# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Local-memory reduction dispatch for MACA GPUs.

Thread scope performs a sequential reduction.  Warp scope additionally
supports layout-driven local views and lane-sharded to replica Wave64 shuffle
reductions.  Warpgroup reductions are not registered because they require a
cross-wave reduction algorithm.
"""

import functools
import operator
from typing import Any

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as T
from tvm.tirx import BufferRegion, PrimFunc, TilePrimitiveCall
from tvm.tirx.layout import TileLayout, laneid
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.operator.tile_primitive.common import ReduceOpType
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch

from ..common import get_indices, get_st_extent
from ..layout_utils import get_local_region, get_sublayout_from_region
from .utils import (
    _analyze_axes,
    _analyze_layout_dims,
    _build_local_dim_map,
    _compute_shuffle_masks,
    _match_reduction_storage_scope,
    _reduction_args,
    _validate_reduction_layout,
    reduce_default_value_table,
    reduce_op_table,
    validate_wave64_shuffle_layout,
)


def _analyze_shuffle_reduce(src_layout, dst_layout):
    """Recognize a full-Wave64 laneid-sharded to replicated reduction."""
    if src_layout.is_swizzle() or dst_layout.is_swizzle():
        return None

    src_canonical = src_layout.canonicalize()
    dst_canonical = dst_layout.canonicalize()
    src_lane_shard = [iterator for iterator in src_canonical.shard if iterator.axis == laneid]
    dst_lane_replica = [iterator for iterator in dst_canonical.replica if iterator.axis == laneid]
    if not src_lane_shard or not dst_lane_replica:
        return None

    if any(
        iterator.axis.is_thread() and iterator.axis != laneid for iterator in src_canonical.shard
    ):
        return None
    if any(
        iterator.axis.is_thread() and iterator.axis != laneid for iterator in dst_canonical.replica
    ):
        return None
    try:
        src_lane_iters = sorted(
            ((int(iterator.stride), int(iterator.extent)) for iterator in src_lane_shard),
        )
        dst_lane_iters = sorted(
            ((int(iterator.stride), int(iterator.extent)) for iterator in dst_lane_replica),
        )
    except (TypeError, ValueError):
        return None

    def is_full_wave(iters):
        next_stride = 1
        for stride, extent in iters:
            if stride != next_stride or extent <= 0 or extent & (extent - 1):
                return False
            next_stride *= extent
        return next_stride == 64

    if not (is_full_wave(src_lane_iters) and is_full_wave(dst_lane_iters)):
        return None
    local_elems = functools.reduce(
        operator.mul,
        [int(iterator.extent) for iterator in src_canonical.shard if not iterator.axis.is_thread()],
        1,
    )
    return 64, local_elems


def _full_wave_active(sctx: DispatchContext) -> tuple[bool, str | None]:
    """Require an unsliced Wave64 before emitting cross-lane shuffles."""
    active_range = sctx.intra.get("laneid")
    if active_range is None:
        return False, "warp reduction is missing laneid active range"
    if len(active_range) not in (2, 3):
        return False, f"invalid laneid active range {active_range}"
    try:
        extent, offset = int(active_range[0]), int(active_range[1])
        stride = int(active_range[2]) if len(active_range) == 3 else 1
    except (TypeError, ValueError):
        return False, f"non-static laneid active range {active_range}"
    if (extent, offset, stride) != (64, 0, 1):
        return False, f"Wave64 shuffle requires contiguous laneid [0, 64), got {active_range}"
    return True, None


def _is_full_buffer_region(buffer_region: BufferRegion) -> bool:
    """Check whether a region covers every logical element of its buffer."""
    buffer = buffer_region.buffer
    if len(buffer_region.region) != len(buffer.shape):
        return False
    analyzer = Analyzer()
    return all(
        analyzer.can_prove_equal(region.min, 0) and analyzer.can_prove_equal(region.extent, shape)
        for region, shape in zip(buffer_region.region, buffer.shape)
    )


def _can_use_shuffle_fast_path(src_br: BufferRegion, dst_br: BufferRegion) -> bool:
    """The flat local shuffle emitter only supports full buffer regions."""
    return _is_full_buffer_region(src_br) and _is_full_buffer_region(dst_br)


def _emit_shuffle_reduce(
    src, dst, reduce_width: int, local_elems: int, accum: bool, reduce_op: ReduceOpType
) -> PrimFunc:
    """Emit a MACA Wave64 XOR tree for each local storage value."""
    op_func = reduce_op_table[reduce_op]
    in_place = src.same_as(dst)
    shuffle_count = reduce_width.bit_length() - 1

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        src_local = src.local(local_elems, layout=src.layout.storage())
        dst_local = dst.local(local_elems, layout=dst.layout.storage())
        old_value = T.alloc_buffer([1], dtype=src.dtype, scope="local")
        for local_index in T.serial(local_elems):
            if accum:
                old_value[0] = dst_local[local_index]
            if not in_place:
                dst_local[local_index] = src_local[local_index]
            active_mask = T.tvm_warp_activemask()
            for bit in T.unroll(shuffle_count):
                value = dst_local[local_index]
                dst_local[local_index] = op_func(
                    value,
                    T.tvm_warp_shuffle_xor(active_mask, value, 1 << bit, reduce_width, 64),
                )
            if accum:
                dst_local[local_index] = op_func(dst_local[local_index], old_value[0])
    # fmt: on

    return impl


def validate_reduction_local(
    op: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    """Validate a thread- or warp-local MACA reduction."""
    if not sctx.is_target("maca"):
        return False, "expected MACA target"
    op = TilePrimitiveCall.downcast(op)
    dst_br, src_br = op.output, op.input
    dst, src = dst_br.buffer, src_br.buffer
    if not (src.scope() == "local" and dst.scope() == "local"):
        return False, "expected local scope for both src and dst"
    if src.dtype != dst.dtype:
        return False, f"dtype mismatch: src={src.dtype} dst={dst.dtype}"
    if sctx.is_thread:
        return True, None
    if not sctx.is_warp:
        return False, f"unsupported exec_scope {sctx.scope_kind} for local reduction"
    if not (src.layout and dst.layout):
        return False, "layouts required for warp-local reduction"
    if not (isinstance(src.layout, TileLayout) and isinstance(dst.layout, TileLayout)):
        return False, "TileLayout required for warp-local reduction"
    if src.layout.is_swizzle() or dst.layout.is_swizzle():
        return False, "swizzle layout unsupported for local reduction"

    try:
        reduce_dims, _ = _analyze_axes(len(src_br.region), tuple(op.reduce_axes))
    except AssertionError as error:
        return False, str(error)

    if (
        _can_use_shuffle_fast_path(src_br, dst_br)
        and _analyze_shuffle_reduce(src.layout, dst.layout) is not None
    ):
        return _full_wave_active(sctx)

    if op.config.get("thread_reduce", False):
        ok, reason = _full_wave_active(sctx)
        if not ok:
            return False, reason
        ok, reason = validate_wave64_shuffle_layout(src.layout, src.shape, reduce_dims)
        if not ok:
            return False, reason

    analyzer = Analyzer()
    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    for layout, buffer, start, extent, name in (
        (src.layout, src, src_st, src_extent, "src"),
        (dst.layout, dst, dst_st, dst_extent, "dst"),
    ):
        if any(
            iterator.axis.is_thread() and analyzer.can_prove_equal(iterator.stride, 0)
            for iterator in layout.shard
        ):
            return False, f"thread dimension with zero stride in {name}"
        if any(iterator.axis.is_thread() for iterator in (getattr(layout, "replica", None) or [])):
            return False, f"thread axis in replica for {name}"
        if get_local_region(layout, list(buffer.shape), start, extent) is None:
            return False, f"get_local_region failed for {name}"

    src_sliced = get_sublayout_from_region(src.layout, src.shape, src_st, src_extent)
    dst_sliced = get_sublayout_from_region(dst.layout, dst.shape, dst_st, dst_extent)
    return _validate_reduction_layout(
        src_sliced, dst_sliced, list(src_extent), list(dst_extent), reduce_dims
    )


def _emit_reduction_local_thread(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    reduce_dims: list[int],
    spatial_dims: list[int],
) -> PrimFunc:
    dst, src = dst_br.buffer, src_br.buffer
    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    spatial_extents = [src_extent[dim] for dim in spatial_dims]
    reduction_extents = [src_extent[dim] for dim in reduce_dims]
    spatial_len = functools.reduce(operator.mul, spatial_extents, 1)
    reduction_len = functools.reduce(operator.mul, reduction_extents, 1)
    op_func = reduce_op_table[reduce_op]
    init_value = reduce_default_value_table(src.dtype)[reduce_op]

    def get_src_indices(spatial_fused, reduction_fused):
        indices = [None] * len(src_extent)
        remaining = spatial_fused
        spatial_values = []
        for extent in reversed(spatial_extents):
            spatial_values.append(remaining % extent)
            remaining //= extent
        remaining = reduction_fused
        reduction_values = []
        for extent in reversed(reduction_extents):
            reduction_values.append(remaining % extent)
            remaining //= extent
        for dim, value in zip(spatial_dims, reversed(spatial_values)):
            indices[dim] = value + src_st[dim]
        for dim, value in zip(reduce_dims, reversed(reduction_values)):
            indices[dim] = value + src_st[dim]
        return indices

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        for spatial_fused in T.serial(spatial_len):
            dst_indices = T.meta_var(get_indices(spatial_fused, dst_st, dst_extent))
            if not accum:
                dst[tuple(dst_indices)] = init_value
            for reduction_fused in T.serial(reduction_len):
                src_indices = T.meta_var(get_src_indices(spatial_fused, reduction_fused))
                dst[tuple(dst_indices)] = op_func(
                    dst[tuple(dst_indices)], src[tuple(src_indices)]
                )
    # fmt: on

    return impl


def _emit_reduction_local_view(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    config: dict[str, Any],
    reduce_dims: list[int],
    src_local_info,
    dst_local_info,
    shuffle_masks: list[int],
) -> PrimFunc:
    dst, src = dst_br.buffer, src_br.buffer
    src_local_shape, src_local_st, src_local_extent = src_local_info
    dst_local_shape, dst_local_st, dst_local_extent = dst_local_info
    src_dim_map = _build_local_dim_map(src.layout, list(src.shape))
    dst_dim_map = _build_local_dim_map(dst.layout, list(dst.shape))
    local_reduce_dims = [dim for dim in reduce_dims if src_dim_map[dim] is not None]
    local_reduce_extent = [src_local_extent[src_dim_map[dim]] for dim in local_reduce_dims]
    local_reduce_start = [src_local_st[src_dim_map[dim]] for dim in local_reduce_dims]
    local_reduce_total = functools.reduce(operator.mul, local_reduce_extent, 1)
    dst_local_total = functools.reduce(operator.mul, dst_local_extent, 1)
    op_func = reduce_op_table[reduce_op]
    init_value = reduce_default_value_table(src.dtype)[reduce_op]
    shuffle = bool(config.get("thread_reduce", False))
    in_place = dst.same_as(src)

    def get_src_local_indices(dst_fused, reduction_fused):
        dst_indices = get_indices(dst_fused, dst_local_st, dst_local_extent)
        reduction_indices = get_indices(reduction_fused, local_reduce_start, local_reduce_extent)
        indices = []
        reduction_position = 0
        for dim in range(len(src_br.region)):
            if src_dim_map[dim] is None:
                continue
            if dim in reduce_dims:
                indices.append(reduction_indices[reduction_position])
                reduction_position += 1
            else:
                indices.append(dst_indices[dst_dim_map[dim]])
        return indices

    def shuffle_value(active_mask, dst_local, dst_indices):
        @T.inline
        def shuffle_once(value, xor_mask):
            dst_local[tuple(dst_indices)] = op_func(
                value, T.tvm_warp_shuffle_xor(active_mask, value, xor_mask, 64, 64)
            )

        for xor_mask in shuffle_masks:
            shuffle_once(dst_local[tuple(dst_indices)], xor_mask)

    save_accum = accum and shuffle

    # fmt: off
    if save_accum:
        @T.prim_func(check_well_formed=False)
        def impl():
            src_local = src.local(*src_local_shape, layout=src.layout.storage())
            dst_local = dst.local(*dst_local_shape, layout=dst.layout.storage())
            old_value = T.alloc_buffer([1], dtype=src.dtype, scope="local")
            for spatial_fused in T.serial(dst_local_total):
                dst_indices = T.meta_var(get_indices(spatial_fused, dst_local_st, dst_local_extent))
                old_value[0] = dst_local[tuple(dst_indices)]
                if not in_place:
                    dst_local[tuple(dst_indices)] = init_value
                    for reduction_fused in T.serial(local_reduce_total):
                        src_indices = T.meta_var(
                            get_src_local_indices(spatial_fused, reduction_fused)
                        )
                        dst_local[tuple(dst_indices)] = op_func(
                            dst_local[tuple(dst_indices)], src_local[tuple(src_indices)]
                        )
                active_mask = T.tvm_warp_activemask()
                shuffle_value(active_mask, dst_local, dst_indices)
                dst_local[tuple(dst_indices)] = op_func(
                    dst_local[tuple(dst_indices)], old_value[0]
                )
    else:
        @T.prim_func(check_well_formed=False)
        def impl():
            src_local = src.local(*src_local_shape, layout=src.layout.storage())
            dst_local = dst.local(*dst_local_shape, layout=dst.layout.storage())
            for spatial_fused in T.serial(dst_local_total):
                dst_indices = T.meta_var(get_indices(spatial_fused, dst_local_st, dst_local_extent))
                if not in_place:
                    if not accum:
                        dst_local[tuple(dst_indices)] = init_value
                    for reduction_fused in T.serial(local_reduce_total):
                        src_indices = T.meta_var(
                            get_src_local_indices(spatial_fused, reduction_fused)
                        )
                        dst_local[tuple(dst_indices)] = op_func(
                            dst_local[tuple(dst_indices)], src_local[tuple(src_indices)]
                        )
                if shuffle:
                    active_mask = T.tvm_warp_activemask()
                    shuffle_value(active_mask, dst_local, dst_indices)
    # fmt: on

    return impl


def reduction_local_impl(
    op: TilePrimitiveCall, op_type: ReduceOpType, sctx: DispatchContext
) -> PrimFunc:
    dst_br, src_br, reduce_axes, accum, config = _reduction_args(op)
    reduce_dims, spatial_dims = _analyze_axes(len(src_br.region), reduce_axes)
    if sctx.is_thread:
        return _emit_reduction_local_thread(
            dst_br, src_br, accum, op_type, reduce_dims, spatial_dims
        )

    src, dst = src_br.buffer, dst_br.buffer
    shuffle_info = (
        _analyze_shuffle_reduce(src.layout, dst.layout)
        if _can_use_shuffle_fast_path(src_br, dst_br)
        else None
    )
    if shuffle_info is not None:
        return _emit_shuffle_reduce(src, dst, *shuffle_info, accum, op_type)

    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    src_local_info = get_local_region(src.layout, list(src.shape), src_st, src_extent)
    dst_local_info = get_local_region(dst.layout, list(dst.shape), dst_st, dst_extent)
    assert src_local_info is not None and dst_local_info is not None
    shuffle_masks = (
        _compute_shuffle_masks(_analyze_layout_dims(src.layout, list(src.shape)), set(reduce_dims))
        if config.get("thread_reduce", False)
        else []
    )
    return _emit_reduction_local_view(
        dst_br,
        src_br,
        accum,
        op_type,
        config,
        reduce_dims,
        src_local_info,
        dst_local_info,
        shuffle_masks,
    )


for _op_name, _op_type in (
    ("sum", ReduceOpType.SUM),
    ("max", ReduceOpType.MAX),
    ("min", ReduceOpType.MIN),
):

    @register_dispatch(
        _op_name,
        "maca",
        variant="local",
        priority=10,
        when=[
            predicate("storage_scope", _match_reduction_storage_scope, expected_scope=["local"]),
            predicate("local_valid", validate_reduction_local),
        ],
    )
    def _local_dispatch(
        op: TilePrimitiveCall, sctx: DispatchContext, _reduce_op=_op_type
    ) -> PrimFunc:
        return reduction_local_impl(TilePrimitiveCall.downcast(op), _reduce_op, sctx)
