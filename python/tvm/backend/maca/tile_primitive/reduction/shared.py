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

"""Shared-memory reduction dispatch for MACA GPUs.

CTA and warp scopes use a Wave64 XOR-shuffle tree.  Thread scope uses a
sequential local reduction.  Warpgroup reductions are intentionally excluded:
their cross-wave synchronization needs a separate implementation.
"""

import functools
import math
import operator

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as T
from tvm.tirx import BufferRegion, PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext, fail
from tvm.tirx.operator.tile_primitive.common import ReduceOpType
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch

from ..common import get_indices, get_st_extent, next_power_of_2
from .utils import (
    _analyze_axes,
    _match_reduction_storage_scope,
    _reduction_args,
    build_src_indices,
    reduce_default_value_table,
    reduce_op_table,
)


def validate_reduction_shared(
    op: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    """Validate a shared-memory reduction that MACA can execute."""
    if not sctx.is_target("maca"):
        return False, "expected MACA target"
    if sctx.scope_kind not in ("cta", "warp", "thread"):
        return False, f"unsupported exec_scope {sctx.scope_kind} for shared reduction"

    op = TilePrimitiveCall.downcast(op)
    dst, src = op.output.buffer, op.input.buffer
    if not (src.scope().startswith("shared") and dst.scope().startswith("shared")):
        return False, "expected shared scope for both src and dst"
    if src.dtype != dst.dtype:
        return False, f"dtype mismatch: src={src.dtype} dst={dst.dtype}"
    if "threadIdx.x" not in sctx.launch_params:
        return False, "threadIdx.x not in launch_params"
    if "threadIdx.y" in sctx.launch_params or "threadIdx.z" in sctx.launch_params:
        return False, "multi-dimensional thread binding not supported for shared reduction"
    if sctx.scope_kind == "cta":
        try:
            thread_count = int(sctx.launch_params["threadIdx.x"].dom.extent)
        except (TypeError, ValueError):
            return False, "CTA shared reduction requires a static threadIdx.x extent"
        if thread_count <= 0 or thread_count % 64:
            return (
                False,
                f"CTA shared reduction requires a multiple of 64 threads, got {thread_count}",
            )

    try:
        reduce_dims, spatial_dims = _analyze_axes(len(op.input.region), tuple(op.reduce_axes))
    except AssertionError as error:
        return False, str(error)

    src_extent = [region.extent for region in op.input.region]
    dst_extent = [region.extent for region in op.output.region]
    expected_dst_len = functools.reduce(operator.mul, [src_extent[dim] for dim in spatial_dims], 1)
    actual_dst_len = functools.reduce(operator.mul, dst_extent, 1)
    if not Analyzer().can_prove_equal(expected_dst_len, actual_dst_len):
        return False, f"dst size {actual_dst_len} != expected spatial size {expected_dst_len}"
    return True, None


def _emit_reduction_shared_wave(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: DispatchContext,
    reduce_dims: list[int],
    spatial_dims: list[int],
) -> PrimFunc:
    scope_kind = sctx.scope_kind
    thread_count = sctx.launch_params["threadIdx.x"].dom.extent if scope_kind == "cta" else 64
    dst, src = dst_br.buffer, src_br.buffer
    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    spatial_len = functools.reduce(operator.mul, [src_extent[dim] for dim in spatial_dims], 1)
    reduction_len = functools.reduce(operator.mul, [src_extent[dim] for dim in reduce_dims], 1)
    op_func = reduce_op_table[reduce_op]
    init_value = reduce_default_value_table(src.dtype)[reduce_op]

    group_size = min(next_power_of_2(int(reduction_len)), 64)
    shuffle_count = int(math.log2(group_size)) if group_size > 1 else 0
    spatial_parallelism = int(thread_count) // group_size

    def get_tid_in_scope():
        thread_idx = sctx.launch_params["threadIdx.x"].var
        return thread_idx if scope_kind == "cta" else thread_idx % 64

    def shuffle_data(thread_data):
        @T.inline
        def shuffle_once(mask, value, xor_mask):
            value[0] = op_func(
                value[0], T.tvm_warp_shuffle_xor(mask, value[0], xor_mask, group_size, 64)
            )

        if shuffle_count:
            active_mask = T.tvm_warp_activemask()
            for bit in range(shuffle_count):
                shuffle_once(active_mask, thread_data, 1 << bit)

    @T.inline
    def sync():
        if scope_kind == "cta":
            T.maca.cta_sync()
        else:
            T.maca.warp_sync()

    # fmt: off
    @T.prim_func
    def impl():
        tid_in_scope = get_tid_in_scope()
        thread_data = T.alloc_buffer([1], dtype=src.dtype, scope="local")
        group_id = T.meta_var(T.floordiv(tid_in_scope, group_size))
        lane_in_group = T.meta_var(tid_in_scope % group_size)
        for step in T.serial(T.ceildiv(spatial_len, spatial_parallelism)):
            spatial_fused = T.meta_var(step * spatial_parallelism + group_id)
            if spatial_fused < spatial_len:
                thread_data[0] = init_value
                for tile in T.serial(T.ceildiv(reduction_len, group_size)):
                    reduction_fused = T.meta_var(tile * group_size + lane_in_group)
                    if reduction_fused < reduction_len:
                        src_indices = T.meta_var(
                            build_src_indices(
                                spatial_fused,
                                reduction_fused,
                                spatial_dims,
                                reduce_dims,
                                src_extent,
                                src_st,
                            )
                        )
                        thread_data[0] = op_func(thread_data[0], src[tuple(src_indices)])
                shuffle_data(thread_data)
                if lane_in_group == 0:
                    dst_indices = T.meta_var(get_indices(spatial_fused, dst_st, dst_extent))
                    dst[tuple(dst_indices)] = T.if_then_else(
                        T.bool(accum),
                        op_func(dst[tuple(dst_indices)], thread_data[0]),
                        thread_data[0],
                    )
        sync()
    # fmt: on

    return impl


def _emit_reduction_shared_thread(
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
    spatial_len = functools.reduce(operator.mul, [src_extent[dim] for dim in spatial_dims], 1)
    reduction_len = functools.reduce(operator.mul, [src_extent[dim] for dim in reduce_dims], 1)
    op_func = reduce_op_table[reduce_op]
    init_value = reduce_default_value_table(src.dtype)[reduce_op]

    # fmt: off
    @T.prim_func
    def impl():
        for spatial_fused in T.serial(spatial_len):
            dst_indices = T.meta_var(get_indices(spatial_fused, dst_st, dst_extent))
            if not accum:
                dst[tuple(dst_indices)] = init_value
            for reduction_fused in T.serial(reduction_len):
                src_indices = T.meta_var(
                    build_src_indices(
                        spatial_fused,
                        reduction_fused,
                        spatial_dims,
                        reduce_dims,
                        src_extent,
                        src_st,
                    )
                )
                dst[tuple(dst_indices)] = op_func(
                    dst[tuple(dst_indices)], src[tuple(src_indices)]
                )
    # fmt: on

    return impl


def reduction_shared_impl(
    op: TilePrimitiveCall, op_type: ReduceOpType, sctx: DispatchContext
) -> PrimFunc:
    dst_br, src_br, reduce_axes, accum, _ = _reduction_args(op)
    reduce_dims, spatial_dims = _analyze_axes(len(src_br.region), reduce_axes)
    if sctx.scope_kind in ("cta", "warp"):
        return _emit_reduction_shared_wave(
            dst_br, src_br, accum, op_type, sctx, reduce_dims, spatial_dims
        )
    if sctx.is_thread:
        return _emit_reduction_shared_thread(
            dst_br, src_br, accum, op_type, reduce_dims, spatial_dims
        )
    fail(f"unsupported exec_scope {sctx.scope_kind} for shared reduction")


for _op_name, _op_type in (
    ("sum", ReduceOpType.SUM),
    ("max", ReduceOpType.MAX),
    ("min", ReduceOpType.MIN),
):

    @register_dispatch(
        _op_name,
        "maca",
        variant="shared",
        priority=10,
        when=[
            predicate("storage_scope", _match_reduction_storage_scope, expected_scope=["shared*"]),
            predicate("shared_valid", validate_reduction_shared),
        ],
    )
    def _shared_dispatch(
        op: TilePrimitiveCall, sctx: DispatchContext, _reduce_op=_op_type
    ) -> PrimFunc:
        return reduction_shared_impl(TilePrimitiveCall.downcast(op), _reduce_op, sctx)
