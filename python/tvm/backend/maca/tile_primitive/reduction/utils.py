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

"""Shared helpers for MACA reduction tile-primitive dispatches."""

import math

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as T
from tvm.tirx import BufferRegion, TilePrimitiveCall
from tvm.tirx.layout import laneid
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.operator.tile_primitive.common import ReduceOpType

from ..common import match_scope

reduce_op_table = {
    ReduceOpType.SUM: lambda a, b: a + b,
    ReduceOpType.MAX: T.max,
    ReduceOpType.MIN: T.min,
}


def reduce_default_value_table(dtype):
    return {
        ReduceOpType.SUM: 0.0,
        ReduceOpType.MAX: T.min_value(dtype),
        ReduceOpType.MIN: T.max_value(dtype),
    }


def _reduction_args(
    op: TilePrimitiveCall,
) -> tuple[BufferRegion, BufferRegion, tuple[int, ...], bool, dict]:
    """Parse a ReduceOp tile primitive."""
    op = TilePrimitiveCall.downcast(op)
    return op.output, op.input, tuple(int(axis) for axis in op.reduce_axes), op.accum, op.config


def _match_reduction_storage_scope(
    op: TilePrimitiveCall, sctx: DispatchContext, expected_scope: list[str]
) -> tuple[bool, str | None]:
    """Check that source and destination match one accepted storage scope."""
    op = TilePrimitiveCall.downcast(op)
    dst_scope = op.output.buffer.scope()
    src_scope = op.input.buffer.scope()
    ok = any(
        match_scope(dst_scope, pattern) and match_scope(src_scope, pattern)
        for pattern in expected_scope
    )
    message = f"storage scope mismatch: dst {dst_scope}, src {src_scope}; expected {expected_scope}"
    return ok, None if ok else message


def _analyze_axes(src_ndim: int, reduce_axes: tuple[int, ...]) -> tuple[list[int], list[int]]:
    """Normalize reduction axes and return reduction and spatial dimensions."""
    reduce_dims = set()
    for axis in reduce_axes:
        normalized = axis if axis >= 0 else axis + src_ndim
        assert 0 <= normalized < src_ndim, f"reduce axis {axis} out of range for ndim={src_ndim}"
        reduce_dims.add(normalized)
    return sorted(reduce_dims), [dim for dim in range(src_ndim) if dim not in reduce_dims]


def _analyze_layout_dims(layout, shape):
    """Split each logical layout dimension into thread and local parts."""
    grouped, seps = layout.group(list(shape))
    result = []
    for dim in range(len(shape)):
        thread_extent = 1
        local_extent = 1
        thread_strides = []
        for index in range(seps[dim], seps[dim + 1]):
            iterator = grouped.shard[index]
            if iterator.axis.is_thread():
                thread_extent *= iterator.extent
                thread_strides.append((iterator.stride, iterator.extent))
            else:
                local_extent *= iterator.extent
        result.append((thread_extent, local_extent, thread_strides))
    return result


def _compute_shuffle_masks(dim_info, reduce_dims: set[int]) -> list[int]:
    """Derive XOR masks for power-of-two thread dimensions being reduced."""
    masks = []
    for dim in reduce_dims:
        _, _, thread_strides = dim_info[dim]
        for stride, extent in thread_strides:
            for bit in range(int(math.log2(int(extent)))):
                masks.append(int(stride) * (1 << bit))
    return sorted(masks)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def validate_wave64_shuffle_layout(
    layout, shape, reduce_dims: list[int]
) -> tuple[bool, str | None]:
    """Validate a full-Wave64 layout that can use XOR shuffle reduction."""
    try:
        grouped, seps = layout.group(list(shape))
    except Exception as error:  # pragma: no cover - layout FFI diagnostics
        return False, f"failed to group layout: {error}"

    masks = []
    for dim in reduce_dims:
        for index in range(seps[dim], seps[dim + 1]):
            iterator = grouped.shard[index]
            if not iterator.axis.is_thread():
                continue
            if iterator.axis != laneid:
                return False, "thread_reduce only supports laneid-partitioned reductions"
            try:
                extent = int(iterator.extent)
                stride = int(iterator.stride)
            except (TypeError, ValueError):
                return False, "thread_reduce requires static laneid layout"
            if not _is_power_of_two(extent):
                return False, f"laneid extent {extent} is not a power of two"
            masks.extend(stride * (1 << bit) for bit in range(extent.bit_length() - 1))

    if not masks:
        return True, None
    if any(mask <= 0 or mask >= 64 or not _is_power_of_two(mask) for mask in masks):
        return False, f"invalid Wave64 XOR masks {sorted(masks)}"
    if len(set(masks)) != len(masks):
        return False, f"duplicate Wave64 XOR masks {sorted(masks)}"
    return True, None


def _build_local_dim_map(layout, buffer_shape):
    """Map logical dimensions to positions in a storage-local view."""
    grouped, seps = layout.group(list(buffer_shape))
    dim_map = {}
    local_position = 0
    for dim in range(len(buffer_shape)):
        has_local = any(
            not grouped.shard[index].axis.is_thread() for index in range(seps[dim], seps[dim + 1])
        )
        dim_map[dim] = local_position if has_local else None
        if has_local:
            local_position += 1
    return dim_map


def _validate_reduction_layout(
    src_layout, dst_layout, src_shape, dst_shape, reduce_dims: list[int]
) -> tuple[bool, str | None]:
    """Validate compatible thread/local partitions for a local reduction."""
    src_info = _analyze_layout_dims(src_layout, src_shape)
    dst_info = _analyze_layout_dims(dst_layout, dst_shape)
    analyzer = Analyzer()
    expected_dst = []
    for src_dim, (thread_extent, local_extent, _) in enumerate(src_info):
        if analyzer.can_prove_equal(thread_extent, 1) and analyzer.can_prove_equal(local_extent, 1):
            continue
        if src_dim in reduce_dims:
            if not analyzer.can_prove_equal(thread_extent, 1):
                expected_dst.append((thread_extent, 1))
        else:
            expected_dst.append((thread_extent, local_extent))

    expected_index = 0
    for thread_extent, local_extent, _ in dst_info:
        if analyzer.can_prove_equal(thread_extent, 1) and analyzer.can_prove_equal(local_extent, 1):
            continue
        if expected_index == len(expected_dst):
            return False, "mismatch dst/src layout for reduction"
        expected_thread, expected_local = expected_dst[expected_index]
        if not (
            analyzer.can_prove_equal(thread_extent, expected_thread)
            and analyzer.can_prove_equal(local_extent, expected_local)
        ):
            return False, "mismatch dst/src layout for reduction"
        expected_index += 1
    if expected_index != len(expected_dst):
        return False, "mismatch dst/src layout for reduction"
    return True, None


def build_src_indices(
    spatial_fused, reduction_fused, spatial_dims, reduce_dims, src_extent, src_st
):
    """Combine fused spatial and reduction indices into source coordinates."""

    def unfuse(fused, dims):
        values = []
        remaining = fused
        for extent in reversed([src_extent[dim] for dim in dims]):
            values.append(remaining % extent)
            remaining //= extent
        return list(reversed(values))

    indices = [None] * len(src_extent)
    for dim, value in zip(spatial_dims, unfuse(spatial_fused, spatial_dims)):
        indices[dim] = value + src_st[dim]
    for dim, value in zip(reduce_dims, unfuse(reduction_fused, reduce_dims)):
        indices[dim] = value + src_st[dim]
    return indices


_REDUCE_OP_TO_STR = {
    ReduceOpType.SUM: "sum",
    ReduceOpType.MAX: "max",
    ReduceOpType.MIN: "min",
}
