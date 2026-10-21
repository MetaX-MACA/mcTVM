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
"""Wave64 BSM global-to-shared asynchronous copies."""

from math import prod

from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Function
from tvm.tirx.tile import DispatchContext, fail
from tvm.tirx.tile.instruction import TensorCall

from ..instructions import _linear_indices, _region, _thread_index

_VALID_BITS = (128, 64, 32)


def _contiguous_aligned(region, byte_width: int) -> bool:
    buffer = region.source
    if buffer.layout is not None and not buffer.layout.is_trivial():
        return False
    if len(buffer.strides):
        return False
    try:
        shape = [int(dim) for dim in buffer.shape]
        starts = [int(axis.min) for axis in region.region]
        extents = [int(axis.extent) for axis in region.region]
    except (TypeError, ValueError):
        return False
    if not extents or len(shape) != len(extents):
        return False
    if any(
        start < 0 or extent <= 0 or start + extent > dim
        for start, extent, dim in zip(starts, extents, shape, strict=True)
    ):
        return False
    pivot = None
    for axis in range(len(shape)):
        if all(extent == 1 for extent in extents[:axis]) and all(
            starts[inner] == 0 and extents[inner] == shape[inner]
            for inner in range(axis + 1, len(shape))
        ):
            pivot = axis
            break
    if pivot is None:
        return False
    flat_offset = 0
    for start, dim in zip(starts, shape, strict=True):
        flat_offset = flat_offset * dim + start
    element_bytes = DataType(buffer.dtype).bits * DataType(buffer.dtype).lanes // 8
    return buffer.data_alignment >= byte_width and flat_offset * element_bytes % byte_width == 0


def lower_copy_async(call: TensorCall, sctx: DispatchContext) -> Function:
    dst_region, src_region, predicate = call.args
    dst_region, src_region = _region(dst_region), _region(src_region)
    dst, src = dst_region.source, src_region.source
    if src.scope() != "global" or not dst.scope().startswith("shared"):
        fail("MACA copy_async supports global-to-shared transfers only")
    if src.dtype != dst.dtype:
        fail("MACA copy_async requires matching source and destination dtypes")
    if call.options.get("direct", False):
        fail("MACA BSM copy_async does not support direct mode")
    if int(call.options.get("prefetch_size", -1)) != -1:
        fail("MACA BSM copy_async does not support prefetch hints")
    fill_mode = call.options.get("fill_mode", "")
    if fill_mode not in ("", "zero"):
        fail("MACA BSM copy_async supports only empty fill_mode or 'zero'")
    predicate_value = predicate.value if hasattr(predicate, "value") else predicate
    if fill_mode == "zero" and isinstance(predicate_value, int) and predicate_value < 0:
        fail("zero-fill copy_async requires a predicate operand")

    extents = [int(r.extent) for r in src_region.region]
    dst_extents = [int(r.extent) for r in dst_region.region]
    if not extents or extents != dst_extents:
        fail("MACA copy_async requires matching non-empty static region shapes")
    elem_bits = DataType(src.dtype).bits * DataType(src.dtype).lanes
    volume = prod(extents) if extents else 0
    threads = prod(int(size) for size, _ in sctx.intra.values()) or 1
    vec_bits = next(
        (
            bits
            for bits in _VALID_BITS
            if bits % elem_bits == 0
            and volume % (threads * (bits // elem_bits)) == 0
            and min(src.data_alignment, dst.data_alignment) >= bits // 8
        ),
        0,
    )
    if vec_bits not in _VALID_BITS or vec_bits % elem_bits:
        fail("MACA copy_async cannot find an aligned transaction width for this region")
    if not _contiguous_aligned(src_region, vec_bits // 8) or not _contiguous_aligned(
        dst_region, vec_bits // 8
    ):
        fail("MACA copy_async requires contiguous, naturally aligned source and destination regions")
    vec_len = vec_bits // elem_bits
    if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"):
        fail(f"MACA copy_async does not support {sctx.scope_kind!r} scope")
    thread = _thread_index(sctx)
    src_start = [r.min for r in src_region.region]
    dst_start = [r.min for r in dst_region.region]
    intrinsic = getattr(T.maca, f"copy_async_{vec_bits}b")
    if fill_mode == "zero":
        intrinsic = getattr(T.maca, f"copy_async_{vec_bits}b_zfill")
    use_zfill = fill_mode == "zero"
    num_transactions = volume // vec_len
    transactions_per_thread = num_transactions // threads

    @T.function(check_well_formed=False)
    def impl():
        for iteration in T.serial(transactions_per_thread):
            transaction = T.meta_var(thread + iteration * threads)
            flat = T.meta_var(transaction * vec_len)
            src_indices = T.meta_var(_linear_indices(flat, src_start, extents))
            dst_indices = T.meta_var(_linear_indices(flat, dst_start, extents))
            src_ptr: T.let = T.address_of(src[src_indices])
            dst_ptr: T.let = T.address_of(dst[dst_indices])
            if T.constexpr(use_zfill):
                intrinsic(dst_ptr, src_ptr, predicate)
            else:
                intrinsic(dst_ptr, src_ptr)

    return impl
