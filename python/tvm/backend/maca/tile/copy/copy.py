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
"""Correctness-first scalar lowering for synchronous tile copies."""

from math import prod

from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Function
from tvm.tirx.tile import DispatchContext, fail
from tvm.tirx.tile.instruction import TensorCall

from ..instructions import _linear_indices, _region, _thread_index


def lower_copy(call: TensorCall, sctx: DispatchContext) -> Function:
    dst_region, src_region = map(_region, call.args)
    dst, src = dst_region.source, src_region.source
    op_name = call.op.name.rsplit(".", 1)[-1]
    if op_name == "ld" and (dst.scope() != "local" or src.scope() not in {"global", "shared"}):
        fail("MACA tile ld requires global/shared-to-local operands")
    if op_name == "st" and (src.scope() != "local" or dst.scope() not in {"global", "shared"}):
        fail("MACA tile st requires local-to-global/shared operands")
    if dst.dtype != src.dtype:
        fail("MACA tile copy requires matching dtypes")
    dst_ext = [int(r.extent) for r in dst_region.region]
    src_ext = [int(r.extent) for r in src_region.region]
    if dst_ext != src_ext or not dst_ext:
        fail(f"MACA tile copy requires matching non-empty regions, got {src_ext} and {dst_ext}")
    for option in ("cache", "l1_evict", "l2_evict", "prefetch_size"):
        if call.options.get(option) is not None:
            fail(f"MACA tile copy does not support {option} hints")
    elements = 1
    for extent in dst_ext:
        elements *= extent
    threads = 1
    for extent, _ in sctx.intra.values():
        threads *= int(extent)
    thread = _thread_index(sctx)
    dst_start = [r.min for r in dst_region.region]
    src_start = [r.min for r in src_region.region]
    vec_bits = call.options.get("vec_bits", 0)
    vec_bits = int(vec_bits.value if hasattr(vec_bits, "value") else vec_bits or 0)

    if vec_bits:
        vec_bytes = vec_bits // 8
        elem_bytes = DataType(src.dtype).bits * DataType(src.dtype).lanes // 8
        if vec_bits not in (16, 32, 64, 128) or vec_bits % (elem_bytes * 8):
            fail("MACA forced vector copy supports 16/32/64/128-bit transactions")
        if sctx.scope_kind != "thread" or elements * elem_bytes != vec_bytes:
            fail("MACA forced vector copy requires one exact-width transaction per thread")
        for region in (dst_region, src_region):
            buffer = region.source
            if buffer.layout is not None and not buffer.layout.is_trivial():
                fail("MACA forced vector copy requires trivial layouts")
            if len(buffer.strides) or buffer.data_alignment < vec_bytes:
                fail("MACA forced vector copy requires naturally aligned contiguous buffers")
            try:
                offset = sum(
                    int(axis.min) * prod(int(dim) for dim in buffer.shape[index + 1 :])
                    for index, axis in enumerate(region.region)
                )
            except (TypeError, ValueError):
                fail("MACA forced vector copy requires static offsets")
            if offset * elem_bytes % vec_bytes:
                fail("MACA forced vector copy region start is misaligned")

        @T.function(check_well_formed=False)
        def vector_impl():
            dst_ptr: T.let = T.address_of(dst[tuple(dst_start)])
            src_ptr: T.let = T.address_of(src[tuple(src_start)])
            T.maca.copy_bytes(dst_ptr, src_ptr, vec_bytes)

        return vector_impl

    @T.function(check_well_formed=False)
    def impl():
        for flat in T.serial(thread, elements, step=threads):
            dst_indices = T.meta_var(_linear_indices(flat, dst_start, dst_ext))
            src_indices = T.meta_var(_linear_indices(flat, src_start, src_ext))
            dst[dst_indices] = src[src_indices]

    return impl
