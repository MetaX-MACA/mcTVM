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

"""Explicit fixed-width vector copy dispatches for MACA.

MACA provides typed 16-, 32-, 64-, and 128-bit load/store helpers.  These
variants deliberately accept only a physically contiguous, naturally aligned
thread-local region, rather than applying CUDA-specific cache semantics or
vector operations across a non-trivial layout.
"""

from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import BufferRegion, TilePrimitiveCall

from ..common import get_vec_len
from .utils import _cache_config_supported, _is_valid_copy, _scope_allowed


def _region_start(buffer_region: BufferRegion):
    return [region.min for region in buffer_region.region]


def _region_elements(buffer_region: BufferRegion):
    product = 1
    for region in buffer_region.region:
        product *= region.extent
    return product


def _can_prove_equal(lhs, rhs) -> bool:
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs == rhs
    return Analyzer().can_prove_equal(lhs, rhs)


def _can_prove_zero(value) -> bool:
    return _can_prove_equal(value, 0)


def _region_is_contiguous(buffer_region: BufferRegion) -> bool:
    """Whether a rectangular logical region is one flat physical interval."""
    buffer: Buffer = buffer_region.buffer
    region = list(buffer_region.region)
    for pivot in range(len(region)):
        if not all(_can_prove_equal(outer.extent, 1) for outer in region[:pivot]):
            continue
        if all(
            _can_prove_zero(inner.min) and _can_prove_equal(inner.extent, buffer.shape[index])
            for index, inner in enumerate(region[pivot + 1 :], start=pivot + 1)
        ):
            return True
    return False


def _vector_region_supported(buffer_region: BufferRegion, num_bytes: int):
    buffer: Buffer = buffer_region.buffer
    layout = buffer.layout
    if layout is None or not layout.is_trivial():
        return False, f"{buffer.scope()} buffer has non-trivial layout"
    if len(buffer.strides) != 0:
        return False, f"{buffer.scope()} buffer has explicit strides"
    if not _region_is_contiguous(buffer_region):
        return False, f"{buffer.scope()} region is not physically contiguous"
    if buffer.data_alignment % num_bytes != 0:
        return (
            False,
            f"{buffer.scope()} buffer alignment {buffer.data_alignment} "
            f"is not a multiple of {num_bytes}",
        )
    # ``ptr_to`` uses a delayed layout offset before LowerTIRx.  This path
    # already requires a trivial layout, so ``offset_of`` is its final,
    # physical row-major element offset and can be checked at dispatch time.
    elem_offset = buffer.offset_of(_region_start(buffer_region))[-1]
    byte_offset = elem_offset * DataType(buffer.dtype).bits // 8
    if not _can_prove_zero(byte_offset % num_bytes):
        return False, f"{buffer.scope()} region byte offset is not aligned to {num_bytes} bytes"
    return True, None


def _is_forced_vec_copy(
    op_call: TilePrimitiveCall,
    sctx: DispatchContext,
    *,
    variant: str,
    num_bytes: int,
):
    if not sctx.is_target("maca"):
        return False, "non-maca target"
    config_ok, config_reason = _cache_config_supported(op_call, sctx)
    if not config_ok:
        return False, config_reason
    if getattr(op_call, "dispatch", None) != variant:
        return False, f"requires explicit dispatch={variant!r}"
    if sctx.scope_kind != "thread":
        return False, f"expected thread exec_scope, got {sctx.scope_kind}"
    valid, valid_reason = _is_valid_copy(op_call, sctx)
    if not valid:
        return False, valid_reason
    scope_ok, scope_reason = _scope_allowed(op_call, sctx)
    if not scope_ok:
        return False, scope_reason

    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    if src.dtype != dst.dtype:
        return False, f"dtype mismatch: src={src.dtype}, dst={dst.dtype}"

    elem_bits = DataType(src.dtype).bits
    width_bits = num_bytes * 8
    if width_bits % elem_bits != 0:
        return False, f"{variant} is not an integral number of {src.dtype} elements"
    expected_elements = width_bits // elem_bits
    if not _can_prove_equal(_region_elements(op_call.src), expected_elements):
        return False, f"src region does not contain exactly {expected_elements} elements"
    if not _can_prove_equal(_region_elements(op_call.dst), expected_elements):
        return False, f"dst region does not contain exactly {expected_elements} elements"

    for name, region in (("src", op_call.src), ("dst", op_call.dst)):
        region_ok, region_reason = _vector_region_supported(region, num_bytes)
        if not region_ok:
            return False, f"{name}: {region_reason}"
    vector_length = get_vec_len(op_call.dst, op_call.src, [expected_elements], thread_cnt=1)
    if vector_length != expected_elements:
        return False, f"{variant} regions do not meet MACA vector alignment requirements"
    return True, None


def _emit_forced_vec_copy(
    op_call: TilePrimitiveCall, _sctx: DispatchContext, num_bytes: int
) -> PrimFunc:
    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    src_ptr = src.ptr_to(_region_start(op_call.src))
    dst_ptr = dst.ptr_to(_region_start(op_call.dst))
    copy_op = getattr(T.maca, f"copy_{num_bytes * 8}b")

    @T.prim_func(check_well_formed=False)
    def impl():
        copy_op(dst_ptr, src_ptr)

    return impl


def _register_forced_vec_copy(variant: str, num_bytes: int) -> None:
    @register_dispatch(
        "copy",
        "maca",
        variant=variant,
        priority=20,
        when=[
            predicate(
                f"{variant}_applicable",
                _is_forced_vec_copy,
                variant=variant,
                num_bytes=num_bytes,
            )
        ],
    )
    def _copy_schedule_forced_vec(
        op_call: TilePrimitiveCall,
        sctx: DispatchContext,
        _num_bytes=num_bytes,
    ) -> PrimFunc:
        return _emit_forced_vec_copy(op_call, sctx, _num_bytes)


_register_forced_vec_copy("vec_128b", 16)
_register_forced_vec_copy("vec_64b", 8)
_register_forced_vec_copy("vec_32b", 4)
_register_forced_vec_copy("vec_16b", 2)
