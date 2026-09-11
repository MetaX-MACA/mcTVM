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

"""Descriptor-driven C500 Wave64 register MMA tile-primitive GEMM."""

from collections.abc import Callable
from dataclasses import dataclass

import tvm_ffi

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as T
from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.layout import S, TileLayout, laneid
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.operator.tile_primitive.dispatcher import fail, predicate, register_dispatch

WAVE_SIZE = 64


def _a_layout(rows: int, reduction: int) -> TileLayout:
    return TileLayout(
        S[
            (rows // 16, 16, reduction // 16, 4, 4) : (
                reduction // 16 * 4,
                1 @ laneid,
                4,
                16 @ laneid,
                1,
            )
        ]
    )


def _b_layout(reduction: int, columns: int) -> TileLayout:
    return TileLayout(
        S[
            (reduction // 16, 4, 4, columns // 16, 16) : (
                columns // 16 * 4,
                16 @ laneid,
                1,
                4,
                1 @ laneid,
            )
        ]
    )


def _d_layout(rows: int, columns: int) -> TileLayout:
    return TileLayout(
        S[
            (rows // 16, 4, 4, columns // 16, 16) : (
                columns // 16 * 4,
                16 @ laneid,
                1,
                4,
                1 @ laneid,
            )
        ]
    )


def _f32_a_layout(rows: int, reduction: int) -> TileLayout:
    return TileLayout(
        S[(rows // 16, 16, reduction // 4, 4) : (reduction // 4, 1 @ laneid, 1, 16 @ laneid)]
    )


def _f32_b_layout(reduction: int, columns: int) -> TileLayout:
    return TileLayout(
        S[(reduction // 4, 4, columns // 16, 16) : (columns // 16, 16 @ laneid, 1, 1 @ laneid)]
    )


def _f64_d_layout(rows: int, columns: int) -> TileLayout:
    return TileLayout(
        S[
            (rows // 16, 4, 4, columns // 16, 16) : (
                columns // 16 * 4,
                1,
                16 @ laneid,
                4,
                1 @ laneid,
            )
        ]
    )


def _packed_a_layout(rows: int, reduction: int, slots: int) -> TileLayout:
    return TileLayout(
        S[
            (rows // 8, 8, reduction // (8 * slots), 8, slots) : (
                reduction // (8 * slots) * slots,
                1 @ laneid,
                slots,
                8 @ laneid,
                1,
            )
        ]
    )


def _packed_b_layout(reduction: int, columns: int, slots: int) -> TileLayout:
    return TileLayout(
        S[
            (reduction // (8 * slots), 8, slots, columns // 8, 8) : (
                columns // 8 * slots,
                8 @ laneid,
                1,
                slots,
                1 @ laneid,
            )
        ]
    )


def _packed_d_layout(rows: int, columns: int) -> TileLayout:
    return TileLayout(
        S[(rows // 8, 8, columns // 8, 8) : (columns // 8, 8 @ laneid, 1, 1 @ laneid)]
    )


@dataclass(frozen=True)
class MmaSignature:
    """A native or software atom and its logical per-lane storage contract."""

    dtypes: tuple[str, str, str, str]
    atom: tuple[int, int, int]
    slots: tuple[int, int, int]
    layouts: tuple[Callable, Callable, Callable]
    intrinsic: Callable
    zero: Callable


_NATIVE_LAYOUTS = (_a_layout, _b_layout, _d_layout)
_F32_LAYOUTS = (_f32_a_layout, _f32_b_layout, _d_layout)
_I4_LAYOUTS = (
    lambda m, k: _packed_a_layout(m, k, 4),
    lambda k, n: _packed_b_layout(k, n, 4),
    _packed_d_layout,
)
_B1_LAYOUTS = (
    lambda m, k: _packed_a_layout(m, k, 16),
    lambda k, n: _packed_b_layout(k, n, 16),
    _packed_d_layout,
)
_SIGNATURES = (
    MmaSignature(
        ("float16", "float16", "float32", "float32"),
        (16, 16, 16),
        (4, 4, 4),
        _NATIVE_LAYOUTS,
        T.maca.mma_m16n16k16_f16_f32,
        T.float32,
    ),
    MmaSignature(
        ("bfloat16", "bfloat16", "float32", "float32"),
        (16, 16, 16),
        (4, 4, 4),
        _NATIVE_LAYOUTS,
        T.maca.mma_m16n16k16_bf16_f32,
        T.float32,
    ),
    MmaSignature(
        ("float16", "float16", "float16", "float16"),
        (16, 16, 16),
        (4, 4, 4),
        _NATIVE_LAYOUTS,
        T.maca.mma_m16n16k16_f16_f16,
        T.float16,
    ),
    MmaSignature(
        ("int8", "int8", "int32", "int32"),
        (16, 16, 16),
        (4, 4, 4),
        _NATIVE_LAYOUTS,
        T.maca.mma_m16n16k16_i8_i32,
        T.int32,
    ),
    MmaSignature(
        ("uint8", "uint8", "int32", "int32"),
        (16, 16, 16),
        (4, 4, 4),
        _NATIVE_LAYOUTS,
        T.maca.mma_m16n16k16_u8_i32,
        T.int32,
    ),
    MmaSignature(
        ("float32", "float32", "float32", "float32"),
        (16, 16, 4),
        (1, 1, 4),
        _F32_LAYOUTS,
        T.maca.mma_m16n16k4_f32_f32,
        T.float32,
    ),
    MmaSignature(
        ("float64", "float64", "float64", "float64"),
        (16, 16, 4),
        (1, 1, 4),
        (_f32_a_layout, _f32_b_layout, _f64_d_layout),
        T.maca.mma_m16n16k4_f64_f64,
        T.float64,
    ),
    MmaSignature(
        ("int4", "int4", "int32", "int32"),
        (8, 8, 32),
        (4, 4, 1),
        _I4_LAYOUTS,
        T.maca.mma_m8n8k32_i4_i32,
        T.int32,
    ),
    MmaSignature(
        ("uint4", "uint4", "int32", "int32"),
        (8, 8, 32),
        (4, 4, 1),
        _I4_LAYOUTS,
        T.maca.mma_m8n8k32_u4_i32,
        T.int32,
    ),
    MmaSignature(
        ("int1", "int1", "int32", "int32"),
        (8, 8, 128),
        (16, 16, 1),
        _B1_LAYOUTS,
        T.maca.bmma_m8n8k128_b1_i32,
        T.int32,
    ),
)
_SIGNATURE_BY_DTYPE = {signature.dtypes: signature for signature in _SIGNATURES}


def _full_wave64(_op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    if sctx.target.kind.name != "maca" or str(sctx.target.attrs.get("mcpu", "")) != "xcore1000":
        return False, "MMA requires the C500/xcore1000 target"
    if not sctx.is_warp:
        return False, "MMA requires warp execution scope"
    active_range = sctx.intra.get("laneid")
    if active_range is None or len(active_range) not in (2, 3):
        return False, "MMA requires a laneid axis"
    try:
        extent, offset = int(active_range[0]), int(active_range[1])
        stride = int(active_range[2]) if len(active_range) == 3 else 1
    except (TypeError, ValueError):
        return False, f"non-static laneid active range {active_range}"
    if (extent, offset, stride) != (WAVE_SIZE, 0, 1):
        return False, f"MMA requires Wave64, got {active_range}"
    return True, None


def _no_replica(op_call: TilePrimitiveCall, _sctx: DispatchContext) -> tuple[bool, str | None]:
    op_call = TilePrimitiveCall.downcast(op_call)
    for region, name in zip(op_call.args[:4], ("D", "A", "B", "C"), strict=True):
        layout = region.buffer.layout
        if not isinstance(layout, TileLayout):
            return False, f"MMA requires a TileLayout for {name}"
        if len(layout.replica) != 0:
            return False, f"MMA does not support replicated {name} layouts"
    return True, None


def _static_int(expr, analyzer: Analyzer, description: str) -> int:
    try:
        return int(analyzer.simplify(expr))
    except (TypeError, ValueError):
        fail(f"MMA requires static {description}")


def _const_scalar(expr, analyzer: Analyzer) -> float | None:
    try:
        return float(analyzer.simplify(expr).value)
    except (AttributeError, TypeError, ValueError):
        return None


def _matrix_region(region, analyzer, name, alignment):
    buffer = region.buffer
    if buffer.scope() != "local":
        fail(f"MMA requires local {name} fragments")
    if len(buffer.shape) != 2 or len(region.region) != 2:
        fail(f"MMA requires a rank-2 {name} buffer and region")
    shape = tuple(_static_int(dim, analyzer, f"{name} dimension") for dim in buffer.shape)
    axes = []
    for axis, (dim, bounds, align) in enumerate(zip(shape, region.region, alignment, strict=True)):
        start = _static_int(bounds.min, analyzer, f"{name} axis {axis} start")
        extent = _static_int(bounds.extent, analyzer, f"{name} axis {axis} extent")
        if dim <= 0 or extent <= 0 or start < 0 or start + extent > dim:
            fail(f"MMA {name} requires positive dimensions and in-bounds nonempty regions")
        if dim % align or start % align or extent % align:
            fail(f"MMA {name} requires buffer and region alignment {alignment}")
        axes.append((start, extent))
    return shape, tuple(axes)


def _transpose_layout(layout: TileLayout, shape: tuple[int, int]) -> TileLayout:
    grouped, separators = layout.group(shape)
    return grouped.permute_by_groups(separators, [1, 0])


def _match_layout(name, actual, expected):
    if not isinstance(actual, TileLayout) or len(actual.replica) != 0:
        fail(f"MMA requires a nonreplicated TileLayout for {name}")
    if not tvm_ffi.structural_equal(actual.canonicalize(), expected.canonicalize()):
        fail(f"MMA received an unsupported {name} fragment layout")


def _check_aliases(d_region, a_region, b_region, c_region, analyzer, sctx):
    roots = sctx.shared_state.get("buffer_storage_roots", {})

    def root(buffer):
        return roots.get(buffer, buffer)

    d_buffer, c_buffer = d_region.buffer, c_region.buffer
    for region, name in ((a_region, "A"), (b_region, "B")):
        if root(d_buffer).same_as(root(region.buffer)):
            fail(f"MMA D must not alias {name}")
    if root(d_buffer).same_as(root(c_buffer)):
        exact = d_buffer.same_as(c_buffer) and tvm_ffi.structural_equal(
            d_region.region, c_region.region
        )
        disjoint = d_buffer.same_as(c_buffer) and any(
            analyzer.can_prove(d.min + d.extent <= c.min)
            or analyzer.can_prove(c.min + c.extent <= d.min)
            for d, c in zip(d_region.region, c_region.region, strict=True)
        )
        if not exact and not disjoint:
            # Different views of one allocation may overlap after layout lowering.
            fail("MMA rejects shifted overlapping C/D regions")


def _lower_mma(op_call: TilePrimitiveCall, sctx: DispatchContext, atom) -> PrimFunc:
    op_call = TilePrimitiveCall.downcast(op_call)
    valid, reason = _full_wave64(op_call, sctx)
    if not valid:
        fail(reason)
    d_region, a_region, b_region, c_region, trans_a, trans_b, alpha, beta = op_call.args
    d_buffer, a_buffer, b_buffer, c_buffer = (
        d_region.buffer,
        a_region.buffer,
        b_region.buffer,
        c_region.buffer,
    )
    dtype = tuple(str(buffer.dtype) for buffer in (a_buffer, b_buffer, c_buffer, d_buffer))
    signature = _SIGNATURE_BY_DTYPE.get(dtype)
    if signature is None or signature.atom != atom:
        fail(f"MMA atom {atom} does not support dtype signature {dtype}")
    analyzer = Analyzer()
    if _const_scalar(alpha, analyzer) != 1.0:
        fail("MMA requires alpha=1")
    beta_value = _const_scalar(beta, analyzer)
    if beta_value not in (0.0, 1.0):
        fail("MMA requires beta in {0, 1}")
    transpose_a, transpose_b = _const_scalar(trans_a, analyzer), _const_scalar(trans_b, analyzer)
    if transpose_a not in (0.0, 1.0) or transpose_b not in (0.0, 1.0):
        fail("MMA requires constant transpose flags")
    m, n, k = signature.atom
    a_shape, a_axes = _matrix_region(a_region, analyzer, "A", (k, m) if transpose_a else (m, k))
    b_shape, b_axes = _matrix_region(b_region, analyzer, "B", (n, k) if transpose_b else (k, n))
    c_shape, c_axes = _matrix_region(c_region, analyzer, "C", (m, n))
    d_shape, d_axes = _matrix_region(d_region, analyzer, "D", (m, n))
    if transpose_a:
        a_shape, a_axes = a_shape[::-1], a_axes[::-1]
    if transpose_b:
        b_shape, b_axes = b_shape[::-1], b_axes[::-1]
    expected_a, expected_b = signature.layouts[0](*a_shape), signature.layouts[1](*b_shape)
    if transpose_a:
        expected_a = _transpose_layout(expected_a, a_shape)
    if transpose_b:
        expected_b = _transpose_layout(expected_b, b_shape)
    for name, buffer, expected in (
        ("A", a_buffer, expected_a),
        ("B", b_buffer, expected_b),
        ("C", c_buffer, signature.layouts[2](*c_shape)),
        ("D", d_buffer, signature.layouts[2](*d_shape)),
    ):
        _match_layout(name, buffer.layout, expected)
    if (
        a_axes[0][1] != d_axes[0][1]
        or b_axes[1][1] != d_axes[1][1]
        or a_axes[1][1] != b_axes[0][1]
        or tuple(ext for _, ext in c_axes) != tuple(ext for _, ext in d_axes)
    ):
        fail("MMA GEMM regions have incompatible dimensions")
    _check_aliases(d_region, a_region, b_region, c_region, analyzer, sctx)
    a_span, b_span, c_span, d_span = (
        _static_int(buffer.layout.storage().span(), analyzer, "storage span")
        for buffer in (a_buffer, b_buffer, c_buffer, d_buffer)
    )
    row_tiles, column_tiles, reduction_tiles = (
        d_axes[0][1] // m,
        d_axes[1][1] // n,
        a_axes[1][1] // k,
    )
    a_stride, b_stride, c_stride, d_stride = (
        a_shape[1] // k,
        b_shape[1] // n,
        c_shape[1] // n,
        d_shape[1] // n,
    )
    am, ak = a_axes[0][0] // m, a_axes[1][0] // k
    bk, bn = b_axes[0][0] // k, b_axes[1][0] // n
    cm, cn = c_axes[0][0] // m, c_axes[1][0] // n
    dm, dn = d_axes[0][0] // m, d_axes[1][0] // n
    a_slots, b_slots, d_slots = signature.slots
    use_c = beta_value == 1.0
    packed = m == 8

    @T.prim_func(check_well_formed=False)
    def impl():
        a_local, b_local = a_buffer.local(a_span), b_buffer.local(b_span)
        c_local, d_local = c_buffer.local(c_span), d_buffer.local(d_span)
        for row_tile in T.unroll(row_tiles):
            for column_tile in T.unroll(column_tiles):
                d_offset = ((dm + row_tile) * d_stride + dn + column_tile) * d_slots
                c_offset = ((cm + row_tile) * c_stride + cn + column_tile) * d_slots
                for element in T.unroll(d_slots):
                    if use_c:
                        d_local[d_offset + element] = c_local[c_offset + element]
                    else:
                        d_local[d_offset + element] = signature.zero(0)
                for reduction_tile in T.unroll(reduction_tiles):
                    a_offset = ((am + row_tile) * a_stride + ak + reduction_tile) * a_slots
                    b_offset = ((bk + reduction_tile) * b_stride + bn + column_tile) * b_slots
                    if packed:
                        # Sub-byte address_of rounds to a 32-bit backing word.
                        # Each atom occupies only 16 bits, so address its halfword
                        # explicitly, including the odd atom in each backing word.
                        signature.intrinsic(
                            T.address_of(d_local[d_offset]),
                            T.ptr_byte_offset(
                                T.address_of(a_local[0]), a_offset // a_slots * 2, "int32"
                            ),
                            T.ptr_byte_offset(
                                T.address_of(b_local[0]), b_offset // b_slots * 2, "int32"
                            ),
                            T.address_of(d_local[d_offset]),
                        )
                    else:
                        signature.intrinsic(
                            T.address_of(d_local[d_offset]),
                            T.address_of(a_local[a_offset]),
                            T.address_of(b_local[b_offset]),
                            T.address_of(d_local[d_offset]),
                        )

    return impl


_PREDICATES = [predicate("full_wave64", _full_wave64), predicate("no_replica", _no_replica)]


@register_dispatch("gemm", "maca", variant="mma.m16n16k16", priority=10, when=_PREDICATES)
def mma_m16n16k16(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Lower a 16x16x16 native MMA signature."""
    return _lower_mma(op_call, sctx, (16, 16, 16))


@register_dispatch("gemm", "maca", variant="mma.m16n16k4", priority=11, when=_PREDICATES)
def mma_m16n16k4(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Lower a 16x16x4 full-F32 MMA signature."""
    return _lower_mma(op_call, sctx, (16, 16, 4))


@register_dispatch("gemm", "maca", variant="mma.m8n8k32", priority=12, when=_PREDICATES)
def mma_m8n8k32(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Lower an 8x8x32 signed or unsigned packed 4-bit signature."""
    return _lower_mma(op_call, sctx, (8, 8, 32))


@register_dispatch("gemm", "maca", variant="mma.m8n8k128", priority=12, when=_PREDICATES)
def mma_m8n8k128(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Lower an 8x8x128 one-bit AND/popcount signature."""
    return _lower_mma(op_call, sctx, (8, 8, 128))
