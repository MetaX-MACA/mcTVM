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

"""C500 Wave64 m16n16k16 MMA tile-primitive GEMM."""

import tvm_ffi

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as T
from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.layout import S, TileLayout, laneid
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.operator.tile_primitive.dispatcher import fail, predicate, register_dispatch


MMA_TILE = 16
MMA_K = 16
WAVE_SIZE = 64


def _full_wave64(_op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    if not sctx.is_warp:
        return False, "MMA m16n16k16 requires warp execution scope"

    active_range = sctx.intra.get("laneid")
    if active_range is None:
        return False, "MMA m16n16k16 requires a laneid axis"
    if len(active_range) not in (2, 3):
        return False, f"invalid laneid active range {active_range}"
    try:
        extent, offset = int(active_range[0]), int(active_range[1])
        stride = int(active_range[2]) if len(active_range) == 3 else 1
    except (TypeError, ValueError):
        return False, f"non-static laneid active range {active_range}"
    if (extent, offset, stride) != (WAVE_SIZE, 0, 1):
        return False, f"MMA m16n16k16 requires Wave64, got {active_range}"
    return True, None


def _no_replica(op_call: TilePrimitiveCall, _sctx: DispatchContext) -> tuple[bool, str | None]:
    op_call = TilePrimitiveCall.downcast(op_call)
    for region, name in zip(op_call.args[:4], ("D", "A", "B", "C"), strict=True):
        layout = region.buffer.layout
        if layout is None:
            return False, f"MMA m16n16k16 requires a layout for {name}"
        if len(layout.replica) != 0:
            return False, f"MMA m16n16k16 does not support replicated {name} layouts"
    return True, None


def _static_int(expr, analyzer: Analyzer, description: str) -> int:
    try:
        return int(analyzer.simplify(expr))
    except (TypeError, ValueError):
        fail(f"MMA m16n16k16 requires static {description}")


def _const_scalar(expr, analyzer: Analyzer) -> float | None:
    value = analyzer.simplify(expr)
    try:
        return float(value.value)
    except (AttributeError, TypeError, ValueError):
        return None


def _matrix_shape(buffer, analyzer: Analyzer, name: str) -> tuple[int, int]:
    if len(buffer.shape) != 2:
        fail(f"MMA m16n16k16 requires a rank-2 {name} buffer")
    rows = _static_int(buffer.shape[0], analyzer, f"{name} rows")
    columns = _static_int(buffer.shape[1], analyzer, f"{name} columns")
    if rows % MMA_TILE != 0 or columns % MMA_TILE != 0:
        fail(f"MMA m16n16k16 requires {name} buffer dimensions divisible by 16")
    return rows, columns


def _matrix_region(buffer_region, analyzer: Analyzer, name: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(buffer_region.region) != 2:
        fail(f"MMA m16n16k16 requires a rank-2 {name} region")

    axes: list[tuple[int, int]] = []
    for axis, region in enumerate(buffer_region.region):
        start = _static_int(region.min, analyzer, f"{name} region axis {axis} start")
        extent = _static_int(region.extent, analyzer, f"{name} region axis {axis} extent")
        if start % MMA_TILE != 0 or extent % MMA_TILE != 0:
            fail(f"MMA m16n16k16 requires 16-aligned {name} regions")
        axes.append((start, extent))
    return axes[0], axes[1]


def _transpose_layout(layout: TileLayout, shape: tuple[int, int]) -> TileLayout:
    grouped, separators = layout.group(shape)
    return grouped.permute_by_groups(separators, [1, 0])


def _a_layout(rows: int, reduction: int) -> TileLayout:
    row_tiles = rows // MMA_TILE
    reduction_tiles = reduction // MMA_K
    return TileLayout(
        S[(row_tiles, MMA_TILE, reduction_tiles, 4, 4) : (
            reduction_tiles * 4,
            1 @ laneid,
            4,
            MMA_TILE @ laneid,
            1,
        )]
    )


def _b_layout(reduction: int, columns: int) -> TileLayout:
    reduction_tiles = reduction // MMA_K
    column_tiles = columns // MMA_TILE
    return TileLayout(
        S[(reduction_tiles, 4, 4, column_tiles, MMA_TILE) : (
            column_tiles * 4,
            MMA_TILE @ laneid,
            1,
            4,
            1 @ laneid,
        )]
    )


def _d_layout(rows: int, columns: int) -> TileLayout:
    row_tiles = rows // MMA_TILE
    column_tiles = columns // MMA_TILE
    return TileLayout(
        S[(row_tiles, 4, 4, column_tiles, MMA_TILE) : (
            column_tiles * 4,
            MMA_TILE @ laneid,
            1,
            4,
            1 @ laneid,
        )]
    )


def _match_layout(name: str, actual, expected: TileLayout) -> None:
    if not isinstance(actual, TileLayout):
        fail(f"MMA m16n16k16 requires a TileLayout for {name}")
    if len(actual.replica) != 0:
        fail(f"MMA m16n16k16 does not support replicated {name} layouts")
    if not tvm_ffi.structural_equal(actual.canonicalize(), expected.canonicalize()):
        fail(f"MMA m16n16k16 received an unsupported {name} fragment layout")


@register_dispatch(
    "gemm",
    "maca",
    variant="mma.m16n16k16",
    priority=10,
    when=[predicate("full_wave64", _full_wave64), predicate("no_replica", _no_replica)],
)
def mma_m16n16k16(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Lower a pure-register C500 GEMM to the direct MMA C builtin."""
    del sctx
    op_call = TilePrimitiveCall.downcast(op_call)
    d_region, a_region, b_region, c_region, transpose_a, transpose_b, alpha, beta = op_call.args
    d_buffer = d_region.buffer
    a_buffer = a_region.buffer
    b_buffer = b_region.buffer
    c_buffer = c_region.buffer
    analyzer = Analyzer()

    for buffer, name in ((d_buffer, "D"), (a_buffer, "A"), (b_buffer, "B"), (c_buffer, "C")):
        if buffer.scope() != "local":
            fail(f"MMA m16n16k16 requires local {name} fragments")

    alpha_value = _const_scalar(alpha, analyzer)
    beta_value = _const_scalar(beta, analyzer)
    if alpha_value != 1.0:
        fail(f"MMA m16n16k16 requires alpha=1, got {alpha}")
    if beta_value not in (0.0, 1.0):
        fail(f"MMA m16n16k16 requires beta in {{0, 1}}, got {beta}")
    transpose_a_value = _const_scalar(transpose_a, analyzer)
    transpose_b_value = _const_scalar(transpose_b, analyzer)
    if transpose_a_value not in (0.0, 1.0) or transpose_b_value not in (0.0, 1.0):
        fail("MMA m16n16k16 requires constant transpose flags")
    transpose_a = transpose_a_value == 1.0
    transpose_b = transpose_b_value == 1.0

    d_shape = _matrix_shape(d_buffer, analyzer, "D")
    a_shape = _matrix_shape(a_buffer, analyzer, "A")
    b_shape = _matrix_shape(b_buffer, analyzer, "B")
    c_shape = _matrix_shape(c_buffer, analyzer, "C")
    d_axes = _matrix_region(d_region, analyzer, "D")
    a_axes = _matrix_region(a_region, analyzer, "A")
    b_axes = _matrix_region(b_region, analyzer, "B")
    c_axes = _matrix_region(c_region, analyzer, "C")

    if transpose_a:
        a_reduction, a_rows = a_shape
        (a_reduction_start, a_reduction_extent), (a_rows_start, a_rows_extent) = a_axes
        expected_a = _transpose_layout(_a_layout(a_rows, a_reduction), (a_rows, a_reduction))
    else:
        a_rows, a_reduction = a_shape
        (a_rows_start, a_rows_extent), (a_reduction_start, a_reduction_extent) = a_axes
        expected_a = _a_layout(a_rows, a_reduction)

    if transpose_b:
        b_columns, b_reduction = b_shape
        (b_columns_start, b_columns_extent), (b_reduction_start, b_reduction_extent) = b_axes
        expected_b = _transpose_layout(_b_layout(b_reduction, b_columns), (b_reduction, b_columns))
    else:
        b_reduction, b_columns = b_shape
        (b_reduction_start, b_reduction_extent), (b_columns_start, b_columns_extent) = b_axes
        expected_b = _b_layout(b_reduction, b_columns)

    (d_rows_start, d_rows_extent), (d_columns_start, d_columns_extent) = d_axes
    (c_rows_start, c_rows_extent), (c_columns_start, c_columns_extent) = c_axes
    _match_layout("A", a_buffer.layout, expected_a)
    _match_layout("B", b_buffer.layout, expected_b)
    _match_layout("C", c_buffer.layout, _d_layout(*c_shape))
    _match_layout("D", d_buffer.layout, _d_layout(*d_shape))

    if (
        a_rows_extent != d_rows_extent
        or b_columns_extent != d_columns_extent
        or a_reduction_extent != b_reduction_extent
        or c_rows_extent != d_rows_extent
        or c_columns_extent != d_columns_extent
    ):
        fail("MMA m16n16k16 GEMM regions have incompatible dimensions")
    dtype = (str(a_buffer.dtype), str(b_buffer.dtype), str(c_buffer.dtype), str(d_buffer.dtype))
    if dtype == ("float16", "float16", "float32", "float32"):
        mma = T.maca.mma_m16n16k16_f16_f32
    elif dtype == ("bfloat16", "bfloat16", "float32", "float32"):
        mma = T.maca.mma_m16n16k16_bf16_f32
    else:
        fail("MMA m16n16k16 supports f16/f16/f32/f32 and bf16/bf16/f32/f32 only")

    a_span = _static_int(a_buffer.layout.storage().span(), analyzer, "A storage span")
    b_span = _static_int(b_buffer.layout.storage().span(), analyzer, "B storage span")
    c_span = _static_int(c_buffer.layout.storage().span(), analyzer, "C storage span")
    d_span = _static_int(d_buffer.layout.storage().span(), analyzer, "D storage span")
    a_column_tiles = a_reduction // MMA_K
    b_column_tiles = b_columns // MMA_TILE
    c_column_tiles = c_shape[1] // MMA_TILE
    d_column_tiles = d_shape[1] // MMA_TILE
    row_tiles = d_rows_extent // MMA_TILE
    column_tiles = d_columns_extent // MMA_TILE
    reduction_tiles = a_reduction_extent // MMA_K
    a_row_tile_start = a_rows_start // MMA_TILE
    a_column_tile_start = a_reduction_start // MMA_K
    b_row_tile_start = b_reduction_start // MMA_K
    b_column_tile_start = b_columns_start // MMA_TILE
    c_row_tile_start = c_rows_start // MMA_TILE
    c_column_tile_start = c_columns_start // MMA_TILE
    d_row_tile_start = d_rows_start // MMA_TILE
    d_column_tile_start = d_columns_start // MMA_TILE
    use_c = beta_value == 1.0

    @T.prim_func(check_well_formed=False)
    def impl():
        a_local = a_buffer.local(a_span)
        b_local = b_buffer.local(b_span)
        c_local = c_buffer.local(c_span)
        d_local = d_buffer.local(d_span)

        for row_tile in T.unroll(row_tiles):
            for column_tile in T.unroll(column_tiles):
                d_offset = (
                    ((d_row_tile_start + row_tile) * d_column_tiles + d_column_tile_start + column_tile)
                    * 4
                )
                c_offset = (
                    ((c_row_tile_start + row_tile) * c_column_tiles + c_column_tile_start + column_tile)
                    * 4
                )
                for element in T.unroll(4):
                    if use_c:
                        d_local[d_offset + element] = c_local[c_offset + element]
                    else:
                        d_local[d_offset + element] = T.float32(0)

                for reduction_tile in T.unroll(reduction_tiles):
                    a_offset = (
                        (
                            (a_row_tile_start + row_tile) * a_column_tiles
                            + a_column_tile_start
                            + reduction_tile
                        )
                        * 4
                    )
                    b_offset = (
                        (
                            (b_row_tile_start + reduction_tile) * b_column_tiles
                            + b_column_tile_start
                            + column_tile
                        )
                        * 4
                    )
                    mma(
                        T.address_of(d_local[d_offset]),
                        T.address_of(a_local[a_offset]),
                        T.address_of(b_local[b_offset]),
                        T.address_of(d_local[d_offset]),
                    )

    return impl
