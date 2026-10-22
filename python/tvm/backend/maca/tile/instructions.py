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
"""MACA tensor instruction contracts and target-specific lowerings."""

from math import prod

from tvm.ir import TensorRegion
from tvm.script import tirx as T
from tvm.tirx import Function
from tvm.tirx.tile import DispatchContext, fail
from tvm.tirx.tile.instruction import Instruction, Operand, TensorCall, namespace


def _thread_index(sctx: DispatchContext):
    if sctx.is_thread:
        return 0
    if sctx.is_warp:
        return T.maca.lane_id()
    if sctx.is_cta:
        return T.maca.linear_thread_id()
    if sctx.is_warpgroup:
        return T.maca.linear_thread_id() % 256
    fail(f"MACA copy does not support {sctx.scope_kind!r} execution scope")


def _region(value):
    if not isinstance(value, TensorRegion):
        raise TypeError("MACA tile copy operands must be tensor regions")
    return value


def _linear_indices(flat, start, extents):
    indices = []
    for axis, (origin, extent) in enumerate(zip(start, extents, strict=True)):
        stride = prod(extents[axis + 1 :])
        indices.append(origin + flat // stride % extent)
    return tuple(indices)


def lower_copy(call: TensorCall, sctx: DispatchContext) -> Function:
    from .copy.copy import lower_copy as lower

    return lower(call, sctx)


def _value_at(value, indices, extents):
    if isinstance(value, TensorRegion):
        source_extents = [int(r.extent) for r in value.region]
        if len(source_extents) > len(extents):
            fail("MACA elementwise source rank exceeds destination rank")
        aligned_extents = [1] * (len(extents) - len(source_extents)) + source_extents
        if any(src not in (1, dst) for src, dst in zip(aligned_extents, extents, strict=True)):
            fail("MACA elementwise tensor operands are not broadcast-compatible")
        starts = [r.min for r in value.region]
        source_indices = [
            start if extent == 1 else start + index
            for start, extent, index in zip(
                starts, source_extents, indices[-len(starts) :], strict=True
            )
        ]
        return value.source[tuple(source_indices)]
    return value


def lower_elementwise(call: TensorCall, sctx: DispatchContext) -> Function:
    """Lower scalar elementwise operations over matching tensor regions."""
    dst_region = _region(call.args[0])
    dst = dst_region.source
    extents = [int(r.extent) for r in dst_region.region]
    if not extents:
        fail("MACA elementwise instructions require a non-scalar output region")
    starts = [r.min for r in dst_region.region]
    volume = prod(extents)
    threads = prod(int(extent) for extent, _offset in sctx.intra.values()) or 1
    thread = _thread_index(sctx)
    kind = call.kind
    operands = call.args[1:]

    def apply(indices):
        values = [_value_at(value, indices, extents) for value in operands]
        if kind == "add":
            return values[0] + values[1]
        if kind == "sub":
            return values[0] - values[1]
        if kind == "mul":
            return values[0] * values[1]
        if kind == "fdiv":
            return values[0] / values[1]
        if kind == "maximum":
            return T.max(values[0], values[1])
        if kind == "cast":
            return T.cast(values[0], str(dst.dtype))
        if kind == "sqrt":
            return T.sqrt(values[0])
        if kind == "exp":
            return T.exp(values[0])
        if kind == "exp2":
            return T.exp2(values[0])
        if kind == "log2":
            return T.log2(values[0])
        if kind == "reciprocal":
            return T.cast(1, str(dst.dtype)) / values[0]
        if kind == "silu":
            return values[0] / (T.cast(1, str(dst.dtype)) + T.exp(-values[0]))
        if kind == "fill":
            return values[0]
        if kind == "copy":
            return values[0]
        if kind == "zero":
            return T.cast(0, str(dst.dtype))
        if kind == "fma":
            return values[0] * values[1] + values[2]
        if kind.endswith("_with_scale_bias"):
            value = values[0] * values[1] + values[2]
            op = kind.removesuffix("_with_scale_bias")
            if op == "exp":
                return T.exp(value)
            if op == "exp2":
                return T.exp2(value)
            if op == "log2":
                return T.log2(value)
            if op == "sqrt":
                return T.sqrt(value)
        fail(f"MACA tile elementwise lowering does not support {kind!r}")

    @T.function(check_well_formed=False)
    def impl():
        for flat in T.serial(thread, volume, step=threads):
            dst_indices = T.meta_var(_linear_indices(flat, starts, extents))
            src_indices = T.meta_var(_linear_indices(flat, [0] * len(extents), extents))
            dst[dst_indices] = T.meta_var(apply(src_indices))

    return impl


def _reduction_indices(output_flat, reduction_flat, src_start, src_extents, dst_extents, axes):
    output = _linear_indices(output_flat, [0] * len(dst_extents), dst_extents)
    reduced = _linear_indices(reduction_flat, [0] * len(axes), [src_extents[axis] for axis in axes])
    red_by_axis = dict(zip(axes, reduced, strict=True))
    return tuple(
        src_start[axis] + red_by_axis[axis]
        if axis in red_by_axis
        else src_start[axis] + output[axis]
        for axis in range(len(src_extents))
    )


def lower_reduce(call: TensorCall, sctx: DispatchContext) -> Function:
    """Lower a static-axis MACA reduction using one output per active thread."""
    dst_region, src_region, axes_expr, op_expr = call.args
    dst_region, src_region = _region(dst_region), _region(src_region)
    src, dst = src_region.source, dst_region.source
    src_extents = [int(r.extent) for r in src_region.region]
    dst_extents = [int(r.extent) for r in dst_region.region]
    if len(src_extents) != len(dst_extents):
        fail("MACA reduction requires source and destination regions of equal rank")
    if isinstance(axes_expr, tuple | list):
        axes_values = list(axes_expr)
    elif hasattr(axes_expr, "fields"):
        axes_values = list(axes_expr.fields)
    else:
        axes_values = [axes_expr]
    axes = tuple(
        sorted({int(getattr(axis, "value", axis)) % len(src_extents) for axis in axes_values})
    )
    if not axes:
        fail("MACA reduction requires at least one reduction axis")
    for axis in range(len(src_extents)):
        if axis in axes and dst_extents[axis] != 1:
            fail("MACA reduction output extent must be one on reduced axes")
        if axis not in axes and src_extents[axis] != dst_extents[axis]:
            fail("MACA reduction input and output extents must match on retained axes")
    op_name = getattr(op_expr, "value", op_expr)
    if op_name not in ("sum", "max", "min", "prod"):
        fail(f"MACA reduction does not support {op_name!r}")
    volume = prod(dst_extents)
    reduction_volume = prod(src_extents[axis] for axis in axes)
    dst_start = [r.min for r in dst_region.region]
    src_start = [r.min for r in src_region.region]
    threads = prod(int(extent) for extent, _offset in sctx.intra.values()) or 1
    thread = _thread_index(sctx)

    @T.function(check_well_formed=False)
    def impl():
        for output_flat in T.serial(thread, volume, step=threads):
            output_indices = T.meta_var(_linear_indices(output_flat, dst_start, dst_extents))
            scratch = T.alloc_local((1,), str(dst.dtype))
            first_indices = T.meta_var(
                _reduction_indices(output_flat, 0, src_start, src_extents, dst_extents, axes)
            )
            if T.constexpr(op_name == "sum"):
                scratch[0] = T.cast(0, str(dst.dtype))
                start = 0
            else:
                scratch[0] = src[first_indices]
                start = 1
            for reduction_flat in T.serial(start, reduction_volume):
                src_indices = T.meta_var(
                    _reduction_indices(
                        output_flat,
                        reduction_flat,
                        src_start,
                        src_extents,
                        dst_extents,
                        axes,
                    )
                )
                value = src[src_indices]
                if T.constexpr(op_name == "sum"):
                    scratch[0] = scratch[0] + value
                elif T.constexpr(op_name == "prod"):
                    scratch[0] = scratch[0] * value
                elif T.constexpr(op_name == "max"):
                    scratch[0] = T.max(scratch[0], value)
                else:
                    scratch[0] = T.min(scratch[0], value)
            dst[output_indices] = scratch[0]

    return impl


def lower_gemm(call: TensorCall, sctx: DispatchContext) -> Function:
    from .gemm.mma_m16n16k16 import lower_gemm as lower

    return lower(call, sctx)


def lower_permute_layout(call: TensorCall, sctx: DispatchContext) -> Function:
    from .permute_layout.wave64_halfwarp_xor_swizzle import lower_permute_layout as lower

    return lower(call, sctx)


def lower_copy_async(call: TensorCall, sctx: DispatchContext) -> Function:
    from .copy_async.ldgsts import lower_copy_async as lower

    return lower(call, sctx)


def _memory_project(values, attrs):
    return "copy", [values["dst"], values["src"]], {}


def _mov_project(values, attrs):
    value = values["src"]
    return ("copy" if isinstance(value, TensorRegion) else "fill"), [values["dst"], value], {}


def make_namespace():
    pair = (Operand("dst", "region"), Operand("src", "region"))
    specs = [
        Instruction(
            "tirx.maca.tile.copy",
            "copy",
            pair,
            ("thread", "warp", "warpgroup", "cta"),
            schema="MemoryAttrs",
            options={
                "vec_bits": 0,
                "cache": None,
                "l1_evict": None,
                "l2_evict": None,
                "prefetch_size": None,
            },
            lower=lower_copy,
        ),
        *[
            Instruction(
                f"tirx.maca.tile.{name}",
                "copy",
                pair,
                ("thread", "warp", "warpgroup", "cta"),
                schema="MemoryAttrs",
                options={
                    "vec_bits": 0,
                    "cache": None,
                    "l1_evict": None,
                    "l2_evict": None,
                    "prefetch_size": None,
                },
                project=_memory_project,
                lower=lower_copy,
            )
            for name in ("ld", "st")
        ],
        Instruction(
            "tirx.maca.tile.mov",
            "fill",
            (Operand("dst", "region"), Operand("src")),
            ("thread", "warp", "warpgroup", "cta"),
            project=_mov_project,
            lower=lower_elementwise,
        ),
        Instruction(
            "tirx.maca.tile.copy_async",
            "copy_async",
            (
                Operand("dst", "region"),
                Operand("src", "region"),
                Operand("predicate", "scalar", -1),
            ),
            ("thread", "warp", "warpgroup", "cta"),
            schema="AsyncAttrs",
            options={"direct": False, "fill_mode": "", "prefetch_size": -1},
            lower=lower_copy_async,
        ),
        Instruction(
            "tirx.maca.tile.permute_layout",
            "permute_layout",
            pair,
            ("warp",),
            lower=lower_permute_layout,
        ),
        Instruction(
            "tirx.maca.tile.gemm",
            "gemm",
            (
                Operand("D", "region"),
                Operand("A", "region"),
                Operand("B", "region"),
                Operand("C", "region"),
                Operand("transpose_A", "scalar", False),
                Operand("transpose_B", "scalar", False),
                Operand("alpha", "scalar", 1.0),
                Operand("beta", "scalar", 0.0),
            ),
            ("warp",),
            schema="MacaMmaAttrs",
            options={"precision": None},
            lower=lower_gemm,
        ),
    ]
    for name, kind in (
        ("add", "add"),
        ("sub", "sub"),
        ("mul", "mul"),
        ("div", "fdiv"),
        ("fdiv", "fdiv"),
        ("max", "maximum"),
        ("maximum", "maximum"),
    ):
        specs.append(
            Instruction(
                f"tirx.maca.tile.{name}",
                kind,
                (Operand("dst", "region"), Operand("lhs"), Operand("rhs")),
                ("thread", "warp", "warpgroup", "cta"),
                schema="MathAttrs",
                options={"rounding_mode": None},
                lower=lower_elementwise,
            )
        )
    for name, kind in (
        ("cast", "cast"),
        ("cvt", "cast"),
        ("sqrt", "sqrt"),
        ("exp", "exp"),
        ("exp2", "exp2"),
        ("log2", "log2"),
        ("reciprocal", "reciprocal"),
        ("silu", "silu"),
        ("fill", "fill"),
        ("zero", "zero"),
    ):
        specs.append(
            Instruction(
                f"tirx.maca.tile.{name}",
                kind,
                (Operand("dst", "region"), Operand("src")),
                ("thread", "warp", "warpgroup", "cta"),
                lower=lower_elementwise,
            )
        )
    specs.append(
        Instruction(
            "tirx.maca.tile.fma",
            "fma",
            (
                Operand("dst", "region"),
                Operand("src"),
                Operand("scale"),
                Operand("bias"),
            ),
            ("thread", "warp", "warpgroup", "cta"),
            lower=lower_elementwise,
        )
    )
    for name in ("exp", "exp2", "log2", "sqrt"):
        specs.append(
            Instruction(
                f"tirx.maca.tile.compose.{name}_with_scale_bias",
                f"{name}_with_scale_bias",
                (
                    Operand("dst", "region"),
                    Operand("src"),
                    Operand("scale", "scalar"),
                    Operand("bias"),
                ),
                ("thread", "warp", "warpgroup", "cta"),
                lower=lower_elementwise,
            )
        )
    specs.append(
        Instruction(
            "tirx.maca.tile.reduce",
            "reduce",
            (
                Operand("dst", "region"),
                Operand("src", "region"),
                Operand("axes"),
                Operand("op"),
            ),
            ("thread", "warp", "warpgroup", "cta"),
            lower=lower_reduce,
        )
    )
    return namespace("maca", specs)


__all__ = ["make_namespace"]
