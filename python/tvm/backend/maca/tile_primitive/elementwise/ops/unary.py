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

"""MACA scalar unary elementwise operation descriptions."""

from __future__ import annotations

from typing import Any

from tvm.ir import is_prim_expr
from tvm.script import tirx as T
from tvm.tirx import BufferRegion, TilePrimitiveCall
from tvm.tirx.expr import FloatImm

from .._common import scalar_dtype
from . import OpSpec, Plan, SrcSpec


def _parse_unary(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
    dst: BufferRegion = op.args[0]
    src = op.args[1]
    bias = op.args[2] if len(op.args) > 2 else None
    scale = op.args[3] if len(op.args) > 2 else None
    if isinstance(src, BufferRegion):
        srcs = [SrcSpec(buf_region=src)]
    elif is_prim_expr(src):
        srcs = [SrcSpec(scalar=src)]
    else:
        return None, f"unsupported src type {type(src).__name__}"

    extras: dict[str, Any] = {
        "scale": scale,
        "bias_const": bias if isinstance(bias, FloatImm) else None,
        "has_bias_buf": isinstance(bias, BufferRegion),
    }
    if isinstance(bias, BufferRegion):
        srcs.append(SrcSpec(buf_region=bias))
    return Plan(dst=dst, srcs=srcs, extras=extras), None


def _check_unary_extras(extras: dict, compute_dtype: str) -> tuple[bool, str | None]:
    scale = extras.get("scale")
    if scale is not None and scalar_dtype(scale) != compute_dtype:
        return False, f"scale dtype {scalar_dtype(scale)} != compute dtype {compute_dtype}"
    bias_const = extras.get("bias_const")
    if bias_const is not None and scalar_dtype(bias_const) != compute_dtype:
        return (
            False,
            f"bias_const dtype {scalar_dtype(bias_const)} != compute dtype {compute_dtype}",
        )
    return True, None


def _with_bias_scale(raw_op):
    def compute(src_vals, extras, dtype):
        value = src_vals[0]
        scale = extras.get("scale")
        if scale is not None:
            value = value * scale
        if extras.get("has_bias_buf"):
            value = value + src_vals[1]
        elif extras.get("bias_const") is not None:
            value = value + extras["bias_const"]
        return raw_op(value)

    return compute


def _compute_zero(src_vals, extras, dtype):
    return 0.0


def _compute_fill(src_vals, extras, dtype):
    return src_vals[0]


def _compute_reciprocal(src_vals, extras, dtype):
    value = src_vals[0]
    return T.FloatImm(value.ty, 1.0) / value


def _compute_silu(src_vals, extras, dtype):
    value = src_vals[0]
    return value / (T.FloatImm(value.ty, 1.0) + T.exp(T.FloatImm(value.ty, 0.0) - value))


UNARY_OPS: dict[str, OpSpec] = {
    "zero": OpSpec("zero", _parse_unary, _compute_zero, _check_unary_extras),
    "fill": OpSpec("fill", _parse_unary, _compute_fill, _check_unary_extras),
    "reciprocal": OpSpec("reciprocal", _parse_unary, _compute_reciprocal, _check_unary_extras),
    "sqrt": OpSpec("sqrt", _parse_unary, _with_bias_scale(T.sqrt), _check_unary_extras),
    "exp": OpSpec("exp", _parse_unary, _with_bias_scale(T.exp), _check_unary_extras),
    "exp2": OpSpec("exp2", _parse_unary, _with_bias_scale(T.exp2), _check_unary_extras),
    "log2": OpSpec("log2", _parse_unary, _with_bias_scale(T.log2), _check_unary_extras),
    "silu": OpSpec("silu", _parse_unary, _compute_silu, _check_unary_extras),
}

__all__ = ["UNARY_OPS"]
