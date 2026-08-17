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

"""MACA scalar binary elementwise operation descriptions."""

from __future__ import annotations

import functools
import operator

from tvm.script import tirx as T
from tvm.tirx import BufferRegion, TilePrimitiveCall

from . import OpSpec, Plan, SrcSpec

_COMMUTATIVE = frozenset({"add", "mul", "maximum"})


def _parse_binary_for(op_name: str):
    def parse(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
        dst: BufferRegion = op.args[0]
        src1 = op.args[1]
        src2 = op.args[2]
        src1_scalar = not isinstance(src1, BufferRegion)
        src2_scalar = not isinstance(src2, BufferRegion)
        if src1_scalar and src2_scalar:
            return None, "both inputs are constants"

        if src1_scalar:
            if op_name not in _COMMUTATIVE:
                return None, f"non-commutative op {op_name} cannot have constant lhs"
            src1, src2 = src2, src1
            src2_scalar = True

        if not src2_scalar:
            src1_n = functools.reduce(operator.mul, [r.extent for r in src1.region], 1)
            src2_n = functools.reduce(operator.mul, [r.extent for r in src2.region], 1)
            if src1_n < src2_n:
                if op_name not in _COMMUTATIVE:
                    return None, f"non-commutative op {op_name} cannot swap to broadcast"
                src1, src2 = src2, src1

        srcs = [SrcSpec(buf_region=src1)]
        srcs.append(SrcSpec(scalar=src2) if src2_scalar else SrcSpec(buf_region=src2))
        extras = {}
        rounding_mode = op.config.get("rounding_mode", None)
        if rounding_mode is not None:
            extras["rounding_mode"] = rounding_mode
        return Plan(dst=dst, srcs=srcs, extras=extras), None

    return parse


def _compute_add(src_vals, extras, dtype):
    return src_vals[0] + src_vals[1]


def _compute_sub(src_vals, extras, dtype):
    return src_vals[0] - src_vals[1]


def _compute_mul(src_vals, extras, dtype):
    return src_vals[0] * src_vals[1]


def _compute_fdiv(src_vals, extras, dtype):
    return src_vals[0] / src_vals[1]


def _compute_maximum(src_vals, extras, dtype):
    return T.max(src_vals[0], src_vals[1])


BINARY_OPS: dict[str, OpSpec] = {
    "add": OpSpec("add", _parse_binary_for("add"), _compute_add),
    "sub": OpSpec("sub", _parse_binary_for("sub"), _compute_sub),
    "mul": OpSpec("mul", _parse_binary_for("mul"), _compute_mul),
    "fdiv": OpSpec("fdiv", _parse_binary_for("fdiv"), _compute_fdiv),
    "maximum": OpSpec("maximum", _parse_binary_for("maximum"), _compute_maximum),
}

__all__ = ["BINARY_OPS"]
