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

"""MACA scalar fused multiply-add operation description."""

from __future__ import annotations

from tvm.tirx import BufferRegion, TilePrimitiveCall

from . import OpSpec, Plan, SrcSpec


def _parse_fma(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
    dst: BufferRegion = op.args[0]
    srcs = []
    for arg in op.args[1:4]:
        srcs.append(
            SrcSpec(buf_region=arg) if isinstance(arg, BufferRegion) else SrcSpec(scalar=arg)
        )
    return Plan(dst=dst, srcs=srcs), None


def _compute_fma(src_vals, extras, dtype):
    return src_vals[0] * src_vals[1] + src_vals[2]


FMA_OPS: dict[str, OpSpec] = {"fma": OpSpec("fma", _parse_fma, _compute_fma)}

__all__ = ["FMA_OPS"]
