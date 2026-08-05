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

"""MACA cast operation description."""

from __future__ import annotations

from tvm.tirx import BufferRegion, TilePrimitiveCall

from ..vec_emit.cast_vec2 import CAST_VEC2_IMPL
from . import OpSpec, Plan, SrcSpec


def _parse_cast(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
    dst: BufferRegion = op.args[0]
    src = op.args[1]
    if not isinstance(src, BufferRegion):
        return None, "cast src must be a buffer region"
    return Plan(dst=dst, srcs=[SrcSpec(buf_region=src)]), None


def _compute_cast(src_vals, extras, dtype):
    return src_vals[0]


CAST_OPS: dict[str, OpSpec] = {
    "cast": OpSpec("cast", _parse_cast, _compute_cast, vec_impls=[CAST_VEC2_IMPL])
}

__all__ = ["CAST_OPS"]
