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

"""MACA-local elementwise operation metadata and registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tvm.ir.expr import Expr
from tvm.tirx import BufferRegion, TilePrimitiveCall


@dataclass
class SrcSpec:
    """One elementwise source, either a buffer region or scalar expression."""

    buf_region: BufferRegion | None = None
    scalar: Expr | None = None
    index_fn: Callable | None = None

    @property
    def is_scalar(self) -> bool:
        return self.scalar is not None

    @property
    def buffer(self):
        return self.buf_region.buffer if self.buf_region is not None else None


@dataclass
class Plan:
    """Parsed elementwise operation consumed by a MACA schedule."""

    dst: BufferRegion
    srcs: list[SrcSpec]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class VecImpl:
    """Compatibility container for future MACA-native vector implementations."""

    vec_len: int
    applies: Callable[[TilePrimitiveCall, Any, Plan], tuple[bool, str | None]]
    emit: Callable


@dataclass
class OpSpec:
    """Metadata for one scalar MACA elementwise operation."""

    name: str
    parse: Callable[[TilePrimitiveCall], tuple[Plan | None, str | None]]
    compute_scalar: Callable[[list, dict, str], Any]
    check_extras: Callable | None = None
    vec_impls: list[VecImpl] = field(default_factory=list)


def _build_all_ops() -> dict[str, OpSpec]:
    """Aggregate MACA scalar operation families after defining the data model."""

    from .binary import BINARY_OPS
    from .cast import CAST_OPS
    from .fma import FMA_OPS
    from .unary import UNARY_OPS

    return {**UNARY_OPS, **BINARY_OPS, **CAST_OPS, **FMA_OPS}


ALL_OPS = _build_all_ops()

__all__ = ["ALL_OPS", "OpSpec", "Plan", "SrcSpec", "VecImpl"]
