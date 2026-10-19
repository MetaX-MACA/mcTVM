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
"""MACA kernel launch configuration supported by the MXMACA runtime."""

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, ClassVar

from tvm.ir import Expr, PrimType
from tvm.ir.prim import IntImm
from tvm.script.ir_builder.stmt import _as_expr

_DYNAMIC_SMEM_TAG = "tirx.use_dyn_shared_memory"


def _scalar(value: Any, name: str, *, allow_zero: bool = False):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer scalar")
    if isinstance(value, Integral):
        value = int(value)
        lower_bound = 0 if allow_zero else 1
        if not lower_bound <= value <= 0xFFFFFFFF:
            raise ValueError(f"{name} must be a uint32 value >= {lower_bound}")
        return value
    if not isinstance(value, Expr) or not isinstance(value.ty, PrimType):
        raise TypeError(f"{name} must be an integer scalar or TIRx scalar expression")
    dtype = value.ty.dtype
    if dtype.lanes != 1 or not dtype.is_integer:
        raise TypeError(f"{name} must be a scalar integer expression")
    if isinstance(value, IntImm):
        constant = int(value)
        lower_bound = 0 if allow_zero else 1
        if not lower_bound <= constant <= 0xFFFFFFFF:
            raise ValueError(f"{name} must be a uint32 value >= {lower_bound}")
    return value


def _dimensions(value: Any, name: str):
    if isinstance(value, tuple | list):
        dimensions = tuple(value)
    else:
        dimensions = (value,)
    if not 1 <= len(dimensions) <= 3:
        raise ValueError(f"{name} expects one to three dimensions")
    return tuple(_scalar(dimension, name) for dimension in dimensions)


@dataclass(frozen=True, kw_only=True)
class LaunchConfig:
    """Grid/block geometry and optional dynamic shared-memory size for MACA.

    ``None`` for ``dynamic_smem_bytes`` uses the kernel's inferred requirement.
    The runtime uses the current MACA stream.
    """

    _config_type: ClassVar[str] = "MacaLaunchConfig"
    grid: Any
    block: Any
    dynamic_smem_bytes: Any = None

    def replace(self, **updates):
        """Return a validated copy with selected launch fields changed."""
        return replace(self, **updates)

    def __post_init__(self):
        grid = _dimensions(self.grid, "grid")
        block = _dimensions(self.block, "block")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "block", block)
        thread_count = 1
        for dimension in block:
            constant = int(dimension) if isinstance(dimension, IntImm) else dimension
            if isinstance(constant, int):
                thread_count *= constant
        if thread_count > 1024:
            raise ValueError("block must contain at most 1024 threads")
        if self.dynamic_smem_bytes is not None:
            value = _scalar(self.dynamic_smem_bytes, "dynamic_smem_bytes", allow_zero=True)
            object.__setattr__(self, "dynamic_smem_bytes", value)


def pack_launch(config: LaunchConfig):
    """Convert configuration to the existing MACA runtime launch-argument ABI."""
    if getattr(config, "_config_type", None) != "MacaLaunchConfig":
        raise TypeError("launch must be a MACA LaunchConfig")
    fields, values = [], []
    for name, dimensions in (("blockIdx", config.grid), ("threadIdx", config.block)):
        padded = dimensions + (1,) * (3 - len(dimensions))
        for axis, extent in zip("xyz", padded, strict=True):
            fields.append(f"{name}.{axis}")
            values.append(_as_expr(extent))
    if config.dynamic_smem_bytes is not None:
        fields.append(_DYNAMIC_SMEM_TAG)
        values.append(_as_expr(config.dynamic_smem_bytes))
    return fields, values


__all__ = ["LaunchConfig", "pack_launch"]
