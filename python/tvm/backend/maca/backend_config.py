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
"""MACA compiler selection and argument validation."""

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict


class BackendConfig(TypedDict, total=False):
    """Optional MXCC overrides. Compiler arguments are passed as argv items."""

    arch: str
    compiler: Literal["mxcc"]
    mxcc: list[str]


def validate_backend_config(config):
    """Validate MACA backend settings and return independently owned lists."""
    if not isinstance(config, Mapping):
        raise TypeError("MACA backend_config must be a mapping")
    result = {}
    for name, value in config.items():
        if name not in BackendConfig.__annotations__:
            raise ValueError(f"Unknown MACA backend_config key: {name!r}")
        if name == "mxcc":
            if isinstance(value, str | bytes) or not isinstance(value, Sequence):
                raise TypeError("MACA backend_config.mxcc must be a sequence of strings")
            if not all(isinstance(item, str) for item in value):
                raise TypeError("MACA backend_config.mxcc must be a sequence of strings")
            result[name] = list(value)
        elif not isinstance(value, str):
            raise TypeError(f"MACA backend_config.{name} must be a string")
        else:
            result[name] = value
    if "arch" in result and not result["arch"].startswith("xcore"):
        raise ValueError("MACA backend_config.arch must name an XCORE architecture, e.g. xcore1000")
    if "compiler" in result and result["compiler"] != "mxcc":
        raise ValueError("MACA backend_config.compiler must be 'mxcc'")
    return result


def resolve_backend_config(config=None, target=None):
    """Resolve compiler and architecture settings from explicit config and target."""
    from tvm.target import Target

    from ..config import copy_backend_config

    result = {"compiler": "mxcc", "mxcc": []}
    if target is not None:
        target = Target(target)
        if target.kind.name != "maca":
            raise ValueError("MACA backend_config requires a MACA target")
        result.update(copy_backend_config(target.attrs.get("backend_config", {})).get("maca", {}))
        if "arch" not in result and target.attrs.get("mcpu") is not None:
            result["arch"] = str(target.attrs["mcpu"])
    result.update(validate_backend_config(config or {}))
    return validate_backend_config(result)
