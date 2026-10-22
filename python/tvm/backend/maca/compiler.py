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
"""Compile one MACA source group using explicit backend settings."""

from typing import NamedTuple

from ..config import copy_backend_config
from .backend_config import resolve_backend_config


class CompilationResult(NamedTuple):
    """Device artifact and the effective settings used to produce it."""

    binary: bytes
    target_format: str
    config: dict


def compile_source(source: str, backend_config=None, *, target=None) -> CompilationResult:
    """Compile source with MXCC using explicit settings or Target defaults."""
    from tvm.support import mxcc

    config = resolve_backend_config(copy_backend_config(backend_config).get("maca"), target)
    if not config.get("arch"):
        config["arch"] = mxcc.get_maca_arch()
    target_format = "mcbin"
    binary = mxcc.compile_mxcc(
        source, target_format=target_format, arch=config["arch"], options=config["mxcc"]
    )
    return CompilationResult(bytes(binary), target_format, {"maca": config})
