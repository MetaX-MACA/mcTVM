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
"""MACA-owned backend hooks."""

from importlib import import_module
from pathlib import Path

from tvm_ffi.libinfo import load_lib_ctypes

from tvm.base import _LOADED_LIBS

_LAZY_SUBMODULES = {"op", "operator", "script"}


def _detect_target_from_device(dev):
    from tvm.target import Target  # pylint: disable=import-outside-toplevel

    return Target(
        {
            "kind": "maca",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
            "mcpu": "xcore" + dev.compute_version.replace(".", "") + "0",
            "mtriple": "mxc-metax-macahca",
        }
    )


def register_backend():
    """Register MACA-owned Python semantics."""
    from tvm.target.detect_target import register_device_target_detector
    from tvm.tirx.script.builder import ir as builder_ir  # pylint: disable=import-outside-toplevel

    runtime_dir = Path(_LOADED_LIBS["tvm_runtime"]._name).resolve().parent
    try:
        _LOADED_LIBS["tvm_runtime_maca"] = load_lib_ctypes(
            package="tvm",
            target_name="tvm_runtime_maca",
            extra_lib_paths=[runtime_dir],
            mode="RTLD_LOCAL",
        )
    except (OSError, FileNotFoundError, RuntimeError):
        pass

    register_device_target_detector("maca", _detect_target_from_device)
    for name, namespace in script_namespaces().items():
        builder_ir.register_script_namespace(name, namespace)

    import_module(f"{__name__}.operator.intrinsics")
    import_module(f"{__name__}.operator.tile_primitive")


def script_namespaces(**_):
    """Return MACA-owned TVMScript namespaces."""
    from .script import (  # pylint: disable=import-outside-toplevel
        MACANamespace,
    )

    return {
        "maca": MACANamespace(),
    }


def script_namespace(**kwargs):
    """Return the MACA TVMScript namespace object."""
    return script_namespaces(**kwargs)["maca"]


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "op",
    "operator",
    "register_backend",
    "script",
    "script_namespace",
    "script_namespaces",
]
