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
"""Tests for MACA architecture detection"""

import os
import subprocess
import unittest
from unittest.mock import patch

import pytest

import tvm
from tvm.support import mxcc


class TestMacaArchDetection(unittest.TestCase):
    """Test cases for get_maca_arch function"""

    def test_env_maca_arch(self):
        """Test MACA_ARCH environment variable is used when set"""
        with patch.dict(os.environ, {"MACA_ARCH": "xcore2000"}):
            arch = mxcc.get_maca_arch()
            self.assertEqual(arch, "xcore2000")

    def test_env_maca_arch_overrides_maca_path(self):
        """Test MACA_ARCH takes priority even when maca_path is given"""
        with patch.dict(os.environ, {"MACA_ARCH": "xcore3000"}):
            arch = mxcc.get_maca_arch("/nonexistent")
            self.assertEqual(arch, "xcore3000")

    def test_maca_not_installed_default(self):
        """Test default arch is returned when MACA is not installed"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                arch = mxcc.get_maca_arch("/nonexistent")
                self.assertEqual(arch, "xcore1000")

    def test_maca_path_env_var(self):
        """Test MACA_PATH environment variable is used as fallback path"""
        with patch.dict(os.environ, {"MACA_PATH": "/custom/maca"}, clear=True):
            with patch("os.path.exists", return_value=True):
                with patch(
                    "subprocess.check_output",
                    return_value=b"Name: XCORE1800\n",
                ):
                    arch = mxcc.get_maca_arch()
                    self.assertEqual(arch, "xcore1800")

    def test_oserror_is_caught(self):
        """Test OSError (e.g. missing binary) is caught gracefully"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=True):
                with patch(
                    "subprocess.check_output",
                    side_effect=OSError("macainfo binary not found"),
                ):
                    arch = mxcc.get_maca_arch("/valid/path")
                    self.assertEqual(arch, "xcore1000")


def test_maca_compile_callback_uses_backend_config_arch():
    config = '{"maca":{"compiler":"mxcc","arch":"xcore1000","mxcc":["-O1"]}}'
    with patch("tvm.support.mxcc.subprocess.Popen", side_effect=RuntimeError("captured")) as popen:
        with pytest.raises(RuntimeError, match="captured"):
            mxcc.tvm_callback_maca_compile("__global__ void kernel() {}", config)

    command = popen.call_args.args[0]
    assert "-offload-arch=xcore1000" in command
    assert "-O1" in command


def test_maca_backend_config_validation():
    from tvm.backend.config import copy_backend_config, parse_backend_config

    config = {"maca": {"compiler": "mxcc", "mxcc": ["-O1"]}}
    assert parse_backend_config('{"maca":{"compiler":"mxcc","mxcc":["-O1"]}}') == config
    assert copy_backend_config(config) == config
    target_config = tvm.target.Target({"kind": "maca", "backend_config": config}).attrs[
        "backend_config"
    ]
    assert copy_backend_config(target_config) == config

    with pytest.raises(ValueError, match="compiler must be 'mxcc'"):
        copy_backend_config({"maca": {"compiler": "nvrtc"}})


def test_maca_compile_source_returns_compilation_result(monkeypatch):
    from tvm.backend.maca.compiler import CompilationResult, compile_source

    calls = []

    def compile_mxcc(code, target_format, arch, options):
        calls.append((code, target_format, arch, options))
        return bytearray(b"artifact")

    monkeypatch.setattr(mxcc, "compile_mxcc", compile_mxcc)
    result = compile_source(
        "kernel source",
        {"maca": {"compiler": "mxcc", "arch": "xcore2000", "mxcc": ["-O2"]}},
    )

    assert isinstance(result, CompilationResult)
    assert result.binary == b"artifact"
    assert result.target_format == "mcbin"
    assert result.config == {"maca": {"compiler": "mxcc", "arch": "xcore2000", "mxcc": ["-O2"]}}
    assert calls == [("kernel source", "mcbin", "xcore2000", ["-O2"])]


def test_maca_target_backend_config_reaches_compiler(monkeypatch):
    from tvm.backend.config import parse_backend_config
    from tvm.script import tirx as T

    calls = []

    def compile_mxcc(code, target_format, arch, options):
        calls.append((target_format, arch, options))
        return bytearray(b"compiled")

    monkeypatch.setattr(mxcc, "compile_mxcc", compile_mxcc)

    @T.function
    def kernel(A: T.Tensor((1,), "float32")):
        with T.launch_thread("threadIdx.x", 1) as tx:
            A[tx] = T.float32(1)

    target = {
        "kind": "maca",
        "mcpu": "xcore1000",
        "backend_config": {"maca": {"compiler": "mxcc", "mxcc": ["-O1"]}},
    }
    result = tvm.tirx.compile(kernel, target=target)
    assert parse_backend_config(result.mod.imports[0].inspect_source("backend_config")) == {
        "maca": {"arch": "xcore1000", "compiler": "mxcc", "mxcc": ["-O1"]}
    }
    assert calls == [("mcbin", "xcore1000", ["-O1"])]


def test_maca_get_arch_calledprocesserror_is_caught():
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.exists", return_value=True):
            with patch(
                "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "macainfo")
            ):
                assert mxcc.get_maca_arch("/valid/path") == "xcore1000"


def test_maca_get_arch_macainfo_parsing():
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.check_output", return_value=b"Name: XCORE2000\n"):
                assert mxcc.get_maca_arch("/valid/path") == "xcore2000"


def test_maca_get_arch_macainfo_parsing_no_match():
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.check_output", return_value=b"Name: UNKNOWN_DEVICE\n"):
                assert mxcc.get_maca_arch("/valid/path") == "xcore1000"


if __name__ == "__main__":
    unittest.main()
