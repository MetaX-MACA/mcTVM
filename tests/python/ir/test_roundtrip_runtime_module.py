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
"""Test roundtrip of runtime modules"""
# pylint: disable=missing-docstring

import pytest
import tvm_ffi

import tvm
import tvm.testing


def test_csource_module():
    mod = tvm.runtime._ffi_api.CSourceModuleCreate("", "cc", [], [])
    assert mod.kind == "c"
    assert mod.is_binary_serializable()
    new_mod = tvm.ir.load_json(tvm.ir.save_json(mod))
    assert new_mod.kind == "c"
    assert new_mod.is_binary_serializable()


def test_maca_module_roundtrip_serializes_backend_config():
    create = tvm_ffi.get_global_func("ffi.Module.create.maca", allow_missing=True)
    load_from_bytes = tvm_ffi.get_global_func("ffi.Module.load_from_bytes.maca", allow_missing=True)
    if create is None or load_from_bytes is None:
        pytest.skip("MACA runtime module is not linked")

    callback_name = "tvm_callback_maca_compile"
    original_callback = tvm_ffi.get_global_func(callback_name, allow_missing=True)
    assert original_callback is not None
    compile_calls = []

    def compile_mock(source, config):
        compile_calls.append((source, config))
        return b"compiled-mcbin"

    tvm_ffi.register_global_func(callback_name, compile_mock, override=True)
    try:
        source = "__global__ void kernel() {}"
        config = '{"maca":{"arch":"xcore-test"}}'
        fmap = tvm_ffi.get_global_func("runtime.LoadMetaDataFromJSON")('{"func_info":{}}')
        module = create(source.encode(), "maca", fmap, source, config)
        restored = tvm.ir.load_json(tvm.ir.save_json(module))
    finally:
        tvm_ffi.register_global_func(callback_name, original_callback, override=True)

    assert compile_calls == [(source, config)]
    assert restored.kind == "maca"
    assert restored.inspect_source("backend_config") == config
    assert restored.inspect_source("mcbin") == "compiled-mcbin"
    assert tvm_ffi.get_global_func("ffi.Module.load_from_file.maca", allow_missing=True) is None


if __name__ == "__main__":
    tvm.testing.main()
