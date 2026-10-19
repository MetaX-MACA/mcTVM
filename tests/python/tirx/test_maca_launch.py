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
# software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for MACA's native kernel launch configuration."""

import pytest
import tvm_ffi

import tvm
from tvm.backend.maca.launch import LaunchConfig, pack_launch
from tvm.script import tirx as T


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid": 0},
        {"block": -1},
        {"block": (1024, 2)},
        {"grid": ()},
        {"block": (1, 2, 3, 4)},
        {"block": 1.5},
        {"grid": True},
        {"grid": 2**32},
    ],
)
def test_invalid_maca_launch_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LaunchConfig(**({"grid": 8, "block": 128} | kwargs))


def test_maca_launch_config_packs_runtime_tags():
    config = T.maca.LaunchConfig(grid=(2, 3), block=(64, 2), dynamic_smem_bytes=0)
    fields, values = pack_launch(config)
    assert fields == [
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "threadIdx.x",
        "threadIdx.y",
        "threadIdx.z",
        "tirx.use_dyn_shared_memory",
    ]
    assert [int(value) for value in values] == [2, 3, 1, 64, 2, 1, 0]


@T.function
def _launch_kernel(A: T.Tensor((64,), "int32")):
    T.func_attr({"global_symbol": "launch_kernel", "target": T.target("maca", host="llvm")})
    T.device_entry(launch=T.maca.LaunchConfig(grid=(2,), block=(64,)))
    block_idx = T.maca.block_idx("x")
    thread_idx = T.maca.thread_idx("x")
    A[thread_idx] = block_idx


@T.function
def _dynamic_shared_kernel(A: T.Tensor((1,), "int32")):
    T.func_attr(
        {"global_symbol": "dynamic_shared_kernel", "target": T.target("maca", host="llvm")}
    )
    T.device_entry(launch=T.maca.LaunchConfig(grid=1, block=64))
    scratch = T.alloc_tensor(ty_args=[T.Tensor((16,), "int32", scope="shared.dyn")])
    thread_idx = T.maca.thread_idx("x")
    scratch[0] = thread_idx
    A[0] = scratch[0]


@T.function
def _undersized_dynamic_shared_kernel(A: T.Tensor((1,), "int32")):
    T.func_attr(
        {
            "global_symbol": "undersized_dynamic_shared_kernel",
            "target": T.target("maca", host="llvm"),
        }
    )
    T.device_entry(launch=T.maca.LaunchConfig(grid=1, block=64, dynamic_smem_bytes=32))
    scratch = T.alloc_tensor(ty_args=[T.Tensor((16,), "int32", scope="shared.dyn")])
    scratch[0] = 1
    A[0] = scratch[0]


def test_maca_launch_config_survives_lower_and_split():
    target = tvm.target.Target("maca", host="llvm")
    func = _launch_kernel.with_attr("target", target)
    mod = tvm.IRModule({"main": func})
    mod = tvm.tirx.transform.LowerTIRx()(mod)
    mod = tvm.tirx.transform.SplitHostDevice()(mod)

    kernels = [
        function
        for function in mod.functions.values()
        if function.attrs and function.attrs.get("calling_conv") == 2
    ]
    assert len(kernels) == 1
    expected = [
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "threadIdx.x",
        "threadIdx.y",
        "threadIdx.z",
    ]
    assert list(kernels[0].attrs["tirx.kernel_launch_params"]) == expected
    assert list(kernels[0].attrs["maca.launch_fields"]) == expected


def test_maca_launch_infers_dynamic_shared_memory_size():
    target = tvm.target.Target("maca", host="llvm")
    func = _dynamic_shared_kernel.with_attr("target", target)
    mod = tvm.tirx.transform.SplitHostDevice()(
        tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": func}))
    )
    kernel = next(
        function
        for function in mod.functions.values()
        if function.attrs and function.attrs.get("calling_conv") == 2
    )
    params = list(kernel.attrs["tirx.kernel_launch_params"])
    assert params[-1] == "tirx.use_dyn_shared_memory"

    calls = []
    tvm_ffi.structural_walk(
        mod["main"],
        lambda node: (
            calls.append(node)
            if isinstance(node, tvm.ir.Call)
            and getattr(node.op, "name", "") == "tirx.call_ffi_kernel"
            else None
        ),
    )
    assert len(calls) == 1
    assert int(calls[0].args[-1]) == 64


def test_maca_launch_rejects_undersized_dynamic_shared_memory():
    target = tvm.target.Target("maca", host="llvm")
    func = _undersized_dynamic_shared_kernel.with_attr("target", target)
    mod = tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": func}))
    with pytest.raises(ValueError, match="smaller than"):
        tvm.tirx.transform.SplitHostDevice()(mod)
