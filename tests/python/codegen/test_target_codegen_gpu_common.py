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
import re
from functools import partial

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.testing import env

VECTOR_N = 8


def _compile_maca_unary(op, symbol):
    @T.prim_func
    def kernel(A: T.Buffer((VECTOR_N,), "float32"), B: T.Buffer((VECTOR_N,), "float32")):
        T.func_attr({"global_symbol": symbol, "tirx.noalias": True})
        for i in T.thread_binding(VECTOR_N, thread="threadIdx.x"):
            B[i] = op(A[i])

    target = tvm.target.Target("maca")
    mod = tvm.IRModule.from_expr(kernel.with_attr("target", target))
    executable = tvm.compile(mod, target=target)
    return executable, executable.mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_maca_rsqrt_lowers_to_native_intrinsic():
    _, source = _compile_maca_unary(T.rsqrt, "rsqrt_kernel")

    assert re.search(r"\brsqrtf\s*\(", source), source
    assert not re.search(r"/\s*sqrtf\s*\(", source), source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_maca_rsqrt_is_numerically_correct():
    executable, _ = _compile_maca_unary(T.rsqrt, "rsqrt_kernel")

    dev = tvm.maca(0)
    a = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 100.0], dtype="float32")
    tvm_a = tvm.runtime.tensor(a, device=dev)
    out = tvm.runtime.empty((VECTOR_N,), "float32", dev)

    executable(tvm_a, out)
    dev.sync()

    np.testing.assert_allclose(out.numpy(), 1.0 / np.sqrt(a), rtol=1e-5, atol=1e-6)


TIES = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5], dtype="float32")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_maca_round_lowers_to_nearbyint():
    _, source = _compile_maca_unary(T.round, "round_kernel")

    assert re.search(r"\bnearbyintf\s*\(", source), source
    assert not re.search(r"(?<!nearbyint)\broundf\s*\(", source), source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_maca_round_is_ties_to_even():
    executable, _ = _compile_maca_unary(T.round, "round_kernel")

    dev = tvm.maca(0)
    tvm_a = tvm.runtime.tensor(TIES, device=dev)
    out = tvm.runtime.empty((len(TIES),), "float32", dev)
    executable(tvm_a, out)
    dev.sync()

    expected = np.array([0.0, 2.0, 2.0, 4.0, -0.0, -2.0, -2.0, -4.0], dtype="float32")
    np.testing.assert_array_equal(out.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("maca", marks=pytest.mark.gpu),
        pytest.param("metal", marks=pytest.mark.gpu),
        pytest.param({"kind": "vulkan", "supports_int64": True}, marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64"])
def test_int_intrin(target, dtype):
    if not tvm.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")
    test_funcs = [
        (T.clz, lambda x, dtype: int(dtype[-2:]) - (len(bin(x)) - 2)),
    ]

    for tvm_intrin, np_func in test_funcs:
        n = 128

        @I.ir_module(s_tir=True)
        class Module:
            @T.prim_func(s_tir=True)
            def main(
                A: T.Buffer((n,), dtype),
                B: T.Buffer((n,), dtype),
            ):
                T.func_attr({"tirx.noalias": True})
                for i0 in T.thread_binding(n, thread="threadIdx.x"):
                    with T.sblock("B"):
                        v_i0 = T.axis.spatial(n, i0)
                        T.reads(A[v_i0])
                        T.writes(B[v_i0])
                        B[v_i0] = tvm_intrin(A[v_i0])

        f = tvm.compile(Module, target=target)

        def run_and_check():
            dev = tvm.device_from_target(target)
            a = tvm.runtime.tensor(np.random.randint(0, 100000, size=n).astype(dtype), dev)
            b = tvm.runtime.tensor(np.zeros(shape=(n,)).astype(dtype), dev)
            f(a, b)
            ref = np.vectorize(partial(np_func, dtype=dtype))(a.numpy())
            tvm.testing.assert_allclose(b.numpy(), ref)

        tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
