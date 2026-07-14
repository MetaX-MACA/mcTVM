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
"""Codegen tests for the MACA intrinsic lowering rules."""

import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env

VECTOR_N = 8


def _compile_rsqrt():
    @T.prim_func
    def kernel(A: T.Buffer((VECTOR_N,), "float32"), B: T.Buffer((VECTOR_N,), "float32")):
        T.func_attr({"global_symbol": "rsqrt_kernel", "tirx.noalias": True})
        for i in T.thread_binding(VECTOR_N, thread="threadIdx.x"):
            B[i] = T.rsqrt(A[i])

    target = tvm.target.Target("maca")
    mod = tvm.IRModule.from_expr(kernel.with_attr("target", target))
    executable = tvm.compile(mod, target=target)
    return executable, executable.mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_rsqrt_lowers_to_the_native_intrinsic():
    # Without a maca.FLowerIntrinsic registration, rsqrt falls back to the generic
    # legalize rule (src/target/intrin_rule.cc:188), which rewrites it as 1 / sqrt(x).
    _, source = _compile_rsqrt()

    assert re.search(r"\brsqrtf\s*\(", source), source
    assert not re.search(r"/\s*sqrtf\s*\(", source), source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_rsqrt_is_numerically_correct():
    executable, _ = _compile_rsqrt()

    dev = tvm.maca(0)
    a = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 100.0], dtype="float32")
    tvm_a = tvm.runtime.tensor(a, device=dev)
    out = tvm.runtime.empty((VECTOR_N,), "float32", dev)

    executable(tvm_a, out)
    dev.sync()

    np.testing.assert_allclose(out.numpy(), 1.0 / np.sqrt(a), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    tvm.testing.main()
