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
"""Rounding semantics of tirx.round on the MACA target."""

import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env

# Only halfway cases separate the two rules: roundf is ties-away-from-zero,
# nearbyintf under the default mode is ties-to-even. Anything else rounds alike.
TIES = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5], dtype="float32")
VECTOR_N = len(TIES)


def _compile_round():
    @T.prim_func
    def kernel(A: T.Buffer((VECTOR_N,), "float32"), B: T.Buffer((VECTOR_N,), "float32")):
        T.func_attr({"global_symbol": "round_kernel", "tirx.noalias": True})
        for i in T.thread_binding(VECTOR_N, thread="threadIdx.x"):
            B[i] = T.round(A[i])

    target = tvm.target.Target("maca")
    mod = tvm.IRModule.from_expr(kernel.with_attr("target", target))
    executable = tvm.compile(mod, target=target)
    return executable, executable.mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_round_lowers_to_nearbyint():
    _, source = _compile_round()

    assert re.search(r"\bnearbyintf\s*\(", source), source
    assert not re.search(r"(?<!nearbyint)\broundf\s*\(", source), source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_round_is_ties_to_even():
    executable, _ = _compile_round()

    dev = tvm.maca(0)
    tvm_a = tvm.runtime.tensor(TIES, device=dev)
    out = tvm.runtime.empty((VECTOR_N,), "float32", dev)
    executable(tvm_a, out)
    dev.sync()

    # ties-to-even, which is what every other backend and TVM's own constant
    # folding produce. roundf would give 1, 2, 3, 4, -1, -2, -3, -4.
    expected = np.array([0.0, 2.0, 2.0, 4.0, -0.0, -2.0, -2.0, -4.0], dtype="float32")
    np.testing.assert_array_equal(out.numpy(), expected)


if __name__ == "__main__":
    tvm.testing.main()
