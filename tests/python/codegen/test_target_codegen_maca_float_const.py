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
"""Float constants survive the round trip through the generated MACA source."""

import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env

VECTOR_N = 8

# Needs all 17 significant digits at float64 and all 9 at float32; a printer that keeps
# fewer cannot reproduce it.
PI = 3.141592653589793

# Far enough below 1 that a fixed-precision printer emits zeros all the way down.
TINY = 1e-20


def _source(dtype, value):
    @T.prim_func
    def kernel(A: T.Buffer((VECTOR_N,), dtype), B: T.Buffer((VECTOR_N,), dtype)):
        T.func_attr({"global_symbol": "const_kernel", "tirx.noalias": True})
        for i in T.thread_binding(VECTOR_N, thread="threadIdx.x"):
            B[i] = A[i] * T.FloatImm(dtype, value)

    target = tvm.target.Target("maca")
    mod = tvm.IRModule.from_expr(kernel.with_attr("target", target))
    executable = tvm.compile(mod, target=target)
    return executable.mod.imports[0].inspect_source()


def _decimal_literals(source):
    return [float(m) for m in re.findall(r"[-+]?\d+\.\d+e[-+]?\d+", source)]


def _hexfloat_literals(source):
    return [float.fromhex(m) for m in re.findall(r"0x[0-9a-fA-F]+\.[0-9a-fA-F]+p[-+]?\d+", source)]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_float64_constant_round_trips():
    source = _source("float64", PI)

    assert PI in _decimal_literals(source), source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_small_float64_constant_is_not_flushed_to_zero():
    source = _source("float64", TINY)

    assert TINY in _decimal_literals(source), source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_float32_constant_round_trips():
    source = _source("float32", PI)

    expected = np.float32(PI)
    assert any(np.float32(h) == expected for h in _hexfloat_literals(source)), source


if __name__ == "__main__":
    tvm.testing.main()
