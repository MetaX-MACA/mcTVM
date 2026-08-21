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
"""tirx.enable_fast_math selects the approximate math lowering on the MACA target."""

import re

import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env

VECTOR_N = 8

# Ops with a distinct approximate `__opf` device function under fast-math, so their
# default (precise) and opt-in (fast) lowerings differ. tan is excluded: both of its
# paths lower to the precise MACAMath symbols, so there is no `__tanf` to assert.
FAST_MATH_OPS = ["exp", "exp10", "log", "log2", "log10", "sin", "cos", "pow"]


def _source(op, enable_fast_math):
    fn = getattr(T, op)

    if op == "pow":

        @T.prim_func
        def kernel(A: T.Buffer((VECTOR_N,), "float32"), B: T.Buffer((VECTOR_N,), "float32")):
            T.func_attr({"global_symbol": op + "_kernel", "tirx.noalias": True})
            for i in T.thread_binding(VECTOR_N, thread="threadIdx.x"):
                B[i] = fn(A[i], A[i])

    else:

        @T.prim_func
        def kernel(A: T.Buffer((VECTOR_N,), "float32"), B: T.Buffer((VECTOR_N,), "float32")):
            T.func_attr({"global_symbol": op + "_kernel", "tirx.noalias": True})
            for i in T.thread_binding(VECTOR_N, thread="threadIdx.x"):
                B[i] = fn(A[i])

    target = tvm.target.Target("maca")
    mod = tvm.IRModule.from_expr(kernel.with_attr("target", target))
    config = {"tirx.enable_fast_math": True} if enable_fast_math else {}
    with tvm.transform.PassContext(config=config):
        executable = tvm.compile(mod, target=target)
    return executable.mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("op", FAST_MATH_OPS)
def test_precise_math_is_the_default(op):
    source = _source(op, enable_fast_math=False)

    assert re.search(r"(?<![_\w])" + op + r"f\s*\(", source), source
    assert not re.search(r"__" + op + r"f\s*\(", source), source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("op", FAST_MATH_OPS)
def test_fast_math_is_opt_in(op):
    source = _source(op, enable_fast_math=True)

    assert re.search(r"__" + op + r"f\s*\(", source), source


if __name__ == "__main__":
    tvm.testing.main()
