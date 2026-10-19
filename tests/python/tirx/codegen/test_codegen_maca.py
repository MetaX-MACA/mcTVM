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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MACA source code generation."""

import re

import tvm
from tvm.script import tirx as T


def _get_source(func):
    target = tvm.target.Target("maca")
    with target:
        mod = tvm.tirx.compile(tvm.IRModule({"main": func}), target=target, pipeline="tirx")
    return mod.mod.imports[0].inspect_source()


def test_maca_for_unroll_codegen():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(launch=T.maca.LaunchConfig(grid=1, block=(32,)))
        tx = T.maca.thread_idx("x")
        if tx == 0:
            for i in T.serial(4, unroll=True):
                A[i] = A[i] + 1

    source = _get_source(main)
    assert re.search(r"#pragma unroll\s+for \(", source)


def test_maca_for_pragma_unroll_count_codegen():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(launch=T.maca.LaunchConfig(grid=1, block=(32,)))
        tx = T.maca.thread_idx("x")
        if tx == 0:
            for i in T.serial(4, unroll=2):
                A[i] = A[i] + 1

    source = _get_source(main)
    assert re.search(r"#pragma unroll 2\s+for \(", source)


def test_maca_for_disable_unroll_codegen():
    @T.function
    def main(A: T.Tensor((4,), "int32"), begin: T.int32, end: T.int32):
        T.device_entry(launch=T.maca.LaunchConfig(grid=1, block=(32,)))
        tx = T.maca.thread_idx("x")
        if tx == 0:
            for i in T.serial(begin, end, unroll=False):
                A[0] = A[0] + i

    source = _get_source(main)
    assert re.search(r"#pragma unroll 1\s+for \(", source)
