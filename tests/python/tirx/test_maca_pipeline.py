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
"""Tests for target-aware TIRx lowering on MACA."""

from unittest.mock import patch

import pytest

import tvm
from tvm.script import tirx as T
from tvm.tirx.compilation_pipeline import tirx_pipeline

MACA_TARGET = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})


@T.prim_func(check_well_formed=False)
def maca_copy(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (64,), "float32")
    B = T.match_buffer(b, (64,), "float32")
    T.device_entry()
    tx = T.thread_id([64])
    B[tx] = A[tx]


@T.prim_func(check_well_formed=False)
def maca_with_ptx() -> None:
    T.device_entry()
    T.ptx.bar.sync(T.uint32(0))


def test_maca_pipeline_does_not_construct_cuda_iket_pass():
    mod = tvm.IRModule.from_expr(maca_copy.with_attr("target", MACA_TARGET))
    with patch(
        "tvm.tirx.compilation_pipeline.cuda_transforms.LowerIket",
        side_effect=AssertionError("CUDA IKET pass must not run for MACA"),
    ):
        pipeline, _, _ = tirx_pipeline(target=MACA_TARGET)
        pipeline(mod)


def test_maca_pipeline_rejects_ptx_before_lowering():
    mod = tvm.IRModule.from_expr(maca_with_ptx.with_attr("target", MACA_TARGET))
    pipeline, _, _ = tirx_pipeline(target=MACA_TARGET)

    with pytest.raises(
        ValueError,
        match=r"MACA.*maca_with_ptx.*tirx\.ptx\.bar.*T\.maca",
    ):
        pipeline(mod)


def test_mcrtc_compile_source_accepts_target_mcpu():
    from pathlib import Path

    helper = Path(__file__).parents[3] / "src/backend/maca/codegen/build_maca_on.cc"
    text = helper.read_text(encoding="utf-8")
    assert "MCRTCCompile(const std::string& code, const Target& target" in text
    assert 'target->GetAttr<ffi::String>("mcpu"' in text
    assert '"-offload-arch=" + arch' in text
