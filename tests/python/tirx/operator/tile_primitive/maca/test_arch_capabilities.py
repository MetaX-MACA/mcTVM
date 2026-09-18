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

"""Tests for MACA tile-primitive architecture capability predicates."""

import pytest

from tvm.backend.maca.tile_primitive.common import (
    IntrinsicCapability,
    maca_intrinsic_capability,
    maca_intrinsic_supported,
    maca_mcpu_is,
)
from tvm.backend.maca.tile_primitive.copy_async.ldgsts import _is_ldgsts
from tvm.backend.maca.tile_primitive.gemm.mma_m16n16k16 import _predicates
from tvm.target import Target
from tvm.tirx import DispatchContext
from tvm.tirx.exec_scope import ExecScope


def _sctx(mcpu: str) -> DispatchContext:
    target = Target({"kind": "maca", "mcpu": mcpu})
    return DispatchContext(target, ExecScope("thread"), {}, {}, scope_kind="thread")


@pytest.mark.parametrize(
    "mcpu,supported,expected",
    [
        ("xcore1000", ("xcore1000",), True),
        ("xcore1000", ("xcore900", "xcore1000"), True),
        ("xcore9999", ("xcore1000",), False),
    ],
)
def test_maca_mcpu_is_exact(mcpu, supported, expected):
    ok, reason = maca_mcpu_is(None, _sctx(mcpu), supported=supported)
    assert ok is expected
    if expected:
        assert reason is None
    else:
        assert reason == f"MACA mcpu {mcpu!r} is not one of {supported!r}"


def test_intrinsic_capability_record_is_immutable():
    capability = IntrinsicCapability("mma.test", "xcore1000", "native", "builtin_test")
    with pytest.raises(AttributeError):
        capability.mcpu = "xcore9999"


@pytest.mark.parametrize(
    "intrinsic",
    ["mma.m16n16k16", "mma.m16n16k4", "mma.m8n8k32", "bmma.m8n8k128", "copy_async.bsm"],
)
def test_xcore1000_supports_registered_intrinsics(intrinsic):
    assert maca_intrinsic_supported(None, _sctx("xcore1000"), intrinsic=intrinsic) == (True, None)


def test_intrinsic_capability_exposes_mode_and_builtin():
    capability = maca_intrinsic_capability(_sctx("xcore1000"), "mma.m16n16k16")
    assert capability is not None
    assert capability.mode == "native"
    assert capability.builtin == "maca_mma_m16n16k16_*"


def test_intrinsic_capability_returns_none_for_unknown_pair():
    assert maca_intrinsic_capability(_sctx("xcore9999"), "mma.m16n16k16") is None


def test_unknown_xcore_has_no_implicit_intrinsic_support():
    ok, reason = maca_intrinsic_supported(None, _sctx("xcore9999"), intrinsic="mma.m16n16k16")
    assert not ok
    assert reason == "MACA intrinsic 'mma.m16n16k16' is unsupported on mcpu 'xcore9999'"


def test_unknown_intrinsic_is_rejected_on_known_mcpu():
    ok, reason = maca_intrinsic_supported(None, _sctx("xcore1000"), intrinsic="unknown")
    assert not ok
    assert reason == "MACA intrinsic 'unknown' is unsupported on mcpu 'xcore1000'"


def test_ldgsts_rejects_unknown_xcore_before_layout_analysis():
    ok, reason = _is_ldgsts(None, _sctx("xcore9999"))
    assert not ok
    assert reason == "MACA intrinsic 'copy_async.bsm' is unsupported on mcpu 'xcore9999'"


def test_native_mma_dispatch_predicate_rejects_unknown_xcore():
    ok, reason = _predicates("mma.m16n16k16")[0].evaluate(None, _sctx("xcore9999"))
    assert not ok
    assert reason == "MACA intrinsic 'mma.m16n16k16' is unsupported on mcpu 'xcore9999'"
