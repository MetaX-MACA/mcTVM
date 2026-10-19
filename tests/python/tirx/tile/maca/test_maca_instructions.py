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
"""Tests for MACA's TIRx tile instruction contracts."""

import tvm
from tvm.script import tirx as T
from tvm.tirx import Function, decl_tensor
from tvm.tirx.exec_scope import ExecScope
from tvm.tirx.tile import DispatchContext
from tvm.tirx.tile.instruction import TensorCall
from tvm.tirx.layout import S, TileLayout


def _copy_call():
    src = decl_tensor([16], dtype="float32", name="src", scope="global")
    dst = decl_tensor([16], dtype="float32", name="dst", scope="shared")
    return T.maca.tile.copy(dst[:], src[:], scope="warp")


def test_maca_tile_instruction_contract_is_registered():
    call = _copy_call()
    assert call.op.name == "tirx.maca.tile.copy"
    view = TensorCall.decode(call)
    assert view.kind == "copy"
    assert view.scope.name == "warp"


def test_maca_tile_copy_lowers_to_tirx_function():
    call = _copy_call()
    view = TensorCall.decode(call)
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    context = DispatchContext(
        target,
        ExecScope("maca", "warp"),
        {},
        {},
        intra={"laneid": [64, 0]},
        scope_kind="warp",
    )
    lowered = view.spec.lower(view, context)
    assert isinstance(lowered, Function)


def test_maca_tile_ld_st_and_mov_aliases_lower():
    global_tensor = decl_tensor([4], dtype="float32", name="global_tensor", scope="global")
    local_tensor = decl_tensor([4], dtype="float32", name="local_tensor", scope="local")
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    context = DispatchContext(
        target,
        ExecScope("maca", "warp"),
        {},
        {},
        intra={"laneid": [64, 0]},
        scope_kind="warp",
    )
    ld_call = T.maca.tile.ld(local_tensor[:], global_tensor[:], scope="warp")
    st_call = T.maca.tile.st(global_tensor[:], local_tensor[:], scope="warp")
    mov_call = T.maca.tile.mov(global_tensor[:], T.float32(2), scope="warp")
    for call in (ld_call, st_call, mov_call):
        view = TensorCall.decode(call)
        assert isinstance(view.spec.lower(view, context), Function)


def test_maca_tile_forced_vector_copy_emits_native_copy_bytes():
    src = decl_tensor([1], dtype="float32", name="src", scope="local")
    dst = decl_tensor([1], dtype="float32", name="dst", scope="local")
    call = T.maca.tile.copy(dst[:], src[:], vec_bits=32, scope="thread")
    view = TensorCall.decode(call)
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    context = DispatchContext(
        target,
        ExecScope("maca", "thread"),
        {},
        {},
        intra={},
        scope_kind="thread",
    )
    lowered = view.spec.lower(view, context)
    assert isinstance(lowered, Function)
    assert "T.maca.copy_bytes" in lowered.script()


def test_maca_tile_elementwise_lowers_to_tirx_function():
    src = decl_tensor([16], dtype="float32", name="src", scope="global")
    dst = decl_tensor([16], dtype="float32", name="dst", scope="global")
    call = T.maca.tile.add(dst[:], src[:], T.float32(1), scope="warp")
    view = TensorCall.decode(call)
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    context = DispatchContext(
        target,
        ExecScope("maca", "warp"),
        {},
        {},
        intra={"laneid": [64, 0]},
        scope_kind="warp",
    )
    lowered = view.spec.lower(view, context)
    assert isinstance(lowered, Function)


def test_maca_tile_reduction_lowers_to_tirx_function():
    src = decl_tensor([16], dtype="float32", name="src", scope="global")
    dst = decl_tensor([1], dtype="float32", name="dst", scope="global")
    call = T.maca.tile.reduce(dst[:], src[:], (0,), "sum", scope="warp")
    view = TensorCall.decode(call)
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    context = DispatchContext(
        target,
        ExecScope("maca", "warp"),
        {},
        {},
        intra={"laneid": [64, 0]},
        scope_kind="warp",
    )
    lowered = view.spec.lower(view, context)
    assert isinstance(lowered, Function)


def test_maca_tile_dispatch_pass_lowers_copy():
    @T.function
    def kernel(src: T.Tensor([16], "float32"), dst: T.Tensor([16], "float32")):
        T.device_entry()
        T.maca.tile.copy(dst[:], src[:], scope="warp")

    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    mod = tvm.tirx.transform.BindTarget(target)(tvm.IRModule({"main": kernel}))
    mod = tvm.tirx.transform.TileDispatch()(mod)
    assert "tirx.maca.tile.copy" not in mod.script()


def test_maca_tile_gemm_lowers_native_wave64_mma():
    from tvm.backend.maca.tile.gemm.mma_m16n16k16 import _a_layout, _b_layout, _d_layout

    A = decl_tensor([16, 16], dtype="float16", name="A", scope="local", layout=_a_layout(16, 16))
    B = decl_tensor([16, 16], dtype="float16", name="B", scope="local", layout=_b_layout(16, 16))
    C = decl_tensor([16, 16], dtype="float32", name="C", scope="local", layout=_d_layout(16, 16))
    D = decl_tensor([16, 16], dtype="float32", name="D", scope="local", layout=_d_layout(16, 16))
    call = T.maca.tile.gemm(D[:, :], A[:, :], B[:, :], C[:, :], scope="warp")
    view = TensorCall.decode(call)
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    context = DispatchContext(
        target,
        ExecScope("maca", "warp"),
        {},
        {},
        intra={"laneid": [64, 0]},
        scope_kind="warp",
    )
    lowered = view.spec.lower(view, context)
    assert isinstance(lowered, Function)


def test_maca_tile_exp_silu_and_maximum_are_registered():
    src = decl_tensor([64], dtype="float32", name="src", scope="global")
    dst = decl_tensor([64], dtype="float32", name="dst", scope="global")
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    context = DispatchContext(
        target,
        ExecScope("maca", "warp"),
        {},
        {},
        intra={"laneid": [64, 0]},
        scope_kind="warp",
    )
    for call in (
        T.maca.tile.exp(dst[:], src[:], scope="warp"),
        T.maca.tile.silu(dst[:], src[:], scope="warp"),
        T.maca.tile.maximum(dst[:], src[:], T.float32(0), scope="warp"),
    ):
        view = TensorCall.decode(call)
        assert isinstance(view.spec.lower(view, context), Function)


def test_maca_tile_copy_async_lowers_bsm_instruction():
    src = decl_tensor([256], dtype="float32", name="src", scope="global")
    dst = decl_tensor([256], dtype="float32", name="dst", scope="shared")
    call = T.maca.tile.copy_async(dst[:], src[:], scope="warp")
    view = TensorCall.decode(call)
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    context = DispatchContext(
        target,
        ExecScope("maca", "warp"),
        {},
        {},
        intra={"laneid": [64, 0]},
        scope_kind="warp",
    )
    lowered = view.spec.lower(view, context)
    assert isinstance(lowered, Function)
    assert "T.maca.copy_async_128b" in lowered.script()


def test_maca_tile_permute_layout_lowers_wave64_staging():
    shape = (4, 32)
    src_layout = TileLayout(S[shape : (32, 1)])
    dst_layout = TileLayout(S[shape : (1, 4)])
    src = decl_tensor([4, 32], dtype="uint32", name="src", scope="shared", layout=src_layout)
    dst = decl_tensor([4, 32], dtype="uint32", name="dst", scope="shared", layout=dst_layout)
    call = T.maca.tile.permute_layout(dst[:, :], src[:, :], scope="warp")
    view = TensorCall.decode(call)
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    context = DispatchContext(
        target,
        ExecScope("maca", "warp"),
        {},
        {},
        intra={"laneid": [64, 0]},
        scope_kind="warp",
    )
    lowered = view.spec.lower(view, context)
    assert isinstance(lowered, Function)
