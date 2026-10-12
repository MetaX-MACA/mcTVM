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
# pylint: disable=missing-function-docstring

"""MACA-specific permute-layout dispatch and storage regression tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.backend.maca.tile_primitive.permute_layout.wave64_halfwarp_xor_swizzle import (
    _choose_xor_k,
)
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.operator.tile_primitive import list_registered_schedules


def _np_layout_offset(extent, strides, multi_idx):
    return int(sum(s * i for s, i in zip(strides, multi_idx)))


def _expected_permute(src_np, src_strides, dst_strides, extent):
    """Compute the expected output: dst at byte offset ``L_dst(i)`` holds the
    value at ``src`` byte offset ``L_src(i)``, for every logical index i.
    """
    V = math.prod(extent)
    dst_np = np.zeros_like(src_np)
    for flat in range(V):
        idx = []
        rem = flat
        for e in reversed(extent):
            idx.append(rem % e)
            rem //= e
        idx = list(reversed(idx))
        src_off = _np_layout_offset(extent, src_strides, idx)
        dst_off = _np_layout_offset(extent, dst_strides, idx)
        dst_np.reshape(-1)[dst_off] = src_np.reshape(-1)[src_off]
    return dst_np


def _compile_and_run(prim_func, np_inputs):
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    with target:
        mod = tvm.IRModule({"main": prim_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    def run_and_check():
        dev = tvm.maca(0)
        tensors = [tvm.runtime.tensor(a, dev) for a in np_inputs]
        mod(*tensors)
        return [tensor.numpy() for tensor in tensors]

    outputs = tvm.testing.run_with_gpu_lock(run_and_check)
    return outputs, mod.mod.imports[0].inspect_source()


@pytest.mark.parametrize("dtype", ["uint32", "float32"])
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_shared_memory_in_place_alias_safety(dtype):
    """Typed shared-memory register staging preserves aliased in-place views."""
    shape = (4, 32)
    pre = TileLayout(S[shape : (32, 1)])  # linear
    post = TileLayout(S[shape : (1, 4)])  # transposed within the 128-block

    # fmt: off
    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, dtype, layout=pre)
        B_buf = T.match_buffer(B, shape, dtype, layout=post)
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        T.lane_id([64])
        storage = T.alloc_buffer((128,), dtype, scope="shared")
        src_view = T.decl_buffer(shape, dtype, data=storage.data, scope="shared", layout=pre)
        dst_view = T.decl_buffer(shape, dtype, data=storage.data, scope="shared", layout=post)
        Tx.cta.copy(src_view[:, :], A_buf[:, :])
        T.maca.cta_sync()
        Tx.warp.permute_layout(dst_view[:, :], src_view[:, :])
        T.maca.cta_sync()
        Tx.cta.copy(B_buf[:, :], dst_view[:, :])
        # fmt: on

    np.random.seed(0)
    A_np = tvm.testing.generate_random_array(dtype, shape)
    B_np = np.zeros_like(A_np)
    [_, B_out], src = _compile_and_run(f, [A_np, B_np])

    ref = _expected_permute(A_np.reshape(-1), [32, 1], [1, 4], list(shape))
    np.testing.assert_array_equal(B_out.reshape(-1), ref)
    assert src.count("tvm_builtin_maca_warp_sync();") >= 2
    assert "__syncwarp()" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64"])
def test_supported_non_32_bit_elements(dtype):
    shape = (4, 32)
    pre = TileLayout(S[shape : (32, 1)])
    post = TileLayout(S[shape : (1, 4)])

    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, dtype, layout=pre)
        B_buf = T.match_buffer(B, shape, dtype, layout=post)
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        T.lane_id([64])
        Tx.warp.permute_layout(B_buf, A_buf)

    np.random.seed(0)
    A_np = tvm.testing.generate_random_array(dtype, shape)
    B_np = np.zeros_like(A_np)
    [_, B_out], _ = _compile_and_run(f, [A_np, B_np])
    ref = _expected_permute(A_np.reshape(-1), [32, 1], [1, 4], list(shape))
    np.testing.assert_array_equal(B_out.reshape(-1), ref)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("volume", [3, 17, 33, 96, 127])
def test_arbitrary_volume_uses_generic_wave64_path(volume):
    """Non-power-of-two volumes must use guarded generic staging correctly."""
    shape = (volume,)
    layout = TileLayout(S[shape : (1,)])

    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, "uint16", layout=layout)
        B_buf = T.match_buffer(B, shape, "uint16", layout=layout)
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        T.lane_id([64])
        Tx.warp.permute_layout(B_buf, A_buf)

    np.random.seed(volume)
    A_np = tvm.testing.generate_random_array("uint16", shape)
    B_np = np.zeros_like(A_np)
    [_, B_out], src = _compile_and_run(f, [A_np, B_np])
    np.testing.assert_array_equal(B_out, A_np)
    assert "tvm_builtin_maca_warp_sync();" in src


def test_permute_layout_schedule_is_registered_for_maca():
    schedules = list_registered_schedules()
    assert schedules["tirx.tile.permute_layout"]["maca"] == [
        "wave64_xor",
        "wave64_generic",
    ]


def test_reject_unsupported_maca_architecture():
    shape = (4, 32)
    pre = TileLayout(S[shape : (32, 1)])
    post = TileLayout(S[shape : (1, 4)])

    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, "uint32", layout=pre)
        B_buf = T.match_buffer(B, shape, "uint32", layout=post)
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        T.lane_id([64])
        Tx.warp.permute_layout(B_buf, A_buf)

    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore9999"})
    with target, pytest.raises(RuntimeError) as exc_info:
        tvm.compile(tvm.IRModule({"main": f}), target=target, tir_pipeline="tirx")
    assert "MACA mcpu 'xcore9999' is not one of ('xcore1000',)" in str(exc_info.value)


def test_choose_xor_k_rejects_out_of_contract_slots():
    assert _choose_xor_k([64, 32], [32, 1], [1, 64], 4, 64, 64) is None


def _lower_forced_dispatch(dispatch, src_scope="global", dst_scope="global", dtype="uint32"):
    shape = (4, 32)
    pre = TileLayout(S[shape : (32, 1)])
    post = TileLayout(S[shape : (1, 4)])

    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, dtype, layout=pre)
        B_buf = T.match_buffer(B, shape, dtype, layout=post)
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        T.lane_id([64])
        if src_scope != "global":
            src_storage = T.alloc_buffer((128,), dtype, scope=src_scope)
            src = T.decl_buffer(shape, dtype, data=src_storage.data, scope=src_scope, layout=pre)
        else:
            src = A_buf
        if dst_scope != "global":
            dst_storage = T.alloc_buffer((128,), dtype, scope=dst_scope)
            dst = T.decl_buffer(shape, dtype, data=dst_storage.data, scope=dst_scope, layout=post)
        else:
            dst = B_buf
        Tx.warp.permute_layout(dst, src, dispatch=dispatch)

    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    with target:
        return tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": f}))


@pytest.mark.parametrize("dispatch", [None, "wave64_xor", "wave64_generic"])
@pytest.mark.parametrize(
    "src_scope, dst_scope",
    [
        ("local", "global"),
        ("global", "local"),
        ("local", "shared"),
        ("shared", "local"),
        ("local", "local"),
    ],
)
def test_reject_thread_private_storage(dispatch, src_scope, dst_scope):
    with pytest.raises(RuntimeError, match="requires global or shared storage"):
        _lower_forced_dispatch(dispatch, src_scope, dst_scope)


@pytest.mark.parametrize("dispatch", [None, "wave64_generic"])
@pytest.mark.parametrize("src_scope", ["global", "shared"])
@pytest.mark.parametrize("dst_scope", ["global", "shared"])
def test_supported_storage_scopes_lower(dispatch, src_scope, dst_scope):
    _lower_forced_dispatch(dispatch, src_scope, dst_scope)


def test_xor_dispatch_requires_shared_shared_and_nonzero_schedule():
    assert _choose_xor_k([4, 32], [32, 1], [1, 4], 4, 4, 32) == 2
    with pytest.raises(RuntimeError, match="shared/shared buffers"):
        _lower_forced_dispatch("wave64_xor")
    with pytest.raises(RuntimeError, match="shared/shared buffers"):
        _lower_forced_dispatch("wave64_xor", "shared", "global")
    with pytest.raises(RuntimeError, match="shared/shared buffers"):
        _lower_forced_dispatch("wave64_xor", "global", "shared")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("shared_side", ["src", "dst"])
def test_mixed_shared_global_fallback(shared_side):
    shape = (4, 32)
    pre = TileLayout(S[shape : (32, 1)])
    post = TileLayout(S[shape : (1, 4)])

    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, "uint32", layout=pre)
        B_buf = T.match_buffer(B, shape, "uint32", layout=post)
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        T.lane_id([64])
        storage = T.alloc_buffer((128,), "uint32", scope="shared")
        shared_src = T.decl_buffer(shape, "uint32", data=storage.data, scope="shared", layout=pre)
        shared_dst = T.decl_buffer(shape, "uint32", data=storage.data, scope="shared", layout=post)
        if shared_side == "src":
            Tx.cta.copy(shared_src[:, :], A_buf[:, :])
            T.maca.cta_sync()
            Tx.warp.permute_layout(B_buf, shared_src)
        else:
            Tx.warp.permute_layout(shared_dst, A_buf)
            T.maca.cta_sync()
            Tx.cta.copy(B_buf[:, :], shared_dst[:, :])

    A_np = tvm.testing.generate_random_array("uint32", shape)
    B_np = np.zeros_like(A_np)
    [_, B_out], src = _compile_and_run(f, [A_np, B_np])
    ref = _expected_permute(A_np.reshape(-1), [32, 1], [1, 4], list(shape))
    np.testing.assert_array_equal(B_out.reshape(-1), ref)
    assert src.count("tvm_builtin_maca_warp_sync();") >= 2


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_shared_shared_uncertified_schedule_uses_generic_fallback():
    shape = (64, 32)
    pre = TileLayout(S[shape : (32, 1)])
    post = TileLayout(S[shape : (1, 64)])
    assert _choose_xor_k([64, 32], [32, 1], [1, 64], 4, 32, 64) is None

    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, "uint32", layout=pre)
        B_buf = T.match_buffer(B, shape, "uint32", layout=post)
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        T.lane_id([64])
        storage = T.alloc_buffer((2048,), "uint32", scope="shared")
        src_view = T.decl_buffer(shape, "uint32", data=storage.data, scope="shared", layout=pre)
        dst_view = T.decl_buffer(shape, "uint32", data=storage.data, scope="shared", layout=post)
        Tx.cta.copy(src_view[:, :], A_buf[:, :])
        T.maca.cta_sync()
        Tx.warp.permute_layout(dst_view, src_view)
        T.maca.cta_sync()
        Tx.cta.copy(B_buf[:, :], dst_view[:, :])

    A_np = tvm.testing.generate_random_array("uint32", shape)
    B_np = np.zeros_like(A_np)
    [_, B_out], src = _compile_and_run(f, [A_np, B_np])
    ref = _expected_permute(A_np.reshape(-1), [32, 1], [1, 64], list(shape))
    np.testing.assert_array_equal(B_out.reshape(-1), ref)
    assert src.count("tvm_builtin_maca_warp_sync();") >= 2


@pytest.mark.parametrize(
    "dtype, message",
    [
        ("float32x4", "scalar elements"),
        ("int128", "1, 2, 4, or 8-byte scalar elements"),
    ],
)
def test_reject_vector_and_128_bit_scalar(dtype, message):
    with pytest.raises(RuntimeError, match=message):
        _lower_forced_dispatch("wave64_generic", dtype=dtype)


if __name__ == "__main__":
    tvm.testing.main()
