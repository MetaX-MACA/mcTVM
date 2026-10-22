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
from tvm.backend.maca.tile.instructions import make_namespace
from tvm.backend.maca.tile.permute_layout.wave64_halfwarp_xor_swizzle import (
    _choose_xor_k,
)
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.tile.instruction import SPECS


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
    @T.function
    def f(
        A: T.Tensor(shape, dtype, layout=pre),
        B: T.Tensor(shape, dtype, layout=post),
    ):
        A_buf = A
        B_buf = B
        T.device_entry(launch=T.maca.LaunchConfig(grid=(1,), block=(64,)))
        storage = T.alloc_tensor(ty_args=[T.Tensor((128,), dtype, scope="shared")])
        src_view = T.decl_tensor(
            storage.data,
            ty_args=[T.Tensor(shape, dtype, scope="shared", layout=pre, data=storage.data)],
        )
        dst_view = T.decl_tensor(
            storage.data,
            ty_args=[T.Tensor(shape, dtype, scope="shared", layout=post, data=storage.data)],
        )
        T.maca.tile.copy(src_view[:, :], A_buf[:, :], scope="cta")
        T.maca.cta_sync()
        T.maca.tile.permute_layout(dst_view[:, :], src_view[:, :], scope="warp")
        T.maca.cta_sync()
        T.maca.tile.copy(B_buf[:, :], dst_view[:, :], scope="cta")
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

    @T.function
    def f(
        A: T.Tensor(shape, dtype, layout=pre),
        B: T.Tensor(shape, dtype, layout=post),
    ):
        A_buf = A
        B_buf = B
        T.device_entry(launch=T.maca.LaunchConfig(grid=(1,), block=(64,)))
        T.maca.tile.permute_layout(B_buf, A_buf, scope="warp")

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

    @T.function
    def f(
        A: T.Tensor(shape, "uint16", layout=layout),
        B: T.Tensor(shape, "uint16", layout=layout),
    ):
        A_buf = A
        B_buf = B
        T.device_entry(launch=T.maca.LaunchConfig(grid=(1,), block=(64,)))
        T.maca.tile.permute_layout(B_buf, A_buf, scope="warp")

    np.random.seed(volume)
    A_np = tvm.testing.generate_random_array("uint16", shape)
    B_np = np.zeros_like(A_np)
    [_, B_out], src = _compile_and_run(f, [A_np, B_np])
    np.testing.assert_array_equal(B_out, A_np)
    assert "tvm_builtin_maca_warp_sync();" in src


def test_permute_layout_schedule_is_registered_for_maca():
    namespace = make_namespace()
    assert namespace.permute_layout.__name__ == "permute_layout"
    assert SPECS["tirx.maca.tile.permute_layout"].kind == "permute_layout"


def test_reject_unsupported_maca_architecture():
    shape = (4, 32)
    pre = TileLayout(S[shape : (32, 1)])
    post = TileLayout(S[shape : (1, 4)])

    @T.function
    def f(
        A: T.Tensor(shape, "uint32", layout=pre),
        B: T.Tensor(shape, "uint32", layout=post),
    ):
        A_buf = A
        B_buf = B
        T.device_entry(launch=T.maca.LaunchConfig(grid=(1,), block=(64,)))
        T.maca.tile.permute_layout(B_buf, A_buf, scope="warp")

    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore9999"})
    with target, pytest.raises(RuntimeError) as exc_info:
        tvm.compile(tvm.IRModule({"main": f}), target=target, tir_pipeline="tirx")
    assert "MACA mcpu 'xcore9999' is not one of ('xcore1000',)" in str(exc_info.value)


def test_choose_xor_k_rejects_out_of_contract_slots():
    assert _choose_xor_k([64, 32], [32, 1], [1, 64], 4, 64, 64) is None


def _lower_permute_layout(src_scope="global", dst_scope="global", dtype="uint32"):
    shape = (4, 32)
    pre = TileLayout(S[shape : (32, 1)])
    post = TileLayout(S[shape : (1, 4)])

    @T.function
    def f(
        A: T.Tensor(shape, dtype, scope=src_scope, layout=pre),
        B: T.Tensor(shape, dtype, scope=dst_scope, layout=post),
    ):
        T.device_entry(launch=T.maca.LaunchConfig(grid=(1,), block=(64,)))
        T.maca.tile.permute_layout(B, A, scope="warp")

    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    with target:
        return tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": f}))


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
def test_reject_thread_private_storage(src_scope, dst_scope):
    with pytest.raises(RuntimeError, match="requires global or shared storage"):
        _lower_permute_layout(src_scope, dst_scope)


@pytest.mark.parametrize("src_scope", ["global", "shared"])
@pytest.mark.parametrize("dst_scope", ["global", "shared"])
def test_supported_storage_scopes_lower(src_scope, dst_scope):
    _lower_permute_layout(src_scope, dst_scope)


def test_xor_dispatch_requires_shared_shared_and_nonzero_schedule():
    assert _choose_xor_k([4, 32], [32, 1], [1, 4], 4, 4, 32) == 2


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("shared_side", ["src", "dst"])
def test_mixed_shared_global_fallback(shared_side):
    shape = (4, 32)
    pre = TileLayout(S[shape : (32, 1)])
    post = TileLayout(S[shape : (1, 4)])
    shared_source = shared_side == "src"

    @T.function
    def f(
        A: T.Tensor(shape, "uint32", layout=pre),
        B: T.Tensor(shape, "uint32", layout=post),
    ):
        A_buf = A
        B_buf = B
        T.device_entry(launch=T.maca.LaunchConfig(grid=(1,), block=(64,)))
        storage = T.alloc_tensor(ty_args=[T.Tensor((128,), "uint32", scope="shared")])
        shared_src = T.decl_tensor(
            storage.data,
            ty_args=[T.Tensor(shape, "uint32", data=storage.data, scope="shared", layout=pre)],
        )
        shared_dst = T.decl_tensor(
            storage.data,
            ty_args=[T.Tensor(shape, "uint32", data=storage.data, scope="shared", layout=post)],
        )
        if T.meta_var(shared_source):
            T.maca.tile.copy(shared_src[:, :], A_buf[:, :], scope="cta")
            T.maca.cta_sync()
            T.maca.tile.permute_layout(B_buf, shared_src, scope="warp")
        else:
            T.maca.tile.permute_layout(shared_dst, A_buf, scope="warp")
            T.maca.cta_sync()
            T.maca.tile.copy(B_buf[:, :], shared_dst[:, :], scope="cta")

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

    @T.function
    def f(
        A: T.Tensor(shape, "uint32", layout=pre),
        B: T.Tensor(shape, "uint32", layout=post),
    ):
        A_buf = A
        B_buf = B
        T.device_entry(launch=T.maca.LaunchConfig(grid=(1,), block=(64,)))
        storage = T.alloc_tensor(ty_args=[T.Tensor((2048,), "uint32", scope="shared")])
        src_view = T.decl_tensor(
            storage.data,
            ty_args=[T.Tensor(shape, "uint32", data=storage.data, scope="shared", layout=pre)],
        )
        dst_view = T.decl_tensor(
            storage.data,
            ty_args=[T.Tensor(shape, "uint32", data=storage.data, scope="shared", layout=post)],
        )
        T.maca.tile.copy(src_view[:, :], A_buf[:, :], scope="cta")
        T.maca.cta_sync()
        T.maca.tile.permute_layout(dst_view, src_view, scope="warp")
        T.maca.cta_sync()
        T.maca.tile.copy(B_buf[:, :], dst_view[:, :], scope="cta")

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
        _lower_permute_layout(dtype=dtype)


if __name__ == "__main__":
    tvm.testing.main()
