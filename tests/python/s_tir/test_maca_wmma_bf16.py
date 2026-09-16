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
"""BF16 MACA WMMA registration, source-generation, and hardware tests."""

from functools import cache

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import te
from tvm.s_tir.tensor_intrin.maca import (
    get_wmma_intrin_group,
    shared_16x16_to_local_64x4_layout_A,
    shared_16x16_to_local_64x4_layout_B,
    shared_16x16_to_local_64x4_layout_C,
)
from tvm.testing import env
from tvm.testing.tir import mfma_schedule
from tvm.tirx import TensorIntrin


@pytest.mark.parametrize(
    "trans_b, load_b, compute",
    [
        (
            False,
            "maca_wmma_load_16x16x16_bf16_b_shared",
            "maca_wmma_sync_16x16x16_bf16bf16f32",
        ),
        (
            True,
            "maca_wmma_load_16x16x16_bf16_b_trans_shared",
            "maca_wmma_sync_16x16x16_bf16bf16f32_trans",
        ),
    ],
)
def test_maca_bf16_wmma_group_names(trans_b, load_b, compute):
    group = get_wmma_intrin_group("shared", "shared", "bfloat16", "float32", trans_b)
    assert group == {
        "init": "maca_wmma_fill_16x16x16_f32",
        "load_a": "maca_wmma_load_16x16x16_bf16_a_shared",
        "load_b": load_b,
        "compute": compute,
        "store": "maca_wmma_store_16x16x16_f32_shared",
    }


@pytest.mark.parametrize(
    "name",
    [
        "maca_wmma_load_16x16x16_bf16_a_shared",
        "maca_wmma_load_16x16x16_bf16_b_shared",
        "maca_wmma_load_16x16x16_bf16_b_trans_shared",
        "maca_wmma_sync_16x16x16_bf16bf16f32",
        "maca_wmma_sync_16x16x16_bf16bf16f32_trans",
        "maca_wmma_fill_16x16x16_f32",
        "maca_wmma_store_16x16x16_f32_shared",
    ],
)
def test_maca_bf16_wmma_intrinsics_are_registered(name):
    assert TensorIntrin.get(name) is not None


def _make_maca_wmma_tile(dtype, trans_b):
    a = te.placeholder((16, 16), name="A", dtype=dtype)
    b = te.placeholder((16, 16), name="B", dtype=dtype)
    k = te.reduce_axis((0, 16), name="k")
    c = te.compute(
        (16, 16),
        lambda i, j: te.sum(
            a[i, k].astype("float32") * (b[j, k] if trans_b else b[k, j]).astype("float32"),
            axis=k,
        ),
        name="C",
    )
    group = get_wmma_intrin_group("shared", "global", dtype, "float32", trans_b)

    def index_map_A(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_A(i % 16, j % 16),
        )

    def index_map_B(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_B(i % 16, j % 16),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_C(i % 16, j % 16),
        )

    return mfma_schedule(
        te.create_prim_func((a, b, c)),
        16,
        dtype,
        trans_b,
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1],
        index_map_A,
        index_map_B,
        index_map_C,
        group["load_a"],
        group["load_b"],
        group["compute"],
        group["init"],
        group["store"],
    )


@cache
def _compile_maca_wmma_tile(dtype, trans_b):
    target = tvm.target.Target({"kind": "maca", "mcpu": "xcore1000"})
    return tvm.compile(_make_maca_wmma_tile(dtype, trans_b).mod["main"], target=target)


@pytest.mark.parametrize(
    "dtype, source_type",
    [("float16", "half"), ("bfloat16", "maca_bfloat16")],
)
@pytest.mark.parametrize("trans_b", [False, True])
def test_maca_wmma_tile_source_generation(dtype, source_type, trans_b):
    mod = _compile_maca_wmma_tile(dtype, trans_b)
    source = mod.mod.imports[0].inspect_source()

    assert "__launch_bounds__(64)" in source
    assert "__syncthreads()" in source
    assert source_type in source
    b_layout = "col_major" if trans_b else "row_major"
    assert (
        f"fragment<mxmaca::wmma::matrix_b, 16, 16, 16, {source_type}, mxmaca::wmma::{b_layout}>"
    ) in source
    for operation in [
        "mxmaca::wmma::fragment",
        "mxmaca::wmma::load_matrix_sync",
        "mxmaca::wmma::fill_fragment",
        "mxmaca::wmma::mma_sync",
        "mxmaca::wmma::store_matrix_sync",
    ]:
        assert operation in source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca_compute(10, 0, exact=True), reason="need C500 (xcore1000)")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_maca_wmma_tile_hardware(dtype):
    mod = _compile_maca_wmma_tile(dtype, False)
    rng = np.random.default_rng(0)
    a_np = rng.uniform(-1, 1, (16, 16)).astype(dtype)
    b_np = rng.uniform(-1, 1, (16, 16)).astype(dtype)
    expected = a_np.astype("float32") @ b_np.astype("float32")

    def run_and_check():
        device = tvm.maca(0)
        a = tvm.runtime.tensor(a_np, device=device)
        b = tvm.runtime.tensor(b_np, device=device)
        c = tvm.runtime.empty((16, 16), dtype="float32", device=device)
        mod(a, b, c)
        device.sync()
        tvm.testing.assert_allclose(c.numpy(), expected, rtol=1e-2, atol=1e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)
