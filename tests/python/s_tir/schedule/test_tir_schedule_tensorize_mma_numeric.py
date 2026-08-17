# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import te
from tvm.s_tir.tensor_intrin.maca import (
    MACA_MMA_F16F16F32_INTRIN,
    MACA_MMA_FILL_16x16_F32_INTRIN,
    MACA_MMA_LOAD_16x16_A_SHARED_F16_INTRIN,
    MACA_MMA_LOAD_16x16_B_SHARED_F16_INTRIN,
    MACA_MMA_STORE_16x16_F32_INTRIN,
    maca_mma_shared_16x16_to_local_64x4_layout_A,
    maca_mma_shared_16x16_to_local_64x4_layout_B,
    maca_mma_shared_16x16_to_local_64x4_layout_C,
)
from tvm.testing import env
from tvm.testing.tir import maca_mma_schedule


M = 1024
N = 1024
K = 1024

measure_perf = False
gflops = (N * M * K) * 2 / 1e9


def matmul(m, n, k, in_dtype, out_dtype, b_transposed):
    b_shape = (n, k) if b_transposed else (k, n)

    a = te.placeholder((m, k), name="A", dtype=in_dtype)
    b = te.placeholder(b_shape, name="B", dtype=in_dtype)
    k = te.reduce_axis((0, k), name="k")

    def maybe_cast(v):
        if in_dtype != out_dtype:
            return tvm.tirx.Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    c = te.compute(
        (m, n),
        lambda i, j: te.sum(
            maybe_cast(a[i, k]) * maybe_cast(b[maybe_swap(k, j)]),
            axis=[k],
        ),
        name="C",
    )

    return a, b, c


def run_test(
    k_inner,
    in_dtype,
    out_dtype,
    b_transposed,
    i_factors,
    j_factors,
    k_factors,
    index_map_A,
    index_map_B,
    index_map_C,
    load_a_intrin,
    load_b_intrin,
    mma_intrin,
    mma_fill_intrin,
    mma_store_intrin,
):
    sch = maca_mma_schedule(
        te.create_prim_func(
            matmul(M, N, K, in_dtype, out_dtype, b_transposed)
        ),
        k_inner,
        in_dtype,
        b_transposed,
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_B,
        index_map_C,
        load_a_intrin,
        load_b_intrin,
        mma_intrin,
        mma_fill_intrin,
        mma_store_intrin,
    )

    f = tvm.compile(sch.mod["main"], target="maca")

    if in_dtype == "float16":
        a_np = np.random.uniform(size=(M, K)).astype("float16")

        if b_transposed:
            b_np = np.random.uniform(size=(N, K)).astype("float16")
            c_np = np.dot(
                a_np.astype("float32"),
                b_np.astype("float32").transpose(),
            ).astype(out_dtype)
        else:
            b_np = np.random.uniform(size=(K, N)).astype("float16")
            c_np = np.dot(
                a_np.astype("float32"),
                b_np.astype("float32"),
            ).astype(out_dtype)

    elif in_dtype == "float32":
        a_np = np.random.uniform(size=(M, K)).astype("float32")

        if b_transposed:
            b_np = np.random.uniform(size=(N, K)).astype("float32")
            c_np = np.dot(
                a_np.astype("float32"),
                b_np.astype("float32").transpose(),
            ).astype(out_dtype)
        else:
            b_np = np.random.uniform(size=(K, N)).astype("float32")
            c_np = np.dot(
                a_np.astype("float32"),
                b_np.astype("float32"),
            ).astype(out_dtype)

    else:
        a_np = np.random.randint(-128, 128, (M, K)).astype(in_dtype)

        if b_transposed:
            b_np = np.random.randint(-128, 128, (N, K)).astype(in_dtype)
            c_np = np.dot(
                a_np.astype("float32"),
                b_np.astype("float32").transpose(),
            ).astype(out_dtype)
        else:
            b_np = np.random.randint(-128, 128, (K, N)).astype(in_dtype)
            c_np = np.dot(
                a_np.astype("float32"),
                b_np.astype("float32"),
            ).astype(out_dtype)

    def run_and_check(measure=False):
        dev = tvm.maca(0)

        a = tvm.runtime.tensor(a_np, dev)
        b = tvm.runtime.tensor(b_np, dev)
        c = tvm.runtime.tensor(
            np.zeros((M, N), dtype=out_dtype),
            dev,
        )

        if measure:
            return f.time_evaluator(
                f.entry_name,
                dev,
                number=500,
            )(a, b, c)

        f(a, b, c)
        dev.sync()

        tvm.testing.assert_allclose(
            c.numpy(),
            c_np,
            rtol=1e-2,
            atol=1e-2,
        )

    tvm.testing.run_with_gpu_lock(run_and_check)

    return lambda: tvm.testing.run_with_gpu_lock(run_and_check, True)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_f16f16f32_m16n16k16():
    def index_map_A(i, j):
        return (
            i // 16,
            j // 16,
            *maca_mma_shared_16x16_to_local_64x4_layout_A(
                i % 16,
                j % 16,
            ),
        )

    def index_map_B(i, j):
        return (
            i // 16,
            j // 16,
            *maca_mma_shared_16x16_to_local_64x4_layout_B(
                i % 16,
                j % 16,
            ),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *maca_mma_shared_16x16_to_local_64x4_layout_C(
                i % 16,
                j % 16,
            ),
        )

    k_inner = 16
    in_dtype = "float16"
    out_dtype = "float32"

    i_factors = [1, 8, 2, 4, 1]
    j_factors = [1, 16, 2, 1, 2]
    k_factors = [32, 2, 1]

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_B,
        index_map_C,
        MACA_MMA_LOAD_16x16_A_SHARED_F16_INTRIN,
        MACA_MMA_LOAD_16x16_B_SHARED_F16_INTRIN,
        MACA_MMA_F16F16F32_INTRIN,
        MACA_MMA_FILL_16x16_F32_INTRIN,
        MACA_MMA_STORE_16x16_F32_INTRIN,
    )

    if measure_perf and timer:
        print(
            "test_f16f16f32_m16n16k16: %f GFLOPS"
            % (gflops / timer().mean)
        )


if __name__ == "__main__":
    tvm.testing.main()