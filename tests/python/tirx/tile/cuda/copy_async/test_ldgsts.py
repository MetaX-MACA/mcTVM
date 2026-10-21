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
# pylint: disable=invalid-name, missing-function-docstring
"""Tests for the non-bulk CTA-level copy_async dispatch (vectorized load)."""

import math

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout

pytestmark = pytest.mark.skipif(
    not env.has_maca_arch("xcore1000"),
    reason="requires MACA xcore1000",
)


@pytest.mark.parametrize(
    "task",
    [
        ################ A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32] ################
        (
            (128, 32),  # g_shape
            (128, 32),  # s_shape
            (0, 0),  # g_st
            (128, 32),  # g_extent
            64,  # thread_cnt
            TileLayout(S[128, 32]),  # layoutA
            TileLayout(S[128, 32]),  # layoutB
            TileLayout(S[128, 32]),  # layoutS
        ),
        ################ A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64] ################
        (
            (64, 64),  # g_shape
            (32, 32),  # s_shape
            (32, 0),  # g_st
            (32, 32),  # g_extent
            64,  # thread_cnt
            TileLayout(S[64, 64]),  # layoutA
            TileLayout(S[64, 64]),  # layoutB
            TileLayout(S[32, 32]),  # layoutS
        ),
        ################ A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32] ################  # noqa: E501
        (
            (4, 32, 32),  # g_shape
            (32, 32),  # s_shape
            (0, 0, 0),  # g_st
            (1, 32, 32),  # g_extent
            64,  # thread_cnt
            TileLayout(S[4, 32, 32]),  # layoutA
            TileLayout(S[4, 32, 32]),  # layoutB
            TileLayout(S[32, 32]),  # layoutS
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
def test_copy_g2s_s2g_cta_vec_load(task, dtype):
    g_shape, s_shape, g_st, g_extent, thread_cnt, layoutA, layoutB, layoutS = task

    r_smem = list(slice(None) for i in range(len(s_shape)))
    r_gmem = list(slice(g_st[i], g_st[i] + g_extent[i]) for i in range(len(g_shape)))

    # fmt: off
    @T.function
    def copy_async(
        A: T.Tensor(g_shape, dtype, layout=layoutA), B: T.Tensor(g_shape, dtype, layout=layoutB)
    ) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(thread_cnt,)))
        cta_id = T.cuda.block_idx('x')
        tid = T.cuda.thread_idx('x')
        A_smem = T.alloc_tensor(ty_args=[T.Tensor(s_shape, dtype, scope="shared", layout=layoutS)])

        T.cuda.tile.cp_async(A_smem[tuple(r_smem)], A[tuple(r_gmem)], scope='cta')
        T.ptx.cp.async_.commit_group()
        T.ptx.cp.async_.wait_group(0)
        T.cuda.cta_sync()
        copy_src_1 = T.meta_var(A_smem[tuple(r_smem)])
        copy_src_tensor_1 = T.meta_var(copy_src_1.source)
        copy_dst_1 = T.meta_var(B[tuple(r_gmem)])
        copy_dst_tensor_1 = T.meta_var(copy_dst_1.source)
        for copy_step_1 in T.serial(
            T.ceildiv(math.prod([int(r.extent) for r in copy_src_1.region]), thread_cnt)
        ):
            copy_index_1 = copy_step_1 * (thread_cnt) + (tid)
            if copy_index_1 < math.prod([int(r.extent) for r in copy_src_1.region]):
                copy_value_1 = copy_src_tensor_1[
                    tuple(
                        [
                            copy_src_1.region[k].min
                            + copy_index_1
                            // math.prod([int(s.extent) for s in copy_src_1.region[k + 1 :]])
                            % copy_src_1.region[k].extent
                            for k in range(len(copy_src_1.region))
                        ]
                    )
                ]
                copy_dst_index_1 = T.meta_var(
                    tuple(
                        [
                            copy_dst_1.region[k].min
                            + copy_index_1
                            // math.prod([int(s.extent) for s in copy_dst_1.region[k + 1 :]])
                            % copy_dst_1.region[k].extent
                            for k in range(len(copy_dst_1.region))
                        ]
                    )
                )
                copy_dst_tensor_1[copy_dst_index_1] = copy_value_1
        # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("maca")
    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(np_dtype)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        B_ref = B_np.copy()
        B_ref[tuple(r_gmem)] = A_np[tuple(r_gmem)]

        def run_and_check():
            dev = tvm.maca(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            np.testing.assert_allclose(B_ref.astype("float32"), B.numpy().astype("float32"))

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.skipif(not env.has_maca(), reason="need maca")
def test_copy_ldgsts_predicate_zero_fill_codegen():
    """ldgsts forwards the predicate to MACA's zero-fill BSM intrinsic."""

    @T.function
    def copy_async(
        A: T.Tensor((32, 16), "uint8", layout=TileLayout(S[32, 16])),
        B: T.Tensor((32, 16), "uint8", layout=TileLayout(S[32, 16])),
    ) -> None:
        T.device_entry(launch=T.maca.LaunchConfig(grid=1, block=(32,)))
        tid = T.maca.thread_idx("x")
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor((32, 16), "uint8", scope="shared", layout=TileLayout(S[32, 16]))]
        )

        T.maca.tile.copy_async(
            A_smem[:, :],
            A[:, :],
            predicate=tid < 16,
            fill_mode="zero",
            scope="cta",
        )
        T.maca.async_wait_gvmcnt(0)
        T.maca.barrier_inst()
        T.maca.cta_sync()
        T.maca.tile.copy(B[:, :], A_smem[:, :], scope="cta")

    target = tvm.target.Target("maca")
    with target:
        mod = tvm.tirx.compile(tvm.IRModule({"main": copy_async}), target=target, pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "__builtin_mxc_ldg_b128_bsm_predicator" in src
    assert "predicate" in src
    assert "s_ptr_ptr" not in src
    assert "g_ptr_ptr" not in src

    A_np = np.arange(32 * 16, dtype="uint8").reshape(32, 16)
    B_np = np.full_like(A_np, 0xA5)

    def run_and_check():
        dev = tvm.maca(0)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)
        expected = np.zeros_like(A_np)
        expected[:16, :] = A_np[:16, :]
        np.testing.assert_array_equal(B.numpy(), expected)

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
