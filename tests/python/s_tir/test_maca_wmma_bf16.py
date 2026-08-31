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
"""BF16 MACA WMMA registration and source-generation tests."""

import pytest

import tvm
from tvm.script import tirx as T
from tvm.s_tir.tensor_intrin.maca import get_wmma_intrin_group
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


def _make_maca_wmma_tile(dtype):
    @T.prim_func(s_tir=True)
    def main(
        A: T.Buffer((16, 16), dtype),
        B: T.Buffer((16, 16), dtype),
        C: T.Buffer((16, 16), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        A_shared = T.sblock_alloc_buffer((16, 16), dtype, scope="shared", align=64)
        B_shared = T.sblock_alloc_buffer((16, 16), dtype, scope="shared", align=64)
        A_frag = T.sblock_alloc_buffer((16, 16), dtype, scope="wmma.matrix_a", align=64)
        B_frag = T.sblock_alloc_buffer((16, 16), dtype, scope="wmma.matrix_b", align=64)
        C_frag = T.sblock_alloc_buffer((16, 16), "float32", scope="wmma.accumulator", align=64)
        for bx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(64, thread="threadIdx.x"):
                for i in range(4):
                    with T.sblock("copy"):
                        v = T.axis.spatial(256, tx * 4 + i)
                        A_shared[v // 16, v % 16] = A[v // 16, v % 16]
                        B_shared[v // 16, v % 16] = B[v // 16, v % 16]
                T.tvm_storage_sync("shared")
                T.tvm_load_matrix_sync(
                    A_frag.data,
                    16,
                    16,
                    16,
                    0,
                    T.tvm_access_ptr(T.type_annotation(dtype), A_shared.data, 0, 256, 1),
                    16,
                    "row_major",
                )
                T.tvm_load_matrix_sync(
                    B_frag.data,
                    16,
                    16,
                    16,
                    0,
                    T.tvm_access_ptr(T.type_annotation(dtype), B_shared.data, 0, 256, 1),
                    16,
                    "row_major",
                )
                T.tvm_fill_fragment(C_frag.data, 16, 16, 16, 0, T.float32(0))
                T.tvm_mma_sync(C_frag.data, 0, A_frag.data, 0, B_frag.data, 0, C_frag.data, 0)
                T.tvm_store_matrix_sync(
                    C_frag.data,
                    16,
                    16,
                    16,
                    0,
                    T.tvm_access_ptr(T.type_annotation("float32"), C.data, 0, 256, 2),
                    16,
                    "row_major",
                )

    return main


@pytest.mark.parametrize(
    "dtype, source_type",
    [("float16", "half"), ("bfloat16", "maca_bfloat16")],
)
def test_maca_wmma_tile_source_generation(dtype, source_type):
    ir_mod = tvm.s_tir.transform.InferFragment()(
        tvm.IRModule({"main": _make_maca_wmma_tile(dtype)})
    )
    mod = tvm.compile(ir_mod, target="maca")
    source = mod.mod.imports[0].inspect_source()

    assert "__launch_bounds__(64)" in source
    assert "__syncthreads()" in source
    assert source_type in source
    for operation in [
        "mxmaca::wmma::fragment",
        "mxmaca::wmma::load_matrix_sync",
        "mxmaca::wmma::fill_fragment",
        "mxmaca::wmma::mma_sync",
        "mxmaca::wmma::store_matrix_sync",
    ]:
        assert operation in source
