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
# pylint: disable=missing-docstring
import pytest

import tvm.testing
from tvm import dlight as dl
from tvm.script import tir as T
from tvm.target import Target


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    @pytest.fixture
    def transform(self):
        def transform(mod):
            with Target("nvidia/geforce-gtx-1080-ti"):
                # Use Matmul rule for Conv for now
                return dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)

        return transform


class TestConv3d(BaseBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(
        A: T.Buffer((14308, 3, 2, 14, 14), "float16"),
        W: T.Buffer((1280, 3, 2, 14, 14), "float16"),
        C: T.Buffer((14308, 1280, 1, 1, 1), "float16"),
    ):
        pad_A = T.alloc_buffer((14308, 3, 2, 14, 14), "float16")
        for i0, i1, i2, i3, i4 in T.grid(14308, 3, 2, 14, 14):
            with T.block("pad_A"):
                v_i0, v_i1, v_i2, v_i3, v_i4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                pad_A[v_i0, v_i1, v_i2, v_i3, v_i4] = A[v_i0, v_i1, v_i2, v_i3, v_i4]
        for nn, ff, yy, xx, zz, rc, ry, rx, rz in T.grid(14308, 1280, 1, 1, 1, 3, 2, 14, 14):
            with T.block("C"):
                v_nn, v_ff, v_yy, v_xx, v_zz, v_rc, v_ry, v_rx, v_rz = T.axis.remap("SSSSSRRRR", [nn, ff, yy, xx, zz, rc, ry, rx, rz])
                with T.init():
                    C[v_nn, v_ff, v_yy, v_xx, v_zz] = T.float16(0.0)
                C[v_nn, v_ff, v_yy, v_xx, v_zz] += pad_A[v_nn, v_rc, v_yy * 2 + v_ry, v_xx * 14 + v_rx, v_zz * 14 + v_rz]* W[v_ff, v_rc, v_ry, v_rx, v_rz]

    @T.prim_func
    def expected(A: T.Buffer((14308, 3, 2, 14, 14), "float16"), W: T.Buffer((1280, 3, 2, 14, 14), "float16"), C: T.Buffer((14308, 1280, 1, 1, 1), "float16")):
        T.func_attr({"tir.is_scheduled": True})
        # with T.block("root"):
        C_reindex_pad_local = T.alloc_buffer((1, 14336, 1280), "float16", scope="local")
        pad_A_reindex_pad_shared = T.alloc_buffer((1, 14336, 1184), "float16", scope="shared")
        W_reindex_pad_shared = T.alloc_buffer((1, 1280, 1184), "float16", scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(20, thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(448, thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(1, thread="vthread.y"):
                    for ax1_1 in T.thread_binding(1, thread="vthread.x"):
                        for ax2_2 in T.thread_binding(16, thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(8, thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(4, 2):
                                    for ax2_3_1_init in T.vectorized(2):
                                        with T.block("C_init"):
                                            v0 = T.axis.spatial(1, 0)
                                            v1 = T.axis.spatial(14336, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3_init)
                                            v2 = T.axis.spatial(1280, ax0_ax2_0_fused * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_0_init * 2 + ax2_3_1_init)
                                            C_reindex_pad_local[0, v1, v2] = T.float16(0.0)
                                for ax3_0 in range(74):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(2):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with T.block("pad_A_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(14336, ax1_0 * 32 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = T.axis.spatial(1184, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_A_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < 14308 and v2 < 1176, A[v1, v2 // 392, v2 // 196 % 2, v2 // 14 % 14, v2 % 14], T.float16(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(4):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with T.block("W_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(1280, ax0_ax2_0_fused * 64 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = T.axis.spatial(1184, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        W_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v2 < 1176, W[v1, v2 // 392, v2 // 196 % 2, v2 // 14 % 14, v2 % 14], T.float16(0.0))
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(16, 4, 2):
                                        for ax2_3_1 in T.vectorized(2):
                                            with T.block("C_update"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(14336, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3)
                                                v2 = T.axis.spatial(1280, ax0_ax2_0_fused * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_0 * 2 + ax2_3_1)
                                                v3 = T.axis.reduce(1184, ax3_0 * 16 + ax3_1)
                                                C_reindex_pad_local[0, v1, v2] = C_reindex_pad_local[0, v1, v2] + pad_A_reindex_pad_shared[0, v1, v3] * W_reindex_pad_shared[0, v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(1, 4, 2):
                                    for ax2_1_1 in T.vectorized(2):
                                        with T.block("C_reindex_pad_local"):
                                            v0 = T.axis.spatial(1, ax0)
                                            v1 = T.axis.spatial(14336, ax1_0 * 32 + ax1_2 * 4 + ax1)
                                            v2 = T.axis.spatial(1280, ax0_ax2_0_fused * 64 + ax2_2 * 4 + ax2_0 * 2 + ax2_1_1)
                                            T.where(ax1_0 * 32 + ax1_2 * 4 + ax1 < 14308)
                                            C[v1, v2, 0, 0, 0] = C_reindex_pad_local[v0, v1, v2]
    # fmt: on


class MACABeforeAfter(tvm.testing.CompareBeforeAfter):
    @pytest.fixture
    def transform(self):
        def transform(mod):
            with Target("maca"):
                # Use Matmul rule for Conv for now
                return dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)

        return transform


@tvm.testing.requires_maca
class TestConv3dMACA(MACABeforeAfter):
    # fmt: off
    @T.prim_func
    def before(
        A: T.Buffer((14308, 3, 2, 14, 14), "float16"),
        W: T.Buffer((1280, 3, 2, 14, 14), "float16"),
        C: T.Buffer((14308, 1280, 1, 1, 1), "float16"),
    ):
        pad_A = T.alloc_buffer((14308, 3, 2, 14, 14), "float16")
        for i0, i1, i2, i3, i4 in T.grid(14308, 3, 2, 14, 14):
            with T.block("pad_A"):
                v_i0, v_i1, v_i2, v_i3, v_i4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                pad_A[v_i0, v_i1, v_i2, v_i3, v_i4] = A[v_i0, v_i1, v_i2, v_i3, v_i4]
        for nn, ff, yy, xx, zz, rc, ry, rx, rz in T.grid(14308, 1280, 1, 1, 1, 3, 2, 14, 14):
            with T.block("C"):
                v_nn, v_ff, v_yy, v_xx, v_zz, v_rc, v_ry, v_rx, v_rz = T.axis.remap("SSSSSRRRR", [nn, ff, yy, xx, zz, rc, ry, rx, rz])
                with T.init():
                    C[v_nn, v_ff, v_yy, v_xx, v_zz] = T.float16(0.0)
                C[v_nn, v_ff, v_yy, v_xx, v_zz] += pad_A[v_nn, v_rc, v_yy * 2 + v_ry, v_xx * 14 + v_rx, v_zz * 14 + v_rz]* W[v_ff, v_rc, v_ry, v_rx, v_rz]

    @T.prim_func
    def expected(A: T.Buffer((14308, 3, 2, 14, 14), "float16"), W: T.Buffer((1280, 3, 2, 14, 14), "float16"), C: T.Buffer((14308, 1280, 1, 1, 1), "float16")):
        T.func_attr({"global_symbol": "before", "tir.is_scheduled": True})
        # with T.block("root"):
        pad_A_reindex_pad_shared_dyn = T.alloc_buffer((1, 14336, 1184), "float16", scope="shared.dyn")
        W_reindex_pad_shared_dyn = T.alloc_buffer((1, 1280, 1184), "float16", scope="shared.dyn")
        pad_A_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((1, 14336, 1184), "float16", scope="wmma.matrix_a")
        W_reindex_pad_shared_dyn_wmma_matrix_b = T.alloc_buffer((1, 1280, 1184), "float16", scope="wmma.matrix_b")
        C_reindex_pad_shared_dyn = T.alloc_buffer((1, 14336, 1280), "float16", scope="shared.dyn")
        C_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((1, 14336, 1280), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(224, thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(20, thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(4, thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(2, 2):
                            with T.block("C_o_init"):
                                v0_o = T.axis.spatial(1, ax0)
                                v1_o = T.axis.spatial(896, ax1_0_0_ax2_0_0_fused * 4 + ax2_0_2_ax1_0_2_fused % 2 * 2 + ax1_0_3_init)
                                v2_o = T.axis.spatial(80, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2_ax1_0_2_fused // 2 * 2 + ax2_0_3_init)
                                T.reads()
                                T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                with T.block("C_init_o"):
                                    v1_i_init_o = T.axis.spatial(1, 0)
                                    v2_i_init_o = T.axis.spatial(1, 0)
                                    T.reads()
                                    T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                    C_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // 16 * (C_1.strides[0] // 16) + C_1.elem_offset % C_1.strides[0] // 16, T.float32(0.0))
                        for ax3_0_0 in T.serial(37, annotations={"software_pipeline_order": [0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage": [0, 0, 0, 0, 0, 1, 1]}):
                            for ax0_ax1_fused_0 in range(2):
                                for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(64, thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(4):
                                            with T.block("pad_A_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(14336, ax1_0_0_ax2_0_0_fused * 64 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 32)
                                                v2 = T.axis.spatial(1184, ax3_0_0 * 32 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 32)
                                                T.reads(A[v1, v2 // 392, v2 // 196 % 2, v2 // 14 % 14, v2 % 14])
                                                T.writes(pad_A_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                pad_A_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < 14308 and v2 < 1176, A[v1, v2 // 392, v2 // 196 % 2, v2 // 14 % 14, v2 % 14], T.float16(0.0))
                            for ax0_ax1_fused_0 in range(2):
                                for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(64, thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(4):
                                            with T.block("W_reindex_pad_shared.dyn"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(1280, ax1_0_1_ax2_0_1_fused * 64 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 32)
                                                v2 = T.axis.spatial(1184, ax3_0_0 * 32 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 256 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 32)
                                                T.reads(W[v1, v2 // 392, v2 // 196 % 2, v2 // 14 % 14, v2 % 14])
                                                T.writes(W_reindex_pad_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                W_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v2 < 1176, W[v1, v2 // 392, v2 // 196 % 2, v2 // 14 % 14, v2 % 14], T.float16(0.0))
                            for ax3_0_1 in T.serial(2, annotations={"software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                for ax0_0 in T.unroll(2):
                                    for ax1_0 in T.unroll(1):
                                        with T.block("pad_A_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(896, ax1_0_0_ax2_0_0_fused * 4 + ax2_0_2_ax1_0_2_fused % 2 * 2 + ax0_0)
                                            v2_o = T.axis.spatial(74, ax3_0_0 * 2 + ax3_0_1 + ax1_0)
                                            T.reads(pad_A_reindex_pad_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(pad_A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            A_1 = T.match_buffer(pad_A_reindex_pad_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C_1 = T.match_buffer(pad_A_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // 16 * (C_1.strides[0] // 16) + C_1.elem_offset % C_1.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * 16, 1), A_1.strides[0], "row_major")
                                for ax0_0 in T.unroll(2):
                                    for ax1_0 in T.unroll(1):
                                        with T.block("W_reindex_pad_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(80, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2_ax1_0_2_fused // 2 * 2 + ax0_0)
                                            v2_o = T.axis.spatial(74, ax3_0_0 * 2 + ax3_0_1 + ax1_0)
                                            T.reads(W_reindex_pad_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(W_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            A_1 = T.match_buffer(W_reindex_pad_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C_1 = T.match_buffer(W_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C_1.data, 16, 16, 16, C_1.elem_offset // C_1.strides[0] // 16 * (C_1.strides[0] // 16) + C_1.elem_offset % C_1.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * 16, 1), A_1.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(2, 2):
                                    with T.block("C_o_update"):
                                        v0_o = T.axis.spatial(1, ax0)
                                        v1_o = T.axis.spatial(896, ax1_0_0_ax2_0_0_fused * 4 + ax2_0_2_ax1_0_2_fused % 2 * 2 + ax1_0_3)
                                        v2_o = T.axis.spatial(80, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2_ax1_0_2_fused // 2 * 2 + ax2_0_3)
                                        v3_o = T.axis.reduce(74, ax3_0_0 * 2 + ax3_0_1)
                                        T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], pad_A_reindex_pad_shared_dyn_wmma_matrix_a[0, v1_o * 16:v1_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], W_reindex_pad_shared_dyn_wmma_matrix_b[0, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                        with T.block("C_o"):
                                            v1_i_o = T.axis.spatial(1, 0)
                                            v2_i_o = T.axis.spatial(1, 0)
                                            v3_i_o = T.axis.reduce(1, 0)
                                            T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], pad_A_reindex_pad_shared_dyn_wmma_matrix_a[0, v1_o * 16:v1_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], W_reindex_pad_shared_dyn_wmma_matrix_b[0, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                            T.writes(C_reindex_pad_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            A_1 = T.match_buffer(pad_A_reindex_pad_shared_dyn_wmma_matrix_a[0, v1_o * 16:v1_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(W_reindex_pad_shared_dyn_wmma_matrix_b[0, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], (16, 16), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C_1.data, C_1.elem_offset // C_1.strides[0] // 16 * (C_1.strides[0] // 16) + C_1.elem_offset % C_1.strides[0] // 16, A_1.data, A_1.elem_offset // A_1.strides[0] // 16 * (A_1.strides[0] // 16) + A_1.elem_offset % A_1.strides[0] // 16, B.data, B.elem_offset // B.strides[0] // 16 * (B.strides[0] // 16) + B.elem_offset % B.strides[0] // 16, C_1.data, C_1.elem_offset // C_1.strides[0] // 16 * (C_1.strides[0] // 16) + C_1.elem_offset % C_1.strides[0] // 16)
                        for ax0_0, ax1_0 in T.grid(2, 2):
                            with T.block("C_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(1, 0)
                                v1_o = T.axis.spatial(896, ax1_0_0_ax2_0_0_fused * 4 + ax2_0_2_ax1_0_2_fused % 2 * 2 + ax0_0)
                                v2_o = T.axis.spatial(80, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2_ax1_0_2_fused // 2 * 2 + ax1_0)
                                T.reads(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                T.writes(C_reindex_pad_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                A_1 = T.match_buffer(C_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C_1 = T.match_buffer(C_reindex_pad_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // 16 * (A_1.strides[0] // 16) + A_1.elem_offset % A_1.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * 16, 2), C_1.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(4):
                            for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("C_reindex_pad_shared.dyn"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(14336, ax1_0_0_ax2_0_0_fused * 64 + ax2_0_2_ax1_0_2_fused % 2 * 32 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 32)
                                        v2 = T.axis.spatial(1280, ax1_0_1_ax2_0_1_fused * 64 + ax2_0_2_ax1_0_2_fused // 2 * 32 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 32)
                                        T.where(ax1_0_0_ax2_0_0_fused * 64 + ax2_0_2_ax1_0_2_fused % 2 * 32 + ((ax0_ax1_fused_0 * 64 + ax0_ax1_fused_1) * 4 + ax0_ax1_fused_2) // 32 < 14308)
                                        T.reads(C_reindex_pad_shared_dyn[v0, v1, v2])
                                        T.writes(C[v1, v2, 0, 0, 0])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        C[v1, v2, 0, 0, 0] = C_reindex_pad_shared_dyn[v0, v1, v2]
    # fmt: on


if __name__ == "__main__":
    tvm.testing.main()
