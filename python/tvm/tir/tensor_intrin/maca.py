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
# pylint: disable=invalid-name,missing-function-docstring,unused-variable
"""Intrinsics for tensorization on MetaX GPU."""
from typing import Dict, Literal, Optional, Tuple

from tvm._ffi import register_func
from tvm.runtime import convert
from tvm.script import tir as T
from tvm.tir import Cast, IntImm, TensorIntrin
from tvm.tir.function import PrimFunc

######## WMMA intrinsics ########


def get_wmma_fragment_index(buffer, stride, m_dim, n_dim):
    """Compute wmma fragment index using elem_offset of the buffer"""
    frag_index_m = buffer.elem_offset // stride // m_dim
    frag_index_n = buffer.elem_offset % stride // n_dim

    num_fragments_per_row = stride // n_dim
    return frag_index_m * num_fragments_per_row + frag_index_n


def get_wmma_load_intrin(
    m_dim: int,
    n_dim: int,
    k_dim: int,
    dtype: str,
    shared_scope: str,
    is_b: bool,
    is_col_major: bool,
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_load intrins"""
    wmma_fragment_scope = f"wmma.matrix_{'b' if is_b else 'a'}"
    layout = "col_major" if is_col_major else "row_major"

    frag_m, frag_n = (k_dim, n_dim) if is_b else (m_dim, k_dim)
    if is_col_major:
        frag_m, frag_n = frag_n, frag_m
    offset_factor = frag_n

    @T.prim_func
    def wmma_load_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (frag_m, frag_n), dtype, align=64, offset_factor=offset_factor, scope=shared_scope
        )
        C = T.match_buffer(
            c,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=wmma_fragment_scope,
        )
        with T.block("root"):
            T.reads(A[0:frag_m, 0:frag_n])
            T.writes(C[0:frag_m, 0:frag_n])
            for i, j in T.grid(frag_m, frag_n):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_load_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.int32()
        s0 = T.int32()
        d1 = T.int32()
        d0 = T.int32()
        A = T.match_buffer(
            a,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=shared_scope,
            strides=[s1, s0],
        )
        C = T.match_buffer(
            c,
            (frag_m, frag_n),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=wmma_fragment_scope,
            strides=[d1, d0],
        )
        with T.block("root"):
            T.reads(A[0:frag_m, 0:frag_n])
            T.writes(C[0:frag_m, 0:frag_n])
            T.evaluate(
                T.tvm_load_matrix_sync(
                    C.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_wmma_fragment_index(C, d1, frag_m, frag_n),
                    A.access_ptr("r"),
                    s1,
                    layout,
                    dtype="handle",
                )
            )

    return wmma_load_desc, wmma_load_impl


def get_wmma_fill_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_fill intrins"""
    zero = IntImm("int32", 0).astype(dtype)
    offset_factor = n_dim

    @T.prim_func
    def wmma_fill_desc(c: T.handle) -> None:
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("init"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = zero

    @T.prim_func
    def wmma_fill_impl(c: T.handle) -> None:
        d1 = T.int32()
        d0 = T.int32()
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
            strides=[d1, d0],
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_fill_fragment(
                    C.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_wmma_fragment_index(C, d1, m_dim, n_dim),
                    T.float32(0),
                    dtype="handle",
                )
            )

    return wmma_fill_desc, wmma_fill_impl


def get_wmma_store_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str, scope: str
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_store intrins"""
    offset_factor = n_dim

    @T.prim_func
    def wmma_store_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=offset_factor, scope=scope
        )
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("store"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_store_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.int32()
        s0 = T.int32()
        d1 = T.int32()
        d0 = T.int32()
        A = T.match_buffer(
            a,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
            strides=[d1, d0],
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope=scope,
            strides=[s1, s0],
        )
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_store_matrix_sync(
                    A.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_wmma_fragment_index(A, d1, m_dim, n_dim),
                    C.access_ptr("w"),
                    s1,
                    "row_major",
                    dtype="handle",
                )
            )

    return wmma_store_desc, wmma_store_impl


def get_wmma_sync_intrin(
    m_dim: int, n_dim: int, k_dim: int, in_dtype: str, out_dtype: str, b_transposed: bool
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_sync intrins"""

    def maybe_cast(v):
        if in_dtype != out_dtype:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    b_shape_0, b_shape_1 = maybe_swap(k_dim, n_dim)

    A_offset_factor = k_dim
    B_offset_factor = b_shape_1
    out_offset_factor = n_dim

    @T.prim_func
    def wmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (m_dim, k_dim),
            in_dtype,
            align=64,
            offset_factor=A_offset_factor,
            scope="wmma.matrix_a",
        )
        B = T.match_buffer(
            b,
            maybe_swap(k_dim, n_dim),
            in_dtype,
            align=64,
            offset_factor=B_offset_factor,
            scope="wmma.matrix_b",
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            out_dtype,
            align=64,
            offset_factor=out_offset_factor,
            scope="wmma.accumulator",
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.block(""):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    B_index_0, B_index_1 = T.meta_var(maybe_swap(vkk, vjj))
                    C[vii, vjj] = C[vii, vjj] + maybe_cast(A[vii, vkk]) * maybe_cast(
                        B[B_index_0, B_index_1]
                    )

    @T.prim_func
    def wmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        a1 = T.int32()
        a0 = T.int32()
        b1 = T.int32()
        b0 = T.int32()
        c1 = T.int32()
        c0 = T.int32()

        A = T.match_buffer(
            a,
            (m_dim, k_dim),
            in_dtype,
            align=64,
            offset_factor=A_offset_factor,
            scope="wmma.matrix_a",
            strides=[a1, a0],
        )
        B = T.match_buffer(
            b,
            maybe_swap(k_dim, n_dim),
            in_dtype,
            align=64,
            offset_factor=B_offset_factor,
            scope="wmma.matrix_b",
            strides=[b1, b0],
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            out_dtype,
            align=64,
            offset_factor=out_offset_factor,
            scope="wmma.accumulator",
            strides=[c1, c0],
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_mma_sync(
                    C.data,
                    get_wmma_fragment_index(C, c1, m_dim, n_dim),
                    A.data,
                    get_wmma_fragment_index(A, a1, m_dim, k_dim),
                    B.data,
                    get_wmma_fragment_index(B, b1, b_shape_0, b_shape_1),
                    C.data,
                    get_wmma_fragment_index(C, c1, m_dim, n_dim),
                    dtype="handle",
                )
            )

    return wmma_sync_desc, wmma_sync_impl


WMMA_SYNC_16x16x4_f32f32f32_INTRIN = "maca_wmma_sync_16x16x4_f32f32f32"
TensorIntrin.register(
    WMMA_SYNC_16x16x4_f32f32f32_INTRIN,
    *get_wmma_sync_intrin(16, 16, 4, "float32", "float32", False),
)

WMMA_SYNC_16x16x16_f16f16f32_INTRIN = "maca_wmma_sync_16x16x16_f16f16f32"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "float16", "float32", False),
)

WMMA_SYNC_16x16x16_f8f8f32_INTRIN = "maca_wmma_sync_16x16x16_f8f8f32"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_f8f8f32_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "float8_e4m3fn", "float32", False),
)

WMMA_SYNC_16x16x4_f32f32f32_TRANS_INTRIN = "maca_wmma_sync_16x16x4_f32f32f32_trans"
TensorIntrin.register(
    WMMA_SYNC_16x16x4_f32f32f32_TRANS_INTRIN,
    *get_wmma_sync_intrin(16, 16, 4, "float32", "float32", True),
)

WMMA_SYNC_16x16x16_f16f16f32_TRANS_INTRIN = "maca_wmma_sync_16x16x16_f16f16f32_trans"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_f16f16f32_TRANS_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "float16", "float32", True),
)

WMMA_SYNC_16x16x16_f16f16f16_INTRIN = "maca_wmma_sync_16x16x16_f16f16f16"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "float16", "float16", False),
)

WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN = "maca_wmma_sync_16x16x16_f16f16f16_trans"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "float16", "float16", True),
)

WMMA_SYNC_16x16x16_s8s8s32_INTRIN = "maca_wmma_sync_16x16x16_s8s8s32"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_s8s8s32_INTRIN, *get_wmma_sync_intrin(16, 16, 16, "int8", "int32", False)
)

WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN = "maca_wmma_sync_16x16x16_s8s8s32_trans"
TensorIntrin.register(
    WMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN,
    *get_wmma_sync_intrin(16, 16, 16, "int8", "int32", True),
)

WMMA_SYNC_8x8x32_s4s4s32_TRANS_INTRIN = "maca_wmma_sync_8x8x32_s4s4s32_trans"
TensorIntrin.register(
    WMMA_SYNC_8x8x32_s4s4s32_TRANS_INTRIN, *get_wmma_sync_intrin(8, 8, 32, "int4", "int32", True)
)

WMMA_LOAD_16x16x16_F16_A_INTRIN = "maca_wmma_load_16x16x16_f16_a_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", False, False),
)

WMMA_LOAD_16x16x4_F32_A_DYN_INTRIN = "maca_wmma_load_16x16x4_f32_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x4_F32_A_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 4, "float32", "shared.dyn", False, False),
)

WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN = "maca_wmma_load_16x16x16_f16_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_A_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared.dyn", False, False),
)

WMMA_LOAD_16x16x16_F16_B_INTRIN = "maca_wmma_load_16x16x16_f16_b_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", True, False),
)

WMMA_LOAD_16x16x16_F8_A_DYN_INTRIN = "maca_wmma_load_16x16x16_f8_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F8_A_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float8_e4m3fn", "shared.dyn", False, False),
)

WMMA_LOAD_16x16x16_F8_B_INTRIN = "maca_wmma_load_16x16x16_f8_b_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F8_B_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float8_e4m3fn", "shared", True, False),
)

WMMA_LOAD_16x16x4_F32_B_DYN_INTRIN = "maca_wmma_load_16x16x4_f32_b_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x4_F32_B_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 4, "float32", "shared.dyn", True, False),
)

WMMA_LOAD_16x16x16_F8_B_DYN_INTRIN = "maca_wmma_load_16x16x16_f8_b_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F8_B_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float8_e4m3fn", "shared.dyn", True, False),
)

WMMA_LOAD_16x16x16_F16_B_DYN_INTRIN = "maca_wmma_load_16x16x16_f16_b_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_B_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared.dyn", True, False),
)

WMMA_LOAD_16x16x16_F16_A_TRANS_INTRIN = "maca_wmma_load_16x16x16_f16_a_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_A_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", False, True),
)

WMMA_LOAD_16x16x16_F16_A_TRANS_DYN_INTRIN = "maca_wmma_load_16x16x16_f16_a_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_A_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared.dyn", False, True),
)

WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN = "maca_wmma_load_16x16x16_f16_b_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared", True, True),
)

WMMA_LOAD_16x16x4_F32_B_TRANS_DYN_INTRIN = "maca_wmma_load_16x16x4_f32_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x4_F32_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 4, "float32", "shared.dyn", True, True),
)

WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN = "maca_wmma_load_16x16x16_f16_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "float16", "shared.dyn", True, True),
)

WMMA_LOAD_16x16x16_S8_A_INTRIN = "maca_wmma_load_16x16x16_s8_a_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_A_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared", False, False),
)

WMMA_LOAD_16x16x16_S8_A_DYN_INTRIN = "maca_wmma_load_16x16x16_s8_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_A_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared.dyn", False, False),
)

WMMA_LOAD_16x16x16_S8_B_INTRIN = "maca_wmma_load_16x16x16_s8_b_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_B_INTRIN, *get_wmma_load_intrin(16, 16, 16, "int8", "shared", True, False)
)

WMMA_LOAD_16x16x16_S8_B_DYN_INTRIN = "maca_wmma_load_16x16x16_s8_b_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_B_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared.dyn", True, False),
)

WMMA_LOAD_16x16x16_S8_A_TRANS_INTRIN = "maca_wmma_load_16x16x16_s8_a_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_A_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared", False, True),
)

WMMA_LOAD_16x16x16_S8_A_TRANS_DYN_INTRIN = "maca_wmma_load_16x16x16_s8_a_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_A_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared.dyn", False, True),
)

WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN = "maca_wmma_load_16x16x16_s8_b_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared", True, True),
)

WMMA_LOAD_16x16x16_S8_B_TRANS_DYN_INTRIN = "maca_wmma_load_16x16x16_s8_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_16x16x16_S8_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(16, 16, 16, "int8", "shared.dyn", True, True),
)

WMMA_LOAD_8x8x32_S4_A_INTRIN = "maca_wmma_load_8x8x32_s4_a_shared"
TensorIntrin.register(
    WMMA_LOAD_8x8x32_S4_A_INTRIN, *get_wmma_load_intrin(8, 8, 32, "int4", "shared", False, False)
)

WMMA_LOAD_8x8x32_S4_A_DYN_INTRIN = "maca_wmma_load_8x8x32_s4_a_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x8x32_S4_A_DYN_INTRIN,
    *get_wmma_load_intrin(8, 8, 32, "int4", "shared.dyn", False, False),
)

WMMA_LOAD_8x8x32_S4_B_TRANS_INTRIN = "maca_wmma_load_8x8x32_s4_b_trans_shared"
TensorIntrin.register(
    WMMA_LOAD_8x8x32_S4_B_TRANS_INTRIN,
    *get_wmma_load_intrin(8, 8, 32, "int4", "shared", True, True),
)

WMMA_LOAD_8x8x32_S4_B_TRANS_DYN_INTRIN = "maca_wmma_load_8x8x32_s4_b_trans_shared_dyn"
TensorIntrin.register(
    WMMA_LOAD_8x8x32_S4_B_TRANS_DYN_INTRIN,
    *get_wmma_load_intrin(8, 8, 32, "int4", "shared.dyn", True, True),
)

WMMA_FILL_16x16x4_F32_INTRIN = "maca_wmma_fill_16x16x4_f32"
TensorIntrin.register(WMMA_FILL_16x16x4_F32_INTRIN, *get_wmma_fill_intrin(16, 16, 4, "float32"))

WMMA_FILL_16x16x16_F32_INTRIN = "maca_wmma_fill_16x16x16_f32"
TensorIntrin.register(WMMA_FILL_16x16x16_F32_INTRIN, *get_wmma_fill_intrin(16, 16, 16, "float32"))

WMMA_FILL_16x16x16_F16_INTRIN = "maca_wmma_fill_16x16x16_f16"
TensorIntrin.register(WMMA_FILL_16x16x16_F16_INTRIN, *get_wmma_fill_intrin(16, 16, 16, "float16"))

WMMA_FILL_16x16x16_S32_INTRIN = "maca_wmma_fill_16x16x16_s32"
TensorIntrin.register(WMMA_FILL_16x16x16_S32_INTRIN, *get_wmma_fill_intrin(16, 16, 16, "int32"))

WMMA_FILL_8x8x32_S32_INTRIN = "maca_wmma_fill_8x8x32_s32"
TensorIntrin.register(WMMA_FILL_8x8x32_S32_INTRIN, *get_wmma_fill_intrin(8, 8, 32, "int32"))

WMMA_STORE_16x16x16_F32_SHARED_INTRIN = "maca_wmma_store_16x16x16_f32_shared"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F32_SHARED_INTRIN, *get_wmma_store_intrin(16, 16, 16, "float32", "shared")
)

WMMA_STORE_16x16x4_F32_SHARED_DYN_INTRIN = "maca_wmma_store_16x16x4_f32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_16x16x4_F32_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(16, 16, 4, "float32", "shared.dyn"),
)

WMMA_STORE_16x16x16_F32_SHARED_DYN_INTRIN = "maca_wmma_store_16x16x16_f32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F32_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(16, 16, 16, "float32", "shared.dyn"),
)

WMMA_STORE_16x16x16_F16_SHARED_INTRIN = "maca_wmma_store_16x16x16_f16_shared"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F16_SHARED_INTRIN, *get_wmma_store_intrin(16, 16, 16, "float16", "shared")
)

WMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN = "maca_wmma_store_16x16x16_f16_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(16, 16, 16, "float16", "shared.dyn"),
)

WMMA_STORE_16x16x16_S32_SHARED_INTRIN = "maca_wmma_store_16x16x16_s32_shared"
TensorIntrin.register(
    WMMA_STORE_16x16x16_S32_SHARED_INTRIN, *get_wmma_store_intrin(16, 16, 16, "int32", "shared")
)

WMMA_STORE_16x16x16_S32_SHARED_DYN_INTRIN = "maca_wmma_store_16x16x16_s32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_16x16x16_S32_SHARED_DYN_INTRIN,
    *get_wmma_store_intrin(16, 16, 16, "int32", "shared.dyn"),
)

WMMA_STORE_8x8x32_S32_SHARED_INTRIN = "maca_wmma_store_8x8x32_s32_shared"
TensorIntrin.register(
    WMMA_STORE_8x8x32_S32_SHARED_INTRIN, *get_wmma_store_intrin(8, 8, 32, "int32", "shared")
)

WMMA_STORE_8x8x32_S32_SHARED_DYN_INTRIN = "maca_wmma_store_8x8x32_s32_shared_dyn"
TensorIntrin.register(
    WMMA_STORE_8x8x32_S32_SHARED_DYN_INTRIN, *get_wmma_store_intrin(8, 8, 32, "int32", "shared.dyn")
)

WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN = "maca_wmma_store_16x16x16_f32_global"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN, *get_wmma_store_intrin(16, 16, 16, "float32", "global")
)

WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN = "maca_wmma_store_16x16x16_f16_global"
TensorIntrin.register(
    WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN, *get_wmma_store_intrin(16, 16, 16, "float16", "global")
)

WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN = "maca_wmma_store_16x16x16_s32_global"
TensorIntrin.register(
    WMMA_STORE_16x16x16_S32_GLOBAL_INTRIN, *get_wmma_store_intrin(16, 16, 16, "int32", "global")
)

WMMA_STORE_8x8x32_S32_GLOBAL_INTRIN = "maca_wmma_store_8x8x32_s32_global"
TensorIntrin.register(
    WMMA_STORE_8x8x32_S32_GLOBAL_INTRIN, *get_wmma_store_intrin(8, 8, 32, "int32", "global")
)


def get_wmma_intrin_group(
    load_scope: Literal["shared", "shared.dyn"],
    store_scope: Literal["global", "shared", "shared.dyn"],
    in_dtype: str,
    out_dtype: str,
    trans_b: bool,
) -> Dict[str, str]:
    """Get a group of intrinsics for wmma tensor core with the given configurations

    Parameters
    ----------
    load_scope : Literal["shared", "shared.dyn"]
        The memory scope of the input buffer.

    store_scope : Literal["global", "shared", "shared.dyn"]
        The memory scope of the result buffer.

    in_dtype : str
        The input data type.

    out_dtype : str
        The output data dtype.

    trans_b : bool
        Whether the input matrix B is transposed.

    Returns
    -------
    ret : Dict[str, str]
        A group of tensor intrinsics.
    """
    assert load_scope in ["shared", "shared.dyn"]
    assert store_scope in ["global", "shared", "shared.dyn"]
    assert in_dtype in ["float16", "int8"]
    assert out_dtype in ["float16", "float32", "int32"]

    shape = "16x16x16"
    in_dtype = "f16" if in_dtype == "float16" else "s8"
    out_dtype = "f16" if out_dtype == "float16" else "f32" if out_dtype == "float32" else "s32"
    # convert "shared.dyn" to "shared_dyn"
    load_scope = load_scope.replace(".", "_")
    store_scope = store_scope.replace(".", "_")
    trans_a = ""
    trans_b = "_trans" if trans_b else ""

    # e.g. wmma_load_16x16x16_f16_a_shared
    load_a_intrin = f"maca_wmma_load_{shape}_{in_dtype}_a{trans_a}_{load_scope}"
    # e.g. wmma_load_16x16x16_f16_b_trans_shared_dyn
    load_b_intrin = f"maca_wmma_load_{shape}_{in_dtype}_b{trans_b}_{load_scope}"
    # e.g. wmma_sync_16x16x16_f16f16f32_trans
    compute_intrin = f"maca_wmma_sync_{shape}_{in_dtype}{in_dtype}{out_dtype}{trans_b}"
    # e.g. wmma_fill_16x16x16_f16
    init_intrin = f"maca_wmma_fill_{shape}_{out_dtype}"
    # e.g. wmma_store_16x16x16_f16_shared_dyn
    store_intrin = f"maca_wmma_store_{shape}_{out_dtype}_{store_scope}"

    return {
        "init": init_intrin,
        "load_a": load_a_intrin,
        "load_b": load_b_intrin,
        "compute": compute_intrin,
        "store": store_intrin,
    }
