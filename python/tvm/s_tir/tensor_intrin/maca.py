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

from typing import Literal

from tvm.script import tirx as T
from tvm.tirx import Cast, IntImm, TensorIntrin
from tvm.tirx.function import PrimFunc

########## MACA MMA intrinsics ##########

MACA_MMA_WARP_SIZE = 64
MACA_MMA_M_DIM = 16
MACA_MMA_N_DIM = 16


MACA_MMA_F16F16F32_SOURCE = r"""
static __device__ __forceinline__ float4
tvm_maca_mma_16x16x16f16(
    half4 a,
    half4 b,
    float4 c) {
    typedef __NATIVE_VECTOR__(4, __fp16) native_f16x4;
    typedef __NATIVE_VECTOR__(4, float) native_f32x4;

    native_f16x4 a_native =
        *reinterpret_cast<const native_f16x4*>(&a);

    native_f16x4 b_native =
        *reinterpret_cast<const native_f16x4*>(&b);

    native_f32x4 c_native =
        *reinterpret_cast<const native_f32x4*>(&c);

    native_f32x4 d_native =
        __builtin_mxc_mma_16x16x16f16(
            a_native,
            b_native,
            c_native);

    return *reinterpret_cast<float4*>(&d_native);
}
"""

MACA_MMA_S8S8S32_SOURCE = r"""
static __device__ __forceinline__ int4
tvm_maca_mma_16x16x16i8(
    int a,
    int b,
    int4 c) {
    typedef __NATIVE_VECTOR__(4, int) native_i32x4;

    native_i32x4 c_native =
        *reinterpret_cast<const native_i32x4*>(&c);

    native_i32x4 d_native =
        __builtin_mxc_mma_16x16x16i8(
            a,
            b,
            c_native);

    return *reinterpret_cast<int4*>(&d_native);
}
"""

MACA_MMA_F32F32F32_SOURCE = r"""
static __device__ __forceinline__ float4
tvm_maca_mma_16x16x4f32(
    float a,
    float b,
    float4 c) {
    typedef __NATIVE_VECTOR__(4, float) native_f32x4;
    native_f32x4 c_native =
        *reinterpret_cast<const native_f32x4*>(&c);

    native_f32x4 d_native =
        __builtin_mxc_mma_16x16x4f32(
            a,
            b,
            c_native);

    return *reinterpret_cast<float4*>(&d_native);
}
"""


def shared_16x16_to_local_64x4_layout_A(i, j):
    thread_id = i + 16 * (j // 4)
    local_id = j % 4
    return thread_id, local_id


def shared_16x16_to_local_64x4_layout_B(i, j):
    thread_id = j + (i // 4) * 16
    local_id = i % 4
    return thread_id, local_id


def shared_16x16_to_local_64x4_layout_C(i, j):
    thread_id = j + (i // 4) * 16
    local_id = i % 4
    return thread_id, local_id


def local_64x4_to_shared_16x16_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 4 + local_id
    return i, j


def local_64x4_to_shared_16x16_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def local_64x4_to_shared_16x16_layout_C(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def shared_16x4_to_local_64x1_layout_A(i, j):
    thread_id = j * 16 + i
    local_id = T.int32(0)
    return thread_id, local_id


def shared_4x16_to_local_64x1_layout_B(i, j):
    thread_id = i * 16 + j
    local_id = T.int32(0)
    return thread_id, local_id


def local_64x1_to_shared_16x4_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = thread_id // 16
    return i, j


def local_64x1_to_shared_4x16_layout_B(thread_id, local_id):
    i = thread_id // 16
    j = thread_id % 16
    return i, j


def get_maca_mma_load_intrin(
    k_dim=16,
    dtype="float16",
    scope="shared",
    is_b=False,
):
    warp_size = MACA_MMA_WARP_SIZE

    if k_dim == 16:
        if dtype not in ("float16", "int8"):
            raise ValueError(f"MACA MMA k_dim=16 does not support dtype={dtype}")

        memory_shape = (16, 16)

        if is_b:
            index_map = shared_16x16_to_local_64x4_layout_B
            reverse_index_map = local_64x4_to_shared_16x16_layout_B
        else:
            index_map = shared_16x16_to_local_64x4_layout_A
            reverse_index_map = local_64x4_to_shared_16x16_layout_A

    elif k_dim == 4:
        if dtype != "float32":
            raise ValueError(f"MACA MMA k_dim=4 currently only supports float32, but got {dtype}")

        if is_b:
            memory_shape = (4, 16)
            index_map = shared_4x16_to_local_64x1_layout_B
            reverse_index_map = local_64x1_to_shared_4x16_layout_B
        else:
            memory_shape = (16, 4)
            index_map = shared_16x4_to_local_64x1_layout_A
            reverse_index_map = local_64x1_to_shared_16x4_layout_A

    else:
        raise ValueError(f"Unsupported MACA MMA k_dim: {k_dim}")

    num_elements = memory_shape[0] * memory_shape[1]

    if num_elements % warp_size != 0:
        raise ValueError(
            f"Memory tile shape {memory_shape} cannot be evenly distributed "
            f"across {warp_size} threads"
        )

    local_size = num_elements // warp_size

    @T.prim_func(s_tir=True)
    def maca_mma_load_desc(
        reg_handle: T.handle,
        memory_handle: T.handle,
    ) -> None:
        memory = T.match_buffer(
            memory_handle,
            memory_shape,
            dtype,
            offset_factor=1,
            scope=scope,
        )

        reg = T.match_buffer(
            reg_handle,
            (warp_size, local_size),
            dtype,
            offset_factor=1,
            scope="warp",
        )

        with T.sblock("root"):
            T.reads(memory[0 : memory_shape[0], 0 : memory_shape[1]])
            T.writes(reg[0:warp_size, 0:local_size])

            for i, j in T.grid(memory_shape[0], memory_shape[1]):
                with T.sblock("memory_reg"):
                    vi, vj = T.axis.remap("SS", [i, j])

                    thread_id, local_id = T.meta_var(index_map(vi, vj))

                    T.reads(memory[vi, vj])
                    T.writes(reg[thread_id, local_id])

                    reg[thread_id, local_id] = memory[vi, vj]

    @T.prim_func(s_tir=True)
    def maca_mma_load_impl(
        reg_handle: T.handle,
        memory_handle: T.handle,
    ) -> None:
        s0 = T.int32()
        s1 = T.int32()

        memory = T.match_buffer(
            memory_handle,
            memory_shape,
            dtype,
            align=64,
            offset_factor=1,
            scope=scope,
            strides=[s0, s1],
        )

        reg = T.match_buffer(
            reg_handle,
            (warp_size, local_size),
            dtype,
            align=64,
            offset_factor=1,
            scope="warp",
        )

        with T.sblock("root"):
            T.reads(memory[0 : memory_shape[0], 0 : memory_shape[1]])
            T.writes(reg[0:warp_size, 0:local_size])

            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, warp_size)

            for local_id in T.serial(0, local_size):
                row, col = T.meta_var(reverse_index_map(tx, local_id))
                reg[tx, local_id] = memory[row, col]

    return maca_mma_load_desc, maca_mma_load_impl


def get_maca_mma_fill_intrin(
    dtype="float32",
    local_size=4,
):
    warp_size = MACA_MMA_WARP_SIZE

    zero = IntImm("int32", 0).astype(dtype)

    index_map = shared_16x16_to_local_64x4_layout_C

    @T.prim_func(s_tir=True)
    def maca_mma_fill_desc(a: T.handle) -> None:
        C_warp = T.match_buffer(
            a,
            (warp_size, local_size),
            dtype=dtype,
            scope="warp",
        )

        with T.sblock("root"):
            T.reads()
            T.writes(C_warp[0:warp_size, 0:local_size])

            for i0, i1 in T.grid(16, 16):
                with T.sblock("C_warp"):
                    i, j = T.axis.remap("SS", [i0, i1])

                    thread_id, local_id = T.meta_var(index_map(i, j))

                    T.reads()
                    T.writes(C_warp[thread_id, local_id])

                    C_warp[thread_id, local_id] = zero

    @T.prim_func(s_tir=True)
    def maca_mma_fill_impl(a: T.handle) -> None:
        C_warp = T.match_buffer(
            a,
            (warp_size, local_size),
            dtype=dtype,
            scope="warp",
            offset_factor=1,
        )

        with T.sblock("root"):
            T.reads()
            T.writes(C_warp[0:warp_size, 0:local_size])

            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, warp_size)

            for local_id in T.serial(0, local_size):
                C_warp[tx, local_id] = zero

    return maca_mma_fill_desc, maca_mma_fill_impl


def get_maca_mma_store_intrin(
    dtype="float32",
    scope="global",
):
    warp_size = MACA_MMA_WARP_SIZE
    local_size = (MACA_MMA_M_DIM * MACA_MMA_N_DIM) // warp_size

    index_map = shared_16x16_to_local_64x4_layout_C
    reverse_index_map = local_64x4_to_shared_16x16_layout_C

    @T.prim_func(s_tir=True)
    def maca_mma_store_desc(
        a: T.handle,
        c: T.handle,
    ) -> None:
        C_warp = T.match_buffer(
            a,
            (warp_size, local_size),
            dtype=dtype,
            scope="warp",
        )

        C = T.match_buffer(
            c,
            (MACA_MMA_M_DIM, MACA_MMA_N_DIM),
            dtype=dtype,
            scope=scope,
        )

        with T.sblock("root"):
            T.reads(C_warp[0:warp_size, 0:local_size])
            T.writes(C[0:MACA_MMA_M_DIM, 0:MACA_MMA_N_DIM])

            for i0, i1 in T.grid(MACA_MMA_M_DIM, MACA_MMA_N_DIM):
                with T.sblock("C_warp"):
                    i, j = T.axis.remap("SS", [i0, i1])

                    thread_id, local_id = T.meta_var(index_map(i, j))

                    T.reads(C_warp[thread_id, local_id])
                    T.writes(C[i, j])

                    C[i, j] = C_warp[thread_id, local_id]

    @T.prim_func(s_tir=True)
    def maca_mma_store_impl(
        a: T.handle,
        c: T.handle,
    ) -> None:
        s0 = T.int32()
        s1 = T.int32()

        C_warp = T.match_buffer(
            a,
            (warp_size, local_size),
            dtype=dtype,
            scope="warp",
            offset_factor=1,
        )

        C = T.match_buffer(
            c,
            (MACA_MMA_M_DIM, MACA_MMA_N_DIM),
            dtype=dtype,
            scope=scope,
            offset_factor=1,
            strides=[s0, s1],
        )

        with T.sblock("root"):
            T.reads(C_warp[0:warp_size, 0:local_size])
            T.writes(C[0:MACA_MMA_M_DIM, 0:MACA_MMA_N_DIM])

            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, warp_size)

            for local_id in T.serial(0, local_size):
                row, col = T.meta_var(reverse_index_map(tx, local_id))
                C[row, col] = C_warp[tx, local_id]

    return maca_mma_store_desc, maca_mma_store_impl


def get_maca_mma_intrin(
    k_dim=16,
    in_dtype="float16",
    out_dtype="float32",
):
    if k_dim not in (4, 16):
        raise ValueError("MACA MMA currently only supports k_dim=4 or 16")

    if (in_dtype, out_dtype) == ("float16", "float32"):
        if k_dim != 16:
            raise ValueError("MACA MMA float16 path currently only supports k_dim=16")
        builtin_name = "tvm_maca_mma_16x16x16f16"
        builtin_source = MACA_MMA_F16F16F32_SOURCE
        cast_dtype = "float32"
    elif (in_dtype, out_dtype) == ("int8", "int32"):
        if k_dim != 16:
            raise ValueError("MACA MMA int8 path currently only supports k_dim=16")
        builtin_name = "tvm_maca_mma_16x16x16i8"
        builtin_source = MACA_MMA_S8S8S32_SOURCE
        cast_dtype = "int32"
    elif (in_dtype, out_dtype) == ("float32", "float32"):
        if k_dim != 4:
            raise ValueError("MACA MMA float32 path currently only supports k_dim=4")
        builtin_name = "tvm_maca_mma_16x16x4f32"
        builtin_source = MACA_MMA_F32F32F32_SOURCE
        cast_dtype = "float32"
    else:
        raise ValueError(
            "MACA MMA currently only supports float16->float32, int8->int32, or float32->float32"
        )

    warp_size = MACA_MMA_WARP_SIZE
    m_dim = MACA_MMA_M_DIM
    n_dim = MACA_MMA_N_DIM

    local_size = (m_dim * k_dim) // warp_size
    local_size_out = (m_dim * n_dim) // warp_size

    if k_dim == 4:
        index_map_A = shared_16x4_to_local_64x1_layout_A
        index_map_B = shared_4x16_to_local_64x1_layout_B
    else:
        index_map_A = shared_16x16_to_local_64x4_layout_A
        index_map_B = shared_16x16_to_local_64x4_layout_B
    index_map_C = shared_16x16_to_local_64x4_layout_C

    @T.prim_func(s_tir=True)
    def maca_mma_sync_desc(
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ) -> None:
        A = T.match_buffer(
            a,
            (warp_size, local_size),
            in_dtype,
            offset_factor=1,
            scope="warp",
        )

        B = T.match_buffer(
            b,
            (warp_size, local_size),
            in_dtype,
            offset_factor=1,
            scope="warp",
        )

        C = T.match_buffer(
            c,
            (warp_size, local_size_out),
            out_dtype,
            offset_factor=1,
            scope="warp",
        )

        with T.sblock("root"):
            T.reads(
                C[0:warp_size, 0:local_size_out],
                A[0:warp_size, 0:local_size],
                B[0:warp_size, 0:local_size],
            )
            T.writes(C[0:warp_size, 0:local_size_out])

            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.sblock("C"):
                    vi, vj, vk = T.axis.remap(
                        "SSR",
                        [i, j, k],
                    )

                    thread_id_C, local_id_C = T.meta_var(index_map_C(vi, vj))
                    thread_id_A, local_id_A = T.meta_var(index_map_A(vi, vk))
                    thread_id_B, local_id_B = T.meta_var(index_map_B(vk, vj))

                    T.reads(
                        C[thread_id_C, local_id_C],
                        A[thread_id_A, local_id_A],
                        B[thread_id_B, local_id_B],
                    )
                    T.writes(C[thread_id_C, local_id_C])

                    C[thread_id_C, local_id_C] += Cast(
                        out_dtype,
                        A[thread_id_A, local_id_A],
                    ) * Cast(
                        out_dtype,
                        B[thread_id_B, local_id_B],
                    )

    @T.prim_func(s_tir=True)
    def maca_mma_sync_impl(
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ) -> None:
        A = T.match_buffer(
            a,
            (warp_size, local_size),
            in_dtype,
            offset_factor=1,
            scope="warp",
        )

        B = T.match_buffer(
            b,
            (warp_size, local_size),
            in_dtype,
            offset_factor=1,
            scope="warp",
        )

        C = T.match_buffer(
            c,
            (warp_size, local_size_out),
            out_dtype,
            offset_factor=1,
            scope="warp",
        )

        with T.sblock("root"):
            T.reads(
                A[0:warp_size, 0:local_size],
                B[0:warp_size, 0:local_size],
                C[0:warp_size, 0:local_size_out],
            )
            T.writes(C[0:warp_size, 0:local_size_out])

            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, warp_size)

            C[tx, 0:local_size_out] = T.call_intrin(
                "float32x4" if cast_dtype == "float32" else "int32x4",
                "tirx.maca.func_call",
                builtin_name,
                A[tx, 0:local_size],
                B[tx, 0:local_size],
                C[tx, 0:local_size_out],
                builtin_source,
            )

    return maca_mma_sync_desc, maca_mma_sync_impl


MACA_MMA_LOAD_16x16_A_SHARED_F16_INTRIN = "maca_mma_load_16x16_a_shared_f16"
TensorIntrin.register(
    MACA_MMA_LOAD_16x16_A_SHARED_F16_INTRIN,
    *get_maca_mma_load_intrin(16, "float16", "shared", is_b=False),
)

MACA_MMA_LOAD_16x16_B_SHARED_F16_INTRIN = "maca_mma_load_16x16_b_shared_f16"
TensorIntrin.register(
    MACA_MMA_LOAD_16x16_B_SHARED_F16_INTRIN,
    *get_maca_mma_load_intrin(16, "float16", "shared", is_b=True),
)

MACA_MMA_LOAD_16x16_A_SHARED_S8_INTRIN = "maca_mma_load_16x16_a_shared_s8"
TensorIntrin.register(
    MACA_MMA_LOAD_16x16_A_SHARED_S8_INTRIN,
    *get_maca_mma_load_intrin(16, "int8", "shared", is_b=False),
)

MACA_MMA_LOAD_16x16_B_SHARED_S8_INTRIN = "maca_mma_load_16x16_b_shared_s8"
TensorIntrin.register(
    MACA_MMA_LOAD_16x16_B_SHARED_S8_INTRIN,
    *get_maca_mma_load_intrin(16, "int8", "shared", is_b=True),
)

MACA_MMA_LOAD_16x4_A_SHARED_F32_INTRIN = "maca_mma_load_16x4_a_shared_f32"
TensorIntrin.register(
    MACA_MMA_LOAD_16x4_A_SHARED_F32_INTRIN,
    *get_maca_mma_load_intrin(4, "float32", "shared", is_b=False),
)

MACA_MMA_LOAD_4x16_B_SHARED_F32_INTRIN = "maca_mma_load_4x16_b_shared_f32"
TensorIntrin.register(
    MACA_MMA_LOAD_4x16_B_SHARED_F32_INTRIN,
    *get_maca_mma_load_intrin(4, "float32", "shared", is_b=True),
)

MACA_MMA_FILL_16x16_F32_INTRIN = "maca_mma_fill_16x16_f32"
TensorIntrin.register(
    MACA_MMA_FILL_16x16_F32_INTRIN,
    *get_maca_mma_fill_intrin("float32", 4),
)

MACA_MMA_FILL_16x16_S32_INTRIN = "maca_mma_fill_16x16_s32"
TensorIntrin.register(
    MACA_MMA_FILL_16x16_S32_INTRIN,
    *get_maca_mma_fill_intrin("int32", 4),
)

MACA_MMA_STORE_16x16_F32_INTRIN = "maca_mma_store_16x16_f32"
TensorIntrin.register(
    MACA_MMA_STORE_16x16_F32_INTRIN,
    *get_maca_mma_store_intrin("float32", "global"),
)

MACA_MMA_STORE_16x16_S32_INTRIN = "maca_mma_store_16x16_s32"
TensorIntrin.register(
    MACA_MMA_STORE_16x16_S32_INTRIN,
    *get_maca_mma_store_intrin("int32", "global"),
)

MACA_MMA_F16F16F32_INTRIN = "maca_mma_f16f16f32"
TensorIntrin.register(
    MACA_MMA_F16F16F32_INTRIN,
    *get_maca_mma_intrin(16, "float16", "float32"),
)

MACA_MMA_S8S8S32_INTRIN = "maca_mma_s8s8s32"
TensorIntrin.register(
    MACA_MMA_S8S8S32_INTRIN,
    *get_maca_mma_intrin(16, "int8", "int32"),
)

MACA_MMA_F32F32F32_INTRIN = "maca_mma_f32f32f32"
TensorIntrin.register(
    MACA_MMA_F32F32F32_INTRIN,
    *get_maca_mma_intrin(4, "float32", "float32"),
)

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
) -> tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_load intrins"""
    wmma_fragment_scope = f"wmma.matrix_{'b' if is_b else 'a'}"
    layout = "col_major" if is_col_major else "row_major"

    frag_m, frag_n = (k_dim, n_dim) if is_b else (m_dim, k_dim)
    if is_col_major:
        frag_m, frag_n = frag_n, frag_m
    offset_factor = frag_n

    @T.prim_func(s_tir=True)
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
        with T.sblock("root"):
            T.reads(A[0:frag_m, 0:frag_n])
            T.writes(C[0:frag_m, 0:frag_n])
            for i, j in T.grid(frag_m, frag_n):
                with T.sblock("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func(s_tir=True)
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
        with T.sblock("root"):
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
) -> tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_fill intrins"""
    zero = IntImm("int32", 0).astype(dtype)
    offset_factor = n_dim

    @T.prim_func(s_tir=True)
    def wmma_fill_desc(c: T.handle) -> None:
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=offset_factor,
            scope="wmma.accumulator",
        )
        with T.sblock("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.sblock("init"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = zero

    @T.prim_func(s_tir=True)
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
        with T.sblock("root"):
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
) -> tuple[PrimFunc, PrimFunc]:
    """Generator of wmma_store intrins"""
    offset_factor = n_dim

    @T.prim_func(s_tir=True)
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
        with T.sblock("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.sblock("store"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func(s_tir=True)
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
        with T.sblock("root"):
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
) -> tuple[PrimFunc, PrimFunc]:
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

    @T.prim_func(s_tir=True)
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

        with T.sblock("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.sblock(""):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    B_index_0, B_index_1 = T.meta_var(maybe_swap(vkk, vjj))
                    C[vii, vjj] = C[vii, vjj] + maybe_cast(A[vii, vkk]) * maybe_cast(
                        B[B_index_0, B_index_1]
                    )

    @T.prim_func(s_tir=True)
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

        with T.sblock("root"):
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

WMMA_STORE_16x16x4_F32_GLOBAL_INTRIN = "maca_wmma_store_16x16x4_f32_global"
TensorIntrin.register(
    WMMA_STORE_16x16x4_F32_GLOBAL_INTRIN,
    *get_wmma_store_intrin(16, 16, 4, "float32", "global"),
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
) -> dict[str, str]:
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
