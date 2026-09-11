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
"""Tests for MACA Wave64 register GEMM datatype families.

The original f16/bf16-to-f32 fixtures follow the SDK WMMA m16n16k16 fragment
mapping. A Wave64 lane maps its four local operand and accumulator slots as
follows:

    lane = 16 * group + row
    A[p] = A[row, 4 * group + p]
    B[p] = B[4 * group + p, row]
    C/D[p] = C/D[4 * group + p, row]

The multi-tile layouts retain these atom coordinates as their innermost axes
and use positive physical strides for the outer tile axes.  Numerical tests
load and store through this mapping directly. The family fixture additionally
covers f16 output, full f32/f64, byte integers, packed four-bit integers, and
binary AND/popcount using independently specified lane maps.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout, laneid
from tvm.tirx.operator.tile_primitive import list_registered_schedules

MMA_M = 16
MMA_N = 16
MMA_K = 16
WAVE_SIZE = 64

# Explicit test-side signatures; these maps do not import dispatch internals.
_NUMERIC_FAMILIES = {
    "f16_f32": ("float16", "float32", 16, 16, 16),
    "bf16_f32": ("bfloat16", "float32", 16, 16, 16),
    "f16_f16": ("float16", "float16", 16, 16, 16),
    "f32_f32": ("float32", "float32", 16, 16, 4),
    "f64_f64": ("float64", "float64", 16, 16, 4),
    "i8_i32": ("int8", "int32", 16, 16, 16),
    "u8_i32": ("uint8", "int32", 16, 16, 16),
    "i4_i32": ("int4", "int32", 8, 8, 32),
    "u4_i32": ("uint4", "int32", 8, 8, 32),
    "b1_i32": ("int1", "int32", 8, 8, 128),
}

_SCRIPT_INTRINSIC = {
    "float16": "T.maca.mma_m16n16k16_f16_f32(",
    "bfloat16": "T.maca.mma_m16n16k16_bf16_f32(",
    "float16_acc": "T.maca.mma_m16n16k16_f16_f16(",
    "int8": "T.maca.mma_m16n16k16_i8_i32(",
    "uint8": "T.maca.mma_m16n16k16_u8_i32(",
}
_SOURCE_BUILTIN = {
    "float16": "__builtin_mxc_mma_16x16x16f16",
    "bfloat16": "__builtin_mxc_mma_16x16x16bf16",
}

# C500 Wave64 m16n16k16 atoms.  For lane = 16 * group + row, the source
# register index p is represented by the final extent-4 axis and maps to the
# SDK WMMA payload coordinate 4 * group + p.
A_ATOM = TileLayout(S[(16, 4, 4) : (1 @ laneid, 16 @ laneid, 1)])
B_ATOM = TileLayout(S[(4, 4, 16) : (16 @ laneid, 1, 1 @ laneid)])
D_ATOM = TileLayout(S[(4, 4, 16) : (16 @ laneid, 1, 1 @ laneid)])


def _transpose_frag(layout, shape):
    """Swap the two logical axes while preserving physical register order."""
    grouped, separators = layout.group(shape)
    return grouped.permute_by_groups(separators, [1, 0])


def _frag(Mt, Nt, Kt):
    """Return Wave64 fragments for an ``Mt x Nt x Kt`` m16n16k16 tiling.

    The innermost axes are exactly the C500 atoms.  The positive outer strides
    make a per-thread local view have physical shape ``[Mt, Kt, 4]`` for A,
    ``[Kt, Nt, 4]`` for B, and ``[Mt, Nt, 4]`` for C/D.
    """
    if min(Mt, Nt, Kt) < 1:
        raise ValueError("tile counts must be positive")

    A = TileLayout(S[(Mt, MMA_M, Kt, 4, 4) : (Kt * 4, 1 @ laneid, 4, MMA_K @ laneid, 1)])
    B = TileLayout(S[(Kt, 4, 4, Nt, MMA_N) : (Nt * 4, MMA_K @ laneid, 1, 4, 1 @ laneid)])
    D = TileLayout(S[(Mt, 4, 4, Nt, MMA_N) : (Nt * 4, 16 @ laneid, 1, 4, 1 @ laneid)])
    return D, A, B


def _build_tiled(Mt, Nt, Kt, *, alpha=1.0, beta=0.0, dtype="float16"):
    """Build one Wave64 GEMM call over an ``Mt x Nt x Kt`` register tiling."""
    Dl, Al, Bl = _frag(Mt, Nt, Kt)
    M, N, K = MMA_M * Mt, MMA_N * Nt, MMA_K * Kt

    @T.prim_func
    def gemm():
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        _lane = T.lane_id([WAVE_SIZE])
        A = T.alloc_buffer((M, K), dtype, scope="local", layout=Al)
        B = T.alloc_buffer((K, N), dtype, scope="local", layout=Bl)
        C = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
        D = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
        Tx.warp.gemm(D, A, B, C, transpose_A=False, transpose_B=False, alpha=alpha, beta=beta)

    return gemm


def _build_gemm(alpha=1.0, beta=0.0, dtype="bfloat16"):
    """Build one m16n16k16 register GEMM call."""
    return _build_tiled(1, 1, 1, alpha=alpha, beta=beta, dtype=dtype)


def _build_dtypes(a_dtype, b_dtype, c_dtype, d_dtype):
    """Build a single tile with explicit operand dtypes for decline checks."""

    @T.prim_func
    def gemm():
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        _lane = T.lane_id([WAVE_SIZE])
        A = T.alloc_buffer((MMA_M, MMA_K), a_dtype, scope="local", layout=A_ATOM)
        B = T.alloc_buffer((MMA_K, MMA_N), b_dtype, scope="local", layout=B_ATOM)
        C = T.alloc_buffer((MMA_M, MMA_N), c_dtype, scope="local", layout=D_ATOM)
        D = T.alloc_buffer((MMA_M, MMA_N), d_dtype, scope="local", layout=D_ATOM)
        Tx.warp.gemm(D, A, B, C, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0)

    return gemm


def _build_f32_gemm():
    """Build the SDK's full-F32 m16n16k4 atom."""
    a_layout = TileLayout(S[(16, 4) : (1 @ laneid, 16 @ laneid)])
    b_layout = TileLayout(S[(4, 16) : (16 @ laneid, 1 @ laneid)])
    d_layout = D_ATOM

    @T.prim_func
    def gemm():
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        _lane = T.lane_id([WAVE_SIZE])
        A = T.alloc_buffer((16, 4), "float32", scope="local", layout=a_layout)
        B = T.alloc_buffer((4, 16), "float32", scope="local", layout=b_layout)
        C = T.alloc_buffer((16, 16), "float32", scope="local", layout=d_layout)
        D = T.alloc_buffer((16, 16), "float32", scope="local", layout=d_layout)
        Tx.warp.gemm(D, A, B, C, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0)

    return gemm


def _build_packed_gemm(dtype, atom):
    """Build a minimal packed C500 atom for lowering coverage."""
    m, n, k = atom
    if dtype in ("int4", "uint4"):
        a_layout = TileLayout(S[(8, 8, 4) : (1 @ laneid, 8 @ laneid, 1)])
        b_layout = TileLayout(S[(8, 4, 8) : (8 @ laneid, 1, 1 @ laneid)])
        d_layout = TileLayout(S[(8, 8) : (8 @ laneid, 1 @ laneid)])
    else:
        a_layout = TileLayout(S[(8, 8, 16) : (1 @ laneid, 8 @ laneid, 1)])
        b_layout = TileLayout(S[(8, 16, 8) : (8 @ laneid, 1, 1 @ laneid)])
        d_layout = TileLayout(S[(8, 8) : (8 @ laneid, 1 @ laneid)])

    @T.prim_func
    def gemm():
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        _lane = T.lane_id([WAVE_SIZE])
        A = T.alloc_buffer((m, k), dtype, scope="local", layout=a_layout)
        B = T.alloc_buffer((k, n), dtype, scope="local", layout=b_layout)
        C = T.alloc_buffer((m, n), "int32", scope="local", layout=d_layout)
        D = T.alloc_buffer((m, n), "int32", scope="local", layout=d_layout)
        Tx.warp.gemm(D, A, B, C, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0)

    return gemm


def _build_tiled_numeric(Mt, Nt, Kt, beta, dtype):
    """Build a source/load/store fixture using the C500 direct lane mapping."""
    Dl, Al, Bl = _frag(Mt, Nt, Kt)
    M, N, K = MMA_M * Mt, MMA_N * Nt, MMA_K * Kt

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, D_ptr: T.handle):
        A_g = T.match_buffer(A_ptr, (M, K), dtype)
        B_g = T.match_buffer(B_ptr, (K, N), dtype)
        C_g = T.match_buffer(C_ptr, (M, N), "float32")
        D_g = T.match_buffer(D_ptr, (M, N), "float32")
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        lane = T.lane_id([WAVE_SIZE])
        group = lane // MMA_M
        row = lane % MMA_M
        A_f = T.alloc_buffer((M, K), dtype, scope="local", layout=Al)
        B_f = T.alloc_buffer((K, N), dtype, scope="local", layout=Bl)
        C_f = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
        D_f = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
        A_local = A_f.local(Mt, Kt, 4)
        B_local = B_f.local(Kt, Nt, 4)
        D_local = D_f.local(Mt, Nt, 4)
        for mt, kt, p in T.grid(Mt, Kt, 4):
            A_local[mt, kt, p] = A_g[mt * MMA_M + row, kt * MMA_K + 4 * group + p]
        for kt, nt, p in T.grid(Kt, Nt, 4):
            B_local[kt, nt, p] = B_g[kt * MMA_K + 4 * group + p, nt * MMA_N + row]
        if beta == 1.0:
            C_local = C_f.local(Mt, Nt, 4)
            for mt, nt, p in T.grid(Mt, Nt, 4):
                C_local[mt, nt, p] = C_g[mt * MMA_M + 4 * group + p, nt * MMA_N + row]
        Tx.warp.gemm(D_f, A_f, B_f, C_f, transpose_A=False, transpose_B=False, alpha=1.0, beta=beta)
        for mt, nt, p in T.grid(Mt, Nt, 4):
            D_g[mt * MMA_M + 4 * group + p, nt * MMA_N + row] = D_local[mt, nt, p]

    return gemm, M, N, K


def _build_transpose(transpose_A, transpose_B, *, dtype="float16"):
    """Build a single tile for one A/B logical input orientation."""
    Al = _transpose_frag(A_ATOM, [MMA_M, MMA_K]) if transpose_A else A_ATOM
    Bl = _transpose_frag(B_ATOM, [MMA_K, MMA_N]) if transpose_B else B_ATOM
    A_shape = (MMA_K, MMA_M) if transpose_A else (MMA_M, MMA_K)
    B_shape = (MMA_N, MMA_K) if transpose_B else (MMA_K, MMA_N)

    @T.prim_func
    def gemm():
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        _lane = T.lane_id([WAVE_SIZE])
        A = T.alloc_buffer(A_shape, dtype, scope="local", layout=Al)
        B = T.alloc_buffer(B_shape, dtype, scope="local", layout=Bl)
        C = T.alloc_buffer((MMA_M, MMA_N), "float32", scope="local", layout=D_ATOM)
        D = T.alloc_buffer((MMA_M, MMA_N), "float32", scope="local", layout=D_ATOM)
        Tx.warp.gemm(
            D,
            A,
            B,
            C,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            alpha=1.0,
            beta=0.0,
        )

    return gemm


def _build_aligned_region_slice(transpose_A, transpose_B, *, dtype="float16"):
    """Build one nonzero, 16-aligned m16n16k16 region GEMM.

    The full buffers contain multiple tiles.  The logical operation selects
    M=[16, 32), N=[16, 32), K=[32, 48), so the dispatcher must retain the
    per-operand tile offsets while normalizing either input orientation.
    """
    M, N, K = 32, 48, 48
    Dl, Al, Bl = _frag(M // MMA_M, N // MMA_N, K // MMA_K)
    Al = _transpose_frag(Al, [M, K]) if transpose_A else Al
    Bl = _transpose_frag(Bl, [K, N]) if transpose_B else Bl
    A_shape = (K, M) if transpose_A else (M, K)
    B_shape = (N, K) if transpose_B else (K, N)

    @T.prim_func
    def gemm():
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        _lane = T.lane_id([WAVE_SIZE])
        A = T.alloc_buffer(A_shape, dtype, scope="local", layout=Al)
        B = T.alloc_buffer(B_shape, dtype, scope="local", layout=Bl)
        C = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
        D = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
        if transpose_A and transpose_B:
            Tx.warp.gemm(
                D[16:32, 16:32],
                A[32:48, 16:32],
                B[16:32, 32:48],
                C[16:32, 16:32],
                transpose_A=True,
                transpose_B=True,
                alpha=1.0,
                beta=0.0,
            )
        elif transpose_A:
            Tx.warp.gemm(
                D[16:32, 16:32],
                A[32:48, 16:32],
                B[32:48, 16:32],
                C[16:32, 16:32],
                transpose_A=True,
                transpose_B=False,
                alpha=1.0,
                beta=0.0,
            )
        elif transpose_B:
            Tx.warp.gemm(
                D[16:32, 16:32],
                A[16:32, 32:48],
                B[16:32, 32:48],
                C[16:32, 16:32],
                transpose_A=False,
                transpose_B=True,
                alpha=1.0,
                beta=0.0,
            )
        else:
            Tx.warp.gemm(
                D[16:32, 16:32],
                A[16:32, 32:48],
                B[32:48, 16:32],
                C[16:32, 16:32],
                transpose_A=False,
                transpose_B=False,
                alpha=1.0,
                beta=0.0,
            )

    return gemm


def _build_transpose_numeric(transpose_A, transpose_B, dtype="float16"):
    """Build a source/load/store single tile for one input orientation."""
    Al = _transpose_frag(A_ATOM, [MMA_M, MMA_K]) if transpose_A else A_ATOM
    Bl = _transpose_frag(B_ATOM, [MMA_K, MMA_N]) if transpose_B else B_ATOM
    A_shape = (MMA_K, MMA_M) if transpose_A else (MMA_M, MMA_K)
    B_shape = (MMA_N, MMA_K) if transpose_B else (MMA_K, MMA_N)

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, D_ptr: T.handle):
        A_g = T.match_buffer(A_ptr, A_shape, dtype)
        B_g = T.match_buffer(B_ptr, B_shape, dtype)
        D_g = T.match_buffer(D_ptr, (MMA_M, MMA_N), "float32")
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        lane = T.lane_id([WAVE_SIZE])
        group = lane // MMA_M
        row = lane % MMA_M
        A_f = T.alloc_buffer(A_shape, dtype, scope="local", layout=Al)
        B_f = T.alloc_buffer(B_shape, dtype, scope="local", layout=Bl)
        C_f = T.alloc_buffer((MMA_M, MMA_N), "float32", scope="local", layout=D_ATOM)
        D_f = T.alloc_buffer((MMA_M, MMA_N), "float32", scope="local", layout=D_ATOM)
        A_local = A_f.local(4)
        B_local = B_f.local(4)
        D_local = D_f.local(4)
        for p in T.unroll(4):
            if transpose_A:
                A_local[p] = A_g[4 * group + p, row]
            else:
                A_local[p] = A_g[row, 4 * group + p]
            if transpose_B:
                B_local[p] = B_g[row, 4 * group + p]
            else:
                B_local[p] = B_g[4 * group + p, row]
        Tx.warp.gemm(
            D_f,
            A_f,
            B_f,
            C_f,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            alpha=1.0,
            beta=0.0,
        )
        for p in T.unroll(4):
            D_g[4 * group + p, row] = D_local[p]

    return gemm


def _lower(func):
    with tvm.target.Target("maca"):
        return tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": func}))


def _build_family_numeric(family, tiles, beta, transpose, region=False, repeat=False):
    """Load SDK lane payloads independently of the production layout builders."""
    dtype, acc, am, an, ak = _NUMERIC_FAMILIES[family]
    mt, nt, kt = tiles
    pad = int(region)
    pm, pn, pk = mt + pad, nt + pad, kt + pad
    M, N, K = pm * am, pn * an, pk * ak
    if am == 8:
        slots = ak // 8
        Al = TileLayout(S[(pm, 8, pk, 8, slots) : (pk * slots, 1 @ laneid, slots, 8 @ laneid, 1)])
        Bl = TileLayout(S[(pk, 8, slots, pn, 8) : (pn * slots, 8 @ laneid, 1, slots, 1 @ laneid)])
        Dl = TileLayout(S[(pm, 8, pn, 8) : (pn, 8 @ laneid, 1, 1 @ laneid)])
    elif ak == 4:
        Al = TileLayout(S[(pm, 16, pk, 4) : (pk, 1 @ laneid, 1, 16 @ laneid)])
        Bl = TileLayout(S[(pk, 4, pn, 16) : (pn, 16 @ laneid, 1, 1 @ laneid)])
        Dl, _, _ = _frag(pm, pn, pk)
        if acc == "float64":
            Dl = TileLayout(S[(pm, 4, 4, pn, 16) : (pn * 4, 1, 16 @ laneid, 4, 1 @ laneid)])
        slots = 1
    else:
        Dl, Al, Bl = _frag(pm, pn, pk)
        slots = 4
    ta, tb = transpose
    Al = _transpose_frag(Al, [M, K]) if ta else Al
    Bl = _transpose_frag(Bl, [K, N]) if tb else Bl
    ashape, bshape = ((K, M) if ta else (M, K)), ((N, K) if tb else (K, N))
    m0, n0, k0 = pad * am, pad * an, pad * ak
    dslots = am * an // WAVE_SIZE
    # Sub-byte inputs arrive as ordinary bytes. Pack their two-byte lane
    # payload explicitly, independently of the intrinsic's unpacking.
    host_dtype = "int8" if dtype in ("int4", "int1") else "uint8" if dtype == "uint4" else dtype

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, D_ptr: T.handle):
        A_g = T.match_buffer(A_ptr, ashape, host_dtype)
        B_g = T.match_buffer(B_ptr, bshape, host_dtype)
        C_g = T.match_buffer(C_ptr, (M, N), acc)
        D_g = T.match_buffer(D_ptr, (M, N), acc)
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        lane = T.lane_id([64])
        A_f = T.alloc_buffer(ashape, dtype, scope="local", layout=Al)
        B_f = T.alloc_buffer(bshape, dtype, scope="local", layout=Bl)
        D_f = T.alloc_buffer((M, N), acc, scope="local", layout=Dl)
        A_l = A_f.local(pm, pk, slots)
        B_l = B_f.local(pk, pn, slots)
        D_l = D_f.local(pm, pn, dslots)
        if am == 8:
            A_bits = T.decl_buffer((pm, pk), "uint16", data=A_l.data, scope="local")
            B_bits = T.decl_buffer((pk, pn), "uint16", data=B_l.data, scope="local")
            for mi, ki in T.grid(pm, pk):
                A_bits[mi, ki] = T.uint16(0)
                for p in T.serial(slots):
                    if ta:
                        A_bits[mi, ki] = A_bits[mi, ki] | (
                            (
                                T.Cast(
                                    "uint16",
                                    A_g[ki * ak + slots * (lane // am) + p, mi * am + lane % am],
                                )
                                & T.uint16((1 << (16 // slots)) - 1)
                            )
                            << T.Cast("uint16", p * (16 // slots))
                        )
                    else:
                        A_bits[mi, ki] = A_bits[mi, ki] | (
                            (
                                T.Cast(
                                    "uint16",
                                    A_g[mi * am + lane % am, ki * ak + slots * (lane // am) + p],
                                )
                                & T.uint16((1 << (16 // slots)) - 1)
                            )
                            << T.Cast("uint16", p * (16 // slots))
                        )
            for ki, ni in T.grid(pk, pn):
                B_bits[ki, ni] = T.uint16(0)
                for p in T.serial(slots):
                    if tb:
                        B_bits[ki, ni] = B_bits[ki, ni] | (
                            (
                                T.Cast(
                                    "uint16",
                                    B_g[ni * an + lane % an, ki * ak + slots * (lane // an) + p],
                                )
                                & T.uint16((1 << (16 // slots)) - 1)
                            )
                            << T.Cast("uint16", p * (16 // slots))
                        )
                    else:
                        B_bits[ki, ni] = B_bits[ki, ni] | (
                            (
                                T.Cast(
                                    "uint16",
                                    B_g[ki * ak + slots * (lane // an) + p, ni * an + lane % an],
                                )
                                & T.uint16((1 << (16 // slots)) - 1)
                            )
                            << T.Cast("uint16", p * (16 // slots))
                        )
        else:
            for mi, ki, p in T.grid(pm, pk, slots):
                if ta:
                    A_l[mi, ki, p] = A_g[ki * ak + slots * (lane // am) + p, mi * am + lane % am]
                else:
                    A_l[mi, ki, p] = A_g[mi * am + lane % am, ki * ak + slots * (lane // am) + p]
            for ki, ni, p in T.grid(pk, pn, slots):
                if tb:
                    B_l[ki, ni, p] = B_g[ni * an + lane % an, ki * ak + slots * (lane // an) + p]
                else:
                    B_l[ki, ni, p] = B_g[ki * ak + slots * (lane // an) + p, ni * an + lane % an]
        for mi, ni, p in T.grid(pm, pn, dslots):
            if acc == "float64":
                D_l[mi, ni, p] = C_g[mi * am + (lane // an) + 4 * p, ni * an + lane % an]
            else:
                D_l[mi, ni, p] = C_g[mi * am + dslots * (lane // an) + p, ni * an + lane % an]
        for iteration in T.unroll(2 if repeat else 1):
            if ta:
                if tb:
                    Tx.warp.gemm(
                        D_f[m0:M, n0:N],
                        A_f[k0:K, m0:M],
                        B_f[n0:N, k0:K],
                        D_f[m0:M, n0:N],
                        transpose_A=True,
                        transpose_B=True,
                        beta=beta,
                    )
                else:
                    Tx.warp.gemm(
                        D_f[m0:M, n0:N],
                        A_f[k0:K, m0:M],
                        B_f[k0:K, n0:N],
                        D_f[m0:M, n0:N],
                        transpose_A=True,
                        transpose_B=False,
                        beta=beta,
                    )
            else:
                if tb:
                    Tx.warp.gemm(
                        D_f[m0:M, n0:N],
                        A_f[m0:M, k0:K],
                        B_f[n0:N, k0:K],
                        D_f[m0:M, n0:N],
                        transpose_A=False,
                        transpose_B=True,
                        beta=beta,
                    )
                else:
                    Tx.warp.gemm(
                        D_f[m0:M, n0:N],
                        A_f[m0:M, k0:K],
                        B_f[k0:K, n0:N],
                        D_f[m0:M, n0:N],
                        transpose_A=False,
                        transpose_B=False,
                        beta=beta,
                    )
        for mi, ni, p in T.grid(pm, pn, dslots):
            if acc == "float64":
                D_g[mi * am + (lane // an) + 4 * p, ni * an + lane % an] = D_l[mi, ni, p]
            else:
                D_g[mi * am + dslots * (lane // an) + p, ni * an + lane % an] = D_l[mi, ni, p]

    return gemm, (M, N, K), (m0, n0, k0)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("family", _NUMERIC_FAMILIES)
@pytest.mark.parametrize("beta", [0.0, 1.0])
@pytest.mark.parametrize(
    "tiles,transpose,region,repeat",
    [
        ((1, 1, 1), (False, False), False, False),
        ((2, 2, 3), (False, False), False, False),
        ((2, 1, 2), (True, False), False, False),
        ((1, 2, 2), (False, True), False, False),
        ((2, 2, 3), (True, True), True, True),
    ],
)
def test_maca_gemm_mma_family_numerical(family, beta, tiles, transpose, region, repeat):
    """Check new signatures, tile offsets, in-place C=D, and repeated calls."""
    func, (M, N, K), (m0, n0, k0) = _build_family_numeric(
        family, tiles, beta, transpose, region, repeat
    )
    dtype, acc, _, _, ak = _NUMERIC_FAMILIES[family]
    rng = np.random.default_rng(20260910)
    if acc == "int32":
        lo, hi = {
            "int8": (-128, 128),
            "uint8": (0, 256),
            "int4": (-8, 8),
            "uint4": (0, 16),
            "int1": (0, 2),
        }[dtype]
        host_dtype = "int8" if dtype in ("int4", "int1") else "uint8" if dtype == "uint4" else dtype
        a = rng.integers(lo, hi, (M, K), dtype=host_dtype)
        b = rng.integers(lo, hi, (K, N), dtype=host_dtype)
        a.flat[:4] = [lo, hi - 1, 0, 1]
        b.flat[:4] = [hi - 1, lo, 1, 0]
        c = rng.integers(-1000, 1000, (M, N), dtype="int32")
        c[m0, n0 : n0 + 4] = [2**31 - 1, 2**31 - 2, -(2**31), -(2**31) + 1]
        if dtype == "int1":
            # The [m0,n0] dot is K-k0 for AND, but zero for XOR.
            a[m0, k0:] = 1
            b[k0:, n0] = 1
    else:
        np_dtype = _numpy_dtype(dtype) if dtype == "bfloat16" else dtype
        a = rng.uniform(-0.5, 0.5, (M, K)).astype(np_dtype)
        b = rng.uniform(-0.5, 0.5, (K, N)).astype(np_dtype)
        c = rng.uniform(-0.5, 0.5, (M, N)).astype(acc)
        if acc == "float16":
            # Dyadic values make each atom's f32 dot exact; differences then
            # measure f16 conversion between atoms, not f32 reduction order.
            a = (rng.integers(-32, 33, (M, K)) / 32).astype(dtype)
            b = (rng.integers(-32, 33, (K, N)) / 32).astype(dtype)
            c = (rng.integers(-32, 33, (M, N)) / 32).astype(acc)
        if beta == 0:
            c.fill(np.nan)
    reference = c.copy()
    result = c[m0:, n0:].copy()
    for _ in range(2 if repeat else 1):
        if beta == 0:
            result = np.zeros_like(result)
        for k in range(k0, K, ak):
            ref_dtype = "int64" if acc == "int32" else acc if acc == "float64" else "float32"
            product = a[m0:, k : k + ak].astype(ref_dtype) @ b[k : k + ak, n0:].astype(ref_dtype)
            result = (result.astype(ref_dtype) + product).astype(acc)
    reference[m0:, n0:] = result
    if acc == "float16" and tiles[2] > 1 and not repeat:
        final_only = a[m0:, k0:].astype("float32") @ b[k0:, n0:].astype("float32")
        if beta == 1:
            final_only += c[m0:, n0:].astype("float32")
        assert np.any(result != final_only.astype("float16")), (
            "fixture must expose per-atom rounding"
        )
    dev = tvm.device("maca", 0)
    a_dev = tvm.runtime.tensor(np.ascontiguousarray(a.T if transpose[0] else a), dev)
    b_dev = tvm.runtime.tensor(np.ascontiguousarray(b.T if transpose[1] else b), dev)
    c_dev = tvm.runtime.tensor(c, dev)
    d_dev = tvm.runtime.tensor(np.zeros((M, N), dtype=acc), dev)
    module = _compile(func)
    source = module.mod.imports[0].inspect_source()
    helper = {
        "f16_f32": "mma_m16n16k16_f16_f32",
        "bf16_f32": "mma_m16n16k16_bf16_f32",
        "f16_f16": "mma_m16n16k16_f16_f16",
        "f32_f32": "mma_m16n16k4_f32_f32",
        "f64_f64": "mma_m16n16k4_f64_f64",
        "i8_i32": "mma_m16n16k16_i8_i32",
        "u8_i32": "mma_m16n16k16_u8_i32",
        "i4_i32": "mma_m8n8k32_i4_i32",
        "u4_i32": "mma_m8n8k32_u4_i32",
        "b1_i32": "bmma_m8n8k128_b1_i32",
    }[family]
    assert source.count("tvm_builtin_maca_" + helper + "(") >= 2
    assert "tcgen05" not in source and "asm volatile" not in source
    module(a_dev, b_dev, c_dev, d_dev)
    if acc == "int32":
        np.testing.assert_array_equal(d_dev.numpy(), reference)
    elif acc == "float16":
        # Native MMA cancellation can leave one minimum half subnormal.
        np.testing.assert_allclose(d_dev.numpy(), reference, rtol=0, atol=2**-24, equal_nan=True)
    elif acc == "float64":
        np.testing.assert_allclose(d_dev.numpy(), reference, rtol=1e-12, atol=1e-14, equal_nan=True)
    else:
        np.testing.assert_allclose(d_dev.numpy(), reference, rtol=1e-5, atol=1e-6, equal_nan=True)


def _numpy_dtype(dtype):
    if dtype == "bfloat16":
        return pytest.importorskip("ml_dtypes").bfloat16
    return np.float16


def _compile(func):
    target = tvm.target.Target("maca")
    with target:
        return tvm.compile(tvm.IRModule({"main": func}), target=target, tir_pipeline="tirx")


def test_cuda_gemm_mma_variant_is_registered():
    schedules = list_registered_schedules()
    maca_gemm = schedules.get("tirx.tile.gemm", {}).get("maca", [])
    assert any(variant.startswith("mma.m16n16k16") for variant in maca_gemm), (
        "m16n16k16 MACA GEMM variant is not registered; "
        f"tirx.tile.gemm schedules = {schedules.get('tirx.tile.gemm')}"
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.gpu
def test_cuda_gemm_mma_lowers_to_mma_sync(dtype):
    """beta=0 clears D then invokes the matching MACA MMA intrinsic."""
    script = _lower(_build_gemm(alpha=1.0, beta=0.0, dtype=dtype))["main"].script()

    assert _SCRIPT_INTRINSIC[dtype] in script
    assert "T.float32(0" in script
    assert "wmma" not in script.lower()


@pytest.mark.gpu
def test_cuda_gemm_mma_accumulates_c_when_beta_one():
    """beta=1 initializes D from C before invoking the MACA intrinsic."""
    script = _lower(_build_gemm(alpha=1.0, beta=1.0))["main"].script()

    assert _SCRIPT_INTRINSIC["bfloat16"] in script
    assert "c_local[" in script
    assert "T.float32(0" not in script


@pytest.mark.parametrize(
    "dtypes, intrinsic",
    [
        (("float16", "float16", "float16", "float16"), _SCRIPT_INTRINSIC["float16_acc"]),
        (("int8", "int8", "int32", "int32"), _SCRIPT_INTRINSIC["int8"]),
        (("uint8", "uint8", "int32", "int32"), _SCRIPT_INTRINSIC["uint8"]),
    ],
)
@pytest.mark.gpu
def test_maca_gemm_mma_lowers_additional_m16n16k16_families(dtypes, intrinsic):
    """Additional C500 signatures select their typed direct intrinsic."""
    script = _lower(_build_dtypes(*dtypes))["main"].script()
    assert intrinsic in script
    assert "wmma" not in script.lower()


@pytest.mark.gpu
def test_maca_gemm_mma_lowers_full_f32_atom():
    script = _lower(_build_f32_gemm())["main"].script()
    assert "T.maca.mma_m16n16k4_f32_f32(" in script


@pytest.mark.parametrize(
    "dtype, atom, intrinsic",
    [
        ("int4", (8, 8, 32), "T.maca.mma_m8n8k32_i4_i32("),
        ("uint4", (8, 8, 32), "T.maca.mma_m8n8k32_u4_i32("),
        ("int1", (8, 8, 128), "T.maca.bmma_m8n8k128_b1_i32("),
    ],
)
@pytest.mark.gpu
def test_maca_gemm_mma_lowers_packed_atoms(dtype, atom, intrinsic):
    script = _lower(_build_packed_gemm(dtype, atom))["main"].script()
    assert intrinsic in script


def test_cuda_gemm_mma_rejects_nonunit_alpha():
    with pytest.raises(RuntimeError, match="dispatch failed"):
        _lower(_build_gemm(alpha=2.0, beta=0.0))


def test_cuda_gemm_mma_rejects_fractional_beta():
    with pytest.raises(RuntimeError, match="dispatch failed"):
        _lower(_build_gemm(alpha=1.0, beta=0.5))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cuda_gemm_mma_numerical(dtype):
    """End-to-end D = A @ B on one Wave64 m16n16k16 tile."""
    np_dtype = _numpy_dtype(dtype)
    func, M, N, K = _build_tiled_numeric(1, 1, 1, 0.0, dtype)
    mod = _compile(func)

    np.random.seed(0)
    A_np = np.random.uniform(-1, 1, (M, K)).astype(np.float32)
    B_np = np.random.uniform(-1, 1, (K, N)).astype(np.float32)
    golden = A_np @ B_np

    def run_and_check():
        dev = tvm.maca(0)
        A_dev = tvm.runtime.tensor(A_np.astype(np_dtype), dev)
        B_dev = tvm.runtime.tensor(B_np.astype(np_dtype), dev)
        C_dev = tvm.runtime.tensor(np.zeros((M, N), np.float32), dev)
        D_dev = tvm.runtime.tensor(np.zeros((M, N), np.float32), dev)
        mod(A_dev, B_dev, C_dev, D_dev)
        tvm.testing.assert_allclose(golden, D_dev.numpy(), atol=2e-2, rtol=2e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


_TILED_SHAPES = [
    (1, 1, 1),
    (2, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
    (2, 2, 2),
    (4, 1, 1),
]
_TILED_MODES = [
    ("float16", 0.0),
    ("bfloat16", 0.0),
    ("float16", 1.0),
    ("bfloat16", 1.0),
]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("Mt, Nt, Kt", _TILED_SHAPES)
@pytest.mark.parametrize("dtype, beta", _TILED_MODES)
def test_cuda_gemm_mma_numerical_tiled(dtype, beta, Mt, Nt, Kt):
    """End-to-end D = A @ B (+ C when beta is one) for tiled fragments."""
    np_dtype = _numpy_dtype(dtype)
    func, M, N, K = _build_tiled_numeric(Mt, Nt, Kt, beta, dtype)
    mod = _compile(func)

    np.random.seed(0)
    A_np = np.random.uniform(-1, 1, (M, K)).astype(np.float32)
    B_np = np.random.uniform(-1, 1, (K, N)).astype(np.float32)
    C_np = np.random.uniform(-1, 1, (M, N)).astype(np.float32)
    golden = A_np @ B_np + (C_np if beta == 1.0 else 0.0)

    def run_and_check():
        dev = tvm.maca(0)
        A_dev = tvm.runtime.tensor(A_np.astype(np_dtype), dev)
        B_dev = tvm.runtime.tensor(B_np.astype(np_dtype), dev)
        C_dev = tvm.runtime.tensor(C_np, dev)
        D_dev = tvm.runtime.tensor(np.zeros((M, N), np.float32), dev)
        mod(A_dev, B_dev, C_dev, D_dev)
        tvm.testing.assert_allclose(golden, D_dev.numpy(), atol=2e-2, rtol=2e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "transpose_A, transpose_B",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_cuda_gemm_mma_numerical_transpose(transpose_A, transpose_B, dtype):
    """End-to-end D = A @ B for every A/B logical orientation."""
    np_dtype = _numpy_dtype(dtype)
    func = _build_transpose_numeric(transpose_A, transpose_B, dtype)
    mod = _compile(func)

    np.random.seed(0)
    A_log = np.random.uniform(-1, 1, (MMA_M, MMA_K)).astype(np.float32)
    B_log = np.random.uniform(-1, 1, (MMA_K, MMA_N)).astype(np.float32)
    A_buf = (A_log.T if transpose_A else A_log).astype(np_dtype)
    B_buf = (B_log.T if transpose_B else B_log).astype(np_dtype)
    golden = A_log @ B_log

    def run_and_check():
        dev = tvm.maca(0)
        A_dev = tvm.runtime.tensor(A_buf, dev)
        B_dev = tvm.runtime.tensor(B_buf, dev)
        D_dev = tvm.runtime.tensor(np.zeros((MMA_M, MMA_N), np.float32), dev)
        mod(A_dev, B_dev, D_dev)
        tvm.testing.assert_allclose(golden, D_dev.numpy(), atol=2e-2, rtol=2e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize("Mt, Nt, Kt", _TILED_SHAPES[1:])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.gpu
def test_cuda_gemm_mma_lowers_tiled(Mt, Nt, Kt, dtype):
    """Every supported m16n16k16 tiling lowers to its MACA intrinsic."""
    script = _lower(_build_tiled(Mt, Nt, Kt, dtype=dtype))["main"].script()
    assert _SCRIPT_INTRINSIC[dtype] in script


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("Mt, Nt, Kt", [(1, 1, 1), (2, 2, 2), (4, 1, 1)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cuda_gemm_mma_codegen_issue_count(Mt, Nt, Kt, dtype):
    """Codegen emits the matching C500 builtin, without a WMMA wrapper."""
    func, _, _, _ = _build_tiled_numeric(Mt, Nt, Kt, 0.0, dtype)
    src = _compile(func).mod.imports[0].inspect_source()
    assert _SOURCE_BUILTIN[dtype] in src
    assert "wmma" not in src.lower()


@pytest.mark.parametrize(
    "transpose_A, transpose_B",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.gpu
def test_cuda_gemm_mma_lowers_transpose(transpose_A, transpose_B):
    """All input orientations use the same m16n16k16 MACA instruction."""
    script = _lower(_build_transpose(transpose_A, transpose_B))["main"].script()
    assert _SCRIPT_INTRINSIC["float16"] in script


@pytest.mark.parametrize(
    "transpose_A, transpose_B",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.gpu
def test_cuda_gemm_mma_lowers_aligned_nonzero_region(transpose_A, transpose_B):
    """A nonzero 16-aligned region lowers for every input orientation."""
    script = _lower(_build_aligned_region_slice(transpose_A, transpose_B))["main"].script()
    assert _SCRIPT_INTRINSIC["float16"] in script


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize(
    "transpose_A, transpose_B",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_cuda_gemm_mma_codegen_transpose(transpose_A, transpose_B):
    """Every input orientation reaches the native f16 C500 builtin."""
    src = (
        _compile(_build_transpose_numeric(transpose_A, transpose_B)).mod.imports[0].inspect_source()
    )
    assert _SOURCE_BUILTIN["float16"] in src
    assert "wmma" not in src.lower()


@pytest.mark.parametrize(
    "a, b, c, d",
    [
        ("bfloat16", "float16", "float32", "float32"),
        ("float32", "float16", "float32", "float32"),
        ("int8", "uint8", "int32", "int32"),
        ("uint8", "int8", "int32", "int32"),
        ("float16", "float16", "int32", "int32"),
        ("float16", "float16", "float32", "float16"),
    ],
)
def test_cuda_gemm_mma_rejects_unsupported_dtype(a, b, c, d):
    """Mixed input signedness and incompatible accumulator signatures decline."""
    with pytest.raises(RuntimeError, match="does not support dtype signature"):
        _lower(_build_dtypes(a, b, c, d))


def _build_contract_case(case):
    """Vary one register-fragment contract while keeping the signature valid."""
    dl, al, _ = _frag(3, 1, 1)
    a_layout = dl if case == "layout" else al
    a_scope = "shared" if case == "scope" else "local"
    start = -16 if case == "negative" else 1 if case == "alignment" else 0
    extent = {"empty": 0, "bounds": 64, "overlap": 32}.get(case, 16)
    c_start = 16 if case in ("overlap", "disjoint") else 0

    @T.prim_func
    def gemm():
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        _lane = T.lane_id([WAVE_SIZE])
        D = T.alloc_buffer((48, 16), "float16", scope="local", layout=dl)
        A_storage = T.alloc_buffer((48, 16), "float16", scope=a_scope, layout=a_layout)
        B_storage = T.alloc_buffer((16, 16), "float16", scope="local", layout=B_ATOM)
        if case == "alias_a":
            A = T.decl_buffer((48, 16), "float16", data=D.data, scope="local", layout=al)
        else:
            A = A_storage
        if case == "alias_b":
            B = T.decl_buffer((16, 16), "float16", data=D.data, scope="local", layout=B_ATOM)
        else:
            B = B_storage
        Tx.warp.gemm(
            D[start : start + extent, 0:16],
            A[0:extent, 0:16],
            B,
            D[c_start : c_start + extent, 0:16],
            transpose_A=False,
            transpose_B=False,
            beta=1.0,
        )

    return gemm


@pytest.mark.parametrize(
    "case, diagnostic",
    [
        ("alignment", "requires buffer and region alignment"),
        ("bounds", "in-bounds nonempty regions"),
        ("empty", "in-bounds nonempty regions"),
        ("negative", "in-bounds nonempty regions"),
        ("layout", "unsupported A fragment layout"),
        ("scope", "requires local A fragments"),
        ("alias_a", "D must not alias A"),
        ("alias_b", "D must not alias B"),
        ("overlap", "rejects shifted overlapping C/D regions"),
    ],
)
def test_maca_gemm_mma_rejects_invalid_contract(case, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        _lower(_build_contract_case(case))


def test_maca_gemm_mma_accepts_disjoint_c_d_regions():
    script = _lower(_build_contract_case("disjoint"))["main"].script()
    assert "T.maca.mma_m16n16k16_f16_f16(" in script


def test_maca_gemm_mma_rejects_other_architecture():
    with tvm.target.Target({"kind": "maca", "mcpu": "xcore1100"}):
        with pytest.raises(RuntimeError, match="requires the C500/xcore1000 target"):
            tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": _build_gemm()}))


if __name__ == "__main__":
    tvm.testing.main()
