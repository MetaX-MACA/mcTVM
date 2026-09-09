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
"""Tests for the MACA Wave64 ``m16n16k16`` register GEMM dispatch.

The dispatch lowers ``tirx.tile.gemm`` over pure-register fragments to the
MACA C builtins for f16/bf16 inputs with f32 accumulation.  This follows the
SDK WMMA m16n16k16 fragment mapping.  A Wave64 lane maps its four local
operand and accumulator slots as follows:

    lane = 16 * group + row
    A[p] = A[row, 4 * group + p]
    B[p] = B[4 * group + p, row]
    C/D[p] = C/D[4 * group + p, row]

The multi-tile layouts retain these atom coordinates as their innermost axes
and use positive physical strides for the outer tile axes.  Numerical tests
load and store through this mapping directly.
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

_SCRIPT_INTRINSIC = {
    "float16": "T.maca.mma_m16n16k16_f16_f32(",
    "bfloat16": "T.maca.mma_m16n16k16_bf16_f32(",
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


def _numpy_dtype(dtype):
    if dtype == "bfloat16":
        return pytest.importorskip("ml_dtypes").bfloat16
    return np.float16


def _compile(func):
    target = tvm.target.Target("maca")
    with target:
        return tvm.compile(tvm.IRModule({"main": func}), target=target, tir_pipeline="tirx")


def test_maca_gemm_mma_variant_is_registered():
    schedules = list_registered_schedules()
    maca_gemm = schedules.get("tirx.tile.gemm", {}).get("maca", [])
    assert any(variant.startswith("mma.m16n16k16") for variant in maca_gemm), (
        "m16n16k16 MACA GEMM variant is not registered; "
        f"tirx.tile.gemm schedules = {schedules.get('tirx.tile.gemm')}"
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.gpu
def test_maca_gemm_mma_lowers_to_builtin(dtype):
    """beta=0 clears D then invokes the matching MACA MMA intrinsic."""
    script = _lower(_build_gemm(alpha=1.0, beta=0.0, dtype=dtype))["main"].script()

    assert _SCRIPT_INTRINSIC[dtype] in script
    assert "T.float32(0" in script
    assert "wmma" not in script.lower()


@pytest.mark.gpu
def test_maca_gemm_mma_accumulates_c_when_beta_one():
    """beta=1 initializes D from C before invoking the MACA intrinsic."""
    script = _lower(_build_gemm(alpha=1.0, beta=1.0))["main"].script()

    assert _SCRIPT_INTRINSIC["bfloat16"] in script
    assert "c_local[" in script
    assert "T.float32(0" not in script


def test_maca_gemm_mma_rejects_nonunit_alpha():
    with pytest.raises(RuntimeError, match="dispatch failed"):
        _lower(_build_gemm(alpha=2.0, beta=0.0))


def test_maca_gemm_mma_rejects_fractional_beta():
    with pytest.raises(RuntimeError, match="dispatch failed"):
        _lower(_build_gemm(alpha=1.0, beta=0.5))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_maca_gemm_mma_numerical(dtype):
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
def test_maca_gemm_mma_numerical_tiled(dtype, beta, Mt, Nt, Kt):
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
def test_maca_gemm_mma_numerical_transpose(transpose_A, transpose_B, dtype):
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
def test_maca_gemm_mma_lowers_tiled(Mt, Nt, Kt, dtype):
    """Every supported m16n16k16 tiling lowers to its MACA intrinsic."""
    script = _lower(_build_tiled(Mt, Nt, Kt, dtype=dtype))["main"].script()
    assert _SCRIPT_INTRINSIC[dtype] in script


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize("Mt, Nt, Kt", [(1, 1, 1), (2, 2, 2), (4, 1, 1)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_maca_gemm_mma_codegen_uses_direct_builtin(Mt, Nt, Kt, dtype):
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
def test_maca_gemm_mma_lowers_transpose(transpose_A, transpose_B):
    """All input orientations use the same m16n16k16 MACA instruction."""
    script = _lower(_build_transpose(transpose_A, transpose_B))["main"].script()
    assert _SCRIPT_INTRINSIC["float16"] in script


@pytest.mark.parametrize(
    "transpose_A, transpose_B",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.gpu
def test_maca_gemm_mma_lowers_aligned_nonzero_region(transpose_A, transpose_B):
    """A nonzero 16-aligned region lowers for every input orientation."""
    script = _lower(_build_aligned_region_slice(transpose_A, transpose_B))["main"].script()
    assert _SCRIPT_INTRINSIC["float16"] in script


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize(
    "transpose_A, transpose_B",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_maca_gemm_mma_codegen_transpose(transpose_A, transpose_B):
    """Every input orientation reaches the native f16 C500 builtin."""
    src = (
        _compile(_build_transpose_numeric(transpose_A, transpose_B)).mod.imports[0].inspect_source()
    )
    assert _SOURCE_BUILTIN["float16"] in src
    assert "wmma" not in src.lower()


@pytest.mark.parametrize(
    "a, b, c, d",
    [
        ("float16", "float16", "float16", "float16"),
        ("bfloat16", "float16", "float32", "float32"),
        ("float32", "float32", "float32", "float32"),
        ("int8", "int8", "int32", "int32"),
    ],
)
def test_maca_gemm_mma_rejects_unsupported_dtype(a, b, c, d):
    """Only matching f16/bf16 inputs with f32 accumulation are supported."""
    with pytest.raises(RuntimeError, match="dispatch failed"):
        _lower(_build_dtypes(a, b, c, d))


if __name__ == "__main__":
    tvm.testing.main()
