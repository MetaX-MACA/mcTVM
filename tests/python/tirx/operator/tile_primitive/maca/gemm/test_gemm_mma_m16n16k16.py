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
"""Tests for the MACA Wave64 register GEMM tile primitive.

For the m16n16k16 families, a lane maps its four local operand and
accumulator slots as follows:

    lane = 16 * group + row
    A[p] = A[row, 4 * group + p]
    B[p] = B[4 * group + p, row]
    C/D[p] = C/D[4 * group + p, row]

The multi-tile layouts retain these atom coordinates as their innermost axes
and use positive physical strides for the outer tile axes.  Numerical tests
load and store through this mapping directly for the active native MMA
families.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.layout import R, S, TileLayout, laneid

pytestmark = pytest.mark.skipif(
    not env.has_maca_arch("xcore1000"),
    reason="requires MACA xcore1000",
)

MMA_M = 16
MMA_N = 16
MMA_K = 16
WAVE_SIZE = 64

# Explicit test-side signatures; these maps do not import dispatch internals.
_NUMERIC_FAMILIES = {
    "f16_f32": ("float16", "float32", 16, 16, 16),
    "bf16_f32": ("bfloat16", "float32", 16, 16, 16),
    "f32_f32": ("float32", "float32", 16, 16, 4),
    "tf32_f32": ("float32", "float32", 16, 16, 8),
    "f64_f64": ("float64", "float64", 16, 16, 4),
    "i8_i32": ("int8", "int32", 16, 16, 16),
}

_SCRIPT_INTRINSIC = {
    "float16": "T.maca.mma_m16n16k16_f16_f32(",
    "bfloat16": "T.maca.mma_m16n16k16_bf16_f32(",
    "int8": "T.maca.mma_m16n16k16_i8_i32(",
}
_LOWERING_CASES = [
    ("f16_f32", "T.maca.mma_m16n16k16_f16_f32("),
    ("bf16_f32", "T.maca.mma_m16n16k16_bf16_f32("),
    ("i8_i32", "T.maca.mma_m16n16k16_i8_i32("),
    ("f32_f32", "T.maca.mma_m16n16k4_f32_f32("),
    ("tf32_f32", "T.maca.mma_m16n16k8_tf32_f32("),
    ("f64_f64", "T.maca.mma_m16n16k4_f64_f64("),
]

_CODEGEN_CASES = [
    ("f16_f32", "tvm_builtin_maca_mma_m16n16k16_f16_f32("),
    ("bf16_f32", "tvm_builtin_maca_mma_m16n16k16_bf16_f32("),
    ("i8_i32", "tvm_builtin_maca_mma_m16n16k16_i8_i32("),
    ("f32_f32", "tvm_builtin_maca_mma_m16n16k4_f32_f32("),
    ("tf32_f32", "tvm_builtin_maca_mma_m16n16k8_tf32_f32("),
    ("f64_f64", "tvm_builtin_maca_mma_m16n16k4_f64_f64("),
]

# For lane = 16 * group + row, the source register index p is represented by
# the final extent-4 axis and maps to the logical coordinate 4 * group + p.
A_FRAG = TileLayout(S[(16, 4, 4) : (1 @ laneid, 16 @ laneid, 1)])
B_FRAG = TileLayout(S[(4, 4, 16) : (16 @ laneid, 1, 1 @ laneid)])
D_FRAG = TileLayout(S[(4, 4, 16) : (16 @ laneid, 1, 1 @ laneid)])


def _transpose_frag(layout, shape):
    """Swap the two logical axes of a 2D fragment layout.

    The transposed input orientations (A as [K, M], B as [N, K]) hold the exact
    same per-lane/per-register element distribution as the K-major fragments --
    only the buffer's logical axes are swapped. So instead of writing them out
    by hand, derive them: ``group`` the shard into the logical dims, then
    ``permute_by_groups`` to exchange the two groups.
    """
    grouped, seps = layout.group(shape)
    return grouped.permute_by_groups(seps, [1, 0])


# Transposed input orientations of the same single tile: A as [K, M], B as
# [N, K]. The dispatch swaps axes per the transpose flags; the .row.col mma is
# unchanged.
A_KM_FRAG = _transpose_frag(A_FRAG, [MMA_M, MMA_K])
B_NK_FRAG = _transpose_frag(B_FRAG, [MMA_K, MMA_N])


def _frag(Mt, Nt, Kt, kinst):
    """Return Wave64 fragments for an ``Mt x Nt x Kt`` m16n16k16 tiling.

    Logical shapes: A = (16*Mt, kinst*Kt), B = (kinst*Kt, 8*Nt), D/C = (16*Mt, 8*Nt).
    Each operand's tiled layout is the single-tile base ``tile_to`` the full
    logical shape -- ``tile_to`` repeats the base's per-lane/per-register element
    map over the tile grid, so a tiling is just a grid of the single-tile
    fragments (the base is the single source of truth, k8 and k16 alike).

    The positive outer strides make a per-thread local view have physical
    shape ``[Mt, Kt, 4]`` for A,
    ``[Kt, Nt, 4]`` for B, and ``[Mt, Nt, 4]`` for C/D.
    """
    if min(Mt, Nt, Kt) < 1:
        raise ValueError("tile counts must be positive")
    if kinst != MMA_K:
        raise ValueError(f"MACA only supports kinst={MMA_K}, but got {kinst}")

    A = TileLayout(S[(Mt, MMA_M, Kt, 4, 4) : (Kt * 4, 1 @ laneid, 4, MMA_K @ laneid, 1)])
    B = TileLayout(S[(Kt, 4, 4, Nt, MMA_N) : (Nt * 4, MMA_K @ laneid, 1, 4, 1 @ laneid)])
    D = TileLayout(S[(Mt, 4, 4, Nt, MMA_N) : (Nt * 4, 16 @ laneid, 1, 4, 1 @ laneid)])
    return D, A, B


def _build_tiled(Mt, Nt, Kt, kinst, *, alpha=1.0, beta=0.0, dtype="float16"):
    """A single-warp kernel issuing one ``T.gemm`` over an Mt x Nt x Kt tiling.

    With ``store=True`` the result is written back to a global buffer (a full
    kernel for codegen); otherwise only the ``T.gemm`` is emitted (for
    ``LowerTIRx`` dispatch checks).
    """
    Dl, Al, Bl = _frag(Mt, Nt, Kt, kinst)
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


def _build_dtypes(a_dtype, b_dtype, c_dtype, d_dtype):
    """Build a single tile with explicit operand dtypes for decline checks."""

    @T.prim_func
    def gemm():
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        _lane = T.lane_id([WAVE_SIZE])
        A = T.alloc_buffer((MMA_M, MMA_K), a_dtype, scope="local", layout=A_FRAG)
        B = T.alloc_buffer((MMA_K, MMA_N), b_dtype, scope="local", layout=B_FRAG)
        C = T.alloc_buffer((MMA_M, MMA_N), c_dtype, scope="local", layout=D_FRAG)
        D = T.alloc_buffer((MMA_M, MMA_N), d_dtype, scope="local", layout=D_FRAG)
        Tx.warp.gemm(D, A, B, C, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0)

    return gemm


_DEFAULT_PRECISION = object()


def _build_signature_gemm(family, precision=_DEFAULT_PRECISION):
    """Build one call from layouts specified independently by this test."""
    dtype, acc, _, _, reduction = _NUMERIC_FAMILIES[family]
    if reduction == 16:
        a_layout, b_layout, d_layout = A_FRAG, B_FRAG, D_FRAG
    elif reduction == 8:
        a_layout = TileLayout(
            S[(1, 16, 1, 4, 2) : (2, 1 @ laneid, 2, -16 @ laneid, 1)] + 48 @ laneid
        )
        b_layout = TileLayout(
            S[(1, 4, 2, 1, 16) : (2, -16 @ laneid, 1, 2, 1 @ laneid)] + 48 @ laneid
        )
        d_layout = D_FRAG
    else:
        a_layout = TileLayout(S[(16, 4) : (1 @ laneid, 16 @ laneid)])
        b_layout = TileLayout(S[(4, 16) : (16 @ laneid, 1 @ laneid)])
        d_layout = (
            TileLayout(S[(4, 4, 16) : (1, 16 @ laneid, 1 @ laneid)]) if acc == "float64" else D_FRAG
        )
    if precision is _DEFAULT_PRECISION:
        precision = "tf32" if family == "tf32_f32" else None

    @T.prim_func
    def gemm():
        T.device_entry()
        _cta = T.cta_id([1])
        _wave = T.warp_id([1])
        _lane = T.lane_id([WAVE_SIZE])
        A = T.alloc_buffer((MMA_M, reduction), dtype, scope="local", layout=a_layout)
        B = T.alloc_buffer((reduction, MMA_N), dtype, scope="local", layout=b_layout)
        C = T.alloc_buffer((MMA_M, MMA_N), acc, scope="local", layout=d_layout)
        D = T.alloc_buffer((MMA_M, MMA_N), acc, scope="local", layout=d_layout)
        Tx.warp.gemm(
            D,
            A,
            B,
            C,
            transpose_A=False,
            transpose_B=False,
            alpha=1.0,
            beta=0.0,
            precision=precision,
        )

    return gemm


def _lower(func):
    with tvm.target.Target("maca"):
        return tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": func}))


@pytest.mark.parametrize("family,intrinsic", _LOWERING_CASES)
def test_maca_gemm_mma_lowers_supported_signatures(family, intrinsic):
    script = _lower(_build_signature_gemm(family))["main"].script()
    assert intrinsic in script


@pytest.mark.parametrize("family,helper", _CODEGEN_CASES)
def test_maca_gemm_mma_codegen_uses_expected_helper(family, helper):
    with tvm.target.Target("maca"):
        module = tvm.compile(
            tvm.IRModule({"main": _build_signature_gemm(family)}),
            target="maca",
            tir_pipeline="tirx",
        )
    assert helper in module.mod.imports[0].inspect_source()


def test_maca_gemm_mma_codegen_tiled_issue_count():
    with tvm.target.Target("maca"):
        module = tvm.compile(
            tvm.IRModule({"main": _build_tiled(2, 2, 3, MMA_K)}),
            target="maca",
            tir_pipeline="tirx",
        )
    source = module.mod.imports[0].inspect_source()
    assert source.count("tvm_builtin_maca_mma_m16n16k16_f16_f32(") - 1 == 2 * 2 * 3


@pytest.mark.parametrize(
    "family,precision,diagnostic",
    [
        ("tf32_f32", None, "TF32 MMA requires precision='tf32'"),
        ("f32_f32", "unsupported", "unsupported GEMM precision 'unsupported'"),
    ],
)
def test_maca_gemm_mma_rejects_invalid_precision(family, precision, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        _lower(_build_signature_gemm(family, precision))


def _build_aligned_region_slice(transpose_A, transpose_B, *, dtype="float16"):
    """Build one nonzero, 16-aligned m16n16k16 region GEMM.

    The full buffers contain multiple tiles.  The logical operation selects
    M=[16, 32), N=[16, 32), K=[32, 48), so the dispatcher must retain the
    per-operand tile offsets while normalizing either input orientation.
    """
    M, N, K = 32, 48, 48
    Dl, Al, Bl = _frag(M // MMA_M, N // MMA_N, K // MMA_K, MMA_K)
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


def _build_family_numeric(family, tiles, beta, transpose, region=False, repeat=False):
    """Load lane payloads independently of the production layout builders."""
    dtype, acc, am, an, ak = _NUMERIC_FAMILIES[family]
    mt, nt, kt = tiles
    pad = int(region)
    pm, pn, pk = mt + pad, nt + pad, kt + pad
    M, N, K = pm * am, pn * an, pk * ak
    if ak == 4:
        Al = TileLayout(S[(pm, 16, pk, 4) : (pk, 1 @ laneid, 1, 16 @ laneid)])
        Bl = TileLayout(S[(pk, 4, pn, 16) : (pn, 16 @ laneid, 1, 1 @ laneid)])
        Dl, _, _ = _frag(pm, pn, pk, MMA_K)
        if acc == "float64":
            Dl = TileLayout(S[(pm, 4, 4, pn, 16) : (pn * 4, 1, 16 @ laneid, 4, 1 @ laneid)])
        slots = 1
    elif ak == 8:
        Al = TileLayout(
            S[(pm, 16, pk, 4, 2) : (pk * 2, 1 @ laneid, 2, -16 @ laneid, 1)] + 48 @ laneid
        )
        Bl = TileLayout(
            S[(pk, 4, 2, pn, 16) : (pn * 2, -16 @ laneid, 1, 2, 1 @ laneid)] + 48 @ laneid
        )
        Dl, _, _ = _frag(pm, pn, pk, MMA_K)
        slots = 2
    else:
        Dl, Al, Bl = _frag(pm, pn, pk, MMA_K)
        slots = 4
    ta, tb = transpose
    Al = _transpose_frag(Al, [M, K]) if ta else Al
    Bl = _transpose_frag(Bl, [K, N]) if tb else Bl
    ashape, bshape = ((K, M) if ta else (M, K)), ((N, K) if tb else (K, N))
    m0, n0, k0 = pad * am, pad * an, pad * ak
    dslots = am * an // WAVE_SIZE
    host_dtype = dtype
    precision = "tf32" if family == "tf32_f32" else None

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
        for mi, ki, p in T.grid(pm, pk, slots):
            k_group = 3 - lane // am if ak == 8 else lane // am
            if ta:
                A_l[mi, ki, p] = A_g[ki * ak + slots * k_group + p, mi * am + lane % am]
            else:
                A_l[mi, ki, p] = A_g[mi * am + lane % am, ki * ak + slots * k_group + p]
        for ki, ni, p in T.grid(pk, pn, slots):
            k_group = 3 - lane // an if ak == 8 else lane // an
            if tb:
                B_l[ki, ni, p] = B_g[ni * an + lane % an, ki * ak + slots * k_group + p]
            else:
                B_l[ki, ni, p] = B_g[ki * ak + slots * k_group + p, ni * an + lane % an]
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
                        precision=precision,
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
                        precision=precision,
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
                        precision=precision,
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
                        precision=precision,
                    )
        for mi, ni, p in T.grid(pm, pn, dslots):
            if acc == "float64":
                D_g[mi * am + (lane // an) + 4 * p, ni * an + lane % an] = D_l[mi, ni, p]
            else:
                D_g[mi * am + dslots * (lane // an) + p, ni * an + lane % an] = D_l[mi, ni, p]

    return gemm, (M, N, K), (m0, n0, k0)


def _quantize_tf32(values):
    """Truncate finite float32 values to the native TF32 multiplier format."""
    values = np.asarray(values, dtype="float32")
    bits = values.view("uint32")
    return (bits & np.uint32(0xFFFFE000)).view("float32")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_maca(), reason="need maca")
@pytest.mark.parametrize(
    "family,beta,tiles,transpose,region,repeat",
    [
        *(
            (family, beta, (1, 1, 1), (False, False), False, False)
            for family in _NUMERIC_FAMILIES
            for beta in (0.0, 1.0)
        ),
        ("f16_f32", 0.0, (2, 2, 3), (False, False), False, False),
        ("f16_f32", 0.0, (1, 1, 1), (True, False), False, False),
        ("f16_f32", 0.0, (1, 1, 1), (False, True), False, False),
        ("f16_f32", 0.0, (1, 1, 1), (True, True), False, False),
        ("f16_f32", 1.0, (2, 2, 3), (True, True), True, True),
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
        }[dtype]
        host_dtype = dtype
        a = rng.integers(lo, hi, (M, K), dtype=host_dtype)
        b = rng.integers(lo, hi, (K, N), dtype=host_dtype)
        a.flat[:4] = [lo, hi - 1, 0, 1]
        b.flat[:4] = [hi - 1, lo, 1, 0]
        c = rng.integers(-1000, 1000, (M, N), dtype="int32")
        c[m0, n0 : n0 + 4] = [2**31 - 1, 2**31 - 2, -(2**31), -(2**31) + 1]
    else:
        np_dtype = pytest.importorskip("ml_dtypes").bfloat16 if dtype == "bfloat16" else dtype
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
    a_reference = _quantize_tf32(a) if family == "tf32_f32" else a
    b_reference = _quantize_tf32(b) if family == "tf32_f32" else b
    for _ in range(2 if repeat else 1):
        if beta == 0:
            result = np.zeros_like(result)
        for k in range(k0, K, ak):
            ref_dtype = "int64" if acc == "int32" else acc if acc == "float64" else "float32"
            product = a_reference[m0:, k : k + ak].astype(ref_dtype) @ b_reference[
                k : k + ak, n0:
            ].astype(ref_dtype)
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
    with tvm.target.Target("maca"):
        module = tvm.compile(tvm.IRModule({"main": func}), target="maca", tir_pipeline="tirx")
    source = module.mod.imports[0].inspect_source()
    helper = {
        "f16_f32": "mma_m16n16k16_f16_f32",
        "bf16_f32": "mma_m16n16k16_bf16_f32",
        "f32_f32": "mma_m16n16k4_f32_f32",
        "tf32_f32": "mma_m16n16k8_tf32_f32",
        "f64_f64": "mma_m16n16k4_f64_f64",
        "i8_i32": "mma_m16n16k16_i8_i32",
    }[family]
    assert source.count("tvm_builtin_maca_" + helper + "(") >= 2
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


@pytest.mark.parametrize(
    "transpose_A, transpose_B",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.gpu
def test_maca_gemm_mma_lowers_aligned_nonzero_region(transpose_A, transpose_B):
    """A nonzero 16-aligned region lowers for every input orientation."""
    script = _lower(_build_aligned_region_slice(transpose_A, transpose_B))["main"].script()
    assert _SCRIPT_INTRINSIC["float16"] in script


def _build_contract_case(case):
    """Vary one register-fragment contract while keeping the signature valid."""
    dl, al, _ = _frag(3, 1, 1, MMA_K)
    replica_layout = TileLayout(
        S[(3, 16, 1, 4, 4) : (4, 1 @ laneid, 4, 16 @ laneid, 1)] + R[2 : 64 @ laneid]
    )
    a_layout = dl if case == "layout" else replica_layout if case == "replica" else al
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
        D = T.alloc_buffer((48, 16), "float32", scope="local", layout=dl)
        A_storage = T.alloc_buffer((48, 16), "float16", scope=a_scope, layout=a_layout)
        B_storage = T.alloc_buffer((16, 16), "float16", scope="local", layout=B_FRAG)
        if case == "alias_a":
            A = T.decl_buffer((48, 16), "float16", data=D.data, scope="local", layout=al)
        else:
            A = A_storage
        if case == "alias_b":
            B = T.decl_buffer((16, 16), "float16", data=D.data, scope="local", layout=B_FRAG)
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
        ("replica", "does not support replicated A layouts"),
        ("scope", "requires local A fragments"),
        ("alias_a", "D must not alias A"),
        ("alias_b", "D must not alias B"),
        ("overlap", "rejects shifted overlapping C/D regions"),
    ],
)
def test_maca_gemm_mma_rejects_invalid_contract(case, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        _lower(_build_contract_case(case))


@pytest.mark.parametrize(
    "dtypes",
    [
        ("float16", "float16", "float16", "float16"),
        ("bfloat16", "float16", "float32", "float32"),
        ("uint8", "uint8", "int32", "int32"),
        ("int8", "int8", "float32", "float32"),
    ],
)
def test_maca_gemm_mma_rejects_unsupported_signature(dtypes):
    with pytest.raises(RuntimeError, match="does not support dtype signature"):
        _lower(_build_dtypes(*dtypes))


def test_maca_gemm_mma_accepts_disjoint_c_d_regions():
    script = _lower(_build_contract_case("disjoint"))["main"].script()
    assert "T.maca.mma_m16n16k16_f16_f32(" in script


if __name__ == "__main__":
    tvm.testing.main()
