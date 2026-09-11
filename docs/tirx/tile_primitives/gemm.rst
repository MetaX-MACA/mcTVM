..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

gemm
====

``gemm`` computes ``D = alpha·A@B + beta·C`` as a fully-unrolled nest of
warp-collective ``mma.sync.aligned.m16n8k{16,8}`` instructions. A warp call is
the usual form. A full warpgroup or CTA scope is also accepted; each contained
warp executes the fragment program, and the operand thread-axis tiling determines
whether those warps own distinct output tiles or repeat the same tile. A and B
fragments and the C/D accumulators **all live in registers** — the caller stages
A and B into register fragments first (typically via :doc:`copy/ldstmatrix`).
The dispatch tiles M/N/K into ``m16n8k`` atoms and emits one ``mma`` per output
tile, accumulating over K in place. Source:
``python/tvm/backend/cuda/tile_primitive/gemm/mma_m16n8k_.py``. (For the
Blackwell async tensor-core path see :doc:`gemm_async`.)

MACA C500 / xcore1000
---------------------

MACA registers synchronous Wave64 variants for C500/xcore1000. All variants
require non-replicated ``local`` register fragments, full Wave64 participation,
constant transpose flags, ``alpha == 1.0``, and ``beta`` equal to 0 or 1. The
verified signature matrix is:

.. list-table::
   :header-rows: 1
   :widths: 26 20 24 30

   * - A / B / C / D
     - Atom
     - Intrinsic
     - Arithmetic and layout
   * - ``float16 / float16 / float32 / float32``
     - ``m16n16k16``
     - ``mma_16x16x16f16``
     - f32 accumulation; four f32 result slots per lane
   * - ``bfloat16 / bfloat16 / float32 / float32``
     - ``m16n16k16``
     - ``mma_16x16x16bf16``
     - BF16 payload bits are preserved; f32 accumulation
   * - ``float16 / float16 / float16 / float16``
     - ``m16n16k16``
     - ``mma_16x16x16f16``
     - C is promoted to f32; native result is rounded to f16 after every K atom
   * - ``float32 / float32 / float32 / float32``
     - ``m16n16k4``
     - ``mma_16x16x4f32``
     - full-f32 arithmetic; A/B have one element per lane
   * - ``float64 / float64 / float64 / float64``
     - ``m16n16k4``
     - ``mma_16x16x4f64``
     - double-precision arithmetic; one A/B and four C/D elements per lane
   * - ``int8 / int8 / int32 / int32``
     - ``m16n16k16``
     - ``mma_16x16x16i8``
     - packed byte payloads and int32 accumulation
   * - ``uint8 / uint8 / int32 / int32``
     - ``m16n16k16``
     - ``mma_16x16x16i8``
     - four signed MMA calls combine low-seven-bit and high-bit products
   * - ``int4 / int4 / int32 / int32``
     - ``m8n8k32``
     - software dot product
     - cross-lane gathering, low-nibble-first unpacking and sign extension
   * - ``uint4 / uint4 / int32 / int32``
     - ``m8n8k32``
     - software dot product
     - cross-lane gathering and unsigned nibble expansion
   * - ``int1 / int1 / int32 / int32``
     - ``m8n8k128``
     - ``bmma`` equivalent
     - AND/popcount products; XOR is a different operation and is not selected

Every row supports all four transpose orientations, multiple M/N/K atoms,
aligned nonzero regions, and exact in-place C=D. Regions must be positive,
in bounds, and aligned to the atom dimensions in their logical orientation.
Beta=0 never reads C; beta=1 initializes each output from C once before the
K loop. Integer accumulation wraps modulo 2**32, including the software
corrections, which use unsigned C++ arithmetic to avoid signed-overflow UB.
The unsigned SDK overload calls the signed i8 builtin without correction;
the tile helper explicitly corrects this for values above 127.

For one atom, lane ``l`` and local element ``p`` own these logical coordinates:

.. list-table::
   :header-rows: 1

   * - Family
     - A (row, K)
     - B (K, column)
     - C/D (row, column)
   * - 16x16x16
     - ``(l % 16, 4*(l // 16) + p)``, p=0..3
     - ``(4*(l // 16) + p, l % 16)``, p=0..3
     - ``(4*(l // 16) + p, l % 16)``, p=0..3
   * - 16x16x4, f32
     - ``(l % 16, l // 16)``
     - ``(l // 16, l % 16)``
     - ``(4*(l // 16) + p, l % 16)``, p=0..3
   * - 16x16x4, f64
     - ``(l % 16, l // 16)``
     - ``(l // 16, l % 16)``
     - ``(l // 16 + 4*p, l % 16)``, p=0..3
   * - 8x8x32, i4/u4
     - ``(l % 8, 4*(l // 8) + p)``, p=0..3
     - ``(4*(l // 8) + p, l % 8)``, p=0..3
     - ``(l // 8, l % 8)``
   * - 8x8x128, int1
     - ``(l % 8, 16*(l // 8) + p)``, p=0..15
     - ``(16*(l // 8) + p, l % 8)``, p=0..15
     - ``(l // 8, l % 8)``

Per-lane storage orders atoms as A[M-tile, K-tile, p],
B[K-tile, N-tile, p], and C/D[M-tile, N-tile, p]. Transposition permutes
the logical axes while preserving that physical order. These mappings use
all 64 lanes without replication. The paired A/B K ordering differs from
the SDK's reversed K groups but preserves the same dot products.

Packed inputs require explicit packed storage: each lane's four nibbles or
sixteen bits occupy one uint16, with increasing K in increasing bit position.
Stage through a uint16 buffer view, as in the numerical tests; scalar int4
and int1 stores do not pack individual logical elements. GEMM uses byte offsets
so adjacent two-byte atoms remain distinct. Signed nibbles encode -8..7 in
two's complement. ``int1`` is interpreted as the SDK's bit payload: set bits
contribute 1 to an AND/popcount dot product (also the product of two signed
one-bit values -1). It does not implement XOR/popcount or bipolar -1/+1 GEMM.
The software helpers gather the distributed row/column chunks before computing
each output; they do not assume each lane already owns a complete SDK fragment.

The implementation rejects mixed signedness, incompatible accumulator types,
non-matching layouts, narrowed execution scopes, unsupported target architectures,
and shifted overlapping C/D views before code generation. D may not alias A/B.
TF32 has a compiling, executable xcore1000 builtin, but the current MACA WMMA
codegen has no mapping from a TIR dtype to ``precision::tf32``; float32 maps to
``float`` and uses full-f32 arithmetic. A reduced-precision mode is not selected
implicitly. The installed MXCC rejects FP8 MMA
(``__builtin_mxc_mma_f32_16x16x16f8_e4m3`` is undeclared), so none of TVM's
e4m3fn/e4m3fnuz/e5m2 encodings are advertised as MMA inputs. The SDK defines no
mixed-signedness overload. Async GEMM, tensor memory, and tcgen05 remain outside
this synchronous implementation.

This matrix was validated on C500 with MACA 3.0.0.0, driver 3.6.11 and
MXCC 1.0.0 (f794f08733), targeting xcore1000. The general SDK type table
(MXMACA C++ guide CN_V03, sections 1.24.2–1.24.4, pp. 224–225) alone does
not establish a working compiler signature or its arithmetic semantics.

What it accepts
---------------

.. code-block:: python

    # register_dispatch("gemm", "cuda", priority=10, when=[
    predicate("full_active_lanes", _full_active_lanes),   # complete warp(s), un-narrowed
    predicate("no_replica", _no_replica),                 # no broadcast axes on D/A/B/C
    # ])
    # in the impl:
    for buf, name in ((D, "D"), (A, "A"), (B, "B"), (C, "C")):
        if buf.scope() != "local":
            fail(f"gemm mma requires {name} in register (local) scope, got {buf.scope()}")

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope / priority
     - ``cuda``; priority ``10``. The predicate has no scope-kind allowlist: it
       requires every axis present in ``sctx.intra`` to be a complete,
       zero-offset ``laneid`` / ``wid_in_wg`` / ``warpid`` axis. This admits the
       normal warp / warpgroup / CTA call sites and rejects unrecognized axes
       such as a cluster axis. ``mma.sync`` remains warp-collective, so callers
       use a warp or a wider scope made of complete warps
   * - operand scope
     - **A, B, C, D all in registers** (``local``); a shared operand makes the
       dispatch ``fail`` (stage with ldmatrix first)
   * - no replica
     - none of D/A/B/C may carry a broadcast/replica axis (``_no_replica``)
   * - shape
     - ``M % 16 == 0``, ``N % 8 == 0``, and ``K % 8 == 0``. The dispatcher
       first tries ``m16n8k16`` and then ``m16n8k8``; the selected instruction
       must tile all operand layouts exactly
   * - dtype
     - A and B are both ``float16`` or both ``bfloat16``; C and D are
       ``float32``
   * - alpha / beta
     - ``alpha == 1.0``; ``beta ∈ {0.0, 1.0}`` (0 → ``D = A@B``; 1 → ``D = A@B + C``)

Demonstration program
----------------------

A single warp computes ``D[16,8] = A[16,16] @ B[16,8]`` in ``float16`` (f32
accumulate) — one ``m16n8k16`` atom (from ``test_gemm_mma_m16n8k_.py``):

.. code-block:: python

    from tvm.tirx.layout import S, TileLayout, laneid

    D_FRAG    = TileLayout(S[(2, 8, 4, 2) : (2, 4 @ laneid, 1 @ laneid, 1)])
    A_FRAG_K8 = TileLayout(S[(2, 8, 4, 2) : (2, 4 @ laneid, 1 @ laneid, 1)])
    B_FRAG_K8 = TileLayout(S[(4, 2, 8) : (1 @ laneid, 1, 4 @ laneid)])
    A_FRAG = A_FRAG_K8.tile_to([16, 16], [16, 8]); B_FRAG = B_FRAG_K8.tile_to([16, 8], [8, 8])

    @Tx.prim_func
    def gemm(A_ptr: Tx.handle, B_ptr: Tx.handle, D_ptr: Tx.handle):
        A_g = Tx.match_buffer(A_ptr, (16, 16), "float16"); B_g = Tx.match_buffer(B_ptr, (16, 8), "float16")
        D_g = Tx.match_buffer(D_ptr, (16, 8), "float32")
        Tx.device_entry(); Tx.cta_id([1]); Tx.warp_id([1]); lane = Tx.lane_id([32])
        A_f = Tx.alloc_buffer((16, 16), "float16", scope="local", layout=A_FRAG)
        B_f = Tx.alloc_buffer((16, 8),  "float16", scope="local", layout=B_FRAG)
        D_f = Tx.alloc_buffer((16, 8),  "float32", scope="local", layout=D_FRAG)
        A_reg = A_f.local(8)                              # stage A into the lane's 8 regs
        for s in Tx.unroll(8):
            kp, rM, kHi = s % 2, (s // 2) % 2, s // 4
            A_reg[s] = A_g[lane // 4 + 8 * rM, 2 * (lane % 4) + kp + 8 * kHi]
        B_reg = B_f.local(4)                              # stage B into the lane's 4 regs
        for s in Tx.unroll(4):
            kp, kHi = s % 2, s // 2
            B_reg[s] = B_g[2 * (lane % 4) + kp + 8 * kHi, lane // 4]
        Tx.tile.warp.gemm(D_f, A_f, B_f, D_f, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0)
        D_reg = D_f.local(4)                              # write the 4 result regs out
        for s in Tx.unroll(4):
            rN, rM = s % 2, s // 2
            D_g[lane // 4 + 8 * rM, 2 * (lane % 4) + rN] = D_reg[s]

Algorithm
---------

**1. Tile and fragment-group.** The dispatch slices each operand's layout to its
region and, for each candidate instruction (``m16n8k16`` then ``m16n8k8``), tries to
group the operand sub-layouts (``D_M, D_N, A_M, A_K, B_K, B_N, C_*``) into the fixed
m16n8k frame, anchoring A/C on D's M, B/C on D's N, and B on A's K. The first
instruction that fits, with matching warp-tiling, wins.

**2. Derive register layouts.** Each operand gets a mediated per-lane view:
D/C as ``[Mo, No, rM, rN]`` (4 f32), A as
``[Mo, Ko, rM, kHi, k_pack]``, and B as
``[Ko, No, kHi, k_pack]``.  The corresponding raw physical register order
exposed by the default ``local`` view is D/C ``[Mo, No, rM, rN]``, A
``[Mo, Ko, kHi, rM, k_pack]``, and B ``[Ko, No, kHi, k_pack]`` — the PTX
operand enumeration that ``mma.sync`` expects.

**3. Emit the unrolled nest** — initialize D (from C if ``beta==1``, else 0), then
accumulate over K in place, one ``mma`` per (m, n) tile:

.. code-block:: python

    for m in Tx.unroll(M_tiles):
        for n in Tx.unroll(N_tiles):
            for rM, rN in ...: d_local[m, n, rM, rN] = c_local[...] if use_c else Tx.float32(0)
            for k in Tx.unroll(K_tiles):
                d_regs = [d_local[m, n, rM, rN] for rM in range(2) for rN in range(2)]  # 4 f32
                a_regs = [a_words[m, k, rM, kHi, 0] for kHi in range(n_kHi) for rM in range(2)]
                b_regs = [b_words[k, n, kHi, 0] for kHi in range(n_kHi)]
                mma_chain = (f"mma.sync.aligned.{shape_str}.row.col"
                             f".f32.{a_elem}.{b_elem}.f32")
                Tx.ptx[mma_chain](*d_regs, *a_regs, *b_regs, *d_regs)   # d = a·b + d

Generated TIRx IR
-----------------

The single 16×8×16 tile lowers to one ``mma`` (4 D regs, 4 A regs, 2 B regs):

.. code-block:: python

    Tx.ptx["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"](
        d_local[0], d_local[1], d_local[2], d_local[3], a_local[0], ...)

Generated CUDA
--------------

.. code-block:: c++

    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"

The accumulator ``{%0..%3}`` is both the C input and the D output (in-place
accumulate); ``{%4..%7}`` are A's four ``b32`` registers, ``{%8, %9}`` B's two.
Verified on ``sm_100a`` (``D == A@B`` within fp16 tolerance).

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - input dtype
     - ``float16`` → ``…f32.f16.f16.f32``; ``bfloat16`` → ``…f32.bf16.bf16.f32``
       (register counts unchanged — 2 elems per ``b32``)
   * - K instruction
     - ``k16`` → A 4 ``b32`` / B 2 ``b32``; ``k8`` → A 2 / B 1
       (``mma.…m16n8k8.…``)
   * - M / N / K extents
     - set the ``M_tiles`` / ``N_tiles`` / ``K_tiles`` unrolled loop counts (one
       ``mma`` per (m, n), K accumulated in place)
   * - beta
     - ``0`` → D zero-initialized; ``1`` → D initialized from C (the ``mma`` itself
       is identical)
   * - transpose_A / transpose_B
     - transpose the logical A or B region before shape and layout matching;
       the transformed operands must still fit the selected ``m16n8k`` frame
   * - operand scope
     - A/B **must** be register fragments; a shared operand makes the dispatch
       ``fail`` (stage via :doc:`copy/ldstmatrix` first)
