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

permute_layout
==============

``permute_layout`` rearranges a warp's data from a source ``TileLayout`` to a
destination one — typically an in-place transpose. CUDA and MACA register
separate implementations because their warp widths and memory instructions
differ. Both stage every lane's values in registers and use a barrier between
the load and store phases, so source and destination may alias.

The CUDA variant (``warp_xor_swizzle``) uses Warp32 and an optional PTX shared
memory path. Source:
``python/tvm/backend/cuda/tile_primitive/permute_layout/warp_xor_swizzle.py``.

MACA variants
-------------

MACA registers ``wave64_xor`` (priority 40), ``wave64_direct`` (30),
``wave64_shared_two_batch`` (25), and ``wave64_generic`` (10). All require
warp scope, a one-dimensional full Wave64 lane range
``[0, 64)``, equal static slice extents, plain ``TileLayout`` objects, and
bijective sliced layouts. Element widths of 1, 2, 4, and 8 bytes are supported.

For volume ``V``, the active lane count is
``min(64, next_power_of_two(V))`` and each lane gets
``ceil(V / active_lanes)`` register slots. Every memory access is guarded by
``lane_id < active_lanes`` and ``flat < V``, so non-power-of-two volumes are
valid. ``wave64_xor`` is only an optimization for 4-byte data, power-of-two
slots, and a certified common XOR schedule. If that schedule is unavailable,
dispatch falls through to direct/shared/generic lowering; bank conflicts never
make a valid operation fail.

Every variant performs complete register loads, an unconditional
``T.maca.warp_sync()``, complete stores, and a second unconditional barrier
across all 64 lanes. This ordering supports aliasing views. The implementation
uses ordinary typed buffer accesses and does not use PTX, inline assembly,
TCGEN05, TMA, tensor-memory, or unverified BSM permutation intrinsics.

What it accepts
---------------

The implementation first builds a common permutation plan:

.. code-block:: python

    if sctx.scope_kind != "warp":                  return "scope is not 'warp'"
    if "threadIdx.y" in launch or "threadIdx.z" in launch: return "multi-dim threadIdx"
    if src_buf.dtype != dst_buf.dtype:             return "dtype mismatch"
    if src_ext_i != dst_ext_i:                     return "extent mismatch"
    if dtype_bytes not in (1, 2, 4, 8):             return "unsupported element width"
    if not isinstance(src_buf.layout, TileLayout): return "src not a plain TileLayout"
    if not isinstance(dst_buf.layout, TileLayout): return "dst not a plain TileLayout"
    # + layouts must slice, regroup, and define bijections on the slice
    # + volume is nonzero and both layouts are bijections
    # Variant-specific XOR checks only select an optimization.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope / priority
     - ``maca``; **warp** scope only; priorities ``40/30/25/10``
   * - operands
     - equal dtype, equal (compile-time) extents; both plain ``TileLayout`` (no
       swizzle wrapper); dtype byte width ∈ {1, 2, 4, 8}; ordinary typed
       buffer loads and stores
   * - launch / volume
     - one-dimensional full Wave64; any nonzero static slice volume
   * - layout mapping
     - after slicing and regrouping, source and destination describe the same
       iteration extents and each is a bijection on the slice
   * - bank-freedom
     - XOR bank-freedom is required only by ``wave64_xor``; lower-priority
       variants remain eligible when no schedule exists

Demonstration program
----------------------

A warp transposes the inner ``4×32`` block of a scale-factor tile — source layout
strides ``(…, 32, 1)``, destination ``(…, 1, 4)`` — for two pipeline stages (the
canonical SF-transpose, from ``test_permute_layout.py``):

.. code-block:: python

    pipe, blk, dtype = 2, 128, "float32"; high = 1
    shape = (pipe, high, 4, 32)
    pre  = TileLayout(S[shape : (blk, 128, 32, 1)])   # source
    post = TileLayout(S[shape : (blk, 128, 1, 4)])    # destination (4↔32 transposed)

    @Tx.prim_func
    def f(A: Tx.handle, B: Tx.handle):
        A_buf = Tx.match_buffer(A, shape, dtype, layout=pre)
        B_buf = Tx.match_buffer(B, shape, dtype, layout=post)
        Tx.device_entry(); Tx.cta_id([1]); Tx.warp_id([1]); Tx.lane_id([64])
        for s in Tx.serial(0, pipe):
            Tx.tile.warp.permute_layout(B_buf[s, 0:1, 0:4, 0:32], A_buf[s, 0:1, 0:4, 0:32])

Algorithm
---------

**1. Align the two layouts.** Both layouts are sliced to the region and
canonicalized; if their shards differ in structure (a linear layout collapses to 1-D
under canon, a transposed one keeps its multi-dim shape) the source is regrouped to
the destination's shape. From the destination shard come the iteration ``extent``
and the per-side strides ``src_str`` / ``dst_str``. The plan computes
``active_lanes`` and ``slots = ceil(volume / active_lanes)``.

**2. Choose a variant.** ``wave64_xor`` simulates both 32-lane service
batches and chooses the smallest conflict-free XOR schedule. If none exists,
the direct/shared/generic variants remain eligible.

**3. Emit two local-staged phases.** Each lane reads its ``slots`` elements through
the source layout into a local temporary (the swizzle permutes which slot holds
which iteration), a ``warp_sync`` follows, then the elements are written back through the
destination layout:

.. code-block:: python

    regs = Tx.alloc_buffer((slots,), dtype, scope="local")
    for r in Tx.unroll(0, slots):
        flat = lane_id + r * active_lanes
        if lane_id < active_lanes and flat < volume:
            regs[r] = src_buf[project(decompose(flat, extent), src_st)]
    Tx.maca.warp_sync()
    for r in Tx.unroll(0, slots):
        flat = lane_id + r * active_lanes
        if lane_id < active_lanes and flat < volume:
            dst_buf[project(decompose(flat, extent), dst_st)] = regs[r]
    Tx.maca.warp_sync()

CUDA reference TIRx IR
----------------------

.. code-block:: python

    regs[r] = A_buf[s*128 + (r ^ ((tx >> 3) & 3)) % 4 * 32 + tx]   # phase 1 (src order)
    Tx.cuda.warp_sync()
    B_buf[s*128 + tx * 4 + (r ^ ((tx >> 3) & 3)) % 4] = regs[r]    # phase 2 (dst order)
    Tx.cuda.warp_sync()

CUDA reference source
---------------------

.. code-block:: c++

    alignas(64) float regs_ptr[4];
    regs_ptr[0] = A_buf_ptr[(s*128) + (((0 ^ ((threadIdx.x >> 3) & 3)) & 3) * 32) + threadIdx.x];
    regs_ptr[1] = A_buf_ptr[(s*128) + (((1 ^ ((threadIdx.x >> 3) & 3)) & 3) * 32) + threadIdx.x];
    regs_ptr[2] = A_buf_ptr[(s*128) + (((2 ^ ((threadIdx.x >> 3) & 3)) & 3) * 32) + threadIdx.x];
    regs_ptr[3] = A_buf_ptr[(s*128) + (((3 ^ ((threadIdx.x >> 3) & 3)) & 3) * 32) + threadIdx.x];
    __syncwarp();
    // ... 4 transposed writes into B_buf_ptr, then __syncwarp();

Each lane owns column ``threadIdx.x`` and stages its 4 rows through ``regs``; the
``(threadIdx.x >> 3)`` XOR rotates the register order per lane-group of 8 so the
write phase hits distinct banks and transposes the ``4×32`` block for every
pipeline stage.

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - layout strides (the permutation)
     - define ``extent`` / ``src_str`` / ``dst_str`` and hence slot indexing and the
       per-element index math (the transpose pattern)
   * - dtype byte width
     - feeds the bank simulation in ``_choose_xor_k`` for the optional 4-byte
       XOR path; unsupported XOR simply falls through to a generic variant
   * - chosen ``k``
     - sets ``shift`` / ``mask`` of the XOR swizzle (``k = 0`` ⇒ no swizzle)
   * - active lanes / slots
     - active lanes are ``min(64, next_power_of_two(volume))`` and slots are
       ``ceil(volume / active_lanes)``; non-power-of-two slots are valid
