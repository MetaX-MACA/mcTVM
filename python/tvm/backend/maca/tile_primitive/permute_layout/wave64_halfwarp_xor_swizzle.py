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

"""MACA permute_layout dispatch: Wave64 register-staged in-place transpose.

The optional per-lane XOR swizzle avoids SMEM bank conflicts on the write phase.

The dispatcher reasons about the **layout's shard**, not the buffer's
declared shape (the two can differ — a buffer with ``shape=(PIPE, M, K)``
may carry a layout whose shard has more dims internally, with grouping
mapping shard segments onto buffer dims).  Concretely:

    src_sliced = src.layout.slice(src.shape, region).canonicalize()
    dst_sliced = dst.layout.slice(dst.shape, region).canonicalize()
    # If the two sliced shards have different structures (which is common —
    # a linear layout collapses to 1D under canon while a transposed one
    # keeps its multi-dim structure), regroup src to dst's shape.
    if src_sliced.shard != dst_sliced.shard:
        src_sliced, _ = src_sliced.group(dst.shard.extents)
    extent  = [int(it.extent) for it in dst_sliced.shard]   # iteration shape
    src_str = [int(it.stride) for it in src_sliced.shard]
    dst_str = [int(it.stride) for it in dst_sliced.shard]

The algorithm:

    regs[P]
    for r in 0..P:
        j  = r XOR ((lane >> SHIFT) & MASK)
        i  = lane + j * 32                             # flat logical index
        idx = decompose(i, extent)                     # iter multi-dim index
        regs[r] = src[project(idx, src.shape, slice_starts)]
    warp_sync()
    for r in 0..P:
        j  = r XOR ((lane >> SHIFT) & MASK)
        i  = lane + j * 32
        idx = decompose(i, extent)
        dst[project(idx, dst.shape, slice_starts)] = regs[r]
    warp_sync()

where ``project`` mixed-radix-folds the iter shard dims back onto the
buffer's iterated slice dims (so the emit's index matches buf.shape rank,
which TIR's BufferLoad/Store requires).

SHIFT and MASK are chosen by simulating the bank pattern at the **shard
granularity** (where strides are affine), trying k = 0, 1, …, log2(P)
and picking the smallest k that makes both phases bank-conflict-free.

Correctness rests on:

* For each lane, ``r ↦ r XOR const`` is a bijection on ``[0, P)``.
* Therefore (lane, r) ↔ flat over [0, V).
* Both layouts are verified bijections on the slice (every logical
  position has a unique byte offset under that layout).
* The mixed-radix projection from iter shard idx to buf coord is exactly
  what TIR's BufferLoad does internally when buf.shape rank < shard rank
  — so iter shard's strides and the buffer-indexed byte offset agree.
"""

from __future__ import annotations

import math

from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.ir import TensorRegion
from tvm.tirx import IntImm, PrimFunc, is_buffer_var
from tvm.tirx.layout import TileLayout, _flatten_coord
from tvm.tirx.operator.tile_primitive import DispatchContext, fail, register_dispatch
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..common import get_indices, get_st_extent, maca_mcpu_is

WAVE_SIZE = 64
ACTIVE_LANES = 32
SMEM_BANKS = 32
SMEM_BANK_BYTES = 4

# ---------- helpers ----------------------------------------------------------


def _as_buffer_and_region(arg):
    """Normalize a Buffer or TensorRegion to (buffer, start_list, extent_list)."""
    if is_buffer_var(arg):
        buf = arg
        extent = list(buf.ty.shape)
        st = [0] * len(extent)
    elif isinstance(arg, TensorRegion):
        buf = arg.source
        st, extent = get_st_extent(arg)
    else:
        raise TypeError(f"unexpected permute_layout arg type: {type(arg)}")
    return buf, list(st), list(extent)


def _as_int(x):
    """Return int(x) if x is int-like, else None."""
    if isinstance(x, int):
        return x
    if isinstance(x, IntImm):
        return int(x.value)
    if hasattr(x, "value") and isinstance(x.value, int):
        return int(x.value)
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _layout_shard_int(layout):
    """Return (extents, strides) as int lists from a TileLayout's shard, or (None, None)."""
    if not isinstance(layout, TileLayout):
        return None, None
    extents, strides = [], []
    for it in layout.shard:
        e = _as_int(it.extent)
        s = _as_int(it.stride)
        if e is None or s is None:
            return None, None
        extents.append(e)
        strides.append(s)
    return extents, strides


def _decompose_row_major(i, extent):
    out, rem = [], i
    for e in reversed(extent):
        out.append(rem % e)
        rem //= e
    return list(reversed(out))


def _eval_offset(idx, strides):
    return sum(i * s for i, s in zip(idx, strides))


def _check_bijection(extent, strides):
    """Iteration extents + strides define a bijection on [0, V)?"""
    V = math.prod(extent)
    seen = set()
    for i in range(V):
        off = _eval_offset(_decompose_row_major(i, extent), strides)
        if off in seen:
            return False
        seen.add(off)
    return len(seen) == V


def _bank_free(extent, strides, dtype_bytes, P, k):
    """For every register slot r ∈ [0, P), do the 32 lanes hit 32 distinct banks?"""
    T, BANKS, BANK_W = ACTIVE_LANES, SMEM_BANKS, SMEM_BANK_BYTES
    shift = int(math.log2(ACTIVE_LANES)) - k
    mask = (1 << k) - 1
    for r in range(P):
        seen = set()
        for lane in range(T):
            j = r ^ ((lane >> shift) & mask)
            flat = lane + j * T
            idx = _decompose_row_major(flat, extent)
            off_bytes = _eval_offset(idx, strides) * dtype_bytes
            bank = (off_bytes // BANK_W) % BANKS
            if bank in seen:
                return False
            seen.add(bank)
    return True


def _choose_xor_k(extent, src_strides, dst_strides, dtype_bytes, P):
    max_k = int(math.log2(P)) if P > 0 else 0
    for k in range(max_k + 1):
        if _bank_free(extent, src_strides, dtype_bytes, P, k) and _bank_free(
            extent, dst_strides, dtype_bytes, P, k
        ):
            return k
    return None


# ---------- validator + dispatch impl ---------------------------------------


def _gather(op_call):
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_arg, src_arg = op_call.args[0], op_call.args[1]
    src_buf, src_st, src_ext = _as_buffer_and_region(src_arg)
    dst_buf, dst_st, dst_ext = _as_buffer_and_region(dst_arg)
    return src_buf, src_st, src_ext, dst_buf, dst_st, dst_ext


def _why_reject(op_call, sctx):
    ok, reason = maca_mcpu_is(op_call, sctx, ("xcore1000",))
    if not ok:
        return reason
    if not sctx.is_warp:
        return f"scope {sctx.scope_kind!r} is not 'warp'"
    active_range = sctx.intra.get("laneid")
    if active_range is None:
        return "warp permute_layout is missing laneid active range"
    if len(active_range) not in (2, 3):
        return f"invalid laneid active range {active_range}"
    try:
        lane_range = tuple(int(x) for x in active_range)
    except (TypeError, ValueError):
        return f"non-static laneid active range {active_range}"
    if lane_range not in ((WAVE_SIZE, 0), (WAVE_SIZE, 0, 1)):
        return f"Wave64 permute_layout requires contiguous laneid [0, 64), got {active_range}"
    if "threadIdx.y" in sctx.launch_params or "threadIdx.z" in sctx.launch_params:
        return "multi-dim threadIdx is not supported"

    src_buf, src_st, src_ext, dst_buf, dst_st, dst_ext = _gather(op_call)

    if src_buf.dtype != dst_buf.dtype:
        return f"dtype mismatch: dst={dst_buf.dtype} vs src={src_buf.dtype}"

    src_ext_i = [_as_int(e) for e in src_ext]
    dst_ext_i = [_as_int(e) for e in dst_ext]
    if None in src_ext_i or None in dst_ext_i:
        return "extents must be compile-time integers"
    if src_ext_i != dst_ext_i:
        return f"slice shape mismatch: src={src_ext_i} vs dst={dst_ext_i}"

    dtype_bits = DataType(src_buf.dtype).bits
    if dtype_bits != 32:
        return "permute_layout requires 32-bit elements"
    dtype_bytes = 4

    if not isinstance(src_buf.layout, TileLayout):
        return "src buffer's layout is not a plain TileLayout"
    if not isinstance(dst_buf.layout, TileLayout):
        return "dst buffer's layout is not a plain TileLayout"

    # Slice + canonicalize both layouts.  The result's shard describes the
    # iteration domain; runtime starts (like ``ks``) are folded into the
    # layout's offset, separate from the shard's affine part.
    src_region = [(s, s + e) for s, e in zip(src_st, src_ext)]
    dst_region = [(s, s + e) for s, e in zip(dst_st, dst_ext)]
    src_sliced = src_buf.layout.slice(list(src_buf.shape), src_region)
    dst_sliced = dst_buf.layout.slice(list(dst_buf.shape), dst_region)
    if src_sliced is None or dst_sliced is None:
        return "layout.slice failed"
    src_sliced = src_sliced.canonicalize()
    dst_sliced = dst_sliced.canonicalize()

    # Iteration shape: regroup dst onto the iterated buf dims; the result's
    # shard may stay finer than iter_buf_extents (one buf dim ↔ several shard
    # dims via seps), which is fine.  Then regroup src to match dst's shard
    # extents exactly so both phases share the same iteration index space.
    iter_buf_extents = [e for e in src_ext_i if e != 1]
    try:
        dst_grouped, dst_seps = dst_sliced.group(iter_buf_extents)
        src_grouped, _ = src_sliced.group([int(it.extent) for it in dst_grouped.shard])
    except Exception as e:
        return f"layout.group failed: {e}"

    dst_ext_, dst_str_ = _layout_shard_int(dst_grouped)
    src_ext_, src_str_ = _layout_shard_int(src_grouped)
    if dst_ext_ is None or src_ext_ is None:
        return "regrouped layout shard contains non-integer extent/stride"
    if src_ext_ != dst_ext_:
        return f"src shard {src_ext_} doesn't match dst shard {dst_ext_} after regrouping"

    extent = dst_ext_
    V = math.prod(extent)
    if V == 0 or V % ACTIVE_LANES != 0:
        return f"volume {V} not divisible by active lane count {ACTIVE_LANES}"
    P = V // ACTIVE_LANES
    if P == 0 or (P & (P - 1)) != 0 or P > ACTIVE_LANES:
        return f"per-thread count {P} must be power of 2 in [1, {ACTIVE_LANES}]"
    if not _check_bijection(extent, src_str_):
        return "src layout (regrouped) is not a bijection on the slice"
    if not _check_bijection(extent, dst_str_):
        return "dst layout is not a bijection on the slice"
    if _choose_xor_k(extent, src_str_, dst_str_, dtype_bytes, P) is None:
        return "no conflict-free XOR schedule for the active 32-lane cohort"
    return None


def _impl(op_call, sctx):
    src_buf, src_st, src_ext, dst_buf, dst_st, dst_ext = _gather(op_call)
    src_ext_i = [_as_int(e) for e in src_ext]

    # Slice away dimensions outside the call region.  Canonicalization leaves
    # only the affine shard that describes the logical permutation.
    src_region = [(s, s + e) for s, e in zip(src_st, src_ext)]
    dst_region = [(s, s + e) for s, e in zip(dst_st, dst_ext)]
    src_sliced = src_buf.layout.slice(list(src_buf.shape), src_region).canonicalize()
    dst_sliced = dst_buf.layout.slice(list(dst_buf.shape), dst_region).canonicalize()

    # Use the destination shard as the common logical iteration space.  The
    # source may canonicalize to a different rank, so regroup it to the exact
    # destination shard extents before deriving either access pattern.
    iter_buf_extents = [e for e in src_ext_i if e != 1]
    dst_grouped, dst_seps = dst_sliced.group(iter_buf_extents)
    src_grouped, _ = src_sliced.group([int(it.extent) for it in dst_grouped.shard])

    extent, dst_str_ = _layout_shard_int(dst_grouped)
    _, src_str_ = _layout_shard_int(src_grouped)
    V = math.prod(extent)
    P = V // ACTIVE_LANES
    dtype_bytes = 4

    k_opt = _choose_xor_k(extent, src_str_, dst_str_, dtype_bytes, P)
    if k_opt is None:
        fail(f"no XOR-bits k ∈ [0, log2(P)={int(math.log2(P))}] makes both phases bank-free")

    shift = int(math.log2(ACTIVE_LANES)) - k_opt
    mask = (1 << k_opt) - 1

    # dst_seps records which consecutive shard dimensions belong to each
    # non-unit buffer dimension.  Fold each group back to one buffer index so
    # emitted BufferLoad/BufferStore nodes match the original buffer rank.
    iter_buf_dims = [i for i, e in enumerate(src_ext_i) if e != 1]
    seps = list(dst_seps)

    def _project(iter_idx, st_list):
        buf_idx = list(st_list)
        for bi in range(len(seps) - 1):
            lo, hi = seps[bi], seps[bi + 1]
            flat = _flatten_coord(iter_idx[lo:hi], extent[lo:hi])
            buf_idx[iter_buf_dims[bi]] = st_list[iter_buf_dims[bi]] + flat
        return tuple(buf_idx)

    dtype = src_buf.dtype

    # fmt: off
    # The dispatcher returns a private helper whose buffer and loop variables
    # are captured from the call site; TilePrimitiveDispatch inlines it before
    # final well-formedness verification.
    @T.prim_func(check_well_formed=False)
    def impl():
        # Re-declare the deferred lane scope in the private helper.  The
        # dispatch pass inlines this function into the caller and resolves
        # the deferred extent to the caller's Wave64 lane binding.
        lane_id = T.lane_id()
        regs = T.alloc_buffer((P,), dtype, scope="local")
        # Only the first consecutive 32-lane cohort issues memory operations.
        # XOR changes register-slot order per lane without changing the set of
        # logical elements covered by the cohort.
        if lane_id < ACTIVE_LANES:
            for r in T.unroll(0, P):
                j = T.meta_var(r ^ ((lane_id >> shift) & mask))
                flat = T.meta_var(lane_id + j * ACTIVE_LANES)
                iter_idx = T.meta_var(get_indices(flat, [0] * len(extent), extent))
                src_idx = T.meta_var(_project(iter_idx, src_st))
                regs[r] = src_buf[tuple(src_idx)]
        # Keep the barrier outside the lane guard: all Wave64 lanes must
        # participate, and all aliased source loads must finish before stores.
        T.maca.warp_sync()
        if lane_id < ACTIVE_LANES:
            for r in T.unroll(0, P):
                j = T.meta_var(r ^ ((lane_id >> shift) & mask))
                flat = T.meta_var(lane_id + j * ACTIVE_LANES)
                iter_idx = T.meta_var(get_indices(flat, [0] * len(extent), extent))
                dst_idx = T.meta_var(_project(iter_idx, dst_st))
                dst_buf[tuple(dst_idx)] = regs[r]
        # Complete the store phase before the caller can reuse aliased storage.
        T.maca.warp_sync()
    # fmt: on
    return impl


@register_dispatch(
    "permute_layout",
    "maca",
    variant="wave64_halfwarp_xor_swizzle",
    priority=20,
)
def permute_layout_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    reason = _why_reject(op, sctx)
    if reason is not None:
        fail(reason)
    return _impl(op, sctx)


__all__ = [
    "_bank_free",
    "_check_bijection",
    "_choose_xor_k",
    "_decompose_row_major",
    "_eval_offset",
    "permute_layout_dispatch",
]
