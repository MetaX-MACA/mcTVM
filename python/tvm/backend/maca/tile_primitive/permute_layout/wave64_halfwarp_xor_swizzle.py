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
    """Normalize an operand to ``(buffer, starts, extents)``.

    A bare buffer denotes its complete shape.  A ``TensorRegion`` carries the
    selected slice explicitly; keeping both forms in the same representation
    lets the layout analysis use one indexing path for full-buffer and sliced
    calls.
    """
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
    """Extract a Python integer from a static TIR expression.

    Dynamic expressions deliberately return ``None`` so dispatch can reject
    them before attempting compile-time layout or volume analysis.
    """
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
    """Read a plain ``TileLayout`` shard as integer extents and strides.

    The affine shard is the iteration-space representation used by the bank
    model and bijection check.  Non-``TileLayout`` wrappers and dynamic fields
    are reported as ``(None, None)`` for a reasoned dispatcher rejection.
    """
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
    """Convert a flattened row-major index into coordinates for ``extent``."""
    out, rem = [], i
    for e in reversed(extent):
        out.append(rem % e)
        rem //= e
    return list(reversed(out))


def _eval_offset(idx, strides):
    """Evaluate an affine layout offset in element units."""
    return sum(i * s for i, s in zip(idx, strides))


def _check_bijection(extent, strides):
    """Return whether ``strides`` map every logical point to a unique offset."""
    V = math.prod(extent)
    seen = set()
    for i in range(V):
        off = _eval_offset(_decompose_row_major(i, extent), strides)
        if off in seen:
            return False
        seen.add(off)
    return len(seen) == V


def _bank_free(extent, strides, dtype_bytes, P, k, active_lanes=ACTIVE_LANES):
    """Check C500 shared-memory bank uniqueness for one XOR schedule.

    C500 has 32 four-byte banks and services a Wave64 in two 32-lane batches.
    ``k`` changes the register-slot order; for multi-byte values every bank
    covered by the element is checked, rather than only its starting bank.
    Partial final slots are ignored because they do not issue a memory access.
    """
    if active_lanes not in (32, 64):
        return False
    shift = 5 - k
    mask = (1 << k) - 1
    volume = math.prod(extent)
    for r in range(P):
        for batch in range(active_lanes // 32):
            seen = set()
            for lane32 in range(32):
                lane = batch * 32 + lane32
                j = r ^ ((lane32 >> shift) & mask)
                flat = lane + j * active_lanes
                if flat >= volume:
                    continue
                idx = _decompose_row_major(flat, extent)
                off_bytes = _eval_offset(idx, strides) * dtype_bytes
                banks = range(off_bytes // SMEM_BANK_BYTES,
                              (off_bytes + dtype_bytes - 1) // SMEM_BANK_BYTES + 1)
                for bank in banks:
                    bank %= SMEM_BANKS
                    if bank in seen:
                        return False
                    seen.add(bank)
    return True


def _choose_xor_k(extent, src_strides, dst_strides, dtype_bytes, P, active_lanes=ACTIVE_LANES):
    """Pick the smallest XOR bit count valid for both load and store layouts."""
    max_k = int(math.log2(P)) if P > 0 else 0
    for k in range(max_k + 1):
        if _bank_free(extent, src_strides, dtype_bytes, P, k, active_lanes) and _bank_free(
            extent, dst_strides, dtype_bytes, P, k, active_lanes
        ):
            return k
    return None


# ---------- validator + dispatch impl ---------------------------------------


def _gather(op_call):
    """Extract normalized source and destination operands from a tile call."""
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_arg, src_arg = op_call.args[0], op_call.args[1]
    src_buf, src_st, src_ext = _as_buffer_and_region(src_arg)
    dst_buf, dst_st, dst_ext = _as_buffer_and_region(dst_arg)
    return src_buf, src_st, src_ext, dst_buf, dst_st, dst_ext


def analyze_common(op_call, sctx):
    """Build the variant-independent permutation plan.

    This validates target/scope, static matching regions, supported element
    widths, plain layouts, and layout bijections.  Optimization-specific
    decisions such as XOR availability are intentionally left to dispatch
    variants so a valid operation can fall back to the generic path.
    """
    ok, reason = maca_mcpu_is(op_call, sctx, ("xcore1000",))
    if not ok:
        return None, reason
    if not sctx.is_warp:
        return None, f"scope {sctx.scope_kind!r} is not 'warp'"
    active_range = sctx.intra.get("laneid")
    if active_range is None:
        return None, "warp permute_layout is missing laneid active range"
    if len(active_range) not in (2, 3):
        return None, f"invalid laneid active range {active_range}"
    try:
        lane_range = tuple(int(x) for x in active_range)
    except (TypeError, ValueError):
        return None, f"non-static laneid active range {active_range}"
    if lane_range not in ((WAVE_SIZE, 0), (WAVE_SIZE, 0, 1)):
        return None, f"Wave64 permute_layout requires contiguous laneid [0, 64), got {active_range}"
    if "threadIdx.y" in sctx.launch_params or "threadIdx.z" in sctx.launch_params:
        return None, "multi-dim threadIdx is not supported"

    src_buf, src_st, src_ext, dst_buf, dst_st, dst_ext = _gather(op_call)

    if src_buf.dtype != dst_buf.dtype:
        return None, f"dtype mismatch: dst={dst_buf.dtype} vs src={src_buf.dtype}"

    src_ext_i = [_as_int(e) for e in src_ext]
    dst_ext_i = [_as_int(e) for e in dst_ext]
    if None in src_ext_i or None in dst_ext_i:
        return None, "extents must be compile-time integers"
    if src_ext_i != dst_ext_i:
        return None, f"slice shape mismatch: src={src_ext_i} vs dst={dst_ext_i}"

    dtype = DataType(src_buf.dtype)
    if dtype.bits % 8 != 0 or dtype.bits // 8 not in (1, 2, 4, 8):
        return None, "permute_layout requires 1, 2, 4, or 8-byte elements"
    dtype_bytes = dtype.bits // 8

    if not isinstance(src_buf.layout, TileLayout):
        return None, "src buffer's layout is not a plain TileLayout"
    if not isinstance(dst_buf.layout, TileLayout):
        return None, "dst buffer's layout is not a plain TileLayout"

    # Slice + canonicalize both layouts.  The result's shard describes the
    # iteration domain; runtime starts (like ``ks``) are folded into the
    # layout's offset, separate from the shard's affine part.
    src_region = [(s, s + e) for s, e in zip(src_st, src_ext)]
    dst_region = [(s, s + e) for s, e in zip(dst_st, dst_ext)]
    src_sliced = src_buf.layout.slice(list(src_buf.shape), src_region)
    dst_sliced = dst_buf.layout.slice(list(dst_buf.shape), dst_region)
    if src_sliced is None or dst_sliced is None:
        return None, "layout.slice failed"
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
        return None, f"layout.group failed: {e}"

    dst_ext_, dst_str_ = _layout_shard_int(dst_grouped)
    src_ext_, src_str_ = _layout_shard_int(src_grouped)
    if dst_ext_ is None or src_ext_ is None:
        return None, "regrouped layout shard contains non-integer extent/stride"
    if src_ext_ != dst_ext_:
        return None, f"src shard {src_ext_} doesn't match dst shard {dst_ext_} after regrouping"

    extent = dst_ext_
    V = math.prod(extent)
    if V == 0:
        return None, "permute_layout rejects an empty slice"
    if not _check_bijection(extent, src_str_):
        return None, "src layout (regrouped) is not a bijection on the slice"
    if not _check_bijection(extent, dst_str_):
        return None, "dst layout is not a bijection on the slice"

    active_lanes = min(WAVE_SIZE, 1 << (V - 1).bit_length())
    slots = (V + active_lanes - 1) // active_lanes
    plan = dict(src_buf=src_buf, dst_buf=dst_buf, src_st=src_st, dst_st=dst_st,
                src_ext=src_ext_i, dst_ext=dst_ext_i, extent=extent, volume=V,
                dtype=src_buf.dtype, dtype_bytes=dtype_bytes, active_lanes=active_lanes,
                slots=slots, src_strides=src_str_, dst_strides=dst_str_,
                dst_seps=list(dst_seps))
    return plan, None


def _emit(plan, xor_k=None):
    src_buf, dst_buf = plan["src_buf"], plan["dst_buf"]
    src_st, dst_st = plan["src_st"], plan["dst_st"]
    src_ext_i, extent = plan["src_ext"], plan["extent"]
    active_lanes, slots = plan["active_lanes"], plan["slots"]
    src_str_, dst_str_ = plan["src_strides"], plan["dst_strides"]

    # Slice away dimensions outside the call region.  Canonicalization leaves
    # only the affine shard that describes the logical permutation.
    shift = 5 - xor_k if xor_k is not None else 0
    mask = (1 << xor_k) - 1 if xor_k is not None else 0

    # dst_seps records which consecutive shard dimensions belong to each
    # non-unit buffer dimension.  Fold each group back to one buffer index so
    # emitted BufferLoad/BufferStore nodes match the original buffer rank.
    iter_buf_dims = [i for i, e in enumerate(src_ext_i) if e != 1]
    seps = plan["dst_seps"]

    def _project(iter_idx, st_list):
        buf_idx = list(st_list)
        for bi in range(len(seps) - 1):
            lo, hi = seps[bi], seps[bi + 1]
            flat = _flatten_coord(iter_idx[lo:hi], extent[lo:hi])
            buf_idx[iter_buf_dims[bi]] = st_list[iter_buf_dims[bi]] + flat
        return tuple(buf_idx)

    dtype = plan["dtype"]

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
        regs = T.alloc_buffer((slots,), dtype, scope="local")
        # Only the first consecutive 32-lane cohort issues memory operations.
        # XOR changes register-slot order per lane without changing the set of
        # logical elements covered by the cohort.
        if lane_id < active_lanes:
            for r in T.unroll(0, slots):
                j = T.meta_var(r ^ (((lane_id & 31) >> shift) & mask)) if xor_k is not None else r
                flat = T.meta_var(lane_id + j * active_lanes)
                if flat < plan["volume"]:
                    iter_idx = T.meta_var(get_indices(flat, [0] * len(extent), extent))
                    src_idx = T.meta_var(_project(iter_idx, src_st))
                    regs[r] = src_buf[tuple(src_idx)]
        # Keep the barrier outside the lane guard: all Wave64 lanes must
        # participate, and all aliased source loads must finish before stores.
        T.maca.warp_sync()
        if lane_id < active_lanes:
            for r in T.unroll(0, slots):
                j = T.meta_var(r ^ (((lane_id & 31) >> shift) & mask)) if xor_k is not None else r
                flat = T.meta_var(lane_id + j * active_lanes)
                if flat < plan["volume"]:
                    iter_idx = T.meta_var(get_indices(flat, [0] * len(extent), extent))
                    dst_idx = T.meta_var(_project(iter_idx, dst_st))
                    dst_buf[tuple(dst_idx)] = regs[r]
        # Complete the store phase before the caller can reuse aliased storage.
        T.maca.warp_sync()
    # fmt: on
    return impl


def _dispatch_plan(op, sctx):
    plan, reason = analyze_common(op, sctx)
    if reason is not None:
        fail(reason)
    return plan


@register_dispatch("permute_layout", "maca", variant="wave64_xor", priority=40)
def permute_layout_xor(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    plan = _dispatch_plan(op, sctx)
    if plan["dtype_bytes"] != 4 or plan["active_lanes"] not in (32, 64):
        fail("XOR variant requires 32-bit elements and 32/64 active lanes")
    if plan["slots"] & (plan["slots"] - 1):
        fail("XOR variant requires a power-of-two slot count")
    k = _choose_xor_k(plan["extent"], plan["src_strides"], plan["dst_strides"],
                      4, plan["slots"], plan["active_lanes"])
    if k is None:
        fail("no certified conflict-free XOR schedule")
    return _emit(plan, k)


@register_dispatch("permute_layout", "maca", variant="wave64_direct", priority=30)
def permute_layout_direct(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    plan = _dispatch_plan(op, sctx)
    if plan["src_buf"].scope().startswith("shared") or plan["dst_buf"].scope().startswith("shared"):
        fail("direct variant is for non-shared buffers")
    return _emit(plan)


@register_dispatch("permute_layout", "maca", variant="wave64_shared_two_batch", priority=25)
def permute_layout_shared(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    plan = _dispatch_plan(op, sctx)
    if not (plan["src_buf"].scope().startswith("shared") or plan["dst_buf"].scope().startswith("shared")):
        fail("shared-two-batch variant requires shared storage")
    return _emit(plan)


@register_dispatch("permute_layout", "maca", variant="wave64_generic", priority=10)
def permute_layout_generic(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return _emit(_dispatch_plan(op, sctx))


permute_layout_dispatch = permute_layout_xor


__all__ = [
    "_bank_free",
    "_check_bijection",
    "_choose_xor_k",
    "_decompose_row_major",
    "_eval_offset",
    "analyze_common",
    "permute_layout_dispatch",
    "permute_layout_xor",
    "permute_layout_direct",
    "permute_layout_shared",
    "permute_layout_generic",
]
