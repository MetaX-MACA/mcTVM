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

"""MACA shared-memory elementwise dispatch using scalar arithmetic."""

from tvm.script import tirx as T
from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.operator.tile_primitive.dispatcher import fail

from ..common import get_indices, get_st_extent, get_thread_cnt
from ..copy.vec_auto_reg import _all_threads_active, _axis_decl
from ._common import (
    _TID_AXIS_FOR_SCOPE,
    _tensor_shape_of,
    buffer_regions,
    compute_dtype_of,
    emit_scope_sync,
    fetch_src_value,
    n_elements,
    shape_broadcast_compat,
)


def is_smem_ewise(spec):
    """Accept shared-memory elementwise operations at thread/Wave64/warpgroup/CTA scope."""

    def check(op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
        if not sctx.is_target("maca"):
            return False, "non-maca target"
        if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"):
            return False, f"unsupported scope {sctx.scope_kind}"
        ok, reason = _all_threads_active(sctx)
        if not ok:
            return False, reason
        if sctx.scope_kind == "warpgroup":
            tx_iv = sctx.launch_params.get("threadIdx.x")
            if tx_iv is None:
                return False, "warpgroup shared elementwise missing threadIdx.x launch_params"
            try:
                cta_threads = int(tx_iv.dom.extent)
            except (TypeError, ValueError):
                return False, f"non-static threadIdx.x extent: {tx_iv.dom.extent}"
            # MACA exposes Wave64 and CTA barriers, but no subgroup barrier
            # spanning four waves. Restrict this path to one warpgroup per CTA
            # so the CTA barrier emitted below is exactly the collective scope.
            if cta_threads != 256:
                return False, (
                    "warpgroup shared elementwise requires one 256-thread CTA on MACA "
                    f"(got {cta_threads})"
                )
        plan, msg = spec.parse(op_call)
        if msg is not None or plan is None:
            return False, msg
        for br in buffer_regions(plan):
            if not br.buffer.scope().startswith("shared"):
                return False, f"operand scope {br.buffer.scope()} != shared*"
            if br.buffer.layout is None:
                return False, "shared operand has no layout"
        if spec.check_extras is not None:
            ok, reason = spec.check_extras(plan.extras, compute_dtype_of(plan))
            if not ok:
                return False, reason
        dst_shape = _tensor_shape_of(plan.dst.region)
        for src in plan.srcs:
            if src.buf_region is None:
                continue
            ok, reason = shape_broadcast_compat(_tensor_shape_of(src.buf_region.region), dst_shape)
            if not ok:
                return False, f"shape incompat: {reason}"
        return True, None

    return check


def _tid_expr(sctx: DispatchContext):
    if sctx.scope_kind == "thread":
        return 0
    return _axis_decl(_TID_AXIS_FOR_SCOPE[sctx.scope_kind], sctx)


def emit_smem(op_call: TilePrimitiveCall, spec, sctx: DispatchContext) -> PrimFunc:
    """Partition a shared tile across active threads and emit scalar ops."""

    plan, msg = spec.parse(op_call)
    if msg is not None or plan is None:
        fail(msg or "parse failed")
    thread_cnt = get_thread_cnt(sctx)
    if thread_cnt is None or int(thread_cnt) <= 0:
        fail(f"unsupported scope {sctx.scope_kind} for shared elementwise")
    thread_cnt = int(thread_cnt)
    if "threadIdx.y" in sctx.launch_params or "threadIdx.z" in sctx.launch_params:
        fail("shared elementwise currently requires 1D threadIdx")

    total = n_elements(plan.dst)
    dst_buf = plan.dst.buffer
    dst_start, dst_extent = get_st_extent(plan.dst)
    dst_dtype = dst_buf.dtype
    n_outer = (total + thread_cnt - 1) // thread_cnt
    compute = spec.compute_scalar
    sync = emit_scope_sync(sctx.scope_kind)

    @T.prim_func(check_well_formed=False)
    def impl():
        tid = _tid_expr(sctx)
        for outer in T.serial(0, n_outer):
            fused = T.meta_var(outer * thread_cnt + tid)
            if fused < total:
                dst_indices = T.meta_var(get_indices(fused, dst_start, dst_extent))
                src_vals = T.meta_var(
                    [
                        fetch_src_value(src, fused, dst_indices, dst_start, dst_extent)
                        for src in plan.srcs
                    ]
                )
                dst_buf[tuple(dst_indices)] = T.cast(
                    compute(src_vals, plan.extras, dst_dtype), dst_dtype
                )
        sync()

    return impl
