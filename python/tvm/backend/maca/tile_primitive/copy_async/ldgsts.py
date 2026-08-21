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

"""MACA BSM implementation of ``copy_async`` global-to-shared copies.

The BSM load instructions accept only 4, 8, and 16-byte transactions.  They
are asynchronous with respect to shared-memory consumers, so this dispatch
only issues transfers.  The caller must explicitly order a later shared-memory
read with ``T.maca.async_wait_gvmcnt()``, ``T.maca.barrier_inst()``, and a CTA
barrier when appropriate.
"""

from numbers import Integral

from tvm.arith import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.expr import IntImm as _IntImm
from tvm.tirx.layout import ComposeLayout, TileLayout
from tvm.tirx.operator.tile_primitive.dispatcher import fail, predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall

from ..copy._common import _TID_AXIS_FOR_SCOPE, _thread_cnt, align_layouts_gs
from ..copy.utils import _is_valid_copy, _scope_allowed
from ..copy.vec_auto_reg import _all_threads_active, _axis_decl, _ptr_off
from ..layout_utils import recompose_swizzle

# BSM is unidirectional: global memory feeds shared memory.
_BSM_PAIRS = [("global", "shared*")]
# The hardware instruction spelling fixes the transaction width.
_BSM_VEC_BITS = (128, 64, 32)
_BSM_CONFIG_KEYS = {"vec_len", "predicate", "fill_mode"}


def _as_static_int(value, analyzer: Analyzer) -> int | None:
    """Return a compile-time integer, or ``None`` for dynamic expressions."""
    try:
        value = analyzer.simplify(value)
    except Exception:  # pragma: no cover - FFI objects can reject simplification
        pass
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _static_complete_layout(
    buffer_region, name: str, analyzer: Analyzer
) -> tuple[bool, str | None]:
    """Require a static, one-to-one layout for the complete async slice.

    BSM has no scalar or partial-copy fallback.  Requiring a static slice here
    keeps the synthesized partition and its vector-alignment proof meaningful,
    and rejecting replicas prevents one global transaction from being treated
    as several independent shared writes.
    """
    buffer: Buffer = buffer_region.buffer
    layout = buffer.layout
    if layout is None:
        return False, f"{name} has no layout"

    shape = []
    region = []
    volume = 1
    for axis, (shape_value, region_value) in enumerate(
        zip(buffer.shape, buffer_region.region, strict=True)
    ):
        shape_int = _as_static_int(shape_value, analyzer)
        start_int = _as_static_int(region_value.min, analyzer)
        extent_int = _as_static_int(region_value.extent, analyzer)
        if shape_int is None or start_int is None or extent_int is None:
            return False, f"{name} axis {axis} must have static shape and region"
        if shape_int <= 0 or start_int < 0 or extent_int <= 0 or start_int + extent_int > shape_int:
            return False, f"{name} axis {axis} has invalid static region"
        shape.append(shape_int)
        region.append((start_int, start_int + extent_int))
        volume *= extent_int

    try:
        sliced = layout.slice(shape, region)
    except Exception as err:  # pragma: no cover - exact FFI exception varies
        return False, f"{name} layout.slice failed: {err}"
    if sliced is None:
        return False, f"{name} layout.slice failed"
    if not sliced.verify_well_formed():
        return False, f"{name} sliced layout is not well formed"

    tile_layout = sliced.tile_layout if isinstance(sliced, ComposeLayout) else sliced
    if not isinstance(tile_layout, TileLayout):
        return False, f"{name} sliced layout is not a TileLayout"
    if len(tile_layout.replica) != 0:
        return False, f"{name} sliced layout has replicas"
    for iterator in tile_layout.shard:
        extent = _as_static_int(iterator.extent, analyzer)
        stride = _as_static_int(iterator.stride, analyzer)
        if extent is None or stride is None or extent <= 0:
            return False, f"{name} sliced layout has a non-static shard iterator"
    for offset in tile_layout.offset.values():
        if _as_static_int(offset, analyzer) is None:
            return False, f"{name} sliced layout has a non-static offset"

    size = _as_static_int(sliced.size(), analyzer)
    span = _as_static_int(sliced.span(), analyzer)
    if size != volume or span is None or span <= 0:
        return False, f"{name} sliced layout is not a complete static mapping"
    return True, None


def _static_complete_layouts(op_call: TilePrimitiveCall) -> tuple[bool, str | None]:
    op_call = TilePrimitiveCall.downcast(op_call)
    analyzer = Analyzer()
    for buffer_region, name in ((op_call.src, "src"), (op_call.dst, "dst")):
        ok, reason = _static_complete_layout(buffer_region, name, analyzer)
        if not ok:
            return False, reason
    return True, None


def _bsm_config(op_call: TilePrimitiveCall) -> tuple[bool, str | None, tuple[int, ...] | None]:
    """Validate the deliberately small BSM configuration surface.

    CUDA's ``cp.async`` accepts PTX cache/prefetch controls and can model a
    partial source size.  MACA BSM exposes neither.  Its predicator only has
    a whole-transaction copy-or-zero behavior, represented by pairing
    ``predicate`` with ``fill_mode=\"zero\"``.
    """
    op_call = TilePrimitiveCall.downcast(op_call)
    config = op_call.config
    unsupported = sorted(str(key) for key in config if str(key) not in _BSM_CONFIG_KEYS)
    if unsupported:
        return False, "BSM copy_async does not support config: " + ", ".join(unsupported), None

    has_predicate = "predicate" in config
    has_fill_mode = "fill_mode" in config
    if has_predicate != has_fill_mode:
        return (
            False,
            "BSM copy_async requires predicate together with fill_mode='zero'",
            None,
        )
    if has_fill_mode and str(config.get("fill_mode")) != "zero":
        return False, "BSM copy_async only supports fill_mode='zero'", None

    elem_bits = DataType(op_call.src.buffer.dtype).bits
    if "vec_len" not in config:
        return True, None, _BSM_VEC_BITS
    vec_len = _as_static_int(config.get("vec_len"), Analyzer())
    if vec_len is None or vec_len <= 0:
        return False, "BSM copy_async vec_len must be a positive static integer", None
    vec_bits = vec_len * elem_bits
    if vec_bits not in _BSM_VEC_BITS:
        return False, "BSM copy_async requires a 4/8/16-byte vec_len", None
    return True, None, (vec_bits,)


def _divides_thread_cnt(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    """Every active thread must receive a whole number of BSM transactions."""
    op_call = TilePrimitiveCall.downcast(op_call)
    thread_cnt = _thread_cnt(sctx)
    if thread_cnt <= 0:
        return False, f"degenerate thread_cnt={thread_cnt} (scope has empty intra)"
    n_elements = 1
    for region_value in op_call.src.region:
        try:
            n_elements *= int(region_value.extent)
        except (TypeError, ValueError):
            return False, f"non-constant region extent {region_value.extent}"
    if n_elements % thread_cnt:
        return False, f"region size {n_elements} not divisible by thread_cnt={thread_cnt}"
    return True, None


def _has_bsm_vector(op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    """Prove that the static layouts admit one native BSM vector width."""
    op_call = TilePrimitiveCall.downcast(op_call)
    ok, reason, candidates = _bsm_config(op_call)
    if not ok:
        return False, reason
    assert candidates is not None

    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    src_region = [
        (region_value.min, region_value.min + region_value.extent)
        for region_value in op_call.src.region
    ]
    dst_region = [
        (region_value.min, region_value.min + region_value.extent)
        for region_value in op_call.dst.region
    ]
    try:
        with sctx.target:
            _g_p, _s_p, vec_len = align_layouts_gs(
                src.layout,
                src.shape,
                src_region,
                dst.layout,
                dst.shape,
                dst_region,
                DataType(src.dtype).bits,
                _thread_cnt(sctx),
                vec_bits_candidates=candidates,
            )
    except Exception as err:  # pragma: no cover - exact FFI exception varies
        return False, f"BSM layout alignment failed: {err}"
    if vec_len * DataType(src.dtype).bits not in _BSM_VEC_BITS:
        return False, "no aligned 4/8/16-byte BSM vector width"
    return True, None


def _is_ldgsts(op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    if not sctx.is_target("maca"):
        return False, "non-maca target"
    if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"):
        return False, f"unsupported exec_scope {sctx.scope_kind}"
    for check in (
        lambda: _all_threads_active(sctx),
        lambda: _is_valid_copy(op_call, sctx),
        lambda: _scope_allowed(op_call, sctx, allowed_pairs=_BSM_PAIRS),
        lambda: _static_complete_layouts(op_call),
        lambda: _divides_thread_cnt(op_call, sctx),
        lambda: _has_bsm_vector(op_call, sctx),
    ):
        ok, reason = check()
        if not ok:
            return False, reason
    return True, None


def _emit_ldgsts(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    # The dispatcher predicate guarantees global -> shared only.
    g_buf, g_br = src, op_call.src
    s_buf, s_br = dst, op_call.dst
    ok, reason, candidates = _bsm_config(op_call)
    if not ok:
        fail(reason)
    assert candidates is not None

    g_region = [
        (region_value.min, region_value.min + region_value.extent) for region_value in g_br.region
    ]
    s_region = [
        (region_value.min, region_value.min + region_value.extent) for region_value in s_br.region
    ]
    elem_bits = DataType(src.dtype).bits
    thread_cnt = _thread_cnt(sctx)
    with sctx.target:
        g_partition, s_partition, vec_len = align_layouts_gs(
            g_buf.layout,
            g_buf.shape,
            g_region,
            s_buf.layout,
            s_buf.shape,
            s_region,
            elem_bits,
            thread_cnt,
            vec_bits_candidates=candidates,
        )
        s_apply_layout = recompose_swizzle(s_buf.layout, s_partition)

    vec_bits = vec_len * elem_bits
    if vec_bits not in _BSM_VEC_BITS:
        fail("BSM copy_async cannot find an aligned 4/8/16-byte vector width")
    copy_op = getattr(T.maca, f"copy_async_{vec_bits}b")
    if "predicate" in op_call.config:
        copy_op = getattr(T.maca, f"copy_async_{vec_bits}b_zfill")
    predicate_expr = op_call.config.get("predicate")

    n_elements = 1
    for iterator in s_partition.shard:
        n_elements *= int(iterator.extent)
    if n_elements % (thread_cnt * vec_len):
        fail(
            f"BSM partition has {n_elements} elements, not divisible by "
            f"thread_cnt({thread_cnt}) * vec_len({vec_len})"
        )
    total_outer = n_elements // (thread_cnt * vec_len)
    apply_shape = [
        _IntImm("int32", total_outer),
        _IntImm("int32", thread_cnt),
        _IntImm("int32", vec_len),
    ]
    s_zero = [0] * len(s_buf.shape)
    g_zero = [0] * len(g_buf.shape)
    tid_axis_name = _TID_AXIS_FOR_SCOPE[sctx.scope_kind] if thread_cnt > 1 else None

    def _decl_tid():
        if tid_axis_name is not None:
            return _axis_decl(tid_axis_name, sctx)
        return _IntImm("int32", 0)

    v0 = _IntImm("int32", 0)

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        tid = _decl_tid()
        for outer in range(total_outer):
            s_offset = s_apply_layout.apply(outer, tid, v0, shape=apply_shape)["m"]
            g_offset = g_partition.apply(outer, tid, v0, shape=apply_shape)["m"]
            s_ptr: T.let = _ptr_off(s_buf.ptr_to(s_zero), s_offset)
            g_ptr: T.let = _ptr_off(g_buf.ptr_to(g_zero), g_offset)
            if predicate_expr is None:
                copy_op(s_ptr, g_ptr)
            else:
                copy_op(s_ptr, g_ptr, predicate_expr)
    # fmt: on
    return impl


@register_dispatch(
    "copy_async",
    "maca",
    variant="ldgsts",
    priority=20,
    when=[predicate("ldgsts_applicable", _is_ldgsts)],
)
def copy_schedule_ldgsts(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return _emit_ldgsts(op_call, sctx)
