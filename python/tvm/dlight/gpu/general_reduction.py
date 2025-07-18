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
# pylint: disable=invalid-name
"""Reduction rule for operators including softmax, layer norm, RMS norm, etc"""
from typing import List, Union

from tvm import arith, tir
from tvm.target import Target

from ..analysis import normalize_prim_func
from ..base import try_inline_contiguous_spatial
from .base import GPUScheduleRule


class GeneralReduction(GPUScheduleRule):
    """General Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        if target.kind.name == "cuda" or target.kind.name == "maca":
            len_tx = 256
            unroll_depth = 256
        elif target.kind.name == "opencl":
            len_tx = 256
            unroll_depth = 64
        else:
            len_tx = 64
            unroll_depth = 64

        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        dom_kind = block_infos[0].dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))

        # Align the number of block iters of the last block.
        num_last_block_iter = len(block_infos[-1].dom_kind())
        if num_last_block_iter < len(dom_kind):
            # If the last block is a scalar value, there is nothing left to
            # tile/parallelise, and  `iters` is an empty tuple.
            # Add a unit thread loop so the final write happens inside a valid
            # GPU thread environment.
            if num_last_block_iter == 0:
                # Put every block (both the running reductions and the final
                # scalar write) inside a trivial GPU thread. The very first block
                # gets a `blockIdx.x` wrapper so that kernels still have a unique
                # block scope.
                for i, info in enumerate(block_infos):
                    loop_rv = sch.add_unit_loop(info.block_rv)
                    if i == 0:
                        sch.bind(loop_rv, "blockIdx.x")
                    else:
                        sch.bind(loop_rv, "threadIdx.x")

                return sch

            def f_layout_mapping(*iters):
                analyzer = arith.Analyzer()
                # Try to match the iters of last block to the iters of the first block.
                # For matched positions, use the iter from the input `iters`.
                # For unmatched positions, use a new iter which is constant 0.
                num_matched = 0
                target_layout_iters = []
                for block_iter in block_infos[0].iters:
                    if num_matched < len(iters) and analyzer.can_prove_equal(
                        block_iter.dom, block_infos[-1].iters[num_matched].dom
                    ):
                        target_layout_iters.append(iters[num_matched])
                        num_matched += 1
                    else:
                        target_layout_iters.append(tir.const(0, iters[0].dtype))

                # If all the iters of the last block can match, return the new layout.
                if num_matched == len(iters):
                    return target_layout_iters
                # Otherwise, fallback to appending zeros in the beginning.
                return [tir.const(0, iters[0].dtype)] * (
                    len(dom_kind) - num_last_block_iter
                ) + list(iters)

            index_map = tir.IndexMap.from_func(f_layout_mapping, ndim=num_last_block_iter)
            sch.transform_block_layout(block_infos[-1].block_rv, index_map)

        try:
            # TODO: fix num_leading_s = 0 case
            assert num_trailing_r > 0
            for block in block_infos[1:-1]:
                assert block.dom_kind() == dom_kind
            assert block_infos[-1].is_injective()
            assert len(block_infos[-1].dom_kind()) <= len(dom_kind)
        except AssertionError:
            return None

        if "R" not in block_infos[-1].dom_kind():
            # The final block is a spatial block.
            # It is possible that the loop order of the last block is not the same as
            # previous blocks.
            # Thus we reorder spatial loops to align with reduction loops for followup schedule.
            # We first collect all the buffers written by reduction blocks,
            # then in the final block, any index of those buffers are spatial.
            reduced_buffers = []
            for block_info in block_infos[:-1]:
                for buffer_write in sch.get(block_info.block_rv).writes:
                    reduced_buffers.append(buffer_write.buffer)

            spatial_block = sch.get(block_infos[-1].block_rv)
            spatial_loops = set()
            block_var_to_loop_var = {}
            loops = sch.get_loops(block_infos[-1].block_rv)
            for block_iter, loop_rv in zip(spatial_block.iter_vars, loops):
                block_var_to_loop_var[block_iter.var] = sch.get(loop_rv).loop_var

            def _visit_expr(e: tir.PrimExpr):
                if isinstance(e, tir.Var) and e in block_var_to_loop_var:
                    spatial_loops.add(block_var_to_loop_var[e])

            for buffer_read in spatial_block.reads:
                buffer = buffer_read.buffer
                if buffer in reduced_buffers:
                    for read_range in buffer_read.region:
                        tir.stmt_functor.post_order_visit(read_range.min, _visit_expr)
                        tir.stmt_functor.post_order_visit(read_range.extent, _visit_expr)

            s_loops = []
            other_loops = []
            for loop_rv in loops:
                loop = sch.get(loop_rv)
                if loop.loop_var in spatial_loops or loop.extent == 1:
                    s_loops.append(loop_rv)
                else:
                    other_loops.append(loop_rv)
            sch.reorder(*s_loops, *other_loops)

        loops = sch.get_loops(block_infos[-1].block_rv)
        bx = sch.fuse(*loops[:num_leading_s])
        r_loop, tx = sch.split(loops[-1], [None, len_tx])
        sch.reorder(tx, r_loop)
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.annotate(r_loop, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)

        for block in reversed(block_infos[:-1]):
            block = block.block_rv
            for i, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, buffer_index=i, storage_scope="shared")
            sch.compute_at(block, bx, preserve_unit_loops=True)
            r_loop = sch.fuse(*sch.get_loops(block)[-num_trailing_r:])
            r_loop, tx = sch.split(r_loop, [None, len_tx])
            sch.reorder(tx, r_loop)
            sch.bind(tx, "threadIdx.x")
            sch.annotate(r_loop, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
            sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)

        # TODO: It's just a workaround to avoid unroll spatial loops, because of the bug of
        # the pass lower-thread-allreduce. We should fix it in the future.
        # sch.annotate(bx, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        # sch.annotate(bx, ann_key="pragma_unroll_explicit", ann_val=1)
        return sch
