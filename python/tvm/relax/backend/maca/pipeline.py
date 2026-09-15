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
"""The Relax MACA backend compilation pipeline and other passes."""

import tvm

from .. import gpu_generic


def library_dispatch_passes(target: tvm.target.Target):
    """Return MACA library dispatch passes requested by ``target.libs``."""
    passes = gpu_generic.library_dispatch_passes(target)
    libs = target.attrs.get("libs", [])
    if "mcdnn" in libs:
        from .mcdnn import partition_for_mcdnn  # pylint: disable=import-outside-toplevel

        @tvm.transform.module_pass(opt_level=0)
        def _partition_for_mcdnn(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
            return partition_for_mcdnn(mod)

        passes.append(_partition_for_mcdnn)

    if "mcblas" not in libs:
        return passes

    from .mcblas import partition_for_mcblas  # pylint: disable=import-outside-toplevel

    @tvm.transform.module_pass(opt_level=0)
    def _partition_for_mcblas(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
        # This must precede LegalizeOps, while high-level relax.matmul is still available.
        return partition_for_mcblas(mod, bind_constants=False)

    return [*passes, _partition_for_mcblas, tvm.relax.transform.RunCodegen()]


def legalize_passes(target: tvm.target.Target):
    """Return the standard GPU legalization passes for MACA."""
    return gpu_generic.legalize_passes(target)


def dataflow_lower_passes(target: tvm.target.Target):
    """Return the standard GPU dataflow lowering passes for MACA."""
    return gpu_generic.dataflow_lower_passes(target)


def finalize_passes(target: tvm.target.Target):
    """Return the standard GPU finalization passes for MACA."""
    return gpu_generic.finalize_passes(target)


def get_default_pipeline(target: tvm.target.Target):
    """Return the default MACA pipeline with opt-in MCBlas partitioning."""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
        with target:
            seq = tvm.transform.Sequential(
                library_dispatch_passes(target)
                + legalize_passes(target)
                + dataflow_lower_passes(target)
                + finalize_passes(target)
            )
            return seq(mod)

    return _pipeline
