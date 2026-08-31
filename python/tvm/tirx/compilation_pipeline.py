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
"""The TIR backend compilation pipeline."""

import tvm
from tvm import tirx
from tvm.backend.cuda import transforms as cuda_transforms


def _verify_maca_module():
    """Reject CUDA-only operations before lowering a MACA module."""

    @tvm.transform.module_pass(opt_level=0)
    def _verify(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        for gvar, func in mod.functions.items():
            if not isinstance(func, tirx.PrimFunc):
                continue

            def _visit(node):
                op = getattr(node, "op", None)
                if not isinstance(op, tvm.ir.Op):
                    return
                op_name = op.name
                if op_name.startswith(("tirx.ptx.", "tirx.ptx_legacy.")):
                    raise ValueError(
                        f"MACA function {gvar.name_hint} contains unsupported operation "
                        f"{op_name}; replace it with the corresponding T.maca operation"
                    )

            tirx.stmt_functor.post_order_visit(func.body, _visit)
        return mod

    return _verify


def _target_specific_passes(target):
    if target is None or tvm.target.Target(target).kind.name == "cuda":
        return [cuda_transforms.LowerIket()]
    if tvm.target.Target(target).kind.name == "maca":
        return []
    return []


def _target_specific_fragment_passes(target):
    if target is not None and tvm.target.Target(target).kind.name == "maca":
        return [tvm.s_tir.transform.InferFragment()]
    return []


def default_tir_pipeline(target=None):
    """The default tirx pipeline used in tvm.tirx.build"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """The default lowering passes for TIR backend."""
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            *([_verify_maca_module()] if target is not None and target.kind.name == "maca" else []),
            tirx.transform.LowerInitBlock(),
            tvm.s_tir.transform.UnifyThreadBinding(),
            tirx.transform.StmtSimplify(),
            tirx.transform.FlattenBuffer(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.NarrowDataType(32),
            tirx.transform.VectorizeLoop(not bool(config.get("tir.disable_vectorize", False))),
            tirx.transform.UnrollLoop(),
            tirx.transform.StmtSimplify(),
        ]
        if not bool(config.get("tir.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        passes.extend(
            [
                tirx.transform.FP8ComputeLegalize(),
                tirx.transform.VerifyMemory(),
                tirx.transform.AnnotateEntryFunc(),
                *_target_specific_fragment_passes(target),
                tirx.transform.SplitHostDevice(),
                *_target_specific_passes(target),
                tirx.transform.MakePackedAPI(),
                tirx.transform.FP8StorageLegalize(),
                tirx.transform.BF16StorageLegalize(),
            ]
        )
        mod = tvm.ir.transform.Sequential(passes)(mod)
        return mod

    return _pipeline, finalize_host_passes, finalize_device_passes


def tirx_pipeline(target=None):
    """The TIRX pipeline used in tvm.tirx.build"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """The default lowering passes for TIR backend."""
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            *([_verify_maca_module()] if target is not None and target.kind.name == "maca" else []),
            tirx.transform.LowerTIRx(),
            tvm.s_tir.transform.UnifyThreadBinding(),
            tirx.transform.StmtSimplify(),
            tirx.transform.LowerTIRxOpaque(),
            tirx.transform.FlattenBuffer(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.NarrowDataType(32),
            tirx.transform.VectorizeLoop(not bool(config.get("tir.disable_vectorize", False))),
            tirx.transform.UnrollLoop(),
            tirx.transform.StmtSimplify(),
        ]
        if not bool(config.get("tir.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        passes.extend(
            [
                tirx.transform.FP8ComputeLegalize(),
                tirx.transform.VerifyMemory(),
                tirx.transform.AnnotateEntryFunc(),
                *_target_specific_fragment_passes(target),
                tirx.transform.SplitHostDevice(),
                *_target_specific_passes(target),
                tirx.transform.MakePackedAPI(),
                tirx.transform.FP8StorageLegalize(),
                tirx.transform.BF16StorageLegalize(),
            ]
        )
        mod = tvm.ir.transform.Sequential(passes)(mod)
        return mod

    return _pipeline, finalize_host_passes, finalize_device_passes


def finalize_host_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    host_pass_list = [
        tirx.transform.LowerTVMBuiltin(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(host_pass_list)


def finalize_device_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    device_pass_list = [
        tirx.transform.LowerWarpMemory(),
        tirx.transform.StmtSimplify(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(device_pass_list)


def finalize_device_passes_tirx():  # pylint: disable=unused-argument
    """The TIRx finalization passes for TIR backend."""
    device_pass_list = [tirx.transform.LowerIntrin()]
    return tvm.ir.transform.Sequential(device_pass_list)


# global map of pre-built pipelines
PIPELINE_MAP = {"default": default_tir_pipeline, "tirx": tirx_pipeline}


def register_tir_pipeline(name: str, pipeline_factory) -> None:
    """Register a named TIR pipeline factory."""

    PIPELINE_MAP[name] = pipeline_factory


def get_tir_pipeline(name: str | None = None, **kwargs) -> tvm.transform.Pass:
    """Get pre-build pipeline by name

    Parameters
    ----------
    name : Optional[str]
        Name of the pipeline
    """
    if name == "default":
        # for now, default to s_tir pipeline
        name = "s_tir"
    if name not in PIPELINE_MAP:
        raise ValueError(
            f"Unknown pre-built pipeline {name},candidates are {list(PIPELINE_MAP.keys())}"
        )
    return PIPELINE_MAP[name](**kwargs)


def get_default_tir_pipeline(
    target: tvm.target.Target,  # pylint: disable=unused-argument
) -> tvm.transform.Pass:
    """Get the default TIR pipeline for the given target."""
    if target.kind.name == "opencl" and "adreno" in target.keys:
        return get_tir_pipeline("adreno")
    else:
        return get_tir_pipeline("s_tir")
