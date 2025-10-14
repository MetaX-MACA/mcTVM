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
"""Definition of MACA/GPU operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from tvm import topi
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.contrib import mxcc, nvcc
# from tvm.contrib.thrust import can_use_thrust
from tvm.meta_schedule import is_meta_schedule_enabled
from tvm.te import SpecializedCondition
from .cuda import batch_matmul_strategy_cuda, conv2d_strategy_cuda, dense_strategy_cuda, judge_winograd
from ....target import Target
from ....tir import IntImm
from .. import op as _op
from .generic import *

@matmul_strategy.register(["maca"])
def matmul_strategy_maca(attrs, inputs, out_type, target):
    """Matmul maca strategy."""
    strategy = _op.OpStrategy()

    if is_auto_scheduler_enabled():
        strategy.add_implementation(
            wrap_compute_matmul(topi.nn.matmul), naive_schedule, name="matmul.maca"
        )
    elif is_meta_schedule_enabled():
        strategy.add_implementation(
            wrap_compute_matmul(topi.nn.matmul), naive_schedule, name="matmul.maca"
        )
    else:
        logger.warning(
            "Matmul is not optimized for maca. Recommend to use mcblas for better performance."
        )
        # Temporary use this as a basic schedule
        strategy.add_implementation(
            wrap_compute_matmul(topi.gpu.matmul_default),
            wrap_topi_schedule(topi.gpu.schedule_matmul_default),
            name="matmul_default.gpu",
        )

    if target.kind.name == "maca" and "mcblas" in target.libs:
        strategy.add_implementation(
            wrap_compute_matmul(topi.maca.matmul_mcblas),
            wrap_topi_schedule(topi.maca.schedule_matmul_mcblas),
            name="matmul_mcblas.maca",
            plevel=25,
        )
    return strategy

@dense_strategy.register(["maca"])
def dense_strategy_maca(attrs, inputs, out_type, target):
    """dense maca strategy"""
    strategy = _op.OpStrategy()
    data, weights = inputs
    b, i = get_const_tuple(data.shape)
    o, _ = get_const_tuple(weights.shape)
    if (
        target.kind.name in ["maca"]
        and data.dtype == "int8"
        and weights.dtype == "int8"
        and out_type.dtype == "int32"
    ):
        strategy.add_implementation(
            wrap_compute_dense(topi.cuda.dense_int8),
            wrap_topi_schedule(topi.cuda.schedule_dense_int8),
            name="dense_int8.maca",
        )

    if target.kind.name == "maca" and "mcblas" in target.libs:
        strategy.add_implementation(
            wrap_compute_dense(topi.maca.dense_mcblas),
            wrap_topi_schedule(topi.maca.schedule_dense_mcblas),
            name="dense_mcblas.maca",
            plevel=25,
        )
    return strategy

@batch_matmul_strategy.register(["maca"])
def batch_matmul_strategy_maca(attrs, inputs, out_type, target):
    """batch_matmul maca strategy"""
    strategy = _op.OpStrategy()
    x, y = inputs

    if target.kind.name == "maca" and "mcblas" in target.libs:
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.maca.batch_matmul_mcblas, need_out_dtype=True),
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="batch_matmul_mcblas.maca",
            plevel=30,
        )

    return strategy

@conv2d_strategy.register(["maca"])
def conv2d_strategy_maca(attrs, inputs, out_type, target):
    """conv2d maca strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    stride_h, stride_w = attrs.get_int_tuple("strides")
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")
    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            if (
                (target.kind.name in ["cuda", "vulkan", "rocm"])
                and data.dtype in ("int8", "uint8")
                and kernel.dtype in ("int8", "uint8")
            ):
                assert data.dtype == kernel.dtype
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_int8),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_int8),
                    name="conv2d_nchw_int8.cuda",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw),
                    name="conv2d_nchw.cuda",
                )
            N, _, H, W = get_const_tuple(data.shape)
            CO, CI, KH, KW = get_const_tuple(kernel.shape)
            (_, _, judge_winograd_auto_scheduler) = judge_winograd(
                N,
                H,
                W,
                KH,
                KW,
                CI,
                CO,
                padding,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                data.dtype,
                kernel.dtype,
                pre_flag=False,
            )
            if is_meta_schedule_enabled() and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nchw),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nchw_winograd.cuda",
                    plevel=15,
                )
            elif (
                (2 < KH < 8 and 2 < KW < 8 and KH == KW)
                and (stride_h == 1 and stride_w == 1)
                and (dilation_h == 1 and dilation_w == 1)
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_winograd),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_winograd),
                    name="conv2d_nchw_winograd.cuda",
                    plevel=5,
                )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_hwcn),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_hwcn),
                name="conv2d_hwcn.cuda",
            )
        elif layout == "NHWC" and kernel_layout == "HWIO":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.gpu.conv2d_nhwc),
                wrap_topi_schedule(topi.gpu.schedule_conv2d_nhwc),
                name="conv2d_nhwc.gpu",
            )

            N, H, W, _ = get_const_tuple(data.shape)
            KH, KW, CI, CO = get_const_tuple(kernel.shape)
            # Winograd shape related judgment
            (
                judge_winograd_tensorcore,
                judge_winograd_autotvm,
                judge_winograd_auto_scheduler,
            ) = judge_winograd(
                N,
                H,
                W,
                KH,
                KW,
                CI,
                CO,
                padding,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                data.dtype,
                kernel.dtype,
                pre_flag=False,
            )
            if judge_winograd_autotvm:
                if (
                    target.kind.name == "cuda"
                    and nvcc.have_tensorcore(target=target)
                    and judge_winograd_tensorcore
                ):
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.cuda.conv2d_nhwc_winograd_tensorcore),
                        wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_winograd_tensorcore),
                        name="conv2d_nhwc_winograd_tensorcore.cuda",
                        plevel=5,
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.cuda.conv2d_nhwc_winograd_direct),
                        wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_winograd_direct),
                        name="conv2d_nhwc_winograd_direct.cuda",
                        plevel=5,
                    )
            if (
                target.kind.name in ["maca"]
                and not is_auto_scheduler_enabled()
                and not is_meta_schedule_enabled()
                and mxcc.have_matrixcore()
                and (
                    (N % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
                    or (N % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
                    or (N % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
                )
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.maca.conv2d_nhwc_tensorcore),
                    wrap_topi_schedule(topi.maca.schedule_conv2d_nhwc_tensorcore),
                    name="conv2d_nhwc_tensorcore.maca",
                    plevel=20,
                )

            # register auto-scheduler implementations
            if is_auto_scheduler_enabled() and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nhwc.winograd",
                    plevel=15,
                )
            # register meta-schedule implementations
            if is_meta_schedule_enabled() and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nhwc.winograd",
                    plevel=15,
                )

        elif layout == "HWNC":
            assert kernel_layout in ["HWOI", "HWOI16o16i", "HWOI8o32i", "HWOI32o16i"]
            _, _, N, in_channels = get_const_tuple(data.shape)
            pre_computed = len(kernel.shape) == 6
            if pre_computed:
                _, _, oc_chunk, _, oc_block_factor, _ = get_const_tuple(kernel.shape)
                out_channels = oc_chunk * oc_block_factor
            else:
                _, _, out_channels, _ = get_const_tuple(kernel.shape)

            tensorcore_dtypes = ["int4", "uint4", "int8", "uint8"]
            if (
                target.kind.name == "cuda"
                and nvcc.have_tensorcore(target=target)
                and kernel.dtype in tensorcore_dtypes
                and (
                    (
                        data.dtype in ["int4", "uint4"]
                        and N % 8 == 0
                        and in_channels % 32 == 0
                        and out_channels % 8 == 0
                    )
                    or (
                        data.dtype in ["int8", "uint8"]
                        and N % 8 == 0
                        and in_channels % 16 == 0
                        and out_channels % 32 == 0
                    )
                )
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_hwnc_tensorcore),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_hwnc_tensorcore),
                    name="conv2d_hwnc_tensorcore_direct.cuda",
                    plevel=20,
                )
            else:
                raise RuntimeError(
                    "Unsupported shape for conv2d HWNC.\
                                    Need to satisfy tensor core schedule."
                )
        elif (
            (target.kind.name in ["cuda", "vulkan", "rocm"])
            and layout == "NCHW4c"
            and data.dtype in ["int8", "uint8"]
        ):
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_NCHWc_int8, need_data_layout=True),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_NCHWc_int8),
                name="conv2d_NCHWc_int8.cuda",
            )
        elif is_auto_scheduler_enabled() or is_meta_schedule_enabled():
            strategy.add_implementation(
                wrap_compute_conv2d(
                    topi.nn.conv, need_data_layout=True, need_kernel_layout=True, has_groups=True
                ),
                naive_schedule,
                name="conv2d.cuda",
                plevel=15,
            )
        elif target.kind.name == "maca" and "mcdnn" not in target.libs:
            # No TVM native kernel applicable
            raise RuntimeError(f"Unsupported conv2d layout {layout} for MACA")

        if (
            target.kind.name == "maca"
            and "mcdnn" in target.libs
            and layout in ["NCHW", "NHWC"]
            and padding[0] == padding[2]
            and padding[1] == padding[3]
            and not (data.dtype in ["uint8", "int8"] or kernel.dtype in ["uint8", "int8"])
        ):
            # add mcdnn implementation
            if layout == "NHWC":
                assert kernel_layout == "OHWI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.maca.conv2d_mcdnn, need_data_layout=True, has_groups=True),
                wrap_topi_schedule(topi.maca.schedule_conv2d_mcdnn),
                name="conv2d_mcdnn.maca",
                plevel=25,
            )

    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups) and (
        layout == "NCHW" or "mcdnn" not in target.libs
    ):  # mcDNN requires a different kernel layout for NHWC inputs.
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.cuda",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.cuda",
            )
        else:
            raise RuntimeError(f"Unsupported depthwise_conv2d layout {layout}")
    else:  # group_conv2d
        # add mcdnn implementation, if any
        mcdnn_impl = False
        if target.kind.name == "maca" and "mcdnn" in target.libs:
            if (
                layout in ["NCHW", "NHWC"]
                and padding[0] == padding[2]
                and padding[1] == padding[3]
                and not (data.dtype in ["uint8", "int8"] or kernel.dtype in ["uint8", "int8"])
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(
                        topi.maca.conv2d_mcdnn, need_data_layout=True, has_groups=True
                    ),
                    wrap_topi_schedule(topi.maca.schedule_conv2d_mcdnn),
                    name="conv2d_mcdnn.maca",
                    plevel=25,
                )
                mcdnn_impl = True

        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            _, channels, _, _ = get_const_tuple(data.shape)
            out_channels, in_channels, _, _ = get_const_tuple(kernel.shape)
            oc_chunk = out_channels // 4
            ic_chunk = in_channels // 4

            if (
                (target.kind.name in ["cuda", "vulkan", "rocm"])
                and data.dtype in ["int8", "uint8"]
                and kernel.dtype in ["int8", "uint8"]
                and channels % groups == 0
                and out_channels % groups == 0
                and channels % 4 == 0
                and out_channels % 4 == 0
                and groups <= oc_chunk
                and groups <= ic_chunk
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.group_conv2d_nchw_int8, has_groups=True),
                    wrap_topi_schedule(topi.cuda.schedule_group_conv2d_nchw_int8),
                    name="group_conv2d_nchw_int8.cuda",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.group_conv2d_nchw, has_groups=True),
                    wrap_topi_schedule(topi.cuda.schedule_group_conv2d_nchw),
                    name="group_conv2d_nchw.cuda",
                )
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.group_conv2d_NCHWc_int8, has_groups=True),
                wrap_topi_schedule(topi.cuda.schedule_group_conv2d_NCHWc_int8),
                name="group_conv2d_NCHWc_int8.cuda",
            )
        elif not mcdnn_impl:
            raise RuntimeError(f"Unsupported group_conv2d layout {layout}")
    return strategy

@softmax_strategy.register(["maca"])
def softmax_strategy_maca(attrs, inputs, out_type, target):
    """softmax maca strategy"""
    strategy = _op.OpStrategy()
    if target.kind.name == "maca" and "mcdnn" in target.libs:
        strategy.add_implementation(
            wrap_compute_softmax(topi.maca.softmax_mcdnn),
            wrap_topi_schedule(topi.maca.schedule_softmax_mcdnn),
            name="softmax.mcdnn",
            plevel=15,
        )
    return strategy

@log_softmax_strategy.register(["maca"])
def log_softmax_strategy_maca(attrs, inputs, out_type, target):
    """log_softmax maca strategy"""
    strategy = _op.OpStrategy()
    if target.kind.name == "maca" and "mcdnn" in target.libs:
        strategy.add_implementation(
            wrap_compute_softmax(topi.maca.log_softmax_mcdnn),
            wrap_topi_schedule(topi.maca.schedule_log_softmax_mcdnn),
            name="log_softmax.mcdnn",
            plevel=15,
        )
    return strategy

@conv2d_transpose_strategy.register(["maca"])
def conv2d_transpose_strategy_maca(attrs, inputs, out_type, target):
    """conv2d_transpose maca strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert dilation == (1, 1), "not support dilate now"
    strategy = _op.OpStrategy()
    num_strategies = 0

    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(topi.cuda.conv2d_transpose_nchw, has_groups=True),
            wrap_topi_schedule(topi.cuda.schedule_conv2d_transpose_nchw),
            name="conv2d_transpose_nchw.cuda",
        )
        num_strategies += 1

    if (
        target.kind.name == "maca"
        and "mcdnn" in target.libs
        and (
            (layout == "NCHW" and attrs.kernel_layout == "IOHW")
            or (layout == "NHWC" and attrs.kernel_layout == "IHWO")
        )
    ):
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(
                topi.maca.conv2d_transpose_mcdnn, add_layout=True, has_groups=True
            ),
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="conv2d_transpose.mcdnn.maca",
            plevel=25,
        )
        num_strategies += 1

    # TODO(masahi): Support conv2d_transpose NHWC for non-cudnn path.
    assert (
        num_strategies > 0
    ), f"Unsupported conv2d_transpose workload, layout = {layout}, groups = {groups}"
    return strategy

@conv3d_strategy.register(["maca"])
def conv3d_strategy_maca(attrs, inputs, out_type, target):
    """conv3d maca strategy"""
    strategy = _op.OpStrategy()
    if target.kind.name == "maca" and "mcdnn" in target.libs:
        strategy.add_implementation(
            wrap_compute_conv3d(topi.maca.conv3d_mcdnn, True),
            wrap_topi_schedule(topi.maca.schedule_conv3d_mcdnn),
            name="conv3d_mcdnn.maca",
            plevel=25,
        )
    return strategy