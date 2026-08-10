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
# pylint: disable=import-outside-toplevel, invalid-name
# ruff: noqa: E501
"""Instantiate a C++ source for profiling MCTLASS kernels."""

from .library import DataTypeTag


class Conv2dProfilerEmitter:
    """Emit a C++ source for profiling MCTLASS kernels."""

    def __init__(self):
        from jinja2 import Template

        self.reduction = """
      ReductionDevice reduction_op;
      static mctlass::conv::Operator const kConvolutionalOperator = ImplicitGemm::kConvolutionalOperator;
      typename ReductionDevice::Arguments reduction_args(
        mctlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn(),
        problem_size.split_k_slices,
        mctlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, problem_size),
        {
           reinterpret_cast<ImplicitGemm::ElementC*> (workspace.get()),
             ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
             },
        {
           tensor_d.device_data(),
             ReductionStrideIndex(tensor_d.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
             },
        {
           tensor_c.device_data(),
             ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
             },
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}
        );

      reduction_op.initialize(reduction_args, nullptr);
      reduction_op();
"""

        self.template = Template(
            """
#include <iostream>
#include "mctlass/mctlass.h"
#include "mctlass/conv/kernel/default_conv2d_fprop.h"
#include "mctlass/conv/kernel/default_conv2d_wgrad.h"
#include "mctlass/conv/kernel/default_conv2d_dgrad.h"
#include "mctlass/conv/device/implicit_gemm_convolution.h"
#include "mctlass/util/command_line.h"
#include "mctlass/util/host_tensor.h"
#include "mctlass/util/reference/host/tensor_fill.h"
#include "mctlass/reduction/device/reduce_split_k.h"
#include "mctlass/reduction/thread/reduction_operators.h"

#define MCTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    mctlass::Status error = status;                                                              \
    if (error != mctlass::Status::kSuccess) {                                                    \
      std::cerr << "Got mctlass error: " << mctlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

{{OperatorDef}}
using ImplicitGemm = mctlass::conv::device::ImplicitGemmConvolution<{{OperatorName}}>;

struct Options {
  mctlass::Tensor4DCoord input_size;
  mctlass::Tensor4DCoord filter_size;
  mctlass::Tensor4DCoord padding;
  mctlass::MatrixCoord conv_stride;
  mctlass::MatrixCoord dilation;

  void parse(int argc, char const **args) {
    mctlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());
    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
    cmd.get_cmd_line_argument("pad_h", pad_h);
    cmd.get_cmd_line_argument("pad_w", pad_w);
    cmd.get_cmd_line_argument("stride_h", stride_h);
    cmd.get_cmd_line_argument("stride_w", stride_w);
    cmd.get_cmd_line_argument("dilation_h", dilation_h);
    cmd.get_cmd_line_argument("dilation_w", dilation_w);
    filter_size.c() = input_size.c();
    padding = {pad_h, pad_h, pad_w, pad_w};
    conv_stride = {stride_h, stride_w};
    dilation = {dilation_h, dilation_w};
  }

  mctlass::Tensor4DCoord output_size() const {
    auto dilated_h = (filter_size.h() - 1) * dilation.row() + 1;
    auto dilated_w = (filter_size.w() - 1) * dilation.column() + 1;
    auto h = (input_size.h() + padding.n() + padding.h() - dilated_h) / conv_stride.row() + 1;
    auto w = (input_size.w() + padding.w() + padding.c() - dilated_w) / conv_stride.column() + 1;
    return mctlass::Tensor4DCoord(input_size.n(), h, w, filter_size.n());
  }
};

double profile_convolution(Options const &options) {
  using ElementOutput = {{ElementOutput}};
  using ElementInputA = typename ImplicitGemm::ElementA;
  using ElementInputB = typename ImplicitGemm::ElementB;

  int split_k_slices = {{SplitK}};
  mctlass::conv::Conv2dProblemSize problem_size(
                        options.input_size,
                        options.filter_size,
                        options.padding,
                        options.conv_stride,
                        options.dilation,
                        options.output_size(),
                        mctlass::conv::Mode::kCrossCorrelation,
                        split_k_slices
                        );

  auto conv_kind = ImplicitGemm::kConvolutionalOperator;
  auto a_extent = implicit_gemm_tensor_a_extent(conv_kind, problem_size);
  auto b_extent = implicit_gemm_tensor_b_extent(conv_kind, problem_size);
  auto c_extent = implicit_gemm_tensor_c_extent(conv_kind, problem_size);

  using LayoutC = typename ImplicitGemm::LayoutC;
  mctlass::HostTensor<ElementInputA, typename ImplicitGemm::LayoutA> tensor_a(a_extent);
  mctlass::HostTensor<ElementInputB, typename ImplicitGemm::LayoutB> tensor_b(b_extent);
  mctlass::HostTensor<ElementOutput, typename ImplicitGemm::LayoutC> tensor_c(c_extent);
  mctlass::HostTensor<ElementOutput, LayoutC> tensor_d(c_extent);
  mctlass::HostTensor<ImplicitGemm::ElementC, LayoutC> tensor_c_gemm(c_extent);

  using ElementComputeEpilogue = typename ImplicitGemm::ElementCompute;

  mctlass::conv::SplitKMode const split_k_mode = split_k_slices > 1 ?
      mctlass::conv::SplitKMode::kParallel : mctlass::conv::SplitKMode::kSerial;

  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_c_gemm.device_ref(),
    tensor_c_gemm.device_ref(),
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
    split_k_mode,
  };

  ImplicitGemm implicit_gemm_op;
  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
  mctlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  auto status = implicit_gemm_op.can_implement(arguments);
  MCTLASS_CHECK(status);

  status = implicit_gemm_op.initialize(arguments, workspace.get());
  MCTLASS_CHECK(status);
  status = implicit_gemm_op();
  MCTLASS_CHECK(status);

  mcEvent_t events[2];
  for (auto & event : events) {
    mcEventCreate(&event);
  }
  mcEventRecord(events[0]);

  for (int iteration = 0; iteration < 100; ++iteration) {
    auto status = implicit_gemm_op();
    MCTLASS_CHECK(status);
    {{Reduction}}
  }

  mcEventRecord(events[1]);
  mcEventSynchronize(events[1]);
  float runtime_ms = 0;
  mcEventElapsedTime(&runtime_ms, events[0], events[1]);

  for (auto event : events) {
    (void)mcEventDestroy(event);
  }
  return double(runtime_ms) / 100.0;
}

int main(int argc, char const **args) {
  Options options;
  options.parse(argc, args);
  std::cout << profile_convolution(options) << std::endl;
  return 0;
}
"""
        )

    def emit(self, op_def, op_name, element_output, split_k_slices=1):
        src = self.template.render(
            OperatorDef=op_def,
            OperatorName=op_name,
            ElementOutput=DataTypeTag[element_output],
            SplitK=split_k_slices,
            Reduction=self.reduction if split_k_slices > 1 else "",
        )
        return src
