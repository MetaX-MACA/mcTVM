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
# pylint: disable=invalid-name,line-too-long
# ruff: noqa: E501
"""Various type definitions to help instantiate MCTLASS kernels."""

import enum
import re
from enum import auto as enum_auto

from tvm.tirx.expr import FloatImm, IntImm


class GeneratorTarget(enum.Enum):
    Library = enum_auto()


class DataType(enum.Enum):
    f16 = enum_auto()
    f32 = enum_auto()
    s8 = enum_auto()
    u8 = enum_auto()
    s32 = enum_auto()


ShortDataTypeNames = {DataType.f16: "h", DataType.f32: "s", DataType.s32: "i"}


DataTypeNames = {
    DataType.f16: "f16",
    DataType.f32: "f32",
    DataType.s8: "s8",
    DataType.u8: "u8",
    DataType.s32: "s32",
}

DataTypeTag = {
    DataType.f16: "mctlass::half_t",
    DataType.f32: "float",
    DataType.s8: "int8_t",
    DataType.s32: "int32_t",
    DataType.u8: "uint8_t",
}

DataTypeSize = {
    DataType.f16: 16,
    DataType.f32: 32,
    DataType.u8: 8,
    DataType.s8: 8,
    DataType.s32: 32,
}


class MathOperation(enum.Enum):
    multiply_add = enum_auto()
    multiply_add_saturate = enum_auto()
    multiply_add_fast_f32 = enum_auto()


MathOperationTag = {
    MathOperation.multiply_add: "mctlass::arch::OpMultiplyAdd",
    MathOperation.multiply_add_saturate: "mctlass::arch::OpMultiplyAddSaturate",
    MathOperation.multiply_add_fast_f32: "mctlass::arch::OpMultiplyAddFastF32",
}


class LayoutType(enum.Enum):
    ColumnMajor = enum_auto()
    RowMajor = enum_auto()
    TensorNHWC = enum_auto()


LayoutTag = {
    LayoutType.ColumnMajor: "mctlass::layout::ColumnMajor",
    LayoutType.RowMajor: "mctlass::layout::RowMajor",
    LayoutType.TensorNHWC: "mctlass::layout::TensorNHWC",
}


TransposedLayout = {
    LayoutType.ColumnMajor: LayoutType.RowMajor,
    LayoutType.RowMajor: LayoutType.ColumnMajor,
    LayoutType.TensorNHWC: LayoutType.TensorNHWC,
}


ShortLayoutTypeNames = {
    LayoutType.ColumnMajor: "n",
    LayoutType.RowMajor: "t",
    LayoutType.TensorNHWC: "nhwc",
}


class OpcodeClass(enum.Enum):
    Simt = enum_auto()
    TensorOp = enum_auto()
    WmmaTensorOp = enum_auto()


OpcodeClassNames = {
    OpcodeClass.Simt: "simt",
    OpcodeClass.TensorOp: "tensorop",
    OpcodeClass.WmmaTensorOp: "wmma_tensorop",
}

OpcodeClassTag = {
    OpcodeClass.Simt: "mctlass::arch::OpClassSimt",
    OpcodeClass.TensorOp: "mctlass::arch::OpClassTensorOp",
    OpcodeClass.WmmaTensorOp: "mctlass::arch::OpClassWmmaTensorOp",
}


class OperationKind(enum.Enum):
    Gemm = enum_auto()
    Conv2d = enum_auto()


OperationKindNames = {OperationKind.Gemm: "gemm", OperationKind.Conv2d: "conv2d"}


class Target(enum.Enum):
    library = enum_auto()


def substitute_template(template, values):
    """Instantiate a kernel template using `values`."""
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            if isinstance(value, int | IntImm):
                value = str(int(value))
            if isinstance(value, float | FloatImm):
                value = str(float(value))
            elif isinstance(value, bool):
                value = str(value).lower()
            regex = f"\\$\\{{{key}\\}}"
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


class GemmKind(enum.Enum):
    Gemm = enum_auto()


GemmKindNames = {GemmKind.Gemm: "gemm"}


class EpilogueFunctor(enum.Enum):
    LinearCombination = enum_auto()
    LinearCombinationRelu = enum_auto()
    LinearCombinationBias = enum_auto()
    LinearCombinationGelu = enum_auto()
    LinearCombinationSigmoid = enum_auto()
    LinearCombinationSilu = enum_auto()
    LinearCombinationHardSwish = enum_auto()
    LinearCombinationResidualBlock = enum_auto()


EpilogueFunctorTag = {
    EpilogueFunctor.LinearCombination: "mctlass::epilogue::thread::LinearCombination",
    EpilogueFunctor.LinearCombinationRelu: "mctlass::epilogue::thread::LinearCombinationRelu",
    EpilogueFunctor.LinearCombinationBias: "mctlass::epilogue::thread::LinearCombination",
    EpilogueFunctor.LinearCombinationGelu: "mctlass::epilogue::thread::LinearCombinationGELU",
    EpilogueFunctor.LinearCombinationSigmoid: "mctlass::epilogue::thread::LinearCombinationSigmoid",
    EpilogueFunctor.LinearCombinationSilu: "mctlass::epilogue::thread::LinearCombinationSilu",
    EpilogueFunctor.LinearCombinationHardSwish: "mctlass::epilogue::thread::LinearCombinationHardSwish",
    EpilogueFunctor.LinearCombinationResidualBlock: "mctlass::epilogue::thread::LinearCombinationResidualBlock",
}


class SwizzlingFunctor(enum.Enum):
    Identity1 = enum_auto()
    Identity2 = enum_auto()
    Identity4 = enum_auto()
    Identity8 = enum_auto()
    Batched = enum_auto()
    StridedDgradIdentity1 = enum_auto()
    StridedDgradIdentity4 = enum_auto()


SwizzlingFunctorTag = {
    SwizzlingFunctor.Identity1: "mctlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>",
    SwizzlingFunctor.Identity2: "mctlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>",
    SwizzlingFunctor.Identity4: "mctlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>",
    SwizzlingFunctor.Identity8: "mctlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>",
    SwizzlingFunctor.Batched: "mctlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle",
    SwizzlingFunctor.StridedDgradIdentity1: "mctlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>",
    SwizzlingFunctor.StridedDgradIdentity4: "mctlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<4>",
}


class ConvKind(enum.Enum):
    Fprop = enum_auto()
    Dgrad = enum_auto()
    Wgrad = enum_auto()


ConvKindTag = {
    ConvKind.Fprop: "mctlass::conv::Operator::kFprop",
    ConvKind.Dgrad: "mctlass::conv::Operator::kDgrad",
    ConvKind.Wgrad: "mctlass::conv::Operator::kWgrad",
}


ConvKindNames = {ConvKind.Fprop: "fprop", ConvKind.Dgrad: "dgrad", ConvKind.Wgrad: "wgrad"}


class StrideSupport(enum.Enum):
    Strided = enum_auto()
    Unity = enum_auto()


StrideSupportTag = {
    StrideSupport.Strided: "mctlass::conv::StrideSupport::kStrided",
    StrideSupport.Unity: "mctlass::conv::StrideSupport::kUnity",
}


StrideSupportNames = {StrideSupport.Strided: "", StrideSupport.Unity: "unity_stride"}


class IteratorAlgorithm(enum.Enum):
    Analytic = enum_auto()
    Optimized = enum_auto()


IteratorAlgorithmTag = {
    IteratorAlgorithm.Analytic: "mctlass::conv::IteratorAlgorithm::kAnalytic",
    IteratorAlgorithm.Optimized: "mctlass::conv::IteratorAlgorithm::kOptimized",
}


IteratorAlgorithmNames = {
    IteratorAlgorithm.Analytic: "analytic",
    IteratorAlgorithm.Optimized: "optimized",
}


class MathInstruction:
    """Describe characteristics of a math instruction."""

    def __init__(
        self,
        instruction_shape,
        element_a,
        element_b,
        element_c,
        element_accumulator,
        opcode_class,
        math_operation=MathOperation.multiply_add,
    ):
        self.instruction_shape = instruction_shape
        self.element_a = element_a
        self.element_b = element_b
        self.element_c = element_c
        self.element_accumulator = element_accumulator
        self.opcode_class = opcode_class
        self.math_operation = math_operation


class TileDescription:
    """Describe characteristics of a GEMM tile."""

    def __init__(
        self, threadblock_shape, stages, warp_count, math_instruction, min_compute, max_compute
    ):
        self.threadblock_shape = threadblock_shape
        self.stages = stages
        self.warp_count = warp_count
        self.math_instruction = math_instruction
        self.minimum_compute_capability = min_compute
        self.maximum_compute_capability = max_compute

    def procedural_name(self):
        return f"{self.threadblock_shape[0]}x{self.threadblock_shape[1]}_{self.threadblock_shape[2]}x{self.stages}"


class TensorDescription:
    def __init__(self, element, layout, alignment=1):
        self.element = element
        self.layout = layout
        self.alignment = alignment
