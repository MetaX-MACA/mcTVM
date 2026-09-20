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
"""MACA matrix multiply-accumulate intrinsic codegens."""

from textwrap import dedent

from ._schema import device_intrinsic

_MMA_SIGNATURE = "(float* d_ptr, const void* a_ptr, const void* b_ptr, const float* c_ptr)"

_V4_INT = "typedef __NATIVE_VECTOR__(4, int) v4i;"
_V4_FLOAT = "typedef __NATIVE_VECTOR__(4, float) v4f;"

device_intrinsic(
    "maca_mma_m16n16k16_f16_f32",
    c_signature=_MMA_SIGNATURE,
    body="""    typedef __NATIVE_VECTOR__(4, _Float16) v4h;
    typedef __NATIVE_VECTOR__(4, float) v4f;
    const _Float16* a = reinterpret_cast<const _Float16*>(a_ptr);
    const _Float16* b = reinterpret_cast<const _Float16*>(b_ptr);
    v4h a_fragment = {a[0], a[1], a[2], a[3]};
    v4h b_fragment = {b[0], b[1], b[2], b[3]};
    v4f c_fragment = {c_ptr[0], c_ptr[1], c_ptr[2], c_ptr[3]};
    v4f result = __builtin_mxc_mma_16x16x16f16(a_fragment, b_fragment, c_fragment);
    d_ptr[0] = result[0];
    d_ptr[1] = result[1];
    d_ptr[2] = result[2];
    d_ptr[3] = result[3];""",
)

device_intrinsic(
    "maca_mma_m16n16k16_i8_i32",
    c_signature="(int* d_ptr, const void* a_ptr, const void* b_ptr, const int* c_ptr)",
    body=dedent(
        f"""
        {_V4_INT}
        const int* a = reinterpret_cast<const int*>(a_ptr);
        const int* b = reinterpret_cast<const int*>(b_ptr);
        v4i c_fragment = {{c_ptr[0], c_ptr[1], c_ptr[2], c_ptr[3]}};
        v4i result = __builtin_mxc_mma_16x16x16i8(a[0], b[0], c_fragment);
        d_ptr[0] = result[0]; d_ptr[1] = result[1];
        d_ptr[2] = result[2]; d_ptr[3] = result[3];
        """
    ),
)


device_intrinsic(
    "maca_mma_m16n16k4_f32_f32",
    c_signature="(float* d_ptr, const float* a_ptr, const float* b_ptr, const float* c_ptr)",
    body=dedent(
        f"""
        {_V4_FLOAT}
        v4f c_fragment = {{c_ptr[0], c_ptr[1], c_ptr[2], c_ptr[3]}};
        v4f result = __builtin_mxc_mma_16x16x4f32(a_ptr[0], b_ptr[0], c_fragment);
        d_ptr[0] = result[0];
        d_ptr[1] = result[1];
        d_ptr[2] = result[2];
        d_ptr[3] = result[3];
        """
    ),
)

device_intrinsic(
    "maca_mma_m16n16k8_tf32_f32",
    c_signature="(float* d_ptr, const float* a_ptr, const float* b_ptr, const float* c_ptr)",
    body=dedent(
        """
        // A/B use FP32 storage and native TF32 multiplier interpretation;
        // C/D remain FP32 accumulators and outputs.
        typedef __NATIVE_VECTOR__(2, float) v2f;
        typedef __NATIVE_VECTOR__(4, float) v4f;
        // Layout storage is low-K then high-K so it has positive physical
        // strides.  The builtin fragment order is high-K then low-K.
        v2f a_fragment = {a_ptr[1], a_ptr[0]};
        v2f b_fragment = {b_ptr[1], b_ptr[0]};
        v4f c_fragment = {c_ptr[0], c_ptr[1], c_ptr[2], c_ptr[3]};
        v4f result = __builtin_mxc_mma_16x16x8tf32(a_fragment, b_fragment, c_fragment);
        d_ptr[0] = result[0]; d_ptr[1] = result[1];
        d_ptr[2] = result[2]; d_ptr[3] = result[3];
        """
    ),
)


device_intrinsic(
    "maca_mma_m16n16k4_f64_f64",
    c_signature="(double* d_ptr, const double* a_ptr, const double* b_ptr, const double* c_ptr)",
    body=dedent(
        """
        typedef __NATIVE_VECTOR__(4, double) v4d;
        v4d c_fragment = {c_ptr[0], c_ptr[1], c_ptr[2], c_ptr[3]};
        v4d result = __builtin_mxc_mma_16x16x4f64(a_ptr[0], b_ptr[0], c_fragment);
        for (unsigned i = 0; i < 4; ++i) d_ptr[i] = result[i];
        """
    ),
)


device_intrinsic(
    "maca_mma_m16n16k16_bf16_f32",
    c_signature=_MMA_SIGNATURE,
    body="""    typedef __NATIVE_VECTOR__(4, _Float16) v4h;
    typedef __NATIVE_VECTOR__(4, float) v4f;
    const unsigned short* a_bits = reinterpret_cast<const unsigned short*>(a_ptr);
    const unsigned short* b_bits = reinterpret_cast<const unsigned short*>(b_ptr);
    v4h a_fragment = {*reinterpret_cast<const _Float16*>(&a_bits[0]),
                      *reinterpret_cast<const _Float16*>(&a_bits[1]),
                      *reinterpret_cast<const _Float16*>(&a_bits[2]),
                      *reinterpret_cast<const _Float16*>(&a_bits[3])};
    v4h b_fragment = {*reinterpret_cast<const _Float16*>(&b_bits[0]),
                      *reinterpret_cast<const _Float16*>(&b_bits[1]),
                      *reinterpret_cast<const _Float16*>(&b_bits[2]),
                      *reinterpret_cast<const _Float16*>(&b_bits[3])};
    v4f c_fragment = {c_ptr[0], c_ptr[1], c_ptr[2], c_ptr[3]};
    v4f result = __builtin_mxc_mma_16x16x16bf16(a_fragment, b_fragment, c_fragment);
    d_ptr[0] = result[0];
    d_ptr[1] = result[1];
    d_ptr[2] = result[2];
    d_ptr[3] = result[3];""",
)
