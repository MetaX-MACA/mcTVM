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
"""MACA m16n16k16 matrix multiply-accumulate intrinsic codegens."""

from ._schema import device_intrinsic

_MMA_SIGNATURE = "(float* d_ptr, const void* a_ptr, const void* b_ptr, const float* c_ptr)"

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
