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
    "maca_mma_m16n16k16_f16_f16",
    c_signature="(void* d_ptr, const void* a_ptr, const void* b_ptr, const void* c_ptr)",
    body=dedent(
        """
        typedef __NATIVE_VECTOR__(4, _Float16) v4h;
        typedef __NATIVE_VECTOR__(4, float) v4f;
        const _Float16* a = reinterpret_cast<const _Float16*>(a_ptr);
        const _Float16* b = reinterpret_cast<const _Float16*>(b_ptr);
        const _Float16* c = reinterpret_cast<const _Float16*>(c_ptr);
        _Float16* d = reinterpret_cast<_Float16*>(d_ptr);
        v4h a_fragment = {a[0], a[1], a[2], a[3]};
        v4h b_fragment = {b[0], b[1], b[2], b[3]};
        v4f c_fragment = {static_cast<float>(c[0]), static_cast<float>(c[1]),
                          static_cast<float>(c[2]), static_cast<float>(c[3])};
        v4f result = __builtin_mxc_mma_16x16x16f16(a_fragment, b_fragment, c_fragment);
        d[0] = static_cast<_Float16>(result[0]);
        d[1] = static_cast<_Float16>(result[1]);
        d[2] = static_cast<_Float16>(result[2]);
        d[3] = static_cast<_Float16>(result[3]);
        """
    ),
)


def _m16n16k16_i8_body(unsigned: bool) -> str:
    """Emit the C500 packed-byte helper for one signedness family."""
    builtin = "__builtin_mxc_mma_16x16x16i8"
    correction = ""
    if unsigned:
        # The SDK's unsigned overload calls the signed instruction unchanged.
        # Split each byte into a positive low 7-bit value and its high bit:
        # (al + 128*ah)*(bl + 128*bh).  All four MMA operands fit signed i8.
        correction = f"""
        v4i zero = {{0, 0, 0, 0}};
        unsigned av = static_cast<unsigned>(a[0]);
        unsigned bv = static_cast<unsigned>(b[0]);
        int al = av & 0x7f7f7f7fu, ah = (av >> 7) & 0x01010101u;
        int bl = bv & 0x7f7f7f7fu, bh = (bv >> 7) & 0x01010101u;
        v4i low = {builtin}(al, bl, zero);
        v4i cross_a = {builtin}(ah, bl, zero);
        v4i cross_b = {builtin}(al, bh, zero);
        v4i high = {builtin}(ah, bh, zero);
        for (unsigned i = 0; i < 4; ++i) {{
          unsigned value = static_cast<unsigned>(c_fragment[i])
              + static_cast<unsigned>(low[i])
              + ((static_cast<unsigned>(cross_a[i]) + cross_b[i]) << 7)
              + (static_cast<unsigned>(high[i]) << 14);
          result[i] = __builtin_bit_cast(int, value);
        }}
        """
    issue = "v4i result;" if unsigned else f"v4i result = {builtin}(a[0], b[0], c_fragment);"
    return dedent(
        f"""
        {_V4_INT}
        const int* a = reinterpret_cast<const int*>(a_ptr);
        const int* b = reinterpret_cast<const int*>(b_ptr);
        const int* c = reinterpret_cast<const int*>(c_ptr);
        int* d = reinterpret_cast<int*>(d_ptr);
        v4i c_fragment = {{c[0], c[1], c[2], c[3]}};
        {issue}
        {correction}
        d[0] = result[0];
        d[1] = result[1];
        d[2] = result[2];
        d[3] = result[3];
        """
    )


for _name, _unsigned in (("i8", False), ("u8", True)):
    device_intrinsic(
        f"maca_mma_m16n16k16_{_name}_i32",
        c_signature="(void* d_ptr, const void* a_ptr, const void* b_ptr, const void* c_ptr)",
        body=_m16n16k16_i8_body(_unsigned),
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


for _name, _signed in (("i4", True), ("u4", False)):
    # Each lane owns four consecutive nibbles. Gather the row and column
    # chunks for its one output, expand them into signed bytes, and perform
    # the SDK compatibility dot product without duplicating C in a reduction.
    device_intrinsic(
        f"maca_mma_m8n8k32_{_name}_i32",
        c_signature="(int* d_ptr, const void* a_ptr, const void* b_ptr, const int* c_ptr)",
        body=dedent(
            f"""
            unsigned a = *reinterpret_cast<const unsigned short*>(a_ptr);
            unsigned b = *reinterpret_cast<const unsigned short*>(b_ptr);
            unsigned lane = __lane_id();
            unsigned count = static_cast<unsigned>(c_ptr[0]);
            for (unsigned chunk = 0; chunk < 8; ++chunk) {{
              unsigned av = __builtin_mxc_bsm_bpermute(((lane / 8) + 8 * chunk) << 2, a);
              unsigned bv = __builtin_mxc_bsm_bpermute(((lane % 8) + 8 * chunk) << 2, b);
              for (unsigned i = 0; i < 4; ++i) {{
                int ax = (av >> (4 * i)) & 15;
                int bx = (bv >> (4 * i)) & 15;
                {"ax = (ax ^ 8) - 8; bx = (bx ^ 8) - 8;" if _signed else ""}
                count += static_cast<unsigned>(ax * bx);
              }}
            }}
            d_ptr[0] = __builtin_bit_cast(int, count);
            """
        ),
    )


device_intrinsic(
    "maca_bmma_m8n8k128_b1_i32",
    c_signature="(int* d_ptr, const void* a_ptr, const void* b_ptr, const int* c_ptr)",
    body=dedent(
        """
        unsigned a = *reinterpret_cast<const unsigned short*>(a_ptr);
        unsigned b = *reinterpret_cast<const unsigned short*>(b_ptr);
        unsigned lane = __lane_id();
        unsigned count = static_cast<unsigned>(c_ptr[0]);
        for (unsigned chunk = 0; chunk < 8; ++chunk) {
          unsigned av = __builtin_mxc_bsm_bpermute(((lane / 8) + 8 * chunk) << 2, a);
          unsigned bv = __builtin_mxc_bsm_bpermute(((lane % 8) + 8 * chunk) << 2, b);
          count += __builtin_popcount(av & bv);
        }
        d_ptr[0] = __builtin_bit_cast(int, count);
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
