/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file maca_half_t.h
 * \brief half_t (fp16) definition for maca codegen.
 */
#ifndef TVM_TARGET_SOURCE_LITERAL_MACA_HALF_T_H_
#define TVM_TARGET_SOURCE_LITERAL_MACA_HALF_T_H_

#include <string>

static constexpr const char* _maca_half_t_def = R"(
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
)";

static constexpr const char* _maca_half_util = R"(
// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define MACA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define MACA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in maca_fp16.h,
// so we define them here to make sure the generated MACA code
// is valid.
#if defined(__MACA_ARCH__)
#if (__MACA_ARCH__ >= 530)
MACA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
MACA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
MACA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
MACA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
MACA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
MACA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef MACA_UNSUPPORTED_HALF_MATH_BINARY
#undef MACA_UNSUPPORTED_HALF_MATH_UNARY
)";

static constexpr const char* _maca_bfloat16_util = R"(
// Pack two bfloat16 values.
static inline __device__ __host__ unsigned
__pack_maca_bfloat162(const maca_bfloat16 x, const maca_bfloat16 y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Some bfp16 math functions are not supported in maca_bfloat16.h,
// so we define them here to make sure the generated MACA code
// is valid.
#define MACA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ maca_bfloat16 HALF_MATH_NAME(maca_bfloat16 x, maca_bfloat16 y) {   \
  float tmp_x = __bfloat162float(x);                                      \
  float tmp_y = __bfloat162float(y);                                      \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2bfloat16(result);                                        \
}

#define MACA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ maca_bfloat16 HALF_MATH_NAME(maca_bfloat16 x) {          \
  float tmp_x = __bfloat162float(x);                                     \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2bfloat16(result);                                       \
}

MACA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
MACA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
MACA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
MACA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
MACA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef MACA_UNSUPPORTED_HALF_MATH_BINARY
#undef MACA_UNSUPPORTED_HALF_MATH_UNARY
)";

static constexpr const char* _maca_warp_intrinsic_util = R"(
#if defined(__MACA_ARCH__) && (__MACA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif

)";

static void declare_vector_type_extensions(std::ostringstream& stream, bool enable_fp16,
                                           bool enable_bf16, bool enable_fp8) {
  if (enable_fp16 || enable_bf16) {
    stream << R"(
#include <type_traits>
template <typename T, typename TVec2>
struct __align__(8) half4_bfloat164 {
  T x, y, z, w;
)";
    if (enable_fp8) {
      stream << R"(
  __host__ __device__ half4_bfloat164() : x(T(0)), y(T(0)), z(T(0)), w(T(0)) {}
  __host__ __device__ half4_bfloat164(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
  __host__ __device__ half4_bfloat164(const __maca_fp8x4_e4m3& fp8x4) {
    if constexpr (std::is_same_v<T, __half>) {
      __maca_fp8x2_e4m3 lo_part, hi_part;
      lo_part.__x = static_cast<__maca_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
      hi_part.__x = static_cast<__maca_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);
      TVec2 lo_half2 = static_cast<TVec2>(lo_part);
      TVec2 hi_half2 = static_cast<TVec2>(hi_part);
      x = reinterpret_cast<T*>(&lo_half2)[0];
      y = reinterpret_cast<T*>(&lo_half2)[1];
      z = reinterpret_cast<T*>(&hi_half2)[0];
      w = reinterpret_cast<T*>(&hi_half2)[1];
    } else {
      __maca_fp8_storage_t elem0_raw = static_cast<__maca_fp8_storage_t>(fp8x4.__x & 0xFF);
      __maca_fp8_storage_t elem1_raw = static_cast<__maca_fp8_storage_t>((fp8x4.__x >> 8) & 0xFF);
      __maca_fp8_storage_t elem2_raw = static_cast<__maca_fp8_storage_t>((fp8x4.__x >> 16) & 0xFF);
      __maca_fp8_storage_t elem3_raw = static_cast<__maca_fp8_storage_t>((fp8x4.__x >> 24) & 0xFF);
      __maca_fp8_e4m3 elem0, elem1, elem2, elem3;
      elem0.__x = elem0_raw;
      elem1.__x = elem1_raw;
      elem2.__x = elem2_raw;
      elem3.__x = elem3_raw;
      x = T(elem0);
      y = T(elem1);
      z = T(elem2);
      w = T(elem3);
    }
  }
  __host__ __device__ explicit operator __maca_fp8x4_e4m3() const {
    __maca_fp8x4_e4m3 result;
    TVec2 lo_half2 = *reinterpret_cast<const TVec2*>(&x);
    TVec2 hi_half2 = *reinterpret_cast<const TVec2*>(&z);
    __maca_fp8x2_e4m3 lo_part(lo_half2), hi_part(hi_half2);
    result.__x =
        (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
  __host__ __device__ explicit half4_bfloat164(const __maca_fp8x4_e5m2& fp8x4) {
      __maca_fp8x2_e5m2 lo_part, hi_part;
      lo_part.__x = static_cast<__maca_fp8x2_storage_t>(fp8x4.__x & 0xFFFF);
      hi_part.__x = static_cast<__maca_fp8x2_storage_t>((fp8x4.__x >> 16) & 0xFFFF);
      TVec2 lo_half2 = static_cast<TVec2>(lo_part);
      TVec2 hi_half2 = static_cast<TVec2>(hi_part);
      x = reinterpret_cast<T*>(&lo_half2)[0];
      y = reinterpret_cast<T*>(&lo_half2)[1];
      z = reinterpret_cast<T*>(&hi_half2)[0];
      w = reinterpret_cast<T*>(&hi_half2)[1];
  }
  __host__ __device__ explicit operator __maca_fp8x4_e5m2() const {
    __maca_fp8x4_e5m2 result;
    TVec2 lo_half2 = *reinterpret_cast<const TVec2*>(&x);
    TVec2 hi_half2 = *reinterpret_cast<const TVec2*>(&z);
    __maca_fp8x2_e5m2 lo_part(lo_half2), hi_part(hi_half2);
    result.__x =
        (static_cast<__uint32_t>(lo_part.__x) | (static_cast<__uint32_t>(hi_part.__x) << 16));
    return result;
  }
  __device__ __maca_fp8x2_e5m2 make_fp8x2_e5m2(__maca_fp8_storage_t x, __maca_fp8_storage_t y) {
      __maca_fp8x2_e5m2 result;
      result.__x = (x) | (y << 8);
      return result;
  }
  __device__ __maca_fp8x4_e5m2 make_fp8x4_e5m2(__maca_fp8_storage_t a, __maca_fp8_storage_t b, __maca_fp8_storage_t c, __maca_fp8_storage_t d) {
      __maca_fp8x4_e5m2 result;
      result.__x = (a) | (b << 8) | (c << 16) | (d << 24);
      return result;
  }
  __device__ __maca_fp8x2_e4m3 make___maca_fp8x2_e4m3(__maca_fp8x2_storage_t x, __maca_fp8x2_storage_t y) {
      __maca_fp8x2_e4m3 result;
      result.__x = (x) | (y << 8);
      return result;
  }
  __device__ __maca_fp8x4_e4m3 make___maca_fp8x4_e4m3(__maca_fp8x4_storage_t a, __maca_fp8x4_storage_t b, __maca_fp8x4_storage_t c, __maca_fp8x4_storage_t d) {
      __maca_fp8x4_e4m3 result;
      result.__x = (a) | (b << 8) | (c << 16) | (d << 24);
      return result;
  }
  )";
    }
    stream << R"(
};
)";
  }
  if (enable_fp16) {
    stream << R"(
using half4 = half4_bfloat164<__half, __half2>;
__host__ __device__ half4 make_half4(__half x, __half y, __half z, __half w) {
    return half4{x, y, z, w};
}
)";
  }
  if (enable_bf16) {
    stream << R"(
using maca_bfloat164 = half4_bfloat164<maca_bfloat16, maca_bfloat162>;
__host__ __device__ maca_bfloat164 make_maca_bfloat164(maca_bfloat16 x, maca_bfloat16 y, maca_bfloat16 z, maca_bfloat16 w) {
    return maca_bfloat164{x, y, z, w};
}
)";
    if (enable_fp8) {
      stream << R"(
__host__ __device__ maca_bfloat162 cast_to_maca_bfloat162(const __maca_fp8x2_e4m3& fp8x2) {
    __maca_fp8_e4m3 elem0, elem1;
    elem0.__x = static_cast<__maca_fp8_storage_t>(fp8x2.__x & 0xFF);
    elem1.__x = static_cast<__maca_fp8_storage_t>((fp8x2.__x >> 8) & 0xFF);
    maca_bfloat16 x = maca_bfloat16(elem0);
    maca_bfloat16 y = maca_bfloat16(elem1);
    return maca_bfloat162(x, y);
}
      )";
    }
  }
  if (enable_fp8) {
    stream << R"(
__host__ __device__ __maca_fp8x2_e4m3 make___maca_fp8x2_e4m3(__maca_fp8_e4m3 x, __maca_fp8_e4m3 y) {
    __maca_fp8x2_e4m3 result;
    result.__x = (x.__x << 8) | y.__x;
    return result;
}
__host__ __device__ __maca_fp8x4_e4m3 make___maca_fp8x4_e4m3(__maca_fp8_e4m3 x, __maca_fp8_e4m3 y, __maca_fp8_e4m3 z, __maca_fp8_e4m3 w) {
    __maca_fp8x4_e4m3 result;
    result.__x = (x.__x << 24) | (y.__x << 16) | (z.__x << 8) | w.__x;
    return result;
}
    )";
  }
}

#endif  // TVM_TARGET_SOURCE_LITERAL_MACA_HALF_T_H_
