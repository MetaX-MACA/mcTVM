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
 * \file Use external mcdnn utils function
 */

#ifndef TVM_RUNTIME_CONTRIB_MCBLAS_MCBLAS_UTILS_H_
#define TVM_RUNTIME_CONTRIB_MCBLAS_MCBLAS_UTILS_H_

// #include <mcblas_v2.h>
#include <mcblas/mcblas.h>
#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_api.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/logging.h>

#include <cstdint>
#include <mcblas/mcblasLt.h>
#include <optional>

namespace tvm {
namespace contrib {

inline const char* GetMcblasErrorString(int error) {
  switch (error) {
    case MCBLAS_STATUS_NOT_INITIALIZED:
      return "MCBLAS_STATUS_NOT_INITIALIZED";
    case MCBLAS_STATUS_ALLOC_FAILED:
      return "MCBLAS_STATUS_ALLOC_FAILED";
    case MCBLAS_STATUS_INVALID_VALUE:
      return "MCBLAS_STATUS_INVALID_VALUE";
    case MCBLAS_STATUS_ARCH_MISMATCH:
      return "MCBLAS_STATUS_ARCH_MISMATCH";
    case MCBLAS_STATUS_MAPPING_ERROR:
      return "MCBLAS_STATUS_MAPPING_ERROR";
    case MCBLAS_STATUS_EXECUTION_FAILED:
      return "MCBLAS_STATUS_EXECUTION_FAILED";
    case MCBLAS_STATUS_INTERNAL_ERROR:
      return "MCBLAS_STATUS_INTERNAL_ERROR";
    case MCBLAS_STATUS_NOT_SUPPORTED:
      return "MCBLAS_STATUS_NOT_SUPPORTED";
    case MCBLAS_STATUS_LICENSE_ERROR:
      return "MCBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unrecognized error";
}

#ifndef CHECK_MCBLAS_ERROR
#define CHECK_MCBLAS_ERROR(fn)                                                            \
  do {                                                                                    \
    int error = static_cast<int>(fn);                                                     \
    ICHECK_EQ(error, MCBLAS_STATUS_SUCCESS) << "MCBLAS: " << GetMcblasErrorString(error); \
  } while (0)  // ; intentionally left off.
#endif         // CHECK_MCBLAS_ERROR

struct McBlasThreadEntry {
  McBlasThreadEntry();
  ~McBlasThreadEntry();
  mcblasHandle_t handle{nullptr};
  static McBlasThreadEntry* ThreadLocal();
};  // McBlasThreadEntry

struct McBlasLtThreadEntry {
  McBlasLtThreadEntry();
  ~McBlasLtThreadEntry();

  mcblasLtHandle_t handle{nullptr};
  mcblasLtMatmulPreference_t matmul_pref_desc{nullptr};
  void* workspace_ptr{nullptr};
  static constexpr const size_t workspace_size = 33554432;

  static McBlasLtThreadEntry* ThreadLocal();
};  // McBlasLtThreadEntry

inline macaDataType_t GetMacaDataType(DLDataType type) {
  if (type.code == kDLInt) {
    switch (type.bits) {
      case 8:
        return MACA_R_8I;
      case 32:
        return MACA_R_32I;
    }
  } else if (type.code == kDLUInt) {
    switch (type.bits) {
      case 8:
        return MACA_R_8U;
      case 32:
        return MACA_R_32U;
    }
  } else if (type.code == kDLFloat) {
    switch (type.bits) {
      case 16:
        return MACA_R_16F;
      case 32:
        return MACA_R_32F;
      case 64:
        return MACA_R_64F;
    }
  }
  LOG(FATAL) << "Unsupported maca type";
}

/*! \brief Execute matrix multiply followed by the specified epilogue, using mcBLASLt. */
void CallMcblasLt(mcblasLtHandle_t hdl, mcStream_t stream,
                  mcblasLtMatmulPreference_t matmul_pref_desc, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* scaleA, const DLTensor* scaleB,
                  const DLTensor* C, bool transa, bool transb, void* workspace_ptr,
                  size_t workspace_size, mcblasLtEpilogue_t epilogue = MCBLASLT_EPILOGUE_DEFAULT,
                  std::optional<float> dq_scale = std::nullopt);

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_MCBLAS_MCBLAS_UTILS_H_
