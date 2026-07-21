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
 * \file Use external mcblas utils function
 */
#include "mcblas_utils.h"

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "../../../../backend/maca/runtime/maca_common.h"

namespace tvm {
namespace contrib {

McBlasThreadEntry::McBlasThreadEntry() { CHECK_MCBLAS_ERROR(mcblasCreate(&handle)); }

McBlasThreadEntry::~McBlasThreadEntry() {
  if (handle) {
    mcblasDestroy(handle);
    handle = nullptr;
  }
}

McBlasThreadEntry* McBlasThreadEntry::ThreadLocal(DLDevice curr_device) {
  static thread_local McBlasThreadEntry inst;
  McBlasThreadEntry* retval = &inst;
  mcStream_t stream =
      static_cast<mcStream_t>(TVMFFIEnvGetStream(curr_device.device_type, curr_device.device_id));
  CHECK_MCBLAS_ERROR(mcblasSetStream(retval->handle, stream));
  return retval;
}

McBlasLtThreadEntry::McBlasLtThreadEntry() {
  CHECK_MCBLAS_ERROR(mcblasLtCreate(&handle));
  CHECK_MCBLAS_ERROR(mcblasLtMatmulPreferenceCreate(&matmul_pref_desc));
  MACA_CALL(mcMalloc(&workspace_ptr, workspace_size));
}

McBlasLtThreadEntry::~McBlasLtThreadEntry() {
  if (handle) {
    mcblasLtDestroy(handle);
    handle = nullptr;
  }
  if (matmul_pref_desc) {
    mcblasLtMatmulPreferenceDestroy(matmul_pref_desc);
    matmul_pref_desc = nullptr;
  }
  if (workspace_ptr != nullptr) {
    mcFree(workspace_ptr);
    workspace_ptr = nullptr;
  }
}

McBlasLtThreadEntry* McBlasLtThreadEntry::ThreadLocal(DLDevice curr_device) {
  static thread_local McBlasLtThreadEntry inst;
  return &inst;
}

}  // namespace contrib
}  // namespace tvm
