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
#include "mcblas_utils.h"

#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include "../../maca/maca_common.h"

namespace tvm {
namespace contrib {

McBlasThreadEntry::McBlasThreadEntry() { CHECK_MCBLAS_ERROR(mcblasCreate(&handle)); }

McBlasThreadEntry::~McBlasThreadEntry() {
  if (handle) {
    mcblasDestroy(handle);
    handle = nullptr;
  }
}

typedef dmlc::ThreadLocalStore<McBlasThreadEntry> McBlasThreadStore;

McBlasThreadEntry* McBlasThreadEntry::ThreadLocal() {
  auto stream = runtime::MACAThreadEntry::ThreadLocal()->stream;
  McBlasThreadEntry* retval = McBlasThreadStore::Get();
  CHECK_MCBLAS_ERROR(mcblasSetStream(retval->handle, static_cast<mcStream_t>(stream)));
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

typedef dmlc::ThreadLocalStore<McBlasLtThreadEntry> McBlasLtThreadStore;

McBlasLtThreadEntry* McBlasLtThreadEntry::ThreadLocal() { return McBlasLtThreadStore::Get(); }

}  // namespace contrib
}  // namespace tvm
