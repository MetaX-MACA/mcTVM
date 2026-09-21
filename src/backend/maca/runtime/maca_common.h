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
 * \file maca_common.h
 * \brief Common utilities for MACA
 */
#ifndef TVM_BACKEND_MACA_RUNTIME_MACA_COMMON_H_
#define TVM_BACKEND_MACA_RUNTIME_MACA_COMMON_H_

#include <mcr/mc_runtime_api.h>
// #include <mcr/mc_version.h>
#include <tvm/ffi/function.h>

#include <string>

#include "../../../runtime/workspace_pool.h"

namespace tvm {
namespace runtime {

#define MACA_DRIVER_CALL(x)                                                        \
  {                                                                                \
    mcError_t result = x;                                                          \
    if (result != mcSuccess && result != mcErrorDeinitialized) {                   \
      TVM_FFI_THROW(InternalError)                                                 \
          << "MACA Error: " #x " failed with error: " << mcGetErrorString(result); \
    }                                                                              \
  }

#define MACA_CALL(func)                                                \
  {                                                                    \
    mcError_t e = (func);                                              \
    TVM_FFI_ICHECK(e == mcSuccess) << "MACA: " << mcGetErrorString(e); \
  }

/*! \brief Thread local workspace */
class MACAThreadEntry {
 public:
  /*! \brief The maca stream */
  mcStream_t stream{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  MACAThreadEntry();
  // get the threadlocal workspace
  static MACAThreadEntry* ThreadLocal();
};

/*!
 * \brief RAII guard that preserves the current MACA device.
 *
 * Switches to the requested device during the guard's lifetime and restores
 * the previous device when it goes out of scope.
 */
class MACADeviceGuard {
 public:
  explicit MACADeviceGuard(int device_id) : previous_device_(-1), changed_(false) {
    if (mcGetDevice(&previous_device_) != mcSuccess) {
      return;
    }

    if (previous_device_ != device_id) {
      if (mcSetDevice(device_id) == mcSuccess) {
        changed_ = true;
      }
    }
  }

  ~MACADeviceGuard() noexcept {
    if (changed_) {
      (void)mcSetDevice(previous_device_);
    }
  }

 private:
  int previous_device_;
  bool changed_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_BACKEND_MACA_RUNTIME_MACA_COMMON_H_
