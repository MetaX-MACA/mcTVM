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
 * \file maca_fallback_module.h
 * \brief Codegen-facing MACA module factory.
 */
#ifndef TVM_BACKEND_MACA_CODEGEN_MACA_FALLBACK_MODULE_H_
#define TVM_BACKEND_MACA_CODEGEN_MACA_FALLBACK_MODULE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <utility>

#include "../../../runtime/metadata.h"
#include "../../../support/env.h"

namespace tvm {
namespace target {

ffi::Module MACAFallbackModuleCreate(ffi::Bytes data, ffi::String fmt,
                                     ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                     ffi::String maca_source);

inline ffi::Module MACAModuleCreateWithFallback(ffi::Bytes data, ffi::String fmt,
                                                ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                                ffi::String maca_source) {
  if (tvm::support::GetEnv<bool>("TVM_COMPILE_FORCE_FALLBACK", false)) {
    return MACAFallbackModuleCreate(std::move(data), std::move(fmt), std::move(fmap),
                                    std::move(maca_source));
  }
  auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.maca");
  if (fcreate.has_value()) {
    return (*fcreate)(data, fmt, fmap, maca_source).cast<ffi::Module>();
  }
  return MACAFallbackModuleCreate(std::move(data), std::move(fmt), std::move(fmap),
                                  std::move(maca_source));
}

}  // namespace target
}  // namespace tvm

#endif  // TVM_BACKEND_MACA_CODEGEN_MACA_FALLBACK_MODULE_H_
