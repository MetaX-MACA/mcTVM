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
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/support/env.h>
#include <tvm/target/target.h>

#include <utility>

#include "../../../runtime/metadata.h"

namespace tvm {
namespace target {

ffi::Module MACAFallbackModuleCreate(ffi::Bytes data, ffi::String fmt,
                                     ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                     ffi::String maca_source, ffi::String backend_config);

inline ffi::String MACABackendConfig(const Target& target) {
  namespace json = tvm::ffi::json;
  json::Object maca_config;
  maca_config.Set("arch", target->GetAttr<ffi::String>("mcpu").value_or(ffi::String()));
  json::Object config;
  config.Set("maca", std::move(maca_config));
  return ffi::String(json::Stringify(config));
}

inline ffi::Module MACAModuleCreateWithFallback(ffi::Bytes source, const Target& target,
                                                ffi::Map<ffi::String, runtime::FunctionInfo> fmap) {
  ffi::String source_text(source.data(), source.size());
  ffi::String config = MACABackendConfig(target);
  if (tvm::support::GetEnv<bool>("TVM_COMPILE_FORCE_FALLBACK", false)) {
    return MACAFallbackModuleCreate(std::move(source), "maca", std::move(fmap),
                                    std::move(source_text), std::move(config));
  }
  auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.maca");
  if (!fcreate.has_value()) {
    return MACAFallbackModuleCreate(std::move(source), "maca", std::move(fmap),
                                    std::move(source_text), std::move(config));
  }

  return (*fcreate)(std::move(source), "maca", std::move(fmap), std::move(source_text),
                    std::move(config))
      .cast<ffi::Module>();
}

}  // namespace target
}  // namespace tvm

#endif  // TVM_BACKEND_MACA_CODEGEN_MACA_FALLBACK_MODULE_H_
