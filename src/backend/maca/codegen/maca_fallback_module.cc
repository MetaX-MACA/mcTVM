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
 * \file maca_fallback_module.cc
 * \brief MACAFallbackModuleNode for codegen-time construction without runtime.
 */
#include "maca_fallback_module.h"

#include <tvm/ffi/extra/module.h>

#include <string>
#include <utility>

#include "../../../support/bytes_io.h"

namespace tvm {
namespace target {

class MACAFallbackModuleNode : public ffi::ModuleObj {
 public:
  MACAFallbackModuleNode(ffi::Bytes data, ffi::String fmt,
                         ffi::Map<ffi::String, runtime::FunctionInfo> fmap, ffi::String maca_source)
      : data_(std::move(data)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        maca_source_(std::move(maca_source)) {}

  const char* kind() const final { return "maca"; }

  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    TVM_FFI_THROW(RuntimeError)
        << "MACA runtime is not linked into this build; cannot launch kernels. "
        << "Re-link with USE_MACA=ON or load this module in a MACA-equipped environment.";
    TVM_FFI_UNREACHABLE();
  }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(data_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (format == fmt_) return ffi::String(data_.data(), data_.size());
    if (maca_source_.length() != 0) return maca_source_;
    if (fmt_ == "fatbin") return ffi::String(data_.data(), data_.size());
    return ffi::String();
  }

 private:
  ffi::Bytes data_;
  ffi::String fmt_;
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap_;
  ffi::String maca_source_;
};

ffi::Module MACAFallbackModuleCreate(ffi::Bytes data, ffi::String fmt,
                                     ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                     ffi::String maca_source) {
  auto n = ffi::make_object<MACAFallbackModuleNode>(std::move(data), std::move(fmt),
                                                    std::move(fmap), std::move(maca_source));
  return ffi::Module(n);
}

}  // namespace target
}  // namespace tvm
