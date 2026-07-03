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
 * \file target_kind.cc
 * \brief MACA compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include <cctype>
#include <string>

namespace tvm {
namespace backend {
namespace maca {

void CheckOrSetAttr(ffi::Map<ffi::String, ffi::Any>* attrs, const ffi::String& name,
                    const ffi::String& value) {
  auto iter = attrs->find(name);
  if (iter == attrs->end()) {
    attrs->Set(name, value);
  } else {
    auto str = (*iter).second.try_cast<ffi::String>();
    TVM_FFI_CHECK(str && str.value() == value, ValueError)
        << "Expects \"" << name << "\" to be \"" << value << "\", but gets: " << (*iter).second;
  }
}

std::string ExtractStringWithPrefix(const std::string& str, const std::string& prefix) {
  if (str.find(prefix) != 0) return "";
  std::size_t pos = prefix.length();
  while (pos < str.length() && (std::isdigit(str[pos]) || std::isalpha(str[pos]))) {
    ++pos;
  }
  return str.substr(prefix.length(), pos - prefix.length());
}

ffi::Map<ffi::String, ffi::Any> UpdateMACAAttrs(ffi::Map<ffi::String, ffi::Any> target) {
  CheckOrSetAttr(&target, "mtriple", "mxc-metax-macahca");
  std::string arch = "xcore1000";
  if (target.count("mcpu")) {
    ffi::String mcpu = target.at("mcpu").as_or_throw<ffi::String>();
    arch = ExtractStringWithPrefix(mcpu, "xcore");
    TVM_FFI_CHECK(!arch.empty(), ValueError)
        << "MACA target gets an invalid XCORE version: -mcpu=" << mcpu;
  } else {
    if (auto f_get_maca_arch = tvm::ffi::Function::GetGlobal("tvm_callback_maca_get_arch")) {
      arch = (*f_get_maca_arch)().cast<std::string>();
    }
    target.Set("mcpu", ffi::String(arch));
  }

  return target;
}

void RegisterTargetKind() {
  namespace refl = tvm::ffi::reflection;

  TVM_REGISTER_TARGET_KIND("maca", kDLMACA)
      .add_attr_option<ffi::String>("mcpu")
      .add_attr_option<ffi::String>("mtriple")
      .add_attr_option<ffi::Array<ffi::String>>("mattr")
      .add_attr_option<int64_t>("max_num_threads", refl::DefaultValue(1024))
      .add_attr_option<int64_t>("max_threads_per_block", refl::DefaultValue(1024))
      .add_attr_option<int64_t>("max_shared_memory_per_block", refl::DefaultValue(65536))
      .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(64))
      .add_attr_option<int64_t>("max_local_memory_per_block", refl::DefaultValue(4095))
      .set_default_keys({"maca", "gpu"})
      .set_target_canonicalizer(UpdateMACAAttrs);
}

}  // namespace maca
}  // namespace backend
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() { tvm::backend::maca::RegisterTargetKind(); }
