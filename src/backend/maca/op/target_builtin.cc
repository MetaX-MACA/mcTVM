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
 * \file backend/maca/op/target_builtin.cc
 *
 *  builtin intrinsic operators specific to MACA target.
 */
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <string>

namespace tvm {
namespace tirx {
namespace builtin {

#define TIRX_DEFINE_BUILTIN_FUNC(OpName)                                           \
  OpRegEntry::RegisterOrGet("tirx." #OpName)                                       \
      .set_name()                                                                  \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String(#OpName), 1) \
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"), /*plevel=*/1)

namespace {
void RegisterDeviceIntrinsicAliases();
}

void RegisterMACATargetBuiltins() {
  // clang-format off
static bool registered = false;
if (registered) return;
registered = true;

RegisterDeviceIntrinsicAliases();
  // clang-format on
}

namespace {

struct DeviceIntrinsicRegistration {
  const char* name;
  const char* namespace_name;
  CallEffectKind effect_kind;
};

void RegisterDeviceIntrinsic(const DeviceIntrinsicRegistration& reg) {
  std::string name(reg.name);
  std::string namespace_name(reg.namespace_name);
  std::string prefix = namespace_name + "_";
  std::string suffix = name;
  if (suffix.rfind(prefix, 0) == 0) {
    suffix = suffix.substr(prefix.size());
  }

  std::string canonical_op_name = "tirx." + namespace_name + "." + suffix;
  ffi::String namespace_attr(namespace_name);
  ffi::String printer_name(namespace_name + "." + suffix);
  int64_t effect = static_cast<int64_t>(reg.effect_kind);

  auto register_one = [&](const std::string& op_name) {
    OpRegEntry::RegisterOrGet(op_name)
        .set_name()
        .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"),
                                  /*plevel=*/15)
        .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", namespace_attr,
                                             /*plevel=*/15)
        .set_attr<TCallEffectKind>("TCallEffectKind", effect, /*plevel=*/15)
        .set_attr<TScriptPrinterName>("TScriptPrinterName", printer_name, /*plevel=*/15);
  };

  register_one(canonical_op_name);
}

#define TIRX_DEVICE_INTRIN_ALIAS(OpName, Namespace, EffectKind) \
  {#OpName, #Namespace, CallEffectKind::EffectKind}

const DeviceIntrinsicRegistration kDeviceIntrinsics[] = {
    TIRX_DEVICE_INTRIN_ALIAS(maca_func_call, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_thread_fence, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_warp_sync, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_cta_sync, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_copy_bytes, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_copy_async_32b, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_copy_async_64b, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_copy_async_128b, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_copy_async_32b_zfill, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_copy_async_64b_zfill, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_copy_async_128b_zfill, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_async_wait_gvmcnt, maca, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(maca_barrier_inst, maca, kOpaque),
};

void RegisterDeviceIntrinsicAliases() {
  for (const auto& reg : kDeviceIntrinsics) {
    RegisterDeviceIntrinsic(reg);
  }
}

#undef TIRX_DEVICE_INTRIN_ALIAS

}  // namespace

#undef TIRX_DEFINE_BUILTIN_FUNC

TVM_FFI_STATIC_INIT_BLOCK() { RegisterMACATargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
