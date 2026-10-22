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
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/runtime/base.h>

#include <string>

namespace tvm {
namespace tirx {
namespace builtin {

namespace {
Type InferTypeMacaAxis(const CallNode* call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError);
  const auto* axis = call->args[0].as<StringImmNode>();
  TVM_FFI_CHECK(axis && (axis->value == "x" || axis->value == "y" || axis->value == "z"),
                ValueError)
      << "MACA index axis must be x, y, or z";
  return PrimType::Int(32);
}

template <size_t N>
Type InferTypeReturnArgType(const CallNode* call) {
  TVM_FFI_CHECK_GT(call->args.size(), N, ValueError)
      << "Return type inference requires argument " << N;
  return call->args[N]->ty;
}

void RegisterDeviceIntrinsicAttrs(OpDef& def, CallEffectKind effect_kind) {
  def.set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind, static_cast<int64_t>(effect_kind));
}

template <typename... Specs>
OpDef& RegisterDeviceIntrinsic(OpDef&& def, CallEffectKind effect_kind, const Specs&... specs) {
  def.signature(specs...);
  RegisterDeviceIntrinsicAttrs(def, effect_kind);
  return def;
}

void RegisterDeviceIntrinsics() {
  for (const char* name : {"block_idx", "thread_idx", "grid_dim", "block_dim"}) {
    RegisterDeviceIntrinsic(OpDef(std::string("tirx.maca.") + name), CallEffectKind::kPure,
                            sig::arg<StringImm>("axis"))
        .set_attr<FInferType>(tvm::op_attr::kInferType,
                              FInferType::FromNative<&InferTypeMacaAxis>());
  }
  for (const char* name : {"linear_thread_id", "warp_id", "lane_id", "warpgroup_id",
                           "warp_in_warpgroup", "thread_in_warpgroup"}) {
    RegisterDeviceIntrinsic(OpDef(std::string("tirx.maca.") + name), CallEffectKind::kPure)
        .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Int(32));
  }

  RegisterDeviceIntrinsic(OpDef("tirx.maca.func_call"), CallEffectKind::kOpaque,
                          sig::arg<StringImm>("func_name"), sig::var_args("args"));
  RegisterDeviceIntrinsic(OpDef("tirx.maca.warp_reduce"), CallEffectKind::kOpaque,
                          sig::arg("value"), sig::arg<StringImm>("op"), sig::arg<IntExpr>("width"))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.cta_reduce"), CallEffectKind::kOpaque, sig::arg("value"),
                          sig::arg<StringImm>("op"), sig::arg<IntExpr>("num_waves"),
                          sig::arg("scratch"))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.thread_fence"), CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.atomic_add"), CallEffectKind::kOpaque,
                          sig::arg("address"), sig::arg("value"))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<1>>());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.atomic_cas"), CallEffectKind::kOpaque,
                          sig::arg("address"), sig::arg("compare"), sig::arg("value"))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<1>>());
  // ldg has both a value-returning form and a vector-store form.
  RegisterDeviceIntrinsic(OpDef("tirx.maca.ldg"), CallEffectKind::kOpaque, sig::var_args("args"));
  RegisterDeviceIntrinsic(OpDef("tirx.maca.warp_sync"), CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.cta_sync"), CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.copy_bytes"), CallEffectKind::kOpaque, sig::arg("dst"),
                          sig::arg("src"), sig::arg<IntExpr>("num_bytes"))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.copy_async_32b"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.copy_async_64b"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.copy_async_128b"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.copy_async_32b_zfill"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg("predicate"))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.copy_async_64b_zfill"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg("predicate"))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.copy_async_128b_zfill"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg("predicate"))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.async_wait_gvmcnt"), CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("count"))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.maca.barrier_inst"), CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());

  for (const char* name :
       {"mma_m16n16k16_f16_f32", "mma_m16n16k16_bf16_f32", "mma_m16n16k16_i8_i32",
        "mma_m16n16k4_f32_f32", "mma_m16n16k8_tf32_f32", "mma_m16n16k4_f64_f64"}) {
    RegisterDeviceIntrinsic(OpDef(std::string("tirx.maca.") + name), CallEffectKind::kOpaque,
                            sig::arg("d"), sig::arg("a"), sig::arg("b"), sig::arg("c"))
        .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void());
  }
}

}  // namespace

void RegisterMACATargetBuiltins() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  RegisterDeviceIntrinsics();
}

TVM_FFI_STATIC_INIT_BLOCK() { RegisterMACATargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
