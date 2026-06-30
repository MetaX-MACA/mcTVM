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
 * \file intrin_rule_maca.cc
 * \brief MACA intrinsic rules.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

#include "../../../target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
// Add float suffix to the intrinsics, MACA fast math.
using tirx::FLowerIntrinsic;

struct MACAMath {
  std::string operator()(const PrimType& t, std::string name) const {
    if (t.MatchesCode(DLDataTypeCode::kDLFloat)) {
      switch (t.bits()) {
        case 64:
          return name;
        case 32:
          return name + 'f';
        case 16: {
          if (name == "fabs") {
            return "__habs";
          } else if (name == "round") {
            return "hrint";
          } else {
            return "h" + name;
          }
        }
        default:
          return "";
      }
    } else if (t.MatchesCode(DLDataTypeCode::kDLBfloat)) {
      return 'h' + name;
    } else if (t.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
      switch (t.bits()) {
        case 32:
          return "__" + name;
        case 64:
          return "__" + name + "ll";
        default:
          return "";
      }
    }
    return "";
  }
};

struct MACAFastMath : public MACAMath {
  std::string operator()(const PrimType& t, std::string name) const {
    if (t.MatchesCode(DLDataTypeCode::kDLFloat) && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return MACAMath::operator()(t, name);
    }
    return "";
  }
};

struct MACAFastMathTan : public MACAMath {
  std::string operator()(const PrimType& t, std::string name) const {
    if (t.MatchesCode(DLDataTypeCode::kDLFloat)) {
      switch (t.bits()) {
        case 64:
          return name;
        // `__tanf` seems to produce some values too deviant from numpy tan version.
        // So, let's use just `tanf` instead.
        case 32:
          return name + 'f';
        case 16:
          return 'h' + name;
        default:
          return "";
      }
    }
    return "";
  }
};

struct MACAPopcount {
  std::string operator()(const PrimType& t, std::string name) const {
    if (t.MatchesCode(DLDataTypeCode::kDLUInt)) {
      switch (t.bits()) {
        case 32:
          return "__popc";
        case 64:
          return "__popcll";
        default:
          return "";
      }
    }
    return "";
  }
};

struct MACAWarpIntrinsic {
  const Op operator()(const PrimType& t, const Op& orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      static const Op& maca_shfl_sync_op = Op::Get("tirx.maca.__shfl_sync");
      return maca_shfl_sync_op;
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      static const Op& maca_shfl_up_sync_op = Op::Get("tirx.maca.__shfl_up_sync");
      return maca_shfl_up_sync_op;
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_down())) {
      static const Op& maca_shfl_down_sync_op = Op::Get("tirx.maca.__shfl_down_sync");
      return maca_shfl_down_sync_op;
    } else {
      static const Op& maca_shfl_xor_sync_op = Op::Get("tirx.maca.__shfl_xor_sync");
      TVM_FFI_ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_xor()));
      return maca_shfl_xor_sync_op;
    }
  }
};

static PrimExpr DispatchMACAWarpActiveMask(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  static const Op& maca_active_mask_op = Op::Get("tirx.maca.__activemask");
  return Call(e.ty(), maca_active_mask_op, call->args);
}

template <typename T>
static PrimExpr DispatchMACAShuffle(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  ffi::Array<PrimExpr> maca_args{{call->args[0], call->args[1], call->args[2], call->args[3]}};
  return Call(e.ty(), T()(e.ty(), call->op.as_or_throw<Op>()), maca_args);
}

void RegisterMACAIntrinRules() {
  // clang-format off
TVM_REGISTER_OP("tirx.clz").set_attr<FLowerIntrinsic>(
    "maca.FLowerIntrinsic", DispatchPureExtern<MACAMath, /*dtype_from_arg=*/true>);

TVM_REGISTER_OP("tirx.floor")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.ceil")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.trunc")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.fabs")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.round")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.nearbyint")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.exp").set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic",
                                                     DispatchPureExtern<MACAFastMath>);

TVM_REGISTER_OP("tirx.exp2")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.exp10")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>);

TVM_REGISTER_OP("tirx.erf").set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic",
                                                     DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.log").set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic",
                                                     DispatchPureExtern<MACAFastMath>);

TVM_REGISTER_OP("tirx.log2")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>);

TVM_REGISTER_OP("tirx.log10")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>);

TVM_REGISTER_OP("tirx.tan").set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic",
                                                     DispatchPureExtern<MACAFastMathTan>);

TVM_REGISTER_OP("tirx.cos").set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic",
                                                     DispatchPureExtern<MACAFastMath>);

TVM_REGISTER_OP("tirx.cosh")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.sin").set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic",
                                                     DispatchPureExtern<MACAFastMath>);

TVM_REGISTER_OP("tirx.sinh")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.atan")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.tanh")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.sqrt")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.pow").set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic",
                                                     DispatchPureExtern<MACAMath>);

TVM_REGISTER_OP("tirx.popcount")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAPopcount>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchMACAShuffle<MACAWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchMACAShuffle<MACAWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchMACAShuffle<MACAWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle_xor")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchMACAShuffle<MACAWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_activemask")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchMACAWarpActiveMask);

TVM_REGISTER_OP("tirx.fmod")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

// Register low-level builtin ops.
// TODO(tvm-team): consider make MACA its own subfolder and create a file for low-level builtins.
TVM_REGISTER_OP("tirx.maca.__shfl_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("lane", "Expr", "The source thread id.")
    .add_argument("width", "Expr", "The warp thread width, must be a power of 2.")
    .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"), 10)
    .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("maca"),
                                               10)
    .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName", ffi::String("maca.__shfl_sync"),
                                        10)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<bool>("maca.need_warp_shuffle", true);

TVM_REGISTER_OP("tirx.maca.__shfl_up_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr", "The source lane id offset to be added.")
    .add_argument("width", "Expr", "The warp thread width, must be a power of 2.")
    .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"), 10)
    .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("maca"),
                                               10)
    .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName", ffi::String("maca.__shfl_up_sync"),
                                        10)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_up_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<bool>("maca.need_warp_shuffle", true);

TVM_REGISTER_OP("tirx.maca.__shfl_down_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr", "The source lane id offset to be subtracted.")
    .add_argument("width", "Expr", "The warp thread width, must be a power of 2.")
    .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"), 10)
    .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("maca"),
                                               10)
    .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName",
                                        ffi::String("maca.__shfl_down_sync"), 10)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_down_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<bool>("maca.need_warp_shuffle", true);

TVM_REGISTER_OP("tirx.maca.__shfl_xor_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("lane_mask", "Expr", "The lane mask.")
    .add_argument("width", "Expr", "The warp thread width, must be a power of 2.")
    .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"), 10)
    .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("maca"),
                                               10)
    .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName", ffi::String("maca.__shfl_xor_sync"),
                                        10)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_xor_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<bool>("maca.need_warp_shuffle", true);

TVM_REGISTER_OP("tirx.maca.__activemask")
    .set_num_inputs(0)
    .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"), 10)
    .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("maca"),
                                               10)
    .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName", ffi::String("maca.__activemask"),
                                        10)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__activemask")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<bool>("maca.need_warp_shuffle", true);
  // clang-format on
}

TVM_FFI_STATIC_INIT_BLOCK() { RegisterMACAIntrinRules(); }

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
