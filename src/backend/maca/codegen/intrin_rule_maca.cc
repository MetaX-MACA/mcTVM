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
#include <tvm/backend/maca/op_attr_types.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op_attr_types.h>

#include "../../../target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
// Add float suffix to the intrinsics, MACA fast math.
using tirx::FLowerIntrinsic;

template <size_t N>
Type InferTypeReturnArgType(const CallNode* call) {
  TVM_FFI_CHECK_GT(call->args.size(), N, ValueError)
      << "Return type inference requires argument " << N;
  return call->args[N]->ty;
}

void RegisterMACAShuffleOp(const char* op_name, const char* symbol, const char* lane_arg) {
  OpDef(op_name)
      .signature(sig::arg<IntExpr>("mask"), sig::arg("value"), sig::arg<IntExpr>(lane_arg),
                 sig::arg<IntExpr>("width"))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<1>>())
      .set_attr<TGlobalSymbol>(tvm::tirx::op_attr::kGlobalSymbol, symbol)
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<bool>(tvm::backend::maca::op_attr::kNeedWarpShuffle, true);
}

struct MACAMath {
  std::string operator()(const PrimType& t, std::string name) const {
    if (t.MatchesCode(DLDataTypeCode::kDLFloat)) {
      switch (t.bits()) {
        case 64:
          // Use nearbyint (ties-to-even) for round to match constant-folding semantics.
          if (name == "round") return "nearbyint";
          return name;
        case 32:
          if (name == "round") return "nearbyintf";
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
    if (orig_op.same_as(tirx::gpu_warp_shuffle_op())) {
      static const Op& maca_shfl_sync_op = Op::Get("tirx.maca.__shfl_sync");
      return maca_shfl_sync_op;
    } else if (orig_op.same_as(tirx::gpu_warp_shuffle_up_op())) {
      static const Op& maca_shfl_up_sync_op = Op::Get("tirx.maca.__shfl_up_sync");
      return maca_shfl_up_sync_op;
    } else if (orig_op.same_as(tirx::gpu_warp_shuffle_down_op())) {
      static const Op& maca_shfl_down_sync_op = Op::Get("tirx.maca.__shfl_down_sync");
      return maca_shfl_down_sync_op;
    } else {
      static const Op& maca_shfl_xor_sync_op = Op::Get("tirx.maca.__shfl_xor_sync");
      TVM_FFI_ICHECK(orig_op.same_as(tirx::gpu_warp_shuffle_xor_op()));
      return maca_shfl_xor_sync_op;
    }
  }
};

static PrimExpr DispatchMACAWarpActiveMask(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  static const Op& maca_active_mask_op = Op::Get("tirx.maca.__activemask");
  ffi::Array<PrimExpr> args = call->args.as_or_throw<ffi::Array<PrimExpr>>();
  return Call(e.ty(), maca_active_mask_op, args).as_or_throw<PrimExpr>();
}

template <typename T>
static PrimExpr DispatchMACAShuffle(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  ffi::Array<PrimExpr> maca_args{
      call->args[0].as_or_throw<PrimExpr>(), call->args[1].as_or_throw<PrimExpr>(),
      call->args[2].as_or_throw<PrimExpr>(), call->args[3].as_or_throw<PrimExpr>()};
  return Call(e.ty(), T()(e.ty(), call->op.as_or_throw<Op>()), maca_args).as_or_throw<PrimExpr>();
}

void RegisterMACAIntrinRules() {
  // clang-format off
OpDef("prim.clz").set_attr<FLowerIntrinsic>(
    "maca.FLowerIntrinsic", DispatchPureExtern<MACAMath, /*dtype_from_arg=*/true>);

OpDef("prim.floor")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.ceil")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.trunc")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.fabs")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.round")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.nearbyint")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.exp")
    .set_attr<FLowerIntrinsic>("maca.fastmath.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>)
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.exp2")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.exp10")
    .set_attr<FLowerIntrinsic>("maca.fastmath.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>)
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.erf").set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic",
                                                     DispatchPureExtern<MACAMath>);

OpDef("prim.log")
    .set_attr<FLowerIntrinsic>("maca.fastmath.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>)
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.log2")
    .set_attr<FLowerIntrinsic>("maca.fastmath.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>)
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.log10")
    .set_attr<FLowerIntrinsic>("maca.fastmath.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>)
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.tan")
    // Now the fast math version of tan and the default version of tan are same.
    .set_attr<FLowerIntrinsic>("maca.fastmath.FLowerIntrinsic",
                               DispatchPureExtern<MACAFastMathTan>)
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.cos")
    .set_attr<FLowerIntrinsic>("maca.fastmath.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>)
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.cosh")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.sin")
    .set_attr<FLowerIntrinsic>("maca.fastmath.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>)
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.sinh")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.atan")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.tanh")
    .set_attr<FLowerIntrinsic>("maca.fastmath.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>)
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.sqrt")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.pow")
    .set_attr<FLowerIntrinsic>("maca.fastmath.FLowerIntrinsic", DispatchPureExtern<MACAFastMath>)
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.rsqrt")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

OpDef("prim.popcount")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAPopcount>);

OpDef("tirx.gpu_warp_shuffle")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchMACAShuffle<MACAWarpIntrinsic>);

OpDef("tirx.gpu_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchMACAShuffle<MACAWarpIntrinsic>);

OpDef("tirx.gpu_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchMACAShuffle<MACAWarpIntrinsic>);

OpDef("tirx.gpu_warp_shuffle_xor")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchMACAShuffle<MACAWarpIntrinsic>);

OpDef("tirx.gpu_warp_activemask")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchMACAWarpActiveMask);

OpDef("prim.fmod")
    .set_attr<FLowerIntrinsic>("maca.FLowerIntrinsic", DispatchPureExtern<MACAMath>);

RegisterMACAShuffleOp("tirx.maca.__shfl_sync", "__shfl_sync", "lane");
RegisterMACAShuffleOp("tirx.maca.__shfl_up_sync", "__shfl_up_sync", "delta");
RegisterMACAShuffleOp("tirx.maca.__shfl_down_sync", "__shfl_down_sync", "delta");
RegisterMACAShuffleOp("tirx.maca.__shfl_xor_sync", "__shfl_xor_sync", "lane_mask");

OpDef("tirx.maca.__activemask")
    .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::UInt(64))
    .set_attr<TGlobalSymbol>(tvm::tirx::op_attr::kGlobalSymbol, "__activemask")
    .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                               static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<bool>(tvm::backend::maca::op_attr::kNeedWarpShuffle, true);
  // clang-format on
}

TVM_FFI_STATIC_INIT_BLOCK() { RegisterMACAIntrinRules(); }

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
