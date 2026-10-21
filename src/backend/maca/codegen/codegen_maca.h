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
 * \file codegen_maca.h
 * \brief Utility to generate maca code
 */
#ifndef TVM_BACKEND_MACA_CODEGEN_CODEGEN_MACA_H_
#define TVM_BACKEND_MACA_CODEGEN_CODEGEN_MACA_H_

#include <tvm/ffi/error.h>
#include <tvm/backend/maca/op_attr_types.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/target/codegen.h>
#include <tvm/ir/op.h>

#include <array>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../target/source/codegen_c.h"

namespace tvm {
namespace codegen {
namespace maca {

class CodeGenMACA final : public CodeGenC {
 public:
  CodeGenMACA();
  void Init(bool output_ssa);
  std::string Finish();
  bool need_include_path() {
    return (enable_fp16_ || enable_bf16_ || enable_int8_ || enable_fp8_ || need_math_constants_h_ ||
            need_mma_h_);
  }
  // override behavior
  void PrintFunctionSignature(const ffi::String& function_name, const Function& func,
                              std::ostream& os) final;
  void InitFuncState(const Function& func) final;
  void PrintStorageSync(const CallNode* op) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  using CodeGenC::PrintType;
  void PrintVecBinaryOp(const std::string& op, const PrimType& t, PrimExpr lhs, PrimExpr rhs,
                        std::ostream& os) final;              // NOLINT(*)
  void PrintType(const PrimType& t, std::ostream& os) final;  // NOLINT(*)
  void PrintVecConstructor(const PrimType& t, std::ostream& os) final;
  void PrintVecElemLoad(const std::string& vec, const PrimType& t, int i,
                        std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(const std::string& vec, const PrimType& t, int i,
                         const std::string& value) final;
  void BindThreadIndex(const PrimVar& var, const ffi::String& thread_tag) final;  // NOLINT(*)
  void PrintVecElemLoadExpr(const PrimType& t, int i, const std::string& value,
                            std::ostream& os) final;
  std::string CastFromTo(std::string value, const PrimType& from, const PrimType& target) final;
  void AddUtilFunction(const std::string& name, const std::string& code);
  // overload visitor
  void Dispatch_(const prim::RampNode* op, std::ostream& os) final;       // NOLINT(*)
  void Dispatch_(const prim::SelectNode* op, std::ostream& os) final;     // NOLINT(*)
  void Dispatch_(const prim::BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void Dispatch_(const FloatImmNode* op, std::ostream& os) final;
  void Dispatch_(const CallNode* op, std::ostream& os) final;
  void Dispatch_(const prim::CastNode* op, std::ostream& os) final;
  void Dispatch_(const EvaluateNode* op) final;
  void Dispatch_(const BindNode* op) final;
  void Dispatch_(const ReturnNode* op) final;
  void Dispatch_(const ForNode* op) final;

 protected:
  void PrintExtraAttrs(const Function& f, std::ostream& os) final;  // NOLINT(*)
  void PrintCallExtern(Type ret_type, ffi::String global_symbol, const ffi::Array<Expr>& args,
                       bool skip_first_arg, std::ostream& os) final;  // NOLINT(*)

 private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string& value, const TensorLoadNode* op,
                           std::ostream& os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;
  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable bf16
  bool enable_bf16_{false};
  // whether enable fp8
  bool enable_fp8_{false};
  // whether enable int8
  bool enable_int8_{false};
  // whether enable warp shuffle intrinsics
  bool enable_warp_shuffle_{false};
  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // Whether the current function is a __global__ kernel entry rather than a
  // __device__ subroutine.
  bool in_kernel_launch_{false};
  std::array<PrimExpr, 3> launch_dimensions_{IntImm::Int32(1), IntImm::Int32(1),
                                             IntImm::Int32(1)};
  std::optional<int64_t> min_blocks_per_sm_;
  // Codegen tags
  std::unordered_set<std::string> codegen_tags_;
  // Op attribute map
  OpAttrMap<bool> op_need_warp_shuffle_ =
      Op::GetAttrMap<bool>(tvm::backend::maca::op_attr::kNeedWarpShuffle);

  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;
  // Functions to be added to the util functions during codegen
  std::unordered_map<std::string, std::string> util_funcs_;

  std::unordered_map<const VarNode*, std::string> fragment_shapes;
  std::unordered_map<const VarNode*, std::string> fragment_layouts;
  friend void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenMACA* p);
  void PrintWmmaScope(const std::string& scope, const PrimType& t, const VarNode* variable,
                      std::ostream& os);
  int32_t GetWmmaFragmentSize(const std::string& scope, const VarNode* variable, int32_t size);

  // cp async
  std::queue<std::string> cp_async_var_names;
  std::queue<size_t> cp_async_remain_nums;
  std::unordered_map<int, int> mcDummyRetNum = {{4, 0}, {8, 0}, {16, 0}};
  // shared buffer ailgnments, {"shd var name": aligns}
  std::unordered_map<ffi::String, uint32_t> shd_aligns;
};
}  // namespace maca
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_BACKEND_MACA_CODEGEN_CODEGEN_MACA_H_
