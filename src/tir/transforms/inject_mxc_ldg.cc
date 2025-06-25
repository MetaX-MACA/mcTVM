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

#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/const_fold.h"
#include "../../arith/pattern_match.h"

namespace tvm {
namespace tir {

static bool is_const_boardcast(const PrimExpr& x, double value) {
  const auto* op = x.as<BroadcastNode>();
  if (op) {
    const IntImmNode* int_value = op->value.as<IntImmNode>();
    const FloatImmNode* float_value = op->value.as<FloatImmNode>();
    return (int_value && (double)int_value->value == value) ||
           (float_value && float_value->value == value);
  }
  return false;
}

class MXCLDGRewriter : public StmtMutator {
 public:
  Stmt VisitStmt_(const BufferStoreNode* store) final {
    Stmt result = StmtMutator::VisitStmt_(store);
    Buffer load_buffer = store->buffer;
    PrimExpr load_value = store->value;
    const CallNode* call = load_value.as<CallNode>();
    if (call != nullptr) {
      const OpNode* op = call->op.as<OpNode>();
      if (op != nullptr && op->name == "tir.if_then_else") {
        const PrimExpr& predicate = call->args[0];
        const PrimExpr& lhs = call->args[1];
        const PrimExpr& rhs = call->args[2];
        PrimExpr global_addr, shared_addr;
        const BufferLoadNode* load = lhs.as<BufferLoadNode>();
        PrimExpr imm_value = rhs;
        PrimExpr pred_exp = IntImm(DataType::Int(32), 1);
        if (load == nullptr) {
          load = rhs.as<BufferLoadNode>();
          imm_value = lhs;
          pred_exp = IntImm(DataType::Int(32), 0);
          if (load == nullptr) {
            return result;
          }
        }
        auto new_load = make_object<BufferLoadNode>(*load);

        // imm_value must be constant, and its value must be zero
        if (!is_const_int(imm_value, 0) && !is_const_float(imm_value, 0) &&
            !is_const_boardcast(imm_value, 0)) {
          VLOG(0) << "imm value is not constant: " << imm_value;
          return result;
        }
        // // store buffer must be shared memory
        // String data_scope =
        // (Downcast<PointerType>(load_buffer->data->type_annotation))->storage_scope; if
        // (data_scope != "shared.dyn") {
        //   LOG(INFO) << "storage scope should be shared memory";
        //   return result;
        // }

        uint32_t total_bits = store->buffer->dtype.bits();
        global_addr = load->indices[0];
        shared_addr = store->indices[0];

        // in case vectorize
        const RampNode* ramp = global_addr.as<RampNode>();
        if (ramp != nullptr) {
          const RampNode* store_indices = shared_addr.as<RampNode>();
          // FIXME: it's ok if the size of store buffer is same as load, no need to be a vector too
          if (store_indices == nullptr) {
            return result;
          }
          int store_stride = static_cast<int>(Downcast<IntImm>(store_indices->stride)->value);
          int store_lanes = static_cast<int>(Downcast<IntImm>(store_indices->lanes)->value);
          int load_stride = static_cast<int>(Downcast<IntImm>(ramp->stride)->value);
          int load_lanes = static_cast<int>(Downcast<IntImm>(ramp->lanes)->value);
          total_bits = store->buffer->dtype.bits() * store_lanes;
          uint32_t load_bits = load->buffer->dtype.bits() * load_lanes;
          // memories of load and store must be continuous and have then same size
          if (store_stride != 1 || load_stride != 1 || total_bits != load_bits) {
            return result;
          }
          global_addr = ramp->base;
          shared_addr = store_indices->base;
        }

        // TODO: support more ldgs
        switch (total_bits) {
          case 16:
          case 32:
          case 64:
          case 128:
            VLOG(0) << "convert if_then_else to mxc_ldg_predicator of " << total_bits << " bits";
            return Evaluate(Call(store->buffer->dtype, tvm::tir::builtin::mxc_ldg_predicator(),
                                 {store->buffer->data, shared_addr, load->buffer->data, global_addr,
                                  predicate, pred_exp, IntImm(DataType::Int(32), total_bits)}));
          default:
            VLOG(0) << "do not support " << total_bits << " bits for now";
            break;
        }
        return result;
      }
    }
    return result;
  }
};

namespace transform {

Pass InjectMXCLDG() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (target.defined() && target.value()->kind->name == "maca") {
      auto* n = f.CopyOnWrite();
      n->body = MXCLDGRewriter()(n->body);
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectMXCLDG", {});
}

// The pass can now be invoked via the pass infrastructure, but we also add a
// Python binding for it
TVM_REGISTER_GLOBAL("tir.transform.InjectMXCLDG").set_body_typed(InjectMXCLDG);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
