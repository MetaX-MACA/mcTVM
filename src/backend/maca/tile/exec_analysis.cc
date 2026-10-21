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

#include "exec_analysis.h"

#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

// Reuse the syntactic comparison/conjunction classifier without changing CUDA.
// MACA resolves the variables and interprets the resulting ranges itself.
#include "../../cuda/tile/filter_canonical.h"

namespace tvm::backend::maca {
using namespace tvm::tirx;
namespace {
constexpr int64_t kWaveSize = 64;
constexpr int64_t kWavesPerGroup = 4;

std::optional<int64_t> AsInt(const PrimExpr& expr) {
  if (const auto* imm = expr.as<IntImmNode>()) return imm->value.as<int64_t>();
  return std::nullopt;
}

std::optional<int64_t> Product(const LaunchParams& params, const std::string& prefix) {
  int64_t product = 1;
  for (char axis : {'x', 'y', 'z'}) {
    auto it = params.find(prefix + axis);
    if (it == params.end()) continue;
    auto value = AsInt(it->second.get<1>());
    if (!value || *value <= 0 || product > std::numeric_limits<int64_t>::max() / *value)
      return std::nullopt;
    product *= *value;
  }
  return product;
}

struct AxisRange {
  int64_t extent;
  int64_t offset = 0;
  int64_t stride = 1;

  bool Intersect(int64_t lo, int64_t hi) {
    lo = std::max(lo, offset);
    if (hi <= lo) return false;
    int64_t first = (lo - offset) / stride + ((lo - offset) % stride != 0);
    int64_t end = std::min(extent, (hi - 1 - offset) / stride + 1);
    if (first >= end) return false;
    offset += first * stride;
    extent = end - first;
    return true;
  }
  ffi::Array<PrimExpr> Encode() const {
    ffi::Array<PrimExpr> result{IntImm::Int64(extent), IntImm::Int64(offset)};
    if (stride != 1) result.push_back(IntImm::Int64(stride));
    return result;
  }
};
using ActiveSet = std::map<std::string, AxisRange>;

class MacaIndexLowerer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit MacaIndexLowerer(LaunchParams params) : params_(std::move(params)) {}

  static PrimExpr Register(const std::string& name) {
    std::string operation;
    if (name.rfind("ctaid.", 0) == 0) operation = "block_idx";
    if (name.rfind("tid.", 0) == 0) operation = "thread_idx";
    if (name.rfind("ntid.", 0) == 0) operation = "block_dim";
    if (name.rfind("nctaid.", 0) == 0) operation = "grid_dim";
    if (!operation.empty()) {
      return Call(PrimType::Int(32), Op::Get("tirx.maca." + operation),
                  {StringImm(name.substr(name.size() - 1))})
          .as_or_throw<PrimExpr>();
    }
    LOG(FATAL) << "Unsupported MACA register: " << name;
    return IntImm::Int32(0);
  }

  static Stmt Lower(const Stmt& body, const LaunchParams& params) {
    auto lowerer = ffi::make_object<MacaIndexLowerer>(params);
    for (const auto& [tag, iv] : params) {
      std::string name(tag);
      std::string reg;
      if (name.rfind("blockIdx.", 0) == 0) reg = "ctaid." + name.substr(9);
      if (name.rfind("threadIdx.", 0) == 0) reg = "tid." + name.substr(10);
      if (!reg.empty()) lowerer->VarRemapSet(iv.get<0>(), Register(reg));
    }
    auto result = lowerer->Mutate(body).ValueOrUnchanged(body);
    ffi::Array<Stmt> prefix;
    if (lowerer->needs_warp_) {
      prefix.push_back(Bind(lowerer->warp_, prim::FloorDiv(lowerer->LinearThread(), 64)));
    }
    prefix.push_back(result);
    return SeqStmt(prefix);
  }

 private:
  PrimExpr Axis(const std::string& tag, int axis) {
    auto it = params_.find(tag + std::string(1, 'x' + axis));
    if (it == params_.end()) return IntImm::Int32(0);
    if (prim::IsOne(it->second.get<1>())) return IntImm::Int32(0);
    std::string reg = tag == "blockIdx." ? "ctaid." : "tid.";
    return Register(reg + std::string(1, 'x' + axis));
  }
  PrimExpr BlockDimension(int axis) {
    auto it = params_.find("threadIdx." + std::string(1, 'x' + axis));
    if (it != params_.end() && it->second.get<1>().as<IntImmNode>()) return it->second.get<1>();
    return Register("ntid." + std::string(1, 'x' + axis));
  }
  PrimExpr LinearThread() {
    sym::Analyzer analyzer;
    return analyzer->Simplify(
        Axis("threadIdx.", 0) +
        BlockDimension(0) * (Axis("threadIdx.", 1) + BlockDimension(1) * Axis("threadIdx.", 2)));
  }
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    const auto* name_op = op->op.as<OpNode>();
    if (!name_op) return StmtExprMutator::Mutate_(op, inplace_mode);
    std::string name(name_op->name);
    if (name.rfind("tirx.maca.", 0) != 0) return StmtExprMutator::Mutate_(op, inplace_mode);
    name = name.substr(10);
    auto axis = [&]() { return op->args[0].as_or_throw<StringImm>()->value.c_str()[0] - 'x'; };
    if (name == "block_idx") return Axis("blockIdx.", axis());
    if (name == "thread_idx") return Axis("threadIdx.", axis());
    if (name == "grid_dim") return Register("nctaid." + std::string(1, 'x' + axis()));
    if (name == "block_dim") return Register("ntid." + std::string(1, 'x' + axis()));
    if (name == "linear_thread_id") return LinearThread();
    if (name == "lane_id") return prim::FloorMod(LinearThread(), 64);
    if (name == "thread_in_warpgroup") return prim::FloorMod(LinearThread(), 256);
    if (name == "warp_id" || name == "warpgroup_id" || name == "warp_in_warpgroup") {
      needs_warp_ = true;
      if (name == "warp_id") return warp_;
      if (name == "warpgroup_id") return prim::FloorDiv(warp_, 4);
      return prim::FloorMod(warp_, 4);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }
  LaunchParams params_;
  PrimVar warp_{"warp_id_in_cta", PrimType::Int(32)};
  bool needs_warp_{false};
};

}  // namespace

struct ExecAnalysis::Impl {
  enum class Index { kLane, kWave, kGroup, kWaveInGroup, kThread, kThreadInGroup };
  LaunchParams params;
  std::vector<ActiveSet> stack;
  std::unordered_map<Var, Index, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> aliases;

  explicit Impl(const LaunchParams& params) : params(params) {
    auto threads = Product(params, "threadIdx.");
    auto ctas = Product(params, "blockIdx.");
    if (!threads || !ctas) return;
    // Partial final waves are overapproximated; never omit participating lanes.
    stack.push_back({{"laneid", {std::min(*threads, kWaveSize)}},
                     {"warpid", {*threads / kWaveSize + (*threads % kWaveSize != 0)}},
                     {"cta_id", {*ctas}}});
  }

  std::optional<Index> Resolve(const PrimExpr& value) const {
    if (const auto* cast = value.as<prim::CastNode>()) return Resolve(cast->value);
    if (auto var = value.as<Var>()) return Resolve(*var);
    const auto* call = value.as<CallNode>();
    if (!call) return std::nullopt;
    const auto* op = call->op.as<OpNode>();
    if (!op) return std::nullopt;
    std::string name(op->name);
    if (name == "tirx.maca.lane_id") return Index::kLane;
    if (name == "tirx.maca.warp_id") return Index::kWave;
    if (name == "tirx.maca.warpgroup_id") return Index::kGroup;
    if (name == "tirx.maca.warp_in_warpgroup") return Index::kWaveInGroup;
    if (name == "tirx.maca.thread_in_warpgroup") return Index::kThreadInGroup;
    if (name == "tirx.maca.linear_thread_id") return Index::kThread;
    if (name == "tirx.maca.thread_idx" && call->args[0].as_or_throw<StringImm>()->value == "x") {
      // threadIdx.x equals the linear index only for a one-dimensional block.
      for (char axis : {'y', 'z'}) {
        auto it = params.find(std::string("threadIdx.") + axis);
        if (it != params.end() && !prim::IsOne(it->second.get<1>())) return std::nullopt;
      }
      return Index::kThread;
    }
    return std::nullopt;
  }

  std::optional<Index> Resolve(const Var& var) const {
    auto it = aliases.find(var);
    return it == aliases.end() ? std::nullopt : std::optional<Index>(it->second);
  }

  bool Narrow(ActiveSet* active, Index index, int64_t lo, int64_t hi) const {
    auto& lanes = active->at("laneid");
    auto& waves = active->at("warpid");
    if (index == Index::kLane) return lanes.Intersect(lo, hi);
    if (index == Index::kWave) return waves.Intersect(lo, hi);
    if (index == Index::kGroup) {
      lo = std::max<int64_t>(lo, 0);
      if (lo > std::numeric_limits<int64_t>::max() / kWavesPerGroup) return false;
      hi = std::min(hi, std::numeric_limits<int64_t>::max() / kWavesPerGroup);
      return waves.Intersect(lo * kWavesPerGroup, hi * kWavesPerGroup);
    }
    if (index == Index::kWaveInGroup) {
      if (waves.stride != 1 || waves.offset / 4 != (waves.offset + waves.extent - 1) / 4)
        return false;
      int64_t base = waves.offset / 4 * 4;
      return waves.Intersect(base + std::max<int64_t>(lo, 0), base + std::min<int64_t>(hi, 4));
    }
    // A flat thread interval can be represented by a lane/wave rectangle only
    // when it lies in one wave or covers complete waves. Other cases stay conservative.
    int64_t base = 0;
    if (index == Index::kThreadInGroup) {
      if (waves.stride != 1 || waves.offset / 4 != (waves.offset + waves.extent - 1) / 4)
        return false;
      base = waves.offset / 4 * 4;
      lo = std::max<int64_t>(lo, 0);
      hi = std::min<int64_t>(hi, 256);
    }
    lo = std::max<int64_t>(lo, 0);
    if (hi <= lo) return false;
    if (lo / kWaveSize == (hi - 1) / kWaveSize) {
      return waves.Intersect(base + lo / kWaveSize, base + lo / kWaveSize + 1) &&
             lanes.Intersect(lo % kWaveSize, (hi - 1) % kWaveSize + 1);
    }
    if (lo % kWaveSize == 0 && hi % kWaveSize == 0)
      return waves.Intersect(base + lo / kWaveSize, base + hi / kWaveSize);
    return false;
  }
};

ExecAnalysis::ExecAnalysis(const LaunchParams& params) : impl_(std::make_unique<Impl>(params)) {}
ExecAnalysis::~ExecAnalysis() = default;

void ExecAnalysis::Bind(const Var& var, const PrimExpr& value) {
  if (auto index = impl_->Resolve(value)) impl_->aliases.insert_or_assign(var, *index);
}

int ExecAnalysis::PushPredicate(const PrimExpr& condition) {
  if (impl_->stack.empty()) return 0;
  auto canonical = cuda::TryClassifyCanonical(
      condition, [&](const Var& var) { return impl_->Resolve(var).has_value(); });
  if (!canonical) return 0;
  struct Constraint {
    int64_t lo = sym::ConstIntBound::kNegInf;
    int64_t hi = sym::ConstIntBound::kPosInf;
  };
  std::map<Impl::Index, Constraint> constraints;
  for (const auto& atom : canonical->atoms) {
    // PTX election semantics are not part of MACA's predicate model.
    if (atom.kind != cuda::FilterAtomKind::kRange) return 0;
    auto index = impl_->Resolve(atom.scopeid_var.value());
    if (!index) return 0;
    auto& range = constraints[*index];
    range.lo = std::max(range.lo, atom.lo);
    range.hi = std::min(range.hi, atom.hi);
  }
  ActiveSet active = impl_->stack.back();
  // Group constraints must be applied before thread-in-group constraints,
  // irrespective of their order in a conjunction.
  for (auto index : {Impl::Index::kGroup, Impl::Index::kWave, Impl::Index::kWaveInGroup,
                     Impl::Index::kThread, Impl::Index::kThreadInGroup, Impl::Index::kLane}) {
    auto it = constraints.find(index);
    if (it == constraints.end()) continue;
    ActiveSet narrowed = active;
    if (impl_->Narrow(&narrowed, index, it->second.lo, it->second.hi)) active = narrowed;
  }
  impl_->stack.push_back(std::move(active));
  return 1;
}

void ExecAnalysis::Pop(int count) {
  while (count-- > 0) impl_->stack.pop_back();
}

std::pair<SplitMap, SplitMap> ExecAnalysis::Split(const ExecScope& scope) const {
  if (impl_->stack.empty()) return {};
  const auto& active = impl_->stack.back();
  SplitMap inter, intra;
  auto put = [&](SplitMap* side, const std::string& name) {
    side->Set(name, active.at(name).Encode());
  };
  auto name = scope->name();
  if (name == "kernel" || name == "thread") {
    put(&inter, "laneid");
    put(&inter, "warpid");
    put(&inter, "cta_id");
  } else if (name == "warp") {
    put(&intra, "laneid");
    put(&inter, "warpid");
    put(&inter, "cta_id");
  } else if (name == "cta") {
    put(&intra, "laneid");
    put(&intra, "warpid");
    put(&inter, "cta_id");
  } else if (name == "warpgroup") {
    const auto& waves = active.at("warpid");
    int64_t first = waves.offset % kWavesPerGroup;
    if (waves.stride != 1) return {};
    if (first == 0 && waves.extent % kWavesPerGroup == 0) {
      intra.Set("wid_in_wg", AxisRange{kWavesPerGroup}.Encode());
      inter.Set("wgid",
                AxisRange{waves.extent / kWavesPerGroup, waves.offset / kWavesPerGroup}.Encode());
    } else if (waves.extent <= kWavesPerGroup - first) {
      intra.Set("wid_in_wg", AxisRange{waves.extent, first}.Encode());
      inter.Set("wgid", AxisRange{1, waves.offset / kWavesPerGroup}.Encode());
    } else {
      return {};
    }
    put(&intra, "laneid");
    put(&inter, "cta_id");
  }
  return {inter, intra};
}

Stmt LowerIndices(const Stmt& stmt, const LaunchParams& params) {
  return MacaIndexLowerer::Lower(stmt, params);
}
}  // namespace tvm::backend::maca
