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

#ifndef TVM_BACKEND_MACA_TILE_EXEC_ANALYSIS_H_
#define TVM_BACKEND_MACA_TILE_EXEC_ANALYSIS_H_

#include <tvm/tirx/stmt.h>
#include <tvm/tirx/tile/exec_scope.h>

#include <memory>
#include <unordered_map>
#include <utility>

namespace tvm::backend::maca {
using LaunchParams = std::unordered_map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>>;
using SplitMap = ffi::Map<ffi::String, ffi::Array<PrimExpr>>;

/*! \brief MACA Wave64 topology and active ranges at each dispatch site. */
class ExecAnalysis {
 public:
  explicit ExecAnalysis(const LaunchParams& params);
  ~ExecAnalysis();
  void Bind(const Var& var, const PrimExpr& value);
  int PushPredicate(const PrimExpr& condition);
  void Pop(int count);
  std::pair<SplitMap, SplitMap> Split(const tirx::ExecScope& scope) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

Stmt LowerIndices(const Stmt& stmt, const LaunchParams& params);
}  // namespace tvm::backend::maca
#endif  // TVM_BACKEND_MACA_TILE_EXEC_ANALYSIS_H_
