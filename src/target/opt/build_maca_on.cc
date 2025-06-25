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
 *  Build maca modules from source.
 *  requires maca to be available.
 *
 * \file build_maca.cc
 */
#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <mcr/mc_runtime.h>
#include <mcr/mcrtc.h>

#include <cstdlib>

#include "../../runtime/maca/maca_common.h"
#include "../../runtime/maca/maca_module.h"
#include "../build_common.h"
#include "../source/codegen_maca.h"

namespace tvm {
namespace codegen {
using namespace maca;
#define MCRTC_CALL(x)                                                                        \
  {                                                                                          \
    mcrtcResult result = x;                                                                  \
    if (result != MCRTC_SUCCESS) {                                                           \
      LOG(FATAL) << "McrtcError: " #x " failed with error: " << mcrtcGetErrorString(result); \
    }                                                                                        \
  }

std::string FindMACAIncludePath() {
#if defined(_WIN32)
  const std::string delimiter = "\\";
#else
  const std::string delimiter = "/";
#endif
  std::string maca_include_path;
  const char* maca_path_env = std::getenv("MACA_PATH");
  if (maca_path_env != nullptr) {
    maca_include_path += maca_path_env;
    maca_include_path += delimiter + "include";
    return maca_include_path;
  }

#if defined(__linux__)
  struct stat st;
  maca_include_path = "/opt/maca/include";
  if (stat(maca_include_path.c_str(), &st) == 0) {
    return maca_include_path;
  }
#endif
  LOG(FATAL) << "Cannot find MACA include path."
             << "MACA_PATH is not set or MACA is not installed in the default installation path."
             << "In other than linux, it is necessary to set MACA_PATH.";
  return maca_include_path;
}

std::string FindCubridgeIncludePath() {
#if defined(_WIN32)
  const std::string delimiter = "\\";
#else
  const std::string delimiter = "/";
#endif
  std::string cubridge_include_path;
  const char* maca_path_env = std::getenv("MACA_PATH");
  if (maca_path_env != nullptr) {
    cubridge_include_path += maca_path_env;
    cubridge_include_path += delimiter + "tools";
    cubridge_include_path += delimiter + "cu-bridge";
    cubridge_include_path += delimiter + "include";
    return cubridge_include_path;
  }

#if defined(__linux__)
  struct stat st;
  cubridge_include_path = "/opt/maca/tools/cu-bridge/include";
  if (stat(cubridge_include_path.c_str(), &st) == 0) {
    return cubridge_include_path;
  }
#endif
  LOG(FATAL) << "Cannot find cu-bridge include path."
             << "MACA_PATH is not set or MACA is not installed in the default installation path."
             << "In other than linux, it is necessary to set MACA_PATH.";
  return cubridge_include_path;
}

std::string MCRTCCompile(const std::string& code, bool include_path = false) {
  std::vector<std::string> compile_params;
  std::vector<const char*> param_cstrings{};
  mcrtcProgram prog;
  std::string cc = "30";
  int major, minor;
  mcError_t e1 = mcDeviceGetAttribute(&major, mcDeviceAttributeComputeCapabilityMajor, 0);
  mcError_t e2 = mcDeviceGetAttribute(&minor, mcDeviceAttributeComputeCapabilityMinor, 0);

  if (e1 == mcSuccess && e2 == mcSuccess) {
    cc = std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
  }
  // FIXME:
  compile_params.push_back("-arch=compute_" + cc);

  if (include_path) {
    std::string include_option = "--include-path=" + FindMACAIncludePath();
    std::string include_cubridge_option = "--include-path=" + FindCubridgeIncludePath();

    compile_params.push_back(include_option);
    compile_params.push_back(include_cubridge_option);
  }

  for (const auto& string : compile_params) {
    param_cstrings.push_back(string.c_str());
  }
  MCRTC_CALL(mcrtcCreateProgram(&prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  mcrtcResult compile_res = mcrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  size_t log_size;
  MCRTC_CALL(mcrtcGetProgramLogSize(prog, &log_size));
  std::string log;
  log.resize(log_size);
  MCRTC_CALL(mcrtcGetProgramLog(prog, &log[0]));
  ICHECK_EQ(compile_res, MCRTC_SUCCESS) << log;
  size_t bitcode_size;
  MCRTC_CALL(mcrtcGetBitcodeSize(prog, &bitcode_size));

  std::string bitcode;
  bitcode.resize(bitcode_size);
  MCRTC_CALL(mcrtcGetBitcode(prog, &bitcode[0]));
  MCRTC_CALL(mcrtcDestroyProgram(&prog));

  return bitcode;
}

runtime::Module BuildMACA(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenMACA cg;
  cg.Init(output_ssa);

  Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenMACA: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    auto calling_conv = prim_func->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenMACA: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    functions.Set(gvar, prim_func);
  }

  for (auto [gvar, prim_func] : functions) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (auto [gvar, prim_func] : functions) {
    cg.AddFunction(gvar, prim_func);
  }

  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_maca_postproc")) {
    code = (*f)(code, target).operator std::string();
  }
  std::string fmt = "mcir";
  std::string mcir;
  const auto* f_enter = Registry::Get("target.TargetEnterScope");
  (*f_enter)(target);
  if (const auto* f = Registry::Get("tvm_callback_maca_compile")) {
    mcir = (*f)(code, target).operator std::string();
    // Dirty matching to check mcir vs mcbin.
    // TODO(tqchen) more reliable checks
    if (mcir[0] != '/') fmt = "mcbin";
  } else {
    mcir = MCRTCCompile(code, cg.need_include_path());
  }
  const auto* f_exit = Registry::Get("target.TargetExitScope");
  (*f_exit)(target);
  return MACAModuleCreate(mcir, fmt, ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.maca").set_body_typed(BuildMACA);
TVM_REGISTER_PASS_CONFIG_OPTION("maca.kernels_output_dir", String);
}  // namespace codegen
}  // namespace tvm
