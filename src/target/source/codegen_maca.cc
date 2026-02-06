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
 * \file codegen_maca.cc
 */

#include "codegen_maca.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../../tir/transforms/ir_utils.h"
#include "literal/maca_half_t.h"

namespace tvm {
namespace codegen {
namespace maca {
/*!
 * \brief Replace patterns with replacement strings.
 * \note should use std::format instead when codebase is ported to C++20.
 */
class Replacer {
 public:
  void register_rule(const std::string& pattern, const std::string& replacement) {
    _rules.emplace_back(pattern, replacement);
  }
  std::string rewrite(std::string str) {
    for (auto&& rule : _rules) {
      auto [pattern, replacement] = rule;
      size_t len = pattern.size();
      size_t new_len = replacement.size();
      size_t pos = str.find(pattern);
      while (pos != std::string::npos) {
        str = str.replace(pos, len, replacement);
        pos = str.find(pattern, pos + new_len);
      }
    }
    return str;
  }
  void empty_rules() { _rules.clear(); }

 private:
  std::vector<std::pair<std::string, std::string>> _rules;
};

std::string PrintMemcpyAsyncAssembly(const std::string& ret_var, const std::string& shared_ptr,
                                     const std::string& shared_elem_offset,
                                     const std::string& global_ptr,
                                     const std::string& global_elem_offset,
                                     const std::string& bytes) {
  std::string code = "{ret}memcpy_async<{copy_size}, false>({shared_ptr}, {global_ptr});\n";
  Replacer replacer;
  replacer.register_rule("{shared_ptr}", shared_ptr + " + " + shared_elem_offset);
  replacer.register_rule("{global_ptr}", global_ptr + " + " + global_elem_offset);
  replacer.register_rule("{copy_size}", bytes);
  if (ret_var.empty()) {
    replacer.register_rule("{ret}", "");
  } else {
    replacer.register_rule("{ret}", ret_var + " = ");
  }
  code = replacer.rewrite(code);
  return code;
}

std::string PrintPredicatedMemcpyAsyncAssembly(const std::string& ret_var,
                                               const std::string& shared_ptr,
                                               const std::string& shared_elem_offset,
                                               const std::string& global_ptr,
                                               const std::string& global_elem_offset,
                                               const std::string& bytes,
                                               const std::string& predicate_value) {
  std::string code = R"(
    {
      int pred_guard = (int){pred_guard};
      {ret}memcpy_async_pred<{copy_size}, MACA_ICMP_NE, false>({shared_ptr}, {global_ptr}, pred_guard, 0);
    }
)";
  Replacer replacer;
  replacer.register_rule("{shared_ptr}", shared_ptr + " + " + shared_elem_offset);
  replacer.register_rule("{global_ptr}", global_ptr + " + " + global_elem_offset);
  replacer.register_rule("{copy_size}", bytes);
  replacer.register_rule("{pred_guard}", predicate_value);
  if (ret_var.empty()) {
    replacer.register_rule("{ret}", "");
  } else {
    replacer.register_rule("{ret}", ret_var + " = ");
  }
  code = replacer.rewrite(code);
  return code;
}

std::string GetFP8Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4, 8, 16) for FP8";
  }
  stream << "__maca_fp8";
  std::string suffix;
  if (type.code() == DataType::kFloat8_e4m3fn) {
    suffix = "_e4m3";
  } else if (type.code() == DataType::kFloat8_e5m2) {
    suffix = "_e5m2";
  } else {
    LOG(FATAL) << "Unsupported FP8 type in MACA codegen";
  }
  stream << vec << suffix;
  return stream.str();
}

CodeGenMACA::CodeGenMACA() { restrict_keyword_ = "__restrict__"; }

void CodeGenMACA::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  vid_global_barrier_state_ = name_supply_->FreshName(runtime::symbol::tvm_global_barrier_state);
  vid_global_barrier_expect_ = name_supply_->FreshName("__barrier_expect");
  ICHECK_EQ(vid_global_barrier_state_, runtime::symbol::tvm_global_barrier_state);
}

void CodeGenMACA::PreFunctionBody(const PrimFunc& f) {
  VisitPipelineCommitQueueScope visitor;
  visitor(f->body);
  this->cp_async_remain_nums = visitor.total_cp_async_nums;
  for (auto& pair : this->mcDummyRetNum) {
    pair.second = 0;
  }
  while (!visitor.last_cp_async_size.empty()) {
    size_t last_cp_size = visitor.last_cp_async_size.front();
    visitor.last_cp_async_size.pop();
    this->mcDummyRetNum[last_cp_size]++;
    static int cp_async_ret_var_index = 0;
    std::string cp_async_ret_var_name = "mcDummyRet" + std::to_string(cp_async_ret_var_index++);
    this->cp_async_var_names.push(cp_async_ret_var_name);
  }

  for (const auto& pair : this->mcDummyRetNum) {
    if (pair.second == 0) {
      continue;
    }
    std::string bit = std::to_string(pair.first * 8);
    this->stream << "  b" << bit << "vectype mcDummyRetB" << bit << "[" << pair.second << "];\n";
    this->stream << "  int read_idx_" << bit << " = 0;\n";
    this->stream << "  int write_idx_" << bit << " = 0;\n";
  }
}

void CodeGenMACA::PrintFuncPrefix(std::ostream& os) { os << "extern \"C\" __global__ "; }

class ThreadIdxExtractor : public tir::StmtVisitor {
 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x" || iv->thread_tag == "threadIdx.x") {
        threadIdx_x_ext = op->value;
      }
      if (iv->var->name_hint == "threadIdx.y" || iv->thread_tag == "threadIdx.y") {
        threadIdx_y_ext = op->value;
      }
      if (iv->var->name_hint == "threadIdx.z" || iv->thread_tag == "threadIdx.z") {
        threadIdx_z_ext = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

 public:
  PrimExpr threadIdx_x_ext = Integer(1);
  PrimExpr threadIdx_y_ext = Integer(1);
  PrimExpr threadIdx_z_ext = Integer(1);
};

void CodeGenMACA::PrintExtraAttrs(const PrimFunc& f, std::ostream& os) {
  ThreadIdxExtractor extractor;
  extractor(f->body);
  arith::Analyzer analyzer;
  PrimExpr threadIdx_ext = analyzer.Simplify(extractor.threadIdx_x_ext * extractor.threadIdx_y_ext *
                                             extractor.threadIdx_z_ext);
  if (const IntImmNode* const threadIdx_ext_int = threadIdx_ext.as<IntImmNode>()) {
    if (threadIdx_ext_int->value == 1) {
      // unable to extract the number of threads per block, hence directly return
      return;
    }
    os << " __launch_bounds__(" << threadIdx_ext_int->value << ")";
  }
}

std::string CodeGenMACA::Finish() {
  if (enable_fp16_) {
    decl_stream << "#if defined(__MACA_ARCH__) && (__MACA_ARCH__ >= 1000)\n";
    decl_stream << "#include <maca_fp16.h>\n";
    decl_stream << "__device__ half max"
                << "(half a, half b)\n"
                << "{\n  return __hgt(__half(a), __half(b)) ? a : b;\n}\n";
    decl_stream << "__device__ half min(half a, half b)\n"
                << "{\n  return __hlt(__half(a), __half(b)) ? a : b;\n}\n";
    decl_stream << "#else\n";
    decl_stream << _maca_half_t_def;
    decl_stream << "#endif\n\n";
    decl_stream << _maca_half_util;
  }

  if (enable_bf16_) {
    decl_stream << "#if defined(__MACA_ARCH__) && (__MACA_ARCH__ >= 1000)\n";
    decl_stream << "#include <maca_bfloat16.h>\n";
    decl_stream << "__device__ maca_bfloat16 max"
                << "(maca_bfloat16 a, maca_bfloat16 b)\n"
                << "{\n  return __hgt(a, b) ? a : b;\n}\n";
    decl_stream << "__device__ maca_bfloat16 min(maca_bfloat16 a, maca_bfloat16 b)\n"
                << "{\n  return __hlt(a, b) ? a : b;\n}\n";
    decl_stream << "#endif\n\n";
    decl_stream << _maca_bfloat16_util;
  }

  if (enable_fp8_) {
    decl_stream << "#if defined(__MACA_ARCH__) && (__MACA_ARCH__ >= 1000)\n";
    decl_stream << "#include <maca_fp8.h>\n";
    decl_stream << "using fp8_e4_t = __maca_fp8_e4m3;\n";
    decl_stream << "using fp8_e4x2_t = __maca_fp8x2_e4m3;\n";
    decl_stream << "using fp8_e4x4_t = __maca_fp8x4_e4m3;\n";
    decl_stream << "struct fp8_e4x8_t {\n fp8_e4_t data[8]; \n};\n";
    decl_stream << "struct fp8_e4x16_t {\n fp8_e4_t data[16]; \n};\n";
    decl_stream << "using fp8_e5_t = __maca_fp8_e5m2;\n";
    decl_stream << "using fp8_e5x2_t = __maca_fp8x2_e5m2;\n";
    decl_stream << "using fp8_e5x4_t = __maca_fp8x4_e5m2;\n";
    decl_stream << "struct fp8_e5x8_t {\n fp8_e5_t data[8]; \n};\n";
    decl_stream << "struct fp8_e5x16_t {\n fp8_e5_t data[16]; \n};\n";
    decl_stream << "#endif\n\n";
  }
  declare_vector_type_extensions(decl_stream, enable_fp16_, enable_bf16_, enable_fp8_);

  if (enable_warp_shuffle_) {
    decl_stream << _maca_warp_intrinsic_util;
  }

  if (need_math_constants_h_) {
    // decl_stream << "#include <math_constants.h>\n";
    decl_stream << "/* single precision constants */\n";
    decl_stream << "#define MACART_INF_F            __int_as_float(0x7f800000U)\n";
    decl_stream << "#define MACART_NAN_F            __int_as_float(0x7fffffffU)\n";
    decl_stream << "/* double precision constants */\n";
    decl_stream << "#define MACART_INF              __longlong_as_double(0x7ff0000000000000ULL)\n";
    decl_stream << "#define MACART_NAN              __longlong_as_double(0xfff8000000000000ULL)\n";
  }

  if (need_mma_h_) {
    decl_stream << "#include <__clang_maca_mma_functions.h>\n";
    decl_stream << "namespace mxmaca {\n";
    decl_stream << "namespace wmma {\n";
    decl_stream << "template <> class fragment<matrix_a, 16, 16, 4, float, row_major> : public "
                   "__frag_base<float, 1> {};\n";
    decl_stream << "template <> class fragment<matrix_a, 16, 16, 4, float, col_major> : public "
                   "__frag_base<float, 1> {};\n";
    decl_stream << "template <> class fragment<matrix_b, 16, 16, 4, float, row_major> : public "
                   "__frag_base<float, 1> {};\n";
    decl_stream << "template <> class fragment<matrix_b, 16, 16, 4, float, col_major> : public "
                   "__frag_base<float, 1> {};\n";
    decl_stream << "template <> class fragment<accumulator, 16, 16, 4, float> : public "
                   "__frag_base<float, 4> {};\n";
    decl_stream << "static __device__ inline void\n";
    decl_stream << "load_matrix_sync(fragment<matrix_a, 16, 16, 4, float, row_major> &f,\n";
    decl_stream << "                const float *p, unsigned int ldm) {\n";
    decl_stream << "  unsigned row = __lane_id() & 0xf;\n";
    decl_stream << "  unsigned col = (__lane_id() >> 4) ^ 0x3;\n";
    decl_stream << "  f.x[0] = p[row * ldm + col];\n";
    decl_stream << "}\n";
    decl_stream << "static __device__ inline void\n";
    decl_stream << "load_matrix_sync(fragment<matrix_a, 16, 16, 4, float, col_major> &f,\n";
    decl_stream << "                const float *p, unsigned int ldm) {\n";
    decl_stream << "  unsigned row = __lane_id() & 0xf;\n";
    decl_stream << "  unsigned col = (__lane_id() >> 4) ^ 0x3;\n";
    decl_stream << "  f.x[0] = p[col * ldm + row];\n";
    decl_stream << "}\n";
    decl_stream << "static __device__ inline void\n";
    decl_stream << "load_matrix_sync(fragment<matrix_b, 16, 16, 4, float, row_major> &f,\n";
    decl_stream << "                const float *p, unsigned int ldm) {\n";
    decl_stream << "  unsigned row = (__lane_id() >> 4) ^ 0x3;\n";
    decl_stream << "  unsigned col = __lane_id() & 0xf;\n";
    decl_stream << "  f.x[0] = p[row * ldm + col];\n";
    decl_stream << "}\n";
    decl_stream << "static __device__ inline void\n";
    decl_stream << "load_matrix_sync(fragment<matrix_b, 16, 16, 4, float, col_major> &f,\n";
    decl_stream << "                const float *p, unsigned int ldm) {\n";
    decl_stream << "  unsigned row = (__lane_id() >> 4) ^ 0x3;\n";
    decl_stream << "  unsigned col = __lane_id() & 0xf;\n";
    decl_stream << "  f.x[0] = p[col * ldm + row];\n";
    decl_stream << "}\n";
    decl_stream << "static __device__ inline void\n";
    decl_stream
        << "store_matrix_sync(float *p, const fragment<accumulator, 16, 16, 4, float> &f,\n";
    decl_stream << "                  unsigned int ldm, layout_t layout) {\n";
    decl_stream << "  unsigned row = (__lane_id() >> 4) << 2;\n";
    decl_stream << "  unsigned col = __lane_id() & 0xf;\n";
    decl_stream << "  if (layout_t::mem_row_major == layout) {\n";
    decl_stream << "    p[row * ldm + col] = f.x[0];\n";
    decl_stream << "    p[(row + 1) * ldm + col] = f.x[1];\n";
    decl_stream << "    p[(row + 2) * ldm + col] = f.x[2];\n";
    decl_stream << "    p[(row + 3) * ldm + col] = f.x[3];\n";
    decl_stream << "  } else {\n";
    decl_stream << "    p[col * ldm + row] = f.x[0];\n";
    decl_stream << "    p[col * ldm + row + 1] = f.x[1];\n";
    decl_stream << "    p[col * ldm + row + 2] = f.x[2];\n";
    decl_stream << "    p[col * ldm + row + 3] = f.x[3];\n";
    decl_stream << "  }\n";
    decl_stream << "}\n";
    decl_stream << "template <typename LayoutA, typename LayoutB>\n";
    decl_stream << "static __device__ inline void\n";
    decl_stream << "mma_sync(fragment<accumulator, 16, 16, 4, float> &d,\n";
    decl_stream << "        const fragment<matrix_a, 16, 16, 4, float, LayoutA> &a,\n";
    decl_stream << "        const fragment<matrix_b, 16, 16, 4, float, LayoutB> &b,\n";
    decl_stream << "        const fragment<accumulator, 16, 16, 4, float> &c) {\n";
    decl_stream << "  d.x = __builtin_mxc_mma_16x16x4f32(a.x[0], b.x[0], c.x);\n";
    decl_stream << "}\n";
    if (enable_fp8_) {
      decl_stream << "template <> class fragment<matrix_a, 16, 16, 16, __maca_fp8_e4m3, row_major> "
                     ": public __frag_base<unsigned int, 1> {};\n";
      decl_stream << "template <> class fragment<matrix_a, 16, 16, 16, __maca_fp8_e4m3, col_major> "
                     ": public __frag_base<unsigned int, 1> {};\n";
      decl_stream << "template <> class fragment<matrix_b, 16, 16, 16, __maca_fp8_e4m3, row_major> "
                     ": public __frag_base<unsigned int, 1> {};\n";
      decl_stream << "template <> class fragment<matrix_b, 16, 16, 16, __maca_fp8_e4m3, col_major> "
                     ": public __frag_base<unsigned int, 1> {};\n";
      decl_stream << "static __device__ inline void\n";
      decl_stream
          << "load_matrix_sync(fragment<matrix_a, 16, 16, 16, __maca_fp8_e4m3, row_major> &f,\n";
      decl_stream << "                const __maca_fp8_e4m3 *p, unsigned int ldm) {\n";
      decl_stream << "  unsigned row = __lane_id() & 0xf;\n";
      decl_stream << "  unsigned col = ((__lane_id() >> 4) << 2);\n";
      decl_stream << "  f.x[0] = *(unsigned int *)(p + row * ldm + col);\n";
      decl_stream << "}\n";
      decl_stream << "static __device__ inline void\n";
      decl_stream
          << "load_matrix_sync(fragment<matrix_a, 16, 16, 16, __maca_fp8_e4m3, col_major> &f,\n";
      decl_stream << "                const __maca_fp8_e4m3 *p, unsigned int ldm) {\n";
      decl_stream << "  unsigned row = __lane_id() & 0xf;\n";
      decl_stream << "  unsigned col = ((__lane_id() >> 4) << 2) ^ 0xf;\n";
      decl_stream << "  union {\n";
      decl_stream << "    unsigned int i;\n";
      decl_stream << "    unsigned char c[4];\n";
      decl_stream << "  } tmp;\n";
      decl_stream << "  tmp.c[0] = p[(col - 3) * ldm + row].__x;\n";
      decl_stream << "  tmp.c[1] = p[(col - 2) * ldm + row].__x;\n";
      decl_stream << "  tmp.c[2] = p[(col - 1) * ldm + row].__x;\n";
      decl_stream << "  tmp.c[3] = p[col * ldm + row].__x;\n";
      decl_stream << "  f.x[0] = tmp.i;\n";
      decl_stream << "}\n";
      decl_stream << "static __device__ inline void\n";
      decl_stream
          << "load_matrix_sync(fragment<matrix_b, 16, 16, 16, __maca_fp8_e4m3, row_major> &f,\n";
      decl_stream << "                const __maca_fp8_e4m3 *p, unsigned int ldm) {\n";
      decl_stream << "  unsigned row = ((__lane_id() >> 4) << 2) ^ 0xf;\n";
      decl_stream << "  unsigned col = __lane_id() & 0xf;\n";
      decl_stream << "  union {\n";
      decl_stream << "    int i;\n";
      decl_stream << "    unsigned char c[4];\n";
      decl_stream << "  } tmp;\n";
      decl_stream << "  tmp.c[0] = p[(row - 3) * ldm + col].__x;\n";
      decl_stream << "  tmp.c[1] = p[(row - 2) * ldm + col].__x;\n";
      decl_stream << "  tmp.c[2] = p[(row - 1) * ldm + col].__x;\n";
      decl_stream << "  tmp.c[3] = p[row * ldm + col].__x;\n";
      decl_stream << "  f.x[0] = tmp.i;\n";
      decl_stream << "}\n";
      decl_stream << "static __device__ inline void\n";
      decl_stream
          << "load_matrix_sync(fragment<matrix_b, 16, 16, 16, __maca_fp8_e4m3, col_major> &f,\n";
      decl_stream << "                const __maca_fp8_e4m3 *p, unsigned int ldm) {\n";
      decl_stream << "  unsigned row = ((__lane_id() >> 4) << 2);\n";
      decl_stream << "  unsigned col = __lane_id() & 0xf;\n";
      decl_stream << "  f.x[0] = *(unsigned int *)(p + col * ldm + row);\n";
      decl_stream << "}\n";
      decl_stream << "template <typename LayoutA, typename LayoutB>\n";
      decl_stream << "static __device__ inline void\n";
      decl_stream << "mma_sync(fragment<accumulator, 16, 16, 16, float> &d,\n";
      decl_stream << "        const fragment<matrix_a, 16, 16, 16, __maca_fp8_e4m3, LayoutA> &a,\n";
      decl_stream << "        const fragment<matrix_b, 16, 16, 16, __maca_fp8_e4m3, LayoutB> &b,\n";
      decl_stream << "        const fragment<accumulator, 16, 16, 16, float> &c) {\n";
      decl_stream << "  d.x = __builtin_mxc_mma_f32_16x16x16f8_e4m3(a.x[0], b.x[0], c.x);\n";
      decl_stream << "}\n";
    }
    decl_stream << "}\n";
    decl_stream << "}\n";
  }

  if (need_cast_smem_ptr_to_int_) {
    decl_stream << "__forceinline__ __device__ unsigned int\n";
    decl_stream << "cast_smem_ptr_to_int(const void* const smem_ptr)\n";
    decl_stream << "{\n";
    decl_stream << "  return __cvta_generic_to_shared(smem_ptr);\n";
    decl_stream << "}\n";
  }

  decl_stream << "\n#if (((__MACACC_VER_MAJOR__ == 11) && (__MACACC_VER_MINOR__ >= 4)) || \\\n";
  decl_stream << "     (__MACACC_VER_MAJOR__ > 11))\n";
  decl_stream << "#define TVM_ENABLE_L2_PREFETCH 1\n";
  decl_stream << "#else\n";
  decl_stream << "#define TVM_ENABLE_L2_PREFETCH 0\n";
  decl_stream << "#endif\n";

  decl_stream << "\n#ifdef _WIN32\n";
  decl_stream << "  using uint = unsigned int;\n";
  decl_stream << "  using uchar = unsigned char;\n";
  decl_stream << "  using ushort = unsigned short;\n";
  decl_stream << "  using int64_t = long long;\n";
  decl_stream << "  using uint64_t = unsigned long long;\n";
  decl_stream << "#else\n";
  decl_stream << "  #define uint unsigned int\n";
  decl_stream << "  #define uchar unsigned char\n";
  decl_stream << "  #define ushort unsigned short\n";
  decl_stream << "  #define int64_t long long\n";
  decl_stream << "  #define uint64_t unsigned long long\n";
  decl_stream << "#endif\n";

  decl_stream << "\n#ifndef b16vectype\n";
  decl_stream << "  typedef __NATIVE_VECTOR__(1, short) b16vectype;\n";
  decl_stream << "#endif\n";
  return CodeGenC::Finish();
}

void CodeGenMACA::VisitStmt_(const tir::ForNode* op) {
  ICHECK(is_const_int(op->min, 0));
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenMACA::VisitStmt_(const DeclBufferNode* op) {
  auto data = op->buffer->data;
  ffi::String data_scope = (Downcast<PointerType>(data->type_annotation))->storage_scope.c_str();
  if (data_scope == "shared.dyn") {
    ffi::String data_name = data->name_hint;
    uint32_t buffer_alignment = std::max(op->buffer->dtype.bytes(), 1);
    auto it = shd_aligns.find(data_name);
    if (it != shd_aligns.end()) {
      shd_aligns[data_name] = std::max(shd_aligns[data_name], buffer_alignment);
    } else {
      shd_aligns[data_name] = buffer_alignment;
    }
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenMACA::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] = CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenMACA::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        enable_fp16_ = true;
        if (t.is_scalar()) {
          os << "half";
        } else if (lanes <= 8) {
          ICHECK_EQ(lanes % 2, 0) << "Only support an even number of lanes for half type";
          if (lanes <= 4) {
            os << "half" << lanes;
          } else {
            os << "uint" << lanes / 2;
          }
        } else {
          fail = true;
        }
        break;
      case 32:
        if (lanes <= 4) {
          os << "float";
        } else if (lanes <= 8) {
          // Emit MACA code to access fp32 vector elements for 4 < lanes <= 8.
          //
          // float8 is stored as ulonglong4
          //
          // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
          // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
          //
          ICHECK_EQ(lanes % 2, 0) << "only support even lane for float type with lanes > 4";
          os << "ulonglong" << lanes / 2;
        } else {
          fail = true;
        }
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16)) return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32)) return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
      os << "maca_bfloat16";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for bfloat16 type";
      if (lanes <= 4) {
        os << "maca_bfloat16" << lanes;
      } else {
        os << "uint" << lanes / 2;
      }
    } else {
      fail = true;
    }
    if (!fail) return;
  } else if (t.is_float8()) {
    enable_fp8_ = true;
    if (t.lanes() <= 4) {
      os << GetFP8Type(t);
    } else {
      os << "uint" << t.lanes() / 4;
    }
    return;
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_vector_bool()) {
    // MACA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
      case 1: {
        if (t.is_scalar()) {
          os << "int";
          return;
        } else if (t.lanes() == 8) {
          os << "int8_t";
          return;
        } else if (t.lanes() == 16) {
          os << "int16_t";
          return;
        } else if (t.lanes() == 32) {
          os << "int";
          return;
        } else {
          LOG(FATAL) << "Cannot convert type " << t << " to MACA type!";
        }
      }
      case 4: {
        if (t.is_scalar()) {
          os << "int";
          return;
        } else if (t.lanes() == 4) {
          os << "int16_t";
          return;
        } else if (t.lanes() == 8) {
          // directly 8 4-bit int in integer.
          os << "int";
          return;
        } else if (t.lanes() == 16) {
          os << "int2";
          return;
        } else if (t.lanes() == 32) {
          os << "int4";
          return;
        } else if (t.lanes() == 64) {
          os << "int8";
          return;
        } else {
          LOG(FATAL) << "Cannot convert type " << t << " to MACA type!";
        }
      }
      case 8: {
        if (t.lanes() == 4) {
          // directly 4 8 bit int in integer.
          enable_int8_ = true;

          // We use int for int8x4 instead of char4 because using char4 is
          // likely to produce extra instructions to pack four int8 elements
          // into 32-bit data.
          os << "int";
          return;
        } else if (t.lanes() == 8) {
          enable_int8_ = true;
          os << "int2";
          return;
        } else if (t.lanes() == 16) {
          enable_int8_ = true;
          os << "int4";
          return;
        } else if (!t.is_uint() && t.is_scalar()) {
          os << "signed char";
          break;
        } else {
          os << "char";
          break;
        }
      }
      case 16: {
        if (t.is_scalar()) {
          os << "short";
        } else if (t.lanes() <= 4) {
          os << "short" << lanes;
        } else if (t.lanes() <= 8) {
          // Emit MACA code to access int16 vector elements.
          //
          // short4 is stored as int2
          //
          // s4.x is emitted as *(short2*)(&(i2.x)).x
          // s4.y is emitted as *(short2*)(&(i2.x)).y
          // s4.z is emitted as *(short2*)(&(i2.y)).x
          // s4.w is emitted as *(short2*)(&(i2.y)).y
          //
          ICHECK_EQ(t.lanes() % 2, 0) << "only support even lane for shorT type with lanes > 4";
          os << "int" << t.lanes() / 2;
        } else {
          fail = true;
        }
        if (!fail) {
          return;
        }
        break;
      }
      case 32: {
        if (t.is_scalar()) {
          os << "int";
        } else if (t.lanes() <= 4) {
          os << "int" << t.lanes();
        } else if (t.lanes() <= 8) {
          // Emit MACA code to access int32 vector elements for 4 < lanes <= 8.
          //
          // int8 is stored as longlong4
          //
          // i8.v1 is emitted as *(int2*)(&(l4.x)).x
          // i8.v2 is emitted as *(int2*)(&(l4.x)).y
          //
          ICHECK_EQ(lanes % 2, 0) << "only support even lane for int32 type with lanes > 4";
          os << "longlong" << lanes / 2;
        } else {
          fail = true;
        }
        if (!fail) {
          return;
        }
        break;
      }
      case 64: {
        if (t.is_scalar()) {
          os << "int64_t";
        } else if (t.lanes() == 2) {
          os << "longlong2";
        } else if (t.lanes() == 3) {
          os << "longlong3";
        } else if (t.lanes() == 4) {
          os << "longlong4";
        }
        return;
      }
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to MACA type";
}

void CodeGenMACA::PrintVecConstructor(DataType t, std::ostream& os) {
  os << "make_";
  PrintType(t, os);
}

void CodeGenMACA::PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                                   std::ostream& os) {  // NOLINT(*)
  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    // Unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());

    for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenMACA::PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                   std::ostream& os) {  // NOLINT(*)
  if (t.is_scalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8 ? 16 : (t.bits() == 16 || t.bits() == 32) ? 8 : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    std::string type_name = t.is_int() ? "char" : "unsigned char";
    if (t.lanes() == 2 || t.lanes() == 3) {
      os << vec << "." << access[i % t.lanes()];
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
    }
  } else if (t.is_float16()) {
    if (t.lanes() <= 4) {
      os << vec << "." << access[i];
    } else {
      os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
    }
  } else if (t.is_bfloat16()) {
    if (t.lanes() <= 4) {
      os << vec << "." << access[i];
    } else {
      os << "((maca_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
    }
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2];
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenMACA::PrintVecElemStore(const std::string& vec, DataType t, int i,
                                    const std::string& value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8 ? 16 : (t.bits() == 16 || t.bits() == 32) ? 8 : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "="
             << "(" << value << ");\n";
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 4 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 4 * 8 << ");\n";
    }
  } else if (t.is_float16()) {
    if (t.lanes() <= 4) {
      stream << vec << "." << access[i] << " = " << value << ";\n";
    } else {
      stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2] << " = "
             << value << ";\n";
    }

  } else if (t.is_bfloat16()) {
    if (t.lanes() <= 4) {
      stream << vec << "." << access[i] << " = " << value << ";\n";
    } else {
      stream << "((maca_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->" << access[i % 2]
             << " = " << value << ";\n";
    }
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2] << ")))->"
           << access[i % 2] << " = " << value << ";\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenMACA::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  } else if (sync == "global") {
    if (!need_global_barrier_) {
      need_global_barrier_ = true;
      this->decl_stream << "extern \"C\" __device__ unsigned " << vid_global_barrier_state_
                        << ";\n";
    }
    // global synchronizer
    std::string is_load = PrintExpr(op->args[1]);
    std::string num_blocks = PrintExpr(op->args[2]);
    this->PrintIndent();
    // In theory only threadfence is needed
    // but we observed problems with only threadfence
    this->stream << "__threadfence_system();\n";
    this->PrintIndent();
    this->stream << "if (" << is_load << ") {\n";
    int wb = this->BeginScope();
    this->PrintIndent();
    this->stream << "atomicAdd(&" << vid_global_barrier_state_ << ", 1);\n";
    this->PrintIndent();
    std::string ptr = name_supply_->FreshName("pf");
    this->stream << "volatile unsigned* " << ptr << " = &" << vid_global_barrier_state_ << ";\n";
    this->PrintIndent();
    this->stream << vid_global_barrier_expect_ << " += " << num_blocks << ";\n";
    this->PrintIndent();
    this->stream << "while (" << ptr << "[0] < " << vid_global_barrier_expect_ << ");\n";
    this->EndScope(wb);
    this->PrintIndent();
    this->stream << "}\n";
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  }
}

void CodeGenMACA::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  ICHECK_NE(scope, "global") << "Cannot allocate global memory when targeting MACA. You must pass "
                                "all global arrays as input instead";
  if (scope == "shared") {
    os << "__shared__ ";
  } else if (scope == "shared.dyn") {
    os << "extern __shared__ ";
  }
}

std::string CodeGenMACA::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")";
  if (from.is_float16() && (target.is_int() || target.is_uint()) && target.bits() == 8) {
    os << "(";
    if (target.is_uint()) {
      os << "u";
    }
    os << "int)";
  }
  os << value << ")";
  return os.str();
}

void CodeGenMACA::VisitExpr_(const CastNode* op, std::ostream& os) {
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  // Emit simple C-style type conversion.
  if (from_ty.is_scalar()) return CodeGenC::VisitExpr_(op, os);

  if (target_ty.code() == DataType::kFloat8_e4m3fn || target_ty.code() == DataType::kFloat8_e5m2 ||
      from_ty.code() == DataType::kFloat8_e4m3fn || from_ty.code() == DataType::kFloat8_e5m2) {
    std::ostringstream val;
    if (target_ty.code() == DataType::kBFloat && target_ty.lanes() == 2) {
      val << "cast_to_maca_bfloat162(" << PrintExpr(op->value) << ")";
    } else {
      val << "(";
      PrintType(target_ty, val);
      val << ")(" << PrintExpr(op->value) << ")";
    }
    os << val.str();
    return;
  }

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  {
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
      std::ostringstream val;
      val << "(";
      PrintType(target_ty.element_of(), val);
      val << ")(";
      PrintVecElemLoad(src, from_ty, i, val);
      val << ")";
      PrintVecElemStore(sret, target_ty, i, val.str());
    }
  }
  os << sret;
}

void CodeGenMACA::PrintCallExtern(Type ret_type, ffi::String global_symbol, const ffi::Array<PrimExpr>& args,
                                  bool skip_first_arg, std::ostream& os) {  // NOLINT(*)
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_fixed_length_vector()) {
    //
    // Emit an unsupported vector call
    //
    // v = intrin_f((float4*)A[0], (float4*)B[0])
    //
    // as
    //
    // float4 __ret;
    // {
    //   float4 __arg0 = ((float4*)A)[0];
    //   float4 __arg1 = ((float4*)B)[0];
    //   __ret.x = intrin_f(__arg0.x, __arg1.x);
    //   __ret.y = intrin_f(__arg0.y, __arg1.y);
    //   __ret.z = intrin_f(__arg0.z, __arg1.z);
    //   __ret.w = intrin_f(__arg0.w, __arg1.w);
    // }
    // v = __ret;
    //
    // Declare the result vector.
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      // Load arguments.
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0) scall << ", ";
          PrintVecElemLoad(sargs[j], args[arg_begin + j].dtype(), i, scall);
        }
        scall << ")";
        PrintVecElemStore(sret, ret_dtype, i, scall.str());
      }
    }
    os << sret;
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg, os);
  }
}
static int stoi(const std::string& str) {
  try {
    return std::stoi(str);
  } catch (std::invalid_argument& e) {
    LOG(FATAL) << "Cannot convert \"" << str << "\" to int";
    throw;
  }
}

void CodeGenMACA::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (auto opt_call_opt = op->op.as<Op>()) {
    Op call_op = opt_call_opt.value();
    // This is only for backward compatibility with __shfl_{up/down}.
    // A macro will be used to replace *_sync calls to legacy ones.
    if (op_need_warp_shuffle_.get(call_op, false)) {
      enable_warp_shuffle_ = true;
    }
  }

  if (op->op.same_as(builtin::tvm_fill_fragment())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 6U);
    os << "mxmaca::wmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_load_matrix_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "mxmaca::wmma::load_matrix_sync(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_store_matrix_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "mxmaca::wmma::store_matrix_sync(";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[6], os);
    if (const StringImmNode* str = op->args[7].as<StringImmNode>()) {
      os << ", mxmaca::wmma::mem_" << str->value;
    } else {
      LOG(FATAL) << "Invalid parameters";
    }
    os << ")";
  } else if (op->op.same_as(builtin::tvm_mma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "mxmaca::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(builtin::tvm_bmma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "mxmaca::wmma::bmma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(builtin::create_barriers())) {
    CHECK_EQ(barrier_count_, -1);
    int barrier_count = Downcast<IntImm>(op->args[0])->value;
    // pad barrier alignment to avoid runtime alignment errors
    CHECK_EQ(barrier_alignment_bytes_ % sizeof(uint64_t), 0);
    int barrier_alignment_count = barrier_alignment_bytes_ / sizeof(uint64_t);
    if (barrier_count % barrier_alignment_count != 0) {
      barrier_count = ((barrier_count / barrier_alignment_count) + 1) * barrier_alignment_count;
    }
    barrier_count_ = barrier_count;
    this->stream << "__shared__ __align__(" << barrier_alignment_bytes_ << ") uint64_t "
                 << barrier_name_ << "[" << barrier_count << "];\n";
    this->stream << "for (int i = 0; i < " << barrier_count << "; ++i) { " << barrier_name_
                 << "[i] = 0; }\n";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenMACA::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::fragment_shape) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* shape_str = op->value.as<StringImmNode>();
    fragment_shapes[buffer] = shape_str->value;
  } else if (op->attr_key == tir::attr::fragment_layout) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* layout_str = op->value.as<StringImmNode>();
    fragment_layouts[buffer] = layout_str->value;
  } else if (op->attr_key == tir::attr::async_commit_queue_scope) {
    const IntImmNode* queue_id = op->value.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0) << "For MACA, the index of an async queue must be 0.";
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == tir::attr::async_wait_queue_scope) {
    auto wait_attrs = GetAsyncWaitAttributes(op);
    auto queue_id = wait_attrs.first.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0) << "For MACA, the index of an async queue must be 0.";
    // TODO(metax): Because the data type of the operation written into this block cannot be
    // obtained temporarily, for a barrier_arrive_and_wait function that only involves one type
    // of data,assume that if there is a method to obtain it in the future, replace the bit here.
    std::string bit = "";
    std::string mask_ = "";
    for (const auto& pair : mcDummyRetNum) {
      if (pair.second != 0) {
        bit = std::to_string(pair.first * 8);
        mask_ = std::to_string(pair.second - 1);
      }
    }
    std::string write_idx_ = "write_idx_" + bit;
    if (!this->cp_async_var_names.empty()) {
      std::string cp_async_ret_var_name = this->cp_async_var_names.front();
      this->cp_async_var_names.pop();
      this->PrintIndent();
      this->stream << "barrier_arrive_and_wait(mcDummyRetB" << bit << "[" << write_idx_ << "]);\n";
      this->PrintIndent();
      this->stream << write_idx_ << "= (" << write_idx_ << " + 1) & " << mask_ << ";\n";
    }
    auto inner = op->body.as<AttrStmtNode>();
    ICHECK(inner);
    this->VisitStmt(inner->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenMACA::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  std::string scope = GetPtrStorageScope(op->buffer_var);
  const VarNode* buffer = op->buffer_var.as<VarNode>();
  if (scope.find("wmma.") == 0) {
    if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
      ICHECK(op->dtype == DataType::Float(16) || op->dtype == DataType::Int(8) ||
             op->dtype == DataType::UInt(8) || op->dtype == DataType::Int(4) ||
             op->dtype == DataType::UInt(4) || op->dtype == DataType::Int(1) ||
             op->dtype == DataType::BFloat(16) || op->dtype == DataType::Float(32))
          << "Matrix_a and matrix_b only support half or char or unsigned char "
          << "or uint4 or int4 or int1 type for now";
    } else {
      ICHECK(op->dtype == DataType::Float(16) || op->dtype == DataType::Float(32) ||
             op->dtype == DataType::Int(32))
          << "Accumulator only support half, float and int type for now";
    }
    PrintWmmaScope(scope, op->dtype, buffer, stream);
  } else {
    PrintStorageScope(scope, stream);
    PrintType(op->dtype, stream);
  }

  if (scope == "shared.dyn") {
    std::string alignment = "";
    auto it = shd_aligns.find(vid);
    if (it != shd_aligns.end()) {
      alignment = "__align__(" + std::to_string(it->second) + ") ";
    }
    stream << ' ' << alignment << vid << "[];\n";
  } else {
    size_t constant_size = op->ConstantAllocationSize();
    ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

    if (scope.find("wmma.") == 0) {
      constant_size = GetWmmaFragmentSize(scope, buffer, constant_size);
    }
    if ((op->dtype == DataType::Int(4) || op->dtype == DataType::UInt(4) ||
         op->dtype == DataType::Int(1)) &&
        scope == "shared") {
      constant_size = constant_size / (32 / op->dtype.bits());
    }
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenMACA::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call && call->op.same_as(builtin::tvm_global_barrier_kinit())) {
    PrintIndent();
    stream << "__shared__ unsigned " << vid_global_barrier_expect_ << ";\n";
    PrintIndent();
    stream << "if (threadIdx.x == 0) {\n";
    PrintIndent();
    stream << "  " << vid_global_barrier_expect_ << " = 0;\n";
    PrintIndent();
    stream << "}\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenMACA::VisitExpr_(const RampNode* op, std::ostream& os) {
  int lanes = op->dtype.lanes();
  CHECK_LE(lanes, 4) << "ValueError: Ramp of more than 4 lanes is not allowed.";
  PrintVecConstructor(op->dtype, os);
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != lanes - 1) os << ", ";
  }
  os << ")";
}

void CodeGenMACA::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  int lanes = op->dtype.lanes();
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8) {
    bool fail = false;
    const int64_t* p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    if (lanes == 4) {
      // make_int8x4
      if (op->dtype.is_uint()) {
        os << "(uint)" << v;
      } else {
        os << "(int)" << v;
      }
    } else if (lanes == 8 || lanes == 16) {
      // [MXMACA] make_int2 or make_int4
      PrintVecConstructor(op->dtype, os);
      os << '(';
      for (int i = 0; i < lanes / 4; ++i) {
        if (i != 0) os << ", ";
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      }
      os << ')';
    } else {
      fail = true;
    }

    if (!fail) {
      return;
    }
  }

  if (op->dtype.is_float16()) {
    std::string v = PrintExpr(op->value);
    PrintVecConstructor(op->dtype, os);
    os << '(';
    if (lanes <= 4) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0) os << ", ";
        os << v << ", " << v;
      }
    } else {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0) os << ", ";
        os << "__pack_half2(" << v << ", " << v << ")";
      }
    }
    os << ')';
    return;
  }

  if (op->dtype.is_bfloat16()) {
    std::string v = PrintExpr(op->value);
    PrintVecConstructor(op->dtype, os);
    os << '(';
    if (lanes > 4) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0) os << ", ";
        os << "__pack_maca_bfloat162(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes; ++i) {
        if (i != 0) os << ", ";
        os << v;
      }
    }
    os << ')';
    return;
  }

  if (op->dtype.is_float8()) {
    int lanes = op->dtype.lanes();
    ICHECK(lanes == 1 || lanes == 2 || lanes == 4);
    std::string v = PrintExpr(op->value);
    // Implicit conversion from float back to fp8
    PrintType(op->dtype, os);
    os << "(make_float" << lanes << "(";
    for (int i = 0; i < lanes; ++i) {
      if (i != 0) os << ", ";
      os << "static_cast<float>(" << v << ")";
    }
    os << "))";
    return;
  }

  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 4) {
    bool fail = false;
    const int64_t* p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xF;

    if (lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (op->dtype.is_uint()) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) | (v << 4) | v;
      if (lanes == 8) {
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (lanes == 16 || lanes == 32) {
        PrintVecConstructor(op->dtype, os);
        os << '(';
        for (int i = 0; i < lanes / 8; ++i) {
          if (i != 0) os << ", ";
          if (op->dtype.is_uint()) {
            os << "(uint)" << v;
          } else {
            os << "(int)" << v;
          }
        }
        os << ')';
      } else {
        fail = true;
      }
    }

    if (!fail) {
      return;
    }
  }

  std::string v = PrintExpr(op->value);
  PrintVecConstructor(op->dtype, os);
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenMACA::VisitExpr_(const SelectNode* op, std::ostream& os) {
  // Non-vector cases.
  if (!op->dtype.is_fixed_length_vector()) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }

  // Codegen vector condition case by serializing the select op.
  ICHECK(op->false_value->dtype == op->dtype && op->true_value->dtype == op->dtype &&
         op->dtype.lanes() == op->condition.dtype().lanes());

  std::string r_var = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(op->dtype, stream);
  stream << ' ' << r_var << ";\n";
  {
    std::string c_var = SSAGetID(PrintExpr(op->condition), op->dtype);
    std::string t_var = SSAGetID(PrintExpr(op->true_value), op->dtype);
    std::string f_var = SSAGetID(PrintExpr(op->false_value), op->dtype);

    // The condition is stored as an ushort vector.
    int lanes = op->dtype.lanes();
    DataType memory_ty(DataType::TypeCode::kUInt, 16, lanes);

    for (int i = 0; i < lanes; ++i) {
      std::ostringstream item;
      item << "(bool(";
      PrintVecElemLoad(c_var, memory_ty, i, item);
      item << ")?";
      PrintVecElemLoad(t_var, op->dtype, i, item);
      item << ':';
      PrintVecElemLoad(f_var, op->dtype, i, item);
      item << ')';
      PrintVecElemStore(r_var, op->dtype, i, item.str());
    }
  }
  os << r_var;
}

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenMACA* p) {  // NOLINT(*)
  // Type code is kBFloat
  if (op->dtype.is_bfloat16()) {
    os << "__float2bfloat16_rn";
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (op->dtype.is_float8()) {
    p->PrintType(op->dtype, os);
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat
  switch (op->dtype.bits()) {
    case 64:
    case 32: {
      std::ostringstream temp;
      if (std::isinf(op->value)) {
        if (op->value < 0) {
          temp << "-";
        }
        temp << ((op->dtype.bits() == 32) ? "MACART_INF_F" : "MACART_INF");
        p->need_math_constants_h_ = true;
      } else if (std::isnan(op->value)) {
        temp << ((op->dtype.bits() == 32) ? "MACART_NAN_F" : "MACART_NAN");
        p->need_math_constants_h_ = true;
      } else {
        temp << std::scientific << op->value;
        if (op->dtype.bits() == 32) temp << 'f';
      }
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << "__float2half" << '(';
      FloatImm const_f32 = FloatImm(DataType::Float(32), op->value);
      PrintConst(const_f32.get(), os, p);
      os << ')';
      break;
    }
    default:
      LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenMACA::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenMACA::PrintWmmaScope(const std::string& scope, DataType t, const VarNode* variable,
                                 std::ostream& os) {
  std::stringstream type;
  PrintType(t, type);
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  if ((t.is_int() || t.is_uint()) && t.bits() < 8 && t.lanes() == 1) {
    type.str(std::string());
    if (t.is_int()) {
      if (t.bits() == 4) {
        type << "mxmaca::wmma::experimental::precision::s4";
      } else if (t.bits() == 1) {
        type << "mxmaca::wmma::experimental::precision::b1";
      } else {
        LOG(FATAL) << "Unhandled interger type for wmma fragment!";
      }
    } else if (t.is_uint()) {
      if (t.bits() == 4) {
        type << "mxmaca::wmma::experimental::precision::u4";
      } else {
        LOG(FATAL) << "Unhandled interger type for wmma fragment!";
      }
    }
  }
  if (scope == "wmma.matrix_a") {
    need_mma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_a";
    os << "mxmaca::wmma::fragment<mxmaca::wmma::matrix_a, " << shape_str << ", " << type.str()
       << ", mxmaca::wmma::" << layout_str << ">";
  } else if (scope == "wmma.matrix_b") {
    need_mma_h_ = true;
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_b";
    os << "mxmaca::wmma::fragment<mxmaca::wmma::matrix_b, " << shape_str << ", " << type.str()
       << ", mxmaca::wmma::" << layout_str << ">";
  } else if (scope == "wmma.accumulator") {
    need_mma_h_ = true;
    os << "mxmaca::wmma::fragment<mxmaca::wmma::accumulator, " << shape_str << ", " << type.str()
       << ">";
  }
}

int32_t CodeGenMACA::GetWmmaFragmentSize(const std::string& scope, const VarNode* variable,
                                         int32_t size) {
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  std::pair<int32_t, int32_t> dim = GetWmmaFragmentDimSize(shape_str, scope);
  if (dim.first * dim.second != 0)
    return size / dim.first / dim.second;
  else
    return 0;
}

void CodeGenMACA::HandleVolatileLoads(const std::string& value, const BufferLoadNode* op,
                                      std::ostream& os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  //
  if ((op->dtype.is_float16() || op->dtype.is_bfloat16()) && IsVolatile(op->buffer->data.get())) {
    os << "(";
    PrintType(op->dtype, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenMACA::PrintVecElemLoadExpr(DataType t, int i, const std::string& value,
                                       std::ostream& os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (!(t.lanes() == 2 || t.lanes() == 3)) {
      if (i != 0) {
        os << "|";
      }
      os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
      return;
    }
  }

  if (t.is_float16()) {
    if (i == 0) {
      PrintVecConstructor(t, os);
      os << '(';
    }
    if (i == t.lanes() - 1) {
      os << value << ")";
    } else {
      os << value << ",";
    }
    return;
  }

  if (t.is_bfloat16()) {
    if (i == 0) {
      PrintVecConstructor(t, os);
      os << '(';
    }
    if (i == t.lanes() - 1) {
      os << value << ")";
    } else {
      os << value << ",";
    }
    return;
  }

  if (i == 0) {
    PrintVecConstructor(t, os);
    os << "(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << ")";
  }
  return;
}
}  // namespace maca
}  // namespace codegen
}  // namespace tvm
