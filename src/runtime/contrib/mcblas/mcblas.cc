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
 * \file Use external cblas library call.
 */
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include "../../3rdparty/compiler-rt/builtin_fp16.h"
#include "../cblas/gemm_common.h"
#include "mcblas_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;
inline mcblasOperation_t MCBLASBooleanToTranspose(bool item) {
  return item ? MCBLAS_OP_T : MCBLAS_OP_N;
}

inline void MCBLASTryEnableTensorCore(mcblasHandle_t hdl) {
  // TensorCores are only supported in mcblas 9.0 or higher
  int version;
  CHECK_MCBLAS_ERROR(mcblasGetVersion(hdl, &version));
  if (version >= 9000) CHECK_MCBLAS_ERROR(mcblasSetMathMode(hdl, MCBLAS_DEFAULT_MATH));
}

struct McblasHgemmOp {
  typedef half TDatatype;
  mcblasHandle_t handle;
  explicit McblasHgemmOp(mcblasHandle_t hdl) : handle(hdl) {}

  void operator()(bool ta, bool tb, int M, int N, int K, half alpha, half* A, int lda, half* B,
                  int ldb, half beta, half* C, int ldc) {
    CHECK_MCBLAS_ERROR(mcblasHgemm(handle, MCBLASBooleanToTranspose(ta),
                                   MCBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda, B, ldb,
                                   &beta, C, ldc));
  }
};

struct McblasSgemmOp {
  typedef float TDatatype;
  mcblasHandle_t handle;
  explicit McblasSgemmOp(mcblasHandle_t hdl) : handle(hdl) {}

  void operator()(bool ta, bool tb, int M, int N, int K, float alpha, float* A, int lda, float* B,
                  int ldb, float beta, float* C, int ldc) {
    CHECK_MCBLAS_ERROR(mcblasSgemm(handle, MCBLASBooleanToTranspose(ta),
                                   MCBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda, B, ldb,
                                   &beta, C, ldc));
  }
};

struct McblasDgemmOp {
  typedef double TDatatype;
  mcblasHandle_t handle;
  explicit McblasDgemmOp(mcblasHandle_t hdl) : handle(hdl) {}
  void operator()(bool ta, bool tb, int M, int N, int K, double alpha, double* A, int lda,
                  double* B, int ldb, double beta, double* C, int ldc) {
    CHECK_MCBLAS_ERROR(mcblasDgemm(handle, MCBLASBooleanToTranspose(ta),
                                   MCBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda, B, ldb,
                                   &beta, C, ldc));
  }
};

struct McblasHgemmBatchOp {
  typedef half TDatatype;
  mcblasHandle_t handle;
  explicit McblasHgemmBatchOp(mcblasHandle_t hdl) : handle(hdl) {}
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, half alpha, half* A,
                  int a_stride, int lda, half* B, int b_stride, int ldb, half beta, half* C,
                  int c_stride, int ldc) {
    CHECK_MCBLAS_ERROR(mcblasHgemmStridedBatched(
        handle, MCBLASBooleanToTranspose(ta), MCBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda,
        a_stride, B, ldb, b_stride, &beta, C, ldc, c_stride, batch_size));
  }
};

struct McblasSgemmBatchOp {
  typedef float TDatatype;
  mcblasHandle_t handle;
  explicit McblasSgemmBatchOp(mcblasHandle_t hdl) : handle(hdl) {}
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, float alpha, float* A,
                  int a_stride, int lda, float* B, int b_stride, int ldb, float beta, float* C,
                  int c_stride, int ldc) {
    CHECK_MCBLAS_ERROR(mcblasSgemmStridedBatched(
        handle, MCBLASBooleanToTranspose(ta), MCBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda,
        a_stride, B, ldb, b_stride, &beta, C, ldc, c_stride, batch_size));
  }
};

struct McblasDgemmBatchOp {
  typedef double TDatatype;
  mcblasHandle_t handle;
  explicit McblasDgemmBatchOp(mcblasHandle_t hdl) : handle(hdl) {}
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, double alpha, double* A,
                  int a_stride, int lda, double* B, int b_stride, int ldb, double beta, double* C,
                  int c_stride, int ldc) {
    CHECK_MCBLAS_ERROR(mcblasDgemmStridedBatched(
        handle, MCBLASBooleanToTranspose(ta), MCBLASBooleanToTranspose(tb), M, N, K, &alpha, A, lda,
        a_stride, B, ldb, b_stride, &beta, C, ldc, c_stride, batch_size));
  }
};

// Check mcblas supported mix-precision computation type and return computeType
bool CheckMixPrecisionType(DLDataType in_dtype, DLDataType out_dtype, bool int_support = true) {
  if (int_support && TypeMatch(out_dtype, kDLInt, 32)) {
    return TypeMatch(in_dtype, kDLInt, 8);
  } else if (TypeMatch(out_dtype, kDLFloat, 32)) {
    return TypeMatch(in_dtype, kDLInt, 8) || TypeMatch(in_dtype, kDLFloat, 16);
  } else {
    return false;
  }
}

int roundoff(int v, int d) { return (v + d - 1) / d * d; }

void CallMcblasLt(mcblasLtHandle_t hdl, mcStream_t stream,
                  mcblasLtMatmulPreference_t matmul_pref_desc, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* scaleA, const DLTensor* scaleB,
                  const DLTensor* C, bool transa, bool transb, void* workspace_ptr,
                  size_t workspace_size, mcblasLtEpilogue_t epilogue,
                  std::optional<float> dq_scale) {
  ICHECK(TypeEqual(A->dtype, B->dtype));
  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  auto compute_type = MCBLAS_COMPUTE_32F;
  auto scale_type = MACA_R_32F;
  macaDataType_t ab_type = MACA_R_32F;
  macaDataType_t c_type = MACA_R_32F;
  float one_fp32 = 1.0;
  float zero_fp32 = 0.0;
  int32_t one_i32 = 1;
  int32_t zero_i32 = 0;
  // Pass dequantization scale through the "alpha" parameter. If there is no dequantization after
  // matmul, then alpha == 1.0
  float alpha_value = dq_scale.value_or(one_fp32);
  void* alpha = &alpha_value;
  void* beta = &zero_fp32;

  if (TypeMatch(A->dtype, kDLFloat, 16)) {
    ab_type = MACA_R_16F;
  } else if (TypeMatch(A->dtype, kDLInt, 8)) {
    ab_type = MACA_R_8I;
  }
  //  TODO: Support MACA_R_8F_E4M3 in mcblas
  // else if (TypeMatch(A->dtype, DataType::TypeCode::kFloat8_e4m3fn, 8)) {
  //   ICHECK(TypeMatch(B->dtype, DataType::TypeCode::kFloat8_e4m3fn, 8));
  //   ab_type = MACA_R_8F_E4M3;
  // }

  if (TypeMatch(C->dtype, kDLFloat, 16)) {
    c_type = MACA_R_16F;
  } else if (TypeMatch(C->dtype, kDLInt, 32)) {
    c_type = MACA_R_32I;
    compute_type = MCBLAS_COMPUTE_32I;
    scale_type = MACA_R_32I;
    alpha = &one_i32;
    beta = &zero_i32;
  }

  mcblasLtMatmulDesc_t op_desc;
  mcblasOperation_t op_transa = MCBLASBooleanToTranspose(transa);
  mcblasOperation_t op_transb = MCBLASBooleanToTranspose(transb);

  CHECK_MCBLAS_ERROR(mcblasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
  CHECK_MCBLAS_ERROR(mcblasLtMatmulDescSetAttribute(op_desc, MCBLASLT_MATMUL_DESC_TRANSA,
                                                    &op_transb, sizeof(op_transb)));
  CHECK_MCBLAS_ERROR(mcblasLtMatmulDescSetAttribute(op_desc, MCBLASLT_MATMUL_DESC_TRANSB,
                                                    &op_transa, sizeof(op_transa)));

  if (bias != nullptr) {
    CHECK_MCBLAS_ERROR(mcblasLtMatmulDescSetAttribute(op_desc, MCBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &bias->data, sizeof(float*)));
  }

  if (scaleA != nullptr) {
    auto scaleA_data = static_cast<char*>(scaleA->data) + scaleA->byte_offset;
    CHECK_MCBLAS_ERROR(mcblasLtMatmulDescSetAttribute(op_desc, MCBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                      &scaleA_data, sizeof(float*)));
  }
  if (scaleB != nullptr) {
    auto scaleB_data = static_cast<char*>(scaleB->data) + scaleB->byte_offset;
    CHECK_MCBLAS_ERROR(mcblasLtMatmulDescSetAttribute(op_desc, MCBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                      &scaleB_data, sizeof(float*)));
  }

  if (epilogue != MCBLASLT_EPILOGUE_DEFAULT) {
    CHECK_MCBLAS_ERROR(mcblasLtMatmulDescSetAttribute(op_desc, MCBLASLT_MATMUL_DESC_EPILOGUE,
                                                      &epilogue, sizeof(epilogue)));
  }

  int batch_offset_A = A->ndim - 2;
  int batch_offset_B = B->ndim - 2;

  int M = ColumnCount(B, transb, batch_offset_B);
  int N = RowCount(A, transa, batch_offset_A);
  int K = ColumnCount(A, transa, batch_offset_A);
  bool use_batched_gemm = A->ndim > 2 || B->ndim > 2;

  // If A is batched but B is not, flatten all non-reduction axes of A to use the regular GEMM.
  // This trick is only applicable if batch axes and the other spatial axis (M or N) are
  // adjacent in both the input and the output matrix. In particular, if A is of shape (M, K)
  // and B matrix is of shape (Batch, N, K) with transb = true, the output shape
  // is (Batch, M, N). Since the Batch and the N axes are not adjacent in the output, we cannot
  // use the regular GEMM if only B is batched.
  if (A->ndim > 2 && B->ndim == 2 && transa == false) {
    N = 1;
    for (int i = 0; i < A->ndim - 1; ++i) {
      N *= A->shape[i];
    }
    use_batched_gemm = false;
  }

  int lda = transb ? K : M;
  int ldb = transa ? N : K;
  int ldc = M;

  mcblasLtMatrixLayout_t A_desc, B_desc, C_desc;
  CHECK_MCBLAS_ERROR(
      mcblasLtMatrixLayoutCreate(&A_desc, ab_type, !transb ? M : K, !transb ? K : M, lda));
  CHECK_MCBLAS_ERROR(
      mcblasLtMatrixLayoutCreate(&B_desc, ab_type, !transa ? K : N, !transa ? N : K, ldb));
  CHECK_MCBLAS_ERROR(mcblasLtMatrixLayoutCreate(&C_desc, c_type, M, N, ldc));

  if (use_batched_gemm) {
    auto get_batch_count = [](int64_t* shape, int batch_offset) {
      int64_t count = 1;
      for (int i = 0; i < batch_offset; ++i) {
        count *= shape[i];
      }
      return count;
    };
    auto set_batch = [](mcblasLtMatrixLayout_t mat_desc, int batch_count, int64_t batch_stride) {
      CHECK_MCBLAS_ERROR(mcblasLtMatrixLayoutSetAttribute(
          mat_desc, MCBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
      CHECK_MCBLAS_ERROR(
          mcblasLtMatrixLayoutSetAttribute(mat_desc, MCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                           &batch_stride, sizeof(batch_stride)));
    };

    int batch_count_A = get_batch_count(A->shape, batch_offset_A);
    int batch_count_B = get_batch_count(B->shape, batch_offset_B);
    int batch_count_C = get_batch_count(C->shape, C->ndim - 2);
    int64_t batch_stride_A = M * K;
    int64_t batch_stride_B = K * N;
    int64_t batch_stride_C = M * N;

    // McblasLt does not seem to support batched GEMM with one of matrices having
    // one batch (with batch_stride 0).
    ICHECK_EQ(batch_count_A, batch_count_B);

    set_batch(A_desc, batch_count_A, batch_stride_A);
    set_batch(B_desc, batch_count_B, batch_stride_B);
    set_batch(C_desc, batch_count_C, batch_stride_C);
  }

  auto A_data = static_cast<char*>(A->data) + A->byte_offset;
  auto B_data = static_cast<char*>(B->data) + B->byte_offset;
  auto C_data = static_cast<char*>(C->data) + C->byte_offset;

  mcblasLtMatmulPreferenceSetAttribute(matmul_pref_desc, MCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &workspace_size, sizeof(size_t));

  mcblasLtMatmulHeuristicResult_t heuristic_result = {};
  int returned_result = 0;
  CHECK_MCBLAS_ERROR(mcblasLtMatmulAlgoGetHeuristic(hdl, op_desc, A_desc, B_desc, C_desc, C_desc,
                                                    matmul_pref_desc, 1, &heuristic_result,
                                                    &returned_result));
  if (returned_result == 0) {
    CHECK_MCBLAS_ERROR(MCBLAS_STATUS_NOT_SUPPORTED);
  }

  CHECK_MCBLAS_ERROR(mcblasLtMatmul(hdl, op_desc, alpha, B_data, A_desc, A_data, B_desc, beta,
                                    C_data, C_desc, C_data, C_desc, &heuristic_result.algo,
                                    workspace_ptr, workspace_size, stream));

  mcblasLtMatmulDescDestroy(op_desc);
  mcblasLtMatrixLayoutDestroy(A_desc);
  mcblasLtMatrixLayoutDestroy(B_desc);
  mcblasLtMatrixLayoutDestroy(C_desc);
}

inline void CallLtIgemm(TVMArgs args, TVMRetValue* ret, mcblasLtHandle_t hdl, mcStream_t stream) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;
  int M = ColumnCount(B, transb);
  int N = RowCount(A, transa);
  int K = ColumnCount(A, transa);
  int N_out = ColumnCount(C, false);
  int m = M;
  int n = m;
  int k = m;
  int lda = M * K / (roundoff(K, 32) / 32);
  int ldb = K * N / (roundoff(K, 32) / 32);
  int ldc = M * N_out / (roundoff(N_out, 32) / 32);
  ICHECK_EQ(A->ndim, 2);
  ICHECK_EQ(B->ndim, 2);
  ICHECK_EQ(C->ndim, 2);

  ICHECK_EQ(ElementStride(A), 1);
  ICHECK_EQ(ElementStride(B), 1);
  ICHECK_EQ(ElementStride(C), 1);

  ICHECK(TypeEqual(A->dtype, B->dtype));
  ICHECK(TypeMatch(A->dtype, kDLInt, 8));
  ICHECK(TypeMatch(C->dtype, kDLInt, 32));

  ICHECK(CheckMixPrecisionType(A->dtype, C->dtype)) << "Unsupported data type";
  int32_t alpha = args.size() > 5 ? args[5] : 1;
  int32_t beta = args.size() > 6 ? args[6] : 0;
  mcblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  auto A_data = reinterpret_cast<void*>(static_cast<char*>(A->data) + A->byte_offset);
  auto B_data = reinterpret_cast<void*>(static_cast<char*>(B->data) + B->byte_offset);
  auto C_data = reinterpret_cast<void*>(static_cast<char*>(C->data) + C->byte_offset);

  mcblasLtOrder_t order_COL32 = MCBLASLT_ORDER_COL32;
  mcblasLtOrder_t order_COL4_4R2_8C = MCBLASLT_ORDER_COL4_4R2_8C;
  mcblasLtMatmulDesc_t operationDesc = nullptr;
  CHECK_MCBLAS_ERROR(mcblasLtMatmulDescCreate(&operationDesc, MCBLAS_COMPUTE_32I, MACA_R_32I));
  mcblasOperation_t opTransA = MCBLASBooleanToTranspose(transa);
  mcblasOperation_t opTransB = MCBLASBooleanToTranspose(transb);
  CHECK_MCBLAS_ERROR(mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_TRANSA,
                                                    &opTransA, sizeof(opTransA)));
  CHECK_MCBLAS_ERROR(mcblasLtMatmulDescSetAttribute(operationDesc, MCBLASLT_MATMUL_DESC_TRANSB,
                                                    &opTransB, sizeof(opTransB)));
  // Create descriptors for the original matrices
  CHECK_MCBLAS_ERROR(mcblasLtMatrixLayoutCreate(&Adesc, MACA_R_8I, opTransA == MCBLAS_OP_N ? m : k,
                                                opTransA == MCBLAS_OP_N ? k : m, lda));
  CHECK_MCBLAS_ERROR(mcblasLtMatrixLayoutCreate(&Bdesc, MACA_R_8I, opTransB == MCBLAS_OP_N ? k : n,
                                                opTransB == MCBLAS_OP_N ? n : k, ldb));
  CHECK_MCBLAS_ERROR(mcblasLtMatrixLayoutCreate(&Cdesc, MACA_R_32I, m, n, ldc));

  CHECK_MCBLAS_ERROR(mcblasLtMatrixLayoutSetAttribute(Adesc, MCBLASLT_MATRIX_LAYOUT_ORDER,
                                                      &order_COL32, sizeof(order_COL32)));
  CHECK_MCBLAS_ERROR(mcblasLtMatrixLayoutSetAttribute(
      Bdesc, MCBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));
  CHECK_MCBLAS_ERROR(mcblasLtMatrixLayoutSetAttribute(Cdesc, MCBLASLT_MATRIX_LAYOUT_ORDER,
                                                      &order_COL32, sizeof(order_COL32)));

  CHECK_MCBLAS_ERROR(mcblasLtMatmul(hdl, operationDesc, &alpha, B_data, Adesc, A_data, Bdesc, &beta,
                                    C_data, Cdesc, C_data, Cdesc, nullptr, nullptr, 0, stream));
}

inline void CallGemmEx(TVMArgs args, TVMRetValue* ret, mcblasHandle_t hdl) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  ICHECK_EQ(A->ndim, 2);
  ICHECK_EQ(B->ndim, 2);
  ICHECK_EQ(C->ndim, 2);

  ICHECK_EQ(ElementStride(A), 1);
  ICHECK_EQ(ElementStride(B), 1);
  ICHECK_EQ(ElementStride(C), 1);

  ICHECK(TypeEqual(A->dtype, B->dtype));

  // C can never be transposed.
  ICHECK(!IsInPlaceTransposed(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  ICHECK(CheckMixPrecisionType(A->dtype, C->dtype)) << "Unsupported data type";
  ICHECK(!TypeMatch(A->dtype, kDLInt, 8) || ColumnStride(A) % 4 == 0)
      << "leading dimension must divide 4 for int8 gemm";
  ICHECK(!TypeMatch(B->dtype, kDLInt, 8) || ColumnStride(B) % 4 == 0)
      << "leading dimension must divide 4 for int8 gemm";
  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;

  macaDataType_t maca_in_type = GetMacaDataType(A->dtype);
  macaDataType_t maca_out_type = GetMacaDataType(C->dtype);
  mcblasGemmAlgo_t algo = MCBLAS_GEMM_DEFAULT;
  void *alpha_ptr = nullptr, *beta_ptr = nullptr;
  auto alpha_int = static_cast<int32_t>(alpha);
  auto beta_int = static_cast<int32_t>(beta);
  auto alpha_float = static_cast<float>(alpha);
  auto beta_float = static_cast<float>(beta);
  if (C->dtype.code == kDLInt) {
    alpha_ptr = &alpha_int;
    beta_ptr = &beta_int;
  } else if (C->dtype.code == kDLFloat) {
    alpha_ptr = &alpha_float;
    beta_ptr = &beta_float;
  }

  auto A_data = reinterpret_cast<void*>(static_cast<char*>(A->data) + A->byte_offset);
  auto B_data = reinterpret_cast<void*>(static_cast<char*>(B->data) + B->byte_offset);
  auto C_data = reinterpret_cast<void*>(static_cast<char*>(C->data) + C->byte_offset);

  CHECK_MCBLAS_ERROR(
      mcblasGemmEx(hdl, MCBLASBooleanToTranspose(transb), MCBLASBooleanToTranspose(transa),
                   ColumnCount(B, transb), RowCount(A, transa), ColumnCount(A, transa), alpha_ptr,
                   B_data, maca_in_type, ColumnStride(B), A_data, maca_in_type, ColumnStride(A),
                   beta_ptr, C_data, maca_out_type, ColumnStride(C), maca_out_type, algo));
}

inline void CallBatchGemmEx(TVMArgs args, TVMRetValue* ret, mcblasHandle_t hdl) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  ICHECK_EQ(A->ndim, 3);
  ICHECK_EQ(B->ndim, 3);
  ICHECK_EQ(C->ndim, 3);

  int batch_size = BatchCount3D(C);
  ICHECK_EQ(ElementStride3D(A), 1);
  ICHECK_EQ(ElementStride3D(B), 1);
  ICHECK_EQ(ElementStride3D(C), 1);

  ICHECK(TypeEqual(A->dtype, B->dtype));

  // C can never be transposed.
  ICHECK(!IsInPlaceTransposed3D(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed3D(A) ? !transa : transa;
  transb = IsInPlaceTransposed3D(B) ? !transb : transb;

  ICHECK(CheckMixPrecisionType(A->dtype, C->dtype, true)) << "Unsupported data type";
  ICHECK(!TypeMatch(A->dtype, kDLInt, 8) || ColumnStride3D(A) % 4 == 0)
      << "leading dimension must divide 4 for int8 gemm";
  ICHECK(!TypeMatch(B->dtype, kDLInt, 8) || ColumnStride3D(B) % 4 == 0)
      << "leading dimension must divide 4 for int8 gemm";
  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;

  int A_stride = A->shape[1] * A->shape[2];
  int B_stride = B->shape[1] * B->shape[2];
  int C_stride = C->shape[1] * C->shape[2];

  // Broadcast A or B by changing its stride.
  int batch_size_a = BatchCount3D(A);
  int batch_size_b = BatchCount3D(B);
  if (batch_size_a != batch_size_b) {
    if (batch_size_a == 1) {
      A_stride = 0;
    } else if (batch_size_b == 1) {
      B_stride = 0;
    }
  } else {
    ICHECK_EQ(batch_size_a, batch_size);
    ICHECK_EQ(batch_size_b, batch_size);
  }

  macaDataType_t maca_in_type = GetMacaDataType(A->dtype);
  macaDataType_t maca_out_type = GetMacaDataType(C->dtype);
  mcblasGemmAlgo_t algo = MCBLAS_GEMM_DEFAULT;
  void *alpha_ptr = nullptr, *beta_ptr = nullptr;
  auto alpha_int = static_cast<int32_t>(alpha);
  auto beta_int = static_cast<int32_t>(beta);
  auto alpha_float = static_cast<float>(alpha);
  auto beta_float = static_cast<float>(beta);
  if (C->dtype.code == kDLInt) {
    alpha_ptr = &alpha_int;
    beta_ptr = &beta_int;
  } else if (C->dtype.code == kDLFloat) {
    alpha_ptr = &alpha_float;
    beta_ptr = &beta_float;
  }

  auto A_data = reinterpret_cast<void*>(static_cast<char*>(A->data) + A->byte_offset);
  auto B_data = reinterpret_cast<void*>(static_cast<char*>(B->data) + B->byte_offset);
  auto C_data = reinterpret_cast<void*>(static_cast<char*>(C->data) + C->byte_offset);
  CHECK_MCBLAS_ERROR(mcblasGemmStridedBatchedEx(
      hdl, MCBLASBooleanToTranspose(transb), MCBLASBooleanToTranspose(transa),
      ColumnCount3D(B, transb), RowCount3D(A, transa), ColumnCount3D(A, transa), alpha_ptr, B_data,
      maca_in_type, ColumnStride3D(B), B_stride, A_data, maca_in_type, ColumnStride3D(A), A_stride,
      beta_ptr, C_data, maca_out_type, ColumnStride3D(C), C_stride, batch_size, maca_out_type,
      algo));
}

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.mcblas.matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  DLTensor* C = args[2];

  McBlasThreadEntry* entry_ptr = McBlasThreadEntry::ThreadLocal();

  MCBLASTryEnableTensorCore(entry_ptr->handle);

  if (TypeEqual(A->dtype, C->dtype)) {
    ICHECK(TypeMatch(A->dtype, kDLFloat, 16) || TypeMatch(A->dtype, kDLFloat, 32) ||
           TypeMatch(A->dtype, kDLFloat, 64));

    if (TypeMatch(A->dtype, kDLFloat, 16))
      CallGemm(args, ret, McblasHgemmOp(entry_ptr->handle));
    else if (TypeMatch(A->dtype, kDLFloat, 32))
      CallGemm(args, ret, McblasSgemmOp(entry_ptr->handle));
    else
      CallGemm(args, ret, McblasDgemmOp(entry_ptr->handle));
  } else {
    CallGemmEx(args, ret, entry_ptr->handle);
  }
});

TVM_REGISTER_GLOBAL("tvm.contrib.mcblaslt.matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];

  McBlasThreadEntry* entry_ptr = McBlasThreadEntry::ThreadLocal();

  MCBLASTryEnableTensorCore(entry_ptr->handle);

  ICHECK(TypeMatch(A->dtype, kDLInt, 8)) << "Expects dtype to be int8\n";
  mcblasLtHandle_t ltHandle;
  CHECK_MCBLAS_ERROR(mcblasLtCreate(&ltHandle));
  auto func = tvm::runtime::Registry::Get("runtime.get_maca_stream");
  ICHECK(func != nullptr);
  mcStream_t stream = static_cast<mcStream_t>((*func)().operator void*());
  CallLtIgemm(args, ret, ltHandle, stream);
  CHECK_MCBLAS_ERROR(mcblasLtDestroy(ltHandle));
});

TVM_REGISTER_GLOBAL("tvm.contrib.mcblas.batch_matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  DLTensor* C = args[2];

  McBlasThreadEntry* entry_ptr = McBlasThreadEntry::ThreadLocal();

  MCBLASTryEnableTensorCore(entry_ptr->handle);
  if (TypeEqual(A->dtype, C->dtype)) {
    ICHECK(TypeMatch(A->dtype, kDLFloat, 16) || TypeMatch(A->dtype, kDLFloat, 32) ||
           TypeMatch(A->dtype, kDLFloat, 64));

    if (TypeMatch(A->dtype, kDLFloat, 16))
      CallBatchGemm(args, ret, McblasHgemmBatchOp(entry_ptr->handle));
    else if (TypeMatch(A->dtype, kDLFloat, 32))
      CallBatchGemm(args, ret, McblasSgemmBatchOp(entry_ptr->handle));
    else
      CallBatchGemm(args, ret, McblasDgemmBatchOp(entry_ptr->handle));
  } else {
    CallBatchGemmEx(args, ret, entry_ptr->handle);
  }
});

}  // namespace contrib
}  // namespace tvm
