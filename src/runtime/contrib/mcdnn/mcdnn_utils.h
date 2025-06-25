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
 * \file Use external mcdnn utils function
 */

#ifndef TVM_RUNTIME_CONTRIB_MCDNN_MCDNN_UTILS_H_
#define TVM_RUNTIME_CONTRIB_MCDNN_MCDNN_UTILS_H_

#include <mcdnn.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>

#include <string>

#include "../../maca/maca_common.h"

namespace tvm {
namespace contrib {
namespace maca {
#define MCDNN_CALL(func)                                                       \
  {                                                                            \
    mcdnnStatus_t e = (func);                                                  \
    ICHECK_EQ(e, MCDNN_STATUS_SUCCESS) << "mcDNN: " << mcdnnGetErrorString(e); \
  }

/*! breif Convert DLTensor type to McDNN type */
struct McDNNDataType {
  static mcdnnDataType_t DLTypeToMcDNNType(const DLDataType& dtype);
  template <int v>
  static const void* GetConst(mcdnnDataType_t type);
};  // struct McDNNDataType

inline void GetStride(int nbdim, const int* dims, int* strides) {
  int mul = 1;
  for (int i = nbdim - 1; i >= 0; --i) {
    mul *= dims[i];
    strides[i] = mul;
  }
}

inline void GetMcdnnStride(int nbdim, const int* dims, int* strides) {
  int mul = 1;
  for (int i = nbdim - 1; i >= 0; --i) {
    strides[i] = mul;
    mul *= dims[i];
  }
}

struct ConvEntry {
  mcdnnConvolutionDescriptor_t conv_desc;
  mcdnnConvolutionMode_t mode{MCDNN_CROSS_CORRELATION};
  mcdnnDataType_t data_type;
  mcdnnTensorFormat_t tensor_format;
  mcdnnTensorDescriptor_t input_desc;
  mcdnnFilterDescriptor_t filter_desc;
  mcdnnTensorDescriptor_t bias_desc;
  mcdnnActivationDescriptor_t activation_desc;
  mcdnnTensorDescriptor_t output_desc;
  mcdnnConvolutionFwdAlgo_t fwd_algo;
  mcdnnConvolutionBwdDataAlgo_t bwd_data_algo;
  mcdnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  // mcdnnMathType_t math_type;
  Device device;
  runtime::DeviceAPI* maca_api;
  void* workspace{nullptr};
  size_t workspace_size{0};
  ConvEntry();
  ~ConvEntry();
  void UpdateWorkspace(const size_t wsize);
  void CleanWorkspace();
};  // ConvThreadEntry

struct SoftmaxEntry {
  mcdnnSoftmaxMode_t mode;
  mcdnnDataType_t data_type;
  mcdnnTensorDescriptor_t shape_desc;
  SoftmaxEntry();
  ~SoftmaxEntry();
};  // SoftmaxEntry

struct McDNNThreadEntry {
  McDNNThreadEntry();
  ~McDNNThreadEntry();

  bool exists() const { return handle; }

  mcdnnHandle_t handle{nullptr};
  ConvEntry conv_entry;
  SoftmaxEntry softmax_entry;
  runtime::DeviceAPI* maca_api{nullptr};
  static McDNNThreadEntry* ThreadLocal(bool check_exists = true);
};  // McDNNThreadEntry

void SetConvDescriptors(McDNNThreadEntry* entry_ptr, int format, int dims, int groups,
                        const int pad[], const int stride[], const int dilation[], int64_t x_dim[],
                        int64_t w_dim[], int64_t y_dim[], DLDataType data_dtype,
                        const std::string& conv_dtype);

void FindAlgo(int format, int dims, int groups, const int pad[], const int stride[],
              const int dilation[], const int x_dim[], const int w_dim[], const int y_dim[],
              const std::string& data_dtype, const std::string& conv_dtype, bool verbose,
              runtime::TVMRetValue* ret);

void ConvolutionForward(int mode, int format, int algo, int dims, int groups, const int pad[],
                        const int stride[], const int dilation[], const DLTensor* x,
                        const DLTensor* w, const DLTensor* y, const std::string& conv_dtype);

void ConvolutionBiasActivationForward(int mode, int format, int algo, int dims, int groups, int act,
                                      double coef, const int pad[], const int stride[],
                                      const int dilation[], const DLTensor* x, const DLTensor* w,
                                      const DLTensor* y, const DLTensor* bias,
                                      const std::string& conv_dtype);
}  // namespace maca
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_MCDNN_MCDNN_UTILS_H_
