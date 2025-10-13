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

#include "mcdnn_utils.h"

#include <dmlc/thread_local.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <vector>

namespace tvm {
namespace contrib {
namespace maca {
// McDNN Data Type
mcdnnDataType_t McDNNDataType::DLTypeToMcDNNType(const DLDataType& dtype) {
  switch (dtype.code) {
    case kDLInt:
      if (dtype.bits == 8 && dtype.lanes == 1)
        return MCDNN_DATA_INT8;
      else if (dtype.bits == 32 && dtype.lanes == 1)
        return MCDNN_DATA_INT32;
      else if (dtype.bits == 8 && dtype.lanes == 4)
        return MCDNN_DATA_INT8x4;
      else
        LOG(FATAL) << "Unsupported type";
      break;
    case kDLUInt:
      LOG(FATAL) << "Unsupported type";
      break;
    case kDLFloat:
      if (dtype.bits == 32 && dtype.lanes == 1)
        return MCDNN_DATA_FLOAT;
      else if (dtype.bits == 64 && dtype.lanes == 1)
        return MCDNN_DATA_DOUBLE;
      else if (dtype.bits == 16 && dtype.lanes == 1)
        return MCDNN_DATA_HALF;
      else
        LOG(FATAL) << "Unsupported type";
      break;
  }
  return MCDNN_DATA_FLOAT;
}

template <>
const void* McDNNDataType::GetConst<0>(mcdnnDataType_t type) {
  static const int int_v = 0;
  static const float float_v = 0;
  static const double double_v = 0;
  if (type == MCDNN_DATA_FLOAT || type == MCDNN_DATA_HALF) {
    return static_cast<const void*>(&float_v);
  }
  if (type == MCDNN_DATA_DOUBLE) {
    return static_cast<const void*>(&double_v);
  }
  if (type == MCDNN_DATA_INT8 || type == MCDNN_DATA_INT32 || type == MCDNN_DATA_INT8x4) {
    return static_cast<const void*>(&int_v);
  }
  return nullptr;
}

template <>
const void* McDNNDataType::GetConst<1>(mcdnnDataType_t type) {
  static const int int_v = 1;
  static const float float_v = 1.f;
  static const double double_v = 1.f;
  if (type == MCDNN_DATA_FLOAT || type == MCDNN_DATA_HALF) {
    return static_cast<const void*>(&float_v);
  }
  if (type == MCDNN_DATA_DOUBLE) {
    return static_cast<const void*>(&double_v);
  }
  if (type == MCDNN_DATA_INT8 || type == MCDNN_DATA_INT32 || type == MCDNN_DATA_INT8x4) {
    return static_cast<const void*>(&int_v);
  }
  return nullptr;
}

// McDNNThreadEntry

McDNNThreadEntry::McDNNThreadEntry() {
  auto stream = runtime::MACAThreadEntry::ThreadLocal()->stream;
  auto func = runtime::Registry::Get("device_api.maca");
  void* ret = (*func)();
  maca_api = static_cast<runtime::DeviceAPI*>(ret);

  // If no McDNN-capable device is present, allow the McDNNThreadEntry
  // object to be created.  This is needed for
  // McDNNThreadEntry::exists.
  {
    mcdnnStatus_t create_res = mcdnnCreate(&handle);
    if (create_res == MCDNN_STATUS_NOT_INITIALIZED) {
      return;
    }
    MCDNN_CALL(create_res);
  }

  MCDNN_CALL(mcdnnSetStream(handle, stream));
  conv_entry.maca_api = maca_api;
}

McDNNThreadEntry::~McDNNThreadEntry() {}

typedef dmlc::ThreadLocalStore<McDNNThreadEntry> McDNNThreadStore;

McDNNThreadEntry* McDNNThreadEntry::ThreadLocal(bool check_exists) {
  auto* res = McDNNThreadStore::Get();
  if (check_exists) {
    ICHECK(res->exists()) << "MCDNN_STATUS_NOT_INITIALIZED";
  }

  return res;
}

// ConvEntry

ConvEntry::ConvEntry() {
  MCDNN_CALL(mcdnnCreateConvolutionDescriptor(&conv_desc));
  MCDNN_CALL(mcdnnCreateFilterDescriptor(&filter_desc));
  MCDNN_CALL(mcdnnCreateTensorDescriptor(&input_desc));
  MCDNN_CALL(mcdnnCreateTensorDescriptor(&output_desc));
  MCDNN_CALL(mcdnnCreateTensorDescriptor(&bias_desc));
  MCDNN_CALL(mcdnnCreateActivationDescriptor(&activation_desc));
}

ConvEntry::~ConvEntry() {
  MCDNN_CALL(mcdnnDestroyFilterDescriptor(filter_desc));
  MCDNN_CALL(mcdnnDestroyConvolutionDescriptor(conv_desc));
  MCDNN_CALL(mcdnnDestroyTensorDescriptor(input_desc));
  MCDNN_CALL(mcdnnDestroyTensorDescriptor(output_desc));
  MCDNN_CALL(mcdnnDestroyTensorDescriptor(bias_desc));
  MCDNN_CALL(mcdnnDestroyActivationDescriptor(activation_desc));
  CleanWorkspace();
}

void ConvEntry::UpdateWorkspace(const size_t wsize) {
  if (workspace_size < wsize) {
    if (workspace != nullptr) {
      CleanWorkspace();
    }
    workspace_size = wsize;
    workspace = maca_api->AllocWorkspace(device, workspace_size);
  }
}

void ConvEntry::CleanWorkspace() {
  if (workspace) maca_api->FreeWorkspace(device, workspace);
  workspace_size = 0;
}

void SetConvDescriptors(McDNNThreadEntry* entry_ptr, int format, int dims, int groups,
                        const int pad[], const int stride[], const int dilation[], int64_t x_dim[],
                        int64_t w_dim[], int64_t y_dim[], DLDataType data_dtype,
                        const std::string& conv_dtype) {
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<mcdnnTensorFormat_t>(format);
  // Set Data Type
  entry_ptr->conv_entry.data_type =
      McDNNDataType::DLTypeToMcDNNType(runtime::String2DLDataType(conv_dtype));

  mcdnnDataType_t mcdnn_data_type = McDNNDataType::DLTypeToMcDNNType(data_dtype);

  // Dims includes N and C
  int full_dims = dims + 2;

  std::vector<int> dim(full_dims);
  std::vector<int> tensor_stride(full_dims);

  // Note: For 2D tenor, using ND setters causes MCDNN_STATUS_NOT_SUPPORTED error
  // in following mcdnnGetConvolutionForwardWorkspaceSize() when data type is fp16, int

  // MCDNN_CALL(mcdnnSetConvolutionGroupCount(entry_ptr->conv_entry.conv_desc, groups));
  if (dims == 2) {
    // Set Desc
    MCDNN_CALL(mcdnnSetConvolution2dDescriptor(
        entry_ptr->conv_entry.conv_desc, pad[0], pad[1], stride[0], stride[1], dilation[0],
        dilation[1], entry_ptr->conv_entry.mode, entry_ptr->conv_entry.data_type));
    int ni, ci, hi, wi;
    if (entry_ptr->conv_entry.tensor_format == MCDNN_TENSOR_NHWC) {
      ni = 0;
      ci = 3;
      hi = 1;
      wi = 2;
    } else {
      ni = 0;
      ci = 1;
      hi = 2;
      wi = 3;
    }

    // Set Input
    MCDNN_CALL(mcdnnSetTensor4dDescriptor(
        entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.tensor_format, mcdnn_data_type,
        static_cast<int>(x_dim[ni]), static_cast<int>(x_dim[ci]), static_cast<int>(x_dim[hi]),
        static_cast<int>(x_dim[wi])));
    // Set Filter
    MCDNN_CALL(mcdnnSetFilter4dDescriptor(
        entry_ptr->conv_entry.filter_desc, mcdnn_data_type, entry_ptr->conv_entry.tensor_format,
        static_cast<int>(w_dim[ni]), static_cast<int>(w_dim[ci]), static_cast<int>(w_dim[hi]),
        static_cast<int>(w_dim[wi])));
    // Set Output
    MCDNN_CALL(mcdnnSetTensor4dDescriptor(
        entry_ptr->conv_entry.output_desc, entry_ptr->conv_entry.tensor_format, mcdnn_data_type,
        static_cast<int>(y_dim[ni]), static_cast<int>(y_dim[ci]), static_cast<int>(y_dim[hi]),
        static_cast<int>(y_dim[wi])));
  } else {
    ICHECK_EQ(format, 0) << "Use of layout MCDNN_TENSOR_NHWC is supported only for 4-D tensors.";

    MCDNN_CALL(mcdnnSetConvolutionNdDescriptor(entry_ptr->conv_entry.conv_desc, dims, pad, stride,
                                               dilation, entry_ptr->conv_entry.mode,
                                               entry_ptr->conv_entry.data_type));

    // Set Filter
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(w_dim[i]);
    }
    MCDNN_CALL(mcdnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc, mcdnn_data_type,
                                          entry_ptr->conv_entry.tensor_format, full_dims,
                                          dim.data()));
    // Set Input
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(x_dim[i]);
    }
    GetMcdnnStride(full_dims, dim.data(), tensor_stride.data());
    MCDNN_CALL(mcdnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc, mcdnn_data_type,
                                          full_dims, dim.data(), tensor_stride.data()));
    // Set Output
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(y_dim[i]);
    }
    GetMcdnnStride(full_dims, dim.data(), tensor_stride.data());
    MCDNN_CALL(mcdnnSetTensorNdDescriptor(entry_ptr->conv_entry.output_desc, mcdnn_data_type,
                                          full_dims, dim.data(), tensor_stride.data()));
  }
  MCDNN_CALL(mcdnnSetConvolutionGroupCount(entry_ptr->conv_entry.conv_desc, groups));
  if (mcdnnGetVersion() > 7000) {
    MCDNN_CALL(mcdnnSetConvolutionMathType(entry_ptr->conv_entry.conv_desc, MCDNN_TENSOR_OP_MATH))
  }
}

// SoftmaxEntry

SoftmaxEntry::SoftmaxEntry() { MCDNN_CALL(mcdnnCreateTensorDescriptor(&shape_desc)); }

SoftmaxEntry::~SoftmaxEntry() { MCDNN_CALL(mcdnnDestroyTensorDescriptor(shape_desc)); }

TVM_REGISTER_GLOBAL("tvm.contrib.mcdnn.exists").set_body_typed([]() -> bool {
  return McDNNThreadEntry::ThreadLocal(false)->exists();
});
}  // namespace maca
}  // namespace contrib
}  // namespace tvm
