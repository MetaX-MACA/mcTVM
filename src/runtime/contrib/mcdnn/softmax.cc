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
 * \file src/runtime/contrib/mcdnn/softmax.cc
 * \brief Use external mcdnn softmax function
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "mcdnn_utils.h"

namespace tvm {
namespace contrib {
namespace maca {
using namespace runtime;

void softmax_impl(mcdnnSoftmaxAlgorithm_t alg, TVMArgs args, TVMRetValue* ret) {
  DLTensor* x = args[0];
  DLTensor* y = args[1];
  int axis = args[2];
  int ndim = x->ndim;
  int64_t* shape = x->shape;
  if (axis < 0) axis += ndim;
  ICHECK(axis >= 0 && axis < ndim);

  McDNNThreadEntry* entry_ptr = McDNNThreadEntry::ThreadLocal();
  entry_ptr->softmax_entry.data_type = McDNNDataType::DLTypeToMcDNNType(x->dtype);

  // Set mode and shape descriptor
  if (axis == ndim - 1) {
    int64_t N = 1;
    for (int i = 0; i < ndim - 1; ++i) {
      N *= shape[i];
    }
    entry_ptr->softmax_entry.mode = MCDNN_SOFTMAX_MODE_INSTANCE;
    MCDNN_CALL(mcdnnSetTensor4dDescriptor(entry_ptr->softmax_entry.shape_desc, MCDNN_TENSOR_NCHW,
                                          entry_ptr->softmax_entry.data_type, static_cast<int>(N),
                                          static_cast<int>(shape[ndim - 1]), 1, 1));
  } else {
    int64_t pre_axis_dim = 1;
    int64_t post_axis_dim = 1;
    for (int i = 0; i < ndim; ++i) {
      if (i < axis) {
        pre_axis_dim *= shape[i];
      } else if (i > axis) {
        post_axis_dim *= shape[i];
      }
    }
    entry_ptr->softmax_entry.mode = MCDNN_SOFTMAX_MODE_CHANNEL;
    MCDNN_CALL(mcdnnSetTensor4dDescriptor(
        entry_ptr->softmax_entry.shape_desc, MCDNN_TENSOR_NCHW, entry_ptr->softmax_entry.data_type,
        static_cast<int>(pre_axis_dim), static_cast<int>(shape[axis]),
        static_cast<int>(post_axis_dim), 1));
  }

  auto alpha = McDNNDataType::GetConst<1>(entry_ptr->softmax_entry.data_type);
  auto beta = McDNNDataType::GetConst<0>(entry_ptr->softmax_entry.data_type);
  MCDNN_CALL(mcdnnSoftmaxForward(entry_ptr->handle, alg, entry_ptr->softmax_entry.mode, alpha,
                                 entry_ptr->softmax_entry.shape_desc, x->data, beta,
                                 entry_ptr->softmax_entry.shape_desc, y->data));
}

TVM_REGISTER_GLOBAL("tvm.contrib.mcdnn.softmax.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      softmax_impl(MCDNN_SOFTMAX_ACCURATE, args, ret);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.mcdnn.log_softmax.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) { softmax_impl(MCDNN_SOFTMAX_LOG, args, ret); });
}  // namespace maca
}  // namespace contrib
}  // namespace tvm
