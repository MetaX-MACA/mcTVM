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
 * \file Use external FlashAttn library call.
 */
#include <flash_attn/flash_attn.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <cmath>

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.flashattn.mha_fwd_inference")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* query = args[0];
      DLTensor* key = args[1];
      DLTensor* value = args[2];
      DLTensor* mask = args[3];
      DLTensor* output = args[4];
      bool is_causal = args[5];
      const int64_t batch = query->shape[0];
      const int64_t seqlen_q = query->shape[1];
      const int64_t seqlen_k = key->shape[1];
      const int64_t head_dim = key->shape[3];
      const int64_t num_heads = query->shape[2];
      const int64_t num_kv_heads = key->shape[2];
      const float attn_scale = float(1.0 / std::sqrt(float(head_dim)));
      Tensor_t q = make_contiguous_tensor4d(query->data, MCFLASHATTN_DATATYPE_FP16, batch, seqlen_q,
                                            num_heads, head_dim);
      Tensor_t k = make_contiguous_tensor4d(key->data, MCFLASHATTN_DATATYPE_FP16, batch, seqlen_k,
                                            num_kv_heads, head_dim);
      Tensor_t v = make_contiguous_tensor4d(value->data, MCFLASHATTN_DATATYPE_FP16, batch, seqlen_k,
                                            num_kv_heads, head_dim);
      Tensor_t attn_mask =
          make_contiguous_tensor4d(mask->data, MCFLASHATTN_DATATYPE_FP16, mask->shape[0],
                                   mask->shape[1], mask->shape[2], mask->shape[3]);
      Tensor_t out = make_contiguous_tensor4d(output->data, MCFLASHATTN_DATATYPE_FP16, batch,
                                              seqlen_q, num_heads, head_dim);
      // auto func = tvm::runtime::Registry::Get("runtime.get_maca_stream");
      // ICHECK(func != nullptr);
      // mcStream_t stream = static_cast<mcStream_t>((*func)().operator void*());
      auto status =
          mha_fwd_inference(batch, seqlen_q, num_heads, seqlen_k, num_heads, head_dim, q, k, v, out,
                            NULL, attn_mask, attn_scale, is_causal, -1, -1, nullptr);
    });

}  // namespace contrib
}  // namespace tvm
