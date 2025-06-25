# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""External function interface to FlashAttn libraries."""
import tvm
from tvm import te


def mha_fwd_inference(query, key, value, mask, is_causal=False):
    """Create an extern op that compute MultiHeadAttention with FlashAttn

    Parameters
    ----------
    query : Tensor
        The left matrix operand
    key : Tensor
        The right matrix operand
    value : Tensor
        The right matrix operand
    mask : Tensor
        The right matrix operand
    is_causal : bool
        Whether transpose lhs

    Returns
    -------
    Out : Tensor
        The result tensor.
    """
    return te.extern(
        query.shape,
        [query, key, value, mask],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.flashattn.mha_fwd_inference",
            ins[0],
            ins[1],
            ins[2],
            ins[3],
            outs[0],
            is_causal,
        ),
        dtype=query.dtype,
        name="flashattn_mha_fwd_inference",
    )
