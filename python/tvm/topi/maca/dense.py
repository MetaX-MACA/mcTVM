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
# pylint: disable=invalid-name, unused-argument
"""Schedule for dense operator"""
import logging
import tvm
from tvm import te, autotvm
from tvm.contrib import mcblas
from .tensor_intrin import dp4a
from .. import tag
from .. import generic
from ..utils import traverse_inline, get_const_tuple

logger = logging.getLogger("topi")


def _matmul_mcblas_common(
    cfg, tensor_a, tensor_b, bias=None, out_dtype=None, transpose_a=False, transpose_b=False
):
    assert len(tensor_a.shape) == 2 and len(tensor_b.shape) == 2, "only support 2-dim matmul"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = tensor_a.dtype
    if out_dtype not in [tensor_a.dtype, "int32"]:
        assert out_dtype == tensor_a.dtype, "Mixed precision other than int8 + int32 not supported."
    batch, in_dim = get_const_tuple(tensor_a.shape)
    out_dim, _ = get_const_tuple(tensor_b.shape)
    matmul = mcblas.matmul(tensor_a, tensor_b, transpose_a, transpose_b, dtype=out_dtype)
    if all(isinstance(d, int) for d in [batch, in_dim, out_dim]):
        cfg.add_flop(batch * in_dim * out_dim * 2)
    if bias is not None:
        matmul = te.compute(
            (batch, out_dim), lambda i, j: matmul[i, j] + bias[j], tag=tag.BROADCAST
        )
    return matmul


@autotvm.register_topi_compute("matmul_mcblas.maca")
def matmul_mcblas(
    cfg, tensor_a, tensor_b, bias=None, out_dtype=None, transpose_a=False, transpose_b=False
):
    """Matmul operator on MACA with MCBLAS"""
    return _matmul_mcblas_common(cfg, tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b)


@autotvm.register_topi_schedule("matmul_mcblas.maca")
def schedule_matmul_mcblas(_, outs):
    """Schedule matmul operator using MCBLAS"""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("dense_mcblas.maca")
def dense_mcblas(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator on MACA with MCBLAS. This is an alias of matmul_nt operator."""
    return _matmul_mcblas_common(cfg, data, weight, bias, out_dtype, False, True)


@autotvm.register_topi_schedule("dense_mcblas.maca")
def schedule_dense_mcblas(_, outs):
    """Schedule dense operator using MCBLAS"""
    return generic.schedule_extern(outs)
