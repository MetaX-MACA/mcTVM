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
# pylint: disable=invalid-name,too-many-locals,unused-variable,unused-argument
"""maca batch_matmul operators"""
import tvm
from tvm import autotvm
from tvm import te
from tvm.contrib import mcblas
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from .. import nn, generic
from ..utils import traverse_inline, get_const_tuple, get_max_power2_factor
from .tensor_intrin import dp4a

@autotvm.register_topi_compute("batch_matmul_mcblas.maca")
def batch_matmul_mcblas(
    cfg, x, y, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """Compute batch matrix multiplication of `x` and `y`.

    Both `x` and `y` can be transposed. For legacy reason, we use NT format
    (transpose_a=False, transpose_b=True) by default.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning space config file.

    x : tvm.te.Tensor
        3-D with shape [batch, M, K] or [batch, K, M].

    y : tvm.te.Tensor
        3-D with shape [batch, K, N] or [batch, N, K].

    out_shape : List[Optional]
        Explicit intended output shape of the computation. Can be useful in cases
        with dynamic input shapes.

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision batch matmul.

    transpose_a : Optional[bool] = False
        Whether the first tensor is in transposed format.

    transpose_b : Optional[bool] = True
        Whether the second tensor is in transposed format.

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    if transpose_a:
        b, k, m = get_const_tuple(x.shape)
    else:
        b, m, k = get_const_tuple(x.shape)
    if transpose_b:
        b, n, k = get_const_tuple(y.shape)
    else:
        b, k, n = get_const_tuple(y.shape)
    if all([isinstance(s, int) for s in [b, m, n, k]]):
        cfg.add_flop(b * m * k * n * 2)
    return mcblas.batch_matmul(x, y, transa=transpose_a, transb=transpose_b, dtype=out_dtype)


@autotvm.register_topi_schedule("batch_matmul_mcblas.maca")
def schedule_batch_matmul_mcblas(_, outs):
    """Schedule batch_matmul operator using MCBLAS"""
    return generic.schedule_extern(outs)
