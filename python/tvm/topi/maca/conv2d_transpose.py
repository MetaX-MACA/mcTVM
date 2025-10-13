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
# pylint: disable=invalid-name
"""Conv2d transpose template for cuda backend"""

import tvm
from tvm import te
from tvm.contrib import mcdnn
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from .. import nn
from ..utils import get_const_tuple, traverse_inline

def conv2d_transpose_mcdnn(
    x, w, stride, padding, out_dtype, output_padding=(0, 0), layout="NCHW", groups=1
):
    """Compute conv2d_tranpose using mcdnn dgrad kernel"""
    tensor_format = 0 if layout == "NCHW" else 1
    return mcdnn.conv_backward_data(
        x,
        w,
        padding,
        stride,
        (1, 1),
        1,
        tensor_format,
        out_dtype,
        groups=groups,
        output_padding=output_padding,
    )
