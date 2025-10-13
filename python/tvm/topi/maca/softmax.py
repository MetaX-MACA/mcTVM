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
# pylint: disable=invalid-name, unused-variable, trailing-whitespace
"""Schedule for softmax operator"""
from tvm.target import Target
from tvm import te
from tvm.contrib import mcdnn
from .. import generic
from .injective import schedule_injective_from_existing
from ..utils import get_const_int, traverse_inline

def softmax_mcdnn(x, axis=-1):
    """Perform softmax on the data using mcdnn"""
    return mcdnn.softmax(x, axis)


def schedule_softmax_mcdnn(outs):
    """Schedule for softmax mcdnn op"""
    return generic.schedule_extern(outs)


def log_softmax_mcdnn(x, axis=-1):
    """Perform log_softmax on the data using mcdnn"""
    return mcdnn.log_softmax(x, axis)


def schedule_log_softmax_mcdnn(outs):
    """Schedule for log_softmax mcdnn op"""
    return generic.schedule_extern(outs)
