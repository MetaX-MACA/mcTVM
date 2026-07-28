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
"""MACA TVMScript namespaces."""

from __future__ import annotations

from tvm.backend.maca import op as _maca_op
from tvm.tirx.script.builder.ir import _op_wrapper

# pylint: disable=protected-access


class MACANamespace:
    """The MACA intrinsics submodule."""

    def __init__(self):
        self.func_call = _op_wrapper(_maca_op.maca_func_call)
        self.thread_fence = _op_wrapper(_maca_op.maca_thread_fence)
        self.warp_sync = _op_wrapper(_maca_op.maca_warp_sync)
        self.cta_sync = _op_wrapper(_maca_op.maca_cta_sync)
        self.copy_bytes = _op_wrapper(_maca_op.maca_copy_bytes)
        self.copy_128b = _op_wrapper(_maca_op.maca_copy_128b)
        self.copy_64b = _op_wrapper(_maca_op.maca_copy_64b)
        self.copy_32b = _op_wrapper(_maca_op.maca_copy_32b)
        self.copy_16b = _op_wrapper(_maca_op.maca_copy_16b)
        self.copy_8b = _op_wrapper(_maca_op.maca_copy_8b)


__all__ = ["MACANamespace"]
