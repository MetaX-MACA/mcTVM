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
from tvm.tirx import is_buffer_var
from tvm.tirx import op as _tir_op
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
        self.copy_async_32b = _op_wrapper(_maca_op.maca_copy_async_32b)
        self.copy_async_64b = _op_wrapper(_maca_op.maca_copy_async_64b)
        self.copy_async_128b = _op_wrapper(_maca_op.maca_copy_async_128b)
        self.copy_async_32b_zfill = _op_wrapper(_maca_op.maca_copy_async_32b_zfill)
        self.copy_async_64b_zfill = _op_wrapper(_maca_op.maca_copy_async_64b_zfill)
        self.copy_async_128b_zfill = _op_wrapper(_maca_op.maca_copy_async_128b_zfill)
        self.async_wait_gvmcnt = _op_wrapper(_maca_op.maca_async_wait_gvmcnt)
        self.barrier_inst = _op_wrapper(_maca_op.maca_barrier_inst)
        self.copy_128b = _op_wrapper(_maca_op.maca_copy_128b)
        self.copy_64b = _op_wrapper(_maca_op.maca_copy_64b)
        self.copy_32b = _op_wrapper(_maca_op.maca_copy_32b)
        self.copy_16b = _op_wrapper(_maca_op.maca_copy_16b)
        self.copy_8b = _op_wrapper(_maca_op.maca_copy_8b)
        setattr(self, "__activemask", self._activemask)
        setattr(self, "__shfl_xor_sync", self._shfl_xor_sync)
        setattr(self, "__shfl_sync", self._shfl_sync)
        setattr(self, "__shfl_up_sync", self._shfl_up_sync)
        setattr(self, "__shfl_down_sync", self._shfl_down_sync)

    @staticmethod
    def _activemask():
        return _tir_op.call_intrin("uint64", "tirx.maca.__activemask")

    @staticmethod
    def _shfl_xor_sync(mask, value, lane_mask, width):
        if is_buffer_var(value):
            value = value[0]
        return _tir_op.call_intrin(
            value.ty, "tirx.maca.__shfl_xor_sync", mask, value, lane_mask, width
        )

    @staticmethod
    def _shfl_sync(mask, value, lane, width):
        if is_buffer_var(value):
            value = value[0]
        return _tir_op.call_intrin(value.ty, "tirx.maca.__shfl_sync", mask, value, lane, width)

    @staticmethod
    def _shfl_up_sync(mask, value, delta, width):
        if is_buffer_var(value):
            value = value[0]
        return _tir_op.call_intrin(value.ty, "tirx.maca.__shfl_up_sync", mask, value, delta, width)

    @staticmethod
    def _shfl_down_sync(mask, value, delta, width):
        if is_buffer_var(value):
            value = value[0]
        return _tir_op.call_intrin(
            value.ty, "tirx.maca.__shfl_down_sync", mask, value, delta, width
        )


__all__ = ["MACANamespace"]
