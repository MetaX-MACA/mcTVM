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
# pylint: disable=invalid-name, too-many-arguments
"""MACA TIR intrinsic builders."""

from __future__ import annotations

from tvm import tirx
from tvm.tirx.op import call_intrin

tir = tirx

########################################################
# MACA native builtins
########################################################


def maca_func_call(func_name, *args, source_code, return_type="void"):
    """TVM intrinsic to call a MACA function. Source code is provided as a string.

    Parameters
    ----------
    func_name: str
        The name of the MACA function.

    args: PrimExpr
        The arguments to the MACA function.

    source_code: str
        The source code of the MACA function.

    return_type: str
        The return type of the MACA function.
    """
    return call_intrin(return_type, "tirx.maca.func_call", func_name, *args, source_code)


def maca_warp_reduce(value, op, width=64):
    """Reduce a scalar over a MACA Wave64 power-of-two subgroup.

    ``width`` must be a power of two in ``[2, 64]``. The code generator
    validates this constraint and emits a full-Wave64-mask XOR butterfly.
    """
    return call_intrin(value.ty, "tirx.maca.warp_reduce", value, op, width)


def maca_warp_sum(value, width=64):
    """Reduce a scalar sum over a MACA Wave64 subgroup."""
    return maca_warp_reduce(value, "sum", width)


def maca_warp_max(value, width=64):
    """Reduce a scalar maximum over a MACA Wave64 subgroup."""
    return maca_warp_reduce(value, "max", width)


def maca_warp_min(value, width=64):
    """Reduce a scalar minimum over a MACA Wave64 subgroup."""
    return maca_warp_reduce(value, "min", width)


def maca_cta_reduce(value, op, num_waves, scratch):
    """Reduce a scalar over a CTA of one to sixteen MACA Wave64 groups."""
    return call_intrin(value.ty, "tirx.maca.cta_reduce", value, op, num_waves, scratch)


def maca_cta_sum(value, num_waves, scratch):
    """Reduce a scalar sum over MACA waves in a CTA."""
    return maca_cta_reduce(value, "sum", num_waves, scratch)


def maca_cta_max(value, num_waves, scratch):
    """Reduce a scalar maximum over MACA waves in a CTA."""
    return maca_cta_reduce(value, "max", num_waves, scratch)


def maca_cta_min(value, num_waves, scratch):
    """Reduce a scalar minimum over MACA waves in a CTA."""
    return maca_cta_reduce(value, "min", num_waves, scratch)


def maca_thread_fence():
    """TVM intrinsic to call maca thread fence instruction

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.maca.thread_fence")


def maca_warp_sync():
    """TVM intrinsic to synchronize threads within the current warp.

    This lowers to a MACA `__syncwarp()` call.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.maca.warp_sync")


def maca_cta_sync():
    """TVM intrinsic to call MACA syncthreads (block-wide barrier)

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.maca.cta_sync")


def maca_copy_bytes(dst, src, num_bytes):
    """Copy 1, 2, 4, 8, or 16 bytes with one typed load/store pair."""
    return call_intrin("void", "tirx.maca.copy_bytes", dst, src, num_bytes)


def maca_copy_128b(dst, src):
    """Copy 128 bits from ``src`` to ``dst``."""
    return maca_copy_bytes(dst, src, 16)


def maca_copy_64b(dst, src):
    """Copy 64 bits from ``src`` to ``dst``."""
    return maca_copy_bytes(dst, src, 8)


def maca_copy_32b(dst, src):
    """Copy 32 bits from ``src`` to ``dst``."""
    return maca_copy_bytes(dst, src, 4)


def maca_copy_16b(dst, src):
    """Copy 16 bits from ``src`` to ``dst``."""
    return maca_copy_bytes(dst, src, 2)


def maca_copy_8b(dst, src):
    """Copy 8 bits from ``src`` to ``dst``."""
    return maca_copy_bytes(dst, src, 1)
