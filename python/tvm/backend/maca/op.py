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


def maca_atomic_add(address, value):
    """Atomically add ``value`` to a MACA device-memory address."""
    value = tir.convert(value)
    return call_intrin(value.ty, "tirx.maca.atomic_add", address, value)


def maca_atomic_cas(address, compare, value):
    """Atomically compare and exchange a MACA device-memory address."""
    compare = tir.convert(compare)
    return call_intrin(compare.ty, "tirx.maca.atomic_cas", address, compare, value)


def maca_ldg(address, dtype, *, dst=None, vec=""):
    """Load through MACA's read-only global-memory path."""
    if dst is None:
        if vec:
            raise ValueError("vector maca.ldg requires dst")
        return call_intrin(dtype, "tirx.maca.ldg", address, dtype)
    if vec not in ("v2", "v4"):
        raise ValueError(f"maca.ldg expects vec in {{'v2', 'v4'}}, got {vec!r}")
    if not isinstance(dst, list | tuple):
        raise ValueError("maca.ldg requires tuple/list dst")
    vec_len = int(vec[1:])
    if len(dst) != vec_len:
        raise ValueError(f"maca.ldg dst length must match {vec}: got {len(dst)}")
    return call_intrin("", "tirx.maca.ldg", *dst, address, dtype, vec, vec_len)


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


def maca_copy_async_32b(dst, src):
    """Issue an asynchronous 32-bit global-to-shared BSM copy."""
    return call_intrin("void", "tirx.maca.copy_async_32b", dst, src)


def maca_copy_async_64b(dst, src):
    """Issue an asynchronous 64-bit global-to-shared BSM copy."""
    return call_intrin("void", "tirx.maca.copy_async_64b", dst, src)


def maca_copy_async_128b(dst, src):
    """Issue an asynchronous 128-bit global-to-shared BSM copy."""
    return call_intrin("void", "tirx.maca.copy_async_128b", dst, src)


def maca_copy_async_32b_zfill(dst, src, predicate):
    """Issue a predicated asynchronous 32-bit BSM copy with zero fill."""
    return call_intrin("void", "tirx.maca.copy_async_32b_zfill", dst, src, predicate)


def maca_copy_async_64b_zfill(dst, src, predicate):
    """Issue a predicated asynchronous 64-bit BSM copy with zero fill."""
    return call_intrin("void", "tirx.maca.copy_async_64b_zfill", dst, src, predicate)


def maca_copy_async_128b_zfill(dst, src, predicate):
    """Issue a predicated asynchronous 128-bit BSM copy with zero fill."""
    return call_intrin("void", "tirx.maca.copy_async_128b_zfill", dst, src, predicate)


def maca_async_wait_gvmcnt(count):
    """Wait for the requested number of outstanding global-memory transfers."""
    return call_intrin("void", "tirx.maca.async_wait_gvmcnt", count)


def maca_barrier_inst():
    """Issue the instruction barrier used after a GVM wait."""
    return call_intrin("void", "tirx.maca.barrier_inst")


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
