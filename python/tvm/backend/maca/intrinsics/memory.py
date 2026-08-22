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
"""MACA typed memory-copy intrinsic codegens."""

from numbers import Integral

from ._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen

_TYPE_MAP = {
    16: "uint4",
    8: "uint2",
    4: "unsigned int",
    2: "unsigned short",
    1: "unsigned char",
}

for _num_bytes, _cpp_type in _TYPE_MAP.items():
    device_intrinsic(
        f"_maca_copy_bytes_{_num_bytes}_impl",
        helper_name=f"tvm_builtin_copy_{_num_bytes * 8}b",
        c_signature="(void* dst_ptr, const void* src_ptr)",
        body=(
            f"    const {_cpp_type}* src_ = reinterpret_cast<const {_cpp_type}*>(src_ptr);\n"
            f"    {_cpp_type}* dst_ = reinterpret_cast<{_cpp_type}*>(dst_ptr);\n"
            "    *dst_ = *src_;"
        ),
    )
del _num_bytes, _cpp_type


# MACA BSM path is an asynchronous global-to-shared transfer.  Keep the
# width-specific helpers explicit: the compiler builtin encodes the transfer
# width in its name and only accepts 32/64/128-bit forms here.
_BSM_ASYNC_BITS = (32, 64, 128)

for _bits in _BSM_ASYNC_BITS:
    device_intrinsic(
        f"maca_copy_async_{_bits}b",
        helper_name=f"tvm_builtin_maca_copy_async_{_bits}b",
        c_signature="(void* dst_ptr, const void* src_ptr)",
        body=(
            f"    __builtin_mxc_ldg_b{_bits}_bsm(\n"
            "        dst_ptr, const_cast<void*>(src_ptr), 0, -1, true, true, false, true);"
        ),
    )

    device_intrinsic(
        f"maca_copy_async_{_bits}b_zfill",
        helper_name=f"tvm_builtin_maca_copy_async_{_bits}b_zfill",
        c_signature="(void* dst_ptr, const void* src_ptr, bool predicate)",
        body=(
            f"    __builtin_mxc_ldg_b{_bits}_bsm_predicator(\n"
            "        dst_ptr, const_cast<void*>(src_ptr), 0, true, true, false, true,\n"
            "        predicate, 1, MACA_ICMP_EQ);"
        ),
    )

del _bits


def _gvmcnt_wait_count(count):
    """Return the required immediate GVM wait count."""
    dtype = getattr(getattr(count, "ty", None), "dtype", "")
    value = count.value if hasattr(count, "value") else count
    if (
        (dtype and not (dtype.startswith("int") or dtype.startswith("uint")))
        or isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise ValueError(
            "maca_async_wait_gvmcnt requires a non-negative compile-time integer count"
        )
    return int(value)


def _gvmcnt_wait_helper_name(count):
    return f"tvm_builtin_maca_async_wait_gvmcnt_{_gvmcnt_wait_count(count)}"


def _gvmcnt_wait_body(count):
    """Inline the GVM wait count because the MACA builtin requires an immediate."""
    count = _gvmcnt_wait_count(count)
    return f"    __builtin_mxc_arrive_gvmcnt({count});"


device_intrinsic(
    "maca_async_wait_gvmcnt",
    helper_name=_gvmcnt_wait_helper_name,
    c_signature="()",
    body=_gvmcnt_wait_body,
    n_attrs=1,
)


device_intrinsic(
    "maca_barrier_inst",
    helper_name="tvm_builtin_maca_barrier_inst",
    c_signature="()",
    body="    __builtin_mxc_barrier_inst();",
)


@register_codegen("maca_copy_bytes")
def codegen_maca_copy_bytes(dst, src, num_bytes):
    """Dispatch ``tirx.maca.copy_bytes`` to a width-specific helper."""
    num_bytes_int = int(num_bytes)
    if num_bytes_int not in _TYPE_MAP:
        raise ValueError(
            f"Unsupported maca_copy_bytes num_bytes {num_bytes_int}, "
            f"expected one of {sorted(_TYPE_MAP)}"
        )
    result = CODEGEN_REGISTRY[f"tirx._maca_copy_bytes_{num_bytes_int}_impl"]([dst, src])
    return result[0] if isinstance(result, tuple) else result
