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
