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
"""MACA math helpers, including Wave64 reductions."""

from __future__ import annotations

from tvm.backend.maca.op import maca_func_call

from .registry import register_codegen
from .utils import parse_str, validate_power_of_two_range

_WAVE_SIZE = 64
_FULL_WAVE_MASK = "0xFFFFFFFFFFFFFFFFULL"
_REDUCE_STEPS = {
    "sum": "val += shuffled;",
    "max": "val = (val > shuffled) ? val : shuffled;",
    "min": "val = (val < shuffled) ? val : shuffled;",
}


def _reduce_step(op, context: str) -> tuple[str, str]:
    """Return the normalized reduction name and device-side combine step."""
    op_name = parse_str(op)
    try:
        return op_name, _REDUCE_STEPS[op_name]
    except KeyError as error:
        expected = ", ".join(sorted(_REDUCE_STEPS))
        raise ValueError(
            f"Unsupported {context} op {op_name!r}; expected one of {expected}"
        ) from error


def _validate_wave_width(width) -> int:
    """Validate a subgroup width within one Wave64."""
    return validate_power_of_two_range(width, 2, _WAVE_SIZE, "maca warp_reduce width")


def _validate_cta_waves(num_waves) -> int:
    """Validate the supported power-of-two number of waves in a CTA."""
    return validate_power_of_two_range(num_waves, 1, 16, "maca cta_reduce num_waves")


def _warp_reduce_source(func_name: str, width: int, combine_step: str) -> str:
    """Emit a full-mask XOR butterfly reduction helper for a Wave64 subgroup."""
    return (
        "\n"
        "template <typename T>\n"
        f"__forceinline__ __device__ T {func_name}(T val) {{\n"
        "    #pragma unroll\n"
        f"    for (int xor_mask = {width} >> 1; xor_mask > 0; xor_mask >>= 1) {{\n"
        f"        T shuffled = __shfl_xor_sync({_FULL_WAVE_MASK}, val, xor_mask, {width});\n"
        f"        {combine_step}\n"
        "    }\n"
        "    return val;\n"
        "}\n"
    )


@register_codegen("maca_warp_reduce")
def codegen_maca_warp_reduce(value, op, width):
    """Lower a scalar all-reduce over a power-of-two Wave64 subgroup."""
    op_name, combine_step = _reduce_step(op, "maca warp_reduce")
    width_int = _validate_wave_width(width)
    func_name = f"tvm_builtin_maca_warp_reduce_{op_name}_{width_int}"
    return maca_func_call(
        func_name,
        value,
        source_code=_warp_reduce_source(func_name, width_int, combine_step),
        return_type=value.ty,
    )


@register_codegen("maca_cta_reduce")
def codegen_maca_cta_reduce(value, op, num_waves, scratch):
    """Lower a CTA all-reduce over one to sixteen Wave64 groups."""
    op_name, combine_step = _reduce_step(op, "maca cta_reduce")
    waves = _validate_cta_waves(num_waves)

    wave_reduce_name = f"tvm_builtin_maca_warp_reduce_{op_name}_{_WAVE_SIZE}"
    cta_reduce_name = f"tvm_builtin_maca_cta_reduce_{op_name}_{waves}"
    source_code = _warp_reduce_source(wave_reduce_name, _WAVE_SIZE, combine_step)

    # The second stage reduces one value per wave.  Restricting wave counts to
    # powers of two avoids injecting a type-specific identity into inactive lanes.
    partial_reduce_name = f"tvm_builtin_maca_warp_reduce_{op_name}_{waves}"
    if waves > 1:
        source_code += _warp_reduce_source(partial_reduce_name, waves, combine_step)

    source_code += (
        "template <typename T>\n"
        f"__forceinline__ __device__ T {cta_reduce_name}(T val, void* scratch_raw) {{\n"
        "    T* scratch = reinterpret_cast<T*>(scratch_raw);\n"
        "    int tid = threadIdx.x + threadIdx.y * blockDim.x"
        " + threadIdx.z * blockDim.x * blockDim.y;\n"
        f"    int wave_id = tid / {_WAVE_SIZE};\n"
        f"    int lane_id = tid % {_WAVE_SIZE};\n"
        f"    val = {wave_reduce_name}(val);\n"
        "    if (lane_id == 0) scratch[wave_id] = val;\n"
        "    __syncthreads();\n"
        "    if (wave_id == 0) {\n"
        f"        T partial = lane_id < {waves} ? scratch[lane_id] : val;\n"
    )
    if waves > 1:
        source_code += f"        partial = {partial_reduce_name}(partial);\n"
    source_code += (
        "        if (lane_id == 0) scratch[0] = partial;\n"
        "    }\n"
        "    __syncthreads();\n"
        "    return scratch[0];\n"
        "}\n"
    )
    return maca_func_call(
        cta_reduce_name, value, scratch, source_code=source_code, return_type=value.ty
    )


__all__ = ["codegen_maca_cta_reduce", "codegen_maca_warp_reduce"]
