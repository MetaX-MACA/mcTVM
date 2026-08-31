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
"""BF16 MACA WMMA registration tests."""

import pytest

from tvm.s_tir.tensor_intrin.maca import get_wmma_intrin_group
from tvm.tirx import TensorIntrin


def test_maca_bf16_wmma_group_names():
    group = get_wmma_intrin_group("shared", "shared", "bfloat16", "float32", True)
    assert group == {
        "init": "maca_wmma_fill_16x16x16_f32",
        "load_a": "maca_wmma_load_16x16x16_bf16_a_shared",
        "load_b": "maca_wmma_load_16x16x16_bf16_b_trans_shared",
        "compute": "maca_wmma_sync_16x16x16_bf16bf16f32_trans",
        "store": "maca_wmma_store_16x16x16_f32_shared",
    }


@pytest.mark.parametrize(
    "name",
    [
        "maca_wmma_load_16x16x16_bf16_a_shared",
        "maca_wmma_load_16x16x16_bf16_b_trans_shared",
        "maca_wmma_sync_16x16x16_bf16bf16f32_trans",
        "maca_wmma_fill_16x16x16_f32",
        "maca_wmma_store_16x16x16_f32_shared",
    ],
)
def test_maca_bf16_wmma_intrinsics_are_registered(name):
    assert TensorIntrin.get(name) is not None
