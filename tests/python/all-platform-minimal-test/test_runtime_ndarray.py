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
"""Basic runtime enablement test."""

import math

import pytest
import numpy as np

import tvm
import tvm.testing
from tvm import te

dtype = tvm.testing.parameter("uint8", "int8", "uint16", "int16", "uint32", "int32", "float32")


def test_nd_create(target, dev, dtype):
    x = np.random.randint(0, 10, size=(3, 4))
    x = np.array(x, dtype=dtype)
    y = tvm.nd.array(x, device=dev)
    z = y.copyto(dev)
    assert y.dtype == x.dtype
    assert y.shape == x.shape
    assert isinstance(y, tvm.nd.NDArray)
    np.testing.assert_equal(x, y.numpy())
    np.testing.assert_equal(x, z.numpy())

    # no need here, just to test usablity
    dev.sync()


def test_memory_usage(target, dev, dtype):
    available_memory_before = dev.available_global_memory
    if available_memory_before is None:
        pytest.skip(reason=f"Target '{target}' does not support queries of available memory")

    shape = [8192, 8192] if target == "maca" else [1024, 1024]
    arr = tvm.nd.empty(shape, dtype=dtype, device=dev)
    available_memory_after = dev.available_global_memory

    num_elements = math.prod(arr.shape)
    element_nbytes = tvm.runtime.DataType(dtype).itemsize
    expected_memory_after = available_memory_before - num_elements * element_nbytes

    # Allocations may be padded out to provide alignment, to match a
    # page boundary, due to additional device-side bookkeeping
    # required by the TVM backend or the driver, etc.  Therefore, the
    # available memory may decrease by more than the requested amount.
    assert available_memory_after <= expected_memory_after

    # TVM's NDArray type is a reference-counted handle to the
    # underlying reference.  After the last reference to an NDArray is
    # cleared, the backing allocation will be freed.
    del arr

    assert dev.available_global_memory == available_memory_before


def test_dtype():
    dtype = tvm.DataType("handle")
    assert dtype.type_code == tvm.DataTypeCode.HANDLE


if __name__ == "__main__":
    tvm.testing.main()
