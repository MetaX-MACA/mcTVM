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
# pylint: disable=unused-import
"""MACA HW intrinsic codegens, grouped by feature domain.

- ``sync`` — barriers, fences, mbarrier, cluster.barrier, warp vote, elect, sync helpers.

"""

# Import op modules to register their codegen functions.
from . import math, memory, sync
from .registry import CODEGEN_REGISTRY, get_codegen, register_codegen

__all__ = [
    "CODEGEN_REGISTRY",
    "get_codegen",
    "register_codegen",
]
