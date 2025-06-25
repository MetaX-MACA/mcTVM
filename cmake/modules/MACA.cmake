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

# MACA Module
find_maca(${USE_MACA})

if(MACA_FOUND)
  # always set the includedir
  # avoid global retrigger of cmake
  include_directories(SYSTEM ${MACA_INCLUDE_DIRS})
  #add_definitions(-D__MC_PLATFORM_HCC__=1)
endif(MACA_FOUND)


if(USE_MACA)
  if(NOT MACA_FOUND)
    message(FATAL_ERROR "Cannot find MACA, USE_MACA=" ${USE_MACA})
  endif()
  message(STATUS "Build with MACA support")
  tvm_file_glob(GLOB RUNTIME_MACA_SRCS src/runtime/maca/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_MACA_SRCS})
  list(APPEND COMPILER_SRCS src/target/opt/build_maca_on.cc)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${MACA_MACAMCC_LIBRARY})
  if (MACA_HCA_LIBRARY)
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${MACA_HCA_LIBRARY})
  endif()

  if(USE_FLASHATTN)
    message(STATUS "Build with FlashAttn support")
    tvm_file_glob(GLOB FLASHATTN_RELAY_CONTRIB_SRC src/relay/backend/contrib/flashattn/*.cc)
    list(APPEND COMPILER_SRCS ${FLASHATTN_RELAY_CONTRIB_SRC})
    tvm_file_glob(GLOB CONTRIB_FLASHATTN_SRCS src/runtime/contrib/flashattn/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_FLASHATTN_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${MACA_FLASHATTN_LIBRARY})
  endif(USE_FLASHATTN)
else(USE_MACA)
  list(APPEND COMPILER_SRCS src/target/opt/build_maca_off.cc)
endif(USE_MACA)
