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

#######################################################
# Enhanced version of find maca.
#
# Usage:
#   find_maca(${USE_MACA})
#
# - When USE_MACA=ON, use auto search
# - When USE_MACA=/path/to/maca-sdk-path, use the sdk
# - When MACA_PATH or MACA_HOME is set, use the env path
#
# Provide variables:
#
# - MACA_FOUND
# - MACA_ROOT_DIR
# - MACA_INCLUDE_DIRS
# - MACA_MACAMCC_LIBRARY

macro(find_maca use_maca)
  set(__use_maca "${use_maca}")
  unset(MACA_FOUND)
  unset(MACA_ROOT_DIR)
  unset(MACA_INCLUDE_DIRS)
  unset(MACA_MACAMCC_LIBRARY CACHE)
  unset(MACA_HCA_LIBRARY CACHE)
  unset(MACA_FLASHATTN_LIBRARY CACHE)

  if(IS_DIRECTORY "${__use_maca}")
    set(__maca_sdk "${__use_maca}")
    message(STATUS "Custom MACA SDK PATH=${__use_maca}")
  elseif(IS_DIRECTORY "$ENV{MACA_PATH}")
    set(__maca_sdk "$ENV{MACA_PATH}")
  elseif(IS_DIRECTORY "$ENV{MACA_HOME}")
    set(__maca_sdk "$ENV{MACA_HOME}")
  elseif(IS_DIRECTORY /opt/maca)
    set(__maca_sdk /opt/maca)
  else()
    set(__maca_sdk "")
  endif()

  if(__maca_sdk)
    set(MACA_ROOT_DIR ${__maca_sdk})
    set(MACA_INCLUDE_DIRS ${__maca_sdk}/include)
    find_library(MACA_MACAMCC_LIBRARY mcruntime PATHS ${__maca_sdk}/lib NO_DEFAULT_PATH)
    find_library(MACA_HCA_LIBRARY mxc-runtime64 PATHS ${__maca_sdk}/lib NO_DEFAULT_PATH)
    find_library(MACA_FLASHATTN_LIBRARY mcFlashAttn PATHS ${__maca_sdk}/lib NO_DEFAULT_PATH)

    if(MACA_MACAMCC_LIBRARY)
      set(MACA_FOUND TRUE)
    endif()
  endif(__maca_sdk)
  if(MACA_FOUND)
    message(STATUS "Found MACA_ROOT_DIR=" ${MACA_ROOT_DIR})
    message(STATUS "Found MACA_INCLUDE_DIRS=" ${MACA_INCLUDE_DIRS})
    message(STATUS "Found MACA_MACAMCC_LIBRARY=" ${MACA_MACAMCC_LIBRARY})
    message(STATUS "Found MACA_FLASHATTN_LIBRARY=" ${MACA_FLASHATTN_LIBRARY})
  endif(MACA_FOUND)
endmacro(find_maca)
