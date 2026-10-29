# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/../build_tools/amdgpu"
  "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake"
)

if(NOT DEFINED LIBHRX_BUILD)
  option(LIBHRX_BUILD
    "Build libhrx and HRX compatibility targets." ON)
endif()
set(_LIBHRX_BUILD_PASSTHROUGH_DEFAULT OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_LIBHRX_BUILD_PASSTHROUGH_DEFAULT ON)
endif()
option(LIBHRX_BUILD_PASSTHROUGH
  "Build libhrx HIP passthrough/interception tools."
  ${_LIBHRX_BUILD_PASSTHROUGH_DEFAULT})
if(LIBHRX_BUILD AND LIBHRX_BUILD_PASSTHROUGH AND
    NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "LIBHRX_BUILD_PASSTHROUGH=ON currently requires Linux because the "
    "passthrough loader and export controls have not been ported yet.")
endif()
option(LIBHRX_BUILD_HIP_BINDING
  "Build libhrx HIP runtime API compatibility binding." ON)
option(LIBHRX_BUILD_CUDA_BINDING
  "Build libhrx CUDA runtime/driver API compatibility binding." OFF)
set(_LIBHRX_BUILD_CTS_DEFAULT OFF)
if(LIBHRX_BUILD AND IREE_BUILD_TESTS)
  set(_LIBHRX_BUILD_CTS_DEFAULT ON)
endif()
option(LIBHRX_BUILD_CTS
  "Build libhrx conformance tests." ${_LIBHRX_BUILD_CTS_DEFAULT})
set(HRX_PUBLIC_DIST_COMPONENT "HrxPublicDist" CACHE STRING
  "Install component for the public HRX distribution.")
include(binary)
include(selectors)

function(libhrx_configure_project)
endfunction()
