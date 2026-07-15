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
if(LIBHRX_BUILD AND IREE_BUILD_TESTS AND IREE_HAL_DRIVER_AMDGPU)
  set(_LIBHRX_BUILD_CTS_DEFAULT ON)
endif()
option(LIBHRX_BUILD_CTS
  "Build libhrx conformance tests." ${_LIBHRX_BUILD_CTS_DEFAULT})
if(LIBHRX_BUILD AND LIBHRX_BUILD_CTS AND NOT IREE_HAL_DRIVER_AMDGPU)
  message(FATAL_ERROR
    "LIBHRX_BUILD_CTS=ON requires IREE_HAL_DRIVER_AMDGPU=ON because the "
    "current CTS embeds AMDGPU test kernels.")
endif()
option(HRX_INSTALL_TESTS
  "Install a relocatable CTest tree and test artifacts." ${IREE_BUILD_TESTS})
set(HRX_PUBLIC_DIST_COMPONENT "HrxPublicDist" CACHE STRING
  "Install component for the public HRX distribution.")
set(HRX_INSTALL_TESTS_COMPONENT "HrxTestsDist" CACHE STRING
  "Install component for the installed HRX system test suite.")
set(HRX_INSTALL_TESTS_DIR "${CMAKE_INSTALL_DATADIR}/hrx-system/tests" CACHE STRING
  "Install directory for the HRX system CTest tree and test artifacts.")

include(hrx_installed_tests)
include(binary)
include(selectors)

function(libhrx_configure_project)
endfunction()
