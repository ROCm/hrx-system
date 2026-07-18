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
option(LIBHRX_BUILD_PASSTHROUGH
  "Build libhrx HIP passthrough/interception tools." ON)
option(LIBHRX_BUILD_HIP_BINDING
  "Build the libhrx HIP runtime API compatibility binding." ON)
option(LIBHRX_BUILD_TOOLS
  "Build libhrx command-line tools." ON)
option(LIBHRX_BUILD_CUDA_BINDING
  "Build libhrx CUDA runtime/driver API compatibility binding." OFF)
option(LIBHRX_BUILD_CTS
  "Build libhrx conformance tests." ${IREE_BUILD_TESTS})
option(LIBHRX_BUILD_TESTS
  "Build focused libhrx unit tests." OFF)
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
