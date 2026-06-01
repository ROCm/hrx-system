# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake"
)

option(LIBHRX_BUILD
  "Build libhrx and HRX compatibility targets." ON)
option(LIBHRX_BUILD_PASSTHROUGH
  "Build libhrx HIP passthrough/interception tools." ON)
option(LIBHRX_BUILD_CUDA_BINDING
  "Build libhrx CUDA runtime/driver API compatibility binding." OFF)
option(LIBHRX_BUILD_CTS
  "Build libhrx conformance tests." ${IREE_BUILD_TESTS})
option(HRX_INSTALL_TESTS
  "Install a relocatable CTest tree and test artifacts." ${IREE_BUILD_TESTS})
set(HRX_PUBLIC_DIST_COMPONENT "HrxPublicDist" CACHE STRING
  "Install component for the public HRX distribution.")
set(HRX_INSTALL_TESTS_COMPONENT "HrxTestsDist" CACHE STRING
  "Install component for the installed HRX system test suite.")
set(HRX_INSTALL_TESTS_DIR "${CMAKE_INSTALL_DATADIR}/hrx-system/tests" CACHE STRING
  "Install directory for the HRX system CTest tree and test artifacts.")

include("${CMAKE_CURRENT_LIST_DIR}/build_tools/third_party/libhrx_dependencies.cmake")

include(hrx_installed_tests)

function(libhrx_configure_project)
  libhrx_configure_dependencies()
endfunction()
