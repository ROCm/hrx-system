# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_loom_binary_test C)

set(IREE_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "fixture")
set(IREE_BUILD_TESTS ON)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/loom/build_tools/cmake/loom_target_profile.cmake")
include("${IREE_REPO_ROOT}/loom/build_tools/cmake/loom_kernel_binary.cmake")

# This exercises host-tool selection without requiring a foreign C toolchain.
if(LOOM_TEST_CROSS)
  set(CMAKE_CROSSCOMPILING TRUE)
endif()
if(NOT IREE_HOST_BIN_DIR)
  add_executable(link_tool tool.c)
  add_executable(compile_tool tool.c)
  add_executable(loom::tools::loom-link ALIAS link_tool)
  add_executable(loom::tools::loom-compile ALIAS compile_tool)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/tools-$<CONFIG>.txt"
    CONTENT "$<TARGET_FILE:link_tool>\n$<TARGET_FILE:compile_tool>\n")
endif()

loom_target_profile(NAME npu4 FAMILY amd.xdna.aie2p SELECTOR amd.xdna.strix.17f0_10)
loom_kernel_binary(NAME explicit
  TARGET ::npu4 OUTPUT explicit.xdna SRCS direct.loom
  DATA direct.h INPUT_FORMAT fixture
  INPUTOPTS fixture:include=direct.h fixture:root=direct
  LIBRARIES ::library ROOTS @second @first CONFIGS a.value=3 z.limit=16 TESTONLY)
loom_kernel_binary(NAME exported
  TARGET ::npu4 OUTPUT exported.xdna LIBRARIES ::library TESTONLY)

# Forward libraries preserve their transitive closure independently of source
# assembly. The kernel producer is reached through an ordinary data consumer.
loom_module(NAME library MODE merge OUTPUT_FORMAT bc
  SRCS library.loom LIBRARIES ::leaf STRICT_DEPS)
loom_module(NAME leaf MODE merge OUTPUT_FORMAT bc SRCS leaf.loom STRICT_DEPS)
add_custom_target(kernel_consumer DEPENDS "${CMAKE_BINARY_DIR}/explicit.xdna")
iree_generated_output_add_consumer("${CMAKE_BINARY_DIR}/explicit.xdna" kernel_consumer)
loom_finalize_module_libraries()
iree_finalize_target_dependencies()
