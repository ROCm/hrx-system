# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_hal_cts_test C CXX)

set(CMAKE_CXX_STANDARD 20)
set(IREE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/hal/drivers/amd/gpu/cts")
set(IREE_BUILD_TESTS ON)
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_macros.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_copts.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_install_support.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_library.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_cc_binary.cmake")
include("${IREE_REPO_ROOT}/build_tools/cmake/iree_c_embed_data.cmake")
include("${IREE_REPO_ROOT}/runtime/build_tools/cmake/iree_runtime_hal_cts_testdata.cmake")
set(IREE_SOURCE_DIR "${IREE_REPO_ROOT}")

add_executable(iree-c-embed-data "${IREE_REPO_ROOT}/build_tools/embed_data/embed_data.cc")
add_library(registry INTERFACE)
target_include_directories(registry INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
add_library(iree::hal::cts::util::registry ALIAS registry)

if(CTS_COMPILED_CORPUS)
  include("${IREE_REPO_ROOT}/loom/build_tools/cmake/loom_target_profile.cmake")
  include("${IREE_REPO_ROOT}/loom/build_tools/cmake/loom_kernel_binary.cmake")
  include("${IREE_REPO_ROOT}/runtime/src/iree/hal/drivers/amd/gpu/cts/testdata.cmake")
  # Only the compiler dependency is substituted. All source, product, embedding,
  # registration, and transitive linkage rules are the real build implementation.
  add_executable(compiler "${IREE_REPO_ROOT}/build_tools/cmake/testdata/loom_binary/tool.c")
  add_executable(loom::tools::loom-link ALIAS compiler)
  add_executable(loom::tools::loom-compile ALIAS compiler)
  set(FIRST_DESCRIPTOR ON)
  set(SECOND_DESCRIPTOR ${CTS_EXTRA_TARGET})
  set(UNAVAILABLE_DESCRIPTOR OFF)
  set(_TARGETS
    first-exact=FIRST_DESCRIPTOR first-generic=FIRST_DESCRIPTOR
    second=SECOND_DESCRIPTOR unavailable=UNAVAILABLE_DESCRIPTOR)
  iree_hal_amd_gpu_cts_testdata(
    NAME primary_corpus SRCS one.kernel.loom two.loom
    BACKEND_NAMES primary alias TARGETS ${_TARGETS})
  iree_hal_amd_gpu_cts_testdata(
    NAME secondary_corpus SRCS one.kernel.loom
    BACKEND_NAMES secondary TARGETS ${_TARGETS})
  iree_cc_binary(NAME probe SRCS corpus_main.cc
    DEPS ::primary_corpus ::secondary_corpus)
  loom_finalize_module_libraries()
else()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/products")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/products/kernel.bin" "kernel bytes")
iree_runtime_hal_cts_testdata(
  NAME asan_report_testdata_gfx12-5-generic
  SRCS "${CMAKE_CURRENT_BINARY_DIR}/products/kernel.bin"
  BACKEND_NAMES primary alias
  TARGET_NAME target
  TARGET_FAMILY family
  TARGET_KEY family:target
)
iree_cc_binary(NAME probe SRCS main.cc
  DEPS ::asan_report_testdata_gfx12-5-generic)
endif()
iree_finalize_target_dependencies()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/probe-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:iree_hal_drivers_amd_gpu_cts_probe>\n")
