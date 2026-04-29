# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0
#
# Streaming HIP runtime library (internal, static, LTO'd into consumers).

add_library(hrx_streaming STATIC
    context.c
    device.c
    event.c
    graph.c
    graph_analysis.c
    graph_exec.c
    init.c
    mem_pool.c
    memory.c
    module.c
    peer.c
    registry.c
    stream.c
    util/buffer_table.c
)

target_include_directories(hrx_streaming PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)
target_include_directories(hrx_streaming PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/libhrx>
)

target_link_libraries(hrx_streaming PUBLIC
    hrx
    iree_async_async
    iree_async_util_proactor_pool
    iree_base_base
    iree_base_internal_arena
    iree_base_threading_threading
    iree_hal_hal
    iree_hal_utils_resource_set
    iree_io_file_handle
)

set_target_properties(hrx_streaming PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

if(BUILD_TESTING AND IREE_BUILD_TESTS)
  if(NOT TARGET iree_testing_gtest OR NOT TARGET iree_testing_gtest_main)
    message(FATAL_ERROR
      "IREE_BUILD_TESTS is ON, but IREE gtest wrapper targets are missing")
  endif()
  add_executable(buffer_table_test util/buffer_table_test.cc)
  target_link_libraries(buffer_table_test PRIVATE
      hrx_streaming
      iree_base_base
      iree_testing_gtest
      iree_testing_gtest_main
  )
  add_test(NAME buffer_table_test COMMAND buffer_table_test)
  if(NOT TARGET iree_testing_benchmark OR NOT TARGET iree_testing_benchmark_main)
    message(FATAL_ERROR
      "IREE_BUILD_TESTS is ON, but IREE benchmark wrapper targets are missing")
  endif()
  add_executable(buffer_table_benchmark util/buffer_table_benchmark.cc)
  target_link_libraries(buffer_table_benchmark PRIVATE
      hrx_streaming
      iree_base_base
      iree_testing_benchmark
      iree_testing_benchmark_main
  )
endif()
