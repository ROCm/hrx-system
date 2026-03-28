# Copyright 2026 The Pyre Authors
# SPDX-License-Identifier: Apache-2.0
#
# Streaming HIP runtime library (internal, static, LTO'd into consumers).

add_library(pyre_streaming STATIC
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

target_include_directories(pyre_streaming PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(pyre_streaming PUBLIC
    hsa-runtime64::hsa-runtime64
    pyre_hsa_registration
    iree_async_async
    iree_async_util_proactor_pool
    iree_base_base
    iree_base_internal_arena
    iree_base_threading_threading
    iree_hal_hal
    iree_hal_drivers_drivers
    iree_hal_remote_client_registration_registration
    iree_hal_utils_resource_set
    iree_io_file_handle
)

set_target_properties(pyre_streaming PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

if(BUILD_TESTING)
  add_executable(buffer_table_test util/buffer_table_test.cc)
  target_link_libraries(buffer_table_test PRIVATE
      pyre_streaming
      iree_base_base
      iree_testing_gtest
      iree_testing_gtest_main
  )
  add_test(NAME buffer_table_test COMMAND buffer_table_test)

  add_executable(buffer_table_benchmark util/buffer_table_benchmark.cc)
  target_link_libraries(buffer_table_benchmark PRIVATE
      pyre_streaming
      iree_base_base
      iree_testing_benchmark
      iree_testing_benchmark_main
  )
endif()
