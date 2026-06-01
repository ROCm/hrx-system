################################################################################
# Common binding runtime — shared by HIP and CUDA bindings.
################################################################################

iree_add_all_subdirs()

iree_cc_library(
  NAME
    common
  COPTS
    "-Wno-unused-variable"
  HDRS
    "fat_binary.h"
    "graph.h"
    "hrx_bridge.h"
    "internal.h"
  SRCS
    "context.c"
    "device.c"
    "event.c"
    "fat_binary.c"
    "graph.c"
    "graph_analysis.c"
    "graph_exec.c"
    "init.c"
    "mem_pool.c"
    "memory.c"
    "module.c"
    "peer.c"
    "registry.c"
    "stream.c"
  DEPS
    flatcc::parsing
    iree::async
    iree::async::util::proactor_pool
    iree::base
    iree::base::internal::arena
    iree::base::threading
    iree::hal
    iree::hal::utils::resource_set
    iree::io::file_handle
    libhrx::src::libhrx::hrx
    libhrx_defs
  INCLUDES
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../libhrx>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../../libhrx>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../../../include>"
  PUBLIC
)

iree_cc_test(
  NAME
    buffer_table_test
  SRCS
    "buffer_table_test.cc"
  DEPS
    ::common
    iree::base
    iree::testing::gtest
    iree::testing::gtest_main
    libhrx_defs
)

iree_cc_binary_benchmark(
  NAME
    buffer_table_benchmark
  SRCS
    "buffer_table_benchmark.cc"
  DEPS
    ::common
    iree::base
    iree::testing::benchmark
    iree::testing::benchmark_main
    libhrx_defs
  TESTONLY
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###

set_target_properties(libhrx_src_binding_common_common PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION TRUE
)
