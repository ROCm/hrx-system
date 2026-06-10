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

# CCOB fat binaries use zstd for compression (used by rocBLAS/hipBLAS and
# most real-world HIP shipping toolchains). If the system has it, wire it in so
# the streaming layer can decompress on the fly.
option(HRX_ENABLE_ZSTD "Enable zstd decompression for CCOB fat binaries" ON)
if(HRX_ENABLE_ZSTD)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD QUIET libzstd)
  endif()
  if(ZSTD_FOUND)
    foreach(_hrx_common_target
        libhrx_src_binding_common_common
        libhrx_src_binding_common_common.objects)
      target_compile_definitions(${_hrx_common_target} PRIVATE
        HRX_ENABLE_ZSTD=1
      )
      target_include_directories(${_hrx_common_target} PRIVATE
        ${ZSTD_INCLUDE_DIRS}
      )
    endforeach()
    target_link_libraries(libhrx_src_binding_common_common PUBLIC
      ${ZSTD_LIBRARIES}
    )
    message(STATUS
      "hrx-runtime: CCOB zstd decompression ENABLED (libzstd ${ZSTD_VERSION})"
    )
  else()
    message(STATUS
      "hrx-runtime: libzstd not found; CCOB fat binaries will fail to load "
      "(install libzstd-dev or disable HRX_ENABLE_ZSTD)"
    )
  endif()
endif()

# Out-of-band code objects (ROCm "kpack" packaging): __hipRegisterFatBinary
# hands us a "HIPK" wrapper whose payload is msgpack metadata pointing at a
# sibling .kpack archive rather than an inline code object. fat_binary.c
# resolves it with the vendor loader librocm_kpack. find_package locates it
# through CMAKE_PREFIX_PATH (for rocm-sdk wheels: `rocm-sdk path --cmake`).
find_package(rocm-kpack CONFIG QUIET)
if(TARGET rocm::rocm_kpack)
  target_link_libraries(libhrx_src_binding_common_common PUBLIC rocm::rocm_kpack)
  target_compile_definitions(libhrx_src_binding_common_common
    PUBLIC HRX_ENABLE_KPACK=1)
else()
  message(WARNING
    "rocm-kpack not found: HRX will build, but HIPK out-of-band code objects "
    "(kpack-packaged ROCm libraries) will fail to load at registration.")
endif()
