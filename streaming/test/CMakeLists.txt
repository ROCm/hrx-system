# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(IREE_STREAMING_BUILD_TESTS)

if(IREE_STREAMING_BUILD_CUDA_BINDING)
  iree_cc_binary(
    NAME
      cuda_driver_api_demo
    SRCS
      "cuda_driver_api_demo.c"
    DEPS
      streaming::binding::cuda
  )

  iree_cc_binary(
    NAME
      cuda_runtime_api_demo
    SRCS
      "cuda_runtime_api_demo.c"
    DEPS
      streaming::binding::cuda
  )
endif()

if(IREE_STREAMING_BUILD_HIP_BINDING)
  iree_cc_binary(
    NAME
      hip_api_demo
    SRCS
      "hip_api_demo.c"
    DEPS
      streaming::binding::hip
  )

  iree_cc_binary(
    NAME
      hip_graph_api_demo
    SRCS
      "hip_graph_api_demo.c"
    DEPS
      streaming::binding::hip
  )

  iree_cc_binary(
    NAME
      hip_graph_capture_api_demo
    SRCS
      "hip_graph_capture_api_demo.c"
    DEPS
      streaming::binding::hip
  )
endif()

iree_cc_library(
  NAME
    graph_util
  HDRS
    "graph_util.h"
  SRCS
    "graph_util.c"
  DEPS
    iree::base
    streaming::streaming
  PUBLIC
)

iree_cc_test(
  NAME
    graph_analysis_benchmark
  SRCS
    "graph_analysis_benchmark.cc"
  DEPS
    ::graph_util
    iree::base
    streaming::streaming
    iree::testing::benchmark
    iree::testing::benchmark_main
  LABELS
    "benchmark"
)

iree_cc_test(
  NAME
    graph_record_benchmark
  SRCS
    "graph_record_benchmark.cc"
  DEPS
    ::graph_util
    iree::base
    streaming::streaming
    iree::testing::benchmark
    iree::testing::benchmark_main
  LABELS
    "benchmark"
)

iree_cc_test(
  NAME
    graph_instantiate_benchmark
  SRCS
    "graph_instantiate_benchmark.cc"
  DEPS
    ::graph_util
    iree::base
    streaming::streaming
    iree::testing::benchmark
    iree::testing::benchmark_main
  LABELS
    "benchmark"
)

iree_cc_test(
  NAME
    graph_exec_benchmark
  SRCS
    "graph_exec_benchmark.cc"
  DEPS
    ::graph_util
    iree::base
    streaming::streaming
    iree::testing::benchmark
    iree::testing::benchmark_main
  LABELS
    "benchmark"
)

endif() # IREE_STREAMING_BUILD_TESTS

# Build test kernels.
add_custom_target(iree_hal_streaming_test_kernels
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/kernels/compile_kernels.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  COMMENT "Compiling CUDA HAL test kernels for multiple targets"
  VERBATIM
)

# HIP demo using HIP C++ compiler (requires IREE_ROCM_PATH)
if(IREE_ROCM_PATH AND IREE_STREAMING_BUILD_HIP_BINDING)
  iree_cc_binary(
    NAME
      hip_demo
    SRCS
      "hip_demo.cc"
    COPTS
      "-xhip"
      "--offload-arch=gfx1100"
      "--rocm-path=${IREE_ROCM_PATH}"
    DEPS
      streaming::binding::hip
  )
else()
  message(STATUS "IREE_ROCM_PATH not specified; skipping HIP C++ demo")
endif()
