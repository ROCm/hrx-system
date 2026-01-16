# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    util
  HDRS
    "buffer_table.h"
  SRCS
    "buffer_table.c"
  DEPS
    iree::base
    iree::base::internal::arena
    iree::base::internal::synchronization
    streaming::internal
    iree::hal
    iree::hal::drivers
  PUBLIC
)

if(IREE_STREAMING_BUILD_TESTS)
  iree_cc_binary_benchmark(
    NAME
      buffer_table_benchmark
    SRCS
      "buffer_table_benchmark.cc"
    DEPS
      ::util
      iree::base
      iree::testing::benchmark
      iree::testing::benchmark_main
    TESTONLY
  )

  iree_cc_test(
    NAME
      buffer_table_test
    SRCS
      "buffer_table_test.cc"
    DEPS
      ::util
      iree::base
      iree::testing::gtest
      iree::testing::gtest_main
  )
endif()
