# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)
include(iree_third_party_helpers)

function(iree_configure_google_benchmark)
  if(NOT IREE_ENABLE_THREADING OR NOT IREE_BUILD_BENCHMARKS)
    return()
  endif()

  find_package(benchmark CONFIG QUIET)
  if(NOT TARGET benchmark AND NOT TARGET benchmark::benchmark)
    iree_fetch_content_assert_allowed("google benchmark")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      google_benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG v1.9.5
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(google_benchmark)
  endif()

  if(TARGET benchmark)
    return()
  endif()
  if(TARGET benchmark::benchmark)
    iree_add_alias_interface(benchmark benchmark::benchmark)
    return()
  endif()
  message(FATAL_ERROR "google benchmark did not provide a benchmark target")
endfunction()
