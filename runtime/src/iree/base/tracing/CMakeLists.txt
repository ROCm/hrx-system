# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_ENABLE_RUNTIME_TRACING)

message(STATUS "Enabling runtime tracing using the ${IREE_TRACING_PROVIDER} provider")
message(STATUS "Tracing mode set at ${IREE_TRACING_MODE}")

if(${IREE_TRACING_PROVIDER} STREQUAL "console")
  iree_cc_library(
    NAME
      provider
    HDRS
      "console.h"
    SRCS
      "console.c"
    DEPS
      iree::base::core_headers
      iree::base::internal::time
    DEFINES
      "IREE_TRACING_PROVIDER_H=\"iree/base/tracing/console.h\""
      "IREE_TRACING_MODE=${IREE_TRACING_MODE}"
    PUBLIC
  )
  if(IREE_TRACING_MODE GREATER 0)
    iree_cc_test(
      NAME
        console_test
      SRCS
        "console_test.cc"
      DEPS
        ::provider
        iree::base::core_headers
        iree::testing::gtest
        iree::testing::gtest_main
    )
  endif()
elseif(${IREE_TRACING_PROVIDER} STREQUAL "tracy")
  include("${IREE_ROOT_DIR}/build_tools/third_party/tracy_client/tracy_client.cmake")
  iree_cc_library(
    NAME
      provider
    HDRS
      "tracy.h"
    SRCS
      "tracy.cc"
    DEPS
      iree::base::core_headers
      tracy_client::runtime
      ${CMAKE_DL_LIBS}
    DEFINES
      "IREE_TRACING_PROVIDER_H=\"iree/base/tracing/tracy.h\""
      "IREE_TRACING_MODE=${IREE_TRACING_MODE}"
    PUBLIC
  )
  if(IREE_TRACING_MODE GREATER 0)
    iree_cc_test(
      NAME
        tracy_test
      SRCS
        "tracy_test.cc"
      DEPS
        ::provider
        iree::base::core_headers
        iree::testing::gtest
        iree::testing::gtest_main
        tracy_client::runtime
    )
  endif()
else()
  # Add the dep so that the user-provided provider is linked in.
  # The user will also need to have set a header to include via
  # IREE_TRACING_PROVIDER_H.
  if(NOT IREE_TRACING_PROVIDER_H)
    message(FATAL_ERROR "Custom tracing providers require a header file be set in IREE_TRACING_PROVIDER_H")
  endif()
  iree_cc_library(
    NAME
      provider
    DEPS
      ${IREE_TRACING_PROVIDER}
    DEFINES
      "IREE_TRACING_PROVIDER_H=\"${IREE_TRACING_PROVIDER_H}\""
      "IREE_TRACING_MODE=${IREE_TRACING_MODE}"
    PUBLIC
  )
endif()

else()
  # Empty rule for when no tracing provider is defined.
  message(STATUS "Runtime tracing disabled")
  iree_cc_library(
    NAME
      provider
    PUBLIC
  )
endif(IREE_ENABLE_RUNTIME_TRACING)
