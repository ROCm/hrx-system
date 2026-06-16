# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT WIN32)
  return()
endif()

iree_cc_library(
  NAME
    kmt_api
  SRCS
    context_blob.cc
    context_blob.h
    kmt_api.cc
    kmt_api.h
  DEPS
    iree::base
  PUBLIC
)

iree_cc_test(
  NAME
    context_blob_test
  SRCS
    context_blob_test.cc
  DEPS
    ::kmt_api
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    kmt_api_test
  SRCS
    kmt_api_test.cc
  DEPS
    ::kmt_api
    iree::testing::gtest
    iree::testing::gtest_main
)
