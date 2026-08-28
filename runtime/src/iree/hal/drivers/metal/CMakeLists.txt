# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    executable_format
  HDRS
    "executable_format.h"
  SRCS
    "executable_format.c"
  DEPS
    iree::base
  PUBLIC
)

iree_cc_test(
  NAME
    executable_format_test
  SRCS
    "executable_format_test.cc"
  DEPS
    ::executable_format
    iree::base
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "iree-build-requirement=runtime.hal.metal"
)

iree_cc_library(
  NAME
    metal
  HDRS
    "api.h"
  SRCS
    "api.h"
    "builtin_executables.h"
    "builtin_executables.m"
    "direct_allocator.h"
    "direct_allocator.m"
    "direct_command_buffer.h"
    "direct_command_buffer.m"
    "executable.h"
    "executable.m"
    "metal_buffer.h"
    "metal_buffer.m"
    "metal_device.m"
    "metal_driver.m"
    "shared_event.h"
    "shared_event.m"
    "staging_buffer.h"
    "staging_buffer.m"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::internal::arena
    iree::hal
    ::executable_format
    iree::hal::drivers::metal::builtin
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::device_spec_builder
    iree::hal::utils::files
    iree::hal::utils::queue_emulation
    iree::hal::utils::queue_host_call_emulation
    iree::hal::utils::resource_set
    "-framework Foundation"
    "-framework Metal"
  PUBLIC
)
