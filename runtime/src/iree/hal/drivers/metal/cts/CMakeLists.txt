# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CTS test binaries for the Metal HAL driver (hand-written CMake, no Bazel).

iree_cc_library(
  NAME
    backends
  SRCS
    "backends.cc"
  DEPS
    iree::async::util::proactor_pool
    iree::base::threading
    iree::hal
    iree::hal::cts::util::registry
    iree::hal::drivers::metal::registration
  TESTONLY
  ALWAYSLINK
  PUBLIC
)

iree_runtime_hal_cts_test_suite(
  BACKENDS
    ::backends
)
