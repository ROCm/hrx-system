# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Logical package outputs can live below the native generator directory.
set(IREE_PACKAGE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/modules")
file(MAKE_DIRECTORY "${IREE_PACKAGE_BINARY_DIR}")

iree_spirv_asm_module(
  PACKAGE
    iree::build_tools::spirv::testdata
  NAME
    minimal
  SRC
    minimal.spvasm
  TARGET_ENV
    vulkan1.3
  C_IDENTIFIER
    minimal_spv
  TESTONLY
  PUBLIC
)
