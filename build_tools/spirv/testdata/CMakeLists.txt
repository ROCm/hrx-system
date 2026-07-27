# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_spirv_asm_module(
  PACKAGE
    build_tools::spirv::testdata
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
