# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_BUILD_TESTS AND (IREE_HAL_DRIVER_VULKAN OR
    (LOOM_BUILD AND LOOM_TARGET_ARCH_SPIRV AND LOOM_EMIT_SPIRV)))
  add_subdirectory(testdata)
endif()
