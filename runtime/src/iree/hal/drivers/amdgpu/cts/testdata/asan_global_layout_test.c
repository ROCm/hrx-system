// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

#define IREE_ASAN_GLOBAL_LAYOUT_DATA_NAME "iree_asan_global_layout_data"

// Define exact dynamic object sizes while retaining one full shadow granule
// after every object. The fixture deliberately uses the same ELF contract as
// Loom output without relying on compiler padding as part of a symbol's size.
__asm__(
    ".section .data.iree_asan_global_layout,\"aw\",@progbits\n"
    ".p2align 3\n"
    ".global " IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME
    "\n"
    ".protected " IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME
    "\n"
    ".type " IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME
    ",@object\n" IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME
    ":\n"
    ".zero 96\n"
    ".size " IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME
    ",.-" IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME
    "\n"
    ".zero 8\n"
    ".p2align 3\n"
    ".global " IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME
    "\n"
    ".protected " IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME
    "\n"
    ".type " IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME
    ",@object\n" IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME
    ":\n"
    ".byte 0\n"
    ".size " IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME
    ",.-" IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME
    "\n"
    ".p2align 3\n"
    ".zero 8\n"
    ".p2align 3\n"
    ".global " IREE_ASAN_GLOBAL_LAYOUT_DATA_NAME
    "\n"
    ".protected " IREE_ASAN_GLOBAL_LAYOUT_DATA_NAME
    "\n"
    ".type " IREE_ASAN_GLOBAL_LAYOUT_DATA_NAME
    ",@object\n" IREE_ASAN_GLOBAL_LAYOUT_DATA_NAME
    ":\n"
    ".long 7,10,13,16\n"
    ".size " IREE_ASAN_GLOBAL_LAYOUT_DATA_NAME
    ",.-" IREE_ASAN_GLOBAL_LAYOUT_DATA_NAME
    "\n"
    ".zero 8\n");

IREE_AMDGPU_ATTRIBUTE_KERNEL void export0(uint32_t* output) { output[0] = 0; }
