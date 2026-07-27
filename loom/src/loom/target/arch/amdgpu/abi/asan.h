// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU ASAN shadow ABI constants used by Loom code generation.
//
// This header intentionally mirrors only the device-visible layout facts needed
// by the compiler. It must not include runtime/src/iree/hal/drivers/amdgpu/abi
// headers because their host side includes HSA headers, while the Loom compiler
// should stay independent from HSA except in execution/tooling code.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ABI_ASAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_ABI_ASAN_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_ASAN_CONFIG_ABI_VERSION 0u

// Log2 application bytes represented by one shadow byte in the current Loom
// AMDGPU access-check lowering.
#define LOOM_AMDGPU_ASAN_SHADOW_SCALE_SHIFT 3u

// Bitset of loom_amdgpu_asan_config_flag_bits_e values.
typedef uint32_t loom_amdgpu_asan_config_flags_t;

enum loom_amdgpu_asan_config_flag_bits_e {
  // ASAN shadow checking is disabled.
  LOOM_AMDGPU_ASAN_CONFIG_FLAG_NONE = 0u,
  // ASAN shadow checking is enabled for the owning logical device.
  LOOM_AMDGPU_ASAN_CONFIG_FLAG_ENABLED = 1u << 0,
};

enum loom_amdgpu_asan_config_layout_e {
  LOOM_AMDGPU_ASAN_CONFIG_BYTE_LENGTH = 96u,
  LOOM_AMDGPU_ASAN_CONFIG_RECORD_LENGTH_OFFSET = 0u,
  LOOM_AMDGPU_ASAN_CONFIG_ABI_VERSION_OFFSET = 4u,
  LOOM_AMDGPU_ASAN_CONFIG_FLAGS_OFFSET = 8u,
  LOOM_AMDGPU_ASAN_CONFIG_SHADOW_SCALE_SHIFT_OFFSET = 12u,
  LOOM_AMDGPU_ASAN_CONFIG_SHADOW_BASE_OFFSET = 16u,
  LOOM_AMDGPU_ASAN_CONFIG_APPLICATION_WINDOW_BASE_OFFSET = 24u,
  LOOM_AMDGPU_ASAN_CONFIG_APPLICATION_WINDOW_SIZE_OFFSET = 32u,
  LOOM_AMDGPU_ASAN_CONFIG_SHADOW_SIZE_OFFSET = 40u,
  LOOM_AMDGPU_ASAN_CONFIG_SHADOW_SLAB_SIZE_OFFSET = 48u,
  LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_0_OFFSET = 56u,
  LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_1_OFFSET = 64u,
  LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_2_OFFSET = 72u,
  LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_3_OFFSET = 80u,
  LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_4_OFFSET = 88u,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ABI_ASAN_H_
