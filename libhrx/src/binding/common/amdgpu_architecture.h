// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_AMDGPU_ARCHITECTURE_H_
#define LIBHRX_SRC_BINDING_COMMON_AMDGPU_ARCHITECTURE_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Structured version decoded from an exact AMDGPU target identifier.
typedef struct iree_hal_streaming_amdgpu_architecture_t {
  // Major gfx IP version.
  uint32_t major;
  // Minor gfx IP version.
  uint32_t minor;
  // Gfx IP stepping, decoded as a hexadecimal digit.
  uint32_t stepping;
} iree_hal_streaming_amdgpu_architecture_t;

// Decodes an exact target such as `gfx942` or `gfx90a:xnack+`.
//
// Feature suffixes use AMDHSA target-ID syntax. The output is left unchanged
// when |value| is malformed or names an unsupported target-ID feature.
bool iree_hal_streaming_parse_amdgpu_architecture(
    const char* value,
    iree_hal_streaming_amdgpu_architecture_t* out_architecture);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LIBHRX_SRC_BINDING_COMMON_AMDGPU_ARCHITECTURE_H_
