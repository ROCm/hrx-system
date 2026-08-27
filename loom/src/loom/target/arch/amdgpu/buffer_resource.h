// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU buffer-resource descriptor record encodings.

#ifndef LOOM_TARGET_ARCH_AMDGPU_BUFFER_RESOURCE_H_
#define LOOM_TARGET_ARCH_AMDGPU_BUFFER_RESOURCE_H_

#include <stdint.h>

#include "loom/target/arch/amdgpu/target_info_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Legacy-format word 3 for a bounds-checked raw 32-bit resource.
#define LOOM_AMDGPU_BUFFER_RESOURCE_BASE48_LEGACY_RAW_WORD3 UINT32_C(0x00027000)

// Unified-format word 3 for a bounds-checked raw 32-bit resource.
#define LOOM_AMDGPU_BUFFER_RESOURCE_BASE48_UNIFIED_RAW_WORD3 \
  UINT32_C(0x31016000)

// Extended-layout word 3 control for a bounds-checked raw resource.
#define LOOM_AMDGPU_BUFFER_RESOURCE_BASE57_RAW_WORD3 UINT32_C(0x00000000)

typedef struct loom_amdgpu_buffer_resource_record_encoding_info_t {
  // Mask applied to the loaded pointer high word, or zero when none is needed.
  uint32_t pointer_high_mask;
  // Final descriptor word 3 control bits for a bounds-checked raw resource.
  uint32_t raw_bounds_checked_word3_control;
  // Low num_records bits packed above the base address in descriptor word 1.
  uint8_t num_records_word1_bit_count;
} loom_amdgpu_buffer_resource_record_encoding_info_t;

// Returns the checked-in physical record facts for |record_encoding|.
//
// The encoding must be a non-NONE value produced by the generated AMDGPU
// descriptor-set target facts.
const loom_amdgpu_buffer_resource_record_encoding_info_t*
loom_amdgpu_buffer_resource_record_encoding_info(
    loom_amdgpu_buffer_resource_record_encoding_t record_encoding);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_BUFFER_RESOURCE_H_
