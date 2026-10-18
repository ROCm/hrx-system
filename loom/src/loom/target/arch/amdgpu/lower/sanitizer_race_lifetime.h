// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU TSAN workgroup shadow-record lifetime lowering.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_LIFETIME_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_LIFETIME_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_shadow_layout.h"

typedef struct loom_amdgpu_sanitizer_race_config_values_t
    loom_amdgpu_sanitizer_race_config_values_t;

#ifdef __cplusplus
extern "C" {
#endif

// Establishes the current dispatch generation and shadow layout as the owner
// of one workgroup record. A stale record's chunk-state bitmap is cooperatively
// cleared before the function leaves the builder at its continuation block.
iree_status_t loom_amdgpu_sanitizer_race_establish_workgroup_lifetime(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    const loom_amdgpu_sanitizer_race_shadow_layout_t* layout,
    loom_value_id_t workitem_linear_id, loom_value_id_t dispatch_id);

// Advances the workgroup barrier epoch. When the compact history epoch wraps,
// the first wave retires the shared chunk-state bitmap before returning. The
// caller must make every wave execute a workgroup barrier immediately after
// this transition so no later access can observe partial retirement.
iree_status_t loom_amdgpu_sanitizer_race_advance_barrier_epoch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t workgroup_shadow_record_offset,
    loom_value_id_t workgroup_chunk_state_bitmap_size,
    loom_value_id_t workitem_linear_id);

// Establishes every fixed history chunk intersecting an application-memory
// access. Concurrent first accesses elect one retiring lane per chunk and wait
// for its release publication before consulting either compact history.
iree_status_t loom_amdgpu_sanitizer_race_establish_access_lifetime(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_config_values_t* config,
    loom_value_id_t workgroup_shadow_record_offset,
    loom_value_id_t workgroup_shadow_history_offset,
    loom_value_id_t memory_byte_offset, uint32_t byte_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_LIFETIME_H_
