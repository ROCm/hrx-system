// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU TSAN entry state and control-flow lowering.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_STATE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_STATE_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race_report.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_report.h"
#include "loom/target/registers.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_sanitizer_race_config_values_t {
  // Address of the runtime-published TSAN config global.
  loom_value_id_t address;
  // TSAN config flags from loom_amdgpu_tsan_config_flag_bits_e.
  loom_value_id_t flags;
  // Queue-local shadow allocation base.
  loom_value_id_t shadow_base;
  // Total bytes in the queue-local shadow allocation.
  loom_value_id_t shadow_size;
} loom_amdgpu_sanitizer_race_config_values_t;

typedef struct loom_amdgpu_sanitizer_race_shadow_entry_base_t {
  // Low entry bits common to every access by this workitem in this dispatch.
  loom_value_id_t low;
  // High entry generation bits common to every access in this dispatch.
  loom_value_id_t high;
} loom_amdgpu_sanitizer_race_shadow_entry_base_t;

typedef struct loom_amdgpu_sanitizer_race_report_coordinates_t {
  // Device-visible pointer to the active AQL dispatch packet.
  loom_value_id_t dispatch_ptr;
  // Three-dimensional workgroup coordinates ordered X, Y, Z.
  loom_value_id_t workgroup_ids[LOOM_KERNEL_DIMENSION_COUNT_];
} loom_amdgpu_sanitizer_race_report_coordinates_t;

// Returns the entry-dominating runtime TSAN configuration.
iree_status_t loom_amdgpu_sanitizer_race_get_config_values(
    loom_low_lower_context_t* context,
    loom_amdgpu_sanitizer_race_config_values_t* out_config);

// Returns the current workitem's flattened VGPR identifier.
iree_status_t loom_amdgpu_sanitizer_race_get_workitem_linear_id(
    loom_low_lower_context_t* context, loom_value_id_t* out_linear_id);

// Returns device report coordinates materialized by entry setup.
iree_status_t loom_amdgpu_sanitizer_race_get_report_coordinates(
    loom_low_lower_context_t* context,
    const loom_amdgpu_sanitizer_race_report_coordinates_t** out_coordinates);

// Returns the byte offset of the current workgroup's shadow record.
iree_status_t loom_amdgpu_sanitizer_race_get_workgroup_shadow_record_offset(
    loom_low_lower_context_t* context, loom_value_id_t* out_offset);

// Returns the byte offset of the current workgroup's padded history array.
iree_status_t loom_amdgpu_sanitizer_race_get_workgroup_shadow_history_offset(
    loom_low_lower_context_t* context, loom_value_id_t* out_offset);

// Returns shadow entry bits shared by the current workitem's observations.
iree_status_t loom_amdgpu_sanitizer_race_get_shadow_entry_base(
    loom_low_lower_context_t* context,
    const loom_amdgpu_sanitizer_race_shadow_entry_base_t** out_base);

// Splits an observation or synchronization site around the runtime TSAN guard
// and returns the entry-dominating configuration to the enabled path.
iree_status_t loom_amdgpu_sanitizer_race_build_config_guard(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_config_values_t* out_config,
    loom_block_t** out_continuation_block);

// Branches to a previously constructed continuation block.
iree_status_t loom_amdgpu_sanitizer_race_branch_to_continuation(
    loom_builder_t* builder, loom_block_t* continuation_block,
    loom_location_id_t location);

// Returns the shared cold report island for failed race observations.
iree_status_t loom_amdgpu_sanitizer_race_get_report_island(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_amdgpu_sanitizer_race_report_island_t** out_island);

// Returns the shared cold trap island for failed sanitizer checks.
iree_status_t loom_amdgpu_sanitizer_race_get_trap_island(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_amdgpu_sanitizer_trap_island_t** out_island);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_STATE_H_
