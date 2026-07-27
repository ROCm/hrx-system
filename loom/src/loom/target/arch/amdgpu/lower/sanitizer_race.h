// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU sanitizer race-observation legality.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/sanitizer/site_collection.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_sanitizer_race_access_plan_t {
  // Compiler-assigned source site identifier for reports.
  loom_sanitizer_site_id_t site_id;
  // Selected memory-space-relative byte address of the observed LDS access.
  loom_amdgpu_memory_access_t address;
  // Report access kind emitted for the current access.
  loom_amdgpu_tsan_access_kind_t report_access_kind;
  // Compact shadow access kind written to the per-granule detector slot.
  loom_amdgpu_tsan_shadow_access_kind_t shadow_access_kind;
  // Static access width in bytes.
  uint32_t access_size;
  // Whether the current access came from an atomic memory operation.
  bool atomic;
} loom_amdgpu_sanitizer_race_access_plan_t;

typedef struct loom_amdgpu_sanitizer_race_sync_plan_t {
  // Workgroup barrier emitted after the epoch update.
  loom_amdgpu_kernel_barrier_plan_t barrier;
} loom_amdgpu_sanitizer_race_sync_plan_t;

// Selects the AMDGPU lowering plan for a sanitizer.race.access op.
iree_status_t loom_amdgpu_select_sanitizer_race_access_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_access_plan_t* out_plan, bool* out_selected);

// Emits entry-block setup needed by sanitizer.race access/sync lowering.
iree_status_t loom_amdgpu_sanitizer_race_emit_entry_setup(
    loom_low_lower_context_t* context);

// Lowers a sanitizer.race.access op into the LDS TSAN detector state machine.
iree_status_t loom_amdgpu_lower_sanitizer_race_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_access_plan_t* plan);

// Selects the AMDGPU lowering plan for a sanitizer.race.sync op.
iree_status_t loom_amdgpu_select_sanitizer_race_sync_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_sync_plan_t* out_plan, bool* out_selected);

// Lowers a sanitizer.race.sync op into a per-workgroup detector epoch update.
iree_status_t loom_amdgpu_lower_sanitizer_race_sync(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_race_sync_plan_t* plan);

// Verifies sanitizer.race.access legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_access(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies sanitizer.race.sync legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_sync(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_H_
