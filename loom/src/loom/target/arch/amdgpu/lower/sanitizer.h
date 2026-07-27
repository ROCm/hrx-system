// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for generic Loom sanitizer assertions.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/ir.h"
#include "loom/sanitizer/site_collection.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_report.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_sanitizer_access_plan_t {
  // Dense sanitizer site ID assigned during source-to-low planning.
  loom_sanitizer_site_id_t site_id;
  // Flat application address plan selected from the asserted view access.
  loom_amdgpu_memory_access_t address;
  // Runtime access kind reported when the assertion fails.
  loom_amdgpu_sanitizer_access_kind_t report_access_kind;
  // Number of application bytes covered by the assertion.
  uint32_t access_size;
  // Minimum byte alignment proven for each repeated access address.
  uint32_t minimum_alignment;
  // Number of regularly-strided sub-accesses covered by the assertion.
  uint16_t static_repeat_count;
  // Static byte stride between consecutive sub-access base addresses.
  uint64_t static_repeat_byte_stride;
} loom_amdgpu_sanitizer_access_plan_t;

// Resolves the shared AMDGPU feedback configuration symbol used by sanitizer
// report producers.
iree_status_t loom_amdgpu_sanitizer_feedback_config_symbol(
    loom_low_lower_context_t* context, loom_symbol_ref_t* out_symbol_ref);

// Resolves the AMDGPU TSAN runtime configuration symbol used by race observers.
iree_status_t loom_amdgpu_sanitizer_tsan_config_symbol(
    loom_low_lower_context_t* context, loom_symbol_ref_t* out_symbol_ref);

// Returns the dense sanitizer site ID assigned to |source_op|.
iree_status_t loom_amdgpu_sanitizer_site_id_for_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_sanitizer_site_id_t* out_site_id);

// Selects an AMDGPU shadow-check lowering for sanitizer.assert.access or
// sanitizer.assert.accesses.
iree_status_t loom_amdgpu_select_sanitizer_assert_access_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_access_plan_t* out_plan, bool* out_selected);

// Verifies sanitizer.assert.access or sanitizer.assert.accesses legality for
// AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_sanitizer_assert_access(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Lowers sanitizer.assert.access or sanitizer.assert.accesses to hot shadow
// checks and cold report CFG.
iree_status_t loom_amdgpu_lower_sanitizer_assert_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_sanitizer_access_plan_t* plan);

// Commits sanitizer site rows discovered while lowering one function into the
// module-scope AMDGPU sanitizer state.
iree_status_t loom_amdgpu_finalize_sanitizer_function(
    loom_low_lower_context_t* context);

// Emits module-level sanitizer metadata required by lowered AMDGPU kernels.
iree_status_t loom_amdgpu_finalize_sanitizer_module(
    loom_module_t* module, loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_H_
