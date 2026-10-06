// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU byte-permutation packet selection and emission.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_BYTE_PERMUTE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_BYTE_PERMUTE_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_byte_permute_selector_cache_entry_t {
  // Byte selector payload represented by this cache entry.
  uint32_t immediate;
  // SGPR materialized for the selector payload.
  loom_value_id_t register_value;
} loom_amdgpu_byte_permute_selector_cache_entry_t;

typedef struct loom_amdgpu_byte_permute_emitter_t {
  // Packet form selected from the active target descriptor set.
  loom_amdgpu_byte_permute_plan_t plan;
  // Resolved descriptor used by every emitted permutation.
  loom_low_lower_resolved_descriptor_t descriptor;
  // SGPR type used to materialize selectors for the regular packet form.
  loom_type_t selector_type;
  // Most recently used selector SGPRs cached by immediate payload.
  loom_amdgpu_byte_permute_selector_cache_entry_t
      selector_cache[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  // Number of populated entries in |selector_cache|.
  uint32_t selector_count;
} loom_amdgpu_byte_permute_emitter_t;

// Selects the best available full-register byte permutation packet.
void loom_amdgpu_select_byte_permute_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_byte_permute_plan_t* out_plan);

// Resolves the selected packet form and initializes selector interning state.
iree_status_t loom_amdgpu_byte_permute_emitter_initialize(
    loom_low_lower_context_t* context,
    const loom_amdgpu_byte_permute_plan_t* plan,
    loom_amdgpu_byte_permute_emitter_t* out_emitter);

// Emits one byte permutation, reusing recent register selectors when required.
iree_status_t loom_amdgpu_byte_permute_emitter_emit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_byte_permute_emitter_t* emitter, loom_value_id_t source0,
    loom_value_id_t source1, uint32_t selector, loom_type_t result_type,
    loom_value_id_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_BYTE_PERMUTE_H_
