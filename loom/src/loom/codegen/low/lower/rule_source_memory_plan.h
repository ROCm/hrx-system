// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_LOWER_RULE_SOURCE_MEMORY_PLAN_H_
#define LOOM_CODEGEN_LOW_LOWER_RULE_SOURCE_MEMORY_PLAN_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower/rules.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_selected_plan_t loom_low_lower_selected_plan_t;
typedef struct loom_low_lower_resolved_emit_t loom_low_lower_resolved_emit_t;
typedef struct loom_low_lower_rule_byte_offset_entry_t
    loom_low_lower_rule_byte_offset_entry_t;

// Construction scratch shared by the final selected-rule traversal. Logical
// identities come from source observation; this table binds their physical
// byte arithmetic within an emission block. No source expressions are walked.
typedef struct loom_low_lower_rule_source_memory_plan_builder_t {
  // Scratch arena lasting through finalization of all selected rules.
  iree_arena_allocator_t* arena;
  // Physical recipes indexed by logical identity, carrier, bias, and block.
  loom_low_lower_rule_byte_offset_entry_t** buckets;
  // Power-of-two bucket count, or zero before the first byte-offset use.
  iree_host_size_t bucket_count;
  // Number of distinct physical recipes.
  iree_host_size_t entry_count;
} loom_low_lower_rule_source_memory_plan_builder_t;

// Retains a direct reference to a function-owned byte-offset binding. The
// earliest actual emit/operand use defines it. Finalization may update that
// definition while every previously retained reference remains valid.
iree_status_t loom_low_lower_rule_source_memory_plan_byte_offset(
    loom_low_lower_context_t* context,
    loom_low_lower_rule_source_memory_plan_builder_t* builder,
    const loom_low_lower_selected_plan_t* selected_plan,
    const loom_low_lower_resolved_emit_t* resolved_emit, uint16_t emit_ordinal,
    uint16_t operand_ordinal, const void** out_binding);

// Executes the original arithmetic recipe at its planned definition, or reads
// the emitted value at a later use. Only the binding's Low value is populated
// during emission; the definition is immutable after finalization. Destructive
// consumers retain their descriptor-required copies of this shared value.
iree_status_t loom_low_lower_rule_source_memory_emit_byte_offset(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    uint16_t operand_ordinal, loom_value_id_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_RULE_SOURCE_MEMORY_PLAN_H_
