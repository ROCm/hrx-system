// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generated lowering-rule value reference resolution.

#ifndef LOOM_CODEGEN_LOW_LOWER_LOWER_RULE_VALUE_H_
#define LOOM_CODEGEN_LOW_LOWER_LOWER_RULE_VALUE_H_

#include "loom/codegen/low/lower/lower_rules.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_rule_emit_state_t {
  // Rule-local low SSA values captured by earlier emit rows.
  loom_value_id_t* temporaries;
  // Number of entries in temporaries.
  uint16_t temporary_count;
} loom_low_lower_rule_emit_state_t;

// Returns the materializer referenced by |value_ref|.
const loom_low_lower_value_materializer_t*
loom_low_lower_rule_value_materializer(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_value_ref_t* value_ref);

// Resolves a source operand or result value reference.
loom_value_id_t loom_low_lower_rule_source_value(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, uint16_t value_ref_index);

// Resolves one rule value reference to its emitted Low SSA value.
iree_status_t loom_low_lower_rule_low_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    uint16_t value_ref_index, loom_value_id_t* out_low_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_LOWER_RULE_VALUE_H_
