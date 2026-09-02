// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generated source-to-Low rule emission.
//
// Emission consumes one rule selected during planning and its pre-resolved Low
// descriptors. It interprets the rule's emit program into target-Low operands,
// attributes, packets, result bindings, aliases, and elisions.

#ifndef LOOM_CODEGEN_LOW_LOWER_RULE_EMIT_H_
#define LOOM_CODEGEN_LOW_LOWER_RULE_EMIT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/rules.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_resolved_emit_t {
  // Static emit-program row selected by planning.
  const loom_low_lower_emit_t* emit;
  // Descriptor row referenced by |emit| and resolved during planning.
  loom_low_lower_resolved_descriptor_t descriptor;
} loom_low_lower_resolved_emit_t;

// Resolves descriptor-backed emit rows after selection. Returned rows are
// function-arena-owned and remain valid for the current lowering run.
iree_status_t loom_low_lower_rule_set_resolve_emit_program(
    loom_low_lower_context_t* context, uint16_t rule_set_index,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_t* rule,
    const loom_low_lower_resolved_emit_t** out_resolved_emits);

// Emits target-Low packets for |source_op| using a previously selected rule.
iree_status_t loom_low_lower_rule_set_emit_rule(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule,
    const loom_low_lower_resolved_emit_t* resolved_emits,
    const loom_low_source_memory_access_plan_t* source_memory_access);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_RULE_EMIT_H_
