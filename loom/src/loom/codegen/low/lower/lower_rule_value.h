// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-value projection for generated source-to-Low lowering rules.
//
// Generated rule tables address source operands and results through compact
// value-ref rows. Matching and emission both consume those rows and must agree
// on how source values and scalar facts are projected. This component owns that
// shared interpretation; it does not select rules or emit target-Low IR.

#ifndef LOOM_CODEGEN_LOW_LOWER_LOWER_RULE_VALUE_H_
#define LOOM_CODEGEN_LOW_LOWER_LOWER_RULE_VALUE_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_u32_divisor_magic_info_t {
  // Multiplication constant consumed by the unsigned quotient recipe.
  uint32_t multiplier;
  // Final logical right shift applied after the high-half multiply.
  uint8_t post_shift;
  // Whether the quotient recipe needs the unsigned add-adjustment step.
  bool is_add;
} loom_low_lower_u32_divisor_magic_info_t;

// Returns the target-owned materializer selected by a nonzero value-ref
// materializer index. Generated table indices are trusted.
const loom_low_lower_value_materializer_t*
loom_low_lower_rule_value_materializer(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_value_ref_t* value_ref);

// Resolves an operand or result value-ref row to its source SSA value.
// Generated table indices and value-ref kinds are trusted.
loom_value_id_t loom_low_lower_rule_source_value(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, uint16_t value_ref_index);

// Resolves an operand or result value-ref row to the complete source field.
// Non-source value refs return an empty span.
loom_value_slice_t loom_low_lower_rule_value_ref_field_span(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, uint16_t value_ref_index);

// Projects scalar or uniform-vector integer facts for one source value.
bool loom_low_lower_rule_integer_immediate_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts);

// Projects scalar or uniform-vector floating-point facts for one source value.
bool loom_low_lower_rule_float_immediate_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts);

// Derives the unsigned 32-bit constant-divisor recipe for an exact source
// value. Returns false when the value is unavailable or outside [2,
// UINT32_MAX].
bool loom_low_lower_rule_value_facts_u32_divisor_magic_info(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id,
    loom_low_lower_u32_divisor_magic_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_LOWER_RULE_VALUE_H_
