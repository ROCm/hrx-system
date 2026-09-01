// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Value-fact queries shared by rule matching and attribute projection.

#ifndef LOOM_CODEGEN_LOW_LOWER_LOWER_RULE_FACTS_H_
#define LOOM_CODEGEN_LOW_LOWER_LOWER_RULE_FACTS_H_

#include "loom/codegen/low/lower/lower_rules.h"

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

// Returns immediate integer facts for a scalar or uniform vector value.
bool loom_low_lower_rule_integer_immediate_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts);

// Returns immediate float facts for a scalar or uniform vector value.
bool loom_low_lower_rule_float_immediate_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts);

// Derives the unsigned division recipe for an exact source value.
bool loom_low_lower_rule_value_facts_u32_divisor_magic_info(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id,
    loom_low_lower_u32_divisor_magic_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_LOWER_RULE_FACTS_H_
