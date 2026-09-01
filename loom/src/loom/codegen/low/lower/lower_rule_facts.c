// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/lower_rule_facts.h"

#include <stdint.h>

#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"

static loom_low_lower_u32_divisor_magic_info_t
loom_low_lower_u32_divisor_magic_info(uint32_t divisor) {
  IREE_ASSERT_GT(divisor, 1u);
  const uint64_t u32_mask = UINT32_MAX;
  const uint64_t two31 = ((uint64_t)1) << 31;
  const uint64_t two31_minus_one = two31 - 1;

  const uint64_t nc = u32_mask - ((u32_mask + 1u - divisor) % divisor);
  uint32_t p = 31;
  uint64_t q1 = two31 / nc;
  uint64_t r1 = two31 % nc;
  uint64_t q2 = two31_minus_one / divisor;
  uint64_t r2 = two31_minus_one % divisor;
  bool is_add = false;
  for (;;) {
    ++p;
    if (r1 >= nc - r1) {
      q1 = ((q1 << 1) + 1) & u32_mask;
      r1 = ((r1 << 1) - nc) & u32_mask;
    } else {
      q1 = (q1 << 1) & u32_mask;
      r1 = (r1 << 1) & u32_mask;
    }
    if (r2 + 1 >= divisor - r2) {
      if (q2 >= two31_minus_one) {
        is_add = true;
      }
      q2 = ((q2 << 1) + 1) & u32_mask;
      r2 = ((r2 << 1) + 1 - divisor) & u32_mask;
    } else {
      if (q2 >= two31) {
        is_add = true;
      }
      q2 = (q2 << 1) & u32_mask;
      r2 = ((r2 << 1) + 1) & u32_mask;
    }
    const uint64_t delta = (divisor - 1 - r2) & u32_mask;
    if (!(p < 64 && (q1 < delta || (q1 == delta && r1 == 0)))) {
      break;
    }
  }

  loom_low_lower_u32_divisor_magic_info_t info = {
      .multiplier = (uint32_t)((q2 + 1) & u32_mask),
      .post_shift = (uint8_t)(p - 32),
      .is_add = is_add,
  };
  if (info.is_add) {
    IREE_ASSERT_GT(info.post_shift, 0);
    --info.post_shift;
  }
  return info;
}

bool loom_low_lower_rule_integer_immediate_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (fact_table == NULL) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_type_is_vector(type)) {
    if (loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
    loom_value_fact_uniform_element_t uniform = {0};
    if (!loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                                &uniform)) {
      return false;
    }
    facts = uniform.element;
  } else if (loom_type_is_scalar(type)) {
    if (loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
  } else {
    return false;
  }
  *out_facts = facts;
  return true;
}

bool loom_low_lower_rule_float_immediate_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (fact_table == NULL) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_type_is_vector(type)) {
    if (!loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
    loom_value_fact_uniform_element_t uniform = {0};
    if (!loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                                &uniform)) {
      return false;
    }
    facts = uniform.element;
  } else if (loom_type_is_scalar(type)) {
    if (!loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
  } else {
    return false;
  }
  *out_facts = facts;
  return true;
}

bool loom_low_lower_rule_value_facts_u32_divisor_magic_info(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id,
    loom_low_lower_u32_divisor_magic_info_t* out_info) {
  *out_info = (loom_low_lower_u32_divisor_magic_info_t){0};
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_immediate_facts(module, fact_table, value_id,
                                                   &facts)) {
    return false;
  }
  int64_t exact_value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &exact_value) || exact_value < 2 ||
      exact_value > UINT32_MAX) {
    return false;
  }
  *out_info = loom_low_lower_u32_divisor_magic_info((uint32_t)exact_value);
  return true;
}
