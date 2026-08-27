// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/loop_domain.h"

#include "loom/ir/facts.h"

static bool loom_loop_domain_value_equal(
    const loom_value_fact_table_t* fact_table, loom_value_id_t lhs,
    loom_value_id_t rhs) {
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  if (lhs == rhs) return true;
  if (!fact_table) return false;

  loom_value_facts_t lhs_facts = loom_value_fact_table_lookup(fact_table, lhs);
  loom_value_facts_t rhs_facts = loom_value_fact_table_lookup(fact_table, rhs);
  return loom_value_facts_is_exact(lhs_facts) &&
         loom_value_facts_is_exact(rhs_facts) &&
         !loom_value_facts_is_float(lhs_facts) &&
         !loom_value_facts_is_float(rhs_facts) &&
         lhs_facts.range_lo == rhs_facts.range_lo;
}

bool loom_loop_domain_equal(const loom_value_fact_table_t* fact_table,
                            loom_loop_domain_t lhs, loom_loop_domain_t rhs) {
  return loom_loop_domain_value_equal(fact_table, lhs.lower_bound,
                                      rhs.lower_bound) &&
         loom_loop_domain_value_equal(fact_table, lhs.upper_bound,
                                      rhs.upper_bound) &&
         loom_loop_domain_value_equal(fact_table, lhs.step, rhs.step);
}

static bool loom_loop_domain_lookup_range_facts(
    const loom_value_fact_table_t* fact_table, loom_loop_domain_t domain,
    loom_value_facts_t* out_lower_bound, loom_value_facts_t* out_upper_bound,
    loom_value_facts_t* out_step) {
  if (!fact_table || domain.lower_bound == LOOM_VALUE_ID_INVALID ||
      domain.upper_bound == LOOM_VALUE_ID_INVALID ||
      domain.step == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  *out_lower_bound =
      loom_value_fact_table_lookup(fact_table, domain.lower_bound);
  *out_upper_bound =
      loom_value_fact_table_lookup(fact_table, domain.upper_bound);
  *out_step = loom_value_fact_table_lookup(fact_table, domain.step);
  return !loom_value_facts_is_float(*out_lower_bound) &&
         !loom_value_facts_is_float(*out_upper_bound) &&
         !loom_value_facts_is_float(*out_step) &&
         loom_value_facts_is_positive(*out_step);
}

bool loom_loop_domain_proven_empty(const loom_value_fact_table_t* fact_table,
                                   loom_loop_domain_t domain) {
  loom_value_facts_t lower_bound = {0};
  loom_value_facts_t upper_bound = {0};
  loom_value_facts_t step = {0};
  if (!loom_loop_domain_lookup_range_facts(fact_table, domain, &lower_bound,
                                           &upper_bound, &step)) {
    return false;
  }
  return lower_bound.range_lo >= upper_bound.range_hi;
}

bool loom_loop_domain_proven_nonempty(const loom_value_fact_table_t* fact_table,
                                      loom_loop_domain_t domain) {
  loom_value_facts_t lower_bound = {0};
  loom_value_facts_t upper_bound = {0};
  loom_value_facts_t step = {0};
  if (!loom_loop_domain_lookup_range_facts(fact_table, domain, &lower_bound,
                                           &upper_bound, &step)) {
    return false;
  }
  return lower_bound.range_hi < upper_bound.range_lo;
}
