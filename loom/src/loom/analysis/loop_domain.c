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
