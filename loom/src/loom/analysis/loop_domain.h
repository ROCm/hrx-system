// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loop-domain comparison helpers.
//
// A loop domain is the half-open counted range [lower_bound, upper_bound) with
// a positive step. The representation deliberately stores only SSA value IDs so
// it can describe scf.for today and tile/vector lane loops later without making
// the analysis depend on any specific dialect.

#ifndef LOOM_ANALYSIS_LOOP_DOMAIN_H_
#define LOOM_ANALYSIS_LOOP_DOMAIN_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_loop_domain_t {
  // Inclusive lower bound of the counted loop domain.
  loom_value_id_t lower_bound;
  // Exclusive upper bound of the counted loop domain.
  loom_value_id_t upper_bound;
  // Positive step between consecutive induction values.
  loom_value_id_t step;
} loom_loop_domain_t;

// Returns true when both domains are proven identical by SSA identity or exact
// integer value facts. Non-exact facts deliberately do not prove equality: two
// values with the same range may still differ at runtime.
bool loom_loop_domain_equal(const loom_value_fact_table_t* fact_table,
                            loom_loop_domain_t lhs, loom_loop_domain_t rhs);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_LOOP_DOMAIN_H_
