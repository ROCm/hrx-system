// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured loop fact solving. The loop owner seeds counted induction values,
// solves the carried-state forwarding equations, and publishes body and result
// facts together. Nested regions use the table's current function context.

#ifndef LOOM_UTIL_FACT_LOOP_H_
#define LOOM_UTIL_FACT_LOOP_H_

#include "loom/analysis/loop_domain.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scoped evaluator for a counted loop whose body equations are already current.
// The solver retains widening, recurrence, and exact-replay policy; the updater
// publishes changed carried arguments and all affected body facts before
// return. Nested loops keep their own evaluator unless explicitly bound in
// turn.
typedef struct loom_value_fact_loop_update_t {
  // Counted loop whose body is maintained by this updater.
  loom_op_t* op;
  // Borrowed owner of the maintained dependency/publication machinery.
  void* user_data;
  // Defines the carried arguments and completes their dependent equations.
  iree_status_t (*fn)(void* user_data, loom_region_t* region,
                      uint16_t argument_offset, const loom_value_facts_t* facts,
                      uint16_t count);
} loom_value_fact_loop_update_t;

// Seeds the counted induction argument when block is the LoopLike body entry.
iree_status_t loom_value_fact_table_seed_loop_iv_arg(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op);

// Retains the current counted domain in the body's fact-scope lifetime.
// Reseeding an induction argument replaces the previous domain identities.
iree_status_t loom_value_fact_table_set_counted_loop_domain(
    loom_value_fact_table_t* table, const loom_region_t* body,
    loom_value_id_t induction_value, loom_loop_domain_t domain);

// Returns the retained domain when value is its body's counted induction
// argument. Lookup follows the defining block directly, without a scope walk.
const loom_loop_domain_t* loom_value_fact_table_lookup_counted_loop_domain(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value);

// Computes the complete LoopLike summary, including nested region facts.
// Reports result changes to the caller-maintained rewrite worklist.
iree_status_t loom_value_fact_table_compute_loop_like_summary(
    loom_value_fact_table_t* table, const loom_module_t* module, loom_op_t* op,
    bool* out_changed);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_LOOP_H_
