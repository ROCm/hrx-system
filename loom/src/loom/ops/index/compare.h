// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared proof helpers for index.cmp.
//
// Source facts use an i64 mathematical envelope while target-scoped index and
// offset values have explicit carrier widths. These helpers keep comparison
// proofs in the predicate's physical carrier domain so a source range outside
// that domain cannot be mistaken for its truncated signed or unsigned view.

#ifndef LOOM_OPS_INDEX_COMPARE_H_
#define LOOM_OPS_INDEX_COMPARE_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/ir/scalar_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// Proves the result of comparing one SSA value to itself.
bool loom_index_cmp_same_value_result(uint8_t predicate, bool* out_result);

// Returns the common carrier width established for two source SSA values.
// Producer-backed index values may select the full source carrier; raw fact
// summaries alone retain the target-native interpretation exposed by the
// helpers below. Zero indicates a targetless fact context and -1 indicates an
// invalid carrier.
int32_t loom_index_cmp_value_carrier_bitwidth(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_scalar_type_t operand_scalar_type, loom_value_id_t lhs,
    const loom_value_facts_t* lhs_facts, loom_value_id_t rhs,
    const loom_value_facts_t* rhs_facts);

// Returns true when both fact summaries fit |carrier_bitwidth| in the view
// required by |predicate|. Zero retains a targetless mathematical source view;
// a negative width is invalid.
bool loom_index_cmp_facts_fit_carrier(int32_t carrier_bitwidth,
                                      uint8_t predicate,
                                      const loom_value_facts_t* lhs_facts,
                                      const loom_value_facts_t* rhs_facts);

// Proves the comparison result from two operand fact summaries interpreted in
// the target carrier view required by |predicate|. Unsigned ordering accepts
// either signed or unsigned representations of that carrier. Ranges outside
// it contribute only the full carrier domain, retaining universal endpoint
// proofs without treating mathematical overflow as a particular bit pattern.
bool loom_index_cmp_result_from_facts(const loom_fact_context_t* context,
                                      loom_scalar_type_t operand_scalar_type,
                                      uint8_t predicate,
                                      const loom_value_facts_t* lhs_facts,
                                      const loom_value_facts_t* rhs_facts,
                                      bool* out_result);

// Proves the comparison result for two source SSA values in their established
// common carrier. This is the module-backed counterpart to the raw-facts query
// above and performs only constant-time value-table lookups.
bool loom_index_cmp_result_from_value_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_scalar_type_t operand_scalar_type, uint8_t predicate,
    loom_value_id_t lhs, const loom_value_facts_t* lhs_facts,
    loom_value_id_t rhs, const loom_value_facts_t* rhs_facts, bool* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_INDEX_COMPARE_H_
