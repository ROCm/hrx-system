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
#include "loom/ir/scalar_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// Proves the result of comparing one SSA value to itself.
bool loom_index_cmp_same_value_result(uint8_t predicate, bool* out_result);

// Returns true when both fact summaries fit the target carrier view required
// by |predicate|. Targetless facts retain their mathematical source view.
bool loom_index_cmp_facts_fit_target_carrier(
    const loom_fact_context_t* context, loom_scalar_type_t operand_scalar_type,
    uint8_t predicate, const loom_value_facts_t* lhs_facts,
    const loom_value_facts_t* rhs_facts);

// Proves the comparison result from two operand fact summaries interpreted in
// the target carrier view required by |predicate|.
bool loom_index_cmp_result_from_facts(const loom_fact_context_t* context,
                                      loom_scalar_type_t operand_scalar_type,
                                      uint8_t predicate,
                                      const loom_value_facts_t* lhs_facts,
                                      const loom_value_facts_t* rhs_facts,
                                      bool* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_INDEX_COMPARE_H_
