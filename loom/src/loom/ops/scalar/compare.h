// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared proof helpers for scalar.cmpi and scalar.cmpf.
//
// These helpers centralize comparison facts used by fact inference and
// structural canonicalization. Integer range proofs use signed fact intervals
// directly and only reuse them for unsigned predicates when both ranges are
// proven non-negative.

#ifndef LOOM_OPS_SCALAR_COMPARE_H_
#define LOOM_OPS_SCALAR_COMPARE_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/ir/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Proves the result of comparing one integer SSA value to itself.
bool loom_scalar_cmpi_same_value_result(uint8_t predicate, bool* out_result);

// Returns the equivalent integer predicate after swapping lhs and rhs.
uint8_t loom_scalar_cmpi_swapped_predicate(uint8_t predicate);

// Proves an integer comparison result from the operand fact summaries.
bool loom_scalar_cmpi_result_from_facts(uint8_t predicate,
                                        const loom_value_facts_t* lhs_facts,
                                        const loom_value_facts_t* rhs_facts,
                                        bool* out_result);

// Proves the result of comparing one non-NaN float SSA value to itself.
bool loom_scalar_cmpf_same_value_result(uint8_t predicate, bool* out_result);

// Proves a float comparison result from exact operand values.
bool loom_scalar_cmpf_exact_result(uint8_t predicate, double lhs, double rhs,
                                   bool* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_SCALAR_COMPARE_H_
