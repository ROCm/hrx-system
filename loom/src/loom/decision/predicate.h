// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_DECISION_PREDICATE_H_
#define LOOM_DECISION_PREDICATE_H_

#include <stdint.h>

#include "loom/ir/facts.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Ternary result of proving a decision predicate from available facts.
typedef uint8_t loom_decision_truth_t;
enum loom_decision_truth_e {
  // Available facts admit values for which the predicate is both true and
  // false.
  LOOM_DECISION_TRUTH_UNKNOWN = 0,
  // Available facts prove that the predicate is false.
  LOOM_DECISION_TRUTH_FALSE = 1,
  // Available facts prove that the predicate is true.
  LOOM_DECISION_TRUTH_TRUE = 2,
};

// Sentinel identity for operands that do not represent a runtime value.
#define LOOM_DECISION_OPERAND_IDENTITY_NONE UINT32_MAX

// A predicate operand resolved into facts and caller-local runtime identity.
//
// Equal non-sentinel identities prove that two operands denote the same
// runtime value even when their scalar facts are otherwise unknown. Identities
// have no meaning across calls and are never serialized.
typedef struct loom_decision_predicate_operand_t {
  // Scalar facts available for the operand.
  loom_value_facts_t facts;
  // Caller-local runtime identity or LOOM_DECISION_OPERAND_IDENTITY_NONE.
  uint32_t identity;
} loom_decision_predicate_operand_t;

// Proves |predicate_kind| from its resolved |operands|.
//
// The caller must provide a verified, type-compatible predicate kind and
// initialize the number of operands declared by
// loom_predicate_kind_argument_count(). Constants are represented by exact
// integer facts with no runtime identity. This operation is allocation-free
// and evaluates only the scalar fact domain; path-sensitive relations and
// contextual target facts are the responsibility of adapters that can prove
// them before falling back to this function.
loom_decision_truth_t loom_decision_predicate_evaluate(
    loom_predicate_kind_t predicate_kind,
    const loom_decision_predicate_operand_t operands[3]);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_DECISION_PREDICATE_H_
