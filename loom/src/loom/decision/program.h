// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable ranked decision programs over SSA value facts.
//
// A program contains one hard conjunction followed by priority-ranked
// alternative conjunctions. Alternatives combine scalar predicates with
// adapter-owned contextual features. Evaluation is allocation-free, visits
// only the priority prefix needed to decide the result, and emits the exact
// action frontier a caller must preserve.

#ifndef LOOM_DECISION_PROGRAM_H_
#define LOOM_DECISION_PROGRAM_H_

#include <stdint.h>

#include "loom/decision/predicate.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Encoded predicate operand reference.
typedef uint32_t loom_decision_program_operand_ref_t;

// Encoded unresolved constraint reference.
typedef uint32_t loom_decision_program_constraint_ref_t;

// Invalid action ordinal.
#define LOOM_DECISION_PROGRAM_ACTION_INVALID UINT32_MAX

// Invalid constraint reference.
#define LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID UINT32_MAX

// Maximum constant-table ordinal representable by an operand reference.
#define LOOM_DECISION_PROGRAM_CONSTANT_ORDINAL_MAX UINT32_C(0x3fffffff)

// Maximum scalar-predicate ordinal representable by a constraint reference.
#define LOOM_DECISION_PROGRAM_PREDICATE_ORDINAL_MAX UINT32_C(0x7fffffff)

// Maximum contextual-feature ordinal representable by a constraint reference.
#define LOOM_DECISION_PROGRAM_FEATURE_ORDINAL_MAX UINT32_C(0x7ffffffe)

// Returns an argument operand reference.
static inline loom_decision_program_operand_ref_t
loom_decision_program_argument_ref(uint32_t ordinal) {
  return ordinal;
}

// Returns a result operand reference.
static inline loom_decision_program_operand_ref_t
loom_decision_program_result_ref(uint32_t ordinal) {
  return UINT32_C(0x40000000) | ordinal;
}

// Returns a constant-table operand reference.
static inline loom_decision_program_operand_ref_t
loom_decision_program_constant_ref(uint32_t ordinal) {
  return UINT32_C(0x80000000) | ordinal;
}

// Returns true when |ref| identifies a constant-table operand.
static inline bool loom_decision_program_operand_is_constant(
    loom_decision_program_operand_ref_t ref) {
  return (ref & UINT32_C(0x80000000)) != 0;
}

// Returns true when |ref| identifies an application result operand.
static inline bool loom_decision_program_operand_is_result(
    loom_decision_program_operand_ref_t ref) {
  return !loom_decision_program_operand_is_constant(ref) &&
         (ref & UINT32_C(0x40000000)) != 0;
}

// Returns the argument, result, or constant ordinal encoded in |ref|.
static inline uint32_t loom_decision_program_operand_ordinal(
    loom_decision_program_operand_ref_t ref) {
  return ref & UINT32_C(0x3fffffff);
}

// Returns a scalar-predicate constraint reference.
static inline loom_decision_program_constraint_ref_t
loom_decision_program_predicate_constraint_ref(uint32_t ordinal) {
  return ordinal;
}

// Returns a contextual-feature constraint reference.
static inline loom_decision_program_constraint_ref_t
loom_decision_program_feature_constraint_ref(uint32_t ordinal) {
  return UINT32_C(0x80000000) | ordinal;
}

// Returns true when |ref| identifies a contextual feature.
static inline bool loom_decision_program_constraint_is_feature(
    loom_decision_program_constraint_ref_t ref) {
  return ref != LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID &&
         (ref & UINT32_C(0x80000000)) != 0;
}

// Returns the predicate or contextual-feature ordinal encoded in |ref|.
static inline uint32_t loom_decision_program_constraint_ordinal(
    loom_decision_program_constraint_ref_t ref) {
  return ref & UINT32_C(0x7fffffff);
}

// One scalar fact predicate over signature inputs and constants.
typedef struct loom_decision_program_predicate_t {
  // Predicate kind from loom_predicate_kind_t.
  uint8_t kind;

  // Number of initialized operand references.
  uint8_t operand_count;

  // Reserved bytes. Always zero.
  uint8_t reserved[2];

  // Operand references in predicate argument order.
  loom_decision_program_operand_ref_t operands[3];
} loom_decision_program_predicate_t;

static_assert(sizeof(loom_decision_program_predicate_t) == 16,
              "decision program predicates must remain 16 bytes");

// Contiguous conjunction of contextual features followed by predicates.
typedef struct loom_decision_program_conjunction_t {
  // First scalar predicate ordinal.
  uint32_t first_predicate;

  // First contextual-feature ordinal.
  uint32_t first_feature;

  // Number of scalar predicates.
  uint16_t predicate_count;

  // Number of contextual features.
  uint16_t feature_count;
} loom_decision_program_conjunction_t;

static_assert(sizeof(loom_decision_program_conjunction_t) == 12,
              "decision program conjunctions must remain 12 bytes");

// One ranked alternative and its caller-owned action ordinal.
typedef struct loom_decision_program_choice_t {
  // Constraints that determine whether the choice is feasible.
  loom_decision_program_conjunction_t conjunction;

  // Dense caller-owned action ordinal.
  uint32_t action_ordinal;
} loom_decision_program_choice_t;

static_assert(sizeof(loom_decision_program_choice_t) == 16,
              "decision program choices must remain 16 bytes");

// One equal-priority group in descending priority order.
typedef struct loom_decision_program_priority_group_t {
  // Number of consecutive choices in the group.
  uint32_t choice_count;
} loom_decision_program_priority_group_t;

static_assert(sizeof(loom_decision_program_priority_group_t) == 4,
              "decision program priority groups must remain 4 bytes");

// Immutable ranked alternative program.
typedef struct loom_decision_program_t {
  // Borrowed scalar predicates grouped by conjunction.
  const loom_decision_program_predicate_t* predicates;

  // Borrowed choices grouped by descending priority.
  const loom_decision_program_choice_t* choices;

  // Borrowed equal-priority groups. Choice offsets are implicit prefix sums.
  const loom_decision_program_priority_group_t* priority_groups;

  // Borrowed signed integer constants referenced by predicates.
  const int64_t* constants;

  // Hard requirements applied before ranked alternatives.
  loom_decision_program_conjunction_t hard_requirements;

  // Total scalar predicate count.
  uint32_t predicate_count;

  // Total contextual-feature count interpreted by the adapter.
  uint32_t feature_count;

  // Total constant count.
  uint32_t constant_count;

  // Total choice count.
  uint32_t choice_count;

  // Total equal-priority group count.
  uint32_t priority_group_count;
} loom_decision_program_t;

// SSA fact binding for one program evaluation.
typedef struct loom_decision_program_binding_t {
  // Borrowed O(1) value-fact table for the active function scope.
  const loom_value_fact_table_t* facts;

  // Borrowed application argument value IDs in signature order.
  const loom_value_id_t* argument_values;

  // Borrowed application result value IDs in signature order.
  const loom_value_id_t* result_values;
} loom_decision_program_binding_t;

// Evaluates one adapter-owned contextual feature.
typedef loom_decision_truth_t (*loom_decision_program_evaluate_feature_fn_t)(
    void* user_data, uint32_t feature_ordinal);

// Contextual-feature evaluator for one application site.
typedef struct loom_decision_program_feature_evaluator_t {
  // Infallible feature evaluator. Required when the program has features.
  loom_decision_program_evaluate_feature_fn_t fn;

  // Opaque adapter state passed to |fn|.
  void* user_data;
} loom_decision_program_feature_evaluator_t;

// Refines a scalar predicate left unknown by ordinary value facts.
// Only the operands declared by |predicate_kind| are initialized.
typedef loom_decision_truth_t (*loom_decision_program_refine_predicate_fn_t)(
    void* user_data, uint8_t predicate_kind,
    const loom_decision_predicate_operand_t operands[3]);

// Optional path- or context-sensitive scalar predicate refiner.
typedef struct loom_decision_program_predicate_refiner_t {
  // Infallible refiner invoked only after scalar facts return unknown.
  loom_decision_program_refine_predicate_fn_t fn;

  // Opaque adapter state passed to |fn|.
  void* user_data;
} loom_decision_program_predicate_refiner_t;

// Returns an empty predicate refiner.
static inline loom_decision_program_predicate_refiner_t
loom_decision_program_predicate_refiner_empty(void) {
  return (loom_decision_program_predicate_refiner_t){0};
}

// Ranked handling of unresolved alternatives.
typedef uint8_t loom_decision_program_resolution_policy_t;
enum loom_decision_program_resolution_policy_e {
  // An unresolved alternative at or above the best proven priority defers
  // selection and remains live.
  LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED = 0,

  // Only proven matches participate in selection. Unresolved alternatives are
  // reported if no proven match exists but do not block one.
  LOOM_DECISION_PROGRAM_SELECT_PROVEN = 1,
};

// Ranked program result kind.
typedef uint8_t loom_decision_program_result_kind_t;
enum loom_decision_program_result_kind_e {
  // Every alternative was disproven.
  LOOM_DECISION_PROGRAM_RESULT_NO_MATCH = 0,

  // Exactly one best-priority action was proven.
  LOOM_DECISION_PROGRAM_RESULT_SELECTED = 1,

  // A material alternative remains unresolved.
  LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED = 2,

  // Multiple best-priority actions were proven.
  LOOM_DECISION_PROGRAM_RESULT_AMBIGUOUS = 3,

  // The hard requirement conjunction was disproven.
  LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT = 4,
};

// Compact ranked program result.
typedef struct loom_decision_program_result_t {
  // Result kind from loom_decision_program_result_kind_t.
  loom_decision_program_result_kind_t kind;

  // Reserved bytes. Always zero.
  uint8_t reserved[3];

  // Selected action, or LOOM_DECISION_PROGRAM_ACTION_INVALID.
  uint32_t action_ordinal;

  // Highest-priority unresolved action, or the invalid action.
  uint32_t unresolved_action_ordinal;

  // First material unresolved constraint, or the invalid constraint.
  loom_decision_program_constraint_ref_t unresolved_constraint;
} loom_decision_program_result_t;

static_assert(sizeof(loom_decision_program_result_t) == 16,
              "decision program results must remain 16 bytes");

// Per-choice evidence produced by full evaluation.
typedef struct loom_decision_program_choice_evidence_t {
  // Feasibility encoded as loom_decision_truth_t.
  loom_decision_truth_t feasibility;

  // Reserved bytes. Always zero.
  uint8_t reserved[3];

  // First unresolved constraint, or the invalid constraint.
  loom_decision_program_constraint_ref_t unresolved_constraint;
} loom_decision_program_choice_evidence_t;

static_assert(sizeof(loom_decision_program_choice_evidence_t) == 8,
              "decision choice evidence must remain 8 bytes");

// Evaluates the minimum ranked prefix needed to decide |program|.
//
// When non-NULL, |live_action_ordinals| must have capacity for
// |program->choice_count| entries. The function writes exactly the actions
// that a caller must retain under |resolution_policy| and returns their count
// in |out_live_action_count|. Callers using the select-proven policy may pass
// NULL when they only need |out_result|. |binding| may be NULL only when the
// program contains no scalar predicates; the feature callback may be empty
// only when it contains no features. All program storage and bindings are
// trusted compiler-owned state.
void loom_decision_program_evaluate(
    const loom_decision_program_t* program,
    const loom_decision_program_binding_t* binding,
    loom_decision_program_feature_evaluator_t feature_evaluator,
    loom_decision_program_predicate_refiner_t predicate_refiner,
    loom_decision_program_resolution_policy_t resolution_policy,
    uint32_t* live_action_ordinals, uint32_t* out_live_action_count,
    loom_decision_program_result_t* out_result);

// Evaluates every choice once after proving the hard requirements and reduces
// the captured evidence. When hard requirements remain unresolved, the defer
// policy evaluates choices to compute their exact live frontier; the
// select-proven policy leaves choice evidence untouched. A disproven hard
// requirement also leaves choice evidence untouched.
//
// This entry point supports explicitly requested reports and analysis. The
// caller provides |program->choice_count| evidence and live-action entries.
// Ordinary JIT selection should use loom_decision_program_evaluate so lower
// priority choices remain untouched after a decisive group.
void loom_decision_program_evaluate_all(
    const loom_decision_program_t* program,
    const loom_decision_program_binding_t* binding,
    loom_decision_program_feature_evaluator_t feature_evaluator,
    loom_decision_program_predicate_refiner_t predicate_refiner,
    loom_decision_program_resolution_policy_t resolution_policy,
    loom_decision_program_choice_evidence_t* choice_evidence,
    uint32_t* live_action_ordinals, uint32_t* out_live_action_count,
    loom_decision_program_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_DECISION_PROGRAM_H_
