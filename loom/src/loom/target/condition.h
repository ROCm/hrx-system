// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Typed static target-applicability conditions over immutable target facts.

#ifndef LOOM_TARGET_CONDITION_H_
#define LOOM_TARGET_CONDITION_H_

#include "iree/base/api.h"
#include "loom/ir/parameterized_attr.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_context_t loom_context_t;

typedef uint8_t loom_target_condition_outcome_t;

enum loom_target_condition_outcome_e {
  // No target facts are bound at this application site.
  LOOM_TARGET_CONDITION_UNBOUND = 0,
  // The right fact family is bound but does not establish the queried value.
  LOOM_TARGET_CONDITION_UNKNOWN = 1,
  // Bound facts disprove the condition or have an incompatible family.
  LOOM_TARGET_CONDITION_REJECT = 2,
  // Bound facts prove the condition.
  LOOM_TARGET_CONDITION_MATCH = 3,
};

// Validates family-specific value constraints at the authored IR boundary.
// Generic structural verification has already established the family schema.
typedef iree_status_t (*loom_target_condition_validate_fn_t)(
    loom_attribute_t condition);

// Evaluates one structurally and semantically verified condition.
//
// |facts| is non-null and has the descriptor-required static fact type. The
// callback performs direct typed reads and cannot fail.
typedef loom_target_condition_outcome_t (*loom_target_condition_evaluate_fn_t)(
    const loom_target_facts_t* facts, loom_attribute_t condition);

// Projects one typed static condition into the ordinary predicate it places
// on a matching contextual query result. Returns false when |query_key|
// identifies a different projection within the same family. The authored
// condition and compiler-produced query origin are already verified.
typedef bool (*loom_target_condition_project_query_predicate_fn_t)(
    loom_attribute_t condition, loom_attribute_t query_key,
    loom_value_id_t query_value_id, loom_predicate_t* out_predicate);

// Static semantics attached to one parameterized attribute family.
struct loom_target_condition_descriptor_t {
  // Required static target fact-family type, or NULL for normalized common
  // target facts.
  const loom_target_fact_type_t* required_fact_type;

  // Optional cold authored-value validator.
  loom_target_condition_validate_fn_t validate;

  // Infallible hot evaluator over immutable facts.
  loom_target_condition_evaluate_fn_t evaluate;

  // Optional projection into the shared SSA fact-predicate domain.
  loom_target_condition_project_query_predicate_fn_t project_query_predicate;
};

static_assert(sizeof(loom_target_condition_descriptor_t) == 32,
              "target condition descriptor must remain 32 bytes");

// One semantically resolved authored target condition.
typedef struct loom_target_condition_t {
  // Static family-owned semantics.
  const loom_target_condition_descriptor_t* descriptor;

  // Borrowed parameterized value owned by the source module.
  loom_attribute_t value;
} loom_target_condition_t;

static_assert(sizeof(loom_target_condition_t) == 24,
              "resolved target condition must remain 24 bytes");

// Resolves and validates one structurally verified parameterized attribute as
// a target condition. This is a cold authored-input boundary. On success,
// |out_descriptor| borrows static process-lifetime storage.
iree_status_t loom_target_condition_resolve(
    const loom_context_t* context, loom_attribute_t condition,
    const loom_target_condition_descriptor_t** out_descriptor);

// Evaluates one resolved descriptor/value pair against bound facts.
// Provider summaries preserve this pair so selection performs only this
// static type check and one direct callback.
static inline loom_target_condition_outcome_t loom_target_condition_evaluate(
    const loom_target_condition_descriptor_t* descriptor,
    loom_attribute_t condition, const loom_target_facts_t* facts) {
  if (facts == NULL) return LOOM_TARGET_CONDITION_UNBOUND;
  if (descriptor->required_fact_type != NULL &&
      facts->fact_type != descriptor->required_fact_type) {
    return LOOM_TARGET_CONDITION_REJECT;
  }
  return descriptor->evaluate(facts, condition);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_CONDITION_H_
