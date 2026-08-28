// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_TEMPLATE_APPLICABILITY_H_
#define LOOM_TRANSFORMS_SYMBOL_TEMPLATE_APPLICABILITY_H_

#include "iree/base/api.h"
#include "loom/analysis/condition_facts.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/decision/predicate.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/target/condition.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Applicability of a template family or provider to one application.
typedef enum loom_template_provider_feasibility_e {
  LOOM_TEMPLATE_PROVIDER_REJECT = 0,
  LOOM_TEMPLATE_PROVIDER_MAYBE = 1,
  LOOM_TEMPLATE_PROVIDER_MATCH = 2,
} loom_template_provider_feasibility_t;

// Requirement category preventing a definitive applicability result.
typedef enum loom_template_provider_unresolved_reason_e {
  LOOM_TEMPLATE_PROVIDER_UNRESOLVED_NONE = 0,
  LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_IDENTITY = 1,
  LOOM_TEMPLATE_PROVIDER_UNRESOLVED_TARGET_CONDITION = 2,
  LOOM_TEMPLATE_PROVIDER_UNRESOLVED_VALUE_PREDICATE = 3,
} loom_template_provider_unresolved_reason_t;

// Detailed applicability result used by selection and diagnostics.
typedef struct loom_template_provider_classification_t {
  // Combined signature, target, condition, and predicate outcome.
  loom_template_provider_feasibility_t feasibility;

  // Target-identity outcome before other requirements are applied.
  loom_template_provider_feasibility_t target_feasibility;

  // First unresolved requirement category in deterministic evaluation order.
  loom_template_provider_unresolved_reason_t unresolved_reason;

  // First unresolved typed target condition, or NULL for other categories.
  const loom_target_condition_t* unresolved_target_condition;
} loom_template_provider_classification_t;

// Target facts established for the function containing an application.
typedef struct loom_template_applicability_target_t {
  // Authored target witness retained for diagnostics and reports.
  loom_symbol_ref_t witness;

  // Function target facts used for provider compatibility.
  const loom_target_facts_t* facts;
} loom_template_applicability_target_t;

// Value and path facts established at one application.
typedef struct loom_template_applicability_facts_t {
  // Function-scoped SSA facts projected with the function target facts.
  const loom_value_fact_table_t* values;

  // Facts established only along the lexical path to this application.
  loom_condition_fact_set_t path;
} loom_template_applicability_facts_t;

// Allocation-free applicability fields shared by families and providers.
typedef struct loom_template_applicability_contract_t {
  // Module whose symbols and values own this contract.
  const loom_module_t* module;

  // Module-local target requirement, or null when target-independent.
  loom_symbol_ref_t target_symbol;

  // Immutable target identity requirement, or NULL when unresolved.
  const loom_target_facts_t* target_facts;

  // Borrowed argument value IDs in signature order.
  const loom_value_id_t* argument_ids;

  // Borrowed result value IDs in signature order.
  const loom_value_id_t* result_ids;

  // Borrowed value predicates.
  const loom_predicate_t* predicates;

  // Borrowed resolved target-condition conjunction.
  const loom_target_condition_t* target_conditions;

  // Number of argument value IDs.
  uint16_t argument_count;

  // Number of result value IDs.
  uint16_t result_count;

  // Number of value predicates.
  uint16_t predicate_count;

  // Number of resolved target conditions.
  uint16_t target_condition_count;
} loom_template_applicability_contract_t;

// Returns an allocation-free applicability contract over |provider|.
loom_template_applicability_contract_t
loom_template_applicability_provider_contract(
    const loom_template_provider_summary_t* provider);

// Evaluates one target identity requirement against |application_target|.
//
// |requirement_module| identifies the symbol domain of |target_symbol|. The
// immutable target facts may satisfy requirements across module boundaries;
// exact symbol identity applies only within one module.
loom_template_provider_feasibility_t
loom_template_applicability_evaluate_target_requirement(
    const loom_module_t* application_module,
    const loom_module_t* requirement_module, loom_symbol_ref_t target_symbol,
    const loom_target_facts_t* target_facts,
    const loom_template_applicability_target_t* application_target);

// Evaluates one resolved target condition at an application site.
loom_template_provider_feasibility_t
loom_template_applicability_evaluate_target_condition(
    const loom_module_t* application_module,
    const loom_target_condition_t* condition,
    const loom_target_facts_t* application_target_facts,
    const loom_template_applicability_facts_t* application_facts);

// Refines an otherwise unknown scalar predicate with lexical path facts.
//
// The supplied operands contain the ordinary function-scoped facts and SSA
// identities. This applies edge-local range refinements and exact symbolic
// relations before returning the final ternary result.
loom_decision_truth_t loom_template_applicability_refine_predicate(
    const loom_template_applicability_facts_t* application_facts,
    uint8_t predicate_kind,
    const loom_decision_predicate_operand_t operands[3]);

// Classifies one family or exact-provider constraint set at |application_op|.
//
// The caller supplies target and value facts from the application module.
void loom_template_applicability_classify_contract(
    const loom_module_t* application_module, const loom_op_t* application_op,
    const loom_template_applicability_contract_t* contract,
    const loom_template_applicability_target_t* application_target,
    const loom_template_applicability_facts_t* application_facts,
    loom_template_provider_classification_t* out_classification);

// Classifies one provider at |application_op|.
//
// Function-contract verification must have established that the provider and
// application signatures both match the provider's template family. Selection
// consumes that trusted contract and does not repeat type comparison at every
// application site.
void loom_template_applicability_classify_provider(
    const loom_module_t* application_module, const loom_op_t* application_op,
    const loom_template_provider_summary_t* provider,
    const loom_template_applicability_target_t* application_target,
    const loom_template_applicability_facts_t* application_facts,
    loom_template_provider_classification_t* out_classification);

// Returns true when path-sensitive value facts can refine |contract|.
bool loom_template_applicability_requires_application_facts(
    const loom_op_t* application_op,
    const loom_template_applicability_contract_t* contract,
    const loom_target_facts_t* application_target_facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_TEMPLATE_APPLICABILITY_H_
