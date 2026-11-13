// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Template application decisions with caller-owned target and value facts.
// Inputs are verified IR and producer-owned model/fact tables. This interface
// consumes their established symbol and signature invariants directly.

#ifndef LOOM_TRANSFORMS_SYMBOL_TEMPLATE_APPLICATION_H_
#define LOOM_TRANSFORMS_SYMBOL_TEMPLATE_APPLICATION_H_

#include "loom/error/emitter.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/transforms/symbol/template_decision_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Semantic reason preventing expansion at one application site.
typedef enum loom_template_application_blocker_e {
  LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE = 0,
  LOOM_TEMPLATE_APPLICATION_BLOCKER_NO_PROVIDER = 1,
  LOOM_TEMPLATE_APPLICATION_BLOCKER_TARGET_MISMATCH = 2,
  LOOM_TEMPLATE_APPLICATION_BLOCKER_ALL_REJECTED = 3,
  LOOM_TEMPLATE_APPLICATION_BLOCKER_MISSING_FACTS = 4,
  LOOM_TEMPLATE_APPLICATION_BLOCKER_AMBIGUOUS = 5,
  LOOM_TEMPLATE_APPLICATION_BLOCKER_MATERIALIZATION = 6,
  LOOM_TEMPLATE_APPLICATION_BLOCKER_FAMILY_REJECTED = 7,
  LOOM_TEMPLATE_APPLICATION_BLOCKER_EXACT_CALL_REJECTED = 8,
} loom_template_application_blocker_t;

// Contract responsible for an application's diagnostic.
typedef enum loom_template_contract_role_e {
  LOOM_TEMPLATE_CONTRACT_NONE = 0,
  LOOM_TEMPLATE_CONTRACT_FAMILY = 1,
  LOOM_TEMPLATE_CONTRACT_PROVIDER = 2,
} loom_template_contract_role_t;

// Allocation-free decision borrowing the site's module and provider catalog.
// A successful apply identifies its selected provider; a successful exact call
// retains its explicit callee. Blockers describe semantic outcomes, not
// statuses.
typedef struct loom_template_application_result_t {
  // Live template.apply or authored template.call operation.
  loom_op_t* application_op;
  // Template family demanded by the application.
  loom_symbol_ref_t family;
  // Borrowed template family symbol name.
  iree_string_view_t family_name;
  // Selected provider for an apply, including when materialization is blocked.
  const loom_template_provider_summary_t* selected_provider;
  // Direct provider referenced by an authored template.call, or null.
  loom_symbol_ref_t exact_provider;
  // Highest-priority provider whose applicability remains unproven, or NULL
  // when the family declaration contract itself remains unproven.
  const loom_template_provider_summary_t* unresolved_provider;
  // First unresolved target condition on the family or provider contract.
  const loom_target_condition_t* unresolved_target_condition;
  // NONE on success; otherwise the reason selection or an exact call is
  // blocked.
  loom_template_application_blocker_t blocker;
  // First unresolved family or provider requirement category.
  loom_template_provider_unresolved_reason_t unresolved_reason;
  // Family or provider contract responsible for the blocker.
  loom_template_contract_role_t blocker_contract;
} loom_template_application_result_t;

// Reusable evaluation storage, sized once to the catalog's maximum choice
// count. Selection overwrites the summary and live-provider count each time.
typedef struct loom_template_application_scratch_t {
  // Caller-owned provider ordinals, with capacity for the largest family.
  uint32_t* live_provider_ordinals;
  // Optional full evidence storage of the same capacity, for detailed reports.
  // NULL selects the minimal-prefix evaluator.
  loom_decision_program_choice_evidence_t* provider_evidence;
  // Number of live provider ordinals written by the last selection.
  uint32_t live_provider_count;
  // Evidence for the last selection; complete only with provider_evidence.
  loom_template_decision_evidence_summary_t summary;
} loom_template_application_scratch_t;

// Borrowed family and provider requirements for one authored exact call.
// The module's symbol-fact table owns all referenced contract storage.
typedef struct loom_template_application_call_t {
  // Provider facts, including the owning family and priority.
  const loom_func_symbol_facts_t* provider_facts;
  // Hard family requirements, checked before provider requirements.
  loom_template_applicability_contract_t family_contract;
  // Requirements of the explicitly selected provider.
  loom_template_applicability_contract_t provider_contract;
} loom_template_application_call_t;

// Selects one template.apply using an immutable model and explicit site facts.
// A NULL model means the family has no available providers. The result borrows
// the site, module and model storage. Selection does not allocate, walk IR,
// acquire facts, mark symbols live or mutate the module. The caller owns those
// lifecycles and consumes scratch's live ordinals before its next evaluation.
void loom_template_application_select(
    const loom_module_t* module, const loom_template_decision_model_t* model,
    const loom_template_decision_site_t* site,
    loom_decision_program_resolution_policy_t resolution_policy,
    loom_template_application_scratch_t* scratch,
    loom_template_application_result_t* out_result);

// Loads an exact call's declared contracts through the shared symbol-fact
// table. The table computes each symbol once; allocation failures propagate.
iree_status_t loom_template_application_load_call(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_fact_table_t* symbol_facts,
    loom_template_application_call_t* out_call);

// Checks an authored template.call against both contracts using explicit site
// facts. Exact calls bypass ranking, not applicability. A successful check has
// blocker NONE and requires no rewrite. Provider evidence is returned
// separately for reports; it remains zero when the family contract prevents
// evaluation.
void loom_template_application_check_call(
    const loom_module_t* module, const loom_template_application_call_t* call,
    const loom_template_decision_site_t* site,
    loom_template_application_result_t* out_result,
    loom_template_provider_classification_t* out_provider_classification);

// Returns the stable diagnostic code for a blocked application.
iree_string_view_t loom_template_application_blocker_code(
    loom_template_application_blocker_t blocker);

// Returns the authored name of a resolved target condition.
iree_string_view_t loom_template_application_condition_name(
    const loom_module_t* module, const loom_target_condition_t* condition);

// Emits a diagnostic for one blocked application at a closed specialization
// boundary. The caller supplies the compilation stage name and emitter; no pass
// object, liveness result or module-wide selection state is required.
iree_status_t loom_template_application_emit_blocker(
    const loom_module_t* module,
    const loom_template_application_result_t* result,
    iree_string_view_t phase_name, iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_TEMPLATE_APPLICATION_H_
