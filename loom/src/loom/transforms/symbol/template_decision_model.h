// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable decision programs synthesized from template provider contracts.

#ifndef LOOM_TRANSFORMS_SYMBOL_TEMPLATE_DECISION_MODEL_H_
#define LOOM_TRANSFORMS_SYMBOL_TEMPLATE_DECISION_MODEL_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_references.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/decision/program.h"
#include "loom/transforms/symbol/template_applicability.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Template-specific contextual feature kind.
typedef uint8_t loom_template_decision_feature_kind_t;
enum loom_template_decision_feature_kind_e {
  // Target identity requirement from a family or provider contract.
  LOOM_TEMPLATE_DECISION_FEATURE_TARGET_IDENTITY = 0,

  // One resolved typed target condition.
  LOOM_TEMPLATE_DECISION_FEATURE_TARGET_CONDITION = 1,
};

// One contextual feature referenced by a generic decision program.
typedef struct loom_template_decision_feature_t {
  // Feature kind from loom_template_decision_feature_kind_t.
  loom_template_decision_feature_kind_t kind;

  // Reserved bytes. Always zero.
  uint8_t reserved[7];

  // Kind-specific immutable requirement.
  union {
    // Target identity requirement.
    struct {
      // Module whose symbol domain owns target_symbol.
      const loom_module_t* module;

      // Module-local target witness, or null when only facts are available.
      loom_symbol_ref_t target_symbol;

      // Immutable target identity facts, or NULL when unavailable.
      const loom_target_facts_t* target_facts;
    } target_identity;

    // Borrowed resolved target condition.
    const loom_target_condition_t* target_condition;
  } value;
} loom_template_decision_feature_t;

// Static properties of one synthesized family model.
typedef uint8_t loom_template_decision_model_flags_t;
enum loom_template_decision_model_flag_bits_e {
  // At least one family or provider predicate reads scalar value facts.
  LOOM_TEMPLATE_DECISION_MODEL_FLAG_HAS_SCALAR_PREDICATES = 1u << 0,

  // At least one target condition can project a lexical query predicate.
  LOOM_TEMPLATE_DECISION_MODEL_FLAG_HAS_PROJECTABLE_TARGET_CONDITIONS = 1u << 1,
};

// One demanded template family's immutable ranked decision model.
typedef struct loom_template_decision_model_t {
  // Module whose symbol and value domains own the family contract.
  const loom_module_t* module;

  // Module-local template family symbol.
  loom_symbol_ref_t family;

  // Providers in stable source order. Program actions index this slice.
  loom_template_provider_slice_t providers;

  // Generic hard-requirement and ranked-provider program.
  loom_decision_program_t program;

  // Borrowed contextual features referenced by program conjunctions.
  const loom_template_decision_feature_t* features;

  // Highest provider priority, used only for reports and statistics.
  int64_t highest_provider_priority;

  // Providers with no target identity requirement.
  uint32_t target_independent_provider_count;

  // Static properties from loom_template_decision_model_flags_t.
  loom_template_decision_model_flags_t flags;

  // Reserved bytes. Always zero.
  uint8_t reserved[3];
} loom_template_decision_model_t;

// One high-byte range in a large model catalog's symbol lookup directory.
typedef struct loom_template_decision_model_symbol_page_t {
  // First model ordinal whose family symbol has this high byte.
  uint16_t first_model_ordinal;

  // Number of consecutive models in this page.
  uint16_t model_count;
} loom_template_decision_model_symbol_page_t;
static_assert(sizeof(loom_template_decision_model_symbol_page_t) == 4,
              "template decision symbol pages must remain compact");

// Sorted catalog of models synthesized for demanded provider-bearing families.
typedef struct loom_template_decision_model_catalog_t {
  // Module summarized by this catalog.
  const loom_module_t* module;

  // Models ordered by ascending module-local family symbol ID.
  const loom_template_decision_model_t* models;

  // Optional high-byte directory narrowing lookup to at most 256 models.
  const loom_template_decision_model_symbol_page_t* symbol_pages;

  // Number of synthesized family models.
  iree_host_size_t model_count;

  // Maximum provider count across synthesized families.
  uint32_t maximum_choice_count;
} loom_template_decision_model_catalog_t;

// Application context bound to one model evaluation.
typedef struct loom_template_decision_site_t {
  // template.apply operation being classified.
  const loom_op_t* application_op;

  // Target facts established for the containing function.
  const loom_template_applicability_target_t* application_target;

  // Function and lexical-path facts prepared for this site.
  const loom_template_applicability_facts_t* application_facts;
} loom_template_decision_site_t;

// Application fact inputs required to evaluate one model at one site.
typedef uint8_t loom_template_decision_fact_requirements_t;
enum loom_template_decision_fact_requirement_bits_e {
  // Function-scoped value facts are required.
  LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_VALUES = 1u << 0,

  // Lexical path relations are required in addition to value facts.
  LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_PATH = 1u << 1,
};

// Template-specific meaning of one unresolved generic constraint.
typedef struct loom_template_decision_constraint_info_t {
  // Requirement category used by selection diagnostics.
  loom_template_provider_unresolved_reason_t reason;

  // Unresolved typed target condition, or NULL for other categories.
  const loom_target_condition_t* target_condition;
} loom_template_decision_constraint_info_t;

// Full provider evidence summary used by reports and rejected-site diagnostics.
typedef struct loom_template_decision_evidence_summary_t {
  // Family target identity feasibility from the hard-requirement evaluation.
  loom_template_provider_feasibility_t family_target_identity;

  // Reserved bytes. Always zero.
  uint8_t reserved[3];

  // Providers whose target identity is proven compatible.
  uint32_t target_identity_match_count;

  // Providers whose target identity remains unresolved.
  uint32_t target_identity_unresolved_count;

  // Providers not rejected by their complete applicability contract.
  uint32_t possible_count;

  // Proven providers at the best proven priority.
  uint32_t best_match_count;

  // Highest-priority unresolved provider, or the invalid action ordinal.
  uint32_t highest_unresolved_provider_ordinal;

  // First unresolved constraint for |highest_unresolved_provider_ordinal|.
  loom_decision_program_constraint_ref_t highest_unresolved_constraint;
} loom_template_decision_evidence_summary_t;

// Synthesizes models for provider-bearing families demanded by |references|.
//
// Construction scans only the unique demanded family list and existing
// fact/provider tables. It walks no IR or unrelated module symbols and
// allocates all persistent model storage from |arena| in a bounded number of
// bulk allocations. The result borrows |module|, symbol-fact payloads, and
// provider summaries; their storage must remain live with the model catalog.
// The |references| table itself is not retained.
iree_status_t loom_template_decision_model_catalog_build(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    const loom_symbol_reference_table_t* references,
    const loom_template_provider_catalog_t* providers,
    iree_arena_allocator_t* arena,
    loom_template_decision_model_catalog_t* out_catalog);

// Returns the synthesized model for |family|, or NULL when the family was not
// demanded or has no providers.
const loom_template_decision_model_t* loom_template_decision_model_lookup(
    const loom_template_decision_model_catalog_t* catalog,
    loom_symbol_ref_t family);

// Returns the bounded fact inputs that may be needed to evaluate |model| at
// |demand|. Flat scalar applications request only the existing function value
// table. A nested projectable condition requests lexical facts without
// rescanning model features to ask whether target facts already settle it.
loom_template_decision_fact_requirements_t
loom_template_decision_model_application_fact_requirements(
    const loom_template_decision_model_t* model,
    const loom_template_demand_t* demand);

// Evaluates the minimum ranked provider prefix required to decide one site.
// Target-identity counters in |out_summary| describe only the evaluated prefix;
// callers requiring complete report evidence must use the full evaluator.
void loom_template_decision_model_evaluate(
    const loom_template_decision_model_t* model,
    const loom_template_decision_site_t* site,
    loom_decision_program_resolution_policy_t resolution_policy,
    loom_template_decision_evidence_summary_t* out_summary,
    uint32_t* live_provider_ordinals, uint32_t* out_live_provider_count,
    loom_decision_program_result_t* out_result);

// Evaluates every provider once and reduces the captured evidence.
void loom_template_decision_model_evaluate_all(
    const loom_template_decision_model_t* model,
    const loom_template_decision_site_t* site,
    loom_decision_program_resolution_policy_t resolution_policy,
    loom_decision_program_choice_evidence_t* provider_evidence,
    loom_template_decision_evidence_summary_t* out_summary,
    uint32_t* live_provider_ordinals, uint32_t* out_live_provider_count,
    loom_decision_program_result_t* out_result);

// Returns the source provider selected by |provider_ordinal|.
static inline const loom_template_provider_summary_t*
loom_template_decision_model_provider(
    const loom_template_decision_model_t* model, uint32_t provider_ordinal) {
  return &model->providers.providers[provider_ordinal];
}

// Describes an unresolved constraint emitted by |model|.
loom_template_decision_constraint_info_t
loom_template_decision_model_constraint_info(
    const loom_template_decision_model_t* model,
    loom_decision_program_constraint_ref_t constraint);

// Adds full choice feasibility, best-priority counts, and the highest-priority
// unresolved provider to |inout_summary|.
void loom_template_decision_model_summarize_choice_evidence(
    const loom_template_decision_model_t* model,
    const loom_decision_program_choice_evidence_t* provider_evidence,
    loom_template_decision_evidence_summary_t* inout_summary);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TRANSFORMS_SYMBOL_TEMPLATE_DECISION_MODEL_H_
