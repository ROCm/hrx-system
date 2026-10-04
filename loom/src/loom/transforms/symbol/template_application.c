// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_application.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/template/ops.h"
#include "loom/util/bstring.h"

static iree_string_view_t loom_template_application_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t fallback) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return fallback;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return fallback;
  }
  return loom_string_table_get(&module->strings, symbol->name_id);
}

static bool loom_template_provider_is_materializable(
    const loom_template_provider_summary_t* provider) {
  return provider->kind == LOOM_TEMPLATE_PROVIDER_KIND_DEF &&
         provider->has_body && loom_symbol_ref_is_valid(provider->symbol);
}

static bool loom_template_provider_is_external_materialization(
    const loom_template_provider_summary_t* provider) {
  return provider->kind == LOOM_TEMPLATE_PROVIDER_KIND_DEF &&
         provider->has_body && provider->origin_ordinal != IREE_HOST_SIZE_MAX;
}

static void loom_template_application_initialize(
    const loom_module_t* module, const loom_op_t* op, loom_symbol_ref_t family,
    loom_template_application_result_t* out_result) {
  *out_result = (loom_template_application_result_t){
      .application_op = (loom_op_t*)op,
      .family = family,
      .family_name = loom_template_application_symbol_name(
          module, family, IREE_SV("<invalid>")),
      .exact_provider = loom_symbol_ref_null(),
  };
}

void loom_template_application_select(
    const loom_module_t* module, const loom_template_decision_model_t* model,
    const loom_template_decision_site_t* site,
    loom_decision_program_resolution_policy_t resolution_policy,
    loom_template_application_scratch_t* scratch,
    loom_template_application_result_t* entry) {
  loom_template_application_initialize(
      module, site->application_op,
      loom_template_apply_family(site->application_op), entry);
  scratch->live_provider_count = 0;
  scratch->summary = (loom_template_decision_evidence_summary_t){0};
  if (model == NULL) {
    entry->blocker = LOOM_TEMPLATE_APPLICATION_BLOCKER_NO_PROVIDER;
    return;
  }
  loom_decision_program_result_t result = {0};
  if (scratch->provider_evidence != NULL) {
    loom_template_decision_model_evaluate_all(
        model, site, resolution_policy, scratch->provider_evidence,
        &scratch->summary, scratch->live_provider_ordinals,
        &scratch->live_provider_count, &result);
  } else {
    loom_template_decision_model_evaluate(
        model, site, resolution_policy, &scratch->summary,
        scratch->live_provider_ordinals, &scratch->live_provider_count,
        &result);
  }
  const bool family_unresolved =
      result.kind == LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED &&
      result.unresolved_action_ordinal == LOOM_DECISION_PROGRAM_ACTION_INVALID;
  if (scratch->provider_evidence != NULL &&
      result.kind != LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT &&
      !family_unresolved) {
    loom_template_decision_model_summarize_choice_evidence(
        model, scratch->provider_evidence, &scratch->summary);
  }

  uint32_t unresolved_provider_ordinal = result.unresolved_action_ordinal;
  loom_decision_program_constraint_ref_t unresolved_constraint =
      result.unresolved_constraint;
  if (scratch->provider_evidence != NULL &&
      scratch->summary.highest_unresolved_provider_ordinal !=
          LOOM_DECISION_PROGRAM_ACTION_INVALID) {
    unresolved_provider_ordinal =
        scratch->summary.highest_unresolved_provider_ordinal;
    unresolved_constraint = scratch->summary.highest_unresolved_constraint;
  }
  if (unresolved_provider_ordinal != LOOM_DECISION_PROGRAM_ACTION_INVALID) {
    const loom_template_decision_constraint_info_t constraint_info =
        loom_template_decision_model_constraint_info(model,
                                                     unresolved_constraint);
    entry->unresolved_provider = loom_template_decision_model_provider(
        model, unresolved_provider_ordinal);
    entry->unresolved_reason = constraint_info.reason;
    entry->unresolved_target_condition = constraint_info.target_condition;
    entry->blocker_contract = LOOM_TEMPLATE_CONTRACT_PROVIDER;
  }

  switch ((loom_decision_program_result_kind_t)result.kind) {
    case LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT: {
      entry->blocker_contract = LOOM_TEMPLATE_CONTRACT_FAMILY;
      entry->blocker = scratch->summary.family_target_identity ==
                               LOOM_TEMPLATE_PROVIDER_REJECT
                           ? LOOM_TEMPLATE_APPLICATION_BLOCKER_TARGET_MISMATCH
                           : LOOM_TEMPLATE_APPLICATION_BLOCKER_FAMILY_REJECTED;
      break;
    }
    case LOOM_DECISION_PROGRAM_RESULT_NO_MATCH:
      entry->blocker =
          scratch->summary.target_identity_match_count == 0 &&
                  scratch->summary.target_identity_unresolved_count == 0
              ? LOOM_TEMPLATE_APPLICATION_BLOCKER_TARGET_MISMATCH
              : LOOM_TEMPLATE_APPLICATION_BLOCKER_ALL_REJECTED;
      break;
    case LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED: {
      if (family_unresolved) {
        const loom_template_decision_constraint_info_t constraint_info =
            loom_template_decision_model_constraint_info(
                model, result.unresolved_constraint);
        entry->unresolved_reason = constraint_info.reason;
        entry->unresolved_target_condition = constraint_info.target_condition;
        entry->blocker_contract = LOOM_TEMPLATE_CONTRACT_FAMILY;
      }
      entry->blocker = LOOM_TEMPLATE_APPLICATION_BLOCKER_MISSING_FACTS;
      break;
    }
    case LOOM_DECISION_PROGRAM_RESULT_AMBIGUOUS:
      entry->blocker = LOOM_TEMPLATE_APPLICATION_BLOCKER_AMBIGUOUS;
      break;
    case LOOM_DECISION_PROGRAM_RESULT_SELECTED: {
      const loom_template_provider_summary_t* selected_provider =
          loom_template_decision_model_provider(model, result.action_ordinal);
      entry->selected_provider = selected_provider;
      if (!loom_template_provider_is_materializable(selected_provider) &&
          !loom_template_provider_is_external_materialization(
              selected_provider)) {
        entry->blocker_contract = LOOM_TEMPLATE_CONTRACT_PROVIDER;
        entry->blocker = LOOM_TEMPLATE_APPLICATION_BLOCKER_MATERIALIZATION;
        break;
      }

      entry->blocker = LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE;
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("invalid template decision result kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_template_application_lookup_target_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    loom_symbol_ref_t target_ref, const loom_target_facts_t** out_target) {
  *out_target = NULL;
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return iree_ok_status();
  }
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      symbol_facts, module, target_ref, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  if (target_facts == NULL) {
    return iree_ok_status();
  }
  *out_target = target_facts->projection;
  return iree_ok_status();
}

static iree_status_t loom_template_application_load_contract_from_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    const loom_func_symbol_facts_t* facts,
    loom_template_applicability_contract_t* out_contract) {
  *out_contract = (loom_template_applicability_contract_t){
      .module = module,
      .target_symbol = facts->target_symbol,
      .argument_ids = facts->argument_ids,
      .result_ids = facts->result_ids,
      .predicates = facts->predicates,
      .target_conditions = facts->target_conditions,
      .argument_count = facts->argument_count,
      .result_count = facts->result_count,
      .predicate_count = facts->predicate_count,
      .target_condition_count = facts->target_condition_count,
  };
  return loom_template_application_lookup_target_facts(
      module, symbol_facts, facts->target_symbol, &out_contract->target_facts);
}

static iree_status_t loom_template_application_lookup_function_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    loom_symbol_ref_t symbol_ref, const loom_func_symbol_facts_t** out_facts) {
  *out_facts = NULL;
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      symbol_facts, module, symbol_ref, &base_facts));
  const loom_func_symbol_facts_t* facts =
      loom_func_symbol_facts_cast(base_facts);
  *out_facts = facts;
  return iree_ok_status();
}

iree_status_t loom_template_application_load_call(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_fact_table_t* symbol_facts,
    loom_template_application_call_t* out_call) {
  IREE_RETURN_IF_ERROR(loom_template_application_lookup_function_facts(
      module, symbol_facts, loom_template_call_callee(call_op),
      &out_call->provider_facts));
  IREE_RETURN_IF_ERROR(loom_template_application_load_contract_from_facts(
      module, symbol_facts, out_call->provider_facts,
      &out_call->provider_contract));
  const loom_func_symbol_facts_t* family_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_template_application_lookup_function_facts(
      module, symbol_facts, out_call->provider_facts->template_family,
      &family_facts));
  return loom_template_application_load_contract_from_facts(
      module, symbol_facts, family_facts, &out_call->family_contract);
}

void loom_template_application_check_call(
    const loom_module_t* module, const loom_template_application_call_t* call,
    const loom_template_decision_site_t* site,
    loom_template_provider_classification_t* out_provider_classification,
    loom_template_application_result_t* entry) {
  loom_template_application_initialize(module, site->application_op,
                                       call->provider_facts->template_family,
                                       entry);
  entry->exact_provider = loom_template_call_callee(site->application_op);
  loom_template_provider_classification_t family_classification = {0};
  *out_provider_classification = (loom_template_provider_classification_t){0};
  loom_template_applicability_classify_contract(
      module, site->application_op, &call->family_contract,
      site->application_target, site->application_facts,
      &family_classification);
  if (family_classification.feasibility == LOOM_TEMPLATE_PROVIDER_MATCH) {
    loom_template_applicability_classify_contract(
        module, site->application_op, &call->provider_contract,
        site->application_target, site->application_facts,
        out_provider_classification);
  }
  if (family_classification.feasibility != LOOM_TEMPLATE_PROVIDER_MATCH) {
    entry->blocker_contract = LOOM_TEMPLATE_CONTRACT_FAMILY;
  } else if (out_provider_classification->feasibility !=
             LOOM_TEMPLATE_PROVIDER_MATCH) {
    entry->blocker_contract = LOOM_TEMPLATE_CONTRACT_PROVIDER;
  } else {
    return;
  }
  const loom_template_provider_classification_t* blocker =
      entry->blocker_contract == LOOM_TEMPLATE_CONTRACT_FAMILY
          ? &family_classification
          : out_provider_classification;
  entry->unresolved_reason = blocker->unresolved_reason;
  entry->unresolved_target_condition = blocker->unresolved_target_condition;
  entry->blocker = blocker->feasibility == LOOM_TEMPLATE_PROVIDER_REJECT
                       ? LOOM_TEMPLATE_APPLICATION_BLOCKER_EXACT_CALL_REJECTED
                       : LOOM_TEMPLATE_APPLICATION_BLOCKER_MISSING_FACTS;
}

iree_string_view_t loom_template_application_condition_name(
    const loom_module_t* module, const loom_target_condition_t* condition) {
  const loom_parameterized_attr_kind_t family_kind =
      loom_attr_as_parameterized_kind(condition->value);
  const loom_parameterized_attr_descriptor_t* family =
      loom_context_resolve_parameterized_attr(module->context, family_kind);
  if (family == NULL) {
    IREE_ASSERT_UNREACHABLE(
        "resolved target condition family disappeared from its context");
    IREE_BUILTIN_UNREACHABLE();
  }
  return loom_bstring_view(family->name);
}

iree_string_view_t loom_template_application_blocker_code(
    loom_template_application_blocker_t blocker) {
  switch (blocker) {
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_NO_PROVIDER:
      return IREE_SV("no_provider");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_TARGET_MISMATCH:
      return IREE_SV("target_mismatch");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_ALL_REJECTED:
      return IREE_SV("all_rejected");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_FAMILY_REJECTED:
      return IREE_SV("family_rejected");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_EXACT_CALL_REJECTED:
      return IREE_SV("exact_call_rejected");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_MISSING_FACTS:
      return IREE_SV("missing_facts");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_AMBIGUOUS:
      return IREE_SV("ambiguous");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_MATERIALIZATION:
      return IREE_SV("materialization_blocked");
    case LOOM_TEMPLATE_APPLICATION_BLOCKER_NONE:
    default:
      return IREE_SV("unresolved");
  }
}

static loom_func_like_t loom_template_application_blocker_contract_function(
    const loom_module_t* module,
    const loom_template_application_result_t* entry) {
  if (entry->blocker_contract == LOOM_TEMPLATE_CONTRACT_FAMILY) {
    const loom_symbol_t* family_symbol =
        &module->symbols.entries[entry->family.symbol_id];
    return loom_func_like_cast(module, family_symbol->defining_op);
  }
  if (entry->blocker_contract != LOOM_TEMPLATE_CONTRACT_PROVIDER) {
    return (loom_func_like_t){0};
  }
  if (loom_symbol_ref_is_valid(entry->exact_provider)) {
    const loom_symbol_t* provider_symbol =
        &module->symbols.entries[entry->exact_provider.symbol_id];
    return loom_func_like_cast(module, provider_symbol->defining_op);
  }
  const loom_template_provider_summary_t* provider =
      entry->unresolved_provider ? entry->unresolved_provider
                                 : entry->selected_provider;
  return provider ? provider->function : (loom_func_like_t){0};
}

static iree_string_view_t loom_template_application_blocker_contract_name(
    const loom_module_t* module,
    const loom_template_application_result_t* entry) {
  if (entry->blocker_contract == LOOM_TEMPLATE_CONTRACT_FAMILY) {
    return entry->family_name;
  }
  if (loom_symbol_ref_is_valid(entry->exact_provider)) {
    return loom_template_application_symbol_name(module, entry->exact_provider,
                                                 IREE_SV("<invalid>"));
  }
  const loom_template_provider_summary_t* provider =
      entry->unresolved_provider ? entry->unresolved_provider
                                 : entry->selected_provider;
  return provider ? provider->name : IREE_SV("<unknown>");
}

iree_status_t loom_template_application_emit_blocker(
    const loom_module_t* module,
    const loom_template_application_result_t* entry,
    iree_string_view_t phase_name, iree_diagnostic_emitter_t emitter) {
  if (entry->blocker == LOOM_TEMPLATE_APPLICATION_BLOCKER_MISSING_FACTS &&
      entry->unresolved_target_condition != NULL) {
    const loom_func_like_t unresolved_contract =
        loom_template_application_blocker_contract_function(module, entry);
    const iree_string_view_t unresolved_contract_name =
        loom_template_application_blocker_contract_name(module, entry);
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(module, entry->application_op)),
        loom_param_string(phase_name),
        loom_param_string(entry->family_name),
        loom_param_string(unresolved_contract_name),
        loom_param_string(loom_template_application_condition_name(
            module, entry->unresolved_target_condition)),
    };
    loom_diagnostic_related_op_t related_op = {
        .label = entry->blocker_contract == LOOM_TEMPLATE_CONTRACT_FAMILY
                     ? IREE_SV("unresolved family condition")
                     : IREE_SV("unresolved provider condition"),
        .module = module,
        .op = loom_func_like_isa(unresolved_contract) ? unresolved_contract.op
                                                      : NULL,
        .field_ref = loom_func_like_isa(unresolved_contract)
                         ? loom_diagnostic_field_ref(
                               LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                               unresolved_contract.vtable->requires_attr_index)
                         : loom_diagnostic_field_ref_none(),
    };
    loom_diagnostic_emission_t emission = {
        .module = module,
        .op = entry->application_op,
        .error = LOOM_ERR_LOWERING_048,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
        .related_ops = related_op.op ? &related_op : NULL,
        .related_op_count = related_op.op ? 1 : 0,
    };
    return iree_diagnostic_emit(emitter, &emission);
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, entry->application_op)),
      loom_param_string(phase_name),
      loom_param_string(entry->family_name),
      loom_param_string(loom_template_application_blocker_code(entry->blocker)),
  };
  const loom_func_like_t blocker_contract =
      loom_template_application_blocker_contract_function(module, entry);
  loom_diagnostic_related_op_t related_op = {
      .label = entry->blocker_contract == LOOM_TEMPLATE_CONTRACT_FAMILY
                   ? IREE_SV("family contract")
                   : IREE_SV("provider contract"),
      .module = module,
      .op = loom_func_like_isa(blocker_contract) ? blocker_contract.op : NULL,
      .field_ref = loom_diagnostic_field_ref_none(),
  };
  loom_diagnostic_emission_t emission = {
      .module = module,
      .op = entry->application_op,
      .error = LOOM_ERR_LOWERING_045,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_op.op ? &related_op : NULL,
      .related_op_count = related_op.op ? 1 : 0,
  };
  return iree_diagnostic_emit(emitter, &emission);
}
