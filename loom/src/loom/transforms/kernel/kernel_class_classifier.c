// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_class_classifier.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/template/ops.h"
#include "loom/util/adaptive_sort.h"

//===----------------------------------------------------------------------===//
// Construction utilities
//===----------------------------------------------------------------------===//

static bool loom_kernel_class_value_id_less(const loom_value_id_t* lhs,
                                            const loom_value_id_t* rhs) {
  return *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_kernel_class_sort_value_ids, loom_value_id_t,
                          loom_kernel_class_value_id_less)

static bool loom_kernel_class_projection_ordinal_less(const uint32_t* lhs,
                                                      const uint32_t* rhs) {
  return *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_kernel_class_sort_projection_ordinals, uint32_t,
                          loom_kernel_class_projection_ordinal_less)

typedef struct loom_kernel_class_argument_map_entry_t {
  // Kernel block argument value ID.
  loom_value_id_t value_id;

  // Kernel ABI argument ordinal.
  uint16_t argument_ordinal;

  // Reserved bytes. Always zero.
  uint8_t reserved[2];
} loom_kernel_class_argument_map_entry_t;

static bool loom_kernel_class_argument_map_entry_less(
    const loom_kernel_class_argument_map_entry_t* lhs,
    const loom_kernel_class_argument_map_entry_t* rhs) {
  return lhs->value_id < rhs->value_id;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_kernel_class_sort_argument_map,
                          loom_kernel_class_argument_map_entry_t,
                          loom_kernel_class_argument_map_entry_less)

typedef struct loom_kernel_class_argument_lookup_t {
  // Kernel argument value IDs in ABI order.
  const loom_value_id_t* argument_ids;

  // Optional sorted map for non-monotonic argument IDs.
  const loom_kernel_class_argument_map_entry_t* sorted_entries;

  // Number of kernel arguments.
  uint16_t argument_count;

  // True when argument_ids are strictly increasing.
  bool arguments_are_ordered;
} loom_kernel_class_argument_lookup_t;

static iree_status_t loom_kernel_class_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_values) {
  *out_values = NULL;
  if (count == 0) return iree_ok_status();
  return iree_arena_allocate_array(arena, count, element_size, out_values);
}

static bool loom_kernel_class_value_ids_are_ordered(
    const loom_value_id_t* value_ids, uint16_t count) {
  for (uint16_t i = 1; i < count; ++i) {
    if (value_ids[i - 1] >= value_ids[i]) return false;
  }
  return true;
}

static iree_status_t loom_kernel_class_prepare_argument_lookup(
    const loom_value_id_t* argument_ids, uint16_t argument_count,
    iree_arena_allocator_t* arena,
    loom_kernel_class_argument_lookup_t* out_lookup) {
  *out_lookup = (loom_kernel_class_argument_lookup_t){
      .argument_ids = argument_ids,
      .argument_count = argument_count,
      .arguments_are_ordered =
          loom_kernel_class_value_ids_are_ordered(argument_ids, argument_count),
  };
  if (out_lookup->arguments_are_ordered || argument_count <= 8) {
    return iree_ok_status();
  }
  loom_kernel_class_argument_map_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, argument_count, sizeof(*entries), (void**)&entries));
  for (uint16_t i = 0; i < argument_count; ++i) {
    entries[i] = (loom_kernel_class_argument_map_entry_t){
        .value_id = argument_ids[i],
        .argument_ordinal = i,
    };
  }
  loom_kernel_class_sort_argument_map(entries, argument_count);
  out_lookup->sorted_entries = entries;
  return iree_ok_status();
}

static bool loom_kernel_class_lookup_argument(
    const loom_kernel_class_argument_lookup_t* lookup, loom_value_id_t value_id,
    uint16_t* out_argument_ordinal) {
  if (!lookup->arguments_are_ordered && lookup->sorted_entries == NULL) {
    for (uint16_t i = 0; i < lookup->argument_count; ++i) {
      if (lookup->argument_ids[i] == value_id) {
        *out_argument_ordinal = i;
        return true;
      }
    }
    return false;
  }

  uint16_t begin = 0;
  uint16_t end = lookup->argument_count;
  while (begin < end) {
    const uint16_t middle = begin + (uint16_t)((end - begin) / 2);
    const loom_value_id_t middle_value =
        lookup->arguments_are_ordered ? lookup->argument_ids[middle]
                                      : lookup->sorted_entries[middle].value_id;
    if (middle_value < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (begin == lookup->argument_count) return false;
  if (lookup->arguments_are_ordered) {
    if (lookup->argument_ids[begin] != value_id) return false;
    *out_argument_ordinal = begin;
  } else {
    if (lookup->sorted_entries[begin].value_id != value_id) return false;
    *out_argument_ordinal = lookup->sorted_entries[begin].argument_ordinal;
  }
  return true;
}

static const loom_kernel_class_projection_t*
loom_kernel_class_lookup_projection(
    const loom_kernel_class_projection_t* projections,
    uint32_t projection_count, loom_value_id_t source_value_id,
    uint32_t* out_projection_ordinal) {
  uint32_t begin = 0;
  uint32_t end = projection_count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    if (projections[middle].source_value_id < source_value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  IREE_ASSERT(begin < projection_count);
  IREE_ASSERT(projections[begin].source_value_id == source_value_id);
  *out_projection_ordinal = begin;
  return &projections[begin];
}

static loom_value_id_t loom_kernel_class_decision_source_value(
    const loom_template_demand_t* demand,
    loom_decision_program_operand_ref_t operand_ref) {
  const uint32_t ordinal = loom_decision_program_operand_ordinal(operand_ref);
  const loom_value_slice_t values =
      loom_decision_program_operand_is_result(operand_ref)
          ? loom_template_apply_results(demand->apply_op)
          : loom_template_apply_operands(demand->apply_op);
  IREE_ASSERT(ordinal < values.count);
  return values.values[ordinal];
}

static iree_host_size_t loom_kernel_class_decision_value_reference_count(
    const loom_template_decision_model_t* model) {
  iree_host_size_t count = 0;
  for (uint32_t i = 0; i < model->program.predicate_count; ++i) {
    const loom_decision_program_predicate_t* predicate =
        &model->program.predicates[i];
    for (uint8_t j = 0; j < predicate->operand_count; ++j) {
      count +=
          !loom_decision_program_operand_is_constant(predicate->operands[j]);
    }
  }
  return count;
}

static loom_decision_truth_t loom_kernel_class_truth_from_feasibility(
    loom_template_provider_feasibility_t feasibility) {
  switch (feasibility) {
    case LOOM_TEMPLATE_PROVIDER_REJECT:
      return LOOM_DECISION_TRUTH_FALSE;
    case LOOM_TEMPLATE_PROVIDER_MAYBE:
      return LOOM_DECISION_TRUTH_UNKNOWN;
    case LOOM_TEMPLATE_PROVIDER_MATCH:
      return LOOM_DECISION_TRUTH_TRUE;
  }
  IREE_ASSERT_UNREACHABLE("invalid template provider feasibility");
  IREE_BUILTIN_UNREACHABLE();
}

static loom_decision_truth_t loom_kernel_class_resolve_feature(
    const loom_template_decision_model_t* model, uint32_t feature_ordinal,
    const loom_template_applicability_target_t* kernel_target,
    const loom_template_applicability_facts_t* kernel_application_facts) {
  const loom_template_decision_feature_t* feature =
      &model->features[feature_ordinal];
  switch ((loom_template_decision_feature_kind_t)feature->kind) {
    case LOOM_TEMPLATE_DECISION_FEATURE_TARGET_IDENTITY:
      return loom_kernel_class_truth_from_feasibility(
          loom_template_applicability_evaluate_target_requirement(
              model->module, feature->value.target_identity.module,
              feature->value.target_identity.target_symbol,
              feature->value.target_identity.target_facts, kernel_target));
    case LOOM_TEMPLATE_DECISION_FEATURE_TARGET_CONDITION:
      return loom_kernel_class_truth_from_feasibility(
          loom_template_applicability_evaluate_target_condition(
              model->module, feature->value.target_condition,
              kernel_target->facts, kernel_application_facts));
  }
  IREE_ASSERT_UNREACHABLE("invalid template decision feature kind");
  IREE_BUILTIN_UNREACHABLE();
}

static loom_decision_truth_t loom_kernel_class_evaluate_feature(
    void* user_data, uint32_t feature_ordinal) {
  const loom_decision_truth_t* outcomes =
      (const loom_decision_truth_t*)user_data;
  return outcomes[feature_ordinal];
}

static loom_decision_program_feature_evaluator_t
loom_kernel_class_feature_evaluator(
    const loom_kernel_class_decision_t* decision) {
  if (decision->model->program.feature_count == 0) {
    return (loom_decision_program_feature_evaluator_t){0};
  }
  return (loom_decision_program_feature_evaluator_t){
      .fn = loom_kernel_class_evaluate_feature,
      .user_data = (void*)decision->feature_outcomes,
  };
}

static loom_kernel_class_contract_flags_t loom_kernel_class_contract_flags(
    const loom_decision_program_t* program,
    loom_decision_program_conjunction_t conjunction) {
  loom_kernel_class_contract_flags_t flags = 0;
  const uint32_t end_predicate =
      conjunction.first_predicate + conjunction.predicate_count;
  for (uint32_t i = conjunction.first_predicate; i < end_predicate; ++i) {
    const loom_decision_program_predicate_t* predicate =
        &program->predicates[i];
    for (uint8_t j = 0; j < predicate->operand_count; ++j) {
      if (loom_decision_program_operand_is_result(predicate->operands[j])) {
        flags |= LOOM_KERNEL_CLASS_CONTRACT_FLAG_RESULT_DEPENDENT;
      }
    }
  }
  return flags;
}

//===----------------------------------------------------------------------===//
// Classifier construction
//===----------------------------------------------------------------------===//

iree_status_t loom_kernel_class_classifier_build(
    const loom_module_t* module, loom_symbol_id_t kernel_symbol_id,
    const loom_symbol_reference_table_t* references,
    const loom_template_decision_model_catalog_t* decision_models,
    const loom_value_fact_table_t* kernel_facts,
    loom_symbolic_expr_context_t* expression_context,
    const loom_template_applicability_target_t* kernel_target,
    iree_arena_allocator_t* arena,
    loom_kernel_class_classifier_t* out_classifier) {
  *out_classifier = (loom_kernel_class_classifier_t){
      .module = module,
      .kernel_symbol_id = kernel_symbol_id,
  };
  IREE_ASSERT(references->module == module);
  IREE_ASSERT(decision_models->module == module);
  IREE_ASSERT(expression_context->module == module);
  IREE_ASSERT(expression_context->fact_table == kernel_facts);

  const loom_symbol_t* kernel_symbol =
      &module->symbols.entries[kernel_symbol_id];
  const loom_func_like_t kernel =
      loom_func_like_cast(module, kernel_symbol->defining_op);
  IREE_ASSERT(loom_func_like_isa(kernel));
  const uint8_t body_region_index = loom_func_like_body_region_index(kernel);
  uint16_t kernel_argument_count = 0;
  const loom_value_id_t* kernel_argument_ids =
      loom_func_like_arg_ids(kernel, &kernel_argument_count);
  out_classifier->kernel_argument_count = kernel_argument_count;

  uint32_t decision_count = 0;
  loom_template_demand_id_t demand_id =
      references->symbols[kernel_symbol_id].first_template_demand_id;
  while (demand_id != LOOM_TEMPLATE_DEMAND_ID_INVALID) {
    const loom_template_demand_t* demand =
        &references->template_demands.values[demand_id];
    if (demand->source_root_region_index_plus_one == body_region_index + 1) {
      ++decision_count;
    }
    demand_id = demand->next_source_demand_id;
  }
  out_classifier->decision_count = decision_count;
  if (decision_count == 0) return iree_ok_status();

  loom_kernel_class_decision_t* decisions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, decision_count, sizeof(*decisions), (void**)&decisions));
  memset(decisions, 0, decision_count * sizeof(*decisions));
  out_classifier->decisions = decisions;

  uint32_t decision_ordinal = decision_count;
  iree_host_size_t value_reference_count = 0;
  iree_host_size_t binding_value_count = 0;
  iree_host_size_t feature_count = 0;
  iree_host_size_t action_count = 0;
  demand_id = references->symbols[kernel_symbol_id].first_template_demand_id;
  while (demand_id != LOOM_TEMPLATE_DEMAND_ID_INVALID) {
    const loom_template_demand_t* demand =
        &references->template_demands.values[demand_id];
    demand_id = demand->next_source_demand_id;
    if (demand->source_root_region_index_plus_one != body_region_index + 1) {
      continue;
    }
    const loom_template_decision_model_t* model =
        loom_template_decision_model_lookup(
            decision_models, (loom_symbol_ref_t){
                                 .module_id = 0,
                                 .symbol_id = demand->family_symbol_id,
                             });
    if (model == NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel template decision has no ranked provider model");
    }
    loom_kernel_class_decision_t* decision = &decisions[--decision_ordinal];
    decision->demand = demand;
    decision->model = model;
    const loom_value_slice_t arguments =
        loom_template_apply_operands(demand->apply_op);
    const loom_value_slice_t results =
        loom_template_apply_results(demand->apply_op);
    decision->argument_count = (uint16_t)arguments.count;
    decision->result_count = (uint16_t)results.count;
    value_reference_count +=
        loom_kernel_class_decision_value_reference_count(model);
    binding_value_count += arguments.count + results.count;
    feature_count += model->program.feature_count;
    if (!iree_host_size_checked_add(action_count, model->program.choice_count,
                                    &action_count)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "kernel decision action contract table exceeds host range");
    }
    out_classifier->maximum_provider_count = iree_max(
        out_classifier->maximum_provider_count, model->program.choice_count);
  }
  IREE_ASSERT(decision_ordinal == 0);

  loom_value_id_t* referenced_value_ids = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, value_reference_count, sizeof(*referenced_value_ids),
      (void**)&referenced_value_ids));
  iree_host_size_t referenced_value_count = 0;
  for (uint32_t i = 0; i < decision_count; ++i) {
    const loom_kernel_class_decision_t* decision = &decisions[i];
    for (uint32_t j = 0; j < decision->model->program.predicate_count; ++j) {
      const loom_decision_program_predicate_t* predicate =
          &decision->model->program.predicates[j];
      for (uint8_t k = 0; k < predicate->operand_count; ++k) {
        const loom_decision_program_operand_ref_t operand_ref =
            predicate->operands[k];
        if (loom_decision_program_operand_is_constant(operand_ref)) continue;
        referenced_value_ids[referenced_value_count++] =
            loom_kernel_class_decision_source_value(decision->demand,
                                                    operand_ref);
      }
    }
  }
  IREE_ASSERT(referenced_value_count == value_reference_count);
  loom_kernel_class_sort_value_ids(referenced_value_ids,
                                   referenced_value_count);
  uint32_t projection_count = 0;
  for (iree_host_size_t i = 0; i < referenced_value_count; ++i) {
    if (i == 0 || referenced_value_ids[i - 1] != referenced_value_ids[i]) {
      referenced_value_ids[projection_count++] = referenced_value_ids[i];
    }
  }
  out_classifier->projection_count = projection_count;

  loom_kernel_class_projection_t* projections = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, projection_count, sizeof(*projections), (void**)&projections));
  out_classifier->projections = projections;

  loom_kernel_class_argument_lookup_t argument_lookup;
  IREE_RETURN_IF_ERROR(loom_kernel_class_prepare_argument_lookup(
      kernel_argument_ids, kernel_argument_count, arena, &argument_lookup));

  iree_host_size_t total_term_count = 0;
  for (uint32_t i = 0; i < projection_count; ++i) {
    loom_symbolic_expr_t expression;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
        expression_context, referenced_value_ids[i], &expression));
    if (loom_symbolic_expr_is_linear(&expression)) {
      if (expression.term_count > UINT16_MAX ||
          !iree_host_size_checked_add(total_term_count, expression.term_count,
                                      &total_term_count)) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "kernel boundary projection term table exceeds host range");
      }
    }
  }
  loom_kernel_class_projection_term_t* terms = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, total_term_count, sizeof(*terms), (void**)&terms));

  iree_host_size_t term_cursor = 0;
  for (uint32_t i = 0; i < projection_count; ++i) {
    loom_symbolic_expr_summary_t summary;
    const bool has_summary = loom_symbolic_expr_context_try_lookup_summary(
        expression_context, referenced_value_ids[i], &summary);
    IREE_ASSERT(has_summary);
    const loom_symbolic_expr_t* expression = &summary.expression;
    loom_kernel_class_projection_t* projection = &projections[i];
    *projection = (loom_kernel_class_projection_t){
        .source_value_id = referenced_value_ids[i],
        .kind = LOOM_KERNEL_CLASS_PROJECTION_STATIC_FACTS,
        .static_facts = expression->facts,
    };
    projection->static_facts.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
    if (!loom_symbolic_expr_is_linear(expression)) continue;
    if (expression->term_count == 0) {
      projection->kind = LOOM_KERNEL_CLASS_PROJECTION_CONSTANT;
      projection->constant = expression->constant;
      projection->static_facts = loom_value_facts_unknown();
      continue;
    }

    bool projects_to_boundary = true;
    for (iree_host_size_t j = 0; j < expression->term_count; ++j) {
      uint16_t argument_ordinal = 0;
      if (!loom_kernel_class_lookup_argument(&argument_lookup,
                                             expression->terms[j].value_id,
                                             &argument_ordinal)) {
        projects_to_boundary = false;
        break;
      }
      terms[term_cursor + j] = (loom_kernel_class_projection_term_t){
          .coefficient = expression->terms[j].coefficient,
          .argument_ordinal = argument_ordinal,
      };
    }
    if (!projects_to_boundary) continue;
    projection->kind = LOOM_KERNEL_CLASS_PROJECTION_AFFINE;
    projection->constant = expression->constant;
    projection->terms = terms + term_cursor;
    projection->term_count = (uint16_t)expression->term_count;
    projection->static_facts = loom_value_facts_unknown();
    term_cursor += expression->term_count;
  }

  loom_value_id_t* binding_values = NULL;
  // Reference discovery is complete, so its storage now owns each decision's
  // compact projection-ordinal slices for the classifier lifetime.
  uint32_t* decision_projection_ordinals = referenced_value_ids;
  loom_decision_truth_t* feature_outcomes = NULL;
  loom_kernel_class_action_t* actions = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, binding_value_count, sizeof(*binding_values),
      (void**)&binding_values));
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, feature_count, sizeof(*feature_outcomes),
      (void**)&feature_outcomes));
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, action_count, sizeof(*actions), (void**)&actions));
  if (binding_value_count > 0) {
    memset(binding_values, 0xFF, binding_value_count * sizeof(*binding_values));
  }

  const loom_template_applicability_facts_t kernel_application_facts = {
      .values = kernel_facts,
  };
  iree_host_size_t binding_cursor = 0;
  iree_host_size_t projection_cursor = 0;
  iree_host_size_t feature_cursor = 0;
  iree_host_size_t action_cursor = 0;
  for (uint32_t i = 0; i < decision_count; ++i) {
    loom_kernel_class_decision_t* decision = &decisions[i];
    decision->argument_values =
        decision->argument_count > 0 ? binding_values + binding_cursor : NULL;
    decision->result_values =
        decision->result_count > 0
            ? binding_values + binding_cursor + decision->argument_count
            : NULL;
    binding_cursor += decision->argument_count + decision->result_count;
    uint32_t* projection_ordinal_slice =
        value_reference_count > 0
            ? decision_projection_ordinals + projection_cursor
            : NULL;
    decision->projection_ordinals = projection_ordinal_slice;
    decision->feature_outcomes = decision->model->program.feature_count > 0
                                     ? feature_outcomes + feature_cursor
                                     : NULL;
    decision->actions = decision->model->program.choice_count > 0
                            ? actions + action_cursor
                            : NULL;
    decision->hard_requirement_flags = loom_kernel_class_contract_flags(
        &decision->model->program, decision->model->program.hard_requirements);
    for (uint32_t j = 0; j < decision->model->program.choice_count; ++j) {
      const loom_decision_program_choice_t* choice =
          &decision->model->program.choices[j];
      IREE_ASSERT(choice->action_ordinal <
                  decision->model->program.choice_count);
      actions[action_cursor + choice->action_ordinal] =
          (loom_kernel_class_action_t){
              .choice_ordinal = j,
              .contract_flags = loom_kernel_class_contract_flags(
                  &decision->model->program, choice->conjunction),
          };
    }
    action_cursor += decision->model->program.choice_count;

    iree_host_size_t raw_projection_count = 0;
    bool has_unprojectable_input = false;
    for (uint32_t j = 0; j < decision->model->program.predicate_count; ++j) {
      const loom_decision_program_predicate_t* predicate =
          &decision->model->program.predicates[j];
      for (uint8_t k = 0; k < predicate->operand_count; ++k) {
        const loom_decision_program_operand_ref_t operand_ref =
            predicate->operands[k];
        if (loom_decision_program_operand_is_constant(operand_ref)) continue;
        const loom_value_id_t source_value_id =
            loom_kernel_class_decision_source_value(decision->demand,
                                                    operand_ref);
        uint32_t projection_ordinal = 0;
        const loom_kernel_class_projection_t* projection =
            loom_kernel_class_lookup_projection(projections, projection_count,
                                                source_value_id,
                                                &projection_ordinal);
        const uint32_t signature_ordinal =
            loom_decision_program_operand_ordinal(operand_ref);
        loom_value_id_t* signature_values =
            loom_decision_program_operand_is_result(operand_ref)
                ? (loom_value_id_t*)decision->result_values
                : (loom_value_id_t*)decision->argument_values;
        signature_values[signature_ordinal] = projection_ordinal;
        has_unprojectable_input |=
            projection->kind == LOOM_KERNEL_CLASS_PROJECTION_STATIC_FACTS;
        decision_projection_ordinals[projection_cursor +
                                     raw_projection_count++] =
            projection_ordinal;
      }
    }
    loom_kernel_class_sort_projection_ordinals(projection_ordinal_slice,
                                               raw_projection_count);
    uint32_t unique_projection_count = 0;
    for (iree_host_size_t j = 0; j < raw_projection_count; ++j) {
      if (j == 0 ||
          projection_ordinal_slice[j - 1] != projection_ordinal_slice[j]) {
        projection_ordinal_slice[unique_projection_count++] =
            projection_ordinal_slice[j];
      }
    }
    decision->projection_count = unique_projection_count;
    projection_cursor += unique_projection_count;

    bool has_unresolved_target = false;
    for (uint32_t j = 0; j < decision->model->program.feature_count; ++j) {
      const loom_decision_truth_t outcome = loom_kernel_class_resolve_feature(
          decision->model, j, kernel_target, &kernel_application_facts);
      feature_outcomes[feature_cursor++] = outcome;
      has_unresolved_target |= outcome == LOOM_DECISION_TRUTH_UNKNOWN;
    }

    const loom_value_slice_t source_arguments =
        loom_template_apply_operands(decision->demand->apply_op);
    const loom_value_slice_t source_results =
        loom_template_apply_results(decision->demand->apply_op);
    const loom_decision_program_binding_t generic_binding = {
        .facts = kernel_facts,
        .argument_values = source_arguments.values,
        .result_values = source_results.values,
    };
    uint32_t generic_live_provider_count = 0;
    loom_decision_program_evaluate(
        &decision->model->program, &generic_binding,
        loom_kernel_class_feature_evaluator(decision),
        loom_decision_program_predicate_refiner_empty(),
        LOOM_DECISION_PROGRAM_SELECT_PROVEN,
        /*live_action_ordinals=*/NULL, &generic_live_provider_count,
        &decision->generic_result);

    if (decision->demand->has_lexical_condition) {
      decision->unavailable_reason =
          LOOM_KERNEL_CLASS_DECISION_LEXICAL_CONDITION;
    } else if (has_unprojectable_input) {
      decision->unavailable_reason =
          LOOM_KERNEL_CLASS_DECISION_UNPROJECTABLE_INPUT;
    } else if (has_unresolved_target) {
      decision->unavailable_reason =
          LOOM_KERNEL_CLASS_DECISION_UNRESOLVED_TARGET;
    }
  }
  IREE_ASSERT(binding_cursor == binding_value_count);
  IREE_ASSERT(projection_cursor <= value_reference_count);
  IREE_ASSERT(feature_cursor == feature_count);
  IREE_ASSERT(action_cursor == action_count);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Site evaluation and partitioning
//===----------------------------------------------------------------------===//

static loom_value_facts_t loom_kernel_class_evaluate_projection_transfer(
    const loom_kernel_class_projection_t* projection,
    const loom_kernel_class_site_t* site) {
  loom_value_facts_t result = loom_value_facts_exact_i64(projection->constant);
  for (uint16_t i = 0; i < projection->term_count; ++i) {
    const loom_kernel_class_projection_term_t* term = &projection->terms[i];
    loom_value_facts_t contribution = loom_value_fact_table_lookup(
        site->facts, site->argument_values[term->argument_ordinal]);
    if (term->coefficient != 1) {
      const loom_value_facts_t coefficient =
          loom_value_facts_exact_i64(term->coefficient);
      loom_value_facts_muli(&contribution, &coefficient, &contribution);
    }
    loom_value_facts_addi(&result, &contribution, &result);
  }
  return result;
}

static loom_value_facts_t loom_kernel_class_evaluate_projection(
    const loom_kernel_class_projection_t* projection,
    const loom_kernel_class_site_t* site, uint16_t ranged_term_limit) {
  switch ((loom_kernel_class_projection_kind_t)projection->kind) {
    case LOOM_KERNEL_CLASS_PROJECTION_CONSTANT:
      return loom_value_facts_exact_i64(projection->constant);
    case LOOM_KERNEL_CLASS_PROJECTION_STATIC_FACTS:
      return projection->static_facts;
    case LOOM_KERNEL_CLASS_PROJECTION_AFFINE:
      break;
  }

  int64_t exact_result = projection->constant;
  for (uint16_t i = 0; i < projection->term_count; ++i) {
    const loom_kernel_class_projection_term_t* term = &projection->terms[i];
    const loom_value_facts_t input = loom_value_fact_table_lookup(
        site->facts, site->argument_values[term->argument_ordinal]);
    int64_t contribution = 0;
    if (loom_value_facts_is_float(input) || !loom_value_facts_is_exact(input) ||
        !iree_checked_mul_i64(input.range_lo, term->coefficient,
                              &contribution) ||
        !iree_checked_add_i64(exact_result, contribution, &exact_result)) {
      return projection->term_count <= ranged_term_limit
                 ? loom_kernel_class_evaluate_projection_transfer(projection,
                                                                  site)
                 : loom_value_facts_unknown();
    }
  }
  return loom_value_facts_exact_i64(exact_result);
}

static loom_kernel_class_decision_state_t
loom_kernel_class_skipped_decision_state(
    loom_kernel_class_decision_unavailable_reason_t reason) {
  switch (reason) {
    case LOOM_KERNEL_CLASS_DECISION_LEXICAL_CONDITION:
      return LOOM_KERNEL_CLASS_DECISION_SKIPPED_LEXICAL_CONDITION;
    case LOOM_KERNEL_CLASS_DECISION_UNPROJECTABLE_INPUT:
      return LOOM_KERNEL_CLASS_DECISION_SKIPPED_UNPROJECTABLE_INPUT;
    case LOOM_KERNEL_CLASS_DECISION_UNRESOLVED_TARGET:
      return LOOM_KERNEL_CLASS_DECISION_SKIPPED_UNRESOLVED_TARGET;
    case LOOM_KERNEL_CLASS_DECISION_AVAILABLE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("available kernel decision cannot be skipped here");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_kernel_class_unresolved_decision_status(
    uint32_t decision_ordinal, const loom_decision_program_result_t* result) {
  switch ((loom_decision_program_result_kind_t)result->kind) {
    case LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel decision %u rejects a launch semantic requirement",
          decision_ordinal);
    case LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel decision %u has no provable provider for a launch",
          decision_ordinal);
    case LOOM_DECISION_PROGRAM_RESULT_AMBIGUOUS:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel decision %u has ambiguous best-priority providers",
          decision_ordinal);
    case LOOM_DECISION_PROGRAM_RESULT_NO_MATCH:
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "kernel decision %u has no matching provider",
                              decision_ordinal);
    case LOOM_DECISION_PROGRAM_RESULT_SELECTED:
      break;
  }
  IREE_ASSERT_UNREACHABLE("selected kernel decision requested failure status");
  IREE_BUILTIN_UNREACHABLE();
}

static bool loom_kernel_class_action_is_publishable(
    const loom_kernel_class_decision_t* decision, uint32_t action_ordinal) {
  IREE_ASSERT(action_ordinal < decision->model->program.choice_count);
  if (decision->generic_result.kind == LOOM_DECISION_PROGRAM_RESULT_SELECTED &&
      decision->generic_result.action_ordinal == action_ordinal) {
    return true;
  }
  const loom_kernel_class_contract_flags_t flags =
      decision->hard_requirement_flags |
      decision->actions[action_ordinal].contract_flags;
  return !iree_any_bit_set(flags,
                           LOOM_KERNEL_CLASS_CONTRACT_FLAG_RESULT_DEPENDENT);
}

iree_status_t loom_kernel_class_classifier_collect(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_kernel_class_collection_t* out_collection) {
  *out_collection = (loom_kernel_class_collection_t){
      .site_count = site_count,
  };
  if (site_count == 0) return iree_ok_status();

  loom_decision_class_partition_t partition;
  IREE_RETURN_IF_ERROR(loom_decision_class_partition_initialize(
      site_count, options->class_limit,
      iree_max(1u, classifier->maximum_provider_count), arena, &partition));

  loom_kernel_class_decision_result_t* decision_results = NULL;
  loom_kernel_class_trace_id_t* current_trace_ids = NULL;
  loom_kernel_class_trace_id_t* candidate_trace_ids = NULL;
  loom_value_facts_t* projection_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, classifier->decision_count, sizeof(*decision_results),
      (void**)&decision_results));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, options->class_limit,
                                                 sizeof(*current_trace_ids),
                                                 (void**)&current_trace_ids));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, options->class_limit,
                                                 sizeof(*candidate_trace_ids),
                                                 (void**)&candidate_trace_ids));
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, classifier->projection_count, sizeof(*projection_facts),
      (void**)&projection_facts));
  memset(current_trace_ids, 0xFF,
         options->class_limit * sizeof(*current_trace_ids));
  if (classifier->decision_count > 0) {
    memset(decision_results, 0,
           classifier->decision_count * sizeof(*decision_results));
  }

  loom_value_fact_table_t decision_facts = {
      .entries = projection_facts,
      .count = classifier->projection_count,
      .capacity = classifier->projection_count,
  };
  loom_kernel_class_trace_t* traces = NULL;
  iree_host_size_t trace_capacity = 0;
  uint32_t trace_count = 0;
  uint32_t accepted_decision_count = 0;
  uint32_t skipped_decision_count = 0;

  for (uint32_t decision_ordinal = 0;
       decision_ordinal < classifier->decision_count; ++decision_ordinal) {
    const loom_kernel_class_decision_t* decision =
        &classifier->decisions[decision_ordinal];
    if (decision->unavailable_reason != LOOM_KERNEL_CLASS_DECISION_AVAILABLE) {
      if (decision->generic_result.kind !=
          LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
        return loom_kernel_class_unresolved_decision_status(
            decision_ordinal, &decision->generic_result);
      }
      decision_results[decision_ordinal].state =
          loom_kernel_class_skipped_decision_state(
              decision->unavailable_reason);
      ++skipped_decision_count;
      continue;
    }

    loom_decision_class_partition_begin(&partition,
                                        decision->model->program.choice_count);
    bool within_class_limit = true;
    bool actions_are_publishable = true;
    for (iree_host_size_t site_ordinal = 0; site_ordinal < site_count;
         ++site_ordinal) {
      const loom_kernel_class_site_t* site = &sites[site_ordinal];
      for (uint32_t i = 0; i < decision->projection_count; ++i) {
        const uint32_t projection_ordinal = decision->projection_ordinals[i];
        projection_facts[projection_ordinal] =
            loom_kernel_class_evaluate_projection(
                &classifier->projections[projection_ordinal], site,
                options->ranged_transfer_term_limit);
      }

      const loom_decision_program_binding_t binding = {
          .facts = &decision_facts,
          .argument_values = decision->argument_values,
          .result_values = decision->result_values,
      };
      loom_decision_program_result_t result;
      uint32_t live_provider_count = 0;
      loom_decision_program_evaluate(
          &decision->model->program, &binding,
          loom_kernel_class_feature_evaluator(decision),
          loom_decision_program_predicate_refiner_empty(),
          LOOM_DECISION_PROGRAM_SELECT_PROVEN,
          /*live_action_ordinals=*/NULL, &live_provider_count, &result);
      if (result.kind != LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
        return loom_kernel_class_unresolved_decision_status(decision_ordinal,
                                                            &result);
      }
      if (!loom_kernel_class_action_is_publishable(decision,
                                                   result.action_ordinal)) {
        actions_are_publishable = false;
        break;
      }
      if (!loom_decision_class_partition_record(&partition, site_ordinal,
                                                result.action_ordinal)) {
        within_class_limit = false;
        break;
      }
    }

    if (!actions_are_publishable) {
      if (decision->generic_result.kind !=
          LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "kernel decision %u selects a result-dependent specialization "
            "and has no generic residual",
            decision_ordinal);
      }
      decision_results[decision_ordinal].state =
          LOOM_KERNEL_CLASS_DECISION_SKIPPED_RESULT_DEPENDENT;
      ++skipped_decision_count;
      continue;
    }

    if (!within_class_limit) {
      if (decision->generic_result.kind !=
          LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "kernel decision %u exceeds the %u-class limit and has no "
            "generic residual",
            decision_ordinal, options->class_limit);
      }
      decision_results[decision_ordinal].state =
          LOOM_KERNEL_CLASS_DECISION_SKIPPED_CLASS_LIMIT;
      ++skipped_decision_count;
      continue;
    }

    const uint32_t new_trace_count = partition.candidate_class_count;
    if (trace_count > UINT32_MAX - new_trace_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "kernel class trace table exceeds uint32_t");
    }
    const uint32_t required_trace_count = trace_count + new_trace_count;
    if (required_trace_count > trace_capacity) {
      void* trace_storage = traces;
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          arena, trace_count, required_trace_count, sizeof(*traces),
          &trace_capacity, &trace_storage));
      traces = (loom_kernel_class_trace_t*)trace_storage;
    }
    for (uint32_t i = 0; i < new_trace_count; ++i) {
      const loom_decision_class_ordinal_t parent_class =
          partition.candidate_parent_classes[i];
      traces[trace_count + i] = (loom_kernel_class_trace_t){
          .parent_trace_id = current_trace_ids[parent_class],
          .decision_ordinal = decision_ordinal,
          .action_ordinal = partition.candidate_outcomes[i],
      };
      candidate_trace_ids[i] = trace_count + i;
    }
    trace_count = required_trace_count;
    loom_decision_class_partition_commit(&partition);
    loom_kernel_class_trace_id_t* previous_trace_ids = current_trace_ids;
    current_trace_ids = candidate_trace_ids;
    candidate_trace_ids = previous_trace_ids;
    decision_results[decision_ordinal].state =
        LOOM_KERNEL_CLASS_DECISION_ACCEPTED;
    ++accepted_decision_count;
  }

  loom_kernel_class_t* classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, partition.class_count, sizeof(*classes), (void**)&classes));
  for (loom_decision_class_ordinal_t i = 0; i < partition.class_count; ++i) {
    classes[i] = (loom_kernel_class_t){
        .trace_id = current_trace_ids[i],
    };
  }
  for (iree_host_size_t i = 0; i < site_count; ++i) {
    ++classes[partition.site_classes[i]].member_count;
  }

  *out_collection = (loom_kernel_class_collection_t){
      .site_classes = partition.site_classes,
      .classes = classes,
      .traces = traces,
      .decision_results = decision_results,
      .site_count = site_count,
      .class_count = partition.class_count,
      .trace_count = trace_count,
      .accepted_decision_count = accepted_decision_count,
      .skipped_decision_count = skipped_decision_count,
  };
  return iree_ok_status();
}
