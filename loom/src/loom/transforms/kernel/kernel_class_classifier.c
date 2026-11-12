// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_class_classifier.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/template/ops.h"
#include "loom/transforms/kernel/kernel_shape_requirements.h"
#include "loom/transforms/kernel/kernel_specialization.h"
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
  if (count == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(arena, count, element_size, out_values);
}

static bool loom_kernel_class_value_ids_are_ordered(
    const loom_value_id_t* value_ids, uint16_t count) {
  for (uint16_t i = 1; i < count; ++i) {
    if (value_ids[i - 1] >= value_ids[i]) {
      return false;
    }
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
  if (begin == lookup->argument_count) {
    return false;
  }
  if (lookup->arguments_are_ordered) {
    if (lookup->argument_ids[begin] != value_id) {
      return false;
    }
    *out_argument_ordinal = begin;
  } else {
    if (lookup->sorted_entries[begin].value_id != value_id) {
      return false;
    }
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
    const loom_op_t* application,
    loom_decision_program_operand_ref_t operand_ref) {
  const uint32_t ordinal = loom_decision_program_operand_ordinal(operand_ref);
  const loom_value_slice_t values =
      loom_decision_program_operand_is_result(operand_ref)
          ? loom_template_apply_results(application)
          : loom_template_apply_operands(application);
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
    const loom_op_t* application,
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
    case LOOM_TEMPLATE_DECISION_FEATURE_INPUT:
      return loom_template_applicability_evaluate_input(
          model->module, loom_op_const_operands(application),
          &feature->value.input, kernel_application_facts);
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

// During construction a shape slice names source SSA values. Once projection
// discovery is complete it is translated in place to projection ordinals.
static iree_status_t loom_kernel_class_shape_values(
    loom_value_set_index_t* sets, loom_value_set_id_t values,
    iree_arena_allocator_t* arena, loom_kernel_class_shape_t* out_shape) {
  *out_shape = (loom_kernel_class_shape_t){0};
  loom_value_set_cursor_t cursor;
  loom_value_set_cursor_begin(sets, values, &cursor);
  while (loom_value_set_cursor_next(&cursor) != LOOM_VALUE_ID_INVALID) {
    ++out_shape->count;
  }
  uint32_t* storage = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, out_shape->count, sizeof(*storage), (void**)&storage));
  out_shape->projections = storage;
  loom_value_set_cursor_begin(sets, values, &cursor);
  for (uint32_t i = 0; i < out_shape->count; ++i) {
    storage[i] = loom_value_set_cursor_next(&cursor);
  }
  return iree_ok_status();
}

// Keep an available boundary expression as one requirement. When a value is
// supplied through a non-affine operation or exact helper result, preserve its
// retained formal dependencies instead of silently merging its static shape.
static iree_status_t loom_kernel_class_shape_resolve(
    const loom_kernel_shape_requirements_t* requirements,
    loom_symbolic_expr_context_t* expression_context,
    const loom_kernel_class_argument_lookup_t* arguments,
    iree_arena_allocator_t* arena, loom_kernel_class_shape_t* shape) {
  loom_value_set_id_t values = 0;
  for (uint32_t i = 0; i < shape->count; ++i) {
    const loom_value_id_t value = shape->projections[i];
    loom_symbolic_expr_t expression;
    IREE_RETURN_IF_ERROR(
        loom_symbolic_expr_from_value(expression_context, value, &expression));
    bool projected = loom_symbolic_expr_is_linear(&expression);
    for (iree_host_size_t j = 0; projected && j < expression.term_count; ++j) {
      uint16_t ordinal;
      projected = loom_kernel_class_lookup_argument(
          arguments, expression.terms[j].value_id, &ordinal);
    }
    if (projected) {
      if (expression.term_count) {
        IREE_RETURN_IF_ERROR(loom_value_set_index_add(requirements->sets,
                                                      values, value, &values));
      }
    } else {
      IREE_RETURN_IF_ERROR(loom_value_set_index_union(
          requirements->sets, values,
          loom_kernel_shape_value_arguments(requirements, value), &values));
    }
  }
  return loom_kernel_class_shape_values(requirements->sets, values, arena,
                                        shape);
}

static iree_status_t loom_kernel_class_provider_shape(
    const loom_module_t* module,
    const loom_kernel_shape_requirements_t* requirements,
    const loom_template_provider_summary_t* provider,
    loom_value_slice_t actuals, iree_arena_allocator_t* arena,
    loom_kernel_class_shape_t* out_shape) {
  *out_shape = (loom_kernel_class_shape_t){0};
  if (provider->module != module ||
      !loom_symbol_ref_is_valid(provider->symbol) ||
      provider->symbol.module_id != 0) {
    return iree_ok_status();
  }
  const loom_func_like_t function = loom_func_like_cast(
      module, module->symbols.entries[provider->symbol.symbol_id].defining_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  const loom_value_set_id_t required =
      requirements->symbols[provider->symbol.symbol_id].arguments;
  loom_value_set_id_t values = 0;
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (loom_value_set_index_contains(requirements->sets, required,
                                      arguments[i])) {
      IREE_RETURN_IF_ERROR(loom_value_set_index_add(
          requirements->sets, values, actuals.values[i], &values));
    }
  }
  return loom_kernel_class_shape_values(requirements->sets, values, arena,
                                        out_shape);
}

static void loom_kernel_class_shape_project(
    loom_kernel_class_shape_t* shape,
    const loom_kernel_class_projection_t* projections,
    uint32_t projection_count) {
  uint32_t* ordinals = (uint32_t*)shape->projections;
  uint32_t count = 0;
  for (uint32_t i = 0; i < shape->count; ++i) {
    uint32_t ordinal = 0;
    const loom_kernel_class_projection_t* projection =
        loom_kernel_class_lookup_projection(projections, projection_count,
                                            ordinals[i], &ordinal);
    if (projection->kind == LOOM_KERNEL_CLASS_PROJECTION_AFFINE) {
      ordinals[count++] = ordinal;
    }
  }
  shape->count = count;
}

static iree_status_t loom_kernel_class_union_shape(
    loom_value_set_index_t* sets, const loom_kernel_class_shape_t* shape,
    loom_value_set_id_t* inout_values) {
  for (uint32_t i = 0; i < shape->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_set_index_add(
        sets, *inout_values, shape->projections[i], inout_values));
  }
  return iree_ok_status();
}

static bool loom_kernel_class_shape_equal(
    const loom_kernel_class_shape_t* lhs,
    const loom_kernel_class_shape_t* rhs) {
  return lhs->count == rhs->count &&
         (lhs->count == 0 ||
          memcmp(lhs->projections, rhs->projections,
                 lhs->count * sizeof(*lhs->projections)) == 0);
}

// A currently proven fallback does not settle a deferred application. Later
// target and class facts can make a higher-ranked provider applicable, so keep
// its structural requirements until an exact choice retires that possibility.
static iree_status_t loom_kernel_class_build_residual_shape(
    loom_kernel_class_decision_t* decision,
    const loom_decision_program_binding_t* binding,
    loom_value_set_index_t* sets, uint32_t maximum_provider_count,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch,
    uint32_t** live_actions) {
  if (decision->generic_result.kind != LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
    return iree_ok_status();
  }
  const loom_kernel_class_shape_t* generic =
      &decision->action_shapes[decision->generic_result.action_ordinal];
  decision->residual_shape = *generic;
  bool different = false;
  for (uint32_t i = 0; i < decision->model->program.choice_count; ++i) {
    const loom_kernel_class_shape_t* shape = &decision->action_shapes[i];
    different |= !loom_kernel_class_shape_equal(shape, generic);
  }
  if (!different) {
    return iree_ok_status();
  }
  if (!*live_actions) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch, maximum_provider_count, sizeof(**live_actions),
        (void**)live_actions));
  }
  uint32_t live_count = 0;
  loom_decision_program_result_t result;
  loom_decision_program_evaluate(
      &decision->model->program, binding,
      loom_kernel_class_feature_evaluator(decision),
      loom_decision_program_predicate_refiner_empty(),
      LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED, *live_actions, &live_count,
      &result);
  const loom_kernel_class_shape_t* first =
      &decision->action_shapes[(*live_actions)[0]];
  for (uint32_t i = 1; i < live_count; ++i) {
    decision->can_reduce_structure |= !loom_kernel_class_shape_equal(
        &decision->action_shapes[(*live_actions)[i]], first);
  }
  // Equal feasible shapes already own the complete union. Otherwise at least
  // one provider omits a requirement from that union and may free class budget.
  if (!decision->can_reduce_structure) {
    decision->residual_shape = *first;
    return iree_ok_status();
  }
  loom_value_set_id_t values = 0;
  for (uint32_t i = 0; i < live_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_kernel_class_union_shape(
        sets, &decision->action_shapes[(*live_actions)[i]], &values));
  }
  return loom_kernel_class_shape_values(sets, values, arena,
                                        &decision->residual_shape);
}

//===----------------------------------------------------------------------===//
// Classifier construction
//===----------------------------------------------------------------------===//

static iree_status_t loom_kernel_class_classifier_build_impl(
    const loom_module_t* module, loom_symbol_id_t kernel_symbol_id,
    const loom_symbol_reference_table_t* references,
    const loom_template_decision_model_catalog_t* decision_models,
    const loom_value_fact_table_t* kernel_facts,
    loom_symbolic_expr_context_t* expression_context,
    const loom_template_applicability_target_t* kernel_target,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
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
  loom_value_facts_t* argument_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, kernel_argument_count, sizeof(*argument_facts),
      (void**)&argument_facts));
  for (uint16_t i = 0; i < kernel_argument_count; ++i) {
    argument_facts[i] =
        loom_value_fact_table_lookup(kernel_facts, kernel_argument_ids[i]);
    argument_facts[i].extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  }
  out_classifier->argument_facts = argument_facts;
  out_classifier->specialization_count =
      (uint16_t)loom_func_like_specialization_count(kernel);

  const loom_symbol_reference_symbol_occurrences_t kernel_references =
      loom_symbol_reference_table_symbol(references, kernel_symbol_id);
  uint32_t decision_count = 0;
  loom_template_demand_id_t demand_id =
      kernel_references.first_template_demand_id;
  while (demand_id != LOOM_TEMPLATE_DEMAND_ID_INVALID) {
    const loom_template_demand_t* demand =
        &references->template_demands.values[demand_id];
    if (demand->source_root_region_index_plus_one == body_region_index + 1) {
      ++decision_count;
    }
    demand_id = demand->next_source_demand_id;
  }
  out_classifier->decision_count = decision_count;

  loom_kernel_class_decision_t* decisions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, decision_count, sizeof(*decisions), (void**)&decisions));
  memset(decisions, 0, decision_count * sizeof(*decisions));
  out_classifier->decisions = decisions;

  loom_kernel_class_argument_lookup_t argument_lookup;
  IREE_RETURN_IF_ERROR(loom_kernel_class_prepare_argument_lookup(
      kernel_argument_ids, kernel_argument_count, scratch_arena,
      &argument_lookup));
  loom_kernel_shape_requirements_t requirements;
  IREE_RETURN_IF_ERROR(loom_kernel_shape_requirements_build(
      module, decision_models, scratch_arena, &requirements));
  IREE_RETURN_IF_ERROR(loom_kernel_class_shape_values(
      requirements.sets, requirements.symbols[kernel_symbol_id].values,
      scratch_arena, &out_classifier->shape));
  IREE_RETURN_IF_ERROR(loom_kernel_class_shape_resolve(
      &requirements, expression_context, &argument_lookup, arena,
      &out_classifier->shape));
  out_classifier->has_shape_requirements = out_classifier->shape.count != 0;
  uint32_t decision_ordinal = decision_count;
  iree_host_size_t value_reference_count = out_classifier->shape.count;
  iree_host_size_t binding_value_count = 0;
  iree_host_size_t feature_count = 0;
  iree_host_size_t action_count = 0;
  demand_id = kernel_references.first_template_demand_id;
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
    decision->application = demand->apply_op;
    if (demand->has_path_condition) {
      decision->unavailable_reason = LOOM_KERNEL_CLASS_DECISION_PATH_CONDITION;
    }
    decision->model = model;
    const loom_value_slice_t arguments =
        loom_template_apply_operands(demand->apply_op);
    const loom_value_slice_t results =
        loom_template_apply_results(demand->apply_op);
    decision->argument_count = (uint16_t)arguments.count;
    decision->result_count = (uint16_t)results.count;
    value_reference_count +=
        loom_kernel_class_decision_value_reference_count(model);
    loom_kernel_class_shape_t* shapes = NULL;
    IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
        arena, model->providers.count, sizeof(*shapes), (void**)&shapes));
    decision->action_shapes = shapes;
    for (uint32_t i = 0; i < model->providers.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_kernel_class_provider_shape(
          module, &requirements,
          loom_template_decision_model_provider(model, i), arguments,
          scratch_arena, &shapes[i]));
      IREE_RETURN_IF_ERROR(
          loom_kernel_class_shape_resolve(&requirements, expression_context,
                                          &argument_lookup, arena, &shapes[i]));
      value_reference_count += shapes[i].count;
      out_classifier->has_shape_requirements |= shapes[i].count != 0;
    }
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
      scratch_arena, value_reference_count, sizeof(*referenced_value_ids),
      (void**)&referenced_value_ids));
  iree_host_size_t referenced_value_count = 0;
  for (uint32_t i = 0; i < out_classifier->shape.count; ++i) {
    referenced_value_ids[referenced_value_count++] =
        out_classifier->shape.projections[i];
  }
  for (uint32_t i = 0; i < decision_count; ++i) {
    for (uint32_t j = 0; j < decisions[i].model->providers.count; ++j) {
      const loom_kernel_class_shape_t* shape = &decisions[i].action_shapes[j];
      for (uint32_t k = 0; k < shape->count; ++k) {
        referenced_value_ids[referenced_value_count++] = shape->projections[k];
      }
    }
    const loom_kernel_class_decision_t* decision = &decisions[i];
    for (uint32_t j = 0; j < decision->model->program.predicate_count; ++j) {
      const loom_decision_program_predicate_t* predicate =
          &decision->model->program.predicates[j];
      for (uint8_t k = 0; k < predicate->operand_count; ++k) {
        const loom_decision_program_operand_ref_t operand_ref =
            predicate->operands[k];
        if (loom_decision_program_operand_is_constant(operand_ref)) {
          continue;
        }
        referenced_value_ids[referenced_value_count++] =
            loom_kernel_class_decision_source_value(decision->application,
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
    if (!loom_symbolic_expr_is_linear(expression)) {
      continue;
    }
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
    if (!projects_to_boundary) {
      continue;
    }
    projection->kind = LOOM_KERNEL_CLASS_PROJECTION_AFFINE;
    projection->constant = expression->constant;
    projection->terms = terms + term_cursor;
    projection->term_count = (uint16_t)expression->term_count;
    projection->static_facts = loom_value_facts_unknown();
    term_cursor += expression->term_count;
  }

  loom_kernel_class_shape_project(&out_classifier->shape, projections,
                                  projection_count);
  out_classifier->has_shape_requirements = out_classifier->shape.count != 0;
  for (uint32_t i = 0; i < decision_count; ++i) {
    for (uint32_t j = 0; j < decisions[i].model->providers.count; ++j) {
      loom_kernel_class_shape_project(
          (loom_kernel_class_shape_t*)&decisions[i].action_shapes[j],
          projections, projection_count);
      out_classifier->has_shape_requirements |=
          decisions[i].action_shapes[j].count != 0;
    }
  }

  loom_value_id_t* binding_values = NULL;
  // Reuse discovery scratch to pack each decision's unique projection ordinals.
  // The occurrence-sized allocation retires after its compact prefix is copied.
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
  uint32_t* live_actions = NULL;
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
        if (loom_decision_program_operand_is_constant(operand_ref)) {
          continue;
        }
        const loom_value_id_t source_value_id =
            loom_kernel_class_decision_source_value(decision->application,
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
          decision->model, j, decision->application, kernel_target,
          &kernel_application_facts);
      feature_outcomes[feature_cursor++] = outcome;
      if (decision->model->features[j].kind ==
          LOOM_TEMPLATE_DECISION_FEATURE_INPUT) {
        has_unprojectable_input |= outcome == LOOM_DECISION_TRUTH_UNKNOWN;
      } else {
        has_unresolved_target |= outcome == LOOM_DECISION_TRUTH_UNKNOWN;
      }
    }

    const loom_value_slice_t source_arguments =
        loom_template_apply_operands(decision->application);
    const loom_value_slice_t source_results =
        loom_template_apply_results(decision->application);
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

    IREE_RETURN_IF_ERROR(loom_kernel_class_build_residual_shape(
        decision, &generic_binding, requirements.sets,
        out_classifier->maximum_provider_count, arena, scratch_arena,
        &live_actions));

    if (decision->unavailable_reason ==
        LOOM_KERNEL_CLASS_DECISION_PATH_CONDITION) {
      continue;
    }
    if (has_unprojectable_input) {
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

  uint32_t* retained_projection_ordinals = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, projection_cursor, sizeof(*retained_projection_ordinals),
      (void**)&retained_projection_ordinals));
  if (projection_cursor != 0) {
    memcpy(retained_projection_ordinals, decision_projection_ordinals,
           projection_cursor * sizeof(*retained_projection_ordinals));
  }
  projection_cursor = 0;
  for (uint32_t i = 0; i < decision_count; ++i) {
    loom_kernel_class_decision_t* decision = &decisions[i];
    decision->projection_ordinals =
        decision->projection_count != 0
            ? retained_projection_ordinals + projection_cursor
            : NULL;
    projection_cursor += decision->projection_count;
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_class_classifier_build(
    const loom_module_t* module, loom_symbol_id_t kernel_symbol_id,
    const loom_symbol_reference_table_t* references,
    const loom_template_decision_model_catalog_t* decision_models,
    const loom_value_fact_table_t* kernel_facts,
    loom_symbolic_expr_context_t* expression_context,
    const loom_template_applicability_target_t* kernel_target,
    iree_arena_allocator_t* arena,
    loom_kernel_class_classifier_t* out_classifier) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(expression_context->arena->block_pool, &scratch_arena);
  iree_status_t status = loom_kernel_class_classifier_build_impl(
      module, kernel_symbol_id, references, decision_models, kernel_facts,
      expression_context, kernel_target, arena, &scratch_arena, out_classifier);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
