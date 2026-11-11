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
          decision->model, j, kernel_target, &kernel_application_facts);
      feature_outcomes[feature_cursor++] = outcome;
      has_unresolved_target |= outcome == LOOM_DECISION_TRUTH_UNKNOWN;
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
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  iree_status_t status = loom_kernel_class_classifier_build_impl(
      module, kernel_symbol_id, references, decision_models, kernel_facts,
      expression_context, kernel_target, arena, &scratch_arena, out_classifier);
  iree_arena_deinitialize(&scratch_arena);
  return status;
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
    case LOOM_KERNEL_CLASS_DECISION_PATH_CONDITION:
      return LOOM_KERNEL_CLASS_DECISION_SKIPPED_PATH_CONDITION;
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

// Finalized decisions refine a bounded prefix partition. Unprocessed optional
// decisions retain their generic structural requirements in a residual
// partition. Admission counts the observed intersection, including mandatory
// staged values, instead of charging provider choices before their shapes.
typedef struct loom_kernel_class_shape_binding_t {
  // Shared earlier structural bindings for this prefix class, or NULL.
  const struct loom_kernel_class_shape_binding_t* parent;
  // Newly captured structural facts owned by the collection arena.
  const loom_kernel_class_shape_value_t* values;
  // Number of facts introduced by this node.
  uint32_t count;
  // Number of distinct facts along the complete shared chain.
  uint32_t total_count;
  // Canonical projection membership for the complete shared chain.
  loom_value_set_id_t keys;
} loom_kernel_class_shape_binding_t;

typedef struct loom_kernel_class_residual_node_t {
  // Previous residual class before this structural distinction was added.
  uint32_t parent;
  // Last decision needing the distinction, or UINT32_MAX for the root.
  uint32_t decision;
  // Construction scratch counting children in the active refinement.
  uint32_t children;
} loom_kernel_class_residual_node_t;

typedef struct loom_kernel_class_residual_t {
  // Bounded split tree owned by the invocation's residual arena.
  loom_kernel_class_residual_node_t* nodes;
  // Current ancestor of each retained leaf after processed decisions retire.
  uint32_t* active_nodes;
  // Immutable leaf ordinal for each observed launch site.
  loom_decision_class_ordinal_t* site_classes;
  // Number of initialized tree nodes, at most twice the class limit minus one.
  uint32_t node_count;
  // Number of retained leaves, bounded by the class limit.
  uint32_t class_count;
  // First reverse refinement that overflowed, or UINT32_MAX when exact.
  uint32_t overflow_decision;
} loom_kernel_class_residual_t;

typedef struct loom_kernel_class_residual_group_t {
  // Tree node refined by this candidate scalar value.
  uint32_t parent;
  // Normalized exact integer or unknown structural facts.
  loom_value_facts_t facts;
} loom_kernel_class_residual_group_t;

static uint32_t loom_kernel_class_bucket_count(uint32_t capacity) {
  uint32_t count = 1;
  while (count < 2 * capacity) {
    count *= 2;
  }
  return count;
}

static loom_value_facts_t loom_kernel_class_evaluate_shape_projection(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_site_t* site, uint32_t projection,
    const loom_kernel_class_collection_options_t* options) {
  loom_value_facts_t facts = loom_kernel_class_evaluate_projection(
      &classifier->projections[projection], site,
      options->ranged_transfer_term_limit);
  return loom_value_facts_is_exact(facts) && !loom_value_facts_is_float(facts)
             ? loom_value_facts_exact_i64(facts.range_lo)
             : loom_value_facts_unknown();
}

static uint32_t loom_kernel_class_hash_shape_fact(uint32_t hash,
                                                  loom_value_facts_t facts) {
  bool exact = loom_value_facts_is_exact(facts);
  hash = loom_structural_hash_mix_u32(hash, exact);
  return exact ? loom_structural_hash_mix_u64(hash, facts.range_lo) : hash;
}

// Build generic requirements in reverse order. Equal requirements are charged
// only at their last use. Stop at the first overflow: earlier residuals cannot
// fit either, and removing that decision exposes the retained exact suffix.
// Splitting a leaf adds at least two children, bounding the tree by 2L-1 nodes.
static iree_status_t loom_kernel_class_residual_build(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch,
    loom_value_set_index_t* sets, loom_value_set_id_t covered,
    loom_kernel_class_residual_t* residual) {
  *residual = (loom_kernel_class_residual_t){.overflow_decision = UINT32_MAX};
  if (!classifier->has_shape_requirements ||
      site_count <= options->class_limit) {
    return iree_ok_status();
  }
  uint32_t optional_count = 0;
  for (uint32_t i = 0; i < classifier->decision_count; ++i) {
    optional_count += classifier->decisions[i].unavailable_reason ==
                          LOOM_KERNEL_CLASS_DECISION_AVAILABLE &&
                      classifier->decisions[i].generic_result.kind ==
                          LOOM_DECISION_PROGRAM_RESULT_SELECTED;
  }
  // With one optional decision, the candidate itself is the complete plan.
  if (optional_count <= 1) {
    return iree_ok_status();
  }
  uint32_t* last = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      scratch, classifier->projection_count, sizeof(*last), (void**)&last));
  if (classifier->projection_count) {
    memset(last, 0xFF, classifier->projection_count * sizeof(*last));
  }
  bool any = false;
  for (uint32_t i = 0; i < classifier->decision_count; ++i) {
    const loom_kernel_class_decision_t* decision = &classifier->decisions[i];
    if (decision->unavailable_reason != LOOM_KERNEL_CLASS_DECISION_AVAILABLE ||
        decision->generic_result.kind !=
            LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
      continue;
    }
    const loom_kernel_class_shape_t* shape = &decision->residual_shape;
    for (uint32_t j = 0; j < shape->count; ++j) {
      if (!loom_value_set_index_contains(sets, covered,
                                         shape->projections[j])) {
        last[shape->projections[j]] = i;
        any = true;
      }
    }
  }
  if (!any) {
    return iree_ok_status();
  }
  uint32_t capacity = (uint32_t)iree_min(site_count, options->class_limit);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, 2 * capacity, sizeof(*residual->nodes), (void**)&residual->nodes));
  residual->nodes[0] =
      (loom_kernel_class_residual_node_t){.parent = 0, .decision = UINT32_MAX};
  residual->node_count = 1;
  residual->class_count = 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*residual->active_nodes),
      (void**)&residual->active_nodes));
  residual->active_nodes[0] = 0;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, site_count, sizeof(*residual->site_classes),
      (void**)&residual->site_classes));
  memset(residual->site_classes, 0,
         site_count * sizeof(*residual->site_classes));
  loom_decision_class_ordinal_t* next = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch, site_count,
                                                 sizeof(*next), (void**)&next));
  loom_kernel_class_residual_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, capacity, sizeof(*groups), (void**)&groups));
  uint32_t* next_nodes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, capacity, sizeof(*next_nodes), (void**)&next_nodes));
  uint32_t bucket_count = loom_kernel_class_bucket_count(capacity);
  uint32_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, bucket_count, sizeof(*buckets), (void**)&buckets));
  for (uint32_t ordinal = classifier->decision_count; ordinal != 0; --ordinal) {
    uint32_t decision_ordinal = ordinal - 1;
    const loom_kernel_class_decision_t* decision =
        &classifier->decisions[decision_ordinal];
    if (decision->unavailable_reason != LOOM_KERNEL_CLASS_DECISION_AVAILABLE ||
        decision->generic_result.kind !=
            LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
      continue;
    }
    const loom_kernel_class_shape_t* shape = &decision->residual_shape;
    for (uint32_t j = 0; j < shape->count; ++j) {
      uint32_t projection = shape->projections[j];
      if (last[projection] != decision_ordinal) {
        continue;
      }
      memset(buckets, 0xFF, bucket_count * sizeof(*buckets));
      uint32_t group_count = 0;
      for (iree_host_size_t site = 0; site < site_count; ++site) {
        uint32_t parent = residual->active_nodes[residual->site_classes[site]];
        loom_value_facts_t facts = loom_kernel_class_evaluate_shape_projection(
            classifier, &sites[site], projection, options);
        uint32_t hash = loom_kernel_class_hash_shape_fact(
            loom_structural_hash_mix_u32(loom_structural_hash_initialize(),
                                         parent),
            facts);
        uint32_t bucket =
            loom_structural_hash_finalize(hash) & (bucket_count - 1);
        while (buckets[bucket] != UINT32_MAX) {
          const loom_kernel_class_residual_group_t* group =
              &groups[buckets[bucket]];
          if (group->parent == parent &&
              loom_value_facts_equal(group->facts, facts)) {
            break;
          }
          bucket = (bucket + 1) & (bucket_count - 1);
        }
        if (buckets[bucket] == UINT32_MAX) {
          if (group_count == capacity) {
            residual->overflow_decision = decision_ordinal;
            return iree_ok_status();
          }
          buckets[bucket] = group_count;
          groups[group_count++] = (loom_kernel_class_residual_group_t){
              .parent = parent, .facts = facts};
        }
        next[site] = (loom_decision_class_ordinal_t)buckets[bucket];
      }
      for (uint32_t group = 0; group < group_count; ++group) {
        ++residual->nodes[groups[group].parent].children;
      }
      for (uint32_t group = 0; group < group_count; ++group) {
        uint32_t parent = groups[group].parent;
        if (residual->nodes[parent].children == 1) {
          next_nodes[group] = parent;
        } else {
          uint32_t node = residual->node_count++;
          IREE_ASSERT(node < 2 * capacity);
          residual->nodes[node] = (loom_kernel_class_residual_node_t){
              .parent = parent, .decision = decision_ordinal};
          next_nodes[group] = node;
        }
      }
      for (uint32_t group = 0; group < group_count; ++group) {
        residual->nodes[groups[group].parent].children = 0;
      }
      memcpy(residual->site_classes, next, site_count * sizeof(*next));
      memcpy(residual->active_nodes, next_nodes,
             group_count * sizeof(*next_nodes));
      residual->class_count = group_count;
    }
  }
  return iree_ok_status();
}

// Retire distinctions after their final generic consumer. Sites keep stable
// leaf IDs; only the bounded leaf-to-ancestor map changes.
static void loom_kernel_class_residual_advance(
    loom_kernel_class_residual_t* residual, uint32_t decision) {
  if (residual->overflow_decision != UINT32_MAX) {
    if (decision < residual->overflow_decision) {
      return;
    }
    residual->overflow_decision = UINT32_MAX;
  }
  for (uint32_t i = 0; i < residual->class_count; ++i) {
    uint32_t node = residual->active_nodes[i];
    while (residual->nodes[node].decision == decision) {
      node = residual->nodes[node].parent;
    }
    residual->active_nodes[i] = node;
  }
}

static bool loom_kernel_class_completion_fits(
    const loom_kernel_class_residual_t* residual,
    const loom_decision_class_ordinal_t* prefix, iree_host_size_t site_count,
    uint32_t prefix_count, uint32_t capacity, uint64_t* buckets,
    uint32_t bucket_count) {
  if (residual->overflow_decision != UINT32_MAX) {
    return false;
  }
  if (residual->class_count <= 1 ||
      prefix_count * residual->class_count <= capacity) {
    return true;
  }
  memset(buckets, 0xFF, bucket_count * sizeof(*buckets));
  uint32_t count = 0;
  for (iree_host_size_t i = 0; i < site_count; ++i) {
    uint64_t key = ((uint64_t)prefix[i] << 32) |
                   residual->active_nodes[residual->site_classes[i]];
    uint32_t hash = loom_structural_hash_finalize(
        loom_structural_hash_mix_u64(loom_structural_hash_initialize(), key));
    uint32_t bucket = hash & (bucket_count - 1);
    while (buckets[bucket] != UINT64_MAX && buckets[bucket] != key) {
      bucket = (bucket + 1) & (bucket_count - 1);
    }
    if (buckets[bucket] == UINT64_MAX) {
      if (count == capacity) {
        return false;
      }
      ++count;
      buckets[bucket] = key;
    }
  }
  return true;
}

typedef struct loom_kernel_class_refinement_t {
  // Newly captured structural values owned by candidate scratch.
  const loom_kernel_class_shape_value_t* values;
  // Selected provider action, or zero for source-only shape refinement.
  uint32_t action;
  // Observed parent/action pair identifying this refinement's shape slice.
  uint32_t pair;
  // Number of new structural values.
  uint32_t count;
  // Current prefix class supplying earlier decisions and structural facts.
  loom_decision_class_ordinal_t parent;
} loom_kernel_class_refinement_t;

typedef struct loom_kernel_class_candidate_t {
  // Observed structural refinements; NULL when parent bindings are unchanged.
  loom_kernel_class_refinement_t* groups;
  // Number of complete prefix classes, including their structural variants.
  uint32_t count;
  // True when the candidate prefix fits the collection's class limit.
  bool fits;
  // True when every selected action can be published from input facts.
  bool publishable;
} loom_kernel_class_candidate_t;

static iree_status_t loom_kernel_class_refine(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_site_t* sites,
    const loom_kernel_class_collection_options_t* options,
    const loom_kernel_class_decision_t* decision, uint32_t decision_ordinal,
    const loom_kernel_class_shape_t* fixed_shape, bool evaluate,
    loom_value_set_index_t* sets,
    const loom_kernel_class_shape_binding_t* const* bindings,
    loom_value_facts_t* projection_facts,
    loom_decision_class_partition_t* partition, iree_arena_allocator_t* scratch,
    loom_kernel_class_candidate_t* candidate) {
  *candidate =
      (loom_kernel_class_candidate_t){.fits = true, .publishable = true};
  uint32_t capacity =
      (uint32_t)iree_min(partition->site_count, partition->class_limit);
  if (evaluate) {
    loom_decision_class_partition_begin(partition,
                                        decision->model->program.choice_count);
    for (iree_host_size_t i = 0; i < partition->site_count; ++i) {
      for (uint32_t j = 0; j < decision->projection_count; ++j) {
        uint32_t projection = decision->projection_ordinals[j];
        projection_facts[projection] = loom_kernel_class_evaluate_projection(
            &classifier->projections[projection], &sites[i],
            options->ranged_transfer_term_limit);
      }
      const loom_decision_program_indexed_binding_t binding = {
          .facts = projection_facts,
          .argument_values = decision->argument_values,
          .result_values = decision->result_values};
      loom_decision_program_result_t result;
      uint32_t live_count;
      loom_decision_program_evaluate_indexed(
          &decision->model->program, &binding,
          loom_kernel_class_feature_evaluator(decision),
          loom_decision_program_predicate_refiner_empty(),
          LOOM_DECISION_PROGRAM_SELECT_PROVEN, NULL, &live_count, &result);
      if (result.kind != LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
        return loom_kernel_class_unresolved_decision_status(decision_ordinal,
                                                            &result);
      }
      if (!loom_kernel_class_action_is_publishable(decision,
                                                   result.action_ordinal)) {
        candidate->publishable = false;
        return iree_ok_status();
      }
      if (!loom_decision_class_partition_record(partition, i,
                                                result.action_ordinal)) {
        candidate->fits = false;
        return iree_ok_status();
      }
    }
  } else {
    partition->candidate_class_count = partition->class_count;
    memcpy(partition->candidate_site_classes, partition->site_classes,
           partition->site_count * sizeof(*partition->site_classes));
    for (uint32_t i = 0; i < partition->class_count; ++i) {
      partition->candidate_parent_classes[i] = (uint16_t)i;
      partition->candidate_outcomes[i] =
          decision ? decision->generic_result.action_ordinal : 0;
    }
  }
  uint32_t pair_count = partition->candidate_class_count;
  if (!classifier->has_shape_requirements) {
    candidate->count = pair_count;
    return iree_ok_status();
  }
  loom_kernel_class_shape_t* deltas = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, pair_count, sizeof(*deltas), (void**)&deltas));
  uint32_t maximum_values = 0;
  for (uint32_t pair = 0; pair < pair_count; ++pair) {
    uint16_t parent = partition->candidate_parent_classes[pair];
    uint32_t action = partition->candidate_outcomes[pair];
    const loom_kernel_class_shape_t* shape =
        !decision  ? fixed_shape
        : evaluate ? &decision->action_shapes[action]
                   : &decision->residual_shape;
    loom_value_set_id_t keys = bindings[parent] ? bindings[parent]->keys : 0;
    uint32_t count = 0;
    for (uint32_t j = 0; j < shape->count; ++j) {
      count +=
          !loom_value_set_index_contains(sets, keys, shape->projections[j]);
    }
    uint32_t* projections = NULL;
    IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
        scratch, count, sizeof(*projections), (void**)&projections));
    uint32_t next = 0;
    for (uint32_t j = 0; j < shape->count; ++j) {
      uint32_t projection = shape->projections[j];
      if (!loom_value_set_index_contains(sets, keys, projection)) {
        projections[next++] = projection;
      }
    }
    deltas[pair] =
        (loom_kernel_class_shape_t){.projections = projections, .count = count};
    maximum_values = iree_max(maximum_values, count);
  }
  if (!maximum_values) {
    candidate->count = pair_count;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch, capacity,
                                                 sizeof(*candidate->groups),
                                                 (void**)&candidate->groups));
  uint32_t* direct_classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, pair_count, sizeof(*direct_classes), (void**)&direct_classes));
  memset(direct_classes, 0xFF, pair_count * sizeof(*direct_classes));
  uint32_t bucket_count = loom_kernel_class_bucket_count(capacity);
  uint32_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0xFF, bucket_count * sizeof(*buckets));
  loom_kernel_class_shape_value_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, maximum_values, sizeof(*values), (void**)&values));
  for (iree_host_size_t i = 0; i < partition->site_count; ++i) {
    uint32_t pair = partition->candidate_site_classes[i];
    uint16_t parent = partition->candidate_parent_classes[pair];
    uint32_t action = partition->candidate_outcomes[pair];
    const loom_kernel_class_shape_t* delta = &deltas[pair];
    if (delta->count == 0) {
      if (direct_classes[pair] == UINT32_MAX) {
        if (candidate->count == capacity) {
          candidate->fits = false;
          return iree_ok_status();
        }
        direct_classes[pair] = candidate->count;
        candidate->groups[candidate->count++] =
            (loom_kernel_class_refinement_t){
                .parent = parent, .action = action, .pair = pair};
      }
      partition->candidate_site_classes[i] = (uint16_t)direct_classes[pair];
      continue;
    }
    uint32_t hash =
        loom_structural_hash_mix_u32(loom_structural_hash_initialize(), pair);
    for (uint32_t j = 0; j < delta->count; ++j) {
      uint32_t projection = delta->projections[j];
      values[j] = (loom_kernel_class_shape_value_t){
          .projection = projection,
          .facts = loom_kernel_class_evaluate_shape_projection(
              classifier, &sites[i], projection, options)};
      hash = loom_kernel_class_hash_shape_fact(hash, values[j].facts);
    }
    uint32_t bucket = loom_structural_hash_finalize(hash) & (bucket_count - 1);
    while (buckets[bucket] != UINT32_MAX) {
      const loom_kernel_class_refinement_t* group =
          &candidate->groups[buckets[bucket]];
      bool equal = group->pair == pair;
      for (uint32_t j = 0; equal && j < delta->count; ++j) {
        equal = loom_value_facts_equal(values[j].facts, group->values[j].facts);
      }
      if (equal) {
        break;
      }
      bucket = (bucket + 1) & (bucket_count - 1);
    }
    if (buckets[bucket] == UINT32_MAX) {
      if (candidate->count == capacity) {
        candidate->fits = false;
        return iree_ok_status();
      }
      loom_kernel_class_shape_value_t* retained = NULL;
      IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
          scratch, delta->count, sizeof(*retained), (void**)&retained));
      if (delta->count) {
        memcpy(retained, values, delta->count * sizeof(*retained));
      }
      buckets[bucket] = candidate->count;
      candidate->groups[candidate->count++] =
          (loom_kernel_class_refinement_t){.parent = parent,
                                           .action = action,
                                           .pair = pair,
                                           .values = retained,
                                           .count = delta->count};
    }
    partition->candidate_site_classes[i] = (uint16_t)buckets[bucket];
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_class_commit(
    const loom_kernel_class_candidate_t* candidate, uint32_t decision_ordinal,
    loom_value_set_index_t* sets,
    loom_kernel_class_shape_binding_t** current_bindings,
    loom_kernel_class_shape_binding_t** next_bindings,
    loom_kernel_class_trace_id_t* current_traces,
    loom_kernel_class_trace_id_t* next_traces,
    loom_kernel_class_trace_t** traces, uint32_t* trace_count,
    iree_host_size_t* trace_capacity,
    loom_decision_class_partition_t* partition, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch) {
  if (!current_bindings) {
    if (candidate->count > UINT32_MAX - *trace_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "kernel class trace table exceeds uint32_t");
    }
    if (decision_ordinal != UINT32_MAX) {
      void* storage = *traces;
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          arena, *trace_count, *trace_count + candidate->count,
          sizeof(**traces), trace_capacity, &storage));
      *traces = storage;
      for (uint32_t i = 0; i < candidate->count; ++i) {
        (*traces)[*trace_count + i] = (loom_kernel_class_trace_t){
            .parent_trace_id =
                current_traces[partition->candidate_parent_classes[i]],
            .decision_ordinal = decision_ordinal,
            .action_ordinal = partition->candidate_outcomes[i]};
        next_traces[i] = *trace_count + i;
      }
      *trace_count += candidate->count;
    } else {
      for (uint32_t i = 0; i < candidate->count; ++i) {
        next_traces[i] = current_traces[partition->candidate_parent_classes[i]];
      }
    }
    memcpy(current_traces, next_traces,
           candidate->count * sizeof(*current_traces));
    loom_decision_class_partition_commit(partition);
    return iree_ok_status();
  }
  uint32_t bucket_count = loom_kernel_class_bucket_count(candidate->count);
  uint32_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0xFF, bucket_count * sizeof(*buckets));
  loom_kernel_class_trace_t* new_traces = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, candidate->count, sizeof(*new_traces), (void**)&new_traces));
  uint32_t added = 0;
  for (uint32_t i = 0; i < candidate->count; ++i) {
    loom_kernel_class_refinement_t unchanged;
    const loom_kernel_class_refinement_t* group;
    if (candidate->groups) {
      group = &candidate->groups[i];
    } else {
      unchanged = (loom_kernel_class_refinement_t){
          .parent = partition->candidate_parent_classes[i],
          .action = partition->candidate_outcomes[i],
      };
      group = &unchanged;
    }
    const loom_kernel_class_shape_binding_t* parent =
        current_bindings[group->parent];
    if (group->count == 0) {
      next_bindings[i] = current_bindings[group->parent];
    } else {
      loom_kernel_class_shape_binding_t* binding = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, sizeof(*binding), (void**)&binding));
      loom_kernel_class_shape_value_t* values = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, group->count, sizeof(*values), (void**)&values));
      memcpy(values, group->values, group->count * sizeof(*values));
      loom_value_set_id_t keys = parent ? parent->keys : 0;
      for (uint32_t j = 0; j < group->count; ++j) {
        IREE_RETURN_IF_ERROR(
            loom_value_set_index_add(sets, keys, values[j].projection, &keys));
      }
      *binding = (loom_kernel_class_shape_binding_t){
          .parent = parent,
          .values = values,
          .count = group->count,
          .total_count = group->count + (parent ? parent->total_count : 0),
          .keys = keys};
      next_bindings[i] = binding;
    }
    if (decision_ordinal == UINT32_MAX) {
      next_traces[i] = current_traces[group->parent];
    } else {
      uint32_t parent_trace = current_traces[group->parent];
      uint64_t key = ((uint64_t)parent_trace << 32) | group->action;
      uint32_t bucket =
          loom_structural_hash_finalize(loom_structural_hash_mix_u64(
              loom_structural_hash_initialize(), key)) &
          (bucket_count - 1);
      while (buckets[bucket] != UINT32_MAX) {
        const loom_kernel_class_trace_t* trace = &new_traces[buckets[bucket]];
        if (trace->parent_trace_id == parent_trace &&
            trace->action_ordinal == group->action) {
          break;
        }
        bucket = (bucket + 1) & (bucket_count - 1);
      }
      if (buckets[bucket] == UINT32_MAX) {
        buckets[bucket] = added;
        new_traces[added++] =
            (loom_kernel_class_trace_t){.parent_trace_id = parent_trace,
                                        .decision_ordinal = decision_ordinal,
                                        .action_ordinal = group->action};
      }
      next_traces[i] = *trace_count + buckets[bucket];
    }
  }
  if (added > UINT32_MAX - *trace_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "kernel class trace table exceeds uint32_t");
  }
  if (added) {
    void* storage = *traces;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, *trace_count, *trace_count + added,
                              sizeof(**traces), trace_capacity, &storage));
    *traces = storage;
    memcpy(*traces + *trace_count, new_traces, added * sizeof(*new_traces));
    *trace_count += added;
  }
  memcpy(current_bindings, next_bindings,
         candidate->count * sizeof(*current_bindings));
  memcpy(current_traces, next_traces,
         candidate->count * sizeof(*current_traces));
  partition->candidate_class_count = (uint16_t)candidate->count;
  loom_decision_class_partition_commit(partition);
  return iree_ok_status();
}

static iree_status_t loom_kernel_class_collect(
    const loom_kernel_class_classifier_t* classifier,
    const loom_module_t* site_module, const loom_kernel_class_site_t* sites,
    iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch,
    iree_arena_allocator_t* residual_arena,
    loom_kernel_class_collection_t* out_collection) {
  *out_collection = (loom_kernel_class_collection_t){.site_count = site_count};
  if (!site_count) {
    return iree_ok_status();
  }
  const uint32_t capacity =
      (uint32_t)iree_min(site_count, options->class_limit);
  loom_decision_class_partition_t partition;
  IREE_RETURN_IF_ERROR(loom_decision_class_partition_initialize(
      site_count, options->class_limit,
      iree_max(1u, classifier->maximum_provider_count), arena, &partition));
  const loom_kernel_specialization_value_t** site_specializations = NULL;
  if (classifier->specialization_count) {
    IREE_RETURN_IF_ERROR(loom_kernel_specializations_collect(
        site_module, sites, classifier->specialization_count, &partition, arena,
        &site_specializations));
  }
  loom_value_set_index_t* sets = NULL;
  if (classifier->has_shape_requirements) {
    IREE_RETURN_IF_ERROR(loom_value_set_index_allocate(arena, &sets));
  }
  loom_kernel_class_shape_binding_t **bindings = NULL, **next_bindings = NULL;
  loom_kernel_class_trace_id_t *trace_ids = NULL, *next_trace_ids = NULL;
  if (classifier->has_shape_requirements) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, capacity, sizeof(*bindings), (void**)&bindings));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, capacity, sizeof(*next_bindings), (void**)&next_bindings));
    memset(bindings, 0, capacity * sizeof(*bindings));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*trace_ids), (void**)&trace_ids));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*next_trace_ids), (void**)&next_trace_ids));
  memset(trace_ids, 0xFF, capacity * sizeof(*trace_ids));
  loom_kernel_class_decision_result_t* results = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, classifier->decision_count, sizeof(*results), (void**)&results));
  if (classifier->decision_count) {
    memset(results, 0, classifier->decision_count * sizeof(*results));
  }
  loom_value_facts_t* projection_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, classifier->projection_count, sizeof(*projection_facts),
      (void**)&projection_facts));
  loom_kernel_class_trace_t* traces = NULL;
  uint32_t trace_count = 0, accepted = 0, skipped = 0;
  iree_host_size_t trace_capacity = 0;
  loom_kernel_class_residual_t residual = {.overflow_decision = UINT32_MAX};
  uint32_t bucket_count = loom_kernel_class_bucket_count(capacity);
  uint64_t* completion_buckets = NULL;
  // Common and unavailable bodies cannot be deferred to optional admission.
  loom_value_set_id_t common_values = 0;
  if (classifier->has_shape_requirements) {
    IREE_RETURN_IF_ERROR(loom_kernel_class_union_shape(sets, &classifier->shape,
                                                       &common_values));
  }
  for (uint32_t i = 0; i < classifier->decision_count; ++i) {
    const loom_kernel_class_decision_t* decision = &classifier->decisions[i];
    if (decision->unavailable_reason == LOOM_KERNEL_CLASS_DECISION_AVAILABLE) {
      continue;
    }
    if (decision->generic_result.kind !=
        LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
      return loom_kernel_class_unresolved_decision_status(
          i, &decision->generic_result);
    }
    results[i].state =
        loom_kernel_class_skipped_decision_state(decision->unavailable_reason);
    ++skipped;
    if (classifier->has_shape_requirements) {
      IREE_RETURN_IF_ERROR(loom_kernel_class_union_shape(
          sets, &decision->residual_shape, &common_values));
    }
  }
  loom_kernel_class_shape_t common = {0};
  if (common_values) {
    IREE_RETURN_IF_ERROR(
        loom_kernel_class_shape_values(sets, common_values, scratch, &common));
  }
  loom_kernel_class_candidate_t candidate;
  if (common.count) {
    IREE_RETURN_IF_ERROR(loom_kernel_class_refine(
        classifier, sites, options, NULL, UINT32_MAX, &common, false, sets,
        (const loom_kernel_class_shape_binding_t* const*)bindings,
        projection_facts, &partition, scratch, &candidate));
    if (!candidate.fits) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "mandatory kernel shapes exceed the class limit");
    }
    IREE_RETURN_IF_ERROR(loom_kernel_class_commit(
        &candidate, UINT32_MAX, sets, bindings, next_bindings, trace_ids,
        next_trace_ids, &traces, &trace_count, &trace_capacity, &partition,
        arena, scratch));
  }
  iree_arena_reset(scratch);
  // Once a complete residual plan fits, every optional change preserves that
  // guarantee. An over-budget generic plan may still become valid when a
  // provider removes structural requirements; the finalized prefix remains
  // the mandatory lower bound while those decisions are processed.
  bool viable = false;
  for (unsigned phase = 0; phase < 2; ++phase) {
    if (phase) {
      IREE_RETURN_IF_ERROR(loom_kernel_class_residual_build(
          classifier, sites, site_count, options, residual_arena, scratch, sets,
          common_values, &residual));
      iree_arena_reset(scratch);
      if (residual.class_count > 1) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            residual_arena, bucket_count, sizeof(*completion_buckets),
            (void**)&completion_buckets));
      }
      viable = loom_kernel_class_completion_fits(
          &residual, partition.site_classes, site_count, partition.class_count,
          options->class_limit, completion_buckets, bucket_count);
    }
    for (uint32_t i = 0; i < classifier->decision_count; ++i) {
      const loom_kernel_class_decision_t* decision = &classifier->decisions[i];
      bool optional = decision->generic_result.kind ==
                      LOOM_DECISION_PROGRAM_RESULT_SELECTED;
      if (decision->unavailable_reason !=
              LOOM_KERNEL_CLASS_DECISION_AVAILABLE ||
          optional != (phase != 0)) {
        continue;
      }
      if (optional) {
        loom_kernel_class_residual_advance(&residual, i);
      }
      IREE_RETURN_IF_ERROR(loom_kernel_class_refine(
          classifier, sites, options, decision, i, NULL, true, sets,
          (const loom_kernel_class_shape_binding_t* const*)bindings,
          projection_facts, &partition, scratch, &candidate));
      bool complete = candidate.fits && candidate.publishable &&
                      ((viable && candidate.count == partition.class_count) ||
                       loom_kernel_class_completion_fits(
                           &residual, partition.candidate_site_classes,
                           site_count, candidate.count, options->class_limit,
                           completion_buckets, bucket_count));
      // Before a complete plan fits, extra classes must buy potential
      // structural relief. A pure refinement only consumes the budget needed
      // by a later decision that can remove generic requirements.
      bool take = candidate.fits && candidate.publishable &&
                  (!optional || complete ||
                   (!viable && (candidate.count == partition.class_count ||
                                decision->can_reduce_structure)));
      if (take) {
        IREE_RETURN_IF_ERROR(loom_kernel_class_commit(
            &candidate, i, sets, bindings, next_bindings, trace_ids,
            next_trace_ids, &traces, &trace_count, &trace_capacity, &partition,
            arena, scratch));
        results[i].state = LOOM_KERNEL_CLASS_DECISION_ACCEPTED;
        ++accepted;
        if (optional) {
          viable = complete;
        }
      } else {
        if (!optional) {
          if (!candidate.publishable) {
            return iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "kernel decision %u selects a result-dependent specialization "
                "and has no generic residual",
                i);
          }
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "kernel decision %u exceeds the %u-class "
                                  "limit and has no generic residual",
                                  i, options->class_limit);
        }
        results[i].state =
            candidate.publishable
                ? LOOM_KERNEL_CLASS_DECISION_SKIPPED_CLASS_LIMIT
                : LOOM_KERNEL_CLASS_DECISION_SKIPPED_RESULT_DEPENDENT;
        ++skipped;
        iree_arena_reset(scratch);
        IREE_RETURN_IF_ERROR(loom_kernel_class_refine(
            classifier, sites, options, decision, i, NULL, false, sets,
            (const loom_kernel_class_shape_binding_t* const*)bindings,
            projection_facts, &partition, scratch, &candidate));
        if (!candidate.fits) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "generic residual exceeds class limit");
        }
        IREE_RETURN_IF_ERROR(loom_kernel_class_commit(
            &candidate, UINT32_MAX, sets, bindings, next_bindings, trace_ids,
            next_trace_ids, &traces, &trace_count, &trace_capacity, &partition,
            arena, scratch));
      }
      iree_arena_reset(scratch);
    }
  }
  loom_kernel_class_t* classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, partition.class_count, sizeof(*classes), (void**)&classes));
  memset(classes, 0, partition.class_count * sizeof(*classes));
  loom_value_facts_t* argument_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena,
      (iree_host_size_t)partition.class_count *
          classifier->kernel_argument_count,
      sizeof(*argument_facts), (void**)&argument_facts));
  for (uint32_t i = 0; i < partition.class_count; ++i) {
    classes[i].trace_id = trace_ids[i];
    classes[i].argument_facts =
        argument_facts ? argument_facts + (iree_host_size_t)i *
                                              classifier->kernel_argument_count
                       : NULL;
    const loom_kernel_class_shape_binding_t* binding =
        bindings ? bindings[i] : NULL;
    if (!binding) {
      continue;
    }
    classes[i].shape_value_count = binding->total_count;
    loom_kernel_class_shape_value_t* values = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, binding->total_count, sizeof(*values), (void**)&values));
    classes[i].shape_values = values;
    for (; binding; binding = binding->parent) {
      memcpy(values, binding->values, binding->count * sizeof(*values));
      values += binding->count;
    }
  }
  for (iree_host_size_t i = 0; i < site_count; ++i) {
    loom_kernel_class_t* entry = &classes[partition.site_classes[i]];
    loom_value_facts_t* facts = (loom_value_facts_t*)entry->argument_facts;
    for (uint16_t j = 0; j < classifier->kernel_argument_count; ++j) {
      loom_value_facts_t observed = loom_value_fact_table_lookup(
          sites[i].facts, sites[i].argument_values[j]);
      observed.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
      if (!entry->member_count) {
        facts[j] = observed;
      } else {
        loom_value_facts_meet(&facts[j], &observed, &facts[j]);
      }
    }
    ++entry->member_count;
    entry->specializations =
        site_specializations ? site_specializations[i] : NULL;
  }
  *out_collection =
      (loom_kernel_class_collection_t){.specialization_module = site_module,
                                       .site_classes = partition.site_classes,
                                       .classes = classes,
                                       .traces = traces,
                                       .decision_results = results,
                                       .site_count = site_count,
                                       .class_count = partition.class_count,
                                       .trace_count = trace_count,
                                       .accepted_decision_count = accepted,
                                       .skipped_decision_count = skipped};
  return iree_ok_status();
}

iree_status_t loom_kernel_class_classifier_collect(
    const loom_kernel_class_classifier_t* classifier,
    const loom_module_t* site_module, const loom_kernel_class_site_t* sites,
    iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_kernel_class_collection_t* out_collection) {
  iree_arena_allocator_t scratch, residual_arena;
  iree_arena_initialize(arena->block_pool, &scratch);
  iree_arena_initialize(arena->block_pool, &residual_arena);
  iree_status_t status = loom_kernel_class_collect(
      classifier, site_module, sites, site_count, options, arena, &scratch,
      &residual_arena, out_collection);
  iree_arena_deinitialize(&scratch);
  iree_arena_deinitialize(&residual_arena);
  return status;
}
