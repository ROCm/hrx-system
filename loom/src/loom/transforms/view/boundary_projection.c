// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/view/boundary_projection.h"

#include <string.h>

#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/view_regions.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/target/provider.h"
#include "loom/transforms/boundary/projection_plan.h"
#include "loom/transforms/view/offset_expression.h"

typedef struct loom_view_boundary_projection_plan_state_t {
  // Shared physical schema for each retained view slot.
  loom_boundary_projection_schema_t schema;
} loom_view_boundary_projection_plan_state_t;

typedef struct loom_view_boundary_offset_t {
  // View value whose definition anchors offset materialization.
  loom_value_id_t anchor_value_id;
  // Existing or once-materialized complete byte offset.
  loom_value_id_t value_id;
  // Existing scalar or future candidate offset added to expression.
  loom_value_id_t base_value_id;
  // Retained affine expression added to base_value_id.
  const loom_symbolic_expr_t* expression;
  // Candidate supplying base_value_id, or IREE_HOST_SIZE_MAX.
  iree_host_size_t dependency;
} loom_view_boundary_offset_t;

typedef struct loom_view_boundary_coordinate_t {
  // Existing materializing buffer, or INVALID when supplied by a candidate.
  loom_value_id_t buffer_value_id;
  // View-region recipe for the byte offset.
  loom_view_region_id_t region_id;
  // Candidate supplying the root and possibly base offset.
  iree_host_size_t dependency;
  // Correlated base root-and-offset selection, or IREE_HOST_SIZE_MAX.
  iree_host_size_t selection;
} loom_view_boundary_coordinate_t;

typedef struct loom_view_boundary_selection_t {
  // Original binary value selection whose alternatives supply the coordinate.
  loom_op_t* op;
  // Selected semantic view result.
  loom_value_id_t result_value_id;
  // Scalar i1 value selecting the true or false coordinate.
  loom_value_id_t condition_value_id;
  // Semantic view selected when condition_value_id is true.
  loom_value_id_t true_value_id;
  // Semantic view selected when condition_value_id is false.
  loom_value_id_t false_value_id;
  // Planned true-value coordinate.
  loom_view_boundary_coordinate_t true_coordinate;
  // Planned false-value coordinate.
  loom_view_boundary_coordinate_t false_coordinate;
  // Materialized selected buffer, or INVALID before application.
  loom_value_id_t buffer_value_id;
  // Materialized selected byte offset, or INVALID before application.
  loom_value_id_t offset_value_id;
  // Whether both alternatives have complete physical coordinates.
  bool selected;
  // Whether recursive planning of this selection is active.
  bool planning;
} loom_view_boundary_selection_t;

typedef struct loom_view_boundary_projection_function_state_t {
  // Symbolic-expression context active while the local domain is acquired.
  loom_symbolic_expr_context_t expressions;
  // View-region analysis over the original function.
  loom_view_region_table_t regions;
  // Offset recipes indexed by view-region ID.
  loom_view_boundary_offset_t* offsets;
  // Correlated selections needed by boundary coordinates.
  loom_view_boundary_selection_t* selections;
  // Number of correlated selections.
  iree_host_size_t selection_count;
  // Allocated correlated-selection capacity.
  iree_host_size_t selection_capacity;
  // Selection index by function-local result ordinal, or IREE_HOST_SIZE_MAX.
  iree_host_size_t* selection_indices;
} loom_view_boundary_projection_function_state_t;

static loom_view_boundary_projection_plan_state_t*
loom_view_boundary_plan_state(const loom_boundary_projection_rule_t* rule,
                              const loom_boundary_projection_plan_t* plan) {
  return (loom_view_boundary_projection_plan_state_t*)
      loom_boundary_projection_rule_state(plan, rule);
}

static loom_view_boundary_projection_function_state_t*
loom_view_boundary_function_state(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  return (loom_view_boundary_projection_function_state_t*)
      loom_boundary_projection_function_rule_state(plan, function, rule);
}

static iree_status_t loom_view_boundary_initialize_schema(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan) {
  loom_view_boundary_projection_plan_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  loom_type_t* component_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, 2, sizeof(*component_types), (void**)&component_types));
  component_types[0] = loom_type_buffer();
  component_types[1] = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);

  iree_string_view_t* component_name_suffixes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, 2, sizeof(*component_name_suffixes),
      (void**)&component_name_suffixes));
  component_name_suffixes[0] = IREE_SV("root");
  component_name_suffixes[1] = IREE_SV("offset");
  state->schema = (loom_boundary_projection_schema_t){
      .rule = rule,
      .component_types = component_types,
      .component_name_suffixes = component_name_suffixes,
      .component_count = 2,
  };
  loom_boundary_projection_set_rule_state(plan, rule, state);
  return iree_ok_status();
}

static bool loom_view_boundary_provider_uses_buffer_offset(
    const loom_target_function_version_t* version) {
  return version != NULL && version->resolved_target.provider != NULL &&
         version->resolved_target.provider->view_boundary_carrier ==
             LOOM_TARGET_VIEW_BOUNDARY_CARRIER_BUFFER_OFFSET;
}

static bool loom_view_boundary_function_applies(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  (void)rule;
  (void)plan;
  return loom_view_boundary_provider_uses_buffer_offset(
      loom_target_function_version_const_cast(function->version));
}

static iree_status_t loom_view_boundary_plan_slot(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  (void)role;
  (void)block;
  *out_schema = (loom_boundary_projection_schema_t){0};
  *out_claimed = false;
  if (!loom_view_boundary_function_applies(rule, plan, function) ||
      !loom_type_is_view(loom_module_value_type(plan->module, value_id))) {
    return iree_ok_status();
  }
  const loom_view_boundary_projection_plan_state_t* state =
      loom_view_boundary_plan_state(rule, plan);
  IREE_ASSERT(state != NULL);
  *out_schema = state->schema;
  *out_claimed = true;
  return iree_ok_status();
}

static void loom_view_boundary_plan_offset(
    const loom_boundary_projection_function_t* function,
    const loom_view_boundary_projection_function_state_t* state,
    const loom_view_region_t* region, loom_view_boundary_offset_t* offset) {
  *offset = (loom_view_boundary_offset_t){
      .anchor_value_id = region->view_value_id,
      .value_id = region->begin_value_id,
      .base_value_id = LOOM_VALUE_ID_INVALID,
      .dependency = IREE_HOST_SIZE_MAX,
  };
  if (offset->value_id != LOOM_VALUE_ID_INVALID) {
    return;
  }
  if (loom_symbolic_expr_is_linear(&region->projection_byte_offset)) {
    const iree_host_size_t dependency = loom_boundary_projection_slot_index(
        function, region->base_view_value_id);
    if (dependency != IREE_HOST_SIZE_MAX) {
      offset->dependency = dependency;
    } else {
      const loom_view_region_t* base_region = NULL;
      if (loom_view_region_table_try_lookup(
              &state->regions, region->base_view_value_id, &base_region)) {
        offset->base_value_id = base_region->begin_value_id;
      }
    }
    if (offset->dependency != IREE_HOST_SIZE_MAX ||
        offset->base_value_id != LOOM_VALUE_ID_INVALID) {
      offset->expression = &region->projection_byte_offset;
      return;
    }
  }
  if (loom_symbolic_expr_is_linear(&region->begin_byte_offset)) {
    offset->expression = &region->begin_byte_offset;
  }
}

static bool loom_view_boundary_schema_uses_rule(
    const loom_boundary_projection_schema_t* schemas, uint16_t count,
    const loom_boundary_projection_rule_t* rule) {
  for (uint16_t i = 0; i < count; ++i) {
    if (schemas[i].rule == rule) {
      return true;
    }
  }
  return false;
}

static bool loom_view_boundary_function_requires_analysis(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  if (loom_view_boundary_schema_uses_rule(function->argument_schemas,
                                          function->argument_count, rule) ||
      loom_view_boundary_schema_uses_rule(function->result_schemas,
                                          function->result_count, rule)) {
    return true;
  }
  for (iree_host_size_t i = 0; i < function->candidate_count; ++i) {
    if (function->candidates[i].schema.rule == rule) {
      return true;
    }
  }
  for (iree_host_size_t i = 0; i < function->call_count; ++i) {
    const loom_boundary_projection_function_t* callee =
        &plan->functions[function->calls[i].callee_index];
    if (loom_view_boundary_schema_uses_rule(callee->argument_schemas,
                                            callee->argument_count, rule) ||
        loom_view_boundary_schema_uses_rule(callee->result_schemas,
                                            callee->result_count, rule)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_view_boundary_prepare_function(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  if (!function->selected || !loom_func_like_body(function->function) ||
      !loom_view_boundary_function_requires_analysis(rule, plan, function)) {
    return iree_ok_status();
  }
  IREE_ASSERT(function->facts != NULL);

  loom_view_boundary_projection_function_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  loom_boundary_projection_set_function_rule_state(plan, function, rule, state);

  if (function->domain.value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, function->domain.value_count,
        sizeof(*state->selection_indices), (void**)&state->selection_indices));
    for (loom_value_ordinal_t i = 0; i < function->domain.value_count; ++i) {
      state->selection_indices[i] = IREE_HOST_SIZE_MAX;
    }
  }

  loom_symbolic_expr_context_initialize(plan->module, &function->domain,
                                        function->facts, plan->arena,
                                        &state->expressions);
  IREE_RETURN_IF_ERROR(loom_view_region_table_initialize(
      &function->domain, &state->expressions, &state->regions));
  IREE_RETURN_IF_ERROR(loom_view_region_table_analyze(&state->regions));
  if (state->regions.region_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, state->regions.region_count, sizeof(*state->offsets),
      (void**)&state->offsets));
  for (iree_host_size_t i = 0; i < state->regions.region_count; ++i) {
    loom_view_boundary_plan_offset(function, state, &state->regions.regions[i],
                                   &state->offsets[i]);
  }
  return iree_ok_status();
}

static iree_status_t loom_view_boundary_plan_coordinate(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, loom_value_id_t source,
    loom_view_boundary_coordinate_t* out_coordinate, bool* out_planned);

static bool loom_view_boundary_match_binary_selection(
    const loom_module_t* module, loom_value_id_t result_value_id,
    loom_op_t** out_op, loom_value_id_t* out_condition_value_id,
    loom_value_id_t* out_true_value_id, loom_value_id_t* out_false_value_id) {
  const loom_value_t* result_value = loom_module_value(module, result_value_id);
  if (loom_value_is_block_arg(result_value)) {
    return false;
  }
  loom_op_t* op = loom_value_def_op(result_value);
  const uint16_t result_index = loom_value_def_index(result_value);
  if (!op || op->result_count == 0 || result_index >= op->result_count) {
    return false;
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  loom_value_id_t condition_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t true_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t false_value_id = LOOM_VALUE_ID_INVALID;
  uint16_t condition_count = 0;
  uint32_t payload_count = 0;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_operand_role_t role = loom_op_operand_role_at(vtable, op, i);
    if (role == LOOM_OPERAND_ROLE_SELECT_CONDITION) {
      condition_value_id = operands[i];
      ++condition_count;
    } else if (role == LOOM_OPERAND_ROLE_SELECT_PAYLOAD) {
      const uint32_t payload_result_index = payload_count % op->result_count;
      const uint32_t payload_group_index = payload_count / op->result_count;
      if (payload_result_index == result_index) {
        if (payload_group_index == 0) {
          true_value_id = operands[i];
        } else if (payload_group_index == 1) {
          false_value_id = operands[i];
        }
      }
      ++payload_count;
    }
  }
  if (condition_count != 1 || payload_count != 2u * op->result_count ||
      true_value_id == LOOM_VALUE_ID_INVALID ||
      false_value_id == LOOM_VALUE_ID_INVALID ||
      !loom_type_equal(loom_module_value_type(module, condition_value_id),
                       loom_type_scalar(LOOM_SCALAR_TYPE_I1))) {
    return false;
  }
  *out_op = op;
  *out_condition_value_id = condition_value_id;
  *out_true_value_id = true_value_id;
  *out_false_value_id = false_value_id;
  return true;
}

static iree_status_t loom_view_boundary_plan_selection(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, loom_op_t* op,
    loom_value_id_t result_value_id, loom_value_id_t condition_value_id,
    loom_value_id_t true_value_id, loom_value_id_t false_value_id,
    loom_view_boundary_coordinate_t* out_coordinate, bool* out_planned) {
  loom_view_boundary_projection_function_state_t* state =
      loom_view_boundary_function_state(rule, plan, function);
  IREE_ASSERT(state != NULL);
  const loom_value_ordinal_t result_ordinal =
      loom_local_value_domain_ordinal(&function->domain, result_value_id);
  iree_host_size_t selection_index = state->selection_indices[result_ordinal];
  if (selection_index != IREE_HOST_SIZE_MAX) {
    const loom_view_boundary_selection_t* selection =
        &state->selections[selection_index];
    *out_planned = selection->selected && !selection->planning;
    if (*out_planned) {
      out_coordinate->selection = selection_index;
    }
    return iree_ok_status();
  }

  if (state->selection_count == state->selection_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, state->selection_count, state->selection_count + 1,
        sizeof(*state->selections), &state->selection_capacity,
        (void**)&state->selections));
  }
  selection_index = state->selection_count++;
  state->selection_indices[result_ordinal] = selection_index;
  state->selections[selection_index] = (loom_view_boundary_selection_t){
      .op = op,
      .result_value_id = result_value_id,
      .condition_value_id = condition_value_id,
      .true_value_id = true_value_id,
      .false_value_id = false_value_id,
      .buffer_value_id = LOOM_VALUE_ID_INVALID,
      .offset_value_id = LOOM_VALUE_ID_INVALID,
      .planning = true,
  };

  // Recursive planning may grow and relocate the selection array. Keep the
  // alternatives local until both recursive calls have completed.
  loom_view_boundary_coordinate_t true_coordinate;
  bool true_planned = false;
  IREE_RETURN_IF_ERROR(loom_view_boundary_plan_coordinate(
      rule, plan, function, true_value_id, &true_coordinate, &true_planned));
  loom_view_boundary_coordinate_t false_coordinate;
  bool false_planned = false;
  IREE_RETURN_IF_ERROR(loom_view_boundary_plan_coordinate(
      rule, plan, function, false_value_id, &false_coordinate, &false_planned));
  loom_view_boundary_selection_t* selection =
      &state->selections[selection_index];
  selection->true_coordinate = true_coordinate;
  selection->false_coordinate = false_coordinate;
  selection->planning = false;
  selection->selected = true_planned && false_planned;
  *out_planned = selection->selected;
  if (*out_planned) {
    out_coordinate->selection = selection_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_view_boundary_plan_coordinate(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, loom_value_id_t source,
    loom_view_boundary_coordinate_t* out_coordinate, bool* out_planned) {
  *out_coordinate = (loom_view_boundary_coordinate_t){
      .buffer_value_id = LOOM_VALUE_ID_INVALID,
      .region_id = LOOM_VIEW_REGION_ID_INVALID,
      .dependency = IREE_HOST_SIZE_MAX,
      .selection = IREE_HOST_SIZE_MAX,
  };
  *out_planned = false;
  const iree_host_size_t direct_candidate =
      loom_boundary_projection_slot_index(function, source);
  if (direct_candidate != IREE_HOST_SIZE_MAX &&
      function->candidates[direct_candidate].selected) {
    out_coordinate->dependency = direct_candidate;
    *out_planned = true;
    return iree_ok_status();
  }

  loom_view_boundary_projection_function_state_t* state =
      loom_view_boundary_function_state(rule, plan, function);
  IREE_ASSERT(state != NULL);
  const loom_view_region_t* region = NULL;
  if (!loom_view_region_table_try_lookup(&state->regions, source, &region)) {
    return iree_ok_status();
  }

  loom_op_t* selection_op = NULL;
  loom_value_id_t condition_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t true_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t false_value_id = LOOM_VALUE_ID_INVALID;
  if (loom_view_boundary_match_binary_selection(
          plan->module, region->base_view_value_id, &selection_op,
          &condition_value_id, &true_value_id, &false_value_id)) {
    if (!loom_symbolic_expr_is_linear(&region->projection_byte_offset)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_view_boundary_plan_selection(
        rule, plan, function, selection_op, region->base_view_value_id,
        condition_value_id, true_value_id, false_value_id, out_coordinate,
        out_planned));
    if (*out_planned && region->base_view_value_id != source) {
      // The region owns the complete displacement through subviews/refinements.
      // Retain it relative to the selected coordinate instead of recovering the
      // projection from the chain of defining operations.
      out_coordinate->region_id = region->region_id;
      state->offsets[region->region_id].expression =
          &region->projection_byte_offset;
    }
    return iree_ok_status();
  }
  const loom_view_boundary_offset_t* offset =
      &state->offsets[region->region_id];
  if (offset->value_id == LOOM_VALUE_ID_INVALID && !offset->expression) {
    return iree_ok_status();
  }
  out_coordinate->region_id = region->region_id;
  if (offset->dependency != IREE_HOST_SIZE_MAX) {
    out_coordinate->dependency = offset->dependency;
    *out_planned = true;
    return iree_ok_status();
  }

  loom_value_fact_view_reference_t reference;
  if (!loom_value_facts_query_view_reference(
          &function->facts->context,
          loom_value_fact_table_lookup(function->facts, source), &reference) ||
      reference.buffer_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  out_coordinate->buffer_value_id = reference.buffer_value_id;
  *out_planned = true;
  return iree_ok_status();
}

static iree_status_t loom_view_boundary_add_coordinate_dependencies(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_view_boundary_coordinate_t* coordinate,
    iree_host_size_t destination_index) {
  if (coordinate->dependency != IREE_HOST_SIZE_MAX) {
    IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
        plan, function, coordinate->dependency, destination_index,
        /*orders_realization=*/false));
  }
  if (coordinate->selection == IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }
  const loom_view_boundary_projection_function_state_t* state =
      loom_view_boundary_function_state(rule, plan, function);
  IREE_ASSERT(state != NULL);
  IREE_ASSERT_LT(coordinate->selection, state->selection_count);
  const loom_view_boundary_selection_t* selection =
      &state->selections[coordinate->selection];
  IREE_RETURN_IF_ERROR(loom_view_boundary_add_coordinate_dependencies(
      rule, plan, function, &selection->true_coordinate, destination_index));
  return loom_view_boundary_add_coordinate_dependencies(
      rule, plan, function, &selection->false_coordinate, destination_index);
}

static iree_status_t loom_view_boundary_plan_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* destination,
    const loom_boundary_projection_schema_t* schema,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_boundary_projection_source_t* out_source, bool* out_planned) {
  (void)destination;
  (void)boundary_op;
  IREE_ASSERT(schema->rule == rule);
  *out_source = (loom_boundary_projection_source_t){0};
  *out_planned = false;
  loom_view_boundary_coordinate_t* coordinate = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, 1, sizeof(*coordinate), (void**)&coordinate));
  *coordinate = (loom_view_boundary_coordinate_t){
      .buffer_value_id = LOOM_VALUE_ID_INVALID,
      .region_id = LOOM_VIEW_REGION_ID_INVALID,
      .dependency = IREE_HOST_SIZE_MAX,
      .selection = IREE_HOST_SIZE_MAX,
  };
  IREE_RETURN_IF_ERROR(loom_view_boundary_plan_coordinate(
      rule, plan, function, source_value_id, coordinate, out_planned));
  if (*out_planned) {
    out_source->rule = rule;
    out_source->rule_plan = coordinate;
    out_source->boundary_op = boundary_op;
  }
  if (*out_planned && destination != NULL) {
    const iree_host_size_t destination_index =
        loom_boundary_projection_slot_index(function, destination->value_id);
    IREE_ASSERT_NE(destination_index, IREE_HOST_SIZE_MAX);
    IREE_RETURN_IF_ERROR(loom_view_boundary_add_coordinate_dependencies(
        rule, plan, function, coordinate, destination_index));
  }
  return iree_ok_status();
}

static iree_status_t loom_view_boundary_materialize_offset(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_view_region_id_t region_id, loom_value_id_t* out_value) {
  loom_view_boundary_projection_function_state_t* state =
      loom_view_boundary_function_state(rule, plan, function);
  IREE_ASSERT(state != NULL);
  loom_view_boundary_offset_t* offset = &state->offsets[region_id];
  if (offset->value_id != LOOM_VALUE_ID_INVALID) {
    *out_value = offset->value_id;
    return iree_ok_status();
  }
  if (offset->dependency != IREE_HOST_SIZE_MAX) {
    const loom_boundary_projection_slot_t* candidate =
        &function->candidates[offset->dependency];
    IREE_ASSERT(candidate->component_value_ids[1] != LOOM_VALUE_ID_INVALID);
    offset->base_value_id = candidate->component_value_ids[1];
  }
  const loom_symbolic_expr_t* expression = offset->expression;
  IREE_ASSERT(expression != NULL);
  IREE_RETURN_IF_ERROR(loom_view_materialize_offset_expression(
      &plan->rewriter.builder, expression, offset->base_value_id,
      offset->anchor_value_id, &offset->value_id));
  *out_value = offset->value_id;
  return iree_ok_status();
}

static iree_status_t loom_view_boundary_materialize_coordinate(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_view_boundary_coordinate_t* coordinate,
    loom_value_id_t out_values[2]);

static iree_status_t loom_view_boundary_materialize_selection(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    iree_host_size_t selection_index, loom_value_id_t out_values[2]) {
  loom_view_boundary_projection_function_state_t* state =
      loom_view_boundary_function_state(rule, plan, function);
  IREE_ASSERT(state != NULL);
  loom_view_boundary_selection_t* selection =
      &state->selections[selection_index];
  if (selection->buffer_value_id != LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT(selection->offset_value_id != LOOM_VALUE_ID_INVALID);
    out_values[0] = selection->buffer_value_id;
    out_values[1] = selection->offset_value_id;
    return iree_ok_status();
  }

  loom_value_id_t true_values[2];
  IREE_RETURN_IF_ERROR(loom_view_boundary_materialize_coordinate(
      rule, plan, function, &selection->true_coordinate, true_values));
  loom_value_id_t false_values[2];
  IREE_RETURN_IF_ERROR(loom_view_boundary_materialize_coordinate(
      rule, plan, function, &selection->false_coordinate, false_values));

  const loom_type_t result_types[2] = {
      loom_module_value_type(plan->module, true_values[0]),
      loom_module_value_type(plan->module, true_values[1]),
  };
  IREE_ASSERT(loom_type_equal(
      result_types[0], loom_module_value_type(plan->module, false_values[0])));
  IREE_ASSERT(loom_type_equal(
      result_types[1], loom_module_value_type(plan->module, false_values[1])));
  loom_builder_t* builder = &plan->rewriter.builder;
  loom_builder_set_before(builder, selection->op);
  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
      selection->condition_value_id, result_types, IREE_ARRAYSIZE(result_types),
      /*tied_results=*/NULL,
      /*tied_result_count=*/0, selection->op->location, &if_op));
  loom_builder_ip_t saved =
      loom_builder_enter_region(builder, if_op, loom_scf_if_then_region(if_op));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, true_values, IREE_ARRAYSIZE(true_values),
                           selection->op->location, &yield_op));
  loom_builder_restore(builder, saved);
  saved =
      loom_builder_enter_region(builder, if_op, loom_scf_if_else_region(if_op));
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, false_values, IREE_ARRAYSIZE(false_values),
                           selection->op->location, &yield_op));
  loom_builder_restore(builder, saved);

  const loom_value_id_t source_value = selection->result_value_id;
  selection->buffer_value_id = loom_op_results(if_op)[0];
  selection->offset_value_id = loom_op_results(if_op)[1];
  IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
      &plan->rewriter, source_value, selection->buffer_value_id,
      IREE_SV("root")));
  IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
      &plan->rewriter, source_value, selection->offset_value_id,
      IREE_SV("offset")));
  out_values[0] = selection->buffer_value_id;
  out_values[1] = selection->offset_value_id;
  loom_boundary_projection_record(plan, rule, 1, 2);
  return iree_ok_status();
}

static iree_status_t loom_view_boundary_materialize_coordinate(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_view_boundary_coordinate_t* coordinate,
    loom_value_id_t out_values[2]) {
  if (coordinate->selection != IREE_HOST_SIZE_MAX) {
    IREE_RETURN_IF_ERROR(loom_view_boundary_materialize_selection(
        rule, plan, function, coordinate->selection, out_values));
    if (coordinate->region_id == LOOM_VIEW_REGION_ID_INVALID) {
      return iree_ok_status();
    }
    loom_view_boundary_projection_function_state_t* state =
        loom_view_boundary_function_state(rule, plan, function);
    state->offsets[coordinate->region_id].base_value_id = out_values[1];
  } else if (coordinate->dependency != IREE_HOST_SIZE_MAX) {
    const loom_boundary_projection_slot_t* candidate =
        &function->candidates[coordinate->dependency];
    IREE_ASSERT(candidate->component_value_ids[0] != LOOM_VALUE_ID_INVALID);
    IREE_ASSERT(candidate->component_value_ids[1] != LOOM_VALUE_ID_INVALID);
    out_values[0] = candidate->component_value_ids[0];
    if (coordinate->region_id == LOOM_VIEW_REGION_ID_INVALID) {
      out_values[1] = candidate->component_value_ids[1];
      return iree_ok_status();
    }
  } else {
    out_values[0] = coordinate->buffer_value_id;
  }
  return loom_view_boundary_materialize_offset(
      rule, plan, function, coordinate->region_id, &out_values[1]);
}

static iree_status_t loom_view_boundary_materialize_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_component_values) {
  IREE_ASSERT(rule == source->rule);
  IREE_ASSERT(source->rule_plan != NULL);
  return loom_view_boundary_materialize_coordinate(
      rule, plan, function,
      (const loom_view_boundary_coordinate_t*)source->rule_plan,
      out_component_values);
}

static iree_status_t loom_view_boundary_reconstruct(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location, loom_value_id_t* out_logical_value) {
  (void)function;
  IREE_ASSERT(rule == slot->schema.rule);
  IREE_ASSERT_EQ(slot->schema.component_count, 2);
  loom_op_t* view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(
      &plan->rewriter.builder, 0, slot->component_value_ids[0],
      slot->component_value_ids[1], 0, logical_type, location, &view_op));
  *out_logical_value = loom_buffer_view_result(view_op);
  loom_boundary_projection_record(plan, rule, 1, slot->schema.component_count);
  return iree_ok_status();
}

static const loom_boundary_projection_rule_t kViewBoundaryRule = {
    .name = IREE_SVL("view-buffer-offset"),
    .type_kind_bits = LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_VIEW),
    .slot_role_bits = LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                          LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_ARGUMENT) |
                      LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                          LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_RESULT) |
                      LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                          LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT) |
                      LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                          LOOM_BOUNDARY_PROJECTION_SLOT_CALL_RESULT),
    .function_applies = loom_view_boundary_function_applies,
    .initialize = loom_view_boundary_initialize_schema,
    .prepare_function = loom_view_boundary_prepare_function,
    .plan_slot = loom_view_boundary_plan_slot,
    .transport =
        {
            .plan_source = loom_view_boundary_plan_source,
            .materialize_source = loom_view_boundary_materialize_source,
            .reconstruct = loom_view_boundary_reconstruct,
        },
};

const loom_boundary_projection_rule_t* loom_view_boundary_projection_rule(
    void) {
  return &kViewBoundaryRule;
}
