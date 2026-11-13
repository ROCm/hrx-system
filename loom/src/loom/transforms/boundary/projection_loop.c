// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/boundary/projection_loop.h"

#include <string.h>

#include "loom/ir/ancestry.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/type_refs.h"
#include "loom/rewrite/loop_like.h"
#include "loom/rewrite/remap.h"

static bool loom_boundary_projection_loop_result_has_candidates(
    const loom_boundary_projection_loop_result_state_t* state) {
  return state->result_candidate != IREE_HOST_SIZE_MAX;
}

static void loom_boundary_projection_reject_loop_result(
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_result_state_t* state) {
  if (state->result_candidate != IREE_HOST_SIZE_MAX) {
    function->candidates[state->result_candidate].selected = false;
  }
  if (state->body_candidate != IREE_HOST_SIZE_MAX) {
    function->candidates[state->body_candidate].selected = false;
  }
}

static bool loom_boundary_projection_loop_result_is_selected(
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_result_state_t* state) {
  if (!loom_boundary_projection_loop_result_has_candidates(state)) {
    return false;
  }
  const bool result_selected =
      function->candidates[state->result_candidate].selected;
  const bool body_selected =
      function->candidates[state->body_candidate].selected;
  return result_selected && body_selected;
}

static void loom_boundary_projection_reject_loop_header(
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_header_state_t* state) {
  if (state->candidate != IREE_HOST_SIZE_MAX) {
    function->candidates[state->candidate].selected = false;
  }
}

static bool loom_boundary_projection_loop_header_is_selected(
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_header_state_t* state) {
  return state->candidate != IREE_HOST_SIZE_MAX &&
         function->candidates[state->candidate].selected;
}

static iree_status_t loom_boundary_projection_add_loop_endpoint(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, loom_value_id_t value_id,
    loom_block_t* block) {
  return loom_boundary_projection_add_provisional_slot(
      plan, function, value_id, LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE,
      block);
}

iree_status_t loom_boundary_projection_collect_loop(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, loom_loop_like_t loop) {
  ++plan->loops_checked;
  const loom_value_slice_t initial_values = loom_loop_like_iter_args(loop);
  const uint16_t result_count = loop.op->result_count;
  loom_region_t* condition_region = loom_loop_like_condition_region(loop);
  const bool condition_loop = condition_region != NULL;
  if (!condition_loop) {
    IREE_ASSERT_EQ(result_count, initial_values.count);
  }

  bool* result_claimable = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, result_count,
                                                   sizeof(*result_claimable),
                                                   (void**)&result_claimable));
  }
  bool any_claimable = false;
  for (uint16_t i = 0; i < result_count; ++i) {
    result_claimable[i] = loom_boundary_projection_may_claim_slot(
        plan, function, LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE,
        loom_op_const_results(loop.op)[i], /*block=*/NULL);
    any_claimable |= result_claimable[i];
  }

  loom_block_t* condition_entry = NULL;
  bool* header_claimable = NULL;
  if (condition_loop) {
    condition_entry = loom_region_entry_block(condition_region);
    IREE_ASSERT_EQ(condition_entry->arg_count, initial_values.count);
    if (initial_values.count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          plan->arena, initial_values.count, sizeof(*header_claimable),
          (void**)&header_claimable));
    }
    for (uint16_t i = 0; i < initial_values.count; ++i) {
      header_claimable[i] = loom_boundary_projection_may_claim_slot(
          plan, function, LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE,
          loom_block_arg_id(condition_entry, i), condition_entry);
      any_claimable |= header_claimable[i];
    }
  }

  if (!any_claimable) {
    return iree_ok_status();
  }

  if (function->loop_count == function->loop_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, function->loop_count, function->loop_count + 1,
        sizeof(*function->loops), &function->loop_capacity,
        (void**)&function->loops));
  }
  loom_boundary_projection_loop_t* loop_plan =
      &function->loops[function->loop_count++];
  *loop_plan = (loom_boundary_projection_loop_t){
      .loop = loop,
      .result_count = result_count,
      .header_count = condition_loop ? initial_values.count : 0,
  };
  if (loop_plan->result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, loop_plan->result_count, sizeof(*loop_plan->result_states),
        (void**)&loop_plan->result_states));
  }
  if (loop_plan->header_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, loop_plan->header_count, sizeof(*loop_plan->header_states),
        (void**)&loop_plan->header_states));
  }

  loom_region_t* body_region = loom_loop_like_body(loop);
  loom_block_t* body_entry = loom_region_entry_block(body_region);
  loop_plan->body_terminator = body_entry->last_op;
  const uint16_t body_state_offset =
      loom_loop_like_iv(loop) == LOOM_VALUE_ID_INVALID ? 0 : 1;
  IREE_ASSERT_EQ(body_entry->arg_count,
                 (uint32_t)body_state_offset + loop_plan->result_count);

  if (condition_loop) {
    loop_plan->condition_terminator = condition_entry->last_op;
  }

  for (uint16_t i = 0; i < loop_plan->result_count; ++i) {
    loom_boundary_projection_loop_result_state_t* state =
        &loop_plan->result_states[i];
    *state = (loom_boundary_projection_loop_result_state_t){
        .result_value_id = loom_op_const_results(loop.op)[i],
        .body_value_id =
            loom_block_arg_id(body_entry, (uint16_t)(body_state_offset + i)),
        .result_candidate = IREE_HOST_SIZE_MAX,
        .body_candidate = IREE_HOST_SIZE_MAX,
    };
    if (!result_claimable[i]) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_boundary_projection_add_loop_endpoint(
        plan, function, state->result_value_id, /*block=*/NULL));
    IREE_RETURN_IF_ERROR(loom_boundary_projection_add_loop_endpoint(
        plan, function, state->body_value_id, body_entry));
  }
  for (uint16_t i = 0; i < loop_plan->header_count; ++i) {
    loom_boundary_projection_loop_header_state_t* state =
        &loop_plan->header_states[i];
    *state = (loom_boundary_projection_loop_header_state_t){
        .value_id = loom_block_arg_id(condition_entry, i),
        .candidate = IREE_HOST_SIZE_MAX,
    };
    if (header_claimable[i]) {
      IREE_RETURN_IF_ERROR(loom_boundary_projection_add_loop_endpoint(
          plan, function, state->value_id, condition_entry));
    }
  }
  return iree_ok_status();
}

void loom_boundary_projection_index_loops(
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t loop_index = 0; loop_index < function->loop_count;
       ++loop_index) {
    loom_boundary_projection_loop_t* loop = &function->loops[loop_index];
    for (uint16_t i = 0; i < loop->result_count; ++i) {
      loom_boundary_projection_loop_result_state_t* state =
          &loop->result_states[i];
      const iree_host_size_t result_candidate =
          loom_boundary_projection_slot_index(function, state->result_value_id);
      if (result_candidate == IREE_HOST_SIZE_MAX) {
        continue;
      }
      state->result_candidate = result_candidate;
      state->body_candidate =
          loom_boundary_projection_slot_index(function, state->body_value_id);
      IREE_ASSERT_NE(state->body_candidate, IREE_HOST_SIZE_MAX);
    }
    for (uint16_t i = 0; i < loop->header_count; ++i) {
      loom_boundary_projection_loop_header_state_t* state =
          &loop->header_states[i];
      const iree_host_size_t candidate =
          loom_boundary_projection_slot_index(function, state->value_id);
      if (candidate != IREE_HOST_SIZE_MAX) {
        state->candidate = candidate;
      }
    }
  }
}

static iree_status_t loom_boundary_projection_couple_loop_endpoint(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, iree_host_size_t lhs,
    iree_host_size_t rhs) {
  IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
      plan, function, lhs, rhs, /*orders_realization=*/false));
  return loom_boundary_projection_add_dependency(plan, function, rhs, lhs,
                                                 /*orders_realization=*/false);
}

static bool loom_boundary_projection_loop_defines_endpoint(
    const loom_module_t* module, const loom_boundary_projection_loop_t* loop,
    loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (!loom_value_is_block_arg(value)) {
    return loom_value_def_op(value) == loop->loop.op;
  }
  const loom_block_t* block = loom_value_def_block(value);
  for (uint8_t i = 0; i < loop->loop.op->region_count; ++i) {
    if (block->parent_region == loom_op_regions(loop->loop.op)[i]) {
      return true;
    }
  }
  return false;
}

static bool loom_boundary_projection_loop_endpoint_is_type_provider(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_loop_t* loop, loom_value_id_t value_id) {
  loom_type_use_iterator_t users;
  loom_module_value_type_users(plan->module, value_id, &users);
  for (loom_value_id_t user = loom_type_users_next(&users);
       user != LOOM_VALUE_ID_INVALID; user = loom_type_users_next(&users)) {
    if (loom_boundary_projection_loop_defines_endpoint(plan->module, loop,
                                                       user)) {
      return true;
    }
  }
  return false;
}

static bool loom_boundary_projection_terminator_supports_state_rebuild(
    const loom_module_t* module, const loom_op_t* terminator,
    uint16_t state_operand_offset, uint16_t state_count) {
  if (!terminator ||
      terminator->operand_count != state_operand_offset + state_count ||
      terminator->result_count != 0 || terminator->successor_count != 0 ||
      terminator->region_count != 0 || terminator->tied_result_count != 0) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, terminator);
  const uint8_t segment_count = loom_op_vtable_operand_segment_count(vtable);
  if (segment_count == 0) {
    return true;
  }
  const uint16_t* segment_counts =
      loom_op_const_operand_segment_counts(terminator);
  uint16_t offset = 0;
  for (uint8_t i = 0; i < segment_count; ++i) {
    if (offset == state_operand_offset && segment_counts[i] == state_count) {
      for (uint8_t j = (uint8_t)(i + 1); j < segment_count; ++j) {
        if (segment_counts[j] != 0) {
          return false;
        }
      }
      return true;
    }
    offset = (uint16_t)(offset + segment_counts[i]);
  }
  return false;
}

static void loom_boundary_projection_reject_loop(
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop) {
  for (uint16_t i = 0; i < loop->result_count; ++i) {
    loom_boundary_projection_reject_loop_result(function,
                                                &loop->result_states[i]);
  }
  for (uint16_t i = 0; i < loop->header_count; ++i) {
    loom_boundary_projection_reject_loop_header(function,
                                                &loop->header_states[i]);
  }
}

static iree_status_t loom_boundary_projection_finalize_loop_result_schema(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_loop_t* loop,
    loom_boundary_projection_loop_result_state_t* state) {
  loom_boundary_projection_schema_t schema = {0};
  bool claimed = false;
  IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_slot_schema(
      plan, function, LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE,
      state->result_value_id, /*block=*/NULL, &schema, &claimed));
  if (!claimed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_boundary_projection_finalize_provisional_slot(
      plan, function, state->result_candidate, &schema));
  IREE_RETURN_IF_ERROR(loom_boundary_projection_finalize_provisional_slot(
      plan, function, state->body_candidate, &schema));
  IREE_RETURN_IF_ERROR(loom_boundary_projection_couple_loop_endpoint(
      plan, function, state->result_candidate, state->body_candidate));

  const loom_value_id_t endpoints[] = {
      state->result_value_id,
      state->body_value_id,
  };
  // All recurrence endpoint types are established before logical values are
  // reconstructed. Keep a column intact when another endpoint type needs it.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(endpoints); ++i) {
    if (endpoints[i] != LOOM_VALUE_ID_INVALID &&
        loom_boundary_projection_loop_endpoint_is_type_provider(plan, loop,
                                                                endpoints[i])) {
      loom_boundary_projection_reject_loop_result(function, state);
      break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_finalize_loop_header_schema(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_loop_t* loop,
    loom_boundary_projection_loop_header_state_t* state) {
  loom_boundary_projection_slot_t* candidate =
      &function->candidates[state->candidate];
  loom_boundary_projection_schema_t schema = {0};
  bool claimed = false;
  IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_slot_schema(
      plan, function, LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE, state->value_id,
      candidate->block, &schema, &claimed));
  if (!claimed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_boundary_projection_finalize_provisional_slot(
      plan, function, state->candidate, &schema));
  if (loom_boundary_projection_loop_endpoint_is_type_provider(
          plan, loop, state->value_id)) {
    loom_boundary_projection_reject_loop_header(function, state);
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_plan_loop_source(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    iree_host_size_t destination_candidate, loom_value_id_t source_value_id,
    loom_op_t* boundary_op, loom_boundary_projection_source_t* out_source,
    bool* out_planned) {
  const loom_boundary_projection_slot_t* destination =
      &function->candidates[destination_candidate];
  return destination->schema.rule->transport.plan_source(
      destination->schema.rule, plan, function, destination,
      &destination->schema, source_value_id, boundary_op, out_source,
      out_planned);
}

iree_status_t loom_boundary_projection_select_loops(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t loop_index = 0; loop_index < function->loop_count;
       ++loop_index) {
    loom_boundary_projection_loop_t* loop = &function->loops[loop_index];
    for (uint16_t i = 0; i < loop->result_count; ++i) {
      loom_boundary_projection_loop_result_state_t* state =
          &loop->result_states[i];
      if (loom_boundary_projection_loop_result_has_candidates(state)) {
        IREE_RETURN_IF_ERROR(
            loom_boundary_projection_finalize_loop_result_schema(plan, function,
                                                                 loop, state));
      }
    }
    for (uint16_t i = 0; i < loop->header_count; ++i) {
      loom_boundary_projection_loop_header_state_t* state =
          &loop->header_states[i];
      if (state->candidate != IREE_HOST_SIZE_MAX) {
        IREE_RETURN_IF_ERROR(
            loom_boundary_projection_finalize_loop_header_schema(plan, function,
                                                                 loop, state));
      }
    }
  }

  return iree_ok_status();
}

iree_status_t loom_boundary_projection_plan_loops(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t loop_index = 0; loop_index < function->loop_count;
       ++loop_index) {
    loom_boundary_projection_loop_t* loop = &function->loops[loop_index];
    const bool condition_loop = loop->condition_terminator != NULL;
    const uint16_t backedge_count =
        condition_loop ? loop->header_count : loop->result_count;
    if (!loom_boundary_projection_terminator_supports_state_rebuild(
            plan->module, loop->body_terminator, /*state_operand_offset=*/0,
            backedge_count) ||
        (condition_loop &&
         !loom_boundary_projection_terminator_supports_state_rebuild(
             plan->module, loop->condition_terminator,
             /*state_operand_offset=*/1, loop->result_count))) {
      loom_boundary_projection_reject_loop(function, loop);
      continue;
    }

    const loom_value_slice_t initial_values =
        loom_loop_like_iter_args(loop->loop);
    const loom_value_id_t* backedge_values =
        loom_op_const_operands(loop->body_terminator);
    const loom_value_id_t* condition_values =
        condition_loop ? loom_op_const_operands(loop->condition_terminator)
                       : NULL;
    for (uint16_t i = 0; i < loop->result_count; ++i) {
      loom_boundary_projection_loop_result_state_t* state =
          &loop->result_states[i];
      if (!loom_boundary_projection_loop_result_is_selected(function, state)) {
        loom_boundary_projection_reject_loop_result(function, state);
        continue;
      }
      bool planned = false;
      if (condition_loop) {
        IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_loop_source(
            plan, function, state->body_candidate, condition_values[1 + i],
            loop->condition_terminator, &state->condition_source, &planned));
      } else {
        IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_loop_source(
            plan, function, state->body_candidate, initial_values.values[i],
            loop->loop.op, &state->initial_source, &planned));
        if (planned) {
          IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_loop_source(
              plan, function, state->body_candidate, backedge_values[i],
              loop->body_terminator, &state->backedge_source, &planned));
        }
      }
      if (!planned) {
        loom_boundary_projection_reject_loop_result(function, state);
      }
    }
    for (uint16_t i = 0; i < loop->header_count; ++i) {
      loom_boundary_projection_loop_header_state_t* state =
          &loop->header_states[i];
      if (!loom_boundary_projection_loop_header_is_selected(function, state)) {
        loom_boundary_projection_reject_loop_header(function, state);
        continue;
      }
      bool planned = false;
      IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_loop_source(
          plan, function, state->candidate, initial_values.values[i],
          loop->loop.op, &state->initial_source, &planned));
      if (planned) {
        IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_loop_source(
            plan, function, state->candidate, backedge_values[i],
            loop->body_terminator, &state->backedge_source, &planned));
      }
      if (!planned) {
        loom_boundary_projection_reject_loop_header(function, state);
      }
    }
  }
  return iree_ok_status();
}

static uint16_t loom_boundary_projection_loop_result_physical_count(
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_result_state_t* state) {
  return loom_boundary_projection_loop_result_is_selected(function, state)
             ? function->candidates[state->result_candidate]
                   .schema.component_count
             : 1;
}

static uint16_t loom_boundary_projection_loop_header_physical_count(
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_header_state_t* state) {
  return loom_boundary_projection_loop_header_is_selected(function, state)
             ? function->candidates[state->candidate].schema.component_count
             : 1;
}

static bool loom_boundary_projection_loop_ties_are_supported(
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop) {
  if (loop->loop.op->tied_result_count == 0) {
    return true;
  }
  const loom_value_slice_t initial = loom_loop_like_iter_args(loop->loop);
  const uint16_t state_operand_offset =
      (uint16_t)(initial.values - loom_op_const_operands(loop->loop.op));
  const loom_tied_result_t* ties = loom_op_tied_results(loop->loop.op);
  uint32_t final_tie_count = 0;
  for (uint16_t i = 0; i < loop->loop.op->tied_result_count; ++i) {
    const loom_tied_result_t* tie = &ties[i];
    IREE_ASSERT_LT(tie->result_index, loop->result_count);
    const uint16_t result_count =
        loom_boundary_projection_loop_result_physical_count(
            function, &loop->result_states[tie->result_index]);
    uint16_t operand_count = 1;
    if (tie->operand_index >= state_operand_offset &&
        tie->operand_index < state_operand_offset + initial.count) {
      const uint16_t state_index =
          (uint16_t)(tie->operand_index - state_operand_offset);
      operand_count = loop->condition_terminator
                          ? loom_boundary_projection_loop_header_physical_count(
                                function, &loop->header_states[state_index])
                          : loom_boundary_projection_loop_result_physical_count(
                                function, &loop->result_states[state_index]);
    }
    if (result_count != operand_count) {
      return false;
    }
    final_tie_count += result_count;
  }
  return final_tie_count <= UINT16_MAX;
}

static bool loom_boundary_projection_type_references_projected_loop_endpoint(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop, loom_type_t type) {
  for (uint16_t i = 0; i < loop->result_count; ++i) {
    const loom_boundary_projection_loop_result_state_t* state =
        &loop->result_states[i];
    if (!loom_boundary_projection_loop_result_is_selected(function, state)) {
      continue;
    }
    if (loom_type_references_value(plan->module, type,
                                   state->result_value_id) ||
        loom_type_references_value(plan->module, type, state->body_value_id)) {
      return true;
    }
  }
  for (uint16_t i = 0; i < loop->header_count; ++i) {
    const loom_boundary_projection_loop_header_state_t* state =
        &loop->header_states[i];
    if (loom_boundary_projection_loop_header_is_selected(function, state) &&
        loom_type_references_value(plan->module, type, state->value_id)) {
      return true;
    }
  }
  return false;
}

void loom_boundary_projection_preflight_loops(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t loop_index = 0; loop_index < function->loop_count;
       ++loop_index) {
    loom_boundary_projection_loop_t* loop = &function->loops[loop_index];
    uint32_t final_result_count = 0;
    for (uint16_t i = 0; i < loop->result_count; ++i) {
      loom_boundary_projection_loop_result_state_t* state =
          &loop->result_states[i];
      if (!loom_boundary_projection_loop_result_is_selected(function, state)) {
        loom_boundary_projection_reject_loop_result(function, state);
        ++final_result_count;
        continue;
      }
      const loom_boundary_projection_slot_t* candidate =
          &function->candidates[state->result_candidate];
      bool references_projected_state = false;
      for (uint16_t component = 0;
           component < candidate->schema.component_count; ++component) {
        const loom_type_t type = candidate->schema.component_types[component];
        references_projected_state |=
            loom_boundary_projection_type_references_projected_loop_endpoint(
                plan, function, loop, type);
      }
      if (references_projected_state) {
        loom_boundary_projection_reject_loop_result(function, state);
        ++final_result_count;
      } else {
        final_result_count += candidate->schema.component_count;
      }
    }

    uint32_t final_header_count = 0;
    for (uint16_t i = 0; i < loop->header_count; ++i) {
      loom_boundary_projection_loop_header_state_t* state =
          &loop->header_states[i];
      if (!loom_boundary_projection_loop_header_is_selected(function, state)) {
        loom_boundary_projection_reject_loop_header(function, state);
        ++final_header_count;
        continue;
      }
      const loom_boundary_projection_slot_t* candidate =
          &function->candidates[state->candidate];
      bool references_projected_state = false;
      for (uint16_t component = 0;
           component < candidate->schema.component_count; ++component) {
        references_projected_state |=
            loom_boundary_projection_type_references_projected_loop_endpoint(
                plan, function, loop,
                candidate->schema.component_types[component]);
      }
      if (references_projected_state) {
        loom_boundary_projection_reject_loop_header(function, state);
        ++final_header_count;
      } else {
        final_header_count += candidate->schema.component_count;
      }
    }
    const bool condition_loop = loop->condition_terminator != NULL;
    if (!condition_loop) {
      final_header_count = final_result_count;
    }
    const uint32_t source_header_count =
        loom_loop_like_iter_args(loop->loop).count;
    const uint32_t target_operand_count =
        (uint32_t)loop->loop.op->operand_count - source_header_count +
        final_header_count;
    const uint32_t body_argument_count =
        final_result_count +
        (loom_loop_like_iv(loop->loop) == LOOM_VALUE_ID_INVALID ? 0u : 1u);
    const uint32_t condition_operand_count =
        condition_loop ? final_result_count + 1u : 0u;
    const uint32_t body_terminator_operand_count =
        condition_loop ? final_header_count : final_result_count;
    if (final_result_count > UINT16_MAX || final_header_count > UINT16_MAX ||
        target_operand_count > UINT16_MAX || body_argument_count > UINT16_MAX ||
        condition_operand_count > UINT16_MAX ||
        body_terminator_operand_count > UINT16_MAX ||
        !loom_boundary_projection_loop_ties_are_supported(function, loop)) {
      loom_boundary_projection_reject_loop(function, loop);
    }
  }
}

iree_status_t loom_boundary_projection_finalize_loops(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t loop_index = 0; loop_index < function->loop_count;
       ++loop_index) {
    loom_boundary_projection_loop_t* loop = &function->loops[loop_index];
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, (iree_host_size_t)loop->result_count + 1,
        sizeof(*loop->result_offsets), (void**)&loop->result_offsets));
    uint32_t final_result_count = 0;
    loop->result_offsets[0] = 0;
    loop->selected = false;
    for (uint16_t i = 0; i < loop->result_count; ++i) {
      const loom_boundary_projection_loop_result_state_t* state =
          &loop->result_states[i];
      const bool selected =
          loom_boundary_projection_loop_result_is_selected(function, state);
      loop->selected |= selected;
      final_result_count +=
          loom_boundary_projection_loop_result_physical_count(function, state);
      IREE_ASSERT_LE(final_result_count, UINT16_MAX);
      loop->result_offsets[i + 1] = (uint16_t)final_result_count;
    }
    loop->final_result_count = (uint16_t)final_result_count;

    if (!loop->condition_terminator) {
      loop->header_offsets = loop->result_offsets;
      loop->final_header_count = loop->final_result_count;
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, (iree_host_size_t)loop->header_count + 1,
        sizeof(*loop->header_offsets), (void**)&loop->header_offsets));
    uint32_t final_header_count = 0;
    loop->header_offsets[0] = 0;
    for (uint16_t i = 0; i < loop->header_count; ++i) {
      const loom_boundary_projection_loop_header_state_t* state =
          &loop->header_states[i];
      const bool selected =
          loom_boundary_projection_loop_header_is_selected(function, state);
      loop->selected |= selected;
      final_header_count +=
          loom_boundary_projection_loop_header_physical_count(function, state);
      IREE_ASSERT_LE(final_header_count, UINT16_MAX);
      loop->header_offsets[i + 1] = (uint16_t)final_header_count;
    }
    loop->final_header_count = (uint16_t)final_header_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_loop_initialize_remap(
    loom_boundary_projection_plan_t* plan, loom_ir_remap_t* out_remap) {
  return loom_ir_remap_initialize(
      plan->module, plan->module, plan->arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
      },
      out_remap);
}

static iree_status_t loom_boundary_projection_materialize_loop_source(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* candidate,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_values) {
  IREE_ASSERT_EQ(source->rule, candidate->schema.rule);
  return candidate->schema.rule->transport.materialize_source(
      candidate->schema.rule, plan, function, source, out_values);
}

static iree_status_t loom_boundary_projection_materialize_initial_state(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop,
    loom_value_id_t* out_initial_values) {
  const loom_value_slice_t source_initial =
      loom_loop_like_iter_args(loop->loop);
  loom_builder_set_before(&plan->rewriter.builder, loop->loop.op);
  if (loop->condition_terminator) {
    for (uint16_t i = 0; i < loop->header_count; ++i) {
      const loom_boundary_projection_loop_header_state_t* state =
          &loop->header_states[i];
      const uint16_t offset = loop->header_offsets[i];
      if (!loom_boundary_projection_loop_header_is_selected(function, state)) {
        out_initial_values[offset] = source_initial.values[i];
        continue;
      }
      const loom_boundary_projection_slot_t* candidate =
          &function->candidates[state->candidate];
      IREE_RETURN_IF_ERROR(loom_boundary_projection_materialize_loop_source(
          plan, function, candidate, &state->initial_source,
          out_initial_values ? out_initial_values + offset : NULL));
    }
  } else {
    for (uint16_t i = 0; i < loop->result_count; ++i) {
      const loom_boundary_projection_loop_result_state_t* state =
          &loop->result_states[i];
      const uint16_t offset = loop->result_offsets[i];
      if (!loom_boundary_projection_loop_result_is_selected(function, state)) {
        out_initial_values[offset] = source_initial.values[i];
        continue;
      }
      const loom_boundary_projection_slot_t* candidate =
          &function->candidates[state->result_candidate];
      IREE_RETURN_IF_ERROR(loom_boundary_projection_materialize_loop_source(
          plan, function, candidate, &state->initial_source,
          out_initial_values ? out_initial_values + offset : NULL));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_build_loop_result_types(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop,
    const loom_value_id_t* reserved_results, loom_type_t* out_result_types) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_loop_initialize_remap(plan, &remap));
  for (uint16_t i = 0; i < loop->result_count; ++i) {
    const loom_boundary_projection_loop_result_state_t* state =
        &loop->result_states[i];
    if (!loom_boundary_projection_loop_result_is_selected(function, state)) {
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_map_value(&remap, state->result_value_id,
                                  reserved_results[loop->result_offsets[i]]));
    }
  }
  for (uint16_t i = 0; i < loop->result_count; ++i) {
    const loom_boundary_projection_loop_result_state_t* state =
        &loop->result_states[i];
    uint16_t target = loop->result_offsets[i];
    if (!loom_boundary_projection_loop_result_is_selected(function, state)) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_type(
          &remap, loom_module_value_type(plan->module, state->result_value_id),
          &out_result_types[target]));
      continue;
    }
    const loom_boundary_projection_schema_t* schema =
        &function->candidates[state->result_candidate].schema;
    for (uint16_t component = 0; component < schema->component_count;
         ++component) {
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_type(&remap, schema->component_types[component],
                             &out_result_types[target++]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_build_loop_header_types(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop,
    const loom_value_id_t* reserved_headers, loom_type_t* out_header_types) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_loop_initialize_remap(plan, &remap));
  for (uint16_t i = 0; i < loop->header_count; ++i) {
    const loom_boundary_projection_loop_header_state_t* state =
        &loop->header_states[i];
    if (!loom_boundary_projection_loop_header_is_selected(function, state)) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
          &remap, state->value_id, reserved_headers[loop->header_offsets[i]]));
    }
  }
  for (uint16_t i = 0; i < loop->header_count; ++i) {
    const loom_boundary_projection_loop_header_state_t* state =
        &loop->header_states[i];
    uint16_t target = loop->header_offsets[i];
    if (!loom_boundary_projection_loop_header_is_selected(function, state)) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_type(
          &remap, loom_module_value_type(plan->module, state->value_id),
          &out_header_types[target]));
      continue;
    }
    const loom_boundary_projection_schema_t* schema =
        &function->candidates[state->candidate].schema;
    for (uint16_t component = 0; component < schema->component_count;
         ++component) {
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_type(&remap, schema->component_types[component],
                             &out_header_types[target++]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_name_loop_components(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_slot_t* candidate) {
  const loom_boundary_projection_schema_t* schema = &candidate->schema;
  if (!schema->component_name_suffixes ||
      !iree_any_bit_set(plan->rewriter.name_policy,
                        LOOM_REWRITER_NAME_POLICY_DERIVE_DEBUG_NAMES)) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < schema->component_count; ++i) {
    // Loop replacement preserves a source name for one-to-one endpoints. A
    // projected endpoint is a physical component, so replace that copied name
    // with the component spelling supplied by the projection rule.
    IREE_RETURN_IF_ERROR(loom_rewriter_clear_value_name(
        &plan->rewriter, candidate->component_value_ids[i]));
    IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
        &plan->rewriter, candidate->value_id, candidate->component_value_ids[i],
        schema->component_name_suffixes[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_bind_loop_endpoint(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, iree_host_size_t candidate,
    const loom_value_id_t* values, uint16_t value_count) {
  if (candidate == IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }
  loom_boundary_projection_slot_t* slot = &function->candidates[candidate];
  if (!slot->selected) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(slot->schema.component_count, value_count);
  if (value_count != 0) {
    memcpy(slot->component_value_ids, values,
           value_count * sizeof(*slot->component_value_ids));
  }
  return loom_boundary_projection_name_loop_components(plan, slot);
}

static iree_status_t loom_boundary_projection_bind_loop_components(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop,
    const loom_loop_like_replacement_t* replacement) {
  for (uint16_t i = 0; i < loop->result_count; ++i) {
    const loom_boundary_projection_loop_result_state_t* state =
        &loop->result_states[i];
    if (!loom_boundary_projection_loop_result_is_selected(function, state)) {
      continue;
    }
    const uint16_t offset = loop->result_offsets[i];
    const uint16_t count = (uint16_t)(loop->result_offsets[i + 1] - offset);
    IREE_RETURN_IF_ERROR(loom_boundary_projection_bind_loop_endpoint(
        plan, function, state->result_candidate,
        replacement->results.values ? replacement->results.values + offset
                                    : NULL,
        count));
    IREE_RETURN_IF_ERROR(loom_boundary_projection_bind_loop_endpoint(
        plan, function, state->body_candidate,
        replacement->body_state.values ? replacement->body_state.values + offset
                                       : NULL,
        count));
  }
  for (uint16_t i = 0; i < loop->header_count; ++i) {
    const loom_boundary_projection_loop_header_state_t* state =
        &loop->header_states[i];
    if (!loom_boundary_projection_loop_header_is_selected(function, state)) {
      continue;
    }
    const uint16_t offset = loop->header_offsets[i];
    const uint16_t count = (uint16_t)(loop->header_offsets[i + 1] - offset);
    IREE_RETURN_IF_ERROR(loom_boundary_projection_bind_loop_endpoint(
        plan, function, state->candidate,
        replacement->condition_state.values
            ? replacement->condition_state.values + offset
            : NULL,
        count));
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_copy_loop_comments(
    loom_boundary_projection_plan_t* plan, const loom_op_t* source,
    const loom_op_t* target) {
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(plan->module, source, &comment_count);
  return comment_count == 0
             ? iree_ok_status()
             : loom_module_attach_op_comments(plan->module, target, comments,
                                              comment_count);
}

static iree_status_t loom_boundary_projection_loop_terminator_segments(
    loom_boundary_projection_plan_t* plan, const loom_op_t* source,
    uint16_t state_operand_offset, uint16_t target_state_count,
    const uint16_t** out_segment_counts, uint8_t* out_segment_count) {
  const loom_op_vtable_t* vtable = loom_op_vtable(plan->module, source);
  *out_segment_count = loom_op_vtable_operand_segment_count(vtable);
  *out_segment_counts = NULL;
  if (*out_segment_count == 0) {
    return iree_ok_status();
  }
  uint16_t* segment_counts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, *out_segment_count, sizeof(*segment_counts),
      (void**)&segment_counts));
  memcpy(segment_counts, loom_op_const_operand_segment_counts(source),
         *out_segment_count * sizeof(*segment_counts));
  uint16_t offset = 0;
  bool replaced = false;
  for (uint8_t i = 0; i < *out_segment_count; ++i) {
    if (offset == state_operand_offset) {
      segment_counts[i] = target_state_count;
      replaced = true;
      break;
    }
    offset = (uint16_t)(offset + segment_counts[i]);
  }
  IREE_ASSERT(replaced);
  *out_segment_counts = segment_counts;
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_build_loop_terminator(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function, loom_op_t* parent,
    loom_region_t* target_region, const loom_op_t* source,
    const loom_value_id_t* source_operands, uint16_t state_operand_offset,
    const loom_value_id_t* state_values, uint16_t state_count,
    loom_op_t** out_terminator) {
  const uint16_t operand_count = (uint16_t)(state_operand_offset + state_count);
  const uint16_t* segment_counts = NULL;
  uint8_t segment_count = 0;
  IREE_RETURN_IF_ERROR(loom_boundary_projection_loop_terminator_segments(
      plan, source, state_operand_offset, state_count, &segment_counts,
      &segment_count));

  loom_builder_t* builder = &plan->rewriter.builder;
  const loom_builder_ip_t saved =
      loom_builder_enter_region(builder, parent, target_region);
  iree_status_t status = iree_ok_status();
  if (segment_count != 0) {
    status = loom_builder_allocate_segmented_op(
        builder, source->kind, operand_count, segment_counts, segment_count,
        /*result_count=*/0, /*region_count=*/0, /*tied_result_count=*/0,
        source->attribute_count, source->location, out_terminator);
  } else {
    status = loom_builder_allocate_op(
        builder, source->kind, operand_count, /*result_count=*/0,
        /*region_count=*/0, /*tied_result_count=*/0, source->attribute_count,
        source->location, out_terminator);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* target = *out_terminator;
    target->instance_flags = source->instance_flags;
    target->traits = source->traits;
    target->intrinsic_read_spaces = source->intrinsic_read_spaces;
    target->intrinsic_write_spaces = source->intrinsic_write_spaces;
    target->flags |= source->flags & LOOM_OP_SOURCE_PRESENTATION_FLAG_MASK;
    for (uint16_t i = 0; i < state_operand_offset; ++i) {
      loom_op_operands(target)[i] =
          loom_boundary_projection_resolve_value(function, source_operands[i]);
    }
    if (state_count != 0) {
      memcpy(loom_op_operands(target) + state_operand_offset, state_values,
             state_count * sizeof(loom_value_id_t));
    }
    if (source->attribute_count != 0) {
      memcpy(loom_op_attrs(target), loom_op_const_attrs(source),
             source->attribute_count * sizeof(loom_attribute_t));
    }
    status = loom_builder_finalize_op(builder, target);
  }
  loom_builder_restore(builder, saved);
  IREE_RETURN_IF_ERROR(status);
  return loom_boundary_projection_copy_loop_comments(plan, source,
                                                     *out_terminator);
}

static iree_status_t loom_boundary_projection_build_loop_terminators(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop,
    const loom_loop_like_replacement_t* replacement,
    loom_op_t** out_condition_terminator, loom_op_t** out_body_terminator) {
  loom_value_id_t* state_values = NULL;
  const uint16_t state_capacity =
      loop->final_header_count > loop->final_result_count
          ? loop->final_header_count
          : loop->final_result_count;
  if (state_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, state_capacity,
                                                   sizeof(*state_values),
                                                   (void**)&state_values));
  }

  if (loop->condition_terminator) {
    const loom_value_id_t* condition_operands =
        loom_op_const_operands(loop->condition_terminator);
    const loom_builder_ip_t saved = loom_builder_enter_region(
        &plan->rewriter.builder, replacement->loop.op,
        loom_loop_like_condition_region(replacement->loop));
    iree_status_t status = iree_ok_status();
    for (uint16_t i = 0; i < loop->result_count && iree_status_is_ok(status);
         ++i) {
      const loom_boundary_projection_loop_result_state_t* state =
          &loop->result_states[i];
      const uint16_t offset = loop->result_offsets[i];
      if (!loom_boundary_projection_loop_result_is_selected(function, state)) {
        state_values[offset] = loom_boundary_projection_resolve_value(
            function, condition_operands[1 + i]);
        continue;
      }
      const loom_boundary_projection_slot_t* candidate =
          &function->candidates[state->body_candidate];
      status = loom_boundary_projection_materialize_loop_source(
          plan, function, candidate, &state->condition_source,
          state_values ? state_values + offset : NULL);
    }
    loom_builder_restore(&plan->rewriter.builder, saved);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(loom_boundary_projection_build_loop_terminator(
        plan, function, replacement->loop.op,
        loom_loop_like_condition_region(replacement->loop),
        loop->condition_terminator, condition_operands,
        /*state_operand_offset=*/1, state_values, loop->final_result_count,
        out_condition_terminator));
  } else {
    *out_condition_terminator = NULL;
  }

  uint16_t final_backedge_count = loop->final_result_count;
  const loom_builder_ip_t saved =
      loom_builder_enter_region(&plan->rewriter.builder, replacement->loop.op,
                                loom_loop_like_body(replacement->loop));
  iree_status_t status = iree_ok_status();
  const loom_value_id_t* body_operands =
      loom_op_const_operands(loop->body_terminator);
  if (loop->condition_terminator) {
    final_backedge_count = loop->final_header_count;
    for (uint16_t i = 0; i < loop->header_count && iree_status_is_ok(status);
         ++i) {
      const loom_boundary_projection_loop_header_state_t* state =
          &loop->header_states[i];
      const uint16_t offset = loop->header_offsets[i];
      if (!loom_boundary_projection_loop_header_is_selected(function, state)) {
        state_values[offset] =
            loom_boundary_projection_resolve_value(function, body_operands[i]);
        continue;
      }
      const loom_boundary_projection_slot_t* candidate =
          &function->candidates[state->candidate];
      status = loom_boundary_projection_materialize_loop_source(
          plan, function, candidate, &state->backedge_source,
          state_values ? state_values + offset : NULL);
    }
  } else {
    for (uint16_t i = 0; i < loop->result_count && iree_status_is_ok(status);
         ++i) {
      const loom_boundary_projection_loop_result_state_t* state =
          &loop->result_states[i];
      const uint16_t offset = loop->result_offsets[i];
      if (!loom_boundary_projection_loop_result_is_selected(function, state)) {
        state_values[offset] =
            loom_boundary_projection_resolve_value(function, body_operands[i]);
        continue;
      }
      const loom_boundary_projection_slot_t* candidate =
          &function->candidates[state->body_candidate];
      status = loom_boundary_projection_materialize_loop_source(
          plan, function, candidate, &state->backedge_source,
          state_values ? state_values + offset : NULL);
    }
  }
  loom_builder_restore(&plan->rewriter.builder, saved);
  IREE_RETURN_IF_ERROR(status);
  return loom_boundary_projection_build_loop_terminator(
      plan, function, replacement->loop.op,
      loom_loop_like_body(replacement->loop), loop->body_terminator,
      body_operands,
      /*state_operand_offset=*/0, state_values, final_backedge_count,
      out_body_terminator);
}

static iree_status_t loom_boundary_projection_map_unprojected_endpoint(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, loom_ir_remap_t* remap,
    loom_value_id_t source, loom_value_id_t target) {
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(remap, source, target));
  return loom_boundary_projection_replace_definition(plan, function, source,
                                                     target);
}

static iree_status_t loom_boundary_projection_realize_loop_candidate(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    iree_host_size_t candidate_index, loom_ir_remap_t* remap,
    loom_location_id_t location) {
  loom_boundary_projection_slot_t* candidate =
      &function->candidates[candidate_index];
  loom_type_t logical_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_type(remap, candidate->logical_type, &logical_type));
  if (candidate->schema.destination_mode ==
      LOOM_BOUNDARY_PROJECTION_DESTINATION_RECONSTRUCT) {
    IREE_RETURN_IF_ERROR(candidate->schema.rule->transport.reconstruct(
        candidate->schema.rule, plan, function, candidate, logical_type,
        location, &candidate->replacement_value_id));
    return loom_boundary_projection_replace_definition(
        plan, function, candidate->value_id, candidate->replacement_value_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_realize_loop_header(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop,
    const loom_value_id_t* target_values, loom_location_id_t location) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_loop_initialize_remap(plan, &remap));
  for (uint16_t i = 0; i < loop->header_count; ++i) {
    const loom_boundary_projection_loop_header_state_t* state =
        &loop->header_states[i];
    if (loom_boundary_projection_loop_header_is_selected(function, state)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_boundary_projection_map_unprojected_endpoint(
        plan, function, &remap, state->value_id,
        target_values[loop->header_offsets[i]]));
  }
  for (uint16_t i = 0; i < loop->header_count; ++i) {
    const loom_boundary_projection_loop_header_state_t* state =
        &loop->header_states[i];
    if (loom_boundary_projection_loop_header_is_selected(function, state)) {
      IREE_RETURN_IF_ERROR(loom_boundary_projection_realize_loop_candidate(
          plan, function, state->candidate, &remap, location));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_realize_loop_body(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop,
    const loom_value_id_t* target_values, loom_location_id_t location) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_loop_initialize_remap(plan, &remap));
  for (uint16_t i = 0; i < loop->result_count; ++i) {
    const loom_boundary_projection_loop_result_state_t* state =
        &loop->result_states[i];
    if (loom_boundary_projection_loop_result_is_selected(function, state)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_boundary_projection_map_unprojected_endpoint(
        plan, function, &remap, state->body_value_id,
        target_values[loop->result_offsets[i]]));
  }
  for (uint16_t i = 0; i < loop->result_count; ++i) {
    const loom_boundary_projection_loop_result_state_t* state =
        &loop->result_states[i];
    if (loom_boundary_projection_loop_result_is_selected(function, state)) {
      IREE_RETURN_IF_ERROR(loom_boundary_projection_realize_loop_candidate(
          plan, function, state->body_candidate, &remap, location));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_realize_loop_results(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_loop_t* loop,
    const loom_loop_like_replacement_t* replacement) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_loop_initialize_remap(plan, &remap));
  for (uint16_t i = 0; i < loop->result_count; ++i) {
    const loom_boundary_projection_loop_result_state_t* state =
        &loop->result_states[i];
    if (loom_boundary_projection_loop_result_is_selected(function, state)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_boundary_projection_map_unprojected_endpoint(
        plan, function, &remap, state->result_value_id,
        replacement->results.values[loop->result_offsets[i]]));
  }

  for (uint16_t i = 0; i < loop->result_count; ++i) {
    const loom_boundary_projection_loop_result_state_t* state =
        &loop->result_states[i];
    if (!loom_boundary_projection_loop_result_is_selected(function, state)) {
      continue;
    }
    loom_boundary_projection_slot_t* candidate =
        &function->candidates[state->result_candidate];
    loom_type_t logical_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, candidate->logical_type, &logical_type));
    loom_builder_set_after(&plan->rewriter.builder, replacement->loop.op);
    if (candidate->schema.destination_mode ==
        LOOM_BOUNDARY_PROJECTION_DESTINATION_RECONSTRUCT) {
      IREE_RETURN_IF_ERROR(candidate->schema.rule->transport.reconstruct(
          candidate->schema.rule, plan, function, candidate, logical_type,
          replacement->loop.op->location, &candidate->replacement_value_id));
      IREE_RETURN_IF_ERROR(loom_boundary_projection_replace_definition(
          plan, function, candidate->value_id,
          candidate->replacement_value_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_move_loop_region(
    loom_boundary_projection_plan_t* plan, loom_block_t* source_block,
    loom_block_t* target_block, loom_op_t* target_parent) {
  loom_op_t* op = source_block->first_op;
  while (op) {
    loom_op_t* next = op->next_op;
    IREE_RETURN_IF_ERROR(loom_rewriter_move_to_block_end(
        &plan->rewriter, op, target_block, target_parent));
    op = next;
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_open_loop(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_loop_t* loop,
    loom_loop_like_replacement_t* out_replacement) {
  *out_replacement = (loom_loop_like_replacement_t){0};
  loom_value_id_t* initial_values = NULL;
  loom_type_t* header_types = NULL;
  loom_type_t* result_types = NULL;
  loom_value_id_t* reserved_values = NULL;
  if (loop->final_header_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, loop->final_header_count, sizeof(*initial_values),
        (void**)&initial_values));
    if (loop->condition_terminator) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          plan->arena, loop->final_header_count, sizeof(*header_types),
          (void**)&header_types));
    }
  }
  if (loop->final_result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, loop->final_result_count, sizeof(*result_types),
        (void**)&result_types));
  }
  const iree_host_size_t reserved_header_count =
      loop->condition_terminator ? loop->final_header_count : 0;
  const iree_host_size_t reserved_value_count =
      reserved_header_count + loop->final_result_count;
  if (reserved_value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, reserved_value_count, sizeof(*reserved_values),
        (void**)&reserved_values));
  }
  IREE_RETURN_IF_ERROR(loom_boundary_projection_materialize_initial_state(
      plan, function, loop, initial_values));

  loom_builder_set_before(&plan->rewriter.builder, loop->loop.op);
  if (reserved_value_count != 0) {
    IREE_RETURN_IF_ERROR(loom_builder_reserve_values(
        &plan->rewriter.builder, reserved_value_count, reserved_values));
  }
  loom_value_id_t* reserved_results =
      reserved_values ? reserved_values + reserved_header_count : NULL;
  if (loop->condition_terminator && loop->final_header_count != 0) {
    IREE_RETURN_IF_ERROR(loom_boundary_projection_build_loop_header_types(
        plan, function, loop, reserved_values, header_types));
  }
  if (loop->final_result_count != 0) {
    IREE_RETURN_IF_ERROR(loom_boundary_projection_build_loop_result_types(
        plan, function, loop, reserved_results, result_types));
  }
  const loom_loop_like_replacement_state_t replacement_state = {
      .initial_values =
          {
              .values = initial_values,
              .count = loop->final_header_count,
          },
      .header_types = header_types,
      .source_header_offsets = loop->header_offsets,
      .result_types = result_types,
      .result_count = loop->final_result_count,
      .source_result_offsets = loop->result_offsets,
  };
  IREE_RETURN_IF_ERROR(loom_loop_like_build_replacement(
      &plan->rewriter.builder, loop->loop, &replacement_state, plan->arena,
      out_replacement));
  for (uint16_t i = 0; i < reserved_header_count; ++i) {
    IREE_ASSERT_EQ(out_replacement->condition_state.values[i],
                   reserved_values[i]);
  }
  for (uint16_t i = 0; i < loop->final_result_count; ++i) {
    IREE_ASSERT_EQ(out_replacement->results.values[i], reserved_results[i]);
  }
  IREE_RETURN_IF_ERROR(loom_boundary_projection_bind_loop_components(
      plan, function, loop, out_replacement));

  const loom_location_id_t condition_location =
      loop->condition_terminator ? loop->condition_terminator->location
                                 : LOOM_LOCATION_UNKNOWN;
  const loom_location_id_t body_location = loop->body_terminator->location;

  // Retained recipes and the immutable original terminator payloads provide
  // every outgoing state value needed to finish the replacement terminators.
  // Retire the original terminators before destination realization so
  // eliminative rules can remove complete source chains. The replacement
  // regions remain in their normal construction interval until dominated
  // child loops have published the component definitions those recipes
  // reference.
  if (loop->condition_terminator) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_erase(&plan->rewriter, loop->condition_terminator));
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_erase(&plan->rewriter, loop->body_terminator));

  if (out_replacement->condition_entry) {
    loom_block_t* source_condition =
        loom_region_entry_block(loom_loop_like_condition_region(loop->loop));
    const loom_builder_ip_t saved = loom_builder_enter_region(
        &plan->rewriter.builder, out_replacement->loop.op,
        loom_loop_like_condition_region(out_replacement->loop));
    iree_status_t status = loom_boundary_projection_realize_loop_header(
        plan, function, loop, out_replacement->condition_state.values,
        condition_location);
    loom_builder_restore(&plan->rewriter.builder, saved);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(loom_boundary_projection_move_loop_region(
        plan, source_condition, out_replacement->condition_entry,
        out_replacement->loop.op));
  }

  loom_block_t* source_body =
      loom_region_entry_block(loom_loop_like_body(loop->loop));
  const loom_builder_ip_t saved = loom_builder_enter_region(
      &plan->rewriter.builder, out_replacement->loop.op,
      loom_loop_like_body(out_replacement->loop));
  iree_status_t status = loom_boundary_projection_realize_loop_body(
      plan, function, loop, out_replacement->body_state.values, body_location);
  loom_builder_restore(&plan->rewriter.builder, saved);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_boundary_projection_move_loop_region(
      plan, source_body, out_replacement->body_entry,
      out_replacement->loop.op));

  const loom_value_id_t source_iv = loom_loop_like_iv(loop->loop);
  if (source_iv != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_boundary_projection_replace_definition(
        plan, function, source_iv, loom_loop_like_iv(out_replacement->loop)));
  }

  IREE_RETURN_IF_ERROR(loom_boundary_projection_realize_loop_results(
      plan, function, loop, out_replacement));
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_close_loop(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_loop_t* loop,
    const loom_loop_like_replacement_t* replacement) {
  loom_op_t* condition_terminator = NULL;
  loom_op_t* body_terminator = NULL;
  IREE_RETURN_IF_ERROR(loom_boundary_projection_build_loop_terminators(
      plan, function, loop, replacement, &condition_terminator,
      &body_terminator));
  ++plan->loops_rewritten;
  return iree_ok_status();
}

typedef struct loom_boundary_projection_active_loop_t {
  loom_boundary_projection_loop_t* plan;
  loom_loop_like_replacement_t replacement;
} loom_boundary_projection_active_loop_t;

iree_status_t loom_boundary_projection_apply_loops(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  loom_boundary_projection_active_loop_t* active_loops = NULL;
  if (function->loop_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, function->loop_count, sizeof(*active_loops),
        (void**)&active_loops));
  }
  iree_host_size_t active_loop_count = 0;
  for (iree_host_size_t i = 0; i < function->loop_count; ++i) {
    loom_boundary_projection_loop_t* loop = &function->loops[i];
    while (active_loop_count != 0 &&
           !loom_op_is_ancestor_of(
               active_loops[active_loop_count - 1].replacement.loop.op,
               loop->loop.op)) {
      loom_boundary_projection_active_loop_t* active =
          &active_loops[--active_loop_count];
      IREE_RETURN_IF_ERROR(loom_boundary_projection_close_loop(
          plan, function, active->plan, &active->replacement));
    }
    if (!loop->selected ||
        iree_any_bit_set(loop->loop.op->flags, LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    loom_boundary_projection_active_loop_t* active =
        &active_loops[active_loop_count++];
    active->plan = loop;
    IREE_RETURN_IF_ERROR(loom_boundary_projection_open_loop(
        plan, function, loop, &active->replacement));
  }
  while (active_loop_count != 0) {
    loom_boundary_projection_active_loop_t* active =
        &active_loops[--active_loop_count];
    IREE_RETURN_IF_ERROR(loom_boundary_projection_close_loop(
        plan, function, active->plan, &active->replacement));
  }
  return iree_ok_status();
}
