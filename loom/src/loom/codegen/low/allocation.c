// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include <string.h>

#include "loom/codegen/low/allocation/copy_decision.h"
#include "loom/codegen/low/allocation/destructive_reuse.h"
#include "loom/codegen/low/allocation/edge_copy.h"
#include "loom/codegen/low/allocation/interval_assignment.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/loop_edge_relocation.h"
#include "loom/codegen/low/allocation/numbering.h"
#include "loom/codegen/low/allocation/packet_move.h"
#include "loom/codegen/low/allocation/placement.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness_builder.h"
#include "loom/codegen/low/allocation/write_interference.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

typedef struct loom_low_allocation_build_state_t {
  // Module containing the allocated low function.
  loom_module_t* module;
  // Caller-provided allocation options.
  const loom_low_allocation_options_t* options;
  // Arena owning all table arrays.
  iree_arena_allocator_t* arena;
  // Body region of the low function.
  loom_region_t* body;
  // Low function definition operation being allocated.
  const loom_op_t* function_op;
  // Resolved target selected by the low function.
  loom_low_resolved_target_t target;
  // Resolved target storage budgets, fixed values, and reserved ranges.
  loom_low_allocation_target_constraints_t target_constraints;
  // Liveness analysis for |body|.
  loom_liveness_analysis_t liveness;
  // Function-local placement relations over |liveness|.
  loom_low_placement_table_t placement;
  // Allocation-only instruction preferences shared by assignment attempts.
  loom_low_placement_preference_index_t preferences;
  // Mutable per-allocation-unit live end points.
  loom_low_allocation_unit_liveness_t unit_liveness;
  // Structural content identity shared by all assignment attempts.
  loom_low_allocation_storage_identity_t storage_identity;
  // Completed interval assignment, spill plan, and remark rows.
  loom_low_allocation_interval_assignment_result_t interval_assignment;
  // Mutable low.copy decision plan being built.
  loom_low_allocation_copy_decision_plan_t copy_decision_plan;
  // Allocation-owned final structural moves and reusable sequencing scratch.
  loom_low_allocation_move_plan_t move_plan;
  // Invocation-only transport from incoming ABI locations to assignments.
  loom_low_move_group_t entry_moves;
  // Final argument and result transport for retained calls, in source order.
  loom_low_allocation_call_moves_t* call_moves;
  // Number of initialized call transport records.
  iree_host_size_t call_move_count;
  // Mutable branch edge-copy plan being built.
  loom_low_allocation_edge_copy_plan_t edge_copy_plan;
  // Mutable packet-local final move plan being built.
  loom_low_allocation_packet_move_plan_t packet_move_plan;
  // Mutable assignment-backed storage leases and release actions being built.
  loom_low_allocation_storage_lease_state_t storage_leases;
} loom_low_allocation_build_state_t;

static loom_low_allocation_interval_assignment_context_t
loom_low_allocation_make_interval_assignment_context(
    loom_low_allocation_build_state_t* state,
    const loom_low_function_model_t* model,
    loom_low_allocation_search_strategy_t search_strategy,
    iree_arena_allocator_t* arena,
    loom_low_allocation_target_constraints_t* target_constraints,
    loom_low_allocation_storage_lease_state_t* storage_leases) {
  return (loom_low_allocation_interval_assignment_context_t){
      .module = state->module,
      .body = state->body,
      .function_op = state->function_op,
      .target = &state->target,
      .liveness = &state->liveness,
      .value_domain = &model->context.value_domain,
      .schedule = state->options->schedule,
      .placement = &state->placement,
      .preferences = &state->preferences,
      .target_constraints = target_constraints,
      .entry_locations = state->options->entry_locations,
      .entry_location_count = state->options->entry_location_count,
      .storage_transport = state->options->storage_transport,
      .required_register_values = state->options->required_register_values,
      .unit_liveness = &state->unit_liveness,
      .residency = state->options->residency,
      .storage_leases = storage_leases,
      .arena = arena,
      .function_cfg_graph = &model->cfg_graph,
      .search_strategy = search_strategy,
  };
}

// Selects between first-fit and an isolated fragmentation-repair coloring.
// A winning repair retains its owned arrays and recycles the first-fit state.
static iree_status_t loom_low_allocation_repair_fragmentation(
    loom_low_allocation_build_state_t* state,
    const loom_low_function_model_t* model,
    const loom_local_value_domain_t* value_domain,
    const iree_arena_checkpoint_t* assignment_checkpoint,
    iree_arena_allocator_t* decision_arena) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(state->arena->block_pool, &scratch_arena);

  loom_low_allocation_target_constraints_t scratch_target_constraints =
      state->target_constraints;
  scratch_target_constraints.emitter = (iree_diagnostic_emitter_t){0};
  scratch_target_constraints.error_count = 0;
  scratch_target_constraints.failure = (loom_low_allocation_failure_t){0};
  scratch_target_constraints.max_assigned_location_end_by_reg_class = NULL;
  scratch_target_constraints.max_fixed_location_end_by_reg_class = NULL;

  iree_status_t status = iree_ok_status();
  const iree_host_size_t reg_class_count =
      state->target.descriptor_set->reg_class_count;
  if (reg_class_count != 0) {
    iree_host_size_t location_end_count = 0;
    if (!iree_host_size_checked_mul(reg_class_count, 2, &location_end_count)) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "register-class extent table exceeds host size");
    }
    uint32_t* location_ends = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(
          &scratch_arena, location_end_count,
          sizeof(*scratch_target_constraints
                      .max_assigned_location_end_by_reg_class),
          (void**)&location_ends);
    }
    if (iree_status_is_ok(status)) {
      memset(location_ends, 0, location_end_count * sizeof(*location_ends));
      scratch_target_constraints.max_assigned_location_end_by_reg_class =
          location_ends;
      scratch_target_constraints.max_fixed_location_end_by_reg_class =
          location_ends + reg_class_count;
    }
  }

  loom_low_allocation_storage_lease_state_t scratch_storage_leases = {0};
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_storage_lease_state_initialize(
        &state->options->storage_leases, state->module, state->function_op,
        value_domain, &state->liveness, &state->storage_identity,
        &state->unit_liveness, &scratch_arena, &scratch_storage_leases);
  }

  loom_low_allocation_interval_assignment_result_t scratch_result = {0};
  if (iree_status_is_ok(status)) {
    const loom_low_allocation_interval_assignment_context_t scratch_context =
        loom_low_allocation_make_interval_assignment_context(
            state, model,
            LOOM_LOW_ALLOCATION_SEARCH_STRATEGY_FRAGMENTATION_REPAIR,
            &scratch_arena, &scratch_target_constraints,
            &scratch_storage_leases);
    status = loom_low_allocation_interval_assignment_build(
        &scratch_context, decision_arena, &scratch_result);
  }
  if (iree_status_is_ok(status)) {
    const bool use_repair =
        scratch_target_constraints.error_count == 0 &&
        (scratch_result.spill_count == 0 ||
         (state->target_constraints.error_count == 0 &&
          scratch_result.spill_traffic_bytes <
              state->interval_assignment.spill_traffic_bytes));
    if (use_repair) {
      iree_arena_checkpoint_restore(assignment_checkpoint);
      iree_arena_transfer(&scratch_arena, state->arena);
      state->target_constraints.error_count = 0;
      state->target_constraints.failure = (loom_low_allocation_failure_t){0};
      state->target_constraints.max_assigned_location_end_by_reg_class =
          scratch_target_constraints.max_assigned_location_end_by_reg_class;
      state->target_constraints.max_fixed_location_end_by_reg_class =
          scratch_target_constraints.max_fixed_location_end_by_reg_class;
      state->storage_leases = scratch_storage_leases;
      state->interval_assignment = scratch_result;
    }
  }

  iree_arena_deinitialize(&scratch_arena);
  return status;
}

// Formal arguments occupy the local value domain's prefix. Unused arguments
// have no assignment, and incoming identities need no move-plan workspace.
static const loom_low_allocation_assignment_t*
loom_low_allocation_entry_destination(
    const loom_low_allocation_build_state_t* state, iree_host_size_t ordinal) {
  const loom_low_allocation_abi_location_t* entry =
      &state->options->entry_locations[ordinal];
  const uint32_t assignment_index =
      state->interval_assignment.assignment_indices_by_value_ordinal[ordinal];
  if (entry->location_kind == LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED ||
      assignment_index == UINT32_MAX) {
    return NULL;
  }
  const loom_low_allocation_assignment_t* assignment =
      &state->interval_assignment.assignments[assignment_index];
  return assignment->descriptor_reg_class_id ==
                     entry->descriptor_reg_class_id &&
                 assignment->location_kind == entry->location_kind &&
                 assignment->location_base == entry->location_base
             ? NULL
             : assignment;
}

static iree_status_t loom_low_allocation_build_entry_moves(
    loom_low_allocation_build_state_t* state) {
  iree_host_size_t raw_move_count = 0;
  for (iree_host_size_t i = 0; i < state->options->entry_location_count; ++i) {
    const loom_low_allocation_assignment_t* destination =
        loom_low_allocation_entry_destination(state, i);
    if (destination == NULL) {
      continue;
    }
    const loom_low_allocation_abi_location_t* entry =
        &state->options->entry_locations[i];
    loom_low_allocation_assignment_t source = *destination;
    source.descriptor_reg_class_id = entry->descriptor_reg_class_id;
    source.location_kind = entry->location_kind;
    source.location_base = entry->location_base;
    loom_low_allocation_move_plan_append_assignment(
        &state->move_plan, &source, 0, destination, 0,
        destination->location_count, &raw_move_count);
    loom_low_allocation_target_constraints_record_location_extent(
        &state->target_constraints, source.descriptor_reg_class_id,
        source.location_kind, source.location_base, source.location_count);
  }
  return loom_low_allocation_move_plan_append_group(
      &state->move_plan, state->function_op, /*read_point=*/0,
      /*write_point=*/0, raw_move_count, &state->entry_moves,
      /*out_input_flags=*/NULL);
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_call_assignment(
    const loom_low_allocation_build_state_t* state,
    loom_value_ordinal_t ordinal) {
  const uint32_t index =
      state->interval_assignment.assignment_indices_by_value_ordinal[ordinal];
  return index == UINT32_MAX ? NULL
                             : &state->interval_assignment.assignments[index];
}

// Reads the already collected call sites and their target convention rows.
// Capacity includes whole register views; memory ABI operands are materialized
// by the target at this same boundary, before/after the register permutation.
static void loom_low_allocation_call_move_capacity(
    const loom_low_allocation_build_state_t* state,
    iree_host_size_t* total_count, iree_host_size_t* group_count) {
  const loom_low_schedule_table_t* schedule = state->options->schedule;
  if (!state->options->call_contracts.query || !schedule) {
    return;
  }
  for (iree_host_size_t c = 0; c < schedule->call_node_count; ++c) {
    const loom_low_schedule_node_t* node =
        &schedule->nodes[schedule->call_node_indices[c]];
    const loom_low_call_contract_t* contract =
        state->options->call_contracts.query(
            state->options->call_contracts.user_data,
            loom_low_func_call_callee(node->op));
    for (uint16_t side = 0; side < 2; ++side) {
      const loom_low_allocation_abi_location_t* registers =
          side ? contract->results : contract->arguments;
      const uint16_t count =
          side ? iree_min(node->result_count, contract->result_count)
               : iree_min(node->operand_count, contract->argument_count);
      const loom_value_ordinal_t* ordinals =
          side ? loom_low_schedule_node_const_result_ordinals(node)
               : loom_low_schedule_node_const_operand_ordinals(node);
      iree_host_size_t units = 0;
      for (uint16_t i = 0; i < count; ++i) {
        const loom_low_allocation_assignment_t* assignment =
            loom_low_allocation_call_assignment(state, ordinals[i]);
        if (registers[i].location_kind !=
                LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED &&
            assignment) {
          units += assignment->location_count;
        }
      }
      *total_count += units;
      *group_count = iree_max(*group_count, units);
    }
  }
}

static iree_status_t loom_low_allocation_build_call_moves(
    loom_low_allocation_build_state_t* state) {
  const loom_low_schedule_table_t* schedule = state->options->schedule;
  if (!state->options->call_contracts.query || !schedule ||
      !schedule->call_node_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, schedule->call_node_count, sizeof(*state->call_moves),
      (void**)&state->call_moves));
  for (iree_host_size_t c = 0; c < schedule->call_node_count; ++c) {
    const uint32_t node_index = schedule->call_node_indices[c];
    const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
    const loom_low_call_contract_t* contract =
        state->options->call_contracts.query(
            state->options->call_contracts.user_data,
            loom_low_func_call_callee(node->op));
    loom_low_allocation_call_moves_t* call =
        &state->call_moves[state->call_move_count++];
    *call = (loom_low_allocation_call_moves_t){
        .source_ordinal = node->source_ordinal,
        .argument_count = node->operand_count,
        .result_count = node->result_count,
    };
    const loom_liveness_operation_point_t* point = loom_liveness_operation_at(
        &state->liveness,
        state->move_plan.operation_indices_by_source_node[node_index]);
    for (uint16_t side = 0; side < 2; ++side) {
      const loom_low_allocation_abi_location_t* registers =
          side ? contract->results : contract->arguments;
      const uint16_t count =
          side ? iree_min(node->result_count, contract->result_count)
               : iree_min(node->operand_count, contract->argument_count);
      const loom_value_ordinal_t* ordinals =
          side ? loom_low_schedule_node_const_result_ordinals(node)
               : loom_low_schedule_node_const_operand_ordinals(node);
      iree_host_size_t raw_count = 0;
      for (uint16_t i = 0; i < count; ++i) {
        const loom_low_allocation_assignment_t* assignment =
            loom_low_allocation_call_assignment(state, ordinals[i]);
        if (registers[i].location_kind ==
                LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED ||
            !assignment) {
          continue;
        }
        loom_low_allocation_assignment_t boundary = *assignment;
        boundary.descriptor_reg_class_id = registers[i].descriptor_reg_class_id;
        boundary.location_kind = registers[i].location_kind;
        boundary.location_base = registers[i].location_base;
        loom_low_allocation_target_constraints_record_location_extent(
            &state->target_constraints, boundary.descriptor_reg_class_id,
            boundary.location_kind, boundary.location_base,
            boundary.location_count);
        loom_low_allocation_move_plan_append_assignment(
            &state->move_plan, side ? &boundary : assignment, 0,
            side ? assignment : &boundary, 0, assignment->location_count,
            &raw_count);
      }
      loom_low_move_group_t group;
      // Overflow arguments have already been stored before register transport.
      // The move set protects its own sources and destinations, including
      // identities; other dying inputs are available for cycle scratch.
      IREE_RETURN_IF_ERROR(loom_low_allocation_move_plan_append_group(
          &state->move_plan, node->op, point->end_point, point->end_point,
          raw_count, &group, /*out_input_flags=*/NULL));
      *(side ? &call->results : &call->arguments) = group.moves;
    }
  }
  return iree_ok_status();
}

// Structural moves consume final assignments. Their permutation index and
// sequencing workspace are private to this construction; only completed move
// rows and scratch-write indices survive in the allocation result.
static iree_status_t loom_low_allocation_build_moves(
    loom_low_allocation_build_state_t* state,
    iree_arena_allocator_t* decision_arena) {
  iree_host_size_t entry_unit_count = 0;
  for (iree_host_size_t i = 0; i < state->options->entry_location_count; ++i) {
    const loom_low_allocation_assignment_t* destination =
        loom_low_allocation_entry_destination(state, i);
    if (destination != NULL) {
      entry_unit_count += destination->location_count;
    }
  }
  iree_host_size_t call_unit_count = 0;
  iree_host_size_t call_group_capacity = 0;
  loom_low_allocation_call_move_capacity(state, &call_unit_count,
                                         &call_group_capacity);
  const bool has_calls = state->options->call_contracts.query &&
                         state->options->schedule &&
                         state->options->schedule->call_node_count;
  if (state->placement.packet_move_group_count == 0 &&
      state->placement.edge_copy_group_count == 0 && entry_unit_count == 0 &&
      !has_calls) {
    return iree_ok_status();
  }
  const iree_arena_checkpoint_t scratch_checkpoint =
      iree_arena_checkpoint_save(decision_arena);
  const loom_low_allocation_move_plan_context_t move_plan_context = {
      .storage_spaces = state->options->move_storage_spaces,
      .descriptor_set = state->target.descriptor_set,
      .target_constraints = &state->target_constraints,
      .unit_liveness = &state->unit_liveness,
      .assignment_map = state->interval_assignment.assignment_map,
      .schedule = state->options->schedule,
  };
  const iree_host_size_t move_input_capacity =
      state->placement.branch_unit_count +
      state->placement.packet_move_unit_count + entry_unit_count +
      call_unit_count;
  const iree_host_size_t raw_group_capacity = iree_max(
      iree_max(state->placement.max_move_group_unit_count, entry_unit_count),
      call_group_capacity);
  iree_status_t status = loom_low_allocation_move_plan_initialize(
      &move_plan_context, move_input_capacity, raw_group_capacity, state->arena,
      decision_arena, &state->move_plan);
  if (iree_status_is_ok(status) && entry_unit_count != 0) {
    status = loom_low_allocation_build_entry_moves(state);
  }
  if (iree_status_is_ok(status) && state->target_constraints.error_count == 0) {
    status = loom_low_allocation_build_call_moves(state);
  }
  if (iree_status_is_ok(status) && state->target_constraints.error_count == 0) {
    const loom_low_allocation_edge_copy_context_t edge_copy_context = {
        .flags = state->options->flags,
        .placement = &state->placement,
        .move_plan = &state->move_plan,
    };
    status = loom_low_allocation_edge_copy_plan_build(
        &edge_copy_context, state->arena, &state->edge_copy_plan);
  }
  if (iree_status_is_ok(status) && state->target_constraints.error_count == 0) {
    const loom_low_allocation_packet_move_context_t packet_move_context = {
        .placement = &state->placement,
        .move_plan = &state->move_plan,
    };
    status = loom_low_allocation_packet_move_plan_build(
        &packet_move_context, state->arena, &state->packet_move_plan);
  }
  iree_arena_checkpoint_restore(&scratch_checkpoint);
  return status;
}

iree_status_t loom_low_allocate_function(
    const loom_low_function_model_t* model,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_table) {
  *out_table = (loom_low_allocation_table_t){
      .module = model->context.module,
      .function_op = model->context.function_op,
      .entry_preamble_end = model->context.requirements.entry_preamble_end,
      .target = model->context.target,
      .error_count = model->context.error_count,
      .cfg_graph = model->cfg_graph,
  };
  const uint8_t allocation_mode =
      loom_low_function_allocation(model->context.function_op);
  IREE_ASSERT(
      allocation_mode == 0 || allocation_mode == LOOM_LOW_ALLOCATION_VIRTUAL,
      "allocation synthesis requires an admitted virtual function");
  if (model->context.error_count != 0) {
    return iree_ok_status();
  }
  IREE_ASSERT(
      loom_local_value_domain_is_acquired(&model->context.value_domain));
  IREE_ASSERT(iree_any_bit_set(model->context.value_domain.flags,
                               LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE));

  loom_low_allocation_build_state_t state = {
      .module = model->context.module,
      .options = options,
      .arena = arena,
      .body = model->context.body,
      .function_op = model->context.function_op,
      .target = model->context.target,
  };
  iree_arena_allocator_t decision_arena;
  iree_arena_initialize(arena->block_pool, &decision_arena);
  iree_status_t status = loom_low_allocation_target_constraints_initialize(
      model->context.module, model->context.function_op, &state.target,
      options->budgets, options->budget_count, options->reserved_ranges,
      options->reserved_range_count, options->emitter, arena,
      &state.target_constraints);

  const loom_local_value_domain_t* value_domain = &model->context.value_domain;
  const loom_liveness_order_t operation_order =
      options->schedule != NULL ? options->schedule->operation_order
                                : loom_liveness_order_empty();
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_liveness_analyze_local_value_domain_with_dataflow(
        value_domain, &model->liveness_dataflow, operation_order, arena,
        &state.liveness);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    const loom_low_placement_pair_use_list_t placement_pair_uses =
        options->schedule != NULL ? options->schedule->placement_pair_uses
                                  : loom_low_placement_pair_use_list_empty();
    status = loom_low_allocation_placement_build(
        &state.target_constraints, state.body, value_domain,
        model->context.storage_origins, &state.liveness, options->fixed_values,
        options->fixed_value_count, placement_pair_uses,
        options->instruction_preferences, arena, &decision_arena,
        &state.placement, &state.preferences);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_unit_liveness_initialize(
        &state.target, &state.placement, value_domain, &state.liveness,
        &model->cfg_graph, options->call_contracts, arena, &decision_arena,
        &state.unit_liveness);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_unit_liveness_retain_tied_storage(
        &state.unit_liveness, &state.liveness, &state.placement, arena,
        &decision_arena);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_refine_destructive_reuse(
        &state.unit_liveness, &state.liveness, &state.placement, arena);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_storage_identity_initialize(
        &options->storage_leases, &state.placement, &state.unit_liveness,
        &decision_arena, &state.storage_identity);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    loom_low_allocation_unit_liveness_propagate_storage_relations(
        &state.unit_liveness, &state.placement);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_target_constraints_finalize_fixed_values(
        &state.target_constraints, &state.liveness, &state.unit_liveness,
        arena);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    for (iree_host_size_t i = 0; i < state.target_constraints.fixed_value_count;
         ++i) {
      const loom_low_allocation_resolved_fixed_value_t* fixed =
          &state.target_constraints.fixed_values[i];
      loom_low_allocation_write_interference_note_fixed(
          state.unit_liveness.write_interference, fixed->value_ordinal,
          &fixed->assignment);
    }
    status = loom_low_allocation_write_interference_finalize(
        state.unit_liveness.write_interference, &state.liveness,
        &model->cfg_graph, &state.placement, &state.unit_liveness,
        &decision_arena, arena);
  }
  const iree_arena_checkpoint_t interval_assignment_checkpoint =
      iree_arena_checkpoint_save(arena);
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_storage_lease_state_initialize(
        &options->storage_leases, model->context.module,
        model->context.function_op, value_domain, &state.liveness,
        &state.storage_identity, &state.unit_liveness, arena,
        &state.storage_leases);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    const loom_low_allocation_interval_assignment_context_t
        interval_assignment_context =
            loom_low_allocation_make_interval_assignment_context(
                &state, model, LOOM_LOW_ALLOCATION_SEARCH_STRATEGY_FIRST_FIT,
                arena, &state.target_constraints, &state.storage_leases);
    status = loom_low_allocation_interval_assignment_build(
        &interval_assignment_context, &decision_arena,
        &state.interval_assignment);
  }
  // Required register values can fail first-fit placement through fragmentation
  // just as spillable values can. Their retained failure is provisional until
  // the alternative packing has also been considered.
  if (iree_status_is_ok(status) &&
      state.interval_assignment.has_packable_aggregates &&
      ((state.target_constraints.error_count == 0 &&
        state.interval_assignment.spill_count != 0) ||
       loom_low_allocation_failure_is_present(
           &state.target_constraints.failure))) {
    status = loom_low_allocation_repair_fragmentation(
        &state, model, value_domain, &interval_assignment_checkpoint,
        &decision_arena);
  }
  // Backedge placement belongs to the final physical assignment. Spill repair
  // rewrites the IR and rebuilds the frame, so relocating a provisional spill
  // assignment would be discarded.
  const bool assignment_is_final =
      state.interval_assignment.spill_plan_count == 0 &&
      state.interval_assignment.spill_count == 0;
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0 &&
      assignment_is_final) {
    const iree_arena_checkpoint_t relocation_checkpoint =
        iree_arena_checkpoint_save(&decision_arena);
    const loom_low_allocation_loop_edge_relocation_context_t
        loop_edge_relocation_context = {
            .module = state.module,
            .body = state.body,
            .cfg_graph = &model->cfg_graph,
            .descriptor_set = state.target.descriptor_set,
            .liveness = &state.liveness,
            .value_domain = &model->context.value_domain,
            .placement = &state.placement,
            .target_constraints = &state.target_constraints,
            .unit_liveness = &state.unit_liveness,
            .storage_leases = &state.storage_leases,
            .assignments = state.interval_assignment.assignments,
            .assignment_count = state.interval_assignment.assignment_count,
            .assignment_indices_by_value_ordinal =
                state.interval_assignment.assignment_indices_by_value_ordinal,
            .arena = &decision_arena,
        };
    loom_low_allocation_loop_edge_relocation_result_t
        loop_edge_relocation_result = {0};
    status = loom_low_allocation_loop_edge_relocate(
        &loop_edge_relocation_context, &loop_edge_relocation_result);
    iree_arena_checkpoint_restore(&relocation_checkpoint);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status =
        loom_low_allocation_storage_lease_state_finalize(&state.storage_leases);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    const loom_low_allocation_copy_decision_context_t copy_decision_context = {
        .body = state.body,
        .descriptor_set = state.target.descriptor_set,
        .assignment_map = state.interval_assignment.assignment_map,
    };
    status = loom_low_allocation_copy_decision_plan_build(
        &copy_decision_context, arena, &state.copy_decision_plan);
  }
  // Final structural moves also belong to the final physical assignment. A
  // provisional spill assignment may report false cycle-scratch conflicts
  // against registers that repair will release.
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0 &&
      assignment_is_final) {
    status = loom_low_allocation_build_moves(&state, &decision_arena);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0 &&
      assignment_is_final) {
    const loom_low_allocation_numbering_context_t numbering_context = {
        .placement = &state.placement,
        .preferences = &state.preferences,
        .target_constraints = &state.target_constraints,
        .entry_locations = options->entry_locations,
        .entry_location_count = options->entry_location_count,
        .unit_liveness = &state.unit_liveness,
        .interval_assignment = &state.interval_assignment,
        .storage_leases = &state.storage_leases,
        .moves = state.move_plan.moves,
        .move_count = state.move_plan.move_count,
    };
    status = loom_low_allocation_number_registers(&numbering_context,
                                                  &decision_arena);
  }
  iree_arena_deinitialize(&decision_arena);

  uint32_t* first_storage_release_action_by_node = NULL;
  if (iree_status_is_ok(status) &&
      state.storage_leases.release_action_count != 0 &&
      iree_any_bit_set(
          options->flags,
          LOOM_LOW_ALLOCATION_FLAG_RETAIN_STORAGE_RELEASE_ACTION_INDEX)) {
    status = loom_low_storage_release_action_index_build(
        state.storage_leases.release_actions,
        state.storage_leases.release_action_count,
        state.storage_leases.lease_table->schedule->node_count, arena,
        &first_storage_release_action_by_node);
  }

  loom_low_allocation_table_t table = {0};
  if (iree_status_is_ok(status)) {
    table = (loom_low_allocation_table_t){
        .module = model->context.module,
        .function_op = model->context.function_op,
        .entry_preamble_end = model->context.requirements.entry_preamble_end,
        .target = state.target,
        .storage_transport = options->storage_transport,
        .liveness = state.liveness,
        .storage_segments = state.unit_liveness.storage_segments.entries,
        .placement = state.placement,
        .fixed_values = state.target_constraints.fixed_values,
        .fixed_value_count = state.target_constraints.fixed_value_count,
        .allocation_mode =
            loom_low_function_allocation(model->context.function_op),
        .error_count = state.target_constraints.error_count,
        .assignments = state.interval_assignment.assignments,
        .assignment_count = state.interval_assignment.assignment_count,
        .physical_extents =
            {
                .ends_by_reg_class =
                    state.target_constraints
                        .max_assigned_location_end_by_reg_class,
                .fixed_ends_by_reg_class =
                    state.target_constraints
                        .max_fixed_location_end_by_reg_class,
                .count = state.target.descriptor_set->reg_class_count,
            },
        .assignment_indices_by_value_ordinal =
            state.interval_assignment.assignment_indices_by_value_ordinal,
        .unit_start_points = state.unit_liveness.start_points,
        .unit_end_points = state.unit_liveness.end_points,
        .unit_point_count = state.unit_liveness.point_count,
        .spill_plans = state.interval_assignment.spill_plans,
        .spill_plan_count = state.interval_assignment.spill_plan_count,
        .retained_fixed_values =
            state.interval_assignment.retained_fixed_values,
        .remarks = state.interval_assignment.remarks,
        .remark_count = state.interval_assignment.remark_count,
        .failure = state.target_constraints.failure,
        .copy_decisions = state.copy_decision_plan.decisions,
        .copy_decision_count = state.copy_decision_plan.decision_count,
        .entry_moves = state.entry_moves,
        .edge_copies = state.edge_copy_plan.copies,
        .edge_copy_count = state.edge_copy_plan.copy_count,
        .first_coalesced_incoming_copy_by_value_ordinal =
            state.edge_copy_plan.first_coalesced_incoming_copy_by_value_ordinal,
        .edge_copy_groups = state.edge_copy_plan.groups,
        .edge_copy_group_count = state.edge_copy_plan.group_count,
        .packet_transfers = state.packet_move_plan.transfers,
        .packet_transfer_count = state.packet_move_plan.transfer_count,
        .packet_move_groups = state.packet_move_plan.groups,
        .packet_move_group_count = state.packet_move_plan.group_count,
        .call_moves = state.call_moves,
        .call_move_count = state.call_move_count,
        .moves = state.move_plan.moves,
        .move_count = state.move_plan.move_count,
        .scratch_move_indices = state.move_plan.scratch_move_indices,
        .move_storage = state.move_plan.storage,
        .move_storage_count = state.move_plan.storage_count,
        .packet_move_count = state.packet_move_plan.move_count,
        .storage_leases = options->storage_leases,
        .storage_lease_instances = state.storage_leases.instances,
        .storage_lease_instance_count = state.storage_leases.instance_count,
        .storage_lease_unit_index = state.storage_leases.unit_index,
        .storage_release_actions = state.storage_leases.release_actions,
        .storage_release_action_count =
            state.storage_leases.release_action_count,
        .first_storage_release_action_by_node =
            first_storage_release_action_by_node,
        .spill_count = state.interval_assignment.spill_count,
        .coalesced_copy_count = state.copy_decision_plan.coalesced_count,
        .materialized_copy_count = state.copy_decision_plan.materialized_count,
        .reserved_ranges = state.target_constraints.reserved_ranges,
        .reserved_range_count = state.target_constraints.reserved_range_count,
        .cfg_graph = model->cfg_graph,
    };
  }
  if (iree_status_is_ok(status)) {
    *out_table = table;
  }
  return status;
}
