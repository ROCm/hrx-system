// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include "loom/codegen/low/allocation/copy_decision.h"
#include "loom/codegen/low/allocation/edge_copy.h"
#include "loom/codegen/low/allocation/interval_assignment.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/loop_edge_relocation.h"
#include "loom/codegen/low/allocation/packet_move.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/allocation/unit_location.h"
#include "loom/codegen/low/diagnostics.h"
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
  // Mutable per-allocation-unit live end points.
  loom_low_allocation_unit_liveness_t unit_liveness;
  // Completed interval assignment, spill plan, and remark rows.
  loom_low_allocation_interval_assignment_result_t interval_assignment;
  // Mutable low.copy decision plan being built.
  loom_low_allocation_copy_decision_plan_t copy_decision_plan;
  // Allocation-owned final structural moves and reusable sequencing scratch.
  loom_low_allocation_move_plan_t move_plan;
  // Mutable branch edge-copy plan being built.
  loom_low_allocation_edge_copy_plan_t edge_copy_plan;
  // Mutable packet-local final move plan being built.
  loom_low_allocation_packet_move_plan_t packet_move_plan;
  // Mutable assignment-backed storage leases and release actions being built.
  loom_low_allocation_storage_lease_state_t storage_leases;
} loom_low_allocation_build_state_t;

static bool loom_low_allocation_mode_can_synthesize(uint8_t allocation_mode) {
  return allocation_mode == 0 || allocation_mode == LOOM_LOW_ALLOCATION_VIRTUAL;
}

static const char* loom_low_allocation_mode_name(uint8_t allocation_mode) {
  switch (allocation_mode) {
    case 0:
    case LOOM_LOW_ALLOCATION_VIRTUAL:
      return "virtual";
    case LOOM_LOW_ALLOCATION_ASSIGNED:
      return "assigned";
    case LOOM_LOW_ALLOCATION_FIXED:
      return "fixed";
    default:
      return "unknown";
  }
}

static iree_status_t loom_low_allocation_validate_synthesis_mode(
    const loom_op_t* low_func_op) {
  uint8_t allocation_mode = loom_low_function_allocation(low_func_op);
  if (loom_low_allocation_mode_can_synthesize(allocation_mode)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "low allocation synthesis requires allocation(virtual), but function has "
      "allocation(%s)",
      loom_low_allocation_mode_name(allocation_mode));
}

iree_status_t loom_low_allocate_function(
    const loom_low_function_model_t* model,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_table) {
  *out_table = (loom_low_allocation_table_t){
      .module = model->module,
      .function_op = model->function_op,
      .target = model->target,
      .error_count = model->error_count,
      .cfg_graph = model->cfg_graph,
  };
  if (model->error_count != 0) return iree_ok_status();
  IREE_ASSERT(loom_local_value_domain_is_acquired(&model->value_domain));
  IREE_ASSERT(iree_any_bit_set(model->value_domain.flags,
                               LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE));

  loom_low_allocation_build_state_t state = {
      .module = model->module,
      .options = options,
      .arena = arena,
      .body = model->body,
      .function_op = model->function_op,
      .target = model->target,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_validate_synthesis_mode(model->function_op));
  iree_status_t status = loom_low_allocation_target_constraints_initialize(
      model->module, model->function_op, &state.target, options->budgets,
      options->budget_count, options->reserved_ranges,
      options->reserved_range_count, options->emitter, arena,
      &state.target_constraints);

  const loom_local_value_domain_t* value_domain = &model->value_domain;
  const loom_liveness_order_t operation_order =
      options->schedule != NULL ? options->schedule->operation_order
                                : loom_liveness_order_empty();
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_liveness_analyze_local_value_domain_with_cfg_graph(
        value_domain, &model->cfg_graph, operation_order, arena,
        &state.liveness);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    const loom_low_placement_pair_use_list_t placement_pair_uses =
        options->schedule != NULL ? options->schedule->placement_pair_uses
                                  : loom_low_placement_pair_use_list_empty();
    status = loom_low_placement_analyze_region(
        model->module, state.body, value_domain, &state.liveness,
        placement_pair_uses, arena, &state.placement);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_unit_liveness_initialize(
        model->module, &state.target, &state.placement, value_domain,
        &state.liveness, arena, &state.unit_liveness);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_unit_liveness_propagate_storage_relations(
        &state.unit_liveness, &state.liveness, &state.placement);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_target_constraints_resolve_fixed_values(
        &state.target_constraints, &state.liveness, value_domain,
        options->fixed_values, options->fixed_value_count, arena);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_storage_lease_state_initialize(
        &options->storage_leases, model->module, model->function_op,
        value_domain, &state.liveness, arena, &state.storage_leases);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    const loom_low_allocation_interval_assignment_context_t
        interval_assignment_context = {
            .module = state.module,
            .body = state.body,
            .function_op = state.function_op,
            .target = &state.target,
            .liveness = &state.liveness,
            .schedule = options->schedule,
            .placement = &state.placement,
            .target_constraints = &state.target_constraints,
            .required_register_values = options->required_register_values,
            .unit_liveness = &state.unit_liveness,
            .residency_model = options->residency_model,
            .storage_leases = &state.storage_leases,
            .arena = arena,
            .function_cfg_graph = &model->cfg_graph,
        };
    status = loom_low_allocation_interval_assignment_build(
        &interval_assignment_context, &state.interval_assignment);
  }
  // Backedge placement belongs to the final physical assignment. Spill repair
  // rewrites the IR and rebuilds the frame, so relocating a provisional spill
  // assignment would be discarded.
  const bool assignment_is_final =
      state.interval_assignment.spill_plan_count == 0 &&
      state.interval_assignment.spill_count == 0;
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0 &&
      assignment_is_final) {
    const loom_low_allocation_loop_edge_relocation_context_t
        loop_edge_relocation_context = {
            .module = state.module,
            .body = state.body,
            .cfg_graph = &model->cfg_graph,
            .descriptor_set = state.target.descriptor_set,
            .liveness = &state.liveness,
            .placement = &state.placement,
            .target_constraints = &state.target_constraints,
            .unit_liveness = &state.unit_liveness,
            .storage_leases = &state.storage_leases,
            .assignments = state.interval_assignment.assignments,
            .assignment_count = state.interval_assignment.assignment_count,
            .assignment_indices_by_value_ordinal =
                state.interval_assignment.assignment_indices_by_value_ordinal,
            .arena = arena,
        };
    loom_low_allocation_loop_edge_relocation_result_t
        loop_edge_relocation_result = {0};
    status = loom_low_allocation_loop_edge_relocate(
        &loop_edge_relocation_context, &loop_edge_relocation_result);
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
    const loom_low_allocation_move_plan_context_t move_plan_context = {
        .descriptor_set = state.target.descriptor_set,
        .target_constraints = &state.target_constraints,
        .unit_liveness = &state.unit_liveness,
        .assignment_map = state.interval_assignment.assignment_map,
    };
    const iree_host_size_t move_input_capacity =
        state.placement.branch_unit_count +
        state.placement.packet_move_unit_count;
    const iree_host_size_t raw_group_capacity =
        iree_max(state.placement.branch_unit_count,
                 state.placement.packet_move_unit_count);
    status = loom_low_allocation_move_plan_initialize(
        &move_plan_context, arena, move_input_capacity, raw_group_capacity,
        &state.move_plan);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0 &&
      assignment_is_final) {
    const loom_low_allocation_edge_copy_context_t edge_copy_context = {
        .placement = &state.placement,
        .move_plan = &state.move_plan,
    };
    status = loom_low_allocation_edge_copy_plan_build(&edge_copy_context, arena,
                                                      &state.edge_copy_plan);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0 &&
      assignment_is_final) {
    const loom_low_allocation_packet_move_context_t packet_move_context = {
        .placement = &state.placement,
        .move_plan = &state.move_plan,
    };
    status = loom_low_allocation_packet_move_plan_build(
        &packet_move_context, arena, &state.packet_move_plan);
  }

  loom_low_allocation_table_t table = {0};
  if (iree_status_is_ok(status)) {
    table = (loom_low_allocation_table_t){
        .module = model->module,
        .function_op = model->function_op,
        .target = state.target,
        .liveness = state.liveness,
        .placement = state.placement,
        .fixed_values = state.target_constraints.fixed_values,
        .fixed_value_count = state.target_constraints.fixed_value_count,
        .allocation_mode = loom_low_function_allocation(model->function_op),
        .error_count = state.target_constraints.error_count,
        .assignments = state.interval_assignment.assignments,
        .assignment_count = state.interval_assignment.assignment_count,
        .physical_extents =
            {
                .ends_by_reg_class =
                    state.target_constraints
                        .max_assigned_location_end_by_reg_class,
                .count = state.target.descriptor_set->reg_class_count,
            },
        .assignment_indices_by_value_ordinal =
            state.interval_assignment.assignment_indices_by_value_ordinal,
        .unit_start_points = state.unit_liveness.start_points,
        .unit_end_points = state.unit_liveness.end_points,
        .unit_point_count = state.unit_liveness.point_count,
        .spill_plans = state.interval_assignment.spill_plans,
        .spill_plan_count = state.interval_assignment.spill_plan_count,
        .remarks = state.interval_assignment.remarks,
        .remark_count = state.interval_assignment.remark_count,
        .failure = state.interval_assignment.failure,
        .copy_decisions = state.copy_decision_plan.decisions,
        .copy_decision_count = state.copy_decision_plan.decision_count,
        .edge_copies = state.edge_copy_plan.copies,
        .edge_copy_count = state.edge_copy_plan.copy_count,
        .edge_copy_groups = state.edge_copy_plan.groups,
        .edge_copy_group_count = state.edge_copy_plan.group_count,
        .packet_move_groups = state.packet_move_plan.groups,
        .packet_move_group_count = state.packet_move_plan.group_count,
        .moves = state.move_plan.moves,
        .scratch_move_indices = state.move_plan.scratch_move_indices,
        .packet_move_count = state.packet_move_plan.move_count,
        .storage_leases = options->storage_leases,
        .storage_lease_instances = state.storage_leases.instances,
        .storage_lease_instance_count = state.storage_leases.instance_count,
        .storage_lease_unit_index = state.storage_leases.unit_index,
        .storage_release_actions = state.storage_leases.release_actions,
        .storage_release_action_count =
            state.storage_leases.release_action_count,
        .spill_count = state.interval_assignment.spill_count,
        .coalesced_copy_count = state.copy_decision_plan.coalesced_count,
        .materialized_copy_count = state.copy_decision_plan.materialized_count,
        .reserved_ranges = state.target_constraints.reserved_ranges,
        .reserved_range_count = state.target_constraints.reserved_range_count,
        .cfg_graph = model->cfg_graph,
    };
  }
  if (iree_status_is_ok(status) && table.error_count == 0) {
    status = loom_low_allocation_diagnostics_emit(
        &table, options->diagnostic_flags, options->emitter);
  }
  if (iree_status_is_ok(status)) {
    *out_table = table;
  }
  return status;
}
