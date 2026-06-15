// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include "loom/codegen/low/allocation/copy_decision.h"
#include "loom/codegen/low/allocation/edge_copy.h"
#include "loom/codegen/low/allocation/interval_assignment.h"
#include "loom/codegen/low/allocation/packet_move.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/allocation/unit_location.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
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
  // Mutable branch edge-copy plan being built.
  loom_low_allocation_edge_copy_plan_t edge_copy_plan;
  // Mutable packet-local move scratch plan being built.
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
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_table) {
  if (!loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.kernel.def");
  }
  *out_table = (loom_low_allocation_table_t){0};

  loom_low_allocation_build_state_t state = {
      .module = module,
      .options = options,
      .arena = arena,
      .body = loom_low_function_body((loom_op_t*)low_func_op),
      .function_op = low_func_op,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_validate_synthesis_mode(low_func_op));

  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, low_func_op, options->descriptor_registry,
      options->target_selection, options->emitter, &state.target));
  if (!state.target.descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function target did not resolve");
  }
  iree_status_t status = loom_low_allocation_target_constraints_initialize(
      module, low_func_op, &state.target, options->budgets,
      options->budget_count, options->reserved_ranges,
      options->reserved_range_count, options->emitter, arena,
      &state.target_constraints);

  loom_local_value_domain_t value_domain = {0};
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_local_value_domain_acquire_for_region_tree(
        module, state.body, arena, &value_domain);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_liveness_analyze_local_value_domain(
        &value_domain, options->liveness_order, arena, &state.liveness);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_placement_analyze_region(module, state.body,
                                               &value_domain, &state.liveness,
                                               arena, &state.placement);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_unit_liveness_initialize(
        module, state.body, &state.target, options->liveness_order,
        &state.liveness, arena, &state.unit_liveness);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_unit_liveness_extend_for_tied_results(
        &state.unit_liveness, &state.liveness, &state.placement);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_target_constraints_resolve_fixed_values(
        &state.target_constraints, &state.liveness, &value_domain,
        options->fixed_values, options->fixed_value_count, arena);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    status = loom_low_allocation_storage_lease_state_initialize(
        &options->storage_leases, module, low_func_op, &state.liveness, arena,
        &state.storage_leases);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    const loom_low_allocation_interval_assignment_context_t
        interval_assignment_context = {
            .module = state.module,
            .body = state.body,
            .function_op = state.function_op,
            .target = &state.target,
            .liveness = &state.liveness,
            .placement = &state.placement,
            .target_constraints = &state.target_constraints,
            .unit_liveness = &state.unit_liveness,
            .storage_leases = &state.storage_leases,
            .arena = arena,
        };
    status = loom_low_allocation_interval_assignment_build(
        &interval_assignment_context, &state.interval_assignment);
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
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    const loom_low_allocation_edge_copy_context_t edge_copy_context = {
        .body = state.body,
        .descriptor_set = state.target.descriptor_set,
        .liveness_order = options->liveness_order,
        .target_constraints = &state.target_constraints,
        .unit_liveness = &state.unit_liveness,
        .placement = &state.placement,
        .assignment_map = state.interval_assignment.assignment_map,
    };
    status = loom_low_allocation_edge_copy_plan_build(&edge_copy_context, arena,
                                                      &state.edge_copy_plan);
  }
  if (iree_status_is_ok(status) && state.target_constraints.error_count == 0) {
    const loom_low_allocation_packet_move_context_t packet_move_context = {
        .module = state.module,
        .body = state.body,
        .descriptor_set = state.target.descriptor_set,
        .liveness_order = options->liveness_order,
        .target_constraints = &state.target_constraints,
        .unit_liveness = &state.unit_liveness,
        .assignment_map = state.interval_assignment.assignment_map,
    };
    status = loom_low_allocation_packet_move_plan_build(
        &packet_move_context, arena, &state.packet_move_plan);
  }

  loom_low_allocation_table_t table = {0};
  if (iree_status_is_ok(status)) {
    table = (loom_low_allocation_table_t){
        .module = module,
        .function_op = low_func_op,
        .target = state.target,
        .liveness = state.liveness,
        .placement = state.placement,
        .allocation_mode = loom_low_function_allocation(low_func_op),
        .error_count = state.target_constraints.error_count,
        .assignments = state.interval_assignment.assignments,
        .assignment_count = state.interval_assignment.assignment_count,
        .assignment_indices_by_value_ordinal =
            state.interval_assignment.assignment_indices_by_value_ordinal,
        .unit_end_points = state.unit_liveness.end_points,
        .unit_end_point_count = state.unit_liveness.end_point_count,
        .spill_plans = state.interval_assignment.spill_plans,
        .spill_plan_count = state.interval_assignment.spill_plan_count,
        .remarks = state.interval_assignment.remarks,
        .remark_count = state.interval_assignment.remark_count,
        .copy_decisions = state.copy_decision_plan.decisions,
        .copy_decision_count = state.copy_decision_plan.decision_count,
        .edge_copies = state.edge_copy_plan.copies,
        .edge_copy_count = state.edge_copy_plan.copy_count,
        .edge_copy_groups = state.edge_copy_plan.groups,
        .edge_copy_group_count = state.edge_copy_plan.group_count,
        .edge_copy_temporaries = state.edge_copy_plan.temporaries,
        .edge_copy_temporary_count = state.edge_copy_plan.temporary_count,
        .packet_move_temporary_groups = state.packet_move_plan.groups,
        .packet_move_temporary_group_count = state.packet_move_plan.group_count,
        .packet_move_temporaries = state.packet_move_plan.temporaries,
        .packet_move_temporary_count = state.packet_move_plan.temporary_count,
        .storage_leases = options->storage_leases,
        .storage_lease_instances = state.storage_leases.instances,
        .storage_lease_instance_count = state.storage_leases.instance_count,
        .storage_release_actions = state.storage_leases.release_actions,
        .storage_release_action_count =
            state.storage_leases.release_action_count,
        .spill_count = state.interval_assignment.spill_count,
        .coalesced_copy_count = state.copy_decision_plan.coalesced_count,
        .materialized_copy_count = state.copy_decision_plan.materialized_count,
    };
    loom_target_bundle_storage_rebind(&table.target.bundle_storage);
  }
  loom_local_value_domain_release(&value_domain);
  if (iree_status_is_ok(status) && table.error_count == 0) {
    status = loom_low_allocation_diagnostics_emit(
        &table, options->diagnostic_flags, options->emitter);
  }
  if (iree_status_is_ok(status)) {
    *out_table = table;
  }
  return status;
}
