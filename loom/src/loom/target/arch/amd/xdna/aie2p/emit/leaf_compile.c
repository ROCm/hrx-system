// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_compile.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/frame.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_capacity.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/spill.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/stack_frame.h"
#include "loom/target/reporting/low.h"

// Uses the scheduler's retained call inventory. A call-bearing frame is
// admitted only when program planning supplied the complete direct-call ABI.
static iree_status_t loom_aie2p_leaf_admit_calls(
    void* user_data, const loom_low_emission_frame_t* frame,
    iree_arena_allocator_t* arena, bool* out_accepted) {
  (void)arena;
  const loom_aie2p_leaf_compile_options_t* options = user_data;
  *out_accepted = frame->schedule.call_node_count == 0 ||
                  options->call_contracts.query != NULL;
  if (*out_accepted) {
    return iree_ok_status();
  }
  const loom_op_t* op =
      frame->schedule.nodes[frame->schedule.call_node_indices[0]].op;
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&frame->target)),
      loom_param_string(loom_low_diagnostic_export_name(&frame->target)),
      loom_param_string(loom_low_diagnostic_config_key(&frame->target)),
      loom_param_string(loom_low_diagnostic_symbol_name(
          frame->module, loom_low_func_def_callee(frame->function_op))),
      loom_param_string(loom_low_diagnostic_operation_name(frame->module, op)),
      loom_param_with_field_ref(
          loom_param_string(loom_low_diagnostic_symbol_name(
              frame->module, loom_low_func_call_callee(op))),
          loom_low_func_call_callee_diagnostic_ref()),
      loom_param_string(IREE_SV(
          "the selected target has no native ABI contract for this call")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TARGET_072,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(options->diagnostic_emitter, &emission);
}

static iree_status_t loom_aie2p_leaf_build_frame(
    loom_module_t* module, loom_op_t* function_op,
    const loom_aie2p_leaf_compile_options_t* options,
    loom_aie2p_stack_layout_t* stack_layout, iree_arena_allocator_t* arena,
    loom_low_emission_frame_t* out_frame, bool* out_accepted) {
  loom_low_planning_statistics_t planning_statistics = {0};
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = options->descriptor_registry,
      .function_target_facts = options->function_target_facts,
      .allocation_fixed_values = options->allocation_fixed_values,
      .allocation_fixed_value_count = options->allocation_fixed_value_count,
      .allocation_entry_locations = options->allocation_entry_locations,
      .allocation_entry_location_count =
          options->allocation_entry_location_count,
      .allocation_exit_locations = options->allocation_exit_locations,
      .allocation_exit_location_count = options->allocation_exit_location_count,
      .call_contracts = options->call_contracts,
      .allocation_reserved_ranges = options->allocation_reserved_ranges,
      .allocation_reserved_range_count =
          options->allocation_reserved_range_count,
      .allocation_diagnostic_flags = options->allocation_diagnostic_flags,
      .memory_accesses = options->memory_accesses,
      .loop_execution = options->loop_execution,
      .schedule_structural_models = loom_aie2p_low_structural_schedule_models(),
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .schedule_flags = LOOM_LOW_SCHEDULE_FLAG_RETAIN_DEPENDENCY_INDEX |
                        loom_target_compile_report_required_schedule_flags(
                            options->compile_report),
      .emitter = options->diagnostic_emitter,
      .statistics =
          options->compile_report != NULL ? &planning_statistics : NULL,
  };
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {
      .validate_frame = loom_aie2p_leaf_admit_calls,
      .validate_frame_user_data = (void*)options,
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_STACK,
              .plan_traffic = loom_aie2p_spill_plan,
              .plan_traffic_user_data = stack_layout,
              .record_materialized_spills =
                  loom_target_compile_report_wants_details(
                      options->compile_report,
                      LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS),
              .emitter = options->diagnostic_emitter,
          },
  };
  iree_status_t status = loom_low_emission_frame_build_spill_free(
      module, function_op, &frame_options, &spill_free_options, arena,
      out_frame, out_accepted);
  if (options->compile_report != NULL) {
    loom_target_compile_report_record_low_planning(options->compile_report,
                                                   &planning_statistics);
  }
  if (iree_status_is_ok(status) && options->compile_report != NULL) {
    if (*out_accepted) {
      const loom_target_bundle_t* bundle =
          loom_low_resolved_target_bundle(&out_frame->target);
      if (bundle != NULL) {
        loom_target_compile_report_record_target_bundle(options->compile_report,
                                                        bundle);
      }
      status = loom_target_compile_report_record_low_emission_frame(
          options->compile_report, out_frame);
    } else if (out_frame->allocation.function_op != NULL) {
      status = loom_target_compile_report_record_low_allocation(
          options->compile_report, &out_frame->allocation);
    }
  }
  return status;
}

iree_status_t loom_aie2p_leaf_compile(
    loom_module_t* module, loom_op_t* function_op,
    const loom_aie2p_leaf_compile_options_t* requested_options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_aie2p_leaf_contribution_t* out_contribution) {
  *out_compiled = false;
  *out_contribution = (loom_aie2p_leaf_contribution_t){0};
  loom_diagnostic_export_scope_t diagnostic_scope = {
      .export_name = loom_target_facts_export_name(
          requested_options->function_target_facts),
      .target = requested_options->diagnostic_emitter,
  };
  loom_aie2p_leaf_compile_options_t scoped_options = *requested_options;
  scoped_options.diagnostic_emitter =
      loom_diagnostic_export_scope_emitter(&diagnostic_scope);
  const loom_aie2p_leaf_compile_options_t* options = &scoped_options;
  // The contribution and optional report retain no frame scratch. Schedule,
  // allocation, and packet-planning storage is reusable by the next worker.
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  iree_arena_allocator_t stack_arena;
  iree_arena_initialize(arena->block_pool, &stack_arena);
  loom_aie2p_stack_layout_t stack_layout = {
      .arena = &stack_arena,
      .byte_length =
          options->return_address.kind == LOOM_AIE2P_RETURN_ADDRESS_FRAME
              ? LOOM_AIE2P_RETURN_ADDRESS_FRAME_BYTES
              : 0,
  };
  loom_low_emission_frame_t frame = {0};
  bool frame_accepted = false;
  iree_status_t status =
      loom_aie2p_leaf_build_frame(module, function_op, options, &stack_layout,
                                  &scratch_arena, &frame, &frame_accepted);
  loom_aie2p_stack_frame_t stack = {0};
  bool stack_accepted = false;
  if (iree_status_is_ok(status) && frame_accepted) {
    status = loom_aie2p_stack_frame_build(
        module, function_op, &frame.schedule.requirements.storage_layout,
        &stack_layout, options->callee_stack_byte_length, options->is_helper,
        options->diagnostic_emitter, &stack_accepted, &stack);
  }
  loom_aie2p_leaf_program_plan_t program_plan = {0};
  bool program_planned = false;
  if (iree_status_is_ok(status) && stack_accepted) {
    const loom_aie2p_bundle_plan_options_t plan_options = {
        .stack = stack,
        .diagnostic_emitter = options->diagnostic_emitter,
        .return_address = options->return_address,
    };
    status = loom_aie2p_bundle_plan_build(&frame, &plan_options, &scratch_arena,
                                          &program_planned, &program_plan);
  }
  if (iree_status_is_ok(status) && program_planned &&
      options->compile_report != NULL) {
    loom_target_compile_report_emission_breakdown_t emission_breakdown = {
        .body_instruction_count = program_plan.issue_cycle_count,
    };
    for (iree_host_size_t i = 0; i < program_plan.bundle_count; ++i) {
      const uint8_t slot_count = program_plan.bundles[i].slot_count;
      if (slot_count <= 1) {
        continue;
      }
      ++emission_breakdown.coissued_instruction_count;
      emission_breakdown.coissued_component_count += slot_count;
    }
    loom_target_compile_report_record_emission(
        options->compile_report, program_plan.issue_cycle_count,
        program_plan.encoded_byte_length, program_plan.encoded_byte_length);
    loom_target_compile_report_record_emission_breakdown(
        options->compile_report, &emission_breakdown);
  }
  bool capacity_admitted = false;
  if (iree_status_is_ok(status) && program_planned) {
    status = loom_aie2p_program_capacity_admit(
        module, function_op, program_plan.encoded_byte_length,
        options->diagnostic_emitter, &capacity_admitted);
  }
  if (iree_status_is_ok(status) && program_planned && capacity_admitted) {
    status =
        loom_aie2p_leaf_object_emit(&program_plan, arena, out_contribution);
  }
  if (iree_status_is_ok(status) && program_planned && capacity_admitted) {
    *out_compiled = true;
  }
  iree_arena_deinitialize(&stack_arena);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
