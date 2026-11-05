// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/function_compile.h"

#include "loom/codegen/low/allocation_materialization.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/planning/occupancy.h"
#include "loom/target/arch/amdgpu/planning/placement.h"
#include "loom/target/arch/amdgpu/planning/storage_lease.h"
#include "loom/target/arch/amdgpu/planning/vopd_plan.h"
#include "loom/target/emit/native/amdgpu/function_plan_build.h"
#include "loom/target/emit/native/amdgpu/spill_lowering.h"
#include "loom/target/reporting/low.h"

typedef struct loom_amdgpu_function_compile_spill_context_t {
  // Selected descriptor set used to rewrite structural spill traffic.
  const loom_low_descriptor_set_t* descriptor_set;
} loom_amdgpu_function_compile_spill_context_t;

static iree_status_t loom_amdgpu_function_compile_lower_spill_traffic(
    void* user_data, loom_module_t* module, loom_op_t* low_function_op,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_emission_frame_lower_spill_traffic_result_t* out_result) {
  const loom_amdgpu_function_compile_spill_context_t* context = user_data;
  loom_amdgpu_spill_lowering_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_spill_traffic(
      module, low_function_op, context->descriptor_set, emitter, arena,
      &result));
  out_result->error_count = result.error_count;
  out_result->lowered_traffic = result.lowered_traffic;
  out_result->required_register_value_ids = result.required_register_value_ids;
  out_result->required_register_value_count =
      result.required_register_value_count;
  out_result->expanded_execution_value_ids =
      result.expanded_execution_value_ids;
  out_result->expanded_execution_value_count =
      result.expanded_execution_value_count;
  out_result->per_user_placement_value_ids =
      result.per_user_placement_value_ids;
  out_result->per_user_placement_value_count =
      result.per_user_placement_value_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_function_compile_record_rejection(
    const loom_amdgpu_function_compilation_t* compilation,
    loom_target_compile_report_t* report) {
  if (report == NULL) {
    return iree_ok_status();
  }
  loom_target_compile_report_record_low_planning(
      report, &compilation->planning_statistics);
  if (compilation->frame.allocation.function_op != NULL) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_record_low_allocation(
        report, &compilation->frame.allocation));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_function_compile(
    loom_module_t* module, loom_op_t* function_op,
    const loom_amdgpu_function_compile_options_t* options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_amdgpu_function_compilation_t* out_compilation) {
  *out_compiled = false;
  *out_compilation = (loom_amdgpu_function_compilation_t){0};

  loom_low_schedule_pair_affinity_list_t schedule_pair_affinities =
      loom_low_schedule_pair_affinity_list_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_build_schedule_pair_affinities(
      options->resolved_target, arena, &schedule_pair_affinities));
  const loom_low_schedule_structural_state_read_list_t schedule_state_reads =
      loom_amdgpu_descriptor_structural_state_reads();
  loom_low_storage_lease_provider_t storage_lease_provider = {0};
  loom_amdgpu_storage_lease_provider(&storage_lease_provider);
  const loom_low_schedule_structural_model_t storage_address_model =
      loom_amdgpu_descriptor_storage_address_model(
          options->resolved_target->descriptor_set);
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = options->descriptor_registry,
      .resolved_target = options->resolved_target,
      .residency_query = loom_amdgpu_occupancy_residency_view,
      .schedule_pair_affinities = schedule_pair_affinities,
      .instruction_preferences = loom_amdgpu_placement_instruction_preferences(
          options->resolved_target),
      .schedule_structural_state_reads = schedule_state_reads,
      .schedule_structural_models = {.values = &storage_address_model,
                                     .count = 1},
      .schedule_flags =
          LOOM_LOW_SCHEDULE_FLAG_RETAIN_VALUE_PRODUCER_NODES |
          LOOM_LOW_SCHEDULE_FLAG_RETAIN_ACYCLIC_MEMORY_COMPLETIONS,
      .allocation_flags =
          LOOM_LOW_ALLOCATION_FLAG_RETAIN_COALESCED_INCOMING_INDEX |
          LOOM_LOW_ALLOCATION_FLAG_RETAIN_STORAGE_RELEASE_ACTION_INDEX,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .memory_accesses = options->memory_accesses,
      .loop_execution = options->loop_execution,
      .allocation_fixed_values = options->allocation_fixed_values,
      .allocation_fixed_value_count = options->allocation_fixed_value_count,
      .allocation_entry_values = options->allocation_entry_values,
      .allocation_entry_value_count = options->allocation_entry_value_count,
      .allocation_entry_locations = options->allocation_entry_locations,
      .allocation_entry_location_count =
          options->allocation_entry_location_count,
      .allocation_exit_locations = options->allocation_exit_locations,
      .allocation_exit_location_count = options->allocation_exit_location_count,
      .call_contracts = options->call_contracts,
      .allocation_reserved_ranges = options->allocation_reserved_ranges,
      .allocation_reserved_range_count =
          options->allocation_reserved_range_count,
      .storage_lease_provider = &storage_lease_provider,
      .emitter = options->emitter,
      .statistics = options->report != NULL
                        ? &out_compilation->planning_statistics
                        : NULL,
  };
  loom_amdgpu_function_compile_spill_context_t spill_context = {
      .descriptor_set = options->resolved_target->descriptor_set,
  };
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_SCRATCH |
                                          LOOM_LOW_STORAGE_SPACE_SET_PRIVATE,
              .record_materialized_spills =
                  loom_target_compile_report_wants_details(
                      options->report,
                      LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS),
              .plan_traffic = loom_amdgpu_plan_spill_materialization_traffic,
              .emitter = options->emitter,
          },
      .lower_spill_traffic = loom_amdgpu_function_compile_lower_spill_traffic,
      .lower_spill_traffic_user_data = &spill_context,
  };
  bool frame_accepted = false;
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build_spill_free(
      module, function_op, &frame_options, &spill_free_options, arena,
      &out_compilation->frame, &frame_accepted));
  if (!frame_accepted) {
    return loom_amdgpu_function_compile_record_rejection(out_compilation,
                                                         options->report);
  }

  bool plan_accepted = false;
  const loom_amdgpu_packet_plan_options_t packet_options = {
      .call_states = options->call_states,
      .callable = options->callable,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_function_plan_build(
      &out_compilation->frame, options->emitter, &packet_options, arena,
      &plan_accepted, &out_compilation->plan));
  if (!plan_accepted) {
    return loom_amdgpu_function_compile_record_rejection(out_compilation,
                                                         options->report);
  }
  *out_compiled = true;
  return iree_ok_status();
}
