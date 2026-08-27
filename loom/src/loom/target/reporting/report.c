// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/report.h"

#include <string.h>

#include "loom/target/reporting/memory_summary.h"
#include "loom/target/reporting/row_list.h"

void loom_target_compile_report_initialize(
    loom_target_compile_report_t* out_report, iree_allocator_t allocator) {
  *out_report = (loom_target_compile_report_t){
      .allocator = allocator,
      .status_code = IREE_STATUS_OK,
  };
}

void loom_target_compile_report_deinitialize(
    loom_target_compile_report_t* report) {
  if (report == NULL) {
    return;
  }
  const iree_allocator_t allocator = report->allocator;
  loom_target_compile_report_row_list_deinitialize(allocator,
                                                   &report->entry_rows);
  loom_target_compile_report_row_list_deinitialize(allocator,
                                                   &report->pressure_summaries);
  loom_target_compile_report_row_list_deinitialize(allocator,
                                                   &report->pressure_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->pressure_origin_rows);
  loom_target_compile_report_row_list_deinitialize(allocator,
                                                   &report->schedule_band_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->schedule_band_summary_rows);
  loom_target_compile_report_row_list_deinitialize(allocator,
                                                   &report->spill_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->allocation_failure_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->allocation_high_water_rows);
  loom_target_compile_report_row_list_deinitialize(allocator,
                                                   &report->wait_counter_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->wait_reason_summary_rows);
  loom_target_compile_report_row_list_deinitialize(allocator,
                                                   &report->wait_action_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->target_insertion_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->config_binding_rows);
  loom_target_compile_report_row_list_deinitialize(allocator,
                                                   &report->source_low_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_target_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_transform_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_selection_summaries);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_memory_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_memory_root_summaries);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_memory_argument_summaries);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_memory_argument_packet_summaries);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_memory_strategy_summaries);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_bank_service_summaries);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_subgroup_access_summaries);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->math_legalization_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->target_legalization_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->target_capability_rows);
  *report = (loom_target_compile_report_t){0};
}

static bool loom_target_compile_report_has_rows(
    const loom_target_compile_report_t* report) {
  return report->pressure_rows.count != 0 || report->spill_rows.count != 0 ||
         report->pressure_summaries.count != 0 ||
         report->pressure_origin_rows.count != 0 ||
         report->schedule_band_rows.count != 0 ||
         report->schedule_band_summary_rows.count != 0 ||
         report->allocation_failure_rows.count != 0 ||
         report->allocation_high_water_rows.count != 0 ||
         report->wait_counter_rows.count != 0 ||
         report->wait_reason_summary_rows.count != 0 ||
         report->wait_action_rows.count != 0 ||
         report->target_insertion_rows.count != 0 ||
         report->config_binding_rows.count != 0 ||
         report->entry_rows.count != 0 || report->source_low_rows.count != 0 ||
         report->source_low_target_rows.count != 0 ||
         report->source_low_transform_rows.count != 0 ||
         report->source_low_selection_summaries.count != 0 ||
         report->source_low_memory_rows.count != 0 ||
         report->source_low_memory_root_summaries.count != 0 ||
         report->source_low_memory_argument_summaries.count != 0 ||
         report->source_low_memory_argument_packet_summaries.count != 0 ||
         report->source_low_memory_strategy_summaries.count != 0 ||
         report->source_low_bank_service_summaries.count != 0 ||
         report->source_low_subgroup_access_summaries.count != 0 ||
         report->math_legalization_rows.count != 0 ||
         report->target_legalization_rows.count != 0 ||
         report->target_capability_rows.count != 0;
}

void loom_target_compile_report_initialize_if_empty(
    loom_target_compile_report_t* report, iree_allocator_t allocator) {
  if (report->detail_flags != LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE ||
      report->requested_detail_flags !=
          LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE ||
      loom_target_compile_report_has_rows(report)) {
    return;
  }
  const loom_target_compile_report_detail_flags_t requested_detail_flags =
      report->requested_detail_flags;
  loom_target_compile_report_initialize(report, allocator);
  report->requested_detail_flags = requested_detail_flags;
}

iree_status_t loom_target_compile_report_clone(
    const loom_target_compile_report_t* source, iree_allocator_t allocator,
    loom_target_compile_report_t* out_target) {
  loom_target_compile_report_t target = *source;
  target.allocator = allocator;
  target.entry_rows = (loom_target_compile_report_row_list_t){0};
  target.pressure_summaries = (loom_target_compile_report_row_list_t){0};
  target.pressure_rows = (loom_target_compile_report_row_list_t){0};
  target.pressure_origin_rows = (loom_target_compile_report_row_list_t){0};
  target.schedule_band_rows = (loom_target_compile_report_row_list_t){0};
  target.schedule_band_summary_rows =
      (loom_target_compile_report_row_list_t){0};
  target.spill_rows = (loom_target_compile_report_row_list_t){0};
  target.allocation_failure_rows = (loom_target_compile_report_row_list_t){0};
  target.allocation_high_water_rows =
      (loom_target_compile_report_row_list_t){0};
  target.wait_counter_rows = (loom_target_compile_report_row_list_t){0};
  target.wait_reason_summary_rows = (loom_target_compile_report_row_list_t){0};
  target.wait_action_rows = (loom_target_compile_report_row_list_t){0};
  target.target_insertion_rows = (loom_target_compile_report_row_list_t){0};
  target.config_binding_rows = (loom_target_compile_report_row_list_t){0};
  target.source_low_rows = (loom_target_compile_report_row_list_t){0};
  target.source_low_target_rows = (loom_target_compile_report_row_list_t){0};
  target.source_low_transform_rows = (loom_target_compile_report_row_list_t){0};
  target.source_low_selection_summaries =
      (loom_target_compile_report_row_list_t){0};
  target.source_low_memory_rows = (loom_target_compile_report_row_list_t){0};
  target.source_low_memory_root_summaries =
      (loom_target_compile_report_row_list_t){0};
  target.source_low_memory_argument_summaries =
      (loom_target_compile_report_row_list_t){0};
  target.source_low_memory_argument_packet_summaries =
      (loom_target_compile_report_row_list_t){0};
  target.source_low_memory_strategy_summaries =
      (loom_target_compile_report_row_list_t){0};
  target.source_low_bank_service_summaries =
      (loom_target_compile_report_row_list_t){0};
  target.source_low_subgroup_access_summaries =
      (loom_target_compile_report_row_list_t){0};
  target.math_legalization_rows = (loom_target_compile_report_row_list_t){0};
  target.target_legalization_rows = (loom_target_compile_report_row_list_t){0};
  target.target_capability_rows = (loom_target_compile_report_row_list_t){0};
  if (source->entry_rows.count == 0 && source->pressure_summaries.count == 0 &&
      source->pressure_rows.count == 0 &&
      source->pressure_origin_rows.count == 0 &&
      source->schedule_band_rows.count == 0 && source->spill_rows.count == 0 &&
      source->schedule_band_summary_rows.count == 0 &&
      source->allocation_failure_rows.count == 0 &&
      source->allocation_high_water_rows.count == 0 &&
      source->wait_counter_rows.count == 0 &&
      source->wait_reason_summary_rows.count == 0 &&
      source->wait_action_rows.count == 0 &&
      source->target_insertion_rows.count == 0 &&
      source->config_binding_rows.count == 0 &&
      source->source_low_rows.count == 0 &&
      source->source_low_target_rows.count == 0 &&
      source->source_low_transform_rows.count == 0 &&
      source->source_low_selection_summaries.count == 0 &&
      source->source_low_memory_rows.count == 0 &&
      source->source_low_memory_root_summaries.count == 0 &&
      source->source_low_memory_argument_summaries.count == 0 &&
      source->source_low_memory_argument_packet_summaries.count == 0 &&
      source->source_low_memory_strategy_summaries.count == 0 &&
      source->source_low_bank_service_summaries.count == 0 &&
      source->source_low_subgroup_access_summaries.count == 0 &&
      source->math_legalization_rows.count == 0 &&
      source->target_legalization_rows.count == 0 &&
      source->target_capability_rows.count == 0) {
    *out_target = target;
    return iree_ok_status();
  }
  if (iree_allocator_is_null(allocator)) {
    *out_target = target;
    return iree_ok_status();
  }
  iree_status_t status = loom_target_compile_report_row_list_clone(
      &source->entry_rows, sizeof(loom_target_compile_report_entry_t),
      allocator, &target.entry_rows);
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->pressure_summaries,
        sizeof(loom_target_compile_report_pressure_summary_t), allocator,
        &target.pressure_summaries);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->pressure_rows,
        sizeof(loom_target_compile_report_pressure_row_t), allocator,
        &target.pressure_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->pressure_origin_rows,
        sizeof(loom_target_compile_report_pressure_origin_row_t), allocator,
        &target.pressure_origin_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->schedule_band_rows,
        sizeof(loom_target_compile_report_schedule_band_row_t), allocator,
        &target.schedule_band_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->schedule_band_summary_rows,
        sizeof(loom_target_compile_report_schedule_band_summary_row_t),
        allocator, &target.schedule_band_summary_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->spill_rows, sizeof(loom_target_compile_report_spill_row_t),
        allocator, &target.spill_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->allocation_failure_rows,
        sizeof(loom_target_compile_report_allocation_failure_row_t), allocator,
        &target.allocation_failure_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->allocation_high_water_rows,
        sizeof(loom_target_compile_report_allocation_high_water_row_t),
        allocator, &target.allocation_high_water_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->wait_counter_rows,
        sizeof(loom_target_compile_report_wait_counter_row_t), allocator,
        &target.wait_counter_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->wait_reason_summary_rows,
        sizeof(loom_target_compile_report_wait_reason_summary_row_t), allocator,
        &target.wait_reason_summary_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->wait_action_rows,
        sizeof(loom_target_compile_report_wait_action_row_t), allocator,
        &target.wait_action_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->target_insertion_rows,
        sizeof(loom_target_compile_report_target_insertion_row_t), allocator,
        &target.target_insertion_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->config_binding_rows,
        sizeof(loom_target_compile_report_config_binding_row_t), allocator,
        &target.config_binding_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_rows,
        sizeof(loom_target_compile_report_source_low_row_t), allocator,
        &target.source_low_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_target_rows,
        sizeof(loom_target_compile_report_source_low_target_row_t), allocator,
        &target.source_low_target_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_transform_rows,
        sizeof(loom_target_compile_report_source_low_transform_row_t),
        allocator, &target.source_low_transform_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_selection_summaries,
        sizeof(loom_target_compile_report_source_low_selection_summary_t),
        allocator, &target.source_low_selection_summaries);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_memory_rows,
        sizeof(loom_target_compile_report_source_low_memory_row_t), allocator,
        &target.source_low_memory_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_memory_root_summaries,
        sizeof(loom_target_compile_report_source_low_memory_root_summary_t),
        allocator, &target.source_low_memory_root_summaries);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_memory_argument_summaries,
        sizeof(loom_target_compile_report_source_low_memory_argument_summary_t),
        allocator, &target.source_low_memory_argument_summaries);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_memory_argument_packet_summaries,
        sizeof(
            loom_target_compile_report_source_low_memory_argument_packet_summary_t),
        allocator, &target.source_low_memory_argument_packet_summaries);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_memory_strategy_summaries,
        sizeof(loom_target_compile_report_source_low_memory_strategy_summary_t),
        allocator, &target.source_low_memory_strategy_summaries);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_bank_service_summaries,
        sizeof(loom_target_compile_report_source_low_bank_service_summary_t),
        allocator, &target.source_low_bank_service_summaries);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->source_low_subgroup_access_summaries,
        sizeof(loom_target_compile_report_source_low_subgroup_access_summary_t),
        allocator, &target.source_low_subgroup_access_summaries);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->math_legalization_rows,
        sizeof(loom_target_compile_report_math_row_t), allocator,
        &target.math_legalization_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->target_legalization_rows,
        sizeof(loom_target_compile_report_legalization_row_t), allocator,
        &target.target_legalization_rows);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_row_list_clone(
        &source->target_capability_rows,
        sizeof(loom_target_compile_report_target_capability_row_t), allocator,
        &target.target_capability_rows);
  }
  if (!iree_status_is_ok(status)) {
    loom_target_compile_report_deinitialize(&target);
    return status;
  }
  *out_target = target;
  return iree_ok_status();
}

void loom_target_compile_report_record_status(
    loom_target_compile_report_t* report, iree_status_code_t status_code) {
  report->status_code = status_code;
}

void loom_target_compile_report_record_target_bundle(
    loom_target_compile_report_t* report, const loom_target_bundle_t* bundle) {
  if (bundle == NULL) {
    return;
  }
  report->target_bundle_name = bundle->name;
  if (bundle->snapshot != NULL) {
    report->target_snapshot_name = bundle->snapshot->name;
  }
  if (bundle->export_plan != NULL) {
    report->target_export_name = bundle->export_plan->name;
    report->target_export_symbol = bundle->export_plan->export_symbol;
  }
  if (bundle->config != NULL) {
    report->target_config_name = bundle->config->name;
  }
}

void loom_target_compile_report_record_artifact_size(
    loom_target_compile_report_t* report, uint64_t artifact_size) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE;
  report->artifact_size = artifact_size;
}

void loom_target_compile_report_record_schedule(
    loom_target_compile_report_t* report, uint64_t node_count,
    uint64_t scheduled_node_count, uint64_t dependency_count,
    uint64_t resource_use_count, uint64_t hazard_gap_count,
    uint64_t model_summary_count, uint64_t pressure_summary_count,
    uint64_t peak_live_units) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE;
  report->schedule_node_count = node_count;
  report->scheduled_node_count = scheduled_node_count;
  report->schedule_dependency_count = dependency_count;
  report->schedule_resource_use_count = resource_use_count;
  report->schedule_hazard_gap_count = hazard_gap_count;
  report->schedule_model_summary_count = model_summary_count;
  report->register_pressure_summary_count = pressure_summary_count;
  report->register_pressure_peak_live_units = peak_live_units;
}

void loom_target_compile_report_record_allocation(
    loom_target_compile_report_t* report, uint64_t assignment_count,
    uint64_t spill_count, uint64_t spill_plan_count,
    uint64_t coalesced_copy_count, uint64_t materialized_copy_count,
    uint64_t storage_lease_count, uint64_t storage_lease_instance_count,
    uint64_t storage_release_action_count) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION;
  report->allocation_assignment_count = assignment_count;
  report->allocation_spill_count = spill_count;
  report->allocation_spill_plan_count = spill_plan_count;
  report->allocation_coalesced_copy_count = coalesced_copy_count;
  report->allocation_materialized_copy_count = materialized_copy_count;
  report->allocation_storage_lease_count = storage_lease_count;
  report->allocation_storage_lease_instance_count =
      storage_lease_instance_count;
  report->allocation_storage_release_action_count =
      storage_release_action_count;
}

void loom_target_compile_report_record_allocation_materialization(
    loom_target_compile_report_t* report, uint64_t spill_storage_count,
    uint64_t spill_storage_bytes, uint64_t spill_store_count,
    uint64_t spill_store_bytes, uint64_t reload_count, uint64_t reload_bytes) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION;
  report->allocation_materialized_spill_storage_count += spill_storage_count;
  report->allocation_materialized_spill_storage_bytes += spill_storage_bytes;
  report->allocation_materialized_spill_store_count += spill_store_count;
  report->allocation_materialized_spill_store_bytes += spill_store_bytes;
  report->allocation_materialized_reload_count += reload_count;
  report->allocation_materialized_reload_bytes += reload_bytes;
}

void loom_target_compile_report_record_move_cause(
    loom_target_compile_report_t* report,
    loom_target_compile_report_move_cause_t cause, uint64_t packet_count,
    uint64_t unit_count) {
  if (packet_count == 0 && unit_count == 0) {
    return;
  }
  IREE_ASSERT(cause > LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_NONE &&
                  cause < LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT,
              "invalid residual move cause");
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES;
  loom_target_compile_report_move_cause_counts_t* counts =
      &report->move_causes[cause];
  counts->packet_count += packet_count;
  counts->unit_count += unit_count;
}

void loom_target_compile_report_record_static_instruction_mix(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_static_instruction_mix_t* mix) {
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX;
  report->static_instruction_mix = *mix;
}

void loom_target_compile_report_record_dynamic_instruction_mix(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_static_instruction_mix_t* mix) {
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX;
  report->dynamic_instruction_mix = *mix;
}

void loom_target_compile_report_record_emission(
    loom_target_compile_report_t* report, uint64_t instruction_count,
    uint64_t code_byte_count, uint64_t code_storage_byte_count) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION;
  report->emitted_instruction_count = instruction_count;
  report->emitted_code_byte_count = code_byte_count;
  report->emitted_code_storage_byte_count = code_storage_byte_count;
}

void loom_target_compile_report_record_emission_breakdown(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_emission_breakdown_t* breakdown) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION_BREAKDOWN;
  report->emission_breakdown = *breakdown;
}

void loom_target_compile_report_record_memory(
    loom_target_compile_report_t* report, uint64_t private_memory_bytes,
    uint64_t local_memory_bytes) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY;
  report->private_memory_bytes = private_memory_bytes;
  report->local_memory_bytes = local_memory_bytes;
}

static uint64_t loom_target_compile_report_register_overhead_units(
    uint64_t final_register_count, uint64_t pressure_peak_live_units) {
  if (final_register_count <= pressure_peak_live_units) {
    return 0;
  }
  return final_register_count - pressure_peak_live_units;
}

static void loom_target_compile_report_update_target_resource_pressure_summary(
    loom_target_compile_report_target_resources_t* target_resources,
    const loom_target_compile_report_pressure_summary_t* summary) {
  if (iree_string_view_equal(summary->register_class,
                             target_resources->scalar_register_class)) {
    target_resources->scalar_pressure_peak_live_units =
        iree_max(target_resources->scalar_pressure_peak_live_units,
                 summary->peak_live_units);
    target_resources->scalar_register_overhead_units =
        loom_target_compile_report_register_overhead_units(
            target_resources->scalar_register_count,
            target_resources->scalar_pressure_peak_live_units);
  }
  if (iree_string_view_equal(summary->register_class,
                             target_resources->vector_register_class)) {
    target_resources->vector_pressure_peak_live_units =
        iree_max(target_resources->vector_pressure_peak_live_units,
                 summary->peak_live_units);
    target_resources->vector_register_overhead_units =
        loom_target_compile_report_register_overhead_units(
            target_resources->vector_register_count,
            target_resources->vector_pressure_peak_live_units);
  }
}

static void loom_target_compile_report_populate_target_resource_pressure(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_target_resources_t* target_resources) {
  target_resources->scalar_pressure_peak_live_units = 0;
  target_resources->scalar_register_overhead_units = 0;
  target_resources->vector_pressure_peak_live_units = 0;
  target_resources->vector_register_overhead_units = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->pressure_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_pressure_summary_t* summaries =
        (const loom_target_compile_report_pressure_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      loom_target_compile_report_update_target_resource_pressure_summary(
          target_resources, &summaries[i]);
    }
  }
}

void loom_target_compile_report_record_target_resources(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_target_resources_t* target_resources) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES;
  report->target_resources = *target_resources;
  loom_target_compile_report_populate_target_resource_pressure(
      report, &report->target_resources);
}

iree_status_t loom_target_compile_report_record_pressure_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_pressure_summary_t* summary) {
  for (loom_target_compile_report_vec_t* vec = report->pressure_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_pressure_summary_t* summaries =
        (loom_target_compile_report_pressure_summary_t*)
            loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      if (iree_string_view_equal(summaries[i].register_class,
                                 summary->register_class)) {
        if (summary->peak_live_units > summaries[i].peak_live_units) {
          summaries[i].peak_live_units = summary->peak_live_units;
          if (iree_any_bit_set(
                  report->detail_flags,
                  LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
            loom_target_compile_report_update_target_resource_pressure_summary(
                &report->target_resources, &summaries[i]);
          }
        }
        return iree_ok_status();
      }
    }
  }

  IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append(
      &report->pressure_summaries, sizeof(*summary), report->allocator,
      summary));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    loom_target_compile_report_update_target_resource_pressure_summary(
        &report->target_resources, summary);
  }
  return iree_ok_status();
}

static void loom_target_compile_report_accumulate_wait_plan(
    loom_target_compile_report_wait_plan_t* target,
    const loom_target_compile_report_wait_plan_t* source) {
  target->action_count += source->action_count;
  target->explicit_action_count += source->explicit_action_count;
  target->planned_action_count += source->planned_action_count;
  target->full_drain_count += source->full_drain_count;
  target->partial_wait_count += source->partial_wait_count;
  target->drained_count += source->drained_count;
  target->max_drained_count =
      iree_max(target->max_drained_count, source->max_drained_count);
  target->max_outstanding_before =
      iree_max(target->max_outstanding_before, source->max_outstanding_before);
  target->max_full_drain_outstanding_before =
      iree_max(target->max_full_drain_outstanding_before,
               source->max_full_drain_outstanding_before);
}

void loom_target_compile_report_record_wait_plan(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_wait_plan_t* wait_plan) {
  if (wait_plan->action_count == 0) {
    return;
  }
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN;
  loom_target_compile_report_accumulate_wait_plan(&report->wait_plan,
                                                  wait_plan);
}

static bool loom_target_compile_report_workgroup_sizes_equal(
    loom_target_workgroup_size_t lhs, loom_target_workgroup_size_t rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

static bool loom_target_compile_report_workgroup_counts_equal(
    loom_target_dispatch_workgroup_count_t lhs,
    loom_target_dispatch_workgroup_count_t rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

static bool loom_target_compile_report_workgroup_cluster_sizes_equal(
    loom_target_workgroup_cluster_size_t lhs,
    loom_target_workgroup_cluster_size_t rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

static bool loom_target_compile_report_checked_mul3_u32(uint32_t x, uint32_t y,
                                                        uint32_t z,
                                                        uint64_t* out_result) {
  uint64_t xy = 0;
  return iree_checked_mul_u64(x, y, &xy) &&
         iree_checked_mul_u64(xy, z, out_result);
}

void loom_target_compile_report_record_workload(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_workload_t* workload) {
  if (workload->flags == LOOM_TARGET_COMPILE_REPORT_WORKLOAD_NONE) {
    return;
  }
  if (report->workload.flags == LOOM_TARGET_COMPILE_REPORT_WORKLOAD_NONE) {
    report->workload = *workload;
  } else {
    loom_target_compile_report_workload_t merged = {0};
    bool has_workgroup_size =
        iree_any_bit_set(report->workload.flags,
                         LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE);
    if (iree_any_bit_set(workload->flags,
                         LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE)) {
      if (!has_workgroup_size) {
        merged.workgroup_size = workload->workgroup_size;
        has_workgroup_size = true;
      } else if (loom_target_compile_report_workgroup_sizes_equal(
                     report->workload.workgroup_size,
                     workload->workgroup_size)) {
        merged.workgroup_size = report->workload.workgroup_size;
      } else {
        has_workgroup_size = false;
      }
    } else if (has_workgroup_size) {
      merged.workgroup_size = report->workload.workgroup_size;
    }
    if (has_workgroup_size) {
      merged.flags |= LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE;
      if (loom_target_compile_report_checked_mul3_u32(
              merged.workgroup_size.x, merged.workgroup_size.y,
              merged.workgroup_size.z, &merged.flat_workgroup_size)) {
        merged.flags |= LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE;
      }
    }

    bool has_workgroup_count =
        iree_any_bit_set(report->workload.flags,
                         LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT);
    if (iree_any_bit_set(workload->flags,
                         LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT)) {
      if (!has_workgroup_count) {
        merged.workgroup_count = workload->workgroup_count;
        has_workgroup_count = true;
      } else if (loom_target_compile_report_workgroup_counts_equal(
                     report->workload.workgroup_count,
                     workload->workgroup_count)) {
        merged.workgroup_count = report->workload.workgroup_count;
      } else {
        has_workgroup_count = false;
      }
    } else if (has_workgroup_count) {
      merged.workgroup_count = report->workload.workgroup_count;
    }
    if (has_workgroup_count) {
      merged.flags |= LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT;
      if (loom_target_compile_report_checked_mul3_u32(
              merged.workgroup_count.x, merged.workgroup_count.y,
              merged.workgroup_count.z, &merged.dispatch_workgroup_count)) {
        merged.flags |=
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT;
      }
    }

    bool has_workgroup_cluster_size = iree_any_bit_set(
        report->workload.flags,
        LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE);
    if (iree_any_bit_set(
            workload->flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE)) {
      if (!has_workgroup_cluster_size) {
        merged.workgroup_cluster_size = workload->workgroup_cluster_size;
        has_workgroup_cluster_size = true;
      } else if (loom_target_compile_report_workgroup_cluster_sizes_equal(
                     report->workload.workgroup_cluster_size,
                     workload->workgroup_cluster_size)) {
        merged.workgroup_cluster_size = report->workload.workgroup_cluster_size;
      } else {
        has_workgroup_cluster_size = false;
      }
    } else if (has_workgroup_cluster_size) {
      merged.workgroup_cluster_size = report->workload.workgroup_cluster_size;
    }
    if (has_workgroup_cluster_size) {
      merged.flags |=
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE;
      if (loom_target_compile_report_checked_mul3_u32(
              merged.workgroup_cluster_size.x, merged.workgroup_cluster_size.y,
              merged.workgroup_cluster_size.z,
              &merged.flat_workgroup_cluster_size)) {
        merged.flags |=
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_CLUSTER_SIZE;
      }
    }

    if (iree_all_bits_set(
            merged.flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE |
                LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT) &&
        iree_checked_mul_u64(merged.flat_workgroup_size,
                             merged.dispatch_workgroup_count,
                             &merged.dispatch_workitem_count)) {
      merged.flags |=
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT;
    }
    report->workload = merged;
  }
  if (report->workload.flags == LOOM_TARGET_COMPILE_REPORT_WORKLOAD_NONE) {
    report->detail_flags &= ~LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD;
  } else {
    report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD;
  }
}

static void loom_target_compile_report_accumulate_instruction_mix(
    loom_target_compile_report_static_instruction_mix_t* target,
    const loom_target_compile_report_static_instruction_mix_t* source) {
  target->descriptor_count += source->descriptor_count;
  target->unknown_count += source->unknown_count;
  target->scalar_alu_count += source->scalar_alu_count;
  target->vector_alu_count += source->vector_alu_count;
  target->matrix_count += source->matrix_count;
  target->mfma_count += source->mfma_count;
  target->smfmac_count += source->smfmac_count;
  target->wmma_count += source->wmma_count;
  target->swmmac_count += source->swmmac_count;
  target->dot_count += source->dot_count;
  target->global_memory_count += source->global_memory_count;
  target->global_load_count += source->global_load_count;
  target->global_store_count += source->global_store_count;
  target->buffer_load_count += source->buffer_load_count;
  target->buffer_store_count += source->buffer_store_count;
  target->flat_memory_count += source->flat_memory_count;
  target->local_memory_count += source->local_memory_count;
  target->scalar_memory_count += source->scalar_memory_count;
  target->private_memory_count += source->private_memory_count;
  target->generic_memory_count += source->generic_memory_count;
  target->memory_read_unknown_width_count +=
      source->memory_read_unknown_width_count;
  target->memory_write_unknown_width_count +=
      source->memory_write_unknown_width_count;
  target->memory_read_byte_count += source->memory_read_byte_count;
  target->memory_write_byte_count += source->memory_write_byte_count;
  target->global_load_byte_count += source->global_load_byte_count;
  target->global_store_byte_count += source->global_store_byte_count;
  target->buffer_load_byte_count += source->buffer_load_byte_count;
  target->buffer_store_byte_count += source->buffer_store_byte_count;
  target->flat_read_byte_count += source->flat_read_byte_count;
  target->flat_write_byte_count += source->flat_write_byte_count;
  target->local_read_byte_count += source->local_read_byte_count;
  target->local_write_byte_count += source->local_write_byte_count;
  target->scalar_read_byte_count += source->scalar_read_byte_count;
  target->scalar_write_byte_count += source->scalar_write_byte_count;
  target->private_read_byte_count += source->private_read_byte_count;
  target->private_write_byte_count += source->private_write_byte_count;
  target->unclassified_read_byte_count += source->unclassified_read_byte_count;
  target->unclassified_write_byte_count +=
      source->unclassified_write_byte_count;
  target->atomic_count += source->atomic_count;
  target->branch_count += source->branch_count;
  target->barrier_count += source->barrier_count;
  target->control_count += source->control_count;
  target->conversion_count += source->conversion_count;
  target->cache_count += source->cache_count;
  target->register_move_count += source->register_move_count;
}

static void loom_target_compile_report_accumulate_move_causes(
    loom_target_compile_report_move_cause_counts_t* target,
    const loom_target_compile_report_move_cause_counts_t* source) {
  for (iree_host_size_t i = 0; i < LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT;
       ++i) {
    target[i].packet_count += source[i].packet_count;
    target[i].unit_count += source[i].unit_count;
  }
}

static bool loom_target_compile_report_strings_equal_or_empty(
    iree_string_view_t lhs, iree_string_view_t rhs) {
  return iree_string_view_is_empty(lhs) || iree_string_view_is_empty(rhs) ||
         iree_string_view_equal(lhs, rhs);
}

static iree_string_view_t loom_target_compile_report_shared_string(
    iree_string_view_t current, iree_string_view_t next) {
  return loom_target_compile_report_strings_equal_or_empty(current, next)
             ? (!iree_string_view_is_empty(current) ? current : next)
             : iree_string_view_empty();
}

static uint32_t loom_target_compile_report_shared_u32(uint32_t current,
                                                      uint32_t next) {
  return current == next ? current : 0;
}

static void loom_target_compile_report_merge_workload(
    loom_target_compile_report_workload_t* target,
    const loom_target_compile_report_workload_t* source) {
  loom_target_compile_report_workload_flags_t flags =
      target->flags & source->flags;
  if (iree_any_bit_set(flags,
                       LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE) &&
      !loom_target_compile_report_workgroup_sizes_equal(
          target->workgroup_size, source->workgroup_size)) {
    flags &= ~LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE;
  }
  if (iree_any_bit_set(flags,
                       LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT) &&
      !loom_target_compile_report_workgroup_counts_equal(
          target->workgroup_count, source->workgroup_count)) {
    flags &= ~LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT;
  }
  if (iree_any_bit_set(
          flags, LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE) &&
      !loom_target_compile_report_workgroup_cluster_sizes_equal(
          target->workgroup_cluster_size, source->workgroup_cluster_size)) {
    flags &= ~LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE;
  }
  if (iree_any_bit_set(
          flags, LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE) &&
      target->flat_workgroup_size != source->flat_workgroup_size) {
    flags &= ~LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE;
  }
  if (iree_any_bit_set(
          flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT) &&
      target->dispatch_workgroup_count != source->dispatch_workgroup_count) {
    flags &= ~LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT;
  }
  if (iree_any_bit_set(
          flags, LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT) &&
      target->dispatch_workitem_count != source->dispatch_workitem_count) {
    flags &= ~LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT;
  }
  if (iree_any_bit_set(
          flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_CLUSTER_SIZE) &&
      target->flat_workgroup_cluster_size !=
          source->flat_workgroup_cluster_size) {
    flags &= ~LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_CLUSTER_SIZE;
  }
  target->flags = flags;
  if (!iree_any_bit_set(flags,
                        LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE)) {
    target->workgroup_size = (loom_target_workgroup_size_t){0};
  }
  if (!iree_any_bit_set(flags,
                        LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT)) {
    target->workgroup_count = (loom_target_dispatch_workgroup_count_t){0};
  }
  if (!iree_any_bit_set(
          flags, LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE)) {
    target->workgroup_cluster_size = (loom_target_workgroup_cluster_size_t){0};
  }
  if (!iree_any_bit_set(
          flags, LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE)) {
    target->flat_workgroup_size = 0;
  }
  if (!iree_any_bit_set(
          flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT)) {
    target->dispatch_workgroup_count = 0;
  }
  if (!iree_any_bit_set(
          flags, LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
    target->dispatch_workitem_count = 0;
  }
  if (!iree_any_bit_set(
          flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_CLUSTER_SIZE)) {
    target->flat_workgroup_cluster_size = 0;
  }
}

static void loom_target_compile_report_merge_target_resources(
    loom_target_compile_report_target_resources_t* target,
    const loom_target_compile_report_target_resources_t* source) {
  target->scalar_register_class = loom_target_compile_report_shared_string(
      target->scalar_register_class, source->scalar_register_class);
  target->scalar_register_count =
      iree_max(target->scalar_register_count, source->scalar_register_count);
  uint64_t scalar_register_overhead_units =
      iree_max(target->scalar_register_overhead_units,
               source->scalar_register_overhead_units);
  target->scalar_pressure_peak_live_units =
      iree_string_view_is_empty(target->scalar_register_class)
          ? 0
          : iree_max(target->scalar_pressure_peak_live_units,
                     source->scalar_pressure_peak_live_units);
  target->scalar_register_overhead_units =
      iree_string_view_is_empty(target->scalar_register_class)
          ? 0
          : (target->scalar_pressure_peak_live_units != 0
                 ? loom_target_compile_report_register_overhead_units(
                       target->scalar_register_count,
                       target->scalar_pressure_peak_live_units)
                 : scalar_register_overhead_units);
  target->vector_register_class = loom_target_compile_report_shared_string(
      target->vector_register_class, source->vector_register_class);
  target->vector_register_count =
      iree_max(target->vector_register_count, source->vector_register_count);
  uint64_t vector_register_overhead_units =
      iree_max(target->vector_register_overhead_units,
               source->vector_register_overhead_units);
  target->vector_pressure_peak_live_units =
      iree_string_view_is_empty(target->vector_register_class)
          ? 0
          : iree_max(target->vector_pressure_peak_live_units,
                     source->vector_pressure_peak_live_units);
  target->vector_register_overhead_units =
      iree_string_view_is_empty(target->vector_register_class)
          ? 0
          : (target->vector_pressure_peak_live_units != 0
                 ? loom_target_compile_report_register_overhead_units(
                       target->vector_register_count,
                       target->vector_pressure_peak_live_units)
                 : vector_register_overhead_units);
  target->subgroup_size = loom_target_compile_report_shared_u32(
      target->subgroup_size, source->subgroup_size);
  target->max_subgroups_per_simd = loom_target_compile_report_shared_u32(
      target->max_subgroups_per_simd, source->max_subgroups_per_simd);
  if (source->occupancy_percent < target->occupancy_percent ||
      (source->occupancy_percent == target->occupancy_percent &&
       source->resident_subgroups_per_simd <
           target->resident_subgroups_per_simd)) {
    target->resident_subgroups_per_simd = source->resident_subgroups_per_simd;
    target->occupancy_percent = source->occupancy_percent;
    target->limiting_resource = source->limiting_resource;
  }
}

static void loom_target_compile_report_merge_entry_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_t* entry_report) {
  const bool first_entry = report->entry_rows.count == 0;
  const bool report_had_emission_breakdown =
      iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION_BREAKDOWN);
  const bool report_had_target_resources = iree_any_bit_set(
      report->detail_flags, LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES);
  const bool report_had_dynamic_instruction_mix = iree_any_bit_set(
      report->detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX);
  const bool report_had_low_planning = iree_any_bit_set(
      report->detail_flags, LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING);
  const bool entry_has_dynamic_instruction_mix = iree_any_bit_set(
      entry_report->detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX);
  const bool entry_has_emission_breakdown =
      iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION_BREAKDOWN);
  const bool entry_has_low_planning =
      iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING);
  const bool entry_has_source_low_data =
      entry_report->source_low_selected_op_count != 0 ||
      entry_report->source_low_emitted_op_count != 0 ||
      entry_report->source_low_memory_summary.packet_count != 0;
  report->detail_flags |=
      entry_report->detail_flags | LOOM_TARGET_COMPILE_REPORT_DETAIL_ENTRIES;
  if (first_entry) {
    report->function_name = entry_report->function_name;
    report->lowered_symbol = entry_report->lowered_symbol;
    report->target_bundle_name = entry_report->target_bundle_name;
    report->target_snapshot_name = entry_report->target_snapshot_name;
    report->target_export_name = entry_report->target_export_name;
    report->target_export_symbol = entry_report->target_export_symbol;
    report->target_config_name = entry_report->target_config_name;
    report->schedule_node_count = entry_report->schedule_node_count;
    report->scheduled_node_count = entry_report->scheduled_node_count;
    report->schedule_dependency_count = entry_report->schedule_dependency_count;
    report->schedule_resource_use_count =
        entry_report->schedule_resource_use_count;
    report->schedule_hazard_gap_count = entry_report->schedule_hazard_gap_count;
    report->schedule_model_summary_count =
        entry_report->schedule_model_summary_count;
    report->register_pressure_summary_count =
        entry_report->register_pressure_summary_count;
    report->register_pressure_peak_live_units =
        entry_report->register_pressure_peak_live_units;
    report->allocation_assignment_count =
        entry_report->allocation_assignment_count;
    report->allocation_spill_count = entry_report->allocation_spill_count;
    report->allocation_spill_plan_count =
        entry_report->allocation_spill_plan_count;
    report->allocation_coalesced_copy_count =
        entry_report->allocation_coalesced_copy_count;
    report->allocation_materialized_copy_count =
        entry_report->allocation_materialized_copy_count;
    report->allocation_materialized_spill_storage_count =
        entry_report->allocation_materialized_spill_storage_count;
    report->allocation_materialized_spill_storage_bytes =
        entry_report->allocation_materialized_spill_storage_bytes;
    report->allocation_materialized_spill_store_count =
        entry_report->allocation_materialized_spill_store_count;
    report->allocation_materialized_spill_store_bytes =
        entry_report->allocation_materialized_spill_store_bytes;
    report->allocation_materialized_reload_count =
        entry_report->allocation_materialized_reload_count;
    report->allocation_materialized_reload_bytes =
        entry_report->allocation_materialized_reload_bytes;
    report->allocation_storage_lease_count =
        entry_report->allocation_storage_lease_count;
    report->allocation_storage_lease_instance_count =
        entry_report->allocation_storage_lease_instance_count;
    report->allocation_storage_release_action_count =
        entry_report->allocation_storage_release_action_count;
    report->emitted_instruction_count = entry_report->emitted_instruction_count;
    report->emitted_code_byte_count = entry_report->emitted_code_byte_count;
    report->emitted_code_storage_byte_count =
        entry_report->emitted_code_storage_byte_count;
    report->emission_breakdown = entry_report->emission_breakdown;
    report->private_memory_bytes = entry_report->private_memory_bytes;
    report->local_memory_bytes = entry_report->local_memory_bytes;
    report->static_instruction_mix = entry_report->static_instruction_mix;
    report->dynamic_instruction_mix = entry_report->dynamic_instruction_mix;
    report->target_resources = entry_report->target_resources;
    report->wait_plan = entry_report->wait_plan;
    report->workload = entry_report->workload;
    report->target_insertion_summary = entry_report->target_insertion_summary;
    if (entry_has_low_planning) {
      report->low_planning = entry_report->low_planning;
    }
    if (report->workload.flags == LOOM_TARGET_COMPILE_REPORT_WORKLOAD_NONE) {
      report->detail_flags &= ~LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD;
    }
    report->math_legalization_rewritten_op_count =
        entry_report->math_legalization_rewritten_op_count;
    report->math_legalization_rejected_op_count =
        entry_report->math_legalization_rejected_op_count;
    report->math_legalization_missing_policy_op_count =
        entry_report->math_legalization_missing_policy_op_count;
    report->math_legalization_missing_recipe_op_count =
        entry_report->math_legalization_missing_recipe_op_count;
    memcpy(report->move_causes, entry_report->move_causes,
           sizeof(report->move_causes));
    if (entry_has_source_low_data) {
      report->source_low_selected_op_count +=
          entry_report->source_low_selected_op_count;
      report->source_low_emitted_op_count +=
          entry_report->source_low_emitted_op_count;
      loom_target_compile_report_accumulate_source_low_memory_summaries(
          &report->source_low_memory_summary,
          &entry_report->source_low_memory_summary);
      loom_target_compile_report_accumulate_bank_service_summaries(
          &report->bank_service_summary, &entry_report->bank_service_summary);
      loom_target_compile_report_accumulate_subgroup_access_summaries(
          &report->subgroup_access_summary,
          &entry_report->subgroup_access_summary);
    }
    return;
  }

  report->function_name = iree_string_view_empty();
  report->lowered_symbol = iree_string_view_empty();
  report->target_export_name = iree_string_view_empty();
  report->target_export_symbol = iree_string_view_empty();
  report->target_bundle_name = loom_target_compile_report_shared_string(
      report->target_bundle_name, entry_report->target_bundle_name);
  report->target_snapshot_name = loom_target_compile_report_shared_string(
      report->target_snapshot_name, entry_report->target_snapshot_name);
  report->target_config_name = loom_target_compile_report_shared_string(
      report->target_config_name, entry_report->target_config_name);
  report->schedule_node_count += entry_report->schedule_node_count;
  report->scheduled_node_count += entry_report->scheduled_node_count;
  report->schedule_dependency_count += entry_report->schedule_dependency_count;
  report->schedule_resource_use_count +=
      entry_report->schedule_resource_use_count;
  report->schedule_hazard_gap_count += entry_report->schedule_hazard_gap_count;
  report->schedule_model_summary_count +=
      entry_report->schedule_model_summary_count;
  report->register_pressure_summary_count +=
      entry_report->register_pressure_summary_count;
  report->register_pressure_peak_live_units =
      iree_max(report->register_pressure_peak_live_units,
               entry_report->register_pressure_peak_live_units);
  report->allocation_assignment_count +=
      entry_report->allocation_assignment_count;
  report->allocation_spill_count += entry_report->allocation_spill_count;
  report->allocation_spill_plan_count +=
      entry_report->allocation_spill_plan_count;
  report->allocation_coalesced_copy_count +=
      entry_report->allocation_coalesced_copy_count;
  report->allocation_materialized_copy_count +=
      entry_report->allocation_materialized_copy_count;
  report->allocation_materialized_spill_storage_count +=
      entry_report->allocation_materialized_spill_storage_count;
  report->allocation_materialized_spill_storage_bytes +=
      entry_report->allocation_materialized_spill_storage_bytes;
  report->allocation_materialized_spill_store_count +=
      entry_report->allocation_materialized_spill_store_count;
  report->allocation_materialized_spill_store_bytes +=
      entry_report->allocation_materialized_spill_store_bytes;
  report->allocation_materialized_reload_count +=
      entry_report->allocation_materialized_reload_count;
  report->allocation_materialized_reload_bytes +=
      entry_report->allocation_materialized_reload_bytes;
  report->allocation_storage_lease_count +=
      entry_report->allocation_storage_lease_count;
  report->allocation_storage_lease_instance_count +=
      entry_report->allocation_storage_lease_instance_count;
  report->allocation_storage_release_action_count +=
      entry_report->allocation_storage_release_action_count;
  report->emitted_instruction_count += entry_report->emitted_instruction_count;
  report->emitted_code_byte_count += entry_report->emitted_code_byte_count;
  report->emitted_code_storage_byte_count +=
      entry_report->emitted_code_storage_byte_count;
  if (report_had_emission_breakdown && entry_has_emission_breakdown) {
    report->emission_breakdown.body_instruction_count +=
        entry_report->emission_breakdown.body_instruction_count;
    report->emission_breakdown.entry_instruction_count +=
        entry_report->emission_breakdown.entry_instruction_count;
    report->emission_breakdown.coissued_instruction_count +=
        entry_report->emission_breakdown.coissued_instruction_count;
    report->emission_breakdown.coissued_component_count +=
        entry_report->emission_breakdown.coissued_component_count;
  } else {
    report->detail_flags &=
        ~LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION_BREAKDOWN;
    report->emission_breakdown =
        (loom_target_compile_report_emission_breakdown_t){0};
  }
  report->private_memory_bytes = iree_max(report->private_memory_bytes,
                                          entry_report->private_memory_bytes);
  report->local_memory_bytes =
      iree_max(report->local_memory_bytes, entry_report->local_memory_bytes);
  loom_target_compile_report_accumulate_wait_plan(&report->wait_plan,
                                                  &entry_report->wait_plan);
  loom_target_compile_report_accumulate_target_insertion_summary(
      &report->target_insertion_summary,
      &entry_report->target_insertion_summary);
  loom_target_compile_report_accumulate_instruction_mix(
      &report->static_instruction_mix, &entry_report->static_instruction_mix);
  if (report_had_dynamic_instruction_mix && entry_has_dynamic_instruction_mix) {
    loom_target_compile_report_accumulate_instruction_mix(
        &report->dynamic_instruction_mix,
        &entry_report->dynamic_instruction_mix);
  } else {
    report->detail_flags &=
        ~LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX;
    report->dynamic_instruction_mix =
        (loom_target_compile_report_static_instruction_mix_t){0};
  }
  if (report_had_low_planning && entry_has_low_planning) {
    loom_low_planning_statistics_accumulate(&report->low_planning,
                                            &entry_report->low_planning);
  } else {
    report->detail_flags &= ~LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING;
    report->low_planning = (loom_low_planning_statistics_t){0};
  }
  report->math_legalization_rewritten_op_count +=
      entry_report->math_legalization_rewritten_op_count;
  report->math_legalization_rejected_op_count +=
      entry_report->math_legalization_rejected_op_count;
  report->math_legalization_missing_policy_op_count +=
      entry_report->math_legalization_missing_policy_op_count;
  report->math_legalization_missing_recipe_op_count +=
      entry_report->math_legalization_missing_recipe_op_count;
  report->source_low_selected_op_count +=
      entry_report->source_low_selected_op_count;
  report->source_low_emitted_op_count +=
      entry_report->source_low_emitted_op_count;
  loom_target_compile_report_accumulate_source_low_memory_summaries(
      &report->source_low_memory_summary,
      &entry_report->source_low_memory_summary);
  loom_target_compile_report_accumulate_bank_service_summaries(
      &report->bank_service_summary, &entry_report->bank_service_summary);
  loom_target_compile_report_accumulate_subgroup_access_summaries(
      &report->subgroup_access_summary, &entry_report->subgroup_access_summary);
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    if (report_had_target_resources) {
      loom_target_compile_report_merge_target_resources(
          &report->target_resources, &entry_report->target_resources);
    } else {
      report->target_resources = entry_report->target_resources;
    }
  }
  loom_target_compile_report_merge_workload(&report->workload,
                                            &entry_report->workload);
  if (report->workload.flags == LOOM_TARGET_COMPILE_REPORT_WORKLOAD_NONE) {
    report->detail_flags &= ~LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD;
  }
  loom_target_compile_report_accumulate_move_causes(report->move_causes,
                                                    entry_report->move_causes);
}

static loom_target_compile_report_entry_t
loom_target_compile_report_entry_from_report(
    const loom_target_compile_report_t* entry_report) {
  return (loom_target_compile_report_entry_t){
      .function_name = entry_report->function_name,
      .source_function_name = entry_report->lowered_symbol,
      .target_bundle_name = entry_report->target_bundle_name,
      .target_snapshot_name = entry_report->target_snapshot_name,
      .target_export_name = entry_report->target_export_name,
      .target_export_symbol = entry_report->target_export_symbol,
      .target_config_name = entry_report->target_config_name,
      .detail_flags = entry_report->detail_flags,
      .schedule_node_count = entry_report->schedule_node_count,
      .scheduled_node_count = entry_report->scheduled_node_count,
      .schedule_dependency_count = entry_report->schedule_dependency_count,
      .schedule_resource_use_count = entry_report->schedule_resource_use_count,
      .schedule_hazard_gap_count = entry_report->schedule_hazard_gap_count,
      .schedule_model_summary_count =
          entry_report->schedule_model_summary_count,
      .register_pressure_summary_count =
          entry_report->register_pressure_summary_count,
      .register_pressure_peak_live_units =
          entry_report->register_pressure_peak_live_units,
      .allocation_assignment_count = entry_report->allocation_assignment_count,
      .allocation_spill_count = entry_report->allocation_spill_count,
      .allocation_spill_plan_count = entry_report->allocation_spill_plan_count,
      .allocation_coalesced_copy_count =
          entry_report->allocation_coalesced_copy_count,
      .allocation_materialized_copy_count =
          entry_report->allocation_materialized_copy_count,
      .allocation_materialized_spill_storage_count =
          entry_report->allocation_materialized_spill_storage_count,
      .allocation_materialized_spill_storage_bytes =
          entry_report->allocation_materialized_spill_storage_bytes,
      .allocation_materialized_spill_store_count =
          entry_report->allocation_materialized_spill_store_count,
      .allocation_materialized_spill_store_bytes =
          entry_report->allocation_materialized_spill_store_bytes,
      .allocation_materialized_reload_count =
          entry_report->allocation_materialized_reload_count,
      .allocation_materialized_reload_bytes =
          entry_report->allocation_materialized_reload_bytes,
      .allocation_storage_lease_count =
          entry_report->allocation_storage_lease_count,
      .allocation_storage_lease_instance_count =
          entry_report->allocation_storage_lease_instance_count,
      .allocation_storage_release_action_count =
          entry_report->allocation_storage_release_action_count,
      .emitted_instruction_count = entry_report->emitted_instruction_count,
      .emitted_code_byte_count = entry_report->emitted_code_byte_count,
      .emitted_code_storage_byte_count =
          entry_report->emitted_code_storage_byte_count,
      .emission_breakdown = entry_report->emission_breakdown,
      .private_memory_bytes = entry_report->private_memory_bytes,
      .local_memory_bytes = entry_report->local_memory_bytes,
      .bank_service_summary = entry_report->bank_service_summary,
      .subgroup_access_summary = entry_report->subgroup_access_summary,
      .static_instruction_mix = entry_report->static_instruction_mix,
      .dynamic_instruction_mix = entry_report->dynamic_instruction_mix,
      .target_resources = entry_report->target_resources,
      .wait_plan = entry_report->wait_plan,
      .workload = entry_report->workload,
      .low_planning = entry_report->low_planning,
      .target_insertion_summary = entry_report->target_insertion_summary,
      .pressure_row_count = entry_report->pressure_rows.count,
      .pressure_origin_row_count = entry_report->pressure_origin_rows.count,
      .schedule_band_row_count = entry_report->schedule_band_rows.count,
      .schedule_band_summary_row_count =
          entry_report->schedule_band_summary_rows.count,
      .spill_row_count = entry_report->spill_rows.count,
      .allocation_high_water_row_count =
          entry_report->allocation_high_water_rows.count,
      .wait_counter_row_count = entry_report->wait_counter_rows.count,
      .wait_reason_summary_row_count =
          entry_report->wait_reason_summary_rows.count,
      .wait_action_row_count = entry_report->wait_action_rows.count,
      .target_capability_row_count = entry_report->target_capability_rows.count,
      .target_insertion_row_count = entry_report->target_insertion_rows.count,
  };
}

static bool loom_target_compile_report_source_low_row_has_summary_key(
    const loom_target_compile_report_source_low_row_t* row) {
  return row->emitted_low_op_count != 0 &&
         (!iree_string_view_is_empty(row->plan_key) ||
          !iree_string_view_is_empty(row->descriptor_key) ||
          !iree_string_view_is_empty(row->descriptor_semantic_tag));
}

static loom_target_compile_report_source_low_selection_summary_t
loom_target_compile_report_source_low_selection_summary_from_row(
    const loom_target_compile_report_source_low_row_t* row) {
  loom_target_compile_report_source_low_selection_summary_t summary = {
      .function_name = row->function_name,
      .source_op_name = row->source_op_name,
      .source_op_kind = row->source_op_kind,
      .selection_kind = row->selection_kind,
      .plan_key = row->plan_key,
      .descriptor_key = row->descriptor_key,
      .descriptor_semantic_tag = row->descriptor_semantic_tag,
      .selected_op_count = 1,
      .emitted_low_op_count = row->emitted_low_op_count,
  };
  if (row->execution_count_plus_one ==
      LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_EXECUTION_COUNT_PLUS_ONE_UNKNOWN) {
    summary.unknown_dynamic_op_count = 1;
    return summary;
  }
  const uint64_t execution_count = row->execution_count_plus_one - 1;
  uint64_t dynamic_emitted_low_op_count = 0;
  if (!iree_checked_mul_u64(row->emitted_low_op_count, execution_count,
                            &dynamic_emitted_low_op_count)) {
    summary.unknown_dynamic_op_count = 1;
    return summary;
  }
  summary.exact_dynamic_op_count = 1;
  summary.dynamic_selected_op_count = execution_count;
  summary.dynamic_emitted_low_op_count = dynamic_emitted_low_op_count;
  return summary;
}

static bool loom_target_compile_report_source_low_selection_summaries_match(
    const loom_target_compile_report_source_low_selection_summary_t* lhs,
    const loom_target_compile_report_source_low_selection_summary_t* rhs) {
  return iree_string_view_equal(lhs->function_name, rhs->function_name) &&
         iree_string_view_equal(lhs->source_op_name, rhs->source_op_name) &&
         lhs->source_op_kind == rhs->source_op_kind &&
         lhs->selection_kind == rhs->selection_kind &&
         iree_string_view_equal(lhs->plan_key, rhs->plan_key) &&
         iree_string_view_equal(lhs->descriptor_key, rhs->descriptor_key) &&
         iree_string_view_equal(lhs->descriptor_semantic_tag,
                                rhs->descriptor_semantic_tag);
}

static loom_target_compile_report_source_low_selection_summary_t*
loom_target_compile_report_find_source_low_selection_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_selection_summary_t* row) {
  for (loom_target_compile_report_vec_t* vec =
           report->source_low_selection_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_source_low_selection_summary_t* summaries =
        (loom_target_compile_report_source_low_selection_summary_t*)
            loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      loom_target_compile_report_source_low_selection_summary_t* summary =
          &summaries[i];
      if (loom_target_compile_report_source_low_selection_summaries_match(
              summary, row)) {
        return summary;
      }
    }
  }
  return NULL;
}

static iree_status_t
loom_target_compile_report_record_source_low_selection_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_selection_summary_t* row) {
  loom_target_compile_report_source_low_selection_summary_t* summary =
      loom_target_compile_report_find_source_low_selection_summary(report, row);
  if (summary != NULL) {
    summary->selected_op_count += row->selected_op_count;
    summary->emitted_low_op_count += row->emitted_low_op_count;
    summary->exact_dynamic_op_count += row->exact_dynamic_op_count;
    summary->unknown_dynamic_op_count += row->unknown_dynamic_op_count;
    summary->dynamic_selected_op_count += row->dynamic_selected_op_count;
    summary->dynamic_emitted_low_op_count += row->dynamic_emitted_low_op_count;
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->source_low_selection_summaries, sizeof(*row), report->allocator,
      row);
}

iree_status_t loom_target_compile_report_record_entry_report(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_t* entry_report) {
  loom_target_compile_report_entry_t entry =
      loom_target_compile_report_entry_from_report(entry_report);
  memcpy(entry.move_causes, entry_report->move_causes,
         sizeof(entry.move_causes));
  loom_target_compile_report_merge_entry_summary(report, entry_report);
  IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append(
      &report->entry_rows, sizeof(entry), report->allocator, &entry));
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->pressure_rows, &entry_report->pressure_rows,
        sizeof(loom_target_compile_report_pressure_row_t), report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->pressure_origin_rows, &entry_report->pressure_origin_rows,
        sizeof(loom_target_compile_report_pressure_origin_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->schedule_band_rows, &entry_report->schedule_band_rows,
        sizeof(loom_target_compile_report_schedule_band_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->schedule_band_summary_rows,
        &entry_report->schedule_band_summary_rows,
        sizeof(loom_target_compile_report_schedule_band_summary_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->spill_rows, &entry_report->spill_rows,
        sizeof(loom_target_compile_report_spill_row_t), report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->allocation_failure_rows,
        &entry_report->allocation_failure_rows,
        sizeof(loom_target_compile_report_allocation_failure_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->allocation_high_water_rows,
        &entry_report->allocation_high_water_rows,
        sizeof(loom_target_compile_report_allocation_high_water_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->wait_counter_rows, &entry_report->wait_counter_rows,
        sizeof(loom_target_compile_report_wait_counter_row_t),
        report->allocator));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->wait_reason_summary_rows,
        &entry_report->wait_reason_summary_rows,
        sizeof(loom_target_compile_report_wait_reason_summary_row_t),
        report->allocator));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->wait_action_rows, &entry_report->wait_action_rows,
        sizeof(loom_target_compile_report_wait_action_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->target_insertion_rows, &entry_report->target_insertion_rows,
        sizeof(loom_target_compile_report_target_insertion_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->config_binding_rows, &entry_report->config_binding_rows,
        sizeof(loom_target_compile_report_config_binding_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->source_low_rows, &entry_report->source_low_rows,
        sizeof(loom_target_compile_report_source_low_row_t),
        report->allocator));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->source_low_target_rows, &entry_report->source_low_target_rows,
        sizeof(loom_target_compile_report_source_low_target_row_t),
        report->allocator));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->source_low_transform_rows,
        &entry_report->source_low_transform_rows,
        sizeof(loom_target_compile_report_source_low_transform_row_t),
        report->allocator));
    for (const loom_target_compile_report_vec_t* vec =
             entry_report->source_low_selection_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_selection_summary_t* rows =
          (const loom_target_compile_report_source_low_selection_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_record_source_low_selection_summary_row(
                report, &rows[i]));
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_merge_source_low_memory_details(
            report, entry_report));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->math_legalization_rows, &entry_report->math_legalization_rows,
        sizeof(loom_target_compile_report_math_row_t), report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append_all(
        &report->target_capability_rows, &entry_report->target_capability_rows,
        sizeof(loom_target_compile_report_target_capability_row_t),
        report->allocator));
  }
  return iree_ok_status();
}

iree_status_t loom_target_compile_report_record_pressure_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_pressure_row_t* row) {
  const loom_target_compile_report_pressure_summary_t summary = {
      .register_class = row->register_class,
      .peak_live_units = row->peak_live_units,
  };
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_pressure_summary(report, &summary));
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->pressure_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_pressure_origin_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_pressure_origin_row_t* row) {
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->pressure_origin_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_schedule_band_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_schedule_band_row_t* row) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->schedule_band_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_schedule_band_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_schedule_band_summary_row_t* row) {
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->schedule_band_summary_rows, sizeof(*row), report->allocator,
      row);
}

iree_status_t loom_target_compile_report_record_spill_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_spill_row_t* row) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->spill_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_allocation_failure_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_allocation_failure_row_t* row) {
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->allocation_failure_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_allocation_high_water_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_allocation_high_water_row_t* row) {
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->allocation_high_water_rows, sizeof(*row), report->allocator,
      row);
}

iree_status_t loom_target_compile_report_record_wait_counter_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_wait_counter_row_t* row) {
  if (row->summary.action_count == 0) {
    return iree_ok_status();
  }
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN;
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->wait_counter_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_wait_reason_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_wait_reason_summary_row_t* row) {
  if (row->summary.action_count == 0) {
    return iree_ok_status();
  }
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN;
  return loom_target_compile_report_row_list_append(
      &report->wait_reason_summary_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_wait_action_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_wait_action_row_t* row) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN;
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->wait_action_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_target_capability_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_target_capability_row_t* row) {
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->target_capability_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_source_low_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_row_t* row) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append(
      &report->source_low_rows, sizeof(*row), report->allocator, row));
  if (!loom_target_compile_report_source_low_row_has_summary_key(row)) {
    return iree_ok_status();
  }
  const loom_target_compile_report_source_low_selection_summary_t summary =
      loom_target_compile_report_source_low_selection_summary_from_row(row);
  return loom_target_compile_report_record_source_low_selection_summary_row(
      report, &summary);
}

iree_status_t loom_target_compile_report_record_config_binding_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_config_binding_row_t* row) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS;
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS)) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->config_binding_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_source_low_target_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_target_row_t* row) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->source_low_target_rows, sizeof(*row), report->allocator, row);
}

iree_status_t loom_target_compile_report_record_source_low_transform_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_transform_row_t* row) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->source_low_transform_rows, sizeof(*row), report->allocator, row);
}

static void loom_target_compile_report_count_math_action(
    loom_target_compile_report_t* report,
    loom_target_compile_report_math_action_t action) {
  switch (action) {
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REWRITTEN:
      ++report->math_legalization_rewritten_op_count;
      break;
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_REJECTED:
      ++report->math_legalization_rejected_op_count;
      break;
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_MISSING_POLICY:
      ++report->math_legalization_missing_policy_op_count;
      break;
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_MISSING_RECIPE:
      ++report->math_legalization_missing_recipe_op_count;
      break;
    case LOOM_TARGET_COMPILE_REPORT_MATH_ACTION_NONE:
    default:
      break;
  }
}

iree_status_t loom_target_compile_report_record_math_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_math_row_t* row) {
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS;
  loom_target_compile_report_count_math_action(report, row->action);
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS)) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->math_legalization_rows, sizeof(*row), report->allocator, row);
}

static void loom_target_compile_report_count_legalization_action(
    loom_target_compile_report_t* report,
    loom_target_compile_report_legalization_action_t action,
    loom_target_compile_report_legalizer_strategy_t legalizer_strategy) {
  switch (action) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_LEGAL:
      ++report->target_legalization_legal_op_count;
      break;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN:
      ++report->target_legalization_rewritten_op_count;
      switch (legalizer_strategy) {
        case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_TARGET:
          ++report->target_legalization_target_rewritten_op_count;
          break;
        case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE:
          ++report->target_legalization_reference_rewritten_op_count;
          break;
        case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_NONE:
        default:
          break;
      }
      break;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_DEFERRED:
      ++report->target_legalization_deferred_op_count;
      break;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_INVALID_IR:
      ++report->target_legalization_invalid_ir_op_count;
      break;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_UNSUPPORTED_FINAL:
      ++report->target_legalization_unsupported_op_count;
      break;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_UNHANDLED:
      ++report->target_legalization_unhandled_op_count;
      break;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_NONE:
    default:
      break;
  }
}

void loom_target_compile_report_record_legalization_summary(
    loom_target_compile_report_t* report,
    loom_target_compile_report_legalization_action_t action,
    loom_target_compile_report_legalizer_strategy_t legalizer_strategy) {
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS;
  loom_target_compile_report_count_legalization_action(report, action,
                                                       legalizer_strategy);
}

iree_status_t loom_target_compile_report_record_legalization_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_legalization_row_t* row) {
  loom_target_compile_report_record_legalization_summary(
      report, row->action, row->legalizer_strategy);
  return loom_target_compile_report_row_list_append(
      &report->target_legalization_rows, sizeof(*row), report->allocator, row);
}
