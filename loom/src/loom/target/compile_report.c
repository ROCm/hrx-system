// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report.h"

#include <string.h>

enum {
  // Default allocation block size for compile report detail rows.
  LOOM_TARGET_COMPILE_REPORT_VEC_DEFAULT_BYTE_LENGTH = 4096,
};

static bool loom_target_compile_report_checked_add_u64(uint64_t lhs,
                                                       uint64_t rhs,
                                                       uint64_t* out_result) {
  if (UINT64_MAX - lhs < rhs) {
    return false;
  }
  *out_result = lhs + rhs;
  return true;
}

static bool loom_target_compile_report_checked_mul_u64(uint64_t lhs,
                                                       uint64_t rhs,
                                                       uint64_t* out_result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) {
    return false;
  }
  *out_result = lhs * rhs;
  return true;
}

void loom_target_compile_report_initialize(
    loom_target_compile_report_t* out_report, iree_allocator_t allocator) {
  *out_report = (loom_target_compile_report_t){
      .allocator = allocator,
      .status_code = IREE_STATUS_OK,
  };
}

static void loom_target_compile_report_row_list_deinitialize(
    iree_allocator_t allocator, loom_target_compile_report_row_list_t* list) {
  loom_target_compile_report_vec_t* vec = list->head;
  while (vec != NULL) {
    loom_target_compile_report_vec_t* next = vec->next;
    iree_allocator_free(allocator, vec);
    vec = next;
  }
  *list = (loom_target_compile_report_row_list_t){0};
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
  loom_target_compile_report_row_list_deinitialize(allocator,
                                                   &report->wait_action_rows);
  loom_target_compile_report_row_list_deinitialize(allocator,
                                                   &report->source_low_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_target_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_selection_summaries);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_memory_rows);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_memory_root_summaries);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_memory_argument_summaries);
  loom_target_compile_report_row_list_deinitialize(
      allocator, &report->source_low_memory_strategy_summaries);
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
         report->wait_action_rows.count != 0 || report->entry_rows.count != 0 ||
         report->source_low_rows.count != 0 ||
         report->source_low_target_rows.count != 0 ||
         report->source_low_selection_summaries.count != 0 ||
         report->source_low_memory_rows.count != 0 ||
         report->source_low_memory_root_summaries.count != 0 ||
         report->source_low_memory_argument_summaries.count != 0 ||
         report->source_low_memory_strategy_summaries.count != 0 ||
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

static iree_status_t loom_target_compile_report_row_list_append(
    loom_target_compile_report_row_list_t* list, iree_host_size_t row_size,
    iree_allocator_t allocator, const void* row) {
  if (row_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "compile report row size must be non-zero");
  }
  if (iree_allocator_is_null(allocator)) {
    return iree_ok_status();
  }
  if (list->tail == NULL || list->tail->count == list->tail->capacity) {
    iree_host_size_t capacity =
        (LOOM_TARGET_COMPILE_REPORT_VEC_DEFAULT_BYTE_LENGTH -
         sizeof(loom_target_compile_report_vec_t)) /
        row_size;
    capacity = iree_max((iree_host_size_t)1, capacity);
    iree_host_size_t row_bytes = 0;
    if (!iree_host_size_checked_mul(capacity, row_size, &row_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "compile report row block is too large");
    }
    iree_host_size_t block_bytes = 0;
    if (!iree_host_size_checked_add(sizeof(loom_target_compile_report_vec_t),
                                    row_bytes, &block_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "compile report row block is too large");
    }
    loom_target_compile_report_vec_t* vec = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, block_bytes, (void**)&vec));
    *vec = (loom_target_compile_report_vec_t){
        .capacity = capacity,
    };
    if (list->tail != NULL) {
      list->tail->next = vec;
    } else {
      list->head = vec;
    }
    list->tail = vec;
  }
  uint8_t* rows = (uint8_t*)loom_target_compile_report_vec_rows(list->tail);
  memcpy(rows + list->tail->count * row_size, row, row_size);
  ++list->tail->count;
  ++list->count;
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_row_list_clone(
    const loom_target_compile_report_row_list_t* source,
    iree_host_size_t row_size, iree_allocator_t allocator,
    loom_target_compile_report_row_list_t* target) {
  *target = (loom_target_compile_report_row_list_t){0};
  for (const loom_target_compile_report_vec_t* vec = source->head; vec != NULL;
       vec = vec->next) {
    const uint8_t* rows =
        (const uint8_t*)loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append(
          target, row_size, allocator, rows + i * row_size));
    }
  }
  return iree_ok_status();
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
  target.wait_action_rows = (loom_target_compile_report_row_list_t){0};
  target.source_low_rows = (loom_target_compile_report_row_list_t){0};
  target.source_low_target_rows = (loom_target_compile_report_row_list_t){0};
  target.source_low_selection_summaries =
      (loom_target_compile_report_row_list_t){0};
  target.source_low_memory_rows = (loom_target_compile_report_row_list_t){0};
  target.source_low_memory_root_summaries =
      (loom_target_compile_report_row_list_t){0};
  target.source_low_memory_argument_summaries =
      (loom_target_compile_report_row_list_t){0};
  target.source_low_memory_strategy_summaries =
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
      source->wait_action_rows.count == 0 &&
      source->source_low_rows.count == 0 &&
      source->source_low_target_rows.count == 0 &&
      source->source_low_selection_summaries.count == 0 &&
      source->source_low_memory_rows.count == 0 &&
      source->source_low_memory_root_summaries.count == 0 &&
      source->source_low_memory_argument_summaries.count == 0 &&
      source->source_low_memory_strategy_summaries.count == 0 &&
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
        &source->wait_action_rows,
        sizeof(loom_target_compile_report_wait_action_row_t), allocator,
        &target.wait_action_rows);
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
        &source->source_low_memory_strategy_summaries,
        sizeof(loom_target_compile_report_source_low_memory_strategy_summary_t),
        allocator, &target.source_low_memory_strategy_summaries);
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

void loom_target_compile_report_record_workload(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_workload_t* workload) {
  if (workload->flags == LOOM_TARGET_COMPILE_REPORT_WORKLOAD_NONE) {
    return;
  }
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD;
  report->workload = *workload;
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

static bool loom_target_compile_report_memory_interval_has_envelope(
    const loom_target_compile_report_memory_interval_t* interval) {
  const loom_target_compile_report_memory_interval_flags_t range_flags =
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE;
  return iree_all_bits_set(interval->flags, range_flags) &&
         interval->end_max_bytes >= interval->begin_min_bytes;
}

static uint64_t loom_target_compile_report_memory_interval_span(
    int64_t begin_bytes, int64_t end_bytes) {
  return end_bytes >= begin_bytes ? (uint64_t)end_bytes - (uint64_t)begin_bytes
                                  : 0;
}

typedef struct loom_target_compile_report_static_memory_interval_t {
  int64_t begin_bytes;
  int64_t end_bytes;
} loom_target_compile_report_static_memory_interval_t;

typedef enum loom_target_compile_report_memory_interval_unique_kind_e {
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_NONE = 0,
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_STATIC = 1,
  LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_SYMBOLIC = 2,
} loom_target_compile_report_memory_interval_unique_kind_t;

typedef struct loom_target_compile_report_memory_interval_unique_delta_t {
  loom_target_compile_report_memory_interval_unique_kind_t kind;
  uint64_t unique_byte_delta;
} loom_target_compile_report_memory_interval_unique_delta_t;

static bool loom_target_compile_report_memory_interval_is_exact_static(
    const loom_target_compile_report_memory_interval_t* interval,
    loom_target_compile_report_static_memory_interval_t* out_static_interval) {
  if (!loom_target_compile_report_memory_interval_has_envelope(interval) ||
      interval->begin_min_bytes != interval->begin_max_bytes ||
      interval->end_min_bytes != interval->end_max_bytes) {
    return false;
  }
  *out_static_interval = (loom_target_compile_report_static_memory_interval_t){
      .begin_bytes = interval->begin_min_bytes,
      .end_bytes = interval->end_max_bytes,
  };
  return true;
}

static bool loom_target_compile_report_memory_interval_has_exact_symbolic(
    const loom_target_compile_report_memory_interval_t* interval) {
  const loom_target_compile_report_memory_interval_flags_t required_flags =
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_RANGE |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_RANGE |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_BEGIN_EXPR |
      LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_END_EXPR;
  return iree_all_bits_set(interval->flags, required_flags) &&
         interval->exact_length_bytes != 0;
}

static bool loom_target_compile_report_source_low_memory_row_has_root(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  return !iree_string_view_is_empty(row->source_root_name) ||
         row->source_root_argument_index != UINT16_MAX;
}

static bool loom_target_compile_report_source_low_memory_rows_match_interval(
    const loom_target_compile_report_source_low_memory_row_t* lhs,
    const loom_target_compile_report_source_low_memory_row_t* rhs,
    bool match_operation_kind) {
  if (!loom_target_compile_report_source_low_memory_row_has_root(lhs) ||
      !loom_target_compile_report_source_low_memory_row_has_root(rhs)) {
    return false;
  }
  if (!iree_string_view_equal(lhs->function_name, rhs->function_name) ||
      !iree_string_view_equal(lhs->source_root_name, rhs->source_root_name) ||
      lhs->source_root_argument_index != rhs->source_root_argument_index ||
      !iree_string_view_equal(lhs->memory_space, rhs->memory_space)) {
    return false;
  }
  return !match_operation_kind ||
         iree_string_view_equal(lhs->operation_kind, rhs->operation_kind);
}

static bool loom_target_compile_report_static_memory_intervals_overlap(
    loom_target_compile_report_static_memory_interval_t lhs,
    loom_target_compile_report_static_memory_interval_t rhs,
    loom_target_compile_report_static_memory_interval_t* out_overlap) {
  const int64_t begin_bytes = iree_max(lhs.begin_bytes, rhs.begin_bytes);
  const int64_t end_bytes = iree_min(lhs.end_bytes, rhs.end_bytes);
  if (begin_bytes >= end_bytes) {
    return false;
  }
  *out_overlap = (loom_target_compile_report_static_memory_interval_t){
      .begin_bytes = begin_bytes,
      .end_bytes = end_bytes,
  };
  return true;
}

static bool loom_target_compile_report_memory_interval_envelopes_are_disjoint(
    const loom_target_compile_report_memory_interval_t* lhs,
    const loom_target_compile_report_memory_interval_t* rhs) {
  if (!loom_target_compile_report_memory_interval_has_envelope(lhs) ||
      !loom_target_compile_report_memory_interval_has_envelope(rhs)) {
    return false;
  }
  return lhs->end_max_bytes <= rhs->begin_min_bytes ||
         rhs->end_max_bytes <= lhs->begin_min_bytes;
}

static void loom_target_compile_report_insert_sorted_static_memory_interval(
    loom_target_compile_report_static_memory_interval_t* intervals,
    iree_host_size_t* inout_count,
    loom_target_compile_report_static_memory_interval_t interval) {
  iree_host_size_t index = *inout_count;
  while (index > 0 &&
         (intervals[index - 1].begin_bytes > interval.begin_bytes ||
          (intervals[index - 1].begin_bytes == interval.begin_bytes &&
           intervals[index - 1].end_bytes > interval.end_bytes))) {
    intervals[index] = intervals[index - 1];
    --index;
  }
  intervals[index] = interval;
  ++*inout_count;
}

static uint64_t loom_target_compile_report_static_memory_interval_union_bytes(
    const loom_target_compile_report_static_memory_interval_t* intervals,
    iree_host_size_t interval_count) {
  if (interval_count == 0) {
    return 0;
  }
  int64_t begin_bytes = intervals[0].begin_bytes;
  int64_t end_bytes = intervals[0].end_bytes;
  uint64_t byte_count = 0;
  for (iree_host_size_t i = 1; i < interval_count; ++i) {
    if (intervals[i].begin_bytes <= end_bytes) {
      end_bytes = iree_max(end_bytes, intervals[i].end_bytes);
      continue;
    }
    byte_count +=
        loom_target_compile_report_memory_interval_span(begin_bytes, end_bytes);
    begin_bytes = intervals[i].begin_bytes;
    end_bytes = intervals[i].end_bytes;
  }
  return byte_count + loom_target_compile_report_memory_interval_span(
                          begin_bytes, end_bytes);
}

static iree_status_t
loom_target_compile_report_calculate_source_low_unique_interval_delta(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row,
    bool match_operation_kind,
    loom_target_compile_report_memory_interval_unique_delta_t* out_delta) {
  *out_delta = (loom_target_compile_report_memory_interval_unique_delta_t){0};
  if (iree_allocator_is_null(report->allocator) ||
      !loom_target_compile_report_source_low_memory_row_has_root(row)) {
    return iree_ok_status();
  }
  loom_target_compile_report_static_memory_interval_t row_interval = {0};
  const bool row_is_exact_static =
      loom_target_compile_report_memory_interval_is_exact_static(
          &row->source_interval, &row_interval);
  const bool row_is_exact_symbolic =
      !row_is_exact_static &&
      loom_target_compile_report_memory_interval_has_exact_symbolic(
          &row->source_interval);
  if (!row_is_exact_static && !row_is_exact_symbolic) {
    return iree_ok_status();
  }
  out_delta->kind =
      row_is_exact_static
          ? LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_STATIC
          : LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_SYMBOLIC;
  const uint64_t row_byte_count =
      row_is_exact_static
          ? loom_target_compile_report_memory_interval_span(
                row_interval.begin_bytes, row_interval.end_bytes)
          : row->source_interval.exact_length_bytes;
  if (report->source_low_memory_rows.count == 0) {
    out_delta->unique_byte_delta = row_byte_count;
    return iree_ok_status();
  }

  loom_target_compile_report_static_memory_interval_t* overlaps = NULL;
  if (row_is_exact_static) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        report->allocator, report->source_low_memory_rows.count,
        sizeof(*overlaps), (void**)&overlaps));
  }

  iree_host_size_t overlap_count = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_row_t* rows =
        (const loom_target_compile_report_source_low_memory_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      const loom_target_compile_report_source_low_memory_row_t* existing =
          &rows[i];
      if (!loom_target_compile_report_source_low_memory_rows_match_interval(
              existing, row, match_operation_kind)) {
        continue;
      }
      if (row_is_exact_static) {
        loom_target_compile_report_static_memory_interval_t existing_interval =
            {0};
        loom_target_compile_report_static_memory_interval_t overlap = {0};
        if (loom_target_compile_report_memory_interval_is_exact_static(
                &existing->source_interval, &existing_interval)) {
          if (loom_target_compile_report_static_memory_intervals_overlap(
                  row_interval, existing_interval, &overlap)) {
            loom_target_compile_report_insert_sorted_static_memory_interval(
                overlaps, &overlap_count, overlap);
          }
          continue;
        }
        if (loom_target_compile_report_memory_interval_has_exact_symbolic(
                &existing->source_interval) &&
            !loom_target_compile_report_memory_interval_envelopes_are_disjoint(
                &row->source_interval, &existing->source_interval)) {
          iree_allocator_free(report->allocator, overlaps);
          *out_delta =
              (loom_target_compile_report_memory_interval_unique_delta_t){0};
          return iree_ok_status();
        }
        continue;
      }

      loom_target_compile_report_static_memory_interval_t existing_interval = {
          0};
      if (loom_target_compile_report_memory_interval_is_exact_static(
              &existing->source_interval, &existing_interval)) {
        if (!loom_target_compile_report_memory_interval_envelopes_are_disjoint(
                &row->source_interval, &existing->source_interval)) {
          *out_delta =
              (loom_target_compile_report_memory_interval_unique_delta_t){0};
          return iree_ok_status();
        }
        continue;
      }
      if (!loom_target_compile_report_memory_interval_has_exact_symbolic(
              &existing->source_interval)) {
        continue;
      }
      if (row->source_interval.begin_expr_id ==
              existing->source_interval.begin_expr_id &&
          row->source_interval.end_expr_id ==
              existing->source_interval.end_expr_id) {
        out_delta->unique_byte_delta = 0;
        return iree_ok_status();
      }
      if (existing->source_interval.end_expr_id ==
              row->source_interval.begin_expr_id ||
          row->source_interval.end_expr_id ==
              existing->source_interval.begin_expr_id ||
          loom_target_compile_report_memory_interval_envelopes_are_disjoint(
              &row->source_interval, &existing->source_interval)) {
        continue;
      }
      *out_delta =
          (loom_target_compile_report_memory_interval_unique_delta_t){0};
      return iree_ok_status();
    }
  }

  if (row_is_exact_static) {
    const uint64_t covered_byte_count =
        loom_target_compile_report_static_memory_interval_union_bytes(
            overlaps, overlap_count);
    iree_allocator_free(report->allocator, overlaps);
    out_delta->unique_byte_delta = row_byte_count > covered_byte_count
                                       ? row_byte_count - covered_byte_count
                                       : 0;
  } else {
    out_delta->unique_byte_delta = row_byte_count;
  }
  return iree_ok_status();
}

static void loom_target_compile_report_merge_memory_interval_envelope_bounds(
    loom_target_compile_report_memory_interval_summary_t* target,
    int64_t begin_min_bytes, int64_t end_max_bytes) {
  if (target->packet_count == 0) {
    target->envelope_begin_min_bytes = begin_min_bytes;
    target->envelope_end_max_bytes = end_max_bytes;
  } else {
    target->envelope_begin_min_bytes =
        iree_min(target->envelope_begin_min_bytes, begin_min_bytes);
    target->envelope_end_max_bytes =
        iree_max(target->envelope_end_max_bytes, end_max_bytes);
  }
  target->envelope_byte_count = loom_target_compile_report_memory_interval_span(
      target->envelope_begin_min_bytes, target->envelope_end_max_bytes);
}

static void loom_target_compile_report_accumulate_memory_interval_summary(
    loom_target_compile_report_memory_interval_summary_t* target,
    const loom_target_compile_report_memory_interval_t* interval,
    loom_target_compile_report_memory_interval_unique_delta_t unique_delta) {
  if (!loom_target_compile_report_memory_interval_has_envelope(interval)) {
    return;
  }
  loom_target_compile_report_merge_memory_interval_envelope_bounds(
      target, interval->begin_min_bytes, interval->end_max_bytes);
  ++target->packet_count;
  switch (unique_delta.kind) {
    case LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_STATIC:
      ++target->exact_static_packet_count;
      target->unique_byte_count += unique_delta.unique_byte_delta;
      break;
    case LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_SYMBOLIC:
      ++target->exact_symbolic_packet_count;
      target->unique_byte_count += unique_delta.unique_byte_delta;
      break;
    case LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_UNIQUE_NONE:
    default:
      break;
  }
}

static void loom_target_compile_report_merge_memory_interval_envelope_summary(
    loom_target_compile_report_memory_interval_summary_t* target,
    const loom_target_compile_report_memory_interval_summary_t* source) {
  if (source->packet_count == 0) {
    return;
  }
  loom_target_compile_report_merge_memory_interval_envelope_bounds(
      target, source->envelope_begin_min_bytes, source->envelope_end_max_bytes);
  target->packet_count += source->packet_count;
}

static void loom_target_compile_report_forget_memory_interval_unique_accounting(
    loom_target_compile_report_memory_interval_summary_t* summary) {
  summary->exact_static_packet_count = 0;
  summary->exact_symbolic_packet_count = 0;
  summary->unique_byte_count = 0;
}

static void loom_target_compile_report_accumulate_source_low_memory_summaries(
    loom_target_compile_report_source_low_memory_summary_t* target,
    const loom_target_compile_report_source_low_memory_summary_t* source) {
  target->packet_count += source->packet_count;
  target->unknown_dynamic_packet_count += source->unknown_dynamic_packet_count;
  target->exact_dynamic_packet_count += source->exact_dynamic_packet_count;
  target->load_packet_count += source->load_packet_count;
  target->store_packet_count += source->store_packet_count;
  target->scalar_packet_count += source->scalar_packet_count;
  target->vector_packet_count += source->vector_packet_count;
  target->source_lane_count += source->source_lane_count;
  target->source_byte_count += source->source_byte_count;
  target->read_byte_count += source->read_byte_count;
  target->write_byte_count += source->write_byte_count;
  target->issued_read_byte_count += source->issued_read_byte_count;
  target->issued_write_byte_count += source->issued_write_byte_count;
  target->issued_read_unknown_width_count +=
      source->issued_read_unknown_width_count;
  target->issued_write_unknown_width_count +=
      source->issued_write_unknown_width_count;
  target->dynamic_packet_count += source->dynamic_packet_count;
  target->dynamic_source_byte_count += source->dynamic_source_byte_count;
  target->dynamic_read_byte_count += source->dynamic_read_byte_count;
  target->dynamic_write_byte_count += source->dynamic_write_byte_count;
  target->dynamic_issued_read_byte_count +=
      source->dynamic_issued_read_byte_count;
  target->dynamic_issued_write_byte_count +=
      source->dynamic_issued_write_byte_count;
  target->dynamic_issued_read_unknown_width_count +=
      source->dynamic_issued_read_unknown_width_count;
  target->dynamic_issued_write_unknown_width_count +=
      source->dynamic_issued_write_unknown_width_count;
  target->contiguous_vector_packet_count +=
      source->contiguous_vector_packet_count;
  target->strided_vector_packet_count += source->strided_vector_packet_count;
  target->unknown_stride_vector_packet_count +=
      source->unknown_stride_vector_packet_count;
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->interval_envelope, &source->interval_envelope);
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->read_interval_envelope, &source->read_interval_envelope);
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->write_interval_envelope, &source->write_interval_envelope);
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

static bool loom_target_compile_report_workgroup_sizes_equal(
    loom_target_workgroup_size_t lhs, loom_target_workgroup_size_t rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

static bool loom_target_compile_report_workgroup_counts_equal(
    loom_target_dispatch_workgroup_count_t lhs,
    loom_target_dispatch_workgroup_count_t rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
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
  const bool report_had_target_resources = iree_any_bit_set(
      report->detail_flags, LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES);
  const bool report_had_dynamic_instruction_mix = iree_any_bit_set(
      report->detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX);
  const bool entry_has_dynamic_instruction_mix = iree_any_bit_set(
      entry_report->detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX);
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
    report->private_memory_bytes = entry_report->private_memory_bytes;
    report->local_memory_bytes = entry_report->local_memory_bytes;
    report->static_instruction_mix = entry_report->static_instruction_mix;
    report->dynamic_instruction_mix = entry_report->dynamic_instruction_mix;
    report->target_resources = entry_report->target_resources;
    report->wait_plan = entry_report->wait_plan;
    report->workload = entry_report->workload;
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
  report->private_memory_bytes = iree_max(report->private_memory_bytes,
                                          entry_report->private_memory_bytes);
  report->local_memory_bytes =
      iree_max(report->local_memory_bytes, entry_report->local_memory_bytes);
  loom_target_compile_report_accumulate_wait_plan(&report->wait_plan,
                                                  &entry_report->wait_plan);
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
      .private_memory_bytes = entry_report->private_memory_bytes,
      .local_memory_bytes = entry_report->local_memory_bytes,
      .static_instruction_mix = entry_report->static_instruction_mix,
      .dynamic_instruction_mix = entry_report->dynamic_instruction_mix,
      .target_resources = entry_report->target_resources,
      .wait_plan = entry_report->wait_plan,
      .workload = entry_report->workload,
      .pressure_row_count = entry_report->pressure_rows.count,
      .pressure_origin_row_count = entry_report->pressure_origin_rows.count,
      .schedule_band_row_count = entry_report->schedule_band_rows.count,
      .schedule_band_summary_row_count =
          entry_report->schedule_band_summary_rows.count,
      .spill_row_count = entry_report->spill_rows.count,
      .allocation_high_water_row_count =
          entry_report->allocation_high_water_rows.count,
      .wait_counter_row_count = entry_report->wait_counter_rows.count,
      .wait_action_row_count = entry_report->wait_action_rows.count,
      .target_capability_row_count = entry_report->target_capability_rows.count,
  };
}

static iree_status_t loom_target_compile_report_append_rows(
    loom_target_compile_report_row_list_t* target,
    const loom_target_compile_report_row_list_t* source,
    iree_host_size_t row_size, iree_allocator_t allocator) {
  for (const loom_target_compile_report_vec_t* vec = source->head; vec != NULL;
       vec = vec->next) {
    const uint8_t* rows =
        (const uint8_t*)loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append(
          target, row_size, allocator, rows + i * row_size));
    }
  }
  return iree_ok_status();
}

static void loom_target_compile_report_merge_source_low_memory_summary(
    loom_target_compile_report_source_low_memory_summary_t* target,
    const loom_target_compile_report_source_low_memory_summary_t* source) {
  target->packet_count += source->packet_count;
  target->unknown_dynamic_packet_count += source->unknown_dynamic_packet_count;
  target->exact_dynamic_packet_count += source->exact_dynamic_packet_count;
  target->load_packet_count += source->load_packet_count;
  target->store_packet_count += source->store_packet_count;
  target->scalar_packet_count += source->scalar_packet_count;
  target->vector_packet_count += source->vector_packet_count;
  target->source_lane_count += source->source_lane_count;
  target->source_byte_count += source->source_byte_count;
  target->read_byte_count += source->read_byte_count;
  target->write_byte_count += source->write_byte_count;
  target->issued_read_byte_count += source->issued_read_byte_count;
  target->issued_write_byte_count += source->issued_write_byte_count;
  target->issued_read_unknown_width_count +=
      source->issued_read_unknown_width_count;
  target->issued_write_unknown_width_count +=
      source->issued_write_unknown_width_count;
  target->dynamic_packet_count += source->dynamic_packet_count;
  target->dynamic_source_byte_count += source->dynamic_source_byte_count;
  target->dynamic_read_byte_count += source->dynamic_read_byte_count;
  target->dynamic_write_byte_count += source->dynamic_write_byte_count;
  target->dynamic_issued_read_byte_count +=
      source->dynamic_issued_read_byte_count;
  target->dynamic_issued_write_byte_count +=
      source->dynamic_issued_write_byte_count;
  target->dynamic_issued_read_unknown_width_count +=
      source->dynamic_issued_read_unknown_width_count;
  target->dynamic_issued_write_unknown_width_count +=
      source->dynamic_issued_write_unknown_width_count;
  target->contiguous_vector_packet_count +=
      source->contiguous_vector_packet_count;
  target->strided_vector_packet_count += source->strided_vector_packet_count;
  target->unknown_stride_vector_packet_count +=
      source->unknown_stride_vector_packet_count;
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->interval_envelope, &source->interval_envelope);
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->read_interval_envelope, &source->read_interval_envelope);
  loom_target_compile_report_merge_memory_interval_envelope_summary(
      &target->write_interval_envelope, &source->write_interval_envelope);
  loom_target_compile_report_forget_memory_interval_unique_accounting(
      &target->interval_envelope);
  loom_target_compile_report_forget_memory_interval_unique_accounting(
      &target->read_interval_envelope);
  loom_target_compile_report_forget_memory_interval_unique_accounting(
      &target->write_interval_envelope);
}

static void loom_target_compile_report_merge_source_low_memory_root_summary(
    loom_target_compile_report_source_low_memory_root_summary_t* target,
    const loom_target_compile_report_source_low_memory_root_summary_t* source) {
  loom_target_compile_report_merge_source_low_memory_summary(&target->summary,
                                                             &source->summary);
}

static void loom_target_compile_report_merge_source_low_memory_argument_summary(
    loom_target_compile_report_source_low_memory_argument_summary_t* target,
    const loom_target_compile_report_source_low_memory_argument_summary_t*
        source) {
  loom_target_compile_report_merge_source_low_memory_summary(&target->summary,
                                                             &source->summary);
}

static void loom_target_compile_report_merge_source_low_memory_strategy_summary(
    loom_target_compile_report_source_low_memory_strategy_summary_t* target,
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        source) {
  loom_target_compile_report_merge_source_low_memory_summary(&target->summary,
                                                             &source->summary);
}

static loom_target_compile_report_source_low_memory_root_summary_t*
loom_target_compile_report_find_source_low_memory_root_summary(
    loom_target_compile_report_t* report, iree_string_view_t function_name,
    iree_string_view_t source_root_name, uint16_t source_root_argument_index,
    iree_string_view_t memory_space) {
  const bool has_root_identity = !iree_string_view_is_empty(source_root_name) ||
                                 source_root_argument_index != UINT16_MAX;
  if (!has_root_identity) {
    return NULL;
  }
  for (loom_target_compile_report_vec_t* vec =
           report->source_low_memory_root_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_source_low_memory_root_summary_t* summaries =
        (loom_target_compile_report_source_low_memory_root_summary_t*)
            loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      loom_target_compile_report_source_low_memory_root_summary_t* summary =
          &summaries[i];
      if (iree_string_view_equal(summary->function_name, function_name) &&
          iree_string_view_equal(summary->source_root_name, source_root_name) &&
          summary->source_root_argument_index == source_root_argument_index &&
          iree_string_view_equal(summary->memory_space, memory_space)) {
        return summary;
      }
    }
  }
  return NULL;
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_root_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_root_summary_t* row) {
  loom_target_compile_report_source_low_memory_root_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_root_summary(
          report, row->function_name, row->source_root_name,
          row->source_root_argument_index, row->memory_space);
  if (summary != NULL) {
    loom_target_compile_report_merge_source_low_memory_root_summary(summary,
                                                                    row);
    return iree_ok_status();
  } else if (iree_string_view_is_empty(row->source_root_name) &&
             row->source_root_argument_index == UINT16_MAX) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_root_summaries, sizeof(*row),
      report->allocator, row);
}

static loom_target_compile_report_source_low_memory_argument_summary_t*
loom_target_compile_report_find_source_low_memory_argument_summary(
    loom_target_compile_report_t* report, iree_string_view_t function_name,
    uint16_t source_root_argument_index, iree_string_view_t memory_space) {
  if (source_root_argument_index == UINT16_MAX) {
    return NULL;
  }
  for (loom_target_compile_report_vec_t* vec =
           report->source_low_memory_argument_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_source_low_memory_argument_summary_t* summaries =
        (loom_target_compile_report_source_low_memory_argument_summary_t*)
            loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      loom_target_compile_report_source_low_memory_argument_summary_t* summary =
          &summaries[i];
      if (iree_string_view_equal(summary->function_name, function_name) &&
          summary->source_root_argument_index == source_root_argument_index &&
          iree_string_view_equal(summary->memory_space, memory_space)) {
        return summary;
      }
    }
  }
  return NULL;
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_argument_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_argument_summary_t*
        row) {
  loom_target_compile_report_source_low_memory_argument_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_argument_summary(
          report, row->function_name, row->source_root_argument_index,
          row->memory_space);
  if (summary != NULL) {
    loom_target_compile_report_merge_source_low_memory_argument_summary(summary,
                                                                        row);
    return iree_ok_status();
  } else if (row->source_root_argument_index == UINT16_MAX) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_argument_summaries, sizeof(*row),
      report->allocator, row);
}

static loom_target_compile_report_source_low_memory_strategy_summary_t
loom_target_compile_report_source_low_memory_strategy_summary_from_row(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  return (loom_target_compile_report_source_low_memory_strategy_summary_t){
      .function_name = row->function_name,
      .memory_space = row->memory_space,
      .operation_kind = row->operation_kind,
      .strategy_key = row->strategy_key,
      .storage_element_format = row->storage_element_format,
      .storage_scale_format = row->storage_scale_format,
      .storage_secondary_scale_format = row->storage_secondary_scale_format,
      .storage_payload_packing = row->storage_payload_packing,
      .storage_scale_topology = row->storage_scale_topology,
      .storage_affine_policy = row->storage_affine_policy,
      .storage_rounding_policy = row->storage_rounding_policy,
      .storage_codebook_policy = row->storage_codebook_policy,
      .storage_sparsity_policy = row->storage_sparsity_policy,
  };
}

static bool
loom_target_compile_report_source_low_memory_strategy_summaries_match(
    const loom_target_compile_report_source_low_memory_strategy_summary_t* lhs,
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        rhs) {
  return iree_string_view_equal(lhs->function_name, rhs->function_name) &&
         iree_string_view_equal(lhs->memory_space, rhs->memory_space) &&
         iree_string_view_equal(lhs->operation_kind, rhs->operation_kind) &&
         iree_string_view_equal(lhs->strategy_key, rhs->strategy_key) &&
         iree_string_view_equal(lhs->storage_element_format,
                                rhs->storage_element_format) &&
         iree_string_view_equal(lhs->storage_scale_format,
                                rhs->storage_scale_format) &&
         iree_string_view_equal(lhs->storage_secondary_scale_format,
                                rhs->storage_secondary_scale_format) &&
         iree_string_view_equal(lhs->storage_payload_packing,
                                rhs->storage_payload_packing) &&
         iree_string_view_equal(lhs->storage_scale_topology,
                                rhs->storage_scale_topology) &&
         iree_string_view_equal(lhs->storage_affine_policy,
                                rhs->storage_affine_policy) &&
         iree_string_view_equal(lhs->storage_rounding_policy,
                                rhs->storage_rounding_policy) &&
         iree_string_view_equal(lhs->storage_codebook_policy,
                                rhs->storage_codebook_policy) &&
         iree_string_view_equal(lhs->storage_sparsity_policy,
                                rhs->storage_sparsity_policy);
}

static loom_target_compile_report_source_low_memory_strategy_summary_t*
loom_target_compile_report_find_source_low_memory_strategy_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        row) {
  if (iree_string_view_is_empty(row->strategy_key)) {
    return NULL;
  }
  for (loom_target_compile_report_vec_t* vec =
           report->source_low_memory_strategy_summaries.head;
       vec != NULL; vec = vec->next) {
    loom_target_compile_report_source_low_memory_strategy_summary_t* summaries =
        (loom_target_compile_report_source_low_memory_strategy_summary_t*)
            loom_target_compile_report_vec_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      loom_target_compile_report_source_low_memory_strategy_summary_t* summary =
          &summaries[i];
      if (loom_target_compile_report_source_low_memory_strategy_summaries_match(
              summary, row)) {
        return summary;
      }
    }
  }
  return NULL;
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_strategy_summary_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        row) {
  loom_target_compile_report_source_low_memory_strategy_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_strategy_summary(report,
                                                                         row);
  if (summary != NULL) {
    loom_target_compile_report_merge_source_low_memory_strategy_summary(summary,
                                                                        row);
    return iree_ok_status();
  } else if (iree_string_view_is_empty(row->strategy_key)) {
    return iree_ok_status();
  }
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_strategy_summaries, sizeof(*row),
      report->allocator, row);
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
  if (!loom_target_compile_report_checked_mul_u64(
          row->emitted_low_op_count, execution_count,
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
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->pressure_rows, &entry_report->pressure_rows,
        sizeof(loom_target_compile_report_pressure_row_t), report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->pressure_origin_rows, &entry_report->pressure_origin_rows,
        sizeof(loom_target_compile_report_pressure_origin_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->schedule_band_rows, &entry_report->schedule_band_rows,
        sizeof(loom_target_compile_report_schedule_band_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->schedule_band_summary_rows,
        &entry_report->schedule_band_summary_rows,
        sizeof(loom_target_compile_report_schedule_band_summary_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->spill_rows, &entry_report->spill_rows,
        sizeof(loom_target_compile_report_spill_row_t), report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->allocation_failure_rows,
        &entry_report->allocation_failure_rows,
        sizeof(loom_target_compile_report_allocation_failure_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->allocation_high_water_rows,
        &entry_report->allocation_high_water_rows,
        sizeof(loom_target_compile_report_allocation_high_water_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->wait_counter_rows, &entry_report->wait_counter_rows,
        sizeof(loom_target_compile_report_wait_counter_row_t),
        report->allocator));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->wait_action_rows, &entry_report->wait_action_rows,
        sizeof(loom_target_compile_report_wait_action_row_t),
        report->allocator));
  }
  if (iree_any_bit_set(entry_report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->source_low_rows, &entry_report->source_low_rows,
        sizeof(loom_target_compile_report_source_low_row_t),
        report->allocator));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->source_low_target_rows, &entry_report->source_low_target_rows,
        sizeof(loom_target_compile_report_source_low_target_row_t),
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
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->source_low_memory_rows, &entry_report->source_low_memory_rows,
        sizeof(loom_target_compile_report_source_low_memory_row_t),
        report->allocator));
    for (const loom_target_compile_report_vec_t* vec =
             entry_report->source_low_memory_root_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_root_summary_t* rows =
          (const loom_target_compile_report_source_low_memory_root_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_record_source_low_memory_root_summary_row(
                report, &rows[i]));
      }
    }
    for (const loom_target_compile_report_vec_t* vec =
             entry_report->source_low_memory_argument_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_argument_summary_t* rows =
          (const loom_target_compile_report_source_low_memory_argument_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_record_source_low_memory_argument_summary_row(
                report, &rows[i]));
      }
    }
    for (const loom_target_compile_report_vec_t* vec =
             entry_report->source_low_memory_strategy_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_strategy_summary_t* rows =
          (const loom_target_compile_report_source_low_memory_strategy_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_record_source_low_memory_strategy_summary_row(
                report, &rows[i]));
      }
    }
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
        &report->math_legalization_rows, &entry_report->math_legalization_rows,
        sizeof(loom_target_compile_report_math_row_t), report->allocator));
  }
  if (iree_any_bit_set(
          entry_report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_rows(
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

iree_status_t loom_target_compile_report_record_source_low_target_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_target_row_t* row) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  return loom_target_compile_report_row_list_append(
      &report->source_low_target_rows, sizeof(*row), report->allocator, row);
}

static bool loom_target_compile_report_source_low_memory_row_is_load(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  return iree_string_view_equal(row->operation_kind, IREE_SV("load"));
}

static bool loom_target_compile_report_source_low_memory_row_is_store(
    const loom_target_compile_report_source_low_memory_row_t* row) {
  return iree_string_view_equal(row->operation_kind, IREE_SV("store"));
}

static void loom_target_compile_report_accumulate_source_low_memory_summary(
    loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_memory_interval_unique_delta_t unique_delta,
    loom_target_compile_report_memory_interval_unique_delta_t
        direction_unique_delta) {
  const uint64_t lane_count = row->vector_lane_count;
  const uint64_t source_byte_count = lane_count * row->element_byte_count;
  const bool is_load =
      loom_target_compile_report_source_low_memory_row_is_load(row);
  const bool is_store =
      loom_target_compile_report_source_low_memory_row_is_store(row);
  ++summary->packet_count;
  summary->source_lane_count += lane_count;
  summary->source_byte_count += source_byte_count;
  if (is_load) {
    ++summary->load_packet_count;
    summary->read_byte_count += source_byte_count;
    loom_target_compile_report_accumulate_memory_interval_summary(
        &summary->read_interval_envelope, &row->source_interval,
        direction_unique_delta);
  } else if (is_store) {
    ++summary->store_packet_count;
    summary->write_byte_count += source_byte_count;
    loom_target_compile_report_accumulate_memory_interval_summary(
        &summary->write_interval_envelope, &row->source_interval,
        direction_unique_delta);
  }
  summary->issued_read_byte_count += row->issued_read_byte_count;
  summary->issued_write_byte_count += row->issued_write_byte_count;
  summary->issued_read_unknown_width_count +=
      row->issued_read_unknown_width_count;
  summary->issued_write_unknown_width_count +=
      row->issued_write_unknown_width_count;
  loom_target_compile_report_accumulate_memory_interval_summary(
      &summary->interval_envelope, &row->source_interval, unique_delta);
  if (row->execution_count_plus_one ==
      LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_EXECUTION_COUNT_PLUS_ONE_UNKNOWN) {
    ++summary->unknown_dynamic_packet_count;
  } else {
    const uint64_t execution_count = row->execution_count_plus_one - 1;
    uint64_t dynamic_source_byte_count = 0;
    uint64_t dynamic_issued_read_byte_count = 0;
    uint64_t dynamic_issued_write_byte_count = 0;
    uint64_t dynamic_issued_read_unknown_width_count = 0;
    uint64_t dynamic_issued_write_unknown_width_count = 0;
    const bool dynamic_counts_ok =
        loom_target_compile_report_checked_mul_u64(
            source_byte_count, execution_count, &dynamic_source_byte_count) &&
        loom_target_compile_report_checked_mul_u64(
            row->issued_read_byte_count, execution_count,
            &dynamic_issued_read_byte_count) &&
        loom_target_compile_report_checked_mul_u64(
            row->issued_write_byte_count, execution_count,
            &dynamic_issued_write_byte_count) &&
        loom_target_compile_report_checked_mul_u64(
            row->issued_read_unknown_width_count, execution_count,
            &dynamic_issued_read_unknown_width_count) &&
        loom_target_compile_report_checked_mul_u64(
            row->issued_write_unknown_width_count, execution_count,
            &dynamic_issued_write_unknown_width_count);
    uint64_t new_dynamic_packet_count = summary->dynamic_packet_count;
    uint64_t new_dynamic_source_byte_count = summary->dynamic_source_byte_count;
    uint64_t new_dynamic_read_byte_count = summary->dynamic_read_byte_count;
    uint64_t new_dynamic_write_byte_count = summary->dynamic_write_byte_count;
    uint64_t new_dynamic_issued_read_byte_count =
        summary->dynamic_issued_read_byte_count;
    uint64_t new_dynamic_issued_write_byte_count =
        summary->dynamic_issued_write_byte_count;
    uint64_t new_dynamic_issued_read_unknown_width_count =
        summary->dynamic_issued_read_unknown_width_count;
    uint64_t new_dynamic_issued_write_unknown_width_count =
        summary->dynamic_issued_write_unknown_width_count;
    bool dynamic_accumulation_ok =
        dynamic_counts_ok &&
        loom_target_compile_report_checked_add_u64(new_dynamic_packet_count,
                                                   execution_count,
                                                   &new_dynamic_packet_count) &&
        loom_target_compile_report_checked_add_u64(
            new_dynamic_source_byte_count, dynamic_source_byte_count,
            &new_dynamic_source_byte_count) &&
        loom_target_compile_report_checked_add_u64(
            new_dynamic_issued_read_byte_count, dynamic_issued_read_byte_count,
            &new_dynamic_issued_read_byte_count) &&
        loom_target_compile_report_checked_add_u64(
            new_dynamic_issued_write_byte_count,
            dynamic_issued_write_byte_count,
            &new_dynamic_issued_write_byte_count) &&
        loom_target_compile_report_checked_add_u64(
            new_dynamic_issued_read_unknown_width_count,
            dynamic_issued_read_unknown_width_count,
            &new_dynamic_issued_read_unknown_width_count) &&
        loom_target_compile_report_checked_add_u64(
            new_dynamic_issued_write_unknown_width_count,
            dynamic_issued_write_unknown_width_count,
            &new_dynamic_issued_write_unknown_width_count);
    if (dynamic_accumulation_ok && is_load) {
      dynamic_accumulation_ok = loom_target_compile_report_checked_add_u64(
          new_dynamic_read_byte_count, dynamic_source_byte_count,
          &new_dynamic_read_byte_count);
    } else if (dynamic_accumulation_ok && is_store) {
      dynamic_accumulation_ok = loom_target_compile_report_checked_add_u64(
          new_dynamic_write_byte_count, dynamic_source_byte_count,
          &new_dynamic_write_byte_count);
    }
    if (!dynamic_accumulation_ok) {
      ++summary->unknown_dynamic_packet_count;
    } else {
      ++summary->exact_dynamic_packet_count;
      summary->dynamic_packet_count = new_dynamic_packet_count;
      summary->dynamic_source_byte_count = new_dynamic_source_byte_count;
      summary->dynamic_read_byte_count = new_dynamic_read_byte_count;
      summary->dynamic_write_byte_count = new_dynamic_write_byte_count;
      summary->dynamic_issued_read_byte_count =
          new_dynamic_issued_read_byte_count;
      summary->dynamic_issued_write_byte_count =
          new_dynamic_issued_write_byte_count;
      summary->dynamic_issued_read_unknown_width_count =
          new_dynamic_issued_read_unknown_width_count;
      summary->dynamic_issued_write_unknown_width_count =
          new_dynamic_issued_write_unknown_width_count;
    }
  }
  if (lane_count == 1) {
    ++summary->scalar_packet_count;
  } else if (lane_count > 1) {
    ++summary->vector_packet_count;
    if (row->element_byte_count == 0 || row->vector_lane_stride_bytes == 0) {
      ++summary->unknown_stride_vector_packet_count;
    } else if (row->vector_lane_stride_bytes == row->element_byte_count) {
      ++summary->contiguous_vector_packet_count;
    } else {
      ++summary->strided_vector_packet_count;
    }
  }
}

static void
loom_target_compile_report_accumulate_source_low_memory_root_summary(
    loom_target_compile_report_source_low_memory_root_summary_t* summary,
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_memory_interval_unique_delta_t unique_delta,
    loom_target_compile_report_memory_interval_unique_delta_t
        direction_unique_delta) {
  loom_target_compile_report_accumulate_source_low_memory_summary(
      &summary->summary, row, unique_delta, direction_unique_delta);
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_root_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_memory_interval_unique_delta_t unique_delta,
    loom_target_compile_report_memory_interval_unique_delta_t
        direction_unique_delta) {
  loom_target_compile_report_source_low_memory_root_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_root_summary(
          report, row->function_name, row->source_root_name,
          row->source_root_argument_index, row->memory_space);
  if (summary != NULL) {
    loom_target_compile_report_accumulate_source_low_memory_root_summary(
        summary, row, unique_delta, direction_unique_delta);
    return iree_ok_status();
  } else if (iree_string_view_is_empty(row->source_root_name) &&
             row->source_root_argument_index == UINT16_MAX) {
    return iree_ok_status();
  }

  loom_target_compile_report_source_low_memory_root_summary_t new_summary = {
      .function_name = row->function_name,
      .source_root_name = row->source_root_name,
      .source_root_argument_index = row->source_root_argument_index,
      .memory_space = row->memory_space,
  };
  loom_target_compile_report_accumulate_source_low_memory_root_summary(
      &new_summary, row, unique_delta, direction_unique_delta);
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_root_summaries, sizeof(new_summary),
      report->allocator, &new_summary);
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_argument_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_memory_interval_unique_delta_t unique_delta,
    loom_target_compile_report_memory_interval_unique_delta_t
        direction_unique_delta) {
  if (row->source_root_argument_index == UINT16_MAX) {
    return iree_ok_status();
  }
  loom_target_compile_report_source_low_memory_argument_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_argument_summary(
          report, row->function_name, row->source_root_argument_index,
          row->memory_space);
  if (summary == NULL) {
    loom_target_compile_report_source_low_memory_argument_summary_t
        new_summary = {
            .function_name = row->function_name,
            .source_root_argument_index = row->source_root_argument_index,
            .memory_space = row->memory_space,
        };
    loom_target_compile_report_accumulate_source_low_memory_summary(
        &new_summary.summary, row, unique_delta, direction_unique_delta);
    return loom_target_compile_report_row_list_append(
        &report->source_low_memory_argument_summaries, sizeof(new_summary),
        report->allocator, &new_summary);
  }
  loom_target_compile_report_accumulate_source_low_memory_summary(
      &summary->summary, row, unique_delta, direction_unique_delta);
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_record_source_low_memory_strategy_summary(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row) {
  if (iree_string_view_is_empty(row->strategy_key)) {
    return iree_ok_status();
  }
  loom_target_compile_report_source_low_memory_strategy_summary_t key =
      loom_target_compile_report_source_low_memory_strategy_summary_from_row(
          row);
  loom_target_compile_report_source_low_memory_strategy_summary_t* summary =
      loom_target_compile_report_find_source_low_memory_strategy_summary(report,
                                                                         &key);
  const loom_target_compile_report_memory_interval_unique_delta_t
      no_unique_delta = {0};
  if (summary != NULL) {
    loom_target_compile_report_accumulate_source_low_memory_summary(
        &summary->summary, row, no_unique_delta, no_unique_delta);
    return iree_ok_status();
  }
  loom_target_compile_report_accumulate_source_low_memory_summary(
      &key.summary, row, no_unique_delta, no_unique_delta);
  return loom_target_compile_report_row_list_append(
      &report->source_low_memory_strategy_summaries, sizeof(key),
      report->allocator, &key);
}

iree_status_t loom_target_compile_report_record_source_low_memory_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_memory_row_t* row) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  loom_target_compile_report_memory_interval_unique_delta_t unique_delta;
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_calculate_source_low_unique_interval_delta(
          report, row, /*match_operation_kind=*/false, &unique_delta));
  loom_target_compile_report_memory_interval_unique_delta_t
      direction_unique_delta;
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_calculate_source_low_unique_interval_delta(
          report, row, /*match_operation_kind=*/true, &direction_unique_delta));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_row_list_append(
      &report->source_low_memory_rows, sizeof(*row), report->allocator, row));
  const loom_target_compile_report_memory_interval_unique_delta_t
      no_unique_delta = {0};
  loom_target_compile_report_accumulate_source_low_memory_summary(
      &report->source_low_memory_summary, row, no_unique_delta,
      no_unique_delta);
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_source_low_memory_root_summary(
          report, row, unique_delta, direction_unique_delta));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_source_low_memory_argument_summary(
          report, row, unique_delta, direction_unique_delta));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_source_low_memory_strategy_summary(
          report, row));
  return iree_ok_status();
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
