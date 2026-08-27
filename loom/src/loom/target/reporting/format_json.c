// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/format_json.h"

#include <stdint.h>

#include "loom/target/reporting/format_planning.h"
#include "loom/target/reporting/schema.h"

iree_status_t loom_target_compile_report_json_write_optional_string_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_string_field(object, name, value);
}

iree_status_t loom_target_compile_report_json_write_optional_u16_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    uint16_t value) {
  if (value == UINT16_MAX) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint32_field(object, name, value);
}

iree_status_t loom_target_compile_report_json_write_optional_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    uint64_t value) {
  if (value == UINT64_MAX) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint64_field(object, name, value);
}

iree_status_t loom_target_compile_report_json_write_optional_u32_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    uint32_t value) {
  if (value == UINT32_MAX) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint32_field(object, name, value);
}

static iree_status_t loom_target_compile_report_format_schedule_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("node_count"), report->schedule_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scheduled_node_count"), report->scheduled_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("dependency_count"), report->schedule_dependency_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("resource_use_count"),
      report->schedule_resource_use_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("hazard_gap_count"), report->schedule_hazard_gap_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("model_summary_count"),
      report->schedule_model_summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("register_pressure_summary_count"),
      report->register_pressure_summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("register_pressure_peak_live_units"),
      report->register_pressure_peak_live_units));
  return loom_json_object_end(&object);
}

iree_status_t loom_target_compile_report_format_instruction_mix_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("descriptor_count"), mix->descriptor_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unknown_count"), mix->unknown_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scalar_alu_count"), mix->scalar_alu_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("vector_alu_count"), mix->vector_alu_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("matrix_count"), mix->matrix_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("mfma_count"), mix->mfma_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("smfmac_count"), mix->smfmac_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("wmma_count"), mix->wmma_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("swmmac_count"), mix->swmmac_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("dot_count"), mix->dot_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("global_memory_count"), mix->global_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("global_load_count"), mix->global_load_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("global_store_count"), mix->global_store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("buffer_load_count"), mix->buffer_load_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("buffer_store_count"), mix->buffer_store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("flat_memory_count"), mix->flat_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_memory_count"), mix->local_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scalar_memory_count"), mix->scalar_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_memory_count"), mix->private_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("generic_memory_count"), mix->generic_memory_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("memory_read_unknown_width_count"),
      mix->memory_read_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("memory_write_unknown_width_count"),
      mix->memory_write_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("memory_read_byte_count"), mix->memory_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("memory_write_byte_count"),
      mix->memory_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("global_load_byte_count"), mix->global_load_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("global_store_byte_count"),
      mix->global_store_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("buffer_load_byte_count"), mix->buffer_load_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("buffer_store_byte_count"),
      mix->buffer_store_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("flat_read_byte_count"), mix->flat_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("flat_write_byte_count"), mix->flat_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_read_byte_count"), mix->local_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_write_byte_count"), mix->local_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scalar_read_byte_count"), mix->scalar_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scalar_write_byte_count"),
      mix->scalar_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_read_byte_count"),
      mix->private_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_write_byte_count"),
      mix->private_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unclassified_read_byte_count"),
      mix->unclassified_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unclassified_write_byte_count"),
      mix->unclassified_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("atomic_count"), mix->atomic_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("branch_count"), mix->branch_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("barrier_count"), mix->barrier_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("control_count"), mix->control_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("conversion_count"), mix->conversion_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("cache_count"), mix->cache_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("register_move_count"), mix->register_move_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_allocation_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint64_field(&object, IREE_SV("assignment_count"),
                                          report->allocation_assignment_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("spill_count"), report->allocation_spill_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint64_field(&object, IREE_SV("spill_plan_count"),
                                          report->allocation_spill_plan_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("coalesced_copy_count"),
      report->allocation_coalesced_copy_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_copy_count"),
      report->allocation_materialized_copy_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_spill_storage_count"),
      report->allocation_materialized_spill_storage_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_spill_storage_bytes"),
      report->allocation_materialized_spill_storage_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_spill_store_count"),
      report->allocation_materialized_spill_store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_spill_store_bytes"),
      report->allocation_materialized_spill_store_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_reload_count"),
      report->allocation_materialized_reload_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("materialized_reload_bytes"),
      report->allocation_materialized_reload_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("storage_lease_count"),
      report->allocation_storage_lease_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("storage_lease_instance_count"),
      report->allocation_storage_lease_instance_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("storage_release_action_count"),
      report->allocation_storage_release_action_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_move_cause_json(
    loom_json_array_writer_t* array,
    const loom_target_compile_report_move_cause_descriptor_t* descriptor,
    const loom_target_compile_report_move_cause_counts_t* counts) {
  if (counts->packet_count == 0 && counts->unit_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(array));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(array->stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("cause"), descriptor->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("packet_count"), counts->packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unit_count"), counts->unit_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_move_cause_counts_json(
    const loom_target_compile_report_move_cause_counts_t* counts,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  uint64_t kind_count = 0;
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  loom_target_compile_report_move_cause_counts_totals(
      counts, &kind_count, &packet_count, &unit_count);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("kind_count"), kind_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("packet_count"), packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unit_count"), unit_count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("causes")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors);
         ++i) {
      const loom_target_compile_report_move_cause_descriptor_t* descriptor =
          &loom_target_compile_report_move_cause_descriptors[i];
      IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_cause_json(
          &array, descriptor, &counts[descriptor->cause]));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_move_causes_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  return loom_target_compile_report_format_move_cause_counts_json(
      report->move_causes, mode, stream);
}

static iree_status_t loom_target_compile_report_format_wait_plan_json(
    const loom_target_compile_report_wait_plan_t* wait_plan,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("action_count"), wait_plan->action_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("explicit_action_count"),
      wait_plan->explicit_action_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("planned_action_count"),
      wait_plan->planned_action_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("full_drain_count"), wait_plan->full_drain_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("partial_wait_count"), wait_plan->partial_wait_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("drained_count"), wait_plan->drained_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("max_drained_count"), wait_plan->max_drained_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("max_outstanding_before"),
      wait_plan->max_outstanding_before));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("max_full_drain_outstanding_before"),
      wait_plan->max_full_drain_outstanding_before));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_wait_counter_row_json(
    const loom_target_compile_report_wait_counter_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("counter"), row->counter_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("counter_id"), row->counter_id));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("summary")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_wait_plan_json(&row->summary, stream));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_wait_counter_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->wait_counter_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->wait_counter_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_wait_counter_row_t* rows =
          (const loom_target_compile_report_wait_counter_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_wait_counter_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_wait_reason_summary_row_json(
    const loom_target_compile_report_wait_reason_summary_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("counter"), row->counter_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("counter_id"), row->counter_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("reason"), row->reason_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("reason_id"), row->reason_id));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("summary")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_wait_plan_json(&row->summary, stream));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_wait_reason_summary_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  (void)mode;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->wait_reason_summary_rows.count));
  if (report->wait_reason_summary_rows.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->wait_reason_summary_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_wait_reason_summary_row_t* rows =
          (const loom_target_compile_report_wait_reason_summary_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_wait_reason_summary_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_wait_action_row_json(
    const loom_target_compile_report_wait_action_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("counter"), row->counter_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("counter_id"), row->counter_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("action"), row->action_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("action_id"), row->action_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("reason"), row->reason_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("reason_id"), row->reason_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("block_index"), row->block_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("node_index"), row->node_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("scheduled_ordinal"), row->scheduled_ordinal));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("producer_node"), row->producer_node));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("producer_scheduled_ordinal"),
      row->producer_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("producer_operation"),
          row->producer_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("producer_descriptor_key"),
          row->producer_descriptor_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("producer_semantic_tag"),
          row->producer_semantic_tag));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("consumer_node"), row->consumer_node));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("consumer_scheduled_ordinal"),
      row->consumer_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("consumer_operation"),
          row->consumer_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("consumer_descriptor_key"),
          row->consumer_descriptor_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("consumer_semantic_tag"),
          row->consumer_semantic_tag));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("target_count"), row->target_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("outstanding_before"), row->outstanding_before));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("outstanding_after"), row->outstanding_after));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("drained_count"), row->drained_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_wait_action_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->wait_action_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->wait_action_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_wait_action_row_t* rows =
          (const loom_target_compile_report_wait_action_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_wait_action_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_target_capability_row_json(
    const loom_target_compile_report_target_capability_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_family"), row->target_family_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("namespace"), row->namespace_name));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("key"), row->key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("value_kind"),
      loom_target_compile_report_capability_value_kind_name(row->value_kind)));
  switch (row->value_kind) {
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL: {
      IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
          &object, IREE_SV("value_bool"), row->value_u64 != 0));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_U64: {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &object, IREE_SV("value_u64"), row->value_u64));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING: {
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &object, IREE_SV("value_string"), row->value_string));
      break;
    }
    case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_NONE:
    default:
      break;
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_target_capability_rows_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->target_capability_rows.count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->target_capability_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_target_capability_row_t* rows =
        (const loom_target_compile_report_target_capability_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_target_capability_row_json(
              &rows[i], row_index, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_emission_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint64_field(&object, IREE_SV("instruction_count"),
                                          report->emitted_instruction_count));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION_BREAKDOWN)) {
    const loom_target_compile_report_emission_breakdown_t* breakdown =
        &report->emission_breakdown;
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("body_instruction_count"),
        breakdown->body_instruction_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("entry_instruction_count"),
        breakdown->entry_instruction_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("coissued_instruction_count"),
        breakdown->coissued_instruction_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("coissued_component_count"),
        breakdown->coissued_component_count));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("code_byte_count"), report->emitted_code_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("code_storage_byte_count"),
      report->emitted_code_storage_byte_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_memory_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_bytes"), report->private_memory_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_bytes"), report->local_memory_bytes));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_target_resource_registers_json(
    iree_string_view_t register_class, uint64_t final_register_count,
    uint64_t scheduled_pressure_peak_live_units, uint64_t final_overhead_units,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (!iree_string_view_is_empty(register_class)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("register_class"), register_class));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("final")));
  loom_json_object_writer_t final_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &final_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &final_object, IREE_SV("register_count"), final_register_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &final_object, IREE_SV("overhead_units"), final_overhead_units));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&final_object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("scheduled_pressure")));
  loom_json_object_writer_t pressure_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &pressure_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &pressure_object, IREE_SV("peak_live_units"),
      scheduled_pressure_peak_live_units));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&pressure_object));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_residency_summary_json(
    const loom_target_residency_summary_t* summary,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("best_tier"), summary->best_tier));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("current_tier"), summary->tier));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("limiting_resource_count"),
      summary->limiting_resource_count));
  if (iree_any_bit_set(
          summary->flags,
          LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("next_better_tier"), summary->next_better_tier));
  }
  if (iree_any_bit_set(
          summary->flags,
          LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_UNIQUE_LIMITING_RESOURCE)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("unique_limiting_resource")));
    loom_json_object_writer_t resource_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &resource_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &resource_object, IREE_SV("name"), summary->limiting_resource));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &resource_object, IREE_SV("units"), summary->limiting_resource_units));
    if (iree_any_bit_set(
            summary->flags,
            LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER)) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &resource_object, IREE_SV("reduction_units_to_next_better_tier"),
          summary->limiting_resource_reduction_units_to_next_better_tier));
    }
    if (iree_any_bit_set(
            summary->flags,
            LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_LIMITING_RESOURCE_NEXT_WORSE_TIER)) {
      IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&resource_object,
                                                        IREE_SV("next_worse")));
      loom_json_object_writer_t next_worse_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &next_worse_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &next_worse_object, IREE_SV("tier"),
          summary->limiting_resource_next_worse_tier));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &next_worse_object, IREE_SV("cliff_units"),
          summary->limiting_resource_next_worse_cliff_units));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &next_worse_object, IREE_SV("additional_units"),
          summary->limiting_resource_additional_units_to_next_worse_tier));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&next_worse_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&resource_object));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_target_resources_json(
    const loom_target_compile_report_target_resources_t* resources,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("scalar")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_target_resource_registers_json(
          resources->scalar_register_class, resources->scalar_register_count,
          resources->scalar_pressure_peak_live_units,
          resources->scalar_register_overhead_units, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("vector")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_target_resource_registers_json(
          resources->vector_register_class, resources->vector_register_count,
          resources->vector_pressure_peak_live_units,
          resources->vector_register_overhead_units, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("subgroup_size"), resources->subgroup_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("max_subgroups_per_simd"),
      resources->max_subgroups_per_simd));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("resident_subgroups_per_simd"),
      resources->resident_subgroups_per_simd));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("occupancy_percent"), resources->occupancy_percent));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("limiting_resource"), resources->limiting_resource));
  if (loom_target_residency_summary_is_valid(&resources->residency_summary)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("residency")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_residency_summary_json(
            &resources->residency_summary, stream));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_dimension3_json(
    uint32_t x, uint32_t y, uint32_t z, uint64_t flat, bool include_flat,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("x"), x));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("y"), y));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&object, IREE_SV("z"), z));
  if (include_flat) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_uint64_field(&object, IREE_SV("flat"), flat));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_workload_json(
    const loom_target_compile_report_workload_t* workload,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (iree_any_bit_set(workload->flags,
                       LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("workgroup_size")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_dimension3_json(
        workload->workgroup_size.x, workload->workgroup_size.y,
        workload->workgroup_size.z, workload->flat_workgroup_size,
        iree_any_bit_set(
            workload->flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE),
        stream));
  }
  if (iree_any_bit_set(workload->flags,
                       LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("workgroup_count")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_dimension3_json(
        workload->workgroup_count.x, workload->workgroup_count.y,
        workload->workgroup_count.z, workload->dispatch_workgroup_count,
        iree_any_bit_set(
            workload->flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT),
        stream));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("cluster_size")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_dimension3_json(
        workload->workgroup_cluster_size.x, workload->workgroup_cluster_size.y,
        workload->workgroup_cluster_size.z,
        workload->flat_workgroup_cluster_size,
        iree_any_bit_set(
            workload->flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_CLUSTER_SIZE),
        stream));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("dispatch_workitem_count"),
        workload->dispatch_workitem_count));
  }
  return loom_json_object_end(&object);
}

iree_status_t loom_target_compile_report_json_write_nonzero_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t key, uint64_t value) {
  if (value == 0) return iree_ok_status();
  return loom_json_object_write_uint64_field(object, key, value);
}

static iree_status_t
loom_target_compile_report_json_write_scaled_nonzero_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t key, uint64_t value,
    uint64_t scale) {
  uint64_t scaled_value = 0;
  if (!iree_checked_mul_u64(value, scale, &scaled_value)) {
    return iree_ok_status();
  }
  return loom_target_compile_report_json_write_nonzero_u64_field(object, key,
                                                                 scaled_value);
}

iree_status_t loom_target_compile_report_format_memory_economics_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    uint64_t scale, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  uint64_t read_byte_count = 0;
  const bool has_read_byte_count = iree_checked_mul_u64(
      mix->memory_read_byte_count, scale, &read_byte_count);
  if (has_read_byte_count) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("read_bytes"), read_byte_count));
  }
  uint64_t write_byte_count = 0;
  const bool has_write_byte_count = iree_checked_mul_u64(
      mix->memory_write_byte_count, scale, &write_byte_count);
  if (has_write_byte_count) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("write_bytes"), write_byte_count));
  }
  uint64_t total_byte_count = 0;
  if (has_read_byte_count && has_write_byte_count &&
      iree_checked_add_u64(read_byte_count, write_byte_count,
                           &total_byte_count)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("total_bytes"), total_byte_count));
  }

#define LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(field) \
  do {                                                                       \
    IREE_RETURN_IF_ERROR(                                                    \
        loom_target_compile_report_json_write_scaled_nonzero_u64_field(      \
            &object, IREE_SV(#field), mix->field, scale));                   \
  } while (0)
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      global_load_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      global_store_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      buffer_load_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      buffer_store_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      flat_memory_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      local_memory_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      scalar_memory_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      private_memory_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD(
      generic_memory_count);
#undef LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_COUNT_FIELD
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_scaled_nonzero_u64_field(
          &object, IREE_SV("read_unknown_width_count"),
          mix->memory_read_unknown_width_count, scale));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_scaled_nonzero_u64_field(
          &object, IREE_SV("write_unknown_width_count"),
          mix->memory_write_unknown_width_count, scale));

  uint64_t byte_count = 0;
#define LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(field) \
  do {                                                                 \
    if (iree_checked_mul_u64(mix->field, scale, &byte_count)) {        \
      IREE_RETURN_IF_ERROR(                                            \
          loom_target_compile_report_json_write_nonzero_u64_field(     \
              &object, IREE_SV(#field), byte_count));                  \
    }                                                                  \
  } while (0)
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      global_load_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      global_store_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      buffer_load_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      buffer_store_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(flat_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      flat_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      local_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      local_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      scalar_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      scalar_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      private_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      private_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      unclassified_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD(
      unclassified_write_byte_count);
#undef LOOM_TARGET_COMPILE_REPORT_WRITE_MEMORY_ECONOMICS_FIELD
  return loom_json_object_end(&object);
}

iree_status_t loom_target_compile_report_format_operation_economics_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    uint64_t scale, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
#define LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(field) \
  do {                                                                    \
    IREE_RETURN_IF_ERROR(                                                 \
        loom_target_compile_report_json_write_scaled_nonzero_u64_field(   \
            &object, IREE_SV(#field), mix->field, scale));                \
  } while (0)
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(scalar_alu_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(vector_alu_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(matrix_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(mfma_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(smfmac_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(wmma_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(swmmac_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(dot_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(atomic_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(branch_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(barrier_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(control_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(conversion_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(cache_count);
  LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD(
      register_move_count);
#undef LOOM_TARGET_COMPILE_REPORT_WRITE_OPERATION_ECONOMICS_FIELD
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_economics_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    const loom_target_compile_report_workload_t* workload,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (loom_target_compile_report_economics_has_operations(mix)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("operations")));
    loom_json_object_writer_t operations;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &operations));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&operations, IREE_SV("per_workitem")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_operation_economics_json(mix, 1,
                                                                   stream));
    if (iree_any_bit_set(
            workload->flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&operations, IREE_SV("dispatch")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_operation_economics_json(
              mix, workload->dispatch_workitem_count, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&operations));
  }
  if (loom_target_compile_report_economics_has_memory(mix)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("memory")));
    loom_json_object_writer_t memory;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &memory));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&memory, IREE_SV("per_workitem_issued")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_memory_economics_json(mix, 1,
                                                                stream));
    if (iree_any_bit_set(
            workload->flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&memory, IREE_SV("dispatch_issued")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_memory_economics_json(
              mix, workload->dispatch_workitem_count, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&memory));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_target_insertion_summary_json(
    const loom_target_compile_report_target_insertion_summary_t* summary,
    iree_host_size_t row_count, loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("static_packet_count"), summary->static_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("exact_dynamic_packet_count"),
      summary->exact_dynamic_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("unknown_dynamic_packet_count"),
      summary->unknown_dynamic_packet_count));
  if (summary->unknown_dynamic_packet_count == 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_packet_count"),
        summary->dynamic_packet_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(
        object, IREE_SV("dynamic_packet_count")));
  }
  return loom_json_object_write_host_size_field(object, IREE_SV("row_count"),
                                                row_count);
}

static iree_status_t
loom_target_compile_report_format_target_insertion_row_json(
    const loom_target_compile_report_target_insertion_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      loom_target_compile_report_target_insertion_kind_name(
          row->insertion_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("kind_id"), (uint32_t)row->insertion_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("packet_key"), row->packet_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("block"), row->block_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("block_index"), row->block_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("node_index"), row->node_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("scheduled_ordinal"), row->scheduled_ordinal));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("boundary_operation"),
          row->boundary_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("boundary_descriptor_key"),
          row->boundary_descriptor_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("static_packet_count"), row->static_packet_count));
  if (iree_any_bit_set(
          row->flags,
          LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_FLAG_DYNAMIC_PACKET_COUNT)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("dynamic_packet_count"), row->dynamic_packet_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(
        &object, IREE_SV("dynamic_packet_count")));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_target_insertions_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_target_insertion_summary_json(
          &report->target_insertion_summary,
          report->target_insertion_rows.count, &object));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->target_insertion_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_target_insertion_row_t* rows =
          (const loom_target_compile_report_target_insertion_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_target_insertion_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_entry_json(
    const loom_target_compile_report_entry_t* row, iree_host_size_t row_index,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_function"), row->source_function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_bundle"), row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_snapshot"), row->target_snapshot_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_export"), row->target_export_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_export_symbol"), row->target_export_symbol));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_config"), row->target_config_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("detail_flags"), row->detail_flags));
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("planning")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_low_planning_json(
        &row->low_planning, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("schedule_node_count"), row->schedule_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("scheduled_node_count"), row->scheduled_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("schedule_dependency_count"),
      row->schedule_dependency_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("schedule_resource_use_count"),
      row->schedule_resource_use_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("schedule_hazard_gap_count"),
      row->schedule_hazard_gap_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("schedule_model_summary_count"),
      row->schedule_model_summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("register_pressure_summary_count"),
      row->register_pressure_summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("register_pressure_peak_live_units"),
      row->register_pressure_peak_live_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_assignment_count"),
      row->allocation_assignment_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_spill_count"), row->allocation_spill_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_spill_plan_count"),
      row->allocation_spill_plan_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_coalesced_copy_count"),
      row->allocation_coalesced_copy_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_copy_count"),
      row->allocation_materialized_copy_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_spill_storage_count"),
      row->allocation_materialized_spill_storage_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_spill_storage_bytes"),
      row->allocation_materialized_spill_storage_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_spill_store_count"),
      row->allocation_materialized_spill_store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_spill_store_bytes"),
      row->allocation_materialized_spill_store_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_reload_count"),
      row->allocation_materialized_reload_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_materialized_reload_bytes"),
      row->allocation_materialized_reload_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_storage_lease_count"),
      row->allocation_storage_lease_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_storage_lease_instance_count"),
      row->allocation_storage_lease_instance_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("allocation_storage_release_action_count"),
      row->allocation_storage_release_action_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("move_causes")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_cause_counts_json(
      row->move_causes, mode, stream));
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("wait_plan")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_wait_plan_json(
        &row->wait_plan, stream));
  }
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("workload")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_workload_json(
        &row->workload, stream));
  }
  if (iree_any_bit_set(
          row->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("target_insertions")));
    loom_json_object_writer_t target_insertions;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &target_insertions));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_insertion_summary_json(
            &row->target_insertion_summary, row->target_insertion_row_count,
            &target_insertions));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&target_insertions));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("instruction_count"), row->emitted_instruction_count));
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION_BREAKDOWN)) {
    const loom_target_compile_report_emission_breakdown_t* breakdown =
        &row->emission_breakdown;
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("body_instruction_count"),
        breakdown->body_instruction_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("entry_instruction_count"),
        breakdown->entry_instruction_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("coissued_instruction_count"),
        breakdown->coissued_instruction_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("coissued_component_count"),
        breakdown->coissued_component_count));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("code_byte_count"), row->emitted_code_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("code_storage_byte_count"),
      row->emitted_code_storage_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_memory_bytes"), row->private_memory_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_memory_bytes"), row->local_memory_bytes));
  if (row->bank_service_summary.modeled_packet_count != 0 ||
      row->subgroup_access_summary.modeled_packet_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("source_low_memory")));
    loom_json_object_writer_t source_low_memory;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &source_low_memory));
    if (row->bank_service_summary.modeled_packet_count != 0) {
      IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
          &source_low_memory, IREE_SV("bank_service")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_bank_service_summary_json(
              &row->bank_service_summary, stream));
    }
    if (row->subgroup_access_summary.modeled_packet_count != 0) {
      IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
          &source_low_memory, IREE_SV("subgroup_access")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_subgroup_access_summary_json(
              &row->subgroup_access_summary, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&source_low_memory));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("pressure_row_count"), row->pressure_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("pressure_origin_row_count"),
      row->pressure_origin_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("schedule_band_row_count"),
      row->schedule_band_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("schedule_band_summary_row_count"),
      row->schedule_band_summary_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("spill_row_count"), row->spill_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("allocation_high_water_row_count"),
      row->allocation_high_water_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("wait_counter_row_count"), row->wait_counter_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("wait_reason_summary_row_count"),
      row->wait_reason_summary_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("wait_action_row_count"), row->wait_action_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("target_capability_row_count"),
      row->target_capability_row_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("target_insertion_row_count"),
      row->target_insertion_row_count));
  if (iree_any_bit_set(row->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("target_resources")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_resources_json(
            &row->target_resources, stream));
  }
  if (iree_any_bit_set(
          row->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("static_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &row->static_instruction_mix, stream));
  }
  if (iree_any_bit_set(
          row->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("dynamic_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &row->dynamic_instruction_mix, stream));
  }
  if (loom_target_compile_report_has_economics(
          row->detail_flags, &row->dynamic_instruction_mix, &row->workload)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("economics")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_economics_json(
        &row->dynamic_instruction_mix, &row->workload, stream));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_entries_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->entry_rows.count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->entry_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_entry_t* rows =
        (const loom_target_compile_report_entry_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_format_entry_json(
          &rows[i], row_index, mode, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_diagnostics_json(
    const loom_target_compile_report_format_options_t* options,
    loom_output_stream_t* stream) {
  return loom_json_write_value_list_array(options->diagnostics.json_objects,
                                          stream);
}

static iree_status_t loom_target_compile_report_format_config_binding_row_json(
    const loom_target_compile_report_config_binding_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("key"), row->key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("value"), row->value));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_config_bindings_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->config_binding_rows.count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->config_binding_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_config_binding_row_t* rows =
        (const loom_target_compile_report_config_binding_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_config_binding_row_json(
              &rows[i], row_index, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

iree_status_t loom_target_compile_report_format_json(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    loom_output_stream_t* stream) {
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"), IREE_SV("loom.compile_report")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("schema_version"),
      LOOM_TARGET_COMPILE_REPORT_SCHEMA_VERSION));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("mode"),
      loom_target_compile_report_format_mode_name(options->mode)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("artifact_kind"),
      loom_target_compile_report_artifact_kind_name(report->artifact_kind)));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("status")));
  IREE_RETURN_IF_ERROR(loom_json_write_status_object(
      stream, report->status_code, iree_string_view_empty()));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("detail_flags"), report->detail_flags));
  if (report->config_binding_rows.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("config_bindings")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_config_bindings_json(report, stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("module"), report->module_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), report->function_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("backend"), report->backend_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_family"), report->target_family_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_key"), report->target_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_bundle"), report->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_snapshot"), report->target_snapshot_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_export"), report->target_export_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_export_symbol"),
          report->target_export_symbol));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_config"), report->target_config_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("lowered"), report->lowered_symbol));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("artifact_format"), report->artifact_format));

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("artifact_size"), report->artifact_size));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("entries")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_entries_json(
      report, options->mode, stream));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("planning")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_low_planning_json(
        &report->low_planning, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("schedule")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("workload")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_workload_json(
        &report->workload, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("static_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &report->static_instruction_mix, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("dynamic_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &report->dynamic_instruction_mix, stream));
  }
  if (loom_target_compile_report_has_report_economics(report)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("economics")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_report_economics_json(report,
                                                                stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("allocation")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("move_causes")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_causes_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("wait_plan")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_wait_plan_json(
        &report->wait_plan, stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("wait_counter_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_counter_rows_json(
            report, options->mode, stream));
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("wait_reason_summary_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_reason_summary_rows_json(
            report, options->mode, stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("wait_action_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_action_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("target_insertions")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_insertions_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("target_capability_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_capability_rows_json(report,
                                                                      stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("emission")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_emission_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("memory")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_memory_json(report, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("target_resources")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_resources_json(
            &report->target_resources, stream));
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_json_planning_details(
      report, options->mode, &object, stream));
  if (options->diagnostics.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("diagnostic_count"), options->diagnostics.count));
    if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("diagnostics")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_diagnostics_json(options, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_json_lowering_details(
      report, options->mode, &object, stream));
  return loom_json_object_end(&object);
}
