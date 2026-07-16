// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/scalar_type.h"
#include "loom/target/math_policy.h"
#include "loom/target/reporting/compile_report_format.h"
#include "loom/target/reporting/compile_report_planning_format.h"
#include "loom/target/reporting/compile_report_schema.h"
#include "loom/util/json.h"

static iree_status_t
loom_target_compile_report_json_write_optional_string_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_string_field(object, name, value);
}

static iree_status_t loom_target_compile_report_json_write_optional_u16_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    uint16_t value) {
  if (value == UINT16_MAX) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint32_field(object, name, value);
}

static iree_status_t loom_target_compile_report_json_write_optional_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    uint64_t value) {
  if (value == UINT64_MAX) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_uint64_field(object, name, value);
}

static iree_status_t loom_target_compile_report_json_write_optional_u32_field(
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

static iree_status_t loom_target_compile_report_format_instruction_mix_json(
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
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("dispatch_workitem_count"),
        workload->dispatch_workitem_count));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_json_write_nonzero_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t key, uint64_t value) {
  if (value == 0) return iree_ok_status();
  return loom_json_object_write_uint64_field(object, key, value);
}

static iree_status_t
loom_target_compile_report_json_write_scaled_nonzero_u64_field(
    loom_json_object_writer_t* object, iree_string_view_t key, uint64_t value,
    uint64_t scale) {
  uint64_t scaled_value = 0;
  if (!loom_target_compile_report_checked_mul_u64(value, scale,
                                                  &scaled_value)) {
    return iree_ok_status();
  }
  return loom_target_compile_report_json_write_nonzero_u64_field(object, key,
                                                                 scaled_value);
}

static iree_status_t loom_target_compile_report_format_memory_economics_json(
    const loom_target_compile_report_static_instruction_mix_t* mix,
    uint64_t scale, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  uint64_t read_byte_count = 0;
  const bool has_read_byte_count = loom_target_compile_report_checked_mul_u64(
      mix->memory_read_byte_count, scale, &read_byte_count);
  if (has_read_byte_count) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("read_bytes"), read_byte_count));
  }
  uint64_t write_byte_count = 0;
  const bool has_write_byte_count = loom_target_compile_report_checked_mul_u64(
      mix->memory_write_byte_count, scale, &write_byte_count);
  if (has_write_byte_count) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("write_bytes"), write_byte_count));
  }
  uint64_t total_byte_count = 0;
  if (has_read_byte_count && has_write_byte_count &&
      loom_target_compile_report_checked_add_u64(
          read_byte_count, write_byte_count, &total_byte_count)) {
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
    if (loom_target_compile_report_checked_mul_u64(mix->field, scale,  \
                                                   &byte_count)) {     \
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

static iree_status_t loom_target_compile_report_format_operation_economics_json(
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
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("instruction_count"), row->emitted_instruction_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("code_byte_count"), row->emitted_code_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("code_storage_byte_count"),
      row->emitted_code_storage_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("private_memory_bytes"), row->private_memory_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("local_memory_bytes"), row->local_memory_bytes));
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

static iree_status_t loom_target_compile_report_format_pressure_row_json(
    const loom_target_compile_report_pressure_row_t* row,
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
          &object, IREE_SV("register_class"), row->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("type_kind"), row->type_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_type"), row->element_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element"),
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("peak_live_units"), row->peak_live_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("peak_live_values"), row->peak_live_values));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("peak_point"), row->peak_point));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("peak_block"), row->peak_block_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("peak_operation"), row->peak_operation_name));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pressure_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->pressure_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->pressure_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_pressure_row_t* rows =
          (const loom_target_compile_report_pressure_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_pressure_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pressure_origin_row_json(
    const loom_target_compile_report_pressure_origin_row_t* row,
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
          &object, IREE_SV("register_class"), row->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("type_kind"), row->type_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_type"), row->element_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element"),
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("peak_point"), row->peak_point));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("peak_block"), row->peak_block_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("peak_operation"), row->peak_operation_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("origin_kind"), row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("origin"),
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("semantic_tag"), row->semantic_tag));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("sample_value"), row->sample_value_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("live_units"), row->live_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("live_values"), row->live_values));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_pressure_origin_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->pressure_origin_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->pressure_origin_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_pressure_origin_row_t* rows =
          (const loom_target_compile_report_pressure_origin_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_pressure_origin_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_schedule_band_row_json(
    const loom_target_compile_report_schedule_band_row_t* row,
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
          &object, IREE_SV("block"), row->block_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("block_index"), row->block_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("first_packet_index"), row->first_packet_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("first_scheduled_ordinal"),
      row->first_scheduled_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("node_count"), row->node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("origin_kind"), row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("origin"),
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("semantic_tag"), row->semantic_tag));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("sample_value"), row->sample_value_name));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("static_instruction_mix")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
      &row->static_instruction_mix, stream));
  if (iree_all_bits_set(
          row->flags,
          LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("dynamic_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &row->dynamic_instruction_mix, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("result_value_count"), row->result_value_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("result_unit_count"), row->result_unit_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_schedule_band_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->schedule_band_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->schedule_band_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_schedule_band_row_t* rows =
          (const loom_target_compile_report_schedule_band_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_schedule_band_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_schedule_band_summary_row_json(
    const loom_target_compile_report_schedule_band_summary_row_t* row,
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
          &object, IREE_SV("block"), row->block_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("block_index"), row->block_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("first_packet_index"), row->first_packet_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("band_count"), row->band_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("node_count"), row->node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("max_band_node_count"), row->max_band_node_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("origin_kind"), row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("origin"),
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("semantic_tag"), row->semantic_tag));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("sample_value"), row->sample_value_name));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("static_instruction_mix")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
      &row->static_instruction_mix, stream));
  if (iree_all_bits_set(
          row->flags,
          LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("dynamic_instruction_mix")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_instruction_mix_json(
        &row->dynamic_instruction_mix, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("result_value_count"), row->result_value_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("result_unit_count"), row->result_unit_count));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_schedule_band_summary_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  (void)mode;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->schedule_band_summary_rows.count));
  if (report->schedule_band_summary_rows.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->schedule_band_summary_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_schedule_band_summary_row_t* rows =
          (const loom_target_compile_report_schedule_band_summary_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_schedule_band_summary_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_spill_row_json(
    const loom_target_compile_report_spill_row_t* row,
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
      loom_target_compile_report_spill_row_kind_name(row->kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("value"), row->value_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("register_class"), row->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("type_kind"), row->type_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_type"), row->element_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element"),
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("origin_kind"), row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("origin"),
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("semantic_tag"), row->semantic_tag));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("assignment_index"), row->assignment_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("slot_index"), row->slot_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("slot_space"), row->slot_space));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("byte_size"), row->byte_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("byte_alignment"), row->byte_alignment));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("store_count"), row->store_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("store_bytes"), row->store_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("reload_count"), row->reload_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("reload_bytes"), row->reload_bytes));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_spill_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->spill_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec = report->spill_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_spill_row_t* rows =
          (const loom_target_compile_report_spill_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(loom_target_compile_report_format_spill_row_json(
            &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_allocation_failure_row_json(
    const loom_target_compile_report_allocation_failure_row_t* row,
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
          &object, IREE_SV("value"), row->value_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("register_class"), row->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("type_kind"), row->type_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_type"), row->element_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element"),
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("failure_code"), row->failure_code));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("blocking_kind"),
      loom_target_compile_report_allocation_failure_blocking_kind_name(
          row->blocking_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_block"), row->origin_block_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("start_point"), row->start_point));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("end_point"), row->end_point));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("required_unit_count"), row->required_unit_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("budget_units"), row->budget_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("peak_live_units"), row->peak_live_units));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("location_kind"), row->location_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("location_base"), row->location_base));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("location_count"), row->location_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("conflict_assignment_index"),
      row->conflict_assignment_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("conflict_value"), row->conflict_value_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("conflict_start_point"), row->conflict_start_point));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("conflict_end_point"), row->conflict_end_point));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("conflict_location_kind"),
          row->conflict_location_kind));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u32_field(
      &object, IREE_SV("conflict_location_base"), row->conflict_location_base));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("conflict_location_count"),
      row->conflict_location_count));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_allocation_failure_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->allocation_failure_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->allocation_failure_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_allocation_failure_row_t* rows =
          (const loom_target_compile_report_allocation_failure_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_allocation_failure_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_allocation_high_water_row_json(
    const loom_target_compile_report_allocation_high_water_row_t* row,
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
          &object, IREE_SV("value"), row->value_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("register_class"), row->register_class));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("type_kind"), row->type_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"),
      loom_target_compile_report_type_kind_name(row->type_kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_type"), row->element_type));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element"),
      loom_target_compile_report_scalar_type_name(row->element_type)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("assignment_index"), row->assignment_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("origin_kind"), row->origin_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("origin"),
      loom_target_compile_report_pressure_origin_kind_name(row->origin_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("origin_operation"), row->origin_operation_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("semantic_tag"), row->semantic_tag));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("start_point"), row->start_point));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("end_point"), row->end_point));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("required_unit_count"), row->required_unit_count));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("location_kind"), row->location_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("location_base"), row->location_base));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("location_count"), row->location_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("high_water_units"), row->high_water_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("lower_free_unit_count"), row->lower_free_unit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("lower_free_run_count"), row->lower_free_run_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("lower_largest_free_run_unit_count"),
      row->lower_largest_free_run_unit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("lower_pressure_releasable_free_unit_count"),
      row->lower_pressure_releasable_free_unit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("lower_pressure_releasable_free_run_count"),
      row->lower_pressure_releasable_free_run_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("lower_pressure_releasable_largest_free_run_unit_count"),
      row->lower_pressure_releasable_largest_free_run_unit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("active_assignment_blocker_count"),
      row->active_assignment_blocker_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("active_assignment_blocker_units"),
      row->active_assignment_blocker_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("active_storage_lease_blocker_count"),
      row->active_storage_lease_blocker_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("active_storage_lease_blocker_units"),
      row->active_storage_lease_blocker_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("active_pressure_storage_lease_blocker_count"),
      row->active_pressure_storage_lease_blocker_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("active_pressure_storage_lease_blocker_units"),
      row->active_pressure_storage_lease_blocker_units));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("active_fallback_storage_lease_blocker_count"),
      row->active_fallback_storage_lease_blocker_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("active_fallback_storage_lease_blocker_units"),
      row->active_fallback_storage_lease_blocker_units));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_allocation_high_water_rows_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->allocation_high_water_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->allocation_high_water_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_allocation_high_water_row_t* rows =
          (const loom_target_compile_report_allocation_high_water_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_allocation_high_water_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_diagnostics_json(
    const loom_target_compile_report_format_options_t* options,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write(stream, options->diagnostic_json_objects));
  return loom_output_stream_write_cstring(stream, "]");
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

static iree_status_t
loom_target_compile_report_format_source_low_target_row_json(
    const loom_target_compile_report_source_low_target_row_t* row,
    iree_host_size_t row_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), row_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("function"), row->function_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("target_source"),
      loom_target_selection_source_name(row->target_source)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_symbol"), row->target_symbol_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_bundle"), row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_snapshot"), row->target_snapshot_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_config"), row->target_config_name));
  if (row->target_subgroup_size != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("target_subgroup_size"), row->target_subgroup_size));
  }
  if (row->target_source == LOOM_TARGET_SELECTION_SOURCE_INVOCATION) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("candidate_target_count"),
        row->candidate_target_count));
    if (row->candidate_target_count != 0) {
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_json_write_optional_string_field(
              &object, IREE_SV("candidate_target_symbol"),
              row->candidate_target_symbol_name));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_json_write_optional_string_field(
              &object, IREE_SV("candidate_target_bundle"),
              row->candidate_target_bundle_name));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_json_write_optional_string_field(
              &object, IREE_SV("candidate_target_snapshot"),
              row->candidate_target_snapshot_name));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_json_write_optional_string_field(
              &object, IREE_SV("candidate_target_config"),
              row->candidate_target_config_name));
      if (row->candidate_target_subgroup_size != 0) {
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
            &object, IREE_SV("candidate_target_subgroup_size"),
            row->candidate_target_subgroup_size));
      }
    }
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_target_rows_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->source_low_target_rows.count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_target_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_target_row_t* rows =
        (const loom_target_compile_report_source_low_target_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_source_low_target_row_json(
              &rows[i], row_index, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_selection_summary_row_json(
    const loom_target_compile_report_source_low_selection_summary_t* row,
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
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("selection"),
      loom_target_compile_report_source_low_selection_name(
          row->selection_kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("plan_key"), row->plan_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("descriptor_key"), row->descriptor_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("descriptor_semantic_tag"),
          row->descriptor_semantic_tag));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("selected_op_count"), row->selected_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("emitted_low_op_count"), row->emitted_low_op_count));
  const bool has_dynamic_delta =
      row->unknown_dynamic_op_count != 0 ||
      row->dynamic_selected_op_count != row->selected_op_count ||
      row->dynamic_emitted_low_op_count != row->emitted_low_op_count;
  if (has_dynamic_delta) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("exact_dynamic_op_count"),
        row->exact_dynamic_op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("unknown_dynamic_op_count"),
        row->unknown_dynamic_op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("dynamic_selected_op_count"),
        row->dynamic_selected_op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("dynamic_emitted_low_op_count"),
        row->dynamic_emitted_low_op_count));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_selection_summaries_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->source_low_selection_summaries.count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_selection_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_selection_summary_t* rows =
        (const loom_target_compile_report_source_low_selection_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_source_low_selection_summary_row_json(
              &rows[i], row_index, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_source_low_row_json(
    const loom_target_compile_report_source_low_row_t* row,
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
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("selection"),
      loom_target_compile_report_source_low_selection_name(
          row->selection_kind)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("rule_set_index"), row->rule_set_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("rule_index"), row->rule_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u64_field(
      &object, IREE_SV("plan_id"), row->plan_id));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("plan_key"), row->plan_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("descriptor_key"), row->descriptor_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("descriptor_semantic_tag"),
          row->descriptor_semantic_tag));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("emitted_low_op_count"), row->emitted_low_op_count));
  if (row->execution_count_plus_one !=
          LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_EXECUTION_COUNT_PLUS_ONE_UNKNOWN &&
      row->execution_count_plus_one != 2) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_uint64_field(&object, IREE_SV("execution_count"),
                                            row->execution_count_plus_one - 1));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_transform_row_json(
    const loom_target_compile_report_source_low_transform_row_t* row,
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
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("transform"), row->transform_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("outcome"), row->outcome));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("reason"), row->reason));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("candidate_value_count"), row->candidate_value_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("selected_value_count"), row->selected_value_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("removed_loop_carried_value_count"),
      row->removed_loop_carried_value_count));
  if (row->removed_loop_carried_payload_register_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("removed_loop_carried_payload_register_count"),
        row->removed_loop_carried_payload_register_count));
  }
  if (row->block_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("block_count"), row->block_count));
  }
  if (row->block_count != 0 || row->row_count != 0 || row->column_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("row_count"), row->row_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("column_count"), row->column_count));
  }
  if (row->workgroup_memory_byte_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("workgroup_memory_byte_count"),
        row->workgroup_memory_byte_count));
  }
  if (row->inserted_load_op_count != 0 || row->inserted_store_op_count != 0 ||
      row->inserted_barrier_op_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("inserted_load_op_count"),
        row->inserted_load_op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("inserted_store_op_count"),
        row->inserted_store_op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("inserted_barrier_op_count"),
        row->inserted_barrier_op_count));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_transforms_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->source_low_transform_rows.count));
  if (mode != LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    return loom_json_object_end(&object);
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("rows")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_transform_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_transform_row_t* rows =
        (const loom_target_compile_report_source_low_transform_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_source_low_transform_row_json(
              &rows[i], row_index, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_memory_interval_json(
    const loom_target_compile_report_memory_interval_t* interval,
    loom_json_object_writer_t* object) {
  if (!loom_target_compile_report_memory_interval_has_range(interval)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("source_interval")));
  loom_json_object_writer_t interval_object;
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin(object->stream, &interval_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &interval_object, IREE_SV("begin_min_bytes"), interval->begin_min_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &interval_object, IREE_SV("begin_max_bytes"), interval->begin_max_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &interval_object, IREE_SV("end_min_bytes"), interval->end_min_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &interval_object, IREE_SV("end_max_bytes"), interval->end_max_bytes));
  if (iree_all_bits_set(
          interval->flags,
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &interval_object, IREE_SV("exact_length_bytes"),
        interval->exact_length_bytes));
  }
  return loom_json_object_end(&interval_object);
}

static iree_status_t
loom_target_compile_report_format_memory_interval_summary_json(
    const char* field_name,
    const loom_target_compile_report_memory_interval_summary_t* summary,
    loom_json_object_writer_t* object) {
  if (summary->packet_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, iree_make_cstring_view(field_name)));
  loom_json_object_writer_t summary_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &summary_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &summary_object, IREE_SV("packet_count"), summary->packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &summary_object, IREE_SV("begin_min_bytes"),
      summary->envelope_begin_min_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &summary_object, IREE_SV("end_max_bytes"),
      summary->envelope_end_max_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &summary_object, IREE_SV("byte_count"), summary->envelope_byte_count));
  if (summary->exact_static_packet_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &summary_object, IREE_SV("exact_static_packet_count"),
        summary->exact_static_packet_count));
  }
  if (summary->exact_symbolic_packet_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &summary_object, IREE_SV("exact_symbolic_packet_count"),
        summary->exact_symbolic_packet_count));
  }
  if (summary->exact_static_packet_count +
          summary->exact_symbolic_packet_count ==
      summary->packet_count) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &summary_object, IREE_SV("unique_byte_count"),
        summary->unique_byte_count));
  }
  return loom_json_object_end(&summary_object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_storage_fields_json(
    const loom_target_compile_report_string_field_t* fields,
    iree_host_size_t field_count, loom_json_object_writer_t* object) {
  const iree_host_size_t first_storage_field =
      loom_target_compile_report_first_non_empty_string_field(fields,
                                                              field_count);
  if (first_storage_field == field_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("storage")));
  loom_json_object_writer_t storage;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &storage));
  for (iree_host_size_t i = first_storage_field; i < field_count; ++i) {
    if (iree_string_view_is_empty(fields[i].value)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &storage, iree_make_cstring_view(fields[i].name), fields[i].value));
  }
  return loom_json_object_end(&storage);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_storage_json(
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_json_object_writer_t* object) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_storage_fields(row, fields);
  return loom_target_compile_report_format_source_low_memory_storage_fields_json(
      fields, field_count, object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_strategy_storage_json(
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        summary,
    loom_json_object_writer_t* object) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_strategy_storage_fields(
          summary, fields);
  return loom_target_compile_report_format_source_low_memory_storage_fields_json(
      fields, field_count, object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_argument_packet_storage_json(
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        summary,
    loom_json_object_writer_t* object) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_argument_packet_storage_fields(
          summary, fields);
  return loom_target_compile_report_format_source_low_memory_storage_fields_json(
      fields, field_count, object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_row_json(
    const loom_target_compile_report_source_low_memory_row_t* row,
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
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("source_root"), row->source_root_name));
  if (row->source_root_argument_index != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("source_root_argument_index"),
        row->source_root_argument_index));
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("memory_space"), row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("operation"), row->operation_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("packet"), row->packet_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("strategy"), row->strategy_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("address_form"), row->address_form));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("dynamic_term_kind"), row->dynamic_term_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("fallback_reason"), row->fallback_reason));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &object, IREE_SV("static_offset_bytes"), row->static_offset_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("element_bytes"), row->element_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("vector_lanes"), row->vector_lane_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("issued_read_byte_count"), row->issued_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("issued_write_byte_count"),
      row->issued_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("issued_read_unknown_width_count"),
      row->issued_read_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("issued_write_unknown_width_count"),
      row->issued_write_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("dynamic_stride_bytes"), row->dynamic_stride_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("vector_lane_stride_bytes"),
      row->vector_lane_stride_bytes));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("bank_stride_words"), row->bank_stride_words));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("bank_conflict_degree"), row->bank_conflict_degree));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("bank_conflict_kind"), row->bank_conflict_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_storage_json(
          row, &object));
  if (row->execution_count_plus_one !=
          LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_EXECUTION_COUNT_PLUS_ONE_UNKNOWN &&
      row->execution_count_plus_one != 2) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_uint64_field(&object, IREE_SV("execution_count"),
                                            row->execution_count_plus_one - 1));
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_memory_interval_json(
      &row->source_interval, &object));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_dispatch_source_json(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_json_object_writer_t* object) {
  loom_target_compile_report_dispatch_memory_bytes_t dispatch_source = {0};
  if (!loom_target_compile_report_source_low_memory_dispatch_source_bytes(
          summary, workload, &dispatch_source)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("dispatch_source")));
  loom_json_object_writer_t dispatch_source_object;
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin(object->stream, &dispatch_source_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dispatch_source_object, IREE_SV("read_bytes"),
      dispatch_source.read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dispatch_source_object, IREE_SV("write_bytes"),
      dispatch_source.write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dispatch_source_object, IREE_SV("total_bytes"),
      dispatch_source.total_byte_count));
  return loom_json_object_end(&dispatch_source_object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_dispatch_issued_json(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_json_object_writer_t* object) {
  if (!loom_target_compile_report_source_low_memory_can_dispatch_scale(
          summary, workload)) {
    return iree_ok_status();
  }
  const uint64_t issued_read_unknown_width_count =
      loom_target_compile_report_source_low_memory_has_dynamic_evidence(summary)
          ? summary->dynamic_issued_read_unknown_width_count
          : summary->issued_read_unknown_width_count;
  const uint64_t issued_write_unknown_width_count =
      loom_target_compile_report_source_low_memory_has_dynamic_evidence(summary)
          ? summary->dynamic_issued_write_unknown_width_count
          : summary->issued_write_unknown_width_count;
  loom_target_compile_report_dispatch_memory_bytes_t dispatch_issued = {0};
  const bool has_dispatch_issued =
      loom_target_compile_report_source_low_memory_dispatch_issued_bytes(
          summary, workload, &dispatch_issued);
  const bool has_unknown_width_count = issued_read_unknown_width_count != 0 ||
                                       issued_write_unknown_width_count != 0;
  if (!has_dispatch_issued && !has_unknown_width_count) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("dispatch_issued")));
  loom_json_object_writer_t dispatch_issued_object;
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin(object->stream, &dispatch_issued_object));
  if (has_dispatch_issued) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &dispatch_issued_object, IREE_SV("read_bytes"),
        dispatch_issued.read_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &dispatch_issued_object, IREE_SV("write_bytes"),
        dispatch_issued.write_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &dispatch_issued_object, IREE_SV("total_bytes"),
        dispatch_issued.total_byte_count));
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_nonzero_u64_field(
      &dispatch_issued_object, IREE_SV("read_unknown_width_count"),
      issued_read_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_nonzero_u64_field(
      &dispatch_issued_object, IREE_SV("write_unknown_width_count"),
      issued_write_unknown_width_count));
  return loom_json_object_end(&dispatch_issued_object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_summary_fields_json(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("packet_count"), summary->packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("load_packet_count"), summary->load_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("store_packet_count"), summary->store_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("scalar_packet_count"), summary->scalar_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("vector_packet_count"), summary->vector_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("source_lane_count"), summary->source_lane_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("source_byte_count"), summary->source_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("read_byte_count"), summary->read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("write_byte_count"), summary->write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("issued_read_byte_count"),
      summary->issued_read_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("issued_write_byte_count"),
      summary->issued_write_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("issued_read_unknown_width_count"),
      summary->issued_read_unknown_width_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("issued_write_unknown_width_count"),
      summary->issued_write_unknown_width_count));
  if (loom_target_compile_report_source_low_memory_should_print_dynamic(
          summary, workload)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("exact_dynamic_packet_count"),
        summary->exact_dynamic_packet_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("unknown_dynamic_packet_count"),
        summary->unknown_dynamic_packet_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_packet_count"),
        summary->dynamic_packet_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_source_byte_count"),
        summary->dynamic_source_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_read_byte_count"),
        summary->dynamic_read_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_write_byte_count"),
        summary->dynamic_write_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_issued_read_byte_count"),
        summary->dynamic_issued_read_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_issued_write_byte_count"),
        summary->dynamic_issued_write_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_issued_read_unknown_width_count"),
        summary->dynamic_issued_read_unknown_width_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        object, IREE_SV("dynamic_issued_write_unknown_width_count"),
        summary->dynamic_issued_write_unknown_width_count));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("contiguous_vector_packet_count"),
      summary->contiguous_vector_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("strided_vector_packet_count"),
      summary->strided_vector_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      object, IREE_SV("unknown_stride_vector_packet_count"),
      summary->unknown_stride_vector_packet_count));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_dispatch_source_json(
          summary, workload, object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_dispatch_issued_json(
          summary, workload, object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_memory_interval_summary_json(
          "interval_envelope", &summary->interval_envelope, object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_memory_interval_summary_json(
          "read_interval_envelope", &summary->read_interval_envelope, object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_memory_interval_summary_json(
          "write_interval_envelope", &summary->write_interval_envelope,
          object));
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_root_summary_json(
    const loom_target_compile_report_source_low_memory_root_summary_t* row,
    const loom_target_compile_report_workload_t* workload,
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
          &object, IREE_SV("source_root"), row->source_root_name));
  if (row->source_root_argument_index != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("source_root_argument_index"),
        row->source_root_argument_index));
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("memory_space"), row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          &row->summary, workload, &object));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_argument_summary_json(
    const loom_target_compile_report_source_low_memory_argument_summary_t* row,
    const loom_target_compile_report_workload_t* workload,
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
          &object, IREE_SV("source_root"), row->source_root_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_root_argument_index"),
      row->source_root_argument_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("memory_space"), row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          &row->summary, workload, &object));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_argument_packet_summary_json(
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        row,
    const loom_target_compile_report_workload_t* workload,
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
          &object, IREE_SV("source_root"), row->source_root_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_root_argument_index"),
      row->source_root_argument_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("memory_space"), row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("operation"), row->operation_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("packet"), row->packet_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("strategy"), row->strategy_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("fallback_reason"), row->fallback_reason));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_argument_packet_storage_json(
          row, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          &row->summary, workload, &object));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_strategy_summary_json(
    const loom_target_compile_report_source_low_memory_strategy_summary_t* row,
    const loom_target_compile_report_workload_t* workload,
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
          &object, IREE_SV("memory_space"), row->memory_space));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("operation"), row->operation_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("packet"), row->packet_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("strategy"), row->strategy_key));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("fallback_reason"), row->fallback_reason));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_strategy_storage_json(
          row, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          &row->summary, workload, &object));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_summary_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  const loom_target_compile_report_source_low_memory_summary_t* summary =
      &report->source_low_memory_summary;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          summary, &report->workload, &object));
  if (report->source_low_memory_root_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("root_count"),
        report->source_low_memory_root_summaries.count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("roots")));
    loom_json_array_writer_t array_0;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_0));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_root_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_root_summary_t* rows =
          (const loom_target_compile_report_source_low_memory_root_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_0));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_root_summary_json(
                &rows[i], &report->workload, row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_0));
  }
  if (report->source_low_memory_argument_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("argument_count"),
        report->source_low_memory_argument_summaries.count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("arguments")));
    loom_json_array_writer_t array_1;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_1));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_argument_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_argument_summary_t* rows =
          (const loom_target_compile_report_source_low_memory_argument_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_1));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_argument_summary_json(
                &rows[i], &report->workload, row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_1));
  }
  if (report->source_low_memory_argument_packet_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("argument_packet_count"),
        report->source_low_memory_argument_packet_summaries.count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("argument_packets")));
    loom_json_array_writer_t array_2;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_2));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_argument_packet_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
          rows =
              (const loom_target_compile_report_source_low_memory_argument_packet_summary_t*)
                  loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_2));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_argument_packet_summary_json(
                &rows[i], &report->workload, row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_2));
  }
  if (report->source_low_memory_strategy_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("strategy_count"),
        report->source_low_memory_strategy_summaries.count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("strategies")));
    loom_json_array_writer_t array_3;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_3));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_strategy_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_strategy_summary_t* rows =
          (const loom_target_compile_report_source_low_memory_strategy_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_3));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_strategy_summary_json(
                &rows[i], &report->workload, row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_3));
  }
  return loom_json_object_end(&object);
}

static bool loom_target_compile_report_has_report_economics(
    const loom_target_compile_report_t* report) {
  return loom_target_compile_report_has_economics(
             report->detail_flags, &report->dynamic_instruction_mix,
             &report->workload) ||
         report->source_low_memory_summary.packet_count != 0;
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_economics_json(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_fields_json(
          summary, workload, &object));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_report_economics_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream) {
  const bool has_dynamic_instruction_mix = iree_any_bit_set(
      report->detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX);
  const bool has_low_dynamic_operations =
      has_dynamic_instruction_mix &&
      loom_target_compile_report_economics_has_operations(
          &report->dynamic_instruction_mix);
  const bool has_low_dynamic_memory =
      has_dynamic_instruction_mix &&
      loom_target_compile_report_economics_has_memory(
          &report->dynamic_instruction_mix);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (has_low_dynamic_operations) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("operations")));
    loom_json_object_writer_t operations;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &operations));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&operations, IREE_SV("per_workitem")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_operation_economics_json(
            &report->dynamic_instruction_mix, 1, stream));
    if (iree_any_bit_set(
            report->workload.flags,
            LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&operations, IREE_SV("dispatch")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_operation_economics_json(
              &report->dynamic_instruction_mix,
              report->workload.dispatch_workitem_count, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&operations));
  }
  if (has_low_dynamic_memory ||
      report->source_low_memory_summary.packet_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("memory")));
    loom_json_object_writer_t memory;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &memory));
    if (has_low_dynamic_memory) {
      IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
          &memory, IREE_SV("per_workitem_issued")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_memory_economics_json(
              &report->dynamic_instruction_mix, 1, stream));
      if (iree_any_bit_set(
              report->workload.flags,
              LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
        IREE_RETURN_IF_ERROR(
            loom_json_object_begin_field(&memory, IREE_SV("dispatch_issued")));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_memory_economics_json(
                &report->dynamic_instruction_mix,
                report->workload.dispatch_workitem_count, stream));
      }
    }
    if (report->source_low_memory_summary.packet_count != 0) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&memory, IREE_SV("source_low")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_source_low_memory_economics_json(
              &report->source_low_memory_summary, &report->workload, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&memory));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_source_low_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("selected_op_count"),
      report->source_low_selected_op_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint64_field(&object, IREE_SV("emitted_op_count"),
                                          report->source_low_emitted_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->source_low_rows.count));
  if (report->source_low_target_rows.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("target_selections")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_target_rows_json(report,
                                                                      stream));
  }
  if (report->source_low_selection_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("selection_summaries")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_selection_summaries_json(
            report, stream));
  }
  if (report->source_low_transform_rows.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("transforms")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_transforms_json(
            report, mode, stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("memory")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_summary_json(report,
                                                                       stream));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array_0;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_0));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_row_t* rows =
          (const loom_target_compile_report_source_low_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_0));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_0));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("memory_rows")));
    loom_json_array_writer_t array_1;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_1));
    row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_memory_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_memory_row_t* rows =
          (const loom_target_compile_report_source_low_memory_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_1));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_memory_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_1));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_math_row_json(
    const loom_target_compile_report_math_row_t* row,
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
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_bundle"), row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_config"), row->target_config_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("policy"), row->policy_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("constraint_key"), row->constraint_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("math_op"),
      loom_target_math_op_name((loom_target_math_op_t)row->math_op)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("lane_domain"),
      loom_target_math_lane_domain_name(
          (loom_target_math_lane_domain_t)row->lane_domain)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("element_type"),
      loom_target_compile_report_scalar_type_name(
          (loom_scalar_type_t)row->element_type)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("action"),
      loom_target_compile_report_math_action_name(row->action)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("recipe"),
      loom_target_math_recipe_name((loom_target_math_recipe_t)row->recipe)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_fastmath_flags"), row->source_fastmath_flags));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("recipe_fastmath_flags"), row->recipe_fastmath_flags));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("created_op_count"), row->created_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("erased_op_count"), row->erased_op_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_math_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("rewritten_op_count"),
      report->math_legalization_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("rejected_op_count"),
      report->math_legalization_rejected_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("missing_policy_op_count"),
      report->math_legalization_missing_policy_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("missing_recipe_op_count"),
      report->math_legalization_missing_recipe_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->math_legalization_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->math_legalization_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_math_row_t* rows =
          (const loom_target_compile_report_math_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(loom_target_compile_report_format_math_row_json(
            &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_legalization_row_json(
    const loom_target_compile_report_legalization_row_t* row,
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
          &object, IREE_SV("source_op"), row->source_op_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_op_kind"), row->source_op_kind));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_bundle"), row->target_bundle_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("target_config"), row->target_config_name));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("legalizer"), row->legalizer_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("legalizer_strategy"),
      loom_target_compile_report_legalizer_strategy_name(
          row->legalizer_strategy)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("mode"),
      loom_target_compile_report_legalization_mode_name(row->mode)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("policy"),
      loom_target_compile_report_legalization_policy_name(row->policy)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("action"),
      loom_target_compile_report_legalization_action_name(row->action)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("legalization_outcome"),
      loom_target_compile_report_legalization_outcome_name(
          row->legalization_outcome)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("contract_outcome"),
      loom_target_compile_report_contract_outcome_name(row->contract_outcome)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("binding_index"), row->binding_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("case_index"), row->case_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("rule_set_index"), row->rule_set_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("rule_index"), row->rule_index));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_json_write_optional_u16_field(
      &object, IREE_SV("diagnostic_index"), row->diagnostic_index));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("descriptor_key"), row->descriptor_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_rejection_bits"), row->source_rejection_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("source_rejection_detail"),
      row->source_rejection_detail));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("target_rejection_bits"), row->target_rejection_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("missing_feature_bits"), row->missing_feature_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("missing_fact_bits"), row->missing_fact_bits));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("created_op_count"), row->created_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("erased_op_count"), row->erased_op_count));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_legalization_json(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("legal_op_count"),
      report->target_legalization_legal_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("rewritten_op_count"),
      report->target_legalization_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("target_rewritten_op_count"),
      report->target_legalization_target_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("reference_rewritten_op_count"),
      report->target_legalization_reference_rewritten_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("deferred_op_count"),
      report->target_legalization_deferred_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("invalid_ir_op_count"),
      report->target_legalization_invalid_ir_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unsupported_op_count"),
      report->target_legalization_unsupported_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unhandled_op_count"),
      report->target_legalization_unhandled_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("count"), report->target_legalization_rows.count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("rows")));
    loom_json_array_writer_t array;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->target_legalization_rows.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_legalization_row_t* rows =
          (const loom_target_compile_report_legalization_row_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_legalization_row_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array));
  }
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
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("pressure_rows")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pressure_rows_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("pressure_origin_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pressure_origin_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("schedule_band_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("schedule_band_summary_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_summary_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("spill_rows")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_spill_rows_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("allocation_failure_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_failure_rows_json(
            report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("allocation_high_water_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_high_water_rows_json(
            report, options->mode, stream));
  }
  if (options->diagnostic_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("diagnostic_count"), options->diagnostic_count));
    if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&object, IREE_SV("diagnostics")));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_diagnostics_json(options, stream));
    }
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("math_legalization")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_math_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("source_low")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_source_low_json(
        report, options->mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("target_legalization")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_legalization_json(
        report, options->mode, stream));
  }
  return loom_json_object_end(&object);
}
