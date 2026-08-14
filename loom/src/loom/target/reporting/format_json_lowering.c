// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/target/math_policy.h"
#include "loom/target/reporting/format_json.h"
#include "loom/target/reporting/schema.h"

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
      loom_target_binding_source_name(row->target_source)));
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
  if (row->target_source == LOOM_TARGET_BINDING_SOURCE_SPECIALIZATION) {
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
loom_target_compile_report_format_source_low_memory_bank_service_u8_array_json(
    iree_string_view_t field_name, const uint8_t* values, uint8_t value_count,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, field_name));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (uint8_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_json_array_write_uint32_element(&array, values[i]));
  }
  return loom_json_array_end(&array);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_bank_service_u16_array_json(
    iree_string_view_t field_name, const uint16_t* values, uint8_t value_count,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, field_name));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (uint8_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_json_array_write_uint32_element(&array, values[i]));
  }
  return loom_json_array_end(&array);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_bank_service_json(
    const loom_target_compile_report_bank_service_t* bank_service,
    loom_json_object_writer_t* row_object) {
  if (iree_string_view_is_empty(bank_service->model_key)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(row_object, IREE_SV("bank_service")));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(row_object->stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("proof"), bank_service->proof));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("classification"), bank_service->classification));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("unknown_reason"), bank_service->unknown_reason));

  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("model")));
  loom_json_object_writer_t model;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object.stream, &model));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &model, IREE_SV("key"), bank_service->model_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &model, IREE_SV("revision"), bank_service->model_revision));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &model, IREE_SV("evidence"), bank_service->model_evidence));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &model, IREE_SV("request_policy"), bank_service->request_policy));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &model, IREE_SV("wave_size"), bank_service->wave_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &model, IREE_SV("bank_count"), bank_service->bank_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &model, IREE_SV("bank_word_bytes"), bank_service->bank_word_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &model, IREE_SV("packet_bank_words"), bank_service->packet_word_count));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_bank_service_u8_array_json(
          IREE_SV("phase_lane_counts"), bank_service->phase_lane_counts,
          bank_service->phase_count, &model));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&model));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("address")));
  loom_json_object_writer_t address;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object.stream, &address));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &address, IREE_SV("lane_address_proof"),
      bank_service->lane_address_proof));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &address, IREE_SV("active_lane_proof"), bank_service->active_lane_proof));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &address, IREE_SV("base_residue_proof"),
      bank_service->base_residue_proof));
  if (bank_service->base_residue_count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &address, IREE_SV("base_residue_count"),
        bank_service->base_residue_count));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&address));

  if (!iree_string_view_is_empty(bank_service->classification)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_memory_bank_service_u16_array_json(
            IREE_SV("phase_required_rounds"),
            bank_service->phase_required_rounds, bank_service->phase_count,
            &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("required_rounds"), bank_service->required_rounds));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("uncontended_rounds"),
        bank_service->uncontended_rounds));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("extra_rounds"), bank_service->extra_rounds));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("maximum_request_multiplicity"),
        bank_service->maximum_request_multiplicity));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_subgroup_access_object_json(
    const loom_target_compile_report_subgroup_access_t* access,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("proof"), access->proof));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_json_write_optional_string_field(
          &object, IREE_SV("unknown_reason"), access->unknown_reason));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("address")));
  loom_json_object_writer_t address;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &address));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &address, IREE_SV("lane_address_proof"), access->lane_address_proof));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &address, IREE_SV("active_lane_proof"), access->active_lane_proof));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &address, IREE_SV("lane_mapping"), access->lane_mapping));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &address, IREE_SV("subgroup_size"), access->subgroup_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &address, IREE_SV("per_lane_packet_bytes"),
      access->per_lane_packet_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &address, IREE_SV("linear_lane_stride_bytes"),
      access->linear_lane_byte_stride));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&address, IREE_SV("lane_terms")));
  loom_json_array_writer_t terms;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &terms));
  for (uint8_t i = 0; i < access->lane_term_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&terms));
    loom_json_object_writer_t term;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &term));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &term, IREE_SV("divisor"), access->lane_terms[i].divisor));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &term, IREE_SV("modulus"), access->lane_terms[i].modulus));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &term, IREE_SV("byte_stride"), access->lane_terms[i].byte_stride));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&term));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&terms));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&address));

  if (iree_string_view_equal(access->proof, IREE_SV("exact"))) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("geometry")));
    loom_json_object_writer_t geometry;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &geometry));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &geometry, IREE_SV("interval_coverage"), access->interval_coverage));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &geometry, IREE_SV("subgroup_requested_bytes"),
        access->subgroup_requested_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &geometry, IREE_SV("subgroup_unique_bytes"),
        access->subgroup_unique_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &geometry, IREE_SV("subgroup_span_bytes"),
        access->subgroup_span_byte_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &geometry, IREE_SV("maximum_adjacent_lane_delta_bytes"),
        access->maximum_adjacent_lane_delta_bytes));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &geometry, IREE_SV("maximum_uncovered_gap_bytes"),
        access->maximum_uncovered_byte_gap_bytes));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &geometry, IREE_SV("distinct_lane_address_count"),
        access->distinct_lane_address_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &geometry, IREE_SV("contiguous_region_count"),
        access->contiguous_region_count));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&geometry));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_subgroup_access_json(
    const loom_target_compile_report_subgroup_access_t* access,
    loom_json_object_writer_t* row_object) {
  if (iree_string_view_is_empty(access->proof)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(row_object, IREE_SV("subgroup_access")));
  return loom_target_compile_report_format_subgroup_access_object_json(
      access, row_object->stream);
}

iree_status_t loom_target_compile_report_format_bank_service_summary_json(
    const loom_target_compile_report_bank_service_summary_t* summary,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("modeled_packet_count"), summary->modeled_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("exact_packet_count"), summary->exact_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unknown_packet_count"), summary->unknown_packet_count));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("structural")));
  loom_json_object_writer_t structural;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &structural));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &structural, IREE_SV("conflict_free_packet_count"),
      summary->conflict_free_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &structural, IREE_SV("conflicted_packet_count"),
      summary->conflicted_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &structural, IREE_SV("required_round_count"),
      summary->required_round_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &structural, IREE_SV("uncontended_round_count"),
      summary->uncontended_round_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &structural, IREE_SV("extra_round_count"), summary->extra_round_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &structural, IREE_SV("maximum_request_multiplicity"),
      summary->maximum_request_multiplicity));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&structural));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("dynamic")));
  loom_json_object_writer_t dynamic;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &dynamic));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("exact_packet_count"),
      summary->exact_dynamic_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("unknown_packet_count"),
      summary->unknown_dynamic_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("packet_count"), summary->dynamic_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("required_round_count"),
      summary->dynamic_required_round_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("uncontended_round_count"),
      summary->dynamic_uncontended_round_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("extra_round_count"),
      summary->dynamic_extra_round_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&dynamic));
  return loom_json_object_end(&object);
}

iree_status_t loom_target_compile_report_format_subgroup_access_summary_json(
    const loom_target_compile_report_subgroup_access_summary_t* summary,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("modeled_packet_count"), summary->modeled_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("exact_packet_count"), summary->exact_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("unknown_packet_count"), summary->unknown_packet_count));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("structural")));
  loom_json_object_writer_t structural;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &structural));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &structural, IREE_SV("dense_packet_count"), summary->dense_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &structural, IREE_SV("gapped_packet_count"),
      summary->gapped_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &structural, IREE_SV("overlapping_packet_count"),
      summary->overlapping_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&structural));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("dynamic")));
  loom_json_object_writer_t dynamic;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &dynamic));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("exact_packet_count"),
      summary->exact_dynamic_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("unknown_packet_count"),
      summary->unknown_dynamic_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("packet_count"), summary->dynamic_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("dense_packet_count"),
      summary->dynamic_dense_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("gapped_packet_count"),
      summary->dynamic_gapped_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &dynamic, IREE_SV("overlapping_packet_count"),
      summary->dynamic_overlapping_packet_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&dynamic));
  return loom_json_object_end(&object);
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
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_storage_json(
          row, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_bank_service_json(
          &row->bank_service, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_subgroup_access_json(
          &row->subgroup_access, &object));
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
loom_target_compile_report_format_source_low_bank_service_summary_json(
    const loom_target_compile_report_source_low_bank_service_summary_t* row,
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

  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("model")));
  loom_json_object_writer_t model;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &model));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &model, IREE_SV("key"), row->model_key));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &model, IREE_SV("revision"), row->model_revision));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &model, IREE_SV("evidence"), row->model_evidence));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &model, IREE_SV("request_policy"), row->request_policy));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &model, IREE_SV("wave_size"), row->wave_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &model, IREE_SV("bank_count"), row->bank_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &model, IREE_SV("bank_word_bytes"), row->bank_word_byte_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &model, IREE_SV("packet_bank_words"), row->packet_word_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&model));

  if (row->summary.unknown_packet_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("unknown_evidence")));
    loom_json_object_writer_t unknown_evidence;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &unknown_evidence));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_json_write_optional_string_field(
            &unknown_evidence, IREE_SV("reason"), row->unknown_reason));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &unknown_evidence, IREE_SV("mixed_reasons"),
        row->has_mixed_unknown_reasons));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&unknown_evidence));
  }

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("summary")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_bank_service_summary_json(&row->summary,
                                                                  stream));
  return loom_json_object_end(&object);
}

static iree_status_t
loom_target_compile_report_format_source_low_subgroup_access_summary_json(
    const loom_target_compile_report_source_low_subgroup_access_summary_t* row,
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
      loom_json_object_begin_field(&object, IREE_SV("access")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_subgroup_access_object_json(
          &row->access, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("summary")));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_subgroup_access_summary_json(
          &row->summary, stream));
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
  if (report->bank_service_summary.modeled_packet_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("bank_service")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_bank_service_summary_json(
            &report->bank_service_summary, stream));
  }
  if (report->subgroup_access_summary.modeled_packet_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("subgroup_access")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_subgroup_access_summary_json(
            &report->subgroup_access_summary, stream));
  }
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
  if (report->source_low_bank_service_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("bank_service_group_count"),
        report->source_low_bank_service_summaries.count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("bank_service_groups")));
    loom_json_array_writer_t array_4;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_4));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_bank_service_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_bank_service_summary_t* rows =
          (const loom_target_compile_report_source_low_bank_service_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_4));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_bank_service_summary_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_4));
  }
  if (report->source_low_subgroup_access_summaries.count != 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("subgroup_access_group_count"),
        report->source_low_subgroup_access_summaries.count));
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("subgroup_access_groups")));
    loom_json_array_writer_t array_5;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array_5));
    iree_host_size_t row_index = 0;
    for (const loom_target_compile_report_vec_t* vec =
             report->source_low_subgroup_access_summaries.head;
         vec != NULL; vec = vec->next) {
      const loom_target_compile_report_source_low_subgroup_access_summary_t* rows =
          (const loom_target_compile_report_source_low_subgroup_access_summary_t*)
              loom_target_compile_report_vec_const_rows(vec);
      for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array_5));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_source_low_subgroup_access_summary_json(
                &rows[i], row_index, stream));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&array_5));
  }
  return loom_json_object_end(&object);
}

bool loom_target_compile_report_has_report_economics(
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

iree_status_t loom_target_compile_report_format_report_economics_json(
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

iree_status_t loom_target_compile_report_format_json_lowering_details(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_json_object_writer_t* root_object, loom_output_stream_t* stream) {
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        root_object, IREE_SV("math_legalization")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_math_json(report, mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(root_object, IREE_SV("source_low")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_source_low_json(
        report, mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        root_object, IREE_SV("target_legalization")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_legalization_json(
        report, mode, stream));
  }
  return iree_ok_status();
}
