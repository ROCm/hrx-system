// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/target/reporting/format_json.h"
#include "loom/target/reporting/schema.h"

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

iree_status_t loom_target_compile_report_format_json_planning_details(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_json_object_writer_t* root_object, loom_output_stream_t* stream) {
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(root_object, IREE_SV("pressure_rows")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pressure_rows_json(
        report, mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        root_object, IREE_SV("pressure_origin_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pressure_origin_rows_json(
            report, mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        root_object, IREE_SV("schedule_band_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_rows_json(report, mode,
                                                                  stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        root_object, IREE_SV("schedule_band_summary_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_schedule_band_summary_rows_json(
            report, mode, stream));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(root_object, IREE_SV("spill_rows")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_spill_rows_json(
        report, mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        root_object, IREE_SV("allocation_failure_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_failure_rows_json(
            report, mode, stream));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        root_object, IREE_SV("allocation_high_water_rows")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_high_water_rows_json(
            report, mode, stream));
  }
  return iree_ok_status();
}
