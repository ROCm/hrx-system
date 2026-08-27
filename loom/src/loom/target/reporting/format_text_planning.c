// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>

#include "loom/ir/scalar_type.h"
#include "loom/target/reporting/format_text.h"
#include "loom/target/reporting/schema.h"

static iree_status_t loom_target_compile_report_append_optional_u32(
    iree_string_builder_t* builder, uint32_t value) {
  if (value == UINT32_MAX) {
    return iree_string_builder_append_string(builder, IREE_SV("-"));
  }
  return iree_string_builder_append_format(builder, "%u", value);
}

static iree_status_t loom_target_compile_report_format_target_capability_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->target_capability_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_target_capability_row_t* rows =
        (const loom_target_compile_report_target_capability_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_target_capability_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t target_family_name =
          loom_target_compile_report_text_non_empty(row->target_family_name);
      const iree_string_view_t namespace_name =
          loom_target_compile_report_text_non_empty(row->namespace_name);
      const iree_string_view_t key =
          loom_target_compile_report_text_non_empty(row->key);
      const iree_string_view_t value_kind =
          loom_target_compile_report_capability_value_kind_name(
              row->value_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: target_capability[%" PRIhsz
          "] function=%.*s target_family=%.*s namespace=%.*s key=%.*s "
          "value_kind=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)target_family_name.size, target_family_name.data,
          (int)namespace_name.size, namespace_name.data, (int)key.size,
          key.data, (int)value_kind.size, value_kind.data));
      switch (row->value_kind) {
        case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, " value=%s\n", row->value_u64 != 0 ? "true" : "false"));
          break;
        }
        case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_U64: {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, " value=%" PRIu64 "\n", row->value_u64));
          break;
        }
        case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING: {
          const iree_string_view_t value =
              loom_target_compile_report_text_non_empty(row->value_string);
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, " value=%.*s\n", (int)value.size, value.data));
          break;
        }
        case LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_NONE:
        default: {
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_cstring(builder, "\n"));
          break;
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_move_cause(
    iree_string_builder_t* builder,
    const loom_target_compile_report_move_cause_descriptor_t* descriptor,
    const loom_target_compile_report_move_cause_counts_t* counts) {
  if (counts->packet_count == 0 && counts->unit_count == 0) {
    return iree_ok_status();
  }
  return iree_string_builder_append_format(
      builder,
      "COMPILE-REPORT: move_cause[%.*s] packets=%" PRIu64 " units=%" PRIu64
      "\n",
      (int)descriptor->name.size, descriptor->name.data, counts->packet_count,
      counts->unit_count);
}

static iree_status_t loom_target_compile_report_format_move_causes(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors);
       ++i) {
    const loom_target_compile_report_move_cause_descriptor_t* descriptor =
        &loom_target_compile_report_move_cause_descriptors[i];
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_cause(
        builder, descriptor, &report->move_causes[descriptor->cause]));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_wait_counter_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->wait_counter_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_wait_counter_row_t* rows =
        (const loom_target_compile_report_wait_counter_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_wait_counter_row_t* row = &rows[i];
      const loom_target_compile_report_wait_plan_t* summary = &row->summary;
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_text_non_empty(row->counter_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: wait_counter[%" PRIhsz
          "] function=%.*s counter=%.*s counter_id=%" PRIu32 " actions=%" PRIu64
          " explicit=%" PRIu64 " planned=%" PRIu64 " full_drains=%" PRIu64
          " partial_waits=%" PRIu64 " drained=%" PRIu64 " max_drained=%" PRIu64
          " max_outstanding=%" PRIu64 " max_full_drain_outstanding=%" PRIu64
          "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)counter_name.size, counter_name.data, row->counter_id,
          summary->action_count, summary->explicit_action_count,
          summary->planned_action_count, summary->full_drain_count,
          summary->partial_wait_count, summary->drained_count,
          summary->max_drained_count, summary->max_outstanding_before,
          summary->max_full_drain_outstanding_before));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_wait_reason_summary_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->wait_reason_summary_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_wait_reason_summary_row_t* rows =
        (const loom_target_compile_report_wait_reason_summary_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_wait_reason_summary_row_t* row =
          &rows[i];
      const loom_target_compile_report_wait_plan_t* summary = &row->summary;
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_text_non_empty(row->counter_name);
      const iree_string_view_t reason_name =
          loom_target_compile_report_text_non_empty(row->reason_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: wait_reason_summary[%" PRIhsz
          "] function=%.*s counter=%.*s counter_id=%" PRIu32
          " reason=%.*s reason_id=%" PRIu32 " actions=%" PRIu64
          " explicit=%" PRIu64 " planned=%" PRIu64 " full_drains=%" PRIu64
          " partial_waits=%" PRIu64 " drained=%" PRIu64 " max_drained=%" PRIu64
          " max_outstanding=%" PRIu64 " max_full_drain_outstanding=%" PRIu64
          "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)counter_name.size, counter_name.data, row->counter_id,
          (int)reason_name.size, reason_name.data, row->reason_id,
          summary->action_count, summary->explicit_action_count,
          summary->planned_action_count, summary->full_drain_count,
          summary->partial_wait_count, summary->drained_count,
          summary->max_drained_count, summary->max_outstanding_before,
          summary->max_full_drain_outstanding_before));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_wait_action_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->wait_action_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_wait_action_row_t* rows =
        (const loom_target_compile_report_wait_action_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_wait_action_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_text_non_empty(row->counter_name);
      const iree_string_view_t action_name =
          loom_target_compile_report_text_non_empty(row->action_name);
      const iree_string_view_t reason_name =
          loom_target_compile_report_text_non_empty(row->reason_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: wait_action[%" PRIhsz
          "] function=%.*s counter=%.*s counter_id=%" PRIu32
          " action=%.*s action_id=%" PRIu32 " reason=%.*s reason_id=%" PRIu32
          " block=%" PRIu32 " node=%" PRIu32 " ordinal=%" PRIu32
          " producer_node=",
          row_index, (int)function_name.size, function_name.data,
          (int)counter_name.size, counter_name.data, row->counter_id,
          (int)action_name.size, action_name.data, row->action_id,
          (int)reason_name.size, reason_name.data, row->reason_id,
          row->block_index, row->node_index, row->scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->producer_node));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" producer_ordinal=")));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->producer_scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("producer_operation"),
          row->producer_operation_name));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("producer_descriptor_key"),
          row->producer_descriptor_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("producer_semantic_tag"),
          row->producer_semantic_tag));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" consumer_node=")));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->consumer_node));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" consumer_ordinal=")));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_u32(
          builder, row->consumer_scheduled_ordinal));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("consumer_operation"),
          row->consumer_operation_name));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("consumer_descriptor_key"),
          row->consumer_descriptor_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("consumer_semantic_tag"),
          row->consumer_semantic_tag));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " target_count=%" PRIu32 " outstanding_before=%" PRIu32
          " outstanding_after=%" PRIu32 " drained=%" PRIu32 "\n",
          row->target_count, row->outstanding_before, row->outstanding_after,
          row->drained_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_pressure_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->pressure_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_pressure_row_t* rows =
        (const loom_target_compile_report_pressure_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_pressure_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_text_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(
              (loom_scalar_type_t)row->element_type);
      const iree_string_view_t peak_block_name =
          loom_target_compile_report_text_non_empty(row->peak_block_name);
      const iree_string_view_t peak_operation_name =
          loom_target_compile_report_text_non_empty(row->peak_operation_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: pressure[%" PRIhsz
          "] function=%.*s class=%.*s type=%.*s element=%.*s "
          "peak_units=%" PRIu64 " peak_values=%" PRIu64
          " point=%u block=%.*s op=%.*s\n",
          row_index, (int)function_name.size, function_name.data,
          (int)register_class.size, register_class.data,
          (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data,
          row->peak_live_units, row->peak_live_values, row->peak_point,
          (int)peak_block_name.size, peak_block_name.data,
          (int)peak_operation_name.size, peak_operation_name.data));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_pressure_origin_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->pressure_origin_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_pressure_origin_row_t* rows =
        (const loom_target_compile_report_pressure_origin_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_pressure_origin_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_text_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t peak_block_name =
          loom_target_compile_report_text_non_empty(row->peak_block_name);
      const iree_string_view_t peak_operation_name =
          loom_target_compile_report_text_non_empty(row->peak_operation_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_text_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_text_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_text_non_empty(row->sample_value_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: pressure_origin[%" PRIhsz
          "] function=%.*s class=%.*s type=%.*s element=%.*s "
          "point=%u block=%.*s op=%.*s origin=%.*s origin_op=%.*s "
          "semantic=%.*s sample=%.*s live_units=%" PRIu64
          " live_values=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)register_class.size, register_class.data,
          (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data, row->peak_point,
          (int)peak_block_name.size, peak_block_name.data,
          (int)peak_operation_name.size, peak_operation_name.data,
          (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data,
          (int)sample_value_name.size, sample_value_name.data, row->live_units,
          row->live_values));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_schedule_band_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->schedule_band_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_schedule_band_row_t* rows =
        (const loom_target_compile_report_schedule_band_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_schedule_band_row_t* row = &rows[i];
      const loom_target_compile_report_static_instruction_mix_t* mix =
          &row->static_instruction_mix;
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t block_name =
          loom_target_compile_report_text_non_empty(row->block_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_text_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_text_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_text_non_empty(row->sample_value_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: schedule_band[%" PRIhsz
          "] function=%.*s block=%.*s block_index=%" PRIu32
          " first_packet=%" PRIu64 " first_ordinal=%" PRIu32 " nodes=%" PRIu32
          " origin=%.*s"
          " origin_op=%.*s semantic=%.*s sample=%.*s"
          " descriptors=%" PRIu64 " unknown=%" PRIu64 " scalar_alu=%" PRIu64
          " vector_alu=%" PRIu64 " matrix=%" PRIu64 " mfma=%" PRIu64
          " smfmac=%" PRIu64 " wmma=%" PRIu64 " swmmac=%" PRIu64 " dot=%" PRIu64
          " global_memory=%" PRIu64 " local_memory=%" PRIu64
          " scalar_memory=%" PRIu64 " generic_memory=%" PRIu64
          " atomic=%" PRIu64 " branch=%" PRIu64 " barrier=%" PRIu64
          " control=%" PRIu64 " conversion=%" PRIu64 " cache=%" PRIu64
          " register_move=%" PRIu64 " result_values=%" PRIu64
          " result_units=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)block_name.size, block_name.data, row->block_index,
          row->first_packet_index, row->first_scheduled_ordinal,
          row->node_count, (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data,
          (int)sample_value_name.size, sample_value_name.data,
          mix->descriptor_count, mix->unknown_count, mix->scalar_alu_count,
          mix->vector_alu_count, mix->matrix_count, mix->mfma_count,
          mix->smfmac_count, mix->wmma_count, mix->swmmac_count, mix->dot_count,
          mix->global_memory_count, mix->local_memory_count,
          mix->scalar_memory_count, mix->generic_memory_count,
          mix->atomic_count, mix->branch_count, mix->barrier_count,
          mix->control_count, mix->conversion_count, mix->cache_count,
          mix->register_move_count, row->result_value_count,
          row->result_unit_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_schedule_band_summary_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->schedule_band_summary_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_schedule_band_summary_row_t* rows =
        (const loom_target_compile_report_schedule_band_summary_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_schedule_band_summary_row_t* row =
          &rows[i];
      const loom_target_compile_report_static_instruction_mix_t* mix =
          &row->static_instruction_mix;
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t block_name =
          loom_target_compile_report_text_non_empty(row->block_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_text_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_text_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_text_non_empty(row->sample_value_name);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: schedule_band_summary[%" PRIhsz
          "] function=%.*s block=%.*s block_index=%" PRIu32
          " first_packet=%" PRIu64 " bands=%" PRIu64 " nodes=%" PRIu64
          " max_band_nodes=%" PRIu32
          " origin=%.*s"
          " origin_op=%.*s semantic=%.*s sample=%.*s"
          " descriptors=%" PRIu64 " unknown=%" PRIu64 " scalar_alu=%" PRIu64
          " vector_alu=%" PRIu64 " matrix=%" PRIu64 " mfma=%" PRIu64
          " smfmac=%" PRIu64 " wmma=%" PRIu64 " swmmac=%" PRIu64 " dot=%" PRIu64
          " global_memory=%" PRIu64 " local_memory=%" PRIu64
          " scalar_memory=%" PRIu64 " generic_memory=%" PRIu64
          " atomic=%" PRIu64 " branch=%" PRIu64 " barrier=%" PRIu64
          " control=%" PRIu64 " conversion=%" PRIu64 " cache=%" PRIu64
          " register_move=%" PRIu64 " result_values=%" PRIu64
          " result_units=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)block_name.size, block_name.data, row->block_index,
          row->first_packet_index, row->band_count, row->node_count,
          row->max_band_node_count, (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data,
          (int)sample_value_name.size, sample_value_name.data,
          mix->descriptor_count, mix->unknown_count, mix->scalar_alu_count,
          mix->vector_alu_count, mix->matrix_count, mix->mfma_count,
          mix->smfmac_count, mix->wmma_count, mix->swmmac_count, mix->dot_count,
          mix->global_memory_count, mix->local_memory_count,
          mix->scalar_memory_count, mix->generic_memory_count,
          mix->atomic_count, mix->branch_count, mix->barrier_count,
          mix->control_count, mix->conversion_count, mix->cache_count,
          mix->register_move_count, row->result_value_count,
          row->result_unit_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_spill_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->spill_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_spill_row_t* rows =
        (const loom_target_compile_report_spill_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_spill_row_t* row = &rows[i];
      const iree_string_view_t kind =
          loom_target_compile_report_spill_row_kind_name(row->kind);
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_text_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_text_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_text_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_text_non_empty(row->semantic_tag);
      const iree_string_view_t slot_space =
          loom_target_compile_report_text_non_empty(row->slot_space);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: spill[%" PRIhsz
          "] kind=%.*s function=%.*s value=%.*s class=%.*s type=%.*s "
          "element=%.*s "
          "origin=%.*s origin_op=%.*s semantic=%.*s "
          "assignment=%u slot=%u space=%.*s bytes=%" PRIu64 " align=%" PRIu64
          " stores=%" PRIu64 " store_bytes=%" PRIu64 " reloads=%" PRIu64
          " reload_bytes=%" PRIu64 "\n",
          row_index, (int)kind.size, kind.data, (int)function_name.size,
          function_name.data, (int)value_name.size, value_name.data,
          (int)register_class.size, register_class.data,
          (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data,
          (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data, row->assignment_index,
          row->slot_index, (int)slot_space.size, slot_space.data,
          row->byte_size, row->byte_alignment, row->store_count,
          row->store_bytes, row->reload_count, row->reload_bytes));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_allocation_failure_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->allocation_failure_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_allocation_failure_row_t* rows =
        (const loom_target_compile_report_allocation_failure_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_allocation_failure_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_text_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_text_non_empty(row->register_class);
      const iree_string_view_t failure_code =
          loom_target_compile_report_text_non_empty(row->failure_code);
      const iree_string_view_t blocking_kind =
          loom_target_compile_report_allocation_failure_blocking_kind_name(
              row->blocking_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_text_non_empty(row->origin_operation_name);
      const iree_string_view_t origin_block_name =
          loom_target_compile_report_text_non_empty(row->origin_block_name);
      const iree_string_view_t location_kind =
          loom_target_compile_report_text_non_empty(row->location_kind);
      const iree_string_view_t conflict_value_name =
          loom_target_compile_report_text_non_empty(row->conflict_value_name);
      const iree_string_view_t conflict_location_kind =
          loom_target_compile_report_text_non_empty(
              row->conflict_location_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: allocation_failure[%" PRIhsz
          "] function=%.*s value=%.*s class=%.*s code=%.*s blocking=%.*s "
          "origin=%.*s block=%.*s start=%u end=%u required_units=%u "
          "budget_units=",
          row_index, (int)function_name.size, function_name.data,
          (int)value_name.size, value_name.data, (int)register_class.size,
          register_class.data, (int)failure_code.size, failure_code.data,
          (int)blocking_kind.size, blocking_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)origin_block_name.size, origin_block_name.data, row->start_point,
          row->end_point, row->required_unit_count));
      if (row->budget_units == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->budget_units));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " peak_live_units=%u location=%.*s[", row->peak_live_units,
          (int)location_kind.size, location_kind.data));
      if (row->location_base == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->location_base));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ":%u] conflict_assignment=", row->location_count));
      if (row->conflict_assignment_index == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_assignment_index));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " conflict_value=%.*s conflict_start=",
          (int)conflict_value_name.size, conflict_value_name.data));
      if (row->conflict_start_point == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_start_point));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, IREE_SV(" conflict_end=")));
      if (row->conflict_end_point == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_end_point));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " conflict_location=%.*s[", (int)conflict_location_kind.size,
          conflict_location_kind.data));
      if (row->conflict_location_base == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("-")));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%u", row->conflict_location_base));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ":%u]\n", row->conflict_location_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_allocation_high_water_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->allocation_high_water_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_allocation_high_water_row_t* rows =
        (const loom_target_compile_report_allocation_high_water_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_allocation_high_water_row_t* row =
          &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_text_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_text_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_text_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_text_non_empty(row->semantic_tag);
      const iree_string_view_t location_kind =
          loom_target_compile_report_text_non_empty(row->location_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: allocation_high_water[%" PRIhsz
          "] function=%.*s value=%.*s class=%.*s type=%.*s element=%.*s "
          "assignment=%u origin=%.*s origin_op=%.*s semantic=%.*s start=%u "
          "end=%u required_units=%u location=%.*s[%u:%u] high_water=%" PRIu64
          " lower_free_units=%" PRIu64
          " lower_free_runs=%u "
          "lower_largest_free_run_units=%u"
          " lower_pressure_releasable_free_units=%" PRIu64
          " lower_pressure_releasable_free_runs=%u"
          " lower_pressure_releasable_largest_free_run_units=%u"
          " active_assignment_blockers=%u "
          "active_assignment_blocker_units=%" PRIu64
          " active_storage_lease_blockers=%u "
          "active_storage_lease_blocker_units=%" PRIu64
          " active_pressure_storage_lease_blockers=%u "
          "active_pressure_storage_lease_blocker_units=%" PRIu64
          " active_fallback_storage_lease_blockers=%u "
          "active_fallback_storage_lease_blocker_units=%" PRIu64 "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)value_name.size, value_name.data, (int)register_class.size,
          register_class.data, (int)type_kind_name.size, type_kind_name.data,
          (int)element_type_name.size, element_type_name.data,
          row->assignment_index, (int)origin_kind.size, origin_kind.data,
          (int)origin_operation_name.size, origin_operation_name.data,
          (int)semantic_tag.size, semantic_tag.data, row->start_point,
          row->end_point, row->required_unit_count, (int)location_kind.size,
          location_kind.data, row->location_base, row->location_count,
          row->high_water_units, row->lower_free_unit_count,
          row->lower_free_run_count, row->lower_largest_free_run_unit_count,
          row->lower_pressure_releasable_free_unit_count,
          row->lower_pressure_releasable_free_run_count,
          row->lower_pressure_releasable_largest_free_run_unit_count,
          row->active_assignment_blocker_count,
          row->active_assignment_blocker_units,
          row->active_storage_lease_blocker_count,
          row->active_storage_lease_blocker_units,
          row->active_pressure_storage_lease_blocker_count,
          row->active_pressure_storage_lease_blocker_units,
          row->active_fallback_storage_lease_blocker_count,
          row->active_fallback_storage_lease_blocker_units));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_config_binding_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->config_binding_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_config_binding_row_t* rows =
        (const loom_target_compile_report_config_binding_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_config_binding_row_t* row = &rows[i];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: config_binding[%" PRIhsz "] key=%.*s value=%.*s\n",
          row_index, (int)row->key.size, row->key.data, (int)row->value.size,
          row->value.data));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_target_compile_report_format_text_planning_details(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_move_causes(report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_wait_counter_rows(report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_wait_reason_summary_rows(report,
                                                                 builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_wait_action_rows(report, builder));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_target_capability_rows(
      report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_config_binding_rows(report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_pressure_rows(report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_pressure_origin_rows(report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_schedule_band_rows(report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_schedule_band_summary_rows(report,
                                                                   builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_spill_rows(report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_allocation_failure_rows(report,
                                                                builder));
  return loom_target_compile_report_format_allocation_high_water_rows(report,
                                                                      builder);
}
