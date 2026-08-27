// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/low_allocation.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/kernel.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"
#include "loom/target/reporting/low_names.h"

typedef struct loom_target_compile_report_pressure_origin_info_t {
  // Structured pressure origin kind.
  loom_target_compile_report_pressure_origin_kind_t kind;
  // Defining operation mnemonic when available.
  iree_string_view_t operation_name;
  // Descriptor semantic tag for descriptor-backed origins.
  iree_string_view_t semantic_tag;
} loom_target_compile_report_pressure_origin_info_t;

static loom_target_compile_report_pressure_origin_kind_t
loom_target_compile_report_pressure_origin_from_instruction_classes(
    loom_low_instruction_class_flags_t instruction_classes) {
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_DOT)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_DOT;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_MATRIX)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_LOCAL_MEMORY)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_LOCAL_MEMORY;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_GLOBAL_MEMORY)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_SCALAR_MEMORY)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SCALAR_MEMORY;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_PRIVATE_MEMORY)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_PRIVATE_MEMORY;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_GENERIC_MEMORY)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GENERIC_MEMORY;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_REGISTER_MOVE)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_REGISTER_MOVE;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_CONVERSION)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONVERSION;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_BARRIER)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BARRIER;
  }
  if (iree_any_bit_set(instruction_classes,
                       LOOM_LOW_INSTRUCTION_CLASS_FLAG_CONTROL |
                           LOOM_LOW_INSTRUCTION_CLASS_FLAG_BRANCH)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONTROL;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_CACHE)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CACHE;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_SCALAR_ALU)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SCALAR_ALU;
  }
  if (iree_all_bits_set(instruction_classes,
                        LOOM_LOW_INSTRUCTION_CLASS_FLAG_VECTOR_ALU)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_VECTOR_ALU;
  }
  return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN;
}

static loom_target_compile_report_pressure_origin_kind_t
loom_target_compile_report_pressure_origin_from_low_op(const loom_op_t* op) {
  if (op == NULL) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN;
  }
  if (loom_low_const_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONSTANT;
  }
  if (loom_low_copy_isa(op) || loom_low_move_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_COPY;
  }
  if (loom_low_slice_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SLICE;
  }
  if (loom_low_concat_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONCAT;
  }
  if (loom_low_storage_reserve_isa(op) || loom_low_storage_view_isa(op) ||
      loom_low_storage_address_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_STORAGE;
  }
  if (loom_low_spill_isa(op) || loom_low_reload_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SPILL_RELOAD;
  }
  if (loom_low_br_isa(op) || loom_low_cond_br_isa(op) ||
      loom_low_return_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONTROL;
  }
  return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_OPERATION;
}

static loom_target_compile_report_pressure_origin_info_t
loom_target_compile_report_pressure_origin_from_node(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node) {
  loom_target_compile_report_pressure_origin_info_t info = {
      .kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN,
      .operation_name = loom_target_compile_report_op_name(module, node->op),
  };
  if (node->kind == LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR &&
      node->descriptor != NULL) {
    info.kind =
        loom_target_compile_report_pressure_origin_from_instruction_classes(
            node->descriptor->instruction_class_flags);
    if (info.kind == LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN) {
      info.kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_OPERATION;
    }
    info.semantic_tag = loom_target_compile_report_descriptor_semantic_tag(
        descriptor_set, node->descriptor);
    return info;
  }
  info.kind = loom_target_compile_report_pressure_origin_from_low_op(node->op);
  return info;
}

static loom_target_compile_report_pressure_origin_info_t
loom_target_compile_report_pressure_origin_from_value(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return (loom_target_compile_report_pressure_origin_info_t){
        .kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN,
    };
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return (loom_target_compile_report_pressure_origin_info_t){
        .kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BLOCK_ARGUMENT,
        .operation_name = IREE_SV("<block-argument>"),
    };
  }
  const loom_op_t* op = loom_value_def_op(value);
  return (loom_target_compile_report_pressure_origin_info_t){
      .kind = loom_target_compile_report_pressure_origin_from_low_op(op),
      .operation_name = loom_target_compile_report_op_name(module, op),
  };
}

static iree_status_t loom_target_compile_report_build_pressure_origin_infos(
    const loom_low_schedule_table_t* schedule,
    loom_target_compile_report_pressure_origin_info_t* origin_infos,
    iree_host_size_t origin_info_count) {
  if (schedule == NULL || origin_infos == NULL) {
    return iree_ok_status();
  }
  const loom_module_t* module = schedule->module;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    if (node->result_count == 0) {
      continue;
    }
    const loom_target_compile_report_pressure_origin_info_t info =
        loom_target_compile_report_pressure_origin_from_node(
            module, descriptor_set, node);
    const loom_value_ordinal_t* result_ordinals =
        loom_low_schedule_node_const_result_ordinals(node);
    for (uint16_t j = 0; j < node->result_count; ++j) {
      const loom_value_ordinal_t result_ordinal = result_ordinals[j];
      if (result_ordinal >= origin_info_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low schedule result ordinal exceeds liveness value domain");
      }
      origin_infos[result_ordinal] = info;
    }
  }
  return iree_ok_status();
}

static bool loom_target_compile_report_schedule_band_matches(
    const loom_target_compile_report_schedule_band_row_t* row,
    const loom_target_compile_report_pressure_origin_info_t* info) {
  if (row->origin_kind != info->kind) {
    return false;
  }
  if (!iree_string_view_is_empty(row->semantic_tag) ||
      !iree_string_view_is_empty(info->semantic_tag)) {
    return iree_string_view_equal(row->semantic_tag, info->semantic_tag);
  }
  return iree_string_view_equal(row->origin_operation_name,
                                info->operation_name);
}

static void loom_target_compile_report_add_schedule_band_node_results(
    const loom_module_t* module, const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_node_t* node,
    loom_target_compile_report_schedule_band_row_t* row) {
  if (liveness == NULL || liveness->value_ids == NULL) {
    return;
  }
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_value_ordinal_t result_ordinal = result_ordinals[i];
    if (result_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
        result_ordinal >= liveness->value_count) {
      continue;
    }
    const loom_value_id_t value_id = liveness->value_ids[result_ordinal];
    if (iree_string_view_is_empty(row->sample_value_name)) {
      row->sample_value_name =
          loom_target_compile_report_value_name(module, value_id);
    }
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, result_ordinal);
    if (interval != NULL) {
      row->result_unit_count += interval->unit_count;
    }
    ++row->result_value_count;
  }
}

static void loom_target_compile_report_accumulate_schedule_band_node(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_node_t* node,
    const loom_target_compile_report_low_dynamic_context_t* dynamic_context,
    loom_target_compile_report_schedule_band_row_t* row) {
  ++row->node_count;
  loom_target_compile_report_static_instruction_mix_t node_mix = {0};
  loom_target_compile_report_accumulate_low_node_static_mix(
      schedule, allocation, schedule->target.descriptor_set, node, &node_mix);
  loom_target_compile_report_accumulate_static_mix(&row->static_instruction_mix,
                                                   &node_mix);
  if (iree_all_bits_set(
          row->flags,
          LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX)) {
    uint64_t multiplier = 1;
    const bool exact =
        dynamic_context != NULL && dynamic_context->exact &&
        loom_target_compile_report_low_node_execution_multiplier(
            schedule->module, &dynamic_context->fact_table,
            dynamic_context->block_multipliers, node, &multiplier) &&
        loom_target_compile_report_accumulate_scaled_static_mix(
            &row->dynamic_instruction_mix, &node_mix, multiplier);
    if (!exact) {
      row->flags &=
          ~LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX;
      row->dynamic_instruction_mix =
          (loom_target_compile_report_static_instruction_mix_t){0};
    }
  }
  loom_target_compile_report_add_schedule_band_node_results(
      schedule->module, liveness, node, row);
}

static bool loom_target_compile_report_schedule_band_summary_matches(
    const loom_target_compile_report_schedule_band_summary_row_t* summary,
    const loom_target_compile_report_schedule_band_row_t* band) {
  return summary->block_index == band->block_index &&
         iree_string_view_equal(summary->block_name, band->block_name) &&
         summary->origin_kind == band->origin_kind &&
         iree_string_view_equal(summary->origin_operation_name,
                                band->origin_operation_name) &&
         iree_string_view_equal(summary->semantic_tag, band->semantic_tag);
}

static void loom_target_compile_report_accumulate_schedule_band_summary(
    loom_target_compile_report_schedule_band_summary_row_t* summary,
    const loom_target_compile_report_schedule_band_row_t* band) {
  ++summary->band_count;
  summary->node_count += band->node_count;
  summary->max_band_node_count =
      iree_max(summary->max_band_node_count, band->node_count);
  if (iree_string_view_is_empty(summary->sample_value_name)) {
    summary->sample_value_name = band->sample_value_name;
  }
  loom_target_compile_report_accumulate_static_mix(
      &summary->static_instruction_mix, &band->static_instruction_mix);
  const bool summary_has_dynamic = iree_all_bits_set(
      summary->flags,
      LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX);
  const bool band_has_dynamic = iree_all_bits_set(
      band->flags,
      LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX);
  if (summary_has_dynamic && band_has_dynamic) {
    loom_target_compile_report_accumulate_static_mix(
        &summary->dynamic_instruction_mix, &band->dynamic_instruction_mix);
  } else if (summary_has_dynamic != band_has_dynamic) {
    summary->flags &=
        ~LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX;
    summary->dynamic_instruction_mix =
        (loom_target_compile_report_static_instruction_mix_t){0};
  }
  summary->result_value_count += band->result_value_count;
  summary->result_unit_count += band->result_unit_count;
}

static iree_status_t loom_target_compile_report_add_schedule_band_summary(
    loom_target_compile_report_schedule_band_summary_row_t* summaries,
    iree_host_size_t summary_capacity, iree_host_size_t* summary_count,
    const loom_target_compile_report_schedule_band_row_t* band) {
  for (iree_host_size_t i = 0; i < *summary_count; ++i) {
    if (loom_target_compile_report_schedule_band_summary_matches(&summaries[i],
                                                                 band)) {
      loom_target_compile_report_accumulate_schedule_band_summary(&summaries[i],
                                                                  band);
      return iree_ok_status();
    }
  }
  if (*summary_count >= summary_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low schedule band summary capacity exceeded");
  }
  loom_target_compile_report_schedule_band_summary_row_t* summary =
      &summaries[(*summary_count)++];
  *summary = (loom_target_compile_report_schedule_band_summary_row_t){
      .flags = band->flags,
      .function_name = band->function_name,
      .block_name = band->block_name,
      .block_index = band->block_index,
      .first_packet_index = band->first_packet_index,
      .origin_kind = band->origin_kind,
      .origin_operation_name = band->origin_operation_name,
      .semantic_tag = band->semantic_tag,
      .sample_value_name = band->sample_value_name,
  };
  loom_target_compile_report_accumulate_schedule_band_summary(summary, band);
  return iree_ok_status();
}

typedef uint32_t loom_target_compile_report_schedule_band_emit_flags_t;

enum {
  LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_EMIT_ROWS = 1u << 0,
};

static iree_status_t loom_target_compile_report_record_schedule_band(
    loom_target_compile_report_t* report,
    loom_target_compile_report_schedule_band_summary_row_t* summaries,
    iree_host_size_t summary_capacity, iree_host_size_t* summary_count,
    loom_target_compile_report_schedule_band_emit_flags_t flags,
    const loom_target_compile_report_schedule_band_row_t* band) {
  if (iree_any_bit_set(flags,
                       LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_EMIT_ROWS)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_schedule_band_row(report, band));
  }
  return loom_target_compile_report_add_schedule_band_summary(
      summaries, summary_capacity, summary_count, band);
}

static iree_status_t loom_target_compile_report_record_schedule_band_rows(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation,
    const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_table_t* schedule,
    const loom_target_compile_report_low_dynamic_context_t* dynamic_context) {
  const bool wants_band_rows = loom_target_compile_report_wants_details(
      report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS);
  const bool wants_band_summary_rows = loom_target_compile_report_wants_details(
      report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS);
  const bool wants_band_summaries = wants_band_rows || wants_band_summary_rows;
  if (!wants_band_summaries) {
    return iree_ok_status();
  }
  if (wants_band_rows) {
    report->detail_flags |=
        LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS;
  }
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS;
  if (schedule == NULL || schedule->scheduled_node_count == 0 ||
      schedule->block_count == 0 || iree_allocator_is_null(report->allocator)) {
    return iree_ok_status();
  }
  if (schedule->scheduled_node_indices == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low schedule band rows require scheduled nodes");
  }
  const loom_module_t* module = schedule->module;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  const loom_target_compile_report_schedule_band_emit_flags_t band_flags =
      wants_band_rows ? LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_EMIT_ROWS : 0;
  iree_host_size_t summary_bytes = 0;
  if (!iree_host_size_checked_mul(
          schedule->scheduled_node_count,
          sizeof(loom_target_compile_report_schedule_band_summary_row_t),
          &summary_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low schedule band summary scratch is too large");
  }
  loom_target_compile_report_schedule_band_summary_row_t* summaries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(report->allocator, summary_bytes,
                                             (void**)&summaries));
  memset(summaries, 0, summary_bytes);
  iree_status_t status = iree_ok_status();
  iree_host_size_t summary_count = 0;
  for (iree_host_size_t block_index = 0;
       iree_status_is_ok(status) && block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    bool has_band = false;
    loom_target_compile_report_schedule_band_row_t band = {0};
    for (uint32_t i = 0;
         iree_status_is_ok(status) && i < block->scheduled_node_count; ++i) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)block->scheduled_node_start + (iree_host_size_t)i;
      if (packet_index >= schedule->scheduled_node_count) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "low schedule band packet index is invalid");
        break;
      }
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      if (node_index >= schedule->node_count) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "low schedule band node index is invalid");
        break;
      }
      const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
      const loom_target_compile_report_pressure_origin_info_t info =
          loom_target_compile_report_pressure_origin_from_node(
              module, descriptor_set, node);
      if (!has_band ||
          !loom_target_compile_report_schedule_band_matches(&band, &info)) {
        if (has_band) {
          status = loom_target_compile_report_record_schedule_band(
              report, summaries, schedule->scheduled_node_count, &summary_count,
              band_flags, &band);
          if (!iree_status_is_ok(status)) {
            break;
          }
        }
        band = (loom_target_compile_report_schedule_band_row_t){
            .flags =
                dynamic_context != NULL && dynamic_context->exact
                    ? LOOM_TARGET_COMPILE_REPORT_SCHEDULE_BAND_DYNAMIC_INSTRUCTION_MIX
                    : 0,
            .function_name = report->function_name,
            .block_name =
                loom_target_compile_report_block_name(module, node->block),
            .block_index = (uint32_t)block_index,
            .first_packet_index = (uint64_t)packet_index,
            .first_scheduled_ordinal = node->scheduled_ordinal,
            .origin_kind = info.kind,
            .origin_operation_name = info.operation_name,
            .semantic_tag = info.semantic_tag,
        };
        has_band = true;
      }
      loom_target_compile_report_accumulate_schedule_band_node(
          schedule, allocation, liveness, node, dynamic_context, &band);
    }
    if (iree_status_is_ok(status) && has_band) {
      status = loom_target_compile_report_record_schedule_band(
          report, summaries, schedule->scheduled_node_count, &summary_count,
          band_flags, &band);
    }
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < summary_count;
       ++i) {
    status = loom_target_compile_report_record_schedule_band_summary_row(
        report, &summaries[i]);
  }
  iree_allocator_free(report->allocator, summaries);
  return status;
}

static bool loom_target_compile_report_interval_is_live_at_point(
    const loom_liveness_interval_t* interval, uint32_t point) {
  return interval->start_point <= point && point < interval->end_point;
}

static bool loom_target_compile_report_pressure_origin_row_matches(
    const loom_target_compile_report_pressure_origin_row_t* row,
    const loom_target_compile_report_pressure_origin_row_t* candidate) {
  return row->origin_kind == candidate->origin_kind &&
         iree_string_view_equal(row->origin_operation_name,
                                candidate->origin_operation_name) &&
         iree_string_view_equal(row->semantic_tag, candidate->semantic_tag);
}

static iree_status_t loom_target_compile_report_record_pressure_origin_rows(
    loom_target_compile_report_t* report,
    const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_table_t* schedule,
    const loom_low_descriptor_set_t* descriptor_set) {
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    return iree_ok_status();
  }
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS;
  if (liveness->value_count == 0 || liveness->pressure_summary_count == 0 ||
      iree_allocator_is_null(report->allocator)) {
    return iree_ok_status();
  }
  iree_host_size_t origin_info_bytes = 0;
  if (!iree_host_size_checked_mul(
          liveness->value_count,
          sizeof(loom_target_compile_report_pressure_origin_info_t),
          &origin_info_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pressure origin info table is too large");
  }
  iree_host_size_t row_bytes = 0;
  if (!iree_host_size_checked_mul(
          liveness->value_count,
          sizeof(loom_target_compile_report_pressure_origin_row_t),
          &row_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pressure origin row scratch is too large");
  }
  loom_target_compile_report_pressure_origin_info_t* origin_infos = NULL;
  loom_target_compile_report_pressure_origin_row_t* rows = NULL;
  iree_status_t status = iree_allocator_malloc(
      report->allocator, origin_info_bytes, (void**)&origin_infos);
  if (iree_status_is_ok(status)) {
    memset(origin_infos, 0, origin_info_bytes);
    status = loom_target_compile_report_build_pressure_origin_infos(
        schedule, origin_infos, liveness->value_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(report->allocator, row_bytes, (void**)&rows);
  }
  for (iree_host_size_t summary_index = 0;
       iree_status_is_ok(status) &&
       summary_index < liveness->pressure_summary_count;
       ++summary_index) {
    const loom_liveness_pressure_summary_t* summary =
        &liveness->pressure_summaries[summary_index];
    iree_host_size_t row_count = 0;
    for (iree_host_size_t value_ordinal = 0;
         value_ordinal < liveness->value_count; ++value_ordinal) {
      const loom_liveness_interval_t* interval =
          loom_liveness_interval_for_value_ordinal(
              liveness, (loom_value_ordinal_t)value_ordinal);
      if (interval == NULL ||
          !loom_liveness_value_class_equal(interval->value_class,
                                           summary->value_class) ||
          !loom_target_compile_report_interval_is_live_at_point(
              interval, summary->peak_point)) {
        continue;
      }
      loom_target_compile_report_pressure_origin_info_t origin_info =
          origin_infos[value_ordinal];
      if (origin_info.kind ==
          LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN) {
        origin_info = loom_target_compile_report_pressure_origin_from_value(
            liveness->module, interval->value_id);
      }
      loom_target_compile_report_pressure_origin_row_t candidate = {
          .function_name = report->function_name,
          .register_class = loom_target_compile_report_value_class_name(
              descriptor_set, summary->value_class),
          .type_kind = summary->value_class.type_kind,
          .element_type = summary->value_class.element_type,
          .peak_point = summary->peak_point,
          .peak_block_name = loom_target_compile_report_block_name(
              liveness->module, summary->peak_block),
          .peak_operation_name = summary->peak_op
                                     ? loom_target_compile_report_op_name(
                                           liveness->module, summary->peak_op)
                                     : IREE_SV("<block-boundary>"),
          .origin_kind = origin_info.kind,
          .origin_operation_name = origin_info.operation_name,
          .semantic_tag = origin_info.semantic_tag,
          .sample_value_name = loom_target_compile_report_value_name(
              liveness->module, interval->value_id),
          .live_units = interval->unit_count,
          .live_values = 1,
      };
      loom_target_compile_report_pressure_origin_row_t* row = NULL;
      for (iree_host_size_t row_index = 0; row_index < row_count; ++row_index) {
        if (loom_target_compile_report_pressure_origin_row_matches(
                &rows[row_index], &candidate)) {
          row = &rows[row_index];
          break;
        }
      }
      if (row == NULL) {
        row = &rows[row_count++];
        *row = candidate;
      } else {
        row->live_units += candidate.live_units;
        row->live_values += candidate.live_values;
      }
    }
    for (iree_host_size_t row_index = 0;
         iree_status_is_ok(status) && row_index < row_count; ++row_index) {
      status = loom_target_compile_report_record_pressure_origin_row(
          report, &rows[row_index]);
    }
  }
  if (rows != NULL) {
    iree_allocator_free(report->allocator, rows);
  }
  if (origin_infos != NULL) {
    iree_allocator_free(report->allocator, origin_infos);
  }
  return status;
}

static bool loom_target_compile_report_liveness_value_ordinal(
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  if (liveness == NULL || liveness->value_ids == NULL) {
    return false;
  }
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    if (liveness->value_ids[i] == value_id) {
      *out_value_ordinal = (loom_value_ordinal_t)i;
      return true;
    }
  }
  return false;
}

static loom_target_compile_report_pressure_origin_info_t
loom_target_compile_report_pressure_origin_for_liveness_value(
    const loom_module_t* module, const loom_liveness_analysis_t* liveness,
    const loom_target_compile_report_pressure_origin_info_t* origin_infos,
    loom_value_id_t value_id) {
  loom_target_compile_report_pressure_origin_info_t origin_info =
      loom_target_compile_report_pressure_origin_from_value(module, value_id);
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (origin_infos != NULL &&
      loom_target_compile_report_liveness_value_ordinal(liveness, value_id,
                                                        &value_ordinal) &&
      value_ordinal < liveness->value_count &&
      origin_infos[value_ordinal].kind !=
          LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN) {
    origin_info = origin_infos[value_ordinal];
  }
  return origin_info;
}

static iree_status_t loom_target_compile_report_acquire_pressure_origin_infos(
    loom_target_compile_report_t* report,
    const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_table_t* schedule,
    loom_target_compile_report_pressure_origin_info_t** out_origin_infos) {
  *out_origin_infos = NULL;
  if (schedule == NULL || liveness == NULL || liveness->value_count == 0 ||
      iree_allocator_is_null(report->allocator)) {
    return iree_ok_status();
  }
  iree_host_size_t origin_info_bytes = 0;
  if (!iree_host_size_checked_mul(
          liveness->value_count,
          sizeof(loom_target_compile_report_pressure_origin_info_t),
          &origin_info_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pressure origin info table is too large");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      report->allocator, origin_info_bytes, (void**)out_origin_infos));
  memset(*out_origin_infos, 0, origin_info_bytes);
  iree_status_t status = loom_target_compile_report_build_pressure_origin_infos(
      schedule, *out_origin_infos, liveness->value_count);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(report->allocator, *out_origin_infos);
    *out_origin_infos = NULL;
  }
  return status;
}

static iree_status_t loom_target_compile_report_record_pressure_rows(
    loom_target_compile_report_t* report,
    const loom_liveness_analysis_t* liveness,
    const loom_low_descriptor_set_t* descriptor_set) {
  const bool wants_pressure_rows = loom_target_compile_report_wants_details(
      report, LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS);
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const loom_liveness_pressure_summary_t* summary =
        &liveness->pressure_summaries[i];
    const iree_string_view_t register_class =
        loom_target_compile_report_value_class_name(descriptor_set,
                                                    summary->value_class);
    const loom_target_compile_report_pressure_summary_t pressure_summary = {
        .register_class = register_class,
        .peak_live_units = summary->peak_live_units,
    };
    if (wants_pressure_rows) {
      const loom_target_compile_report_pressure_row_t row = {
          .function_name = report->function_name,
          .register_class = register_class,
          .type_kind = summary->value_class.type_kind,
          .element_type = summary->value_class.element_type,
          .peak_live_units = summary->peak_live_units,
          .peak_live_values = summary->peak_live_values,
          .peak_point = summary->peak_point,
          .peak_block_name = loom_target_compile_report_block_name(
              liveness->module, summary->peak_block),
          .peak_operation_name = summary->peak_op
                                     ? loom_target_compile_report_op_name(
                                           liveness->module, summary->peak_op)
                                     : IREE_SV("<block-boundary>"),
      };
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_pressure_row(report, &row));
    } else {
      IREE_RETURN_IF_ERROR(loom_target_compile_report_record_pressure_summary(
          report, &pressure_summary));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_record_spill_rows(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_table_t* schedule) {
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    return iree_ok_status();
  }
  if (allocation->spill_plan_count == 0) {
    return iree_ok_status();
  }
  const loom_liveness_analysis_t* liveness = &allocation->liveness;
  loom_target_compile_report_pressure_origin_info_t* origin_infos = NULL;
  iree_status_t status =
      loom_target_compile_report_acquire_pressure_origin_infos(
          report, liveness, schedule, &origin_infos);
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < allocation->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan =
        &allocation->spill_plans[i];
    const loom_low_allocation_assignment_t* assignment = NULL;
    if (spill_plan->assignment_index < allocation->assignment_count) {
      assignment = &allocation->assignments[spill_plan->assignment_index];
    }
    const loom_liveness_value_class_t value_class =
        assignment != NULL ? assignment->value_class
                           : (loom_liveness_value_class_t){0};
    const loom_target_compile_report_pressure_origin_info_t origin_info =
        loom_target_compile_report_pressure_origin_for_liveness_value(
            allocation->module, liveness, origin_infos, spill_plan->value_id);
    const loom_target_compile_report_spill_row_t row = {
        .kind = LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_PLANNED,
        .function_name = report->function_name,
        .value_name = loom_target_compile_report_value_name(
            allocation->module, spill_plan->value_id),
        .register_class = loom_target_compile_report_value_class_name(
            allocation->target.descriptor_set, value_class),
        .type_kind = value_class.type_kind,
        .element_type = value_class.element_type,
        .origin_kind = origin_info.kind,
        .origin_operation_name = origin_info.operation_name,
        .semantic_tag = origin_info.semantic_tag,
        .assignment_index = spill_plan->assignment_index,
        .slot_index = spill_plan->slot_index,
        .slot_space = loom_low_spill_slot_space_name(spill_plan->slot_space),
        .byte_size = spill_plan->byte_size,
        .byte_alignment = spill_plan->byte_alignment,
        .store_count = spill_plan->store_count,
        .store_bytes = iree_math_saturating_mul_u64(spill_plan->byte_size,
                                                    spill_plan->store_count),
        .reload_count = spill_plan->reload_count,
        .reload_bytes = iree_math_saturating_mul_u64(spill_plan->byte_size,
                                                     spill_plan->reload_count),
    };
    status = loom_target_compile_report_record_spill_row(report, &row);
  }
  if (origin_infos != NULL) {
    iree_allocator_free(report->allocator, origin_infos);
  }
  return status;
}

iree_status_t loom_target_compile_report_record_materialized_spill_rows(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    return iree_ok_status();
  }
  if (frame->materialized_spills.record_count == 0) {
    return iree_ok_status();
  }
  const loom_liveness_analysis_t* liveness = &frame->allocation.liveness;
  loom_target_compile_report_pressure_origin_info_t* origin_infos = NULL;
  iree_status_t status =
      loom_target_compile_report_acquire_pressure_origin_infos(
          report, liveness, &frame->schedule, &origin_infos);
  for (const loom_low_allocation_materialized_spill_vec_t* vec =
           frame->materialized_spills.head;
       iree_status_is_ok(status) && vec != NULL; vec = vec->next) {
    for (iree_host_size_t i = 0;
         iree_status_is_ok(status) && i < vec->record_count; ++i) {
      const loom_low_allocation_materialized_spill_t* spill = &vec->records[i];
      loom_target_compile_report_pressure_origin_info_t origin_info =
          loom_target_compile_report_pressure_origin_for_liveness_value(
              frame->module, liveness, origin_infos, spill->value_id);
      if (origin_info.kind ==
              LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN &&
          iree_all_bits_set(
              spill->flags,
              LOOM_LOW_ALLOCATION_MATERIALIZED_SPILL_FLAG_VALUE_WAS_BLOCK_ARGUMENT)) {
        origin_info = (loom_target_compile_report_pressure_origin_info_t){
            .kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BLOCK_ARGUMENT,
            .operation_name = IREE_SV("<block-argument>"),
        };
      }
      const loom_target_compile_report_spill_row_t row = {
          .kind = LOOM_TARGET_COMPILE_REPORT_SPILL_ROW_MATERIALIZED,
          .function_name = report->function_name,
          .value_name = loom_target_compile_report_value_name(frame->module,
                                                              spill->value_id),
          .register_class = loom_target_compile_report_value_class_name(
              frame->target.descriptor_set, spill->value_class),
          .type_kind = spill->value_class.type_kind,
          .element_type = spill->value_class.element_type,
          .origin_kind = origin_info.kind,
          .origin_operation_name = origin_info.operation_name,
          .semantic_tag = origin_info.semantic_tag,
          .assignment_index = spill->assignment_index,
          .slot_index = spill->slot_index,
          .slot_space = loom_low_spill_slot_space_name(spill->slot_space),
          .byte_size = spill->byte_size,
          .byte_alignment = spill->byte_alignment,
          .store_count = spill->store_count,
          .store_bytes = spill->store_bytes,
          .reload_count = spill->reload_count,
          .reload_bytes = spill->reload_bytes,
      };
      status = loom_target_compile_report_record_spill_row(report, &row);
    }
  }
  if (origin_infos != NULL) {
    iree_allocator_free(report->allocator, origin_infos);
  }
  return status;
}

typedef struct loom_target_compile_report_allocation_high_water_scratch_t {
  // Assignment index that reaches |high_water_units|, or UINT32_MAX.
  uint32_t assignment_index;
  // One-past-last physical register unit reached by the assignment.
  uint64_t high_water_units;
} loom_target_compile_report_allocation_high_water_scratch_t;

typedef struct loom_target_compile_report_allocation_high_water_blockers_t {
  // Number of active assignment blockers below the high-water assignment.
  uint32_t active_assignment_count;
  // Total units owned by active assignment blockers.
  uint64_t active_assignment_units;
  // Number of active target storage-lease blockers below the high-water
  // assignment.
  uint32_t active_storage_lease_count;
  // Total units owned by active target storage-lease blockers.
  uint64_t active_storage_lease_units;
  // Number of pressure-releasable storage-lease blockers below the high-water
  // assignment.
  uint32_t active_pressure_storage_lease_count;
  // Total units owned by pressure-releasable storage-lease blockers.
  uint64_t active_pressure_storage_lease_units;
  // Number of fallback-release storage-lease blockers below the high-water
  // assignment.
  uint32_t active_fallback_storage_lease_count;
  // Total units owned by fallback-release storage-lease blockers.
  uint64_t active_fallback_storage_lease_units;
} loom_target_compile_report_allocation_high_water_blockers_t;

typedef struct loom_target_compile_report_allocation_lower_free_runs_t {
  // Number of unoccupied physical units below the high-water assignment base.
  uint64_t free_unit_count;
  // Number of contiguous unoccupied-unit runs below the assignment base.
  uint32_t free_run_count;
  // Largest contiguous unoccupied-unit run below the assignment base.
  uint32_t largest_free_run_unit_count;
} loom_target_compile_report_allocation_lower_free_runs_t;

typedef enum loom_target_compile_report_storage_lease_occupancy_e {
  // Treat active storage leases as occupied.
  LOOM_TARGET_COMPILE_REPORT_STORAGE_LEASE_OCCUPANCY_ACTIVE = 0,
  // Treat pressure-releasable storage leases as free.
  LOOM_TARGET_COMPILE_REPORT_STORAGE_LEASE_OCCUPANCY_PRESSURE_RELEASE_UPPER_BOUND =
      1,
} loom_target_compile_report_storage_lease_occupancy_t;

static bool loom_target_compile_report_assignment_active_at(
    const loom_low_allocation_assignment_t* assignment, uint32_t point) {
  return assignment->start_point <= point && point < assignment->end_point;
}

static bool loom_target_compile_report_unit_inside_range(
    uint32_t unit, uint32_t location_base, uint32_t location_count) {
  return unit >= location_base &&
         (uint64_t)unit < (uint64_t)location_base + location_count;
}

static bool loom_target_compile_report_storage_lease_is_pressure_releasable(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_storage_lease_t* lease) {
  if (allocation->storage_leases.records == NULL ||
      lease->lease_record_index >= allocation->storage_leases.record_count) {
    return false;
  }
  return iree_all_bits_set(
      allocation->storage_leases.records[lease->lease_record_index].flags,
      LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE);
}

static bool loom_target_compile_report_unit_blocked_by_assignment(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint32_t point, uint32_t unit) {
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    if (i == high_water_assignment_index) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->descriptor_reg_class_id !=
            high_water_assignment->descriptor_reg_class_id ||
        assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        !loom_target_compile_report_assignment_active_at(assignment, point)) {
      continue;
    }
    if (loom_target_compile_report_unit_inside_range(
            unit, assignment->location_base, assignment->location_count)) {
      return true;
    }
  }
  return false;
}

static bool loom_target_compile_report_unit_blocked_by_storage_lease(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint32_t point, uint32_t unit,
    loom_target_compile_report_storage_lease_occupancy_t occupancy) {
  for (iree_host_size_t i = 0; i < allocation->storage_lease_instance_count;
       ++i) {
    const loom_low_allocation_storage_lease_t* lease =
        &allocation->storage_lease_instances[i];
    if (lease->assignment_index == high_water_assignment_index ||
        lease->descriptor_reg_class_id !=
            high_water_assignment->descriptor_reg_class_id ||
        lease->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        point < lease->start_point || point >= lease->end_point) {
      continue;
    }
    if (occupancy ==
            LOOM_TARGET_COMPILE_REPORT_STORAGE_LEASE_OCCUPANCY_PRESSURE_RELEASE_UPPER_BOUND &&
        loom_target_compile_report_storage_lease_is_pressure_releasable(
            allocation, lease)) {
      continue;
    }
    if (loom_target_compile_report_unit_inside_range(unit, lease->location_base,
                                                     lease->location_count)) {
      return true;
    }
  }
  return false;
}

static bool loom_target_compile_report_unit_blocked_in_lower_window(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint32_t point, uint32_t unit,
    loom_target_compile_report_storage_lease_occupancy_t occupancy) {
  return loom_target_compile_report_unit_blocked_by_assignment(
             allocation, high_water_assignment, high_water_assignment_index,
             point, unit) ||
         loom_target_compile_report_unit_blocked_by_storage_lease(
             allocation, high_water_assignment, high_water_assignment_index,
             point, unit, occupancy);
}

static loom_target_compile_report_allocation_lower_free_runs_t
loom_target_compile_report_allocation_lower_free_runs(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index,
    loom_target_compile_report_storage_lease_occupancy_t occupancy) {
  loom_target_compile_report_allocation_lower_free_runs_t free_runs = {0};
  const uint32_t point = high_water_assignment->start_point;
  uint32_t current_free_run = 0;
  for (uint32_t unit = 0; unit < high_water_assignment->location_base; ++unit) {
    if (loom_target_compile_report_unit_blocked_in_lower_window(
            allocation, high_water_assignment, high_water_assignment_index,
            point, unit, occupancy)) {
      current_free_run = 0;
      continue;
    }
    ++free_runs.free_unit_count;
    if (current_free_run == 0) {
      ++free_runs.free_run_count;
    }
    ++current_free_run;
    free_runs.largest_free_run_unit_count =
        iree_max(free_runs.largest_free_run_unit_count, current_free_run);
  }
  return free_runs;
}

static uint64_t loom_target_compile_report_units_below_high_water(
    uint32_t location_base, uint32_t location_count,
    uint64_t high_water_units) {
  if (location_count == 0 || (uint64_t)location_base >= high_water_units) {
    return 0;
  }
  const uint64_t location_end = (uint64_t)location_base + location_count;
  const uint64_t clipped_end = iree_min(location_end, high_water_units);
  return clipped_end - location_base;
}

static void loom_target_compile_report_add_high_water_assignment_blockers(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint64_t high_water_units,
    loom_target_compile_report_allocation_high_water_blockers_t* blockers) {
  const uint32_t point = high_water_assignment->start_point;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    if (i == high_water_assignment_index) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->descriptor_reg_class_id !=
            high_water_assignment->descriptor_reg_class_id ||
        assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        !loom_target_compile_report_assignment_active_at(assignment, point)) {
      continue;
    }
    const uint64_t blocker_units =
        loom_target_compile_report_units_below_high_water(
            assignment->location_base, assignment->location_count,
            high_water_units);
    if (blocker_units == 0) {
      continue;
    }
    ++blockers->active_assignment_count;
    blockers->active_assignment_units += blocker_units;
  }
}

static void loom_target_compile_report_add_high_water_storage_lease_blockers(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint64_t high_water_units,
    loom_target_compile_report_allocation_high_water_blockers_t* blockers) {
  const uint32_t point = high_water_assignment->start_point;
  for (iree_host_size_t i = 0; i < allocation->storage_lease_instance_count;
       ++i) {
    const loom_low_allocation_storage_lease_t* lease =
        &allocation->storage_lease_instances[i];
    if (lease->assignment_index == high_water_assignment_index ||
        lease->descriptor_reg_class_id !=
            high_water_assignment->descriptor_reg_class_id ||
        lease->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        point < lease->start_point || point >= lease->end_point) {
      continue;
    }
    const uint64_t blocker_units =
        loom_target_compile_report_units_below_high_water(
            lease->location_base, lease->location_count, high_water_units);
    if (blocker_units == 0) {
      continue;
    }
    ++blockers->active_storage_lease_count;
    blockers->active_storage_lease_units += blocker_units;
    const bool pressure_releasable =
        lease->lease_record_index < allocation->storage_leases.record_count &&
        allocation->storage_leases.records != NULL &&
        iree_all_bits_set(
            allocation->storage_leases.records[lease->lease_record_index].flags,
            LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE);
    if (pressure_releasable) {
      ++blockers->active_pressure_storage_lease_count;
      blockers->active_pressure_storage_lease_units += blocker_units;
    } else {
      ++blockers->active_fallback_storage_lease_count;
      blockers->active_fallback_storage_lease_units += blocker_units;
    }
  }
}

static loom_target_compile_report_allocation_high_water_blockers_t
loom_target_compile_report_allocation_high_water_blockers(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint64_t high_water_units) {
  loom_target_compile_report_allocation_high_water_blockers_t blockers = {0};
  loom_target_compile_report_add_high_water_assignment_blockers(
      allocation, high_water_assignment, high_water_assignment_index,
      high_water_units, &blockers);
  loom_target_compile_report_add_high_water_storage_lease_blockers(
      allocation, high_water_assignment, high_water_assignment_index,
      high_water_units, &blockers);
  return blockers;
}

static iree_status_t
loom_target_compile_report_record_allocation_high_water_rows(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_table_t* schedule) {
  if (!loom_target_compile_report_wants_details(
          report,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      allocation->target.descriptor_set;
  if (descriptor_set == NULL || descriptor_set->reg_class_count == 0 ||
      allocation->assignment_count == 0 ||
      iree_allocator_is_null(report->allocator)) {
    return iree_ok_status();
  }
  iree_host_size_t scratch_bytes = 0;
  if (!iree_host_size_checked_mul(
          descriptor_set->reg_class_count,
          sizeof(loom_target_compile_report_allocation_high_water_scratch_t),
          &scratch_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation high-water scratch is too large");
  }
  loom_target_compile_report_allocation_high_water_scratch_t* scratch = NULL;
  iree_status_t status =
      iree_allocator_malloc(report->allocator, scratch_bytes, (void**)&scratch);
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
      scratch[i] = (loom_target_compile_report_allocation_high_water_scratch_t){
          .assignment_index = UINT32_MAX,
      };
    }
  }

  const loom_liveness_analysis_t* liveness = &allocation->liveness;
  loom_target_compile_report_pressure_origin_info_t* origin_infos = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_acquire_pressure_origin_infos(
        report, liveness, schedule, &origin_infos);
  }

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        assignment->location_count == 0) {
      continue;
    }
    if (assignment->descriptor_reg_class_id >=
        descriptor_set->reg_class_count) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "allocation assignment register class exceeds descriptor set");
      break;
    }
    const uint64_t high_water_units =
        (uint64_t)assignment->location_base + assignment->location_count;
    loom_target_compile_report_allocation_high_water_scratch_t* entry =
        &scratch[assignment->descriptor_reg_class_id];
    if (high_water_units > entry->high_water_units) {
      entry->assignment_index = (uint32_t)i;
      entry->high_water_units = high_water_units;
    }
  }

  for (uint32_t i = 0;
       iree_status_is_ok(status) && i < descriptor_set->reg_class_count; ++i) {
    const loom_target_compile_report_allocation_high_water_scratch_t* entry =
        &scratch[i];
    if (entry->assignment_index == UINT32_MAX ||
        entry->assignment_index >= allocation->assignment_count) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[entry->assignment_index];
    const loom_target_compile_report_allocation_high_water_blockers_t blockers =
        loom_target_compile_report_allocation_high_water_blockers(
            allocation, assignment, entry->assignment_index,
            entry->high_water_units);
    const loom_target_compile_report_allocation_lower_free_runs_t free_runs =
        loom_target_compile_report_allocation_lower_free_runs(
            allocation, assignment, entry->assignment_index,
            LOOM_TARGET_COMPILE_REPORT_STORAGE_LEASE_OCCUPANCY_ACTIVE);
    const loom_target_compile_report_allocation_lower_free_runs_t
        pressure_release_free_runs =
            loom_target_compile_report_allocation_lower_free_runs(
                allocation, assignment, entry->assignment_index,
                LOOM_TARGET_COMPILE_REPORT_STORAGE_LEASE_OCCUPANCY_PRESSURE_RELEASE_UPPER_BOUND);
    const loom_target_compile_report_pressure_origin_info_t origin_info =
        loom_target_compile_report_pressure_origin_for_liveness_value(
            allocation->module, liveness, origin_infos, assignment->value_id);
    const loom_target_compile_report_allocation_high_water_row_t row = {
        .function_name = report->function_name,
        .value_name = loom_target_compile_report_value_name(
            allocation->module, assignment->value_id),
        .register_class = loom_target_compile_report_value_class_name(
            descriptor_set, assignment->value_class),
        .type_kind = assignment->value_class.type_kind,
        .element_type = assignment->value_class.element_type,
        .assignment_index = entry->assignment_index,
        .origin_operation_name = origin_info.operation_name,
        .origin_kind = origin_info.kind,
        .semantic_tag = origin_info.semantic_tag,
        .start_point = assignment->start_point,
        .end_point = assignment->end_point,
        .required_unit_count = assignment->unit_count,
        .location_kind =
            loom_low_allocation_location_kind_name(assignment->location_kind),
        .location_base = assignment->location_base,
        .location_count = assignment->location_count,
        .high_water_units = entry->high_water_units,
        .lower_free_unit_count = free_runs.free_unit_count,
        .lower_free_run_count = free_runs.free_run_count,
        .lower_largest_free_run_unit_count =
            free_runs.largest_free_run_unit_count,
        .lower_pressure_releasable_free_unit_count =
            pressure_release_free_runs.free_unit_count,
        .lower_pressure_releasable_free_run_count =
            pressure_release_free_runs.free_run_count,
        .lower_pressure_releasable_largest_free_run_unit_count =
            pressure_release_free_runs.largest_free_run_unit_count,
        .active_assignment_blocker_count = blockers.active_assignment_count,
        .active_assignment_blocker_units = blockers.active_assignment_units,
        .active_storage_lease_blocker_count =
            blockers.active_storage_lease_count,
        .active_storage_lease_blocker_units =
            blockers.active_storage_lease_units,
        .active_pressure_storage_lease_blocker_count =
            blockers.active_pressure_storage_lease_count,
        .active_pressure_storage_lease_blocker_units =
            blockers.active_pressure_storage_lease_units,
        .active_fallback_storage_lease_blocker_count =
            blockers.active_fallback_storage_lease_count,
        .active_fallback_storage_lease_blocker_units =
            blockers.active_fallback_storage_lease_units,
    };
    status = loom_target_compile_report_record_allocation_high_water_row(report,
                                                                         &row);
  }

  if (origin_infos != NULL) {
    iree_allocator_free(report->allocator, origin_infos);
  }
  if (scratch != NULL) {
    iree_allocator_free(report->allocator, scratch);
  }
  return status;
}

static loom_target_compile_report_allocation_failure_blocking_kind_t
loom_target_compile_report_allocation_failure_blocking_kind(
    loom_low_allocation_failure_blocking_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET:
      return LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET;
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT:
      return LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT;
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT:
      return LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT;
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION:
      return LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION;
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_UNKNOWN:
    default:
      return LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_UNKNOWN;
  }
}

static iree_string_view_t
loom_target_compile_report_allocation_failure_conflict_value_name(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_failure_t* failure) {
  return failure->conflict_value_id == LOOM_VALUE_ID_INVALID
             ? iree_string_view_empty()
             : loom_target_compile_report_value_name(
                   allocation->module, failure->conflict_value_id);
}

static const loom_block_t*
loom_target_compile_report_allocation_failure_origin_block(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_failure_t* failure, const loom_op_t* origin_op) {
  if (allocation->module != NULL &&
      failure->value_id < allocation->module->values.count) {
    const loom_value_t* value =
        loom_module_value(allocation->module, failure->value_id);
    if (loom_value_is_block_arg(value)) {
      return loom_value_def_block(value);
    }
  }
  return origin_op != NULL ? origin_op->parent_block : NULL;
}

static iree_status_t loom_target_compile_report_record_allocation_failure_rows(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  if (!loom_low_allocation_failure_is_present(&allocation->failure) ||
      !loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    return iree_ok_status();
  }
  const loom_low_allocation_failure_t* failure = &allocation->failure;
  const loom_op_t* origin_op = loom_low_diagnostic_value_origin_op(
      allocation->module, failure->value_id, allocation->function_op);
  const loom_block_t* origin_block =
      loom_target_compile_report_allocation_failure_origin_block(
          allocation, failure, origin_op);
  const loom_target_compile_report_allocation_failure_row_t row = {
      .function_name = report->function_name,
      .value_name = loom_target_compile_report_value_name(allocation->module,
                                                          failure->value_id),
      .register_class = loom_target_compile_report_value_class_name(
          allocation->target.descriptor_set, failure->value_class),
      .type_kind = failure->value_class.type_kind,
      .element_type = failure->value_class.element_type,
      .failure_code = failure->failure_code,
      .blocking_kind =
          loom_target_compile_report_allocation_failure_blocking_kind(
              failure->blocking_kind),
      .origin_operation_name =
          loom_target_compile_report_op_name(allocation->module, origin_op),
      .origin_block_name = loom_target_compile_report_block_name(
          allocation->module, origin_block),
      .start_point = failure->start_point,
      .end_point = failure->end_point,
      .required_unit_count = failure->required_unit_count,
      .budget_units = failure->budget_units,
      .peak_live_units = failure->peak_live_units,
      .location_kind =
          loom_low_allocation_location_kind_name(failure->location_kind),
      .location_base = failure->location_base,
      .location_count = failure->location_count,
      .conflict_assignment_index = failure->conflict_assignment_index,
      .conflict_value_name =
          loom_target_compile_report_allocation_failure_conflict_value_name(
              allocation, failure),
      .conflict_start_point = failure->conflict_start_point,
      .conflict_end_point = failure->conflict_end_point,
      .conflict_location_kind =
          failure->conflict_value_id == LOOM_VALUE_ID_INVALID
              ? iree_string_view_empty()
              : loom_low_allocation_location_kind_name(
                    failure->conflict_location_kind),
      .conflict_location_base = failure->conflict_location_base,
      .conflict_location_count = failure->conflict_location_count,
  };
  return loom_target_compile_report_record_allocation_failure_row(report, &row);
}

static void loom_target_compile_report_record_low_allocation_identity(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  const loom_target_bundle_t* bundle =
      loom_low_resolved_target_bundle(&allocation->target);
  const iree_string_view_t export_symbol = bundle->export_plan->export_symbol;
  report->function_name =
      !iree_string_view_is_empty(export_symbol)
          ? export_symbol
          : loom_low_diagnostic_function_name(allocation->module,
                                              allocation->function_op);
  report->lowered_symbol = loom_low_diagnostic_function_name(
      allocation->module, allocation->function_op);
  report->target_bundle_name = bundle->name;
  report->target_snapshot_name = bundle->snapshot->name;
  report->target_export_name = bundle->export_plan->name;
  report->target_export_symbol = bundle->export_plan->export_symbol;
  report->target_config_name = bundle->config->name;
}

iree_status_t loom_target_compile_report_record_low_allocation_contents(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_table_t* schedule,
    const loom_target_compile_report_low_dynamic_context_t* dynamic_context) {
  const loom_liveness_analysis_t* liveness = &allocation->liveness;
  loom_target_compile_report_record_allocation(
      report, allocation->assignment_count, allocation->spill_count,
      allocation->spill_plan_count, allocation->coalesced_copy_count,
      allocation->materialized_copy_count,
      allocation->storage_leases.record_count,
      allocation->storage_lease_instance_count,
      allocation->storage_release_action_count);
  iree_status_t status = loom_target_compile_report_record_pressure_rows(
      report, liveness, allocation->target.descriptor_set);
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_pressure_origin_rows(
        report, liveness, schedule, allocation->target.descriptor_set);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_schedule_band_rows(
        report, allocation, liveness, schedule, dynamic_context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_spill_rows(report, allocation,
                                                          schedule);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_allocation_failure_rows(
        report, allocation);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_allocation_high_water_rows(
        report, allocation, schedule);
  }
  return status;
}

iree_status_t loom_target_compile_report_record_low_allocation(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  loom_target_compile_report_record_low_allocation_identity(report, allocation);
  return loom_target_compile_report_record_low_allocation_contents(
      report, allocation, /*schedule=*/NULL, /*dynamic_context=*/NULL);
}
