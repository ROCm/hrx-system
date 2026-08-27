// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>

#include "loom/target/math_policy.h"
#include "loom/target/reporting/format_text.h"
#include "loom/target/reporting/schema.h"

static iree_status_t
loom_target_compile_report_append_source_low_memory_storage_fields_text(
    const loom_target_compile_report_string_field_t* fields,
    iree_host_size_t field_count, iree_string_builder_t* builder) {
  const iree_host_size_t first_storage_field =
      loom_target_compile_report_first_non_empty_string_field(fields,
                                                              field_count);
  if (first_storage_field == field_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, " storage={"));
  bool first_field = true;
  for (iree_host_size_t i = first_storage_field; i < field_count; ++i) {
    if (iree_string_view_is_empty(fields[i].value)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%s%s:%.*s", first_field ? "" : ",", fields[i].name,
        (int)fields[i].value.size, fields[i].value.data));
    first_field = false;
  }
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t
loom_target_compile_report_append_source_low_memory_storage_text(
    const loom_target_compile_report_source_low_memory_row_t* row,
    iree_string_builder_t* builder) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_storage_fields(row, fields);
  return loom_target_compile_report_append_source_low_memory_storage_fields_text(
      fields, field_count, builder);
}

static iree_status_t
loom_target_compile_report_append_source_low_memory_strategy_storage_text(
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        summary,
    iree_string_builder_t* builder) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_strategy_storage_fields(
          summary, fields);
  return loom_target_compile_report_append_source_low_memory_storage_fields_text(
      fields, field_count, builder);
}

static iree_status_t
loom_target_compile_report_append_source_low_memory_argument_packet_storage_text(
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        summary,
    iree_string_builder_t* builder) {
  loom_target_compile_report_string_field_t
      fields[LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT];
  const iree_host_size_t field_count =
      loom_target_compile_report_source_low_memory_argument_packet_storage_fields(
          summary, fields);
  return loom_target_compile_report_append_source_low_memory_storage_fields_text(
      fields, field_count, builder);
}

static iree_status_t
loom_target_compile_report_append_source_low_memory_bank_service_text(
    const loom_target_compile_report_bank_service_t* bank_service,
    iree_string_builder_t* builder) {
  if (iree_string_view_is_empty(bank_service->model_key)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " bank_service={proof:%.*s,model:%.*s,revision:%.*s,evidence:%.*s,"
      "request_policy:%.*s,wave_size:%u,bank_count:%u,bank_word_bytes:%u,"
      "packet_bank_words:%u,phase_lane_counts:[",
      (int)bank_service->proof.size, bank_service->proof.data,
      (int)bank_service->model_key.size, bank_service->model_key.data,
      (int)bank_service->model_revision.size, bank_service->model_revision.data,
      (int)bank_service->model_evidence.size, bank_service->model_evidence.data,
      (int)bank_service->request_policy.size, bank_service->request_policy.data,
      bank_service->wave_size, bank_service->bank_count,
      bank_service->bank_word_byte_count, bank_service->packet_word_count));
  for (uint8_t phase = 0; phase < bank_service->phase_count; ++phase) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%s%u", phase == 0 ? "" : ",",
        bank_service->phase_lane_counts[phase]));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "],lane_address_proof:%.*s,active_lane_proof:%.*s,"
      "base_residue_proof:%.*s",
      (int)bank_service->lane_address_proof.size,
      bank_service->lane_address_proof.data,
      (int)bank_service->active_lane_proof.size,
      bank_service->active_lane_proof.data,
      (int)bank_service->base_residue_proof.size,
      bank_service->base_residue_proof.data));
  if (bank_service->base_residue_count != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",base_residue_count:%u", bank_service->base_residue_count));
  }
  if (!iree_string_view_is_empty(bank_service->unknown_reason)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",unknown_reason:%.*s", (int)bank_service->unknown_reason.size,
        bank_service->unknown_reason.data));
  }
  if (!iree_string_view_is_empty(bank_service->classification)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",classification:%.*s,phase_required_rounds:[",
        (int)bank_service->classification.size,
        bank_service->classification.data));
    for (uint8_t phase = 0; phase < bank_service->phase_count; ++phase) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "%s%u", phase == 0 ? "" : ",",
          bank_service->phase_required_rounds[phase]));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "],required_rounds:%u,uncontended_rounds:%u,extra_rounds:%u,"
        "maximum_request_multiplicity:%u",
        bank_service->required_rounds, bank_service->uncontended_rounds,
        bank_service->extra_rounds,
        bank_service->maximum_request_multiplicity));
  }
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t
loom_target_compile_report_append_source_low_memory_subgroup_access_text(
    const loom_target_compile_report_subgroup_access_t* access,
    iree_string_builder_t* builder) {
  if (iree_string_view_is_empty(access->proof)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " subgroup_access={proof:%.*s,lane_address_proof:%.*s,"
      "active_lane_proof:%.*s,lane_mapping:%.*s,subgroup_size:%" PRIu8
      ",per_lane_packet_bytes:%" PRIu32 ",linear_lane_stride_bytes:%" PRIu32
      ",lane_terms:[",
      (int)access->proof.size, access->proof.data,
      (int)access->lane_address_proof.size, access->lane_address_proof.data,
      (int)access->active_lane_proof.size, access->active_lane_proof.data,
      (int)access->lane_mapping.size, access->lane_mapping.data,
      access->subgroup_size, access->per_lane_packet_byte_count,
      access->linear_lane_byte_stride));
  for (uint8_t i = 0; i < access->lane_term_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "%s{divisor:%" PRIu16 ",modulus:%" PRIu16 ",byte_stride:%" PRIu32 "}",
        i == 0 ? "" : ",", access->lane_terms[i].divisor,
        access->lane_terms[i].modulus, access->lane_terms[i].byte_stride));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "]"));
  if (!iree_string_view_is_empty(access->unknown_reason)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",unknown_reason:%.*s", (int)access->unknown_reason.size,
        access->unknown_reason.data));
  }
  if (iree_string_view_equal(access->proof, IREE_SV("exact"))) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",interval_coverage:%.*s,subgroup_requested_bytes:%" PRIu64
        ",subgroup_unique_bytes:%" PRIu64 ",subgroup_span_bytes:%" PRIu64
        ",maximum_adjacent_lane_delta_bytes:%" PRIu64
        ",maximum_uncovered_gap_bytes:%" PRIu64
        ",distinct_lane_addresses:%" PRIu16 ",contiguous_regions:%" PRIu16,
        (int)access->interval_coverage.size, access->interval_coverage.data,
        access->subgroup_requested_byte_count,
        access->subgroup_unique_byte_count, access->subgroup_span_byte_count,
        access->maximum_adjacent_lane_delta_bytes,
        access->maximum_uncovered_byte_gap_bytes,
        access->distinct_lane_address_count, access->contiguous_region_count));
  }
  return iree_string_builder_append_cstring(builder, "}");
}

iree_status_t
loom_target_compile_report_append_bank_service_summary_text_fields(
    const loom_target_compile_report_bank_service_summary_t* summary,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_format(
      builder,
      " modeled_packets=%" PRIu64 " exact_packets=%" PRIu64
      " unknown_packets=%" PRIu64 " conflict_free_packets=%" PRIu64
      " conflicted_packets=%" PRIu64 " structural_required_rounds=%" PRIu64
      " structural_uncontended_rounds=%" PRIu64
      " structural_extra_rounds=%" PRIu64
      " maximum_request_multiplicity=%" PRIu16 " dynamic_exact_packets=%" PRIu64
      " dynamic_unknown_packets=%" PRIu64 " dynamic_packets=%" PRIu64
      " dynamic_required_rounds=%" PRIu64 " dynamic_uncontended_rounds=%" PRIu64
      " dynamic_extra_rounds=%" PRIu64,
      summary->modeled_packet_count, summary->exact_packet_count,
      summary->unknown_packet_count, summary->conflict_free_packet_count,
      summary->conflicted_packet_count, summary->required_round_count,
      summary->uncontended_round_count, summary->extra_round_count,
      summary->maximum_request_multiplicity,
      summary->exact_dynamic_packet_count,
      summary->unknown_dynamic_packet_count, summary->dynamic_packet_count,
      summary->dynamic_required_round_count,
      summary->dynamic_uncontended_round_count,
      summary->dynamic_extra_round_count);
}

iree_status_t
loom_target_compile_report_append_subgroup_access_summary_text_fields(
    const loom_target_compile_report_subgroup_access_summary_t* summary,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_format(
      builder,
      " modeled_packets=%" PRIu64 " exact_packets=%" PRIu64
      " unknown_packets=%" PRIu64 " dense_packets=%" PRIu64
      " gapped_packets=%" PRIu64 " overlapping_packets=%" PRIu64
      " dynamic_exact_packets=%" PRIu64 " dynamic_unknown_packets=%" PRIu64
      " dynamic_packets=%" PRIu64 " dynamic_dense_packets=%" PRIu64
      " dynamic_gapped_packets=%" PRIu64
      " dynamic_overlapping_packets=%" PRIu64,
      summary->modeled_packet_count, summary->exact_packet_count,
      summary->unknown_packet_count, summary->dense_packet_count,
      summary->gapped_packet_count, summary->overlapping_packet_count,
      summary->exact_dynamic_packet_count,
      summary->unknown_dynamic_packet_count, summary->dynamic_packet_count,
      summary->dynamic_dense_packet_count, summary->dynamic_gapped_packet_count,
      summary->dynamic_overlapping_packet_count);
}

static iree_status_t loom_target_compile_report_append_memory_interval_text(
    const loom_target_compile_report_memory_interval_t* interval,
    iree_string_builder_t* builder) {
  if (!loom_target_compile_report_memory_interval_has_range(interval)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " source_interval={begin_min_bytes:%" PRId64 ",begin_max_bytes:%" PRId64
      ",end_min_bytes:%" PRId64 ",end_max_bytes:%" PRId64,
      interval->begin_min_bytes, interval->begin_max_bytes,
      interval->end_min_bytes, interval->end_max_bytes));
  if (iree_all_bits_set(
          interval->flags,
          LOOM_TARGET_COMPILE_REPORT_MEMORY_INTERVAL_EXACT_LENGTH)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",exact_length_bytes:%" PRIu64, interval->exact_length_bytes));
  }
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t
loom_target_compile_report_append_memory_interval_summary_text(
    const char* field_name,
    const loom_target_compile_report_memory_interval_summary_t* summary,
    iree_string_builder_t* builder) {
  if (summary->packet_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " %s={packets:%" PRIu64 ",begin_min_bytes:%" PRId64
      ",end_max_bytes:%" PRId64 ",byte_count:%" PRIu64,
      field_name, summary->packet_count, summary->envelope_begin_min_bytes,
      summary->envelope_end_max_bytes, summary->envelope_byte_count));
  if (summary->exact_static_packet_count != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",exact_static_packet_count:%" PRIu64,
        summary->exact_static_packet_count));
  }
  if (summary->exact_symbolic_packet_count != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",exact_symbolic_packet_count:%" PRIu64,
        summary->exact_symbolic_packet_count));
  }
  if (summary->exact_static_packet_count +
          summary->exact_symbolic_packet_count ==
      summary->packet_count) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",unique_byte_count:%" PRIu64, summary->unique_byte_count));
  }
  return iree_string_builder_append_cstring(builder, "}");
}

iree_status_t loom_target_compile_report_format_text_source_low_memory_summary(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " packets=%" PRIu64 " loads=%" PRIu64 " stores=%" PRIu64
      " scalar_packets=%" PRIu64 " vector_packets=%" PRIu64
      " source_lanes=%" PRIu64 " source_bytes=%" PRIu64 " read_bytes=%" PRIu64
      " write_bytes=%" PRIu64 " issued_read_bytes=%" PRIu64
      " issued_write_bytes=%" PRIu64 " issued_read_unknown_widths=%" PRIu64
      " issued_write_unknown_widths=%" PRIu64
      " contiguous_vector_packets=%" PRIu64 " strided_vector_packets=%" PRIu64
      " unknown_stride_vector_packets=%" PRIu64,
      summary->packet_count, summary->load_packet_count,
      summary->store_packet_count, summary->scalar_packet_count,
      summary->vector_packet_count, summary->source_lane_count,
      summary->source_byte_count, summary->read_byte_count,
      summary->write_byte_count, summary->issued_read_byte_count,
      summary->issued_write_byte_count,
      summary->issued_read_unknown_width_count,
      summary->issued_write_unknown_width_count,
      summary->contiguous_vector_packet_count,
      summary->strided_vector_packet_count,
      summary->unknown_stride_vector_packet_count));
  if (loom_target_compile_report_source_low_memory_should_print_dynamic(
          summary, workload)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        " exact_dynamic_packets=%" PRIu64 " unknown_dynamic_packets=%" PRIu64
        " dynamic_packets=%" PRIu64 " dynamic_source_bytes=%" PRIu64
        " dynamic_read_bytes=%" PRIu64 " dynamic_write_bytes=%" PRIu64
        " dynamic_issued_read_bytes=%" PRIu64
        " dynamic_issued_write_bytes=%" PRIu64
        " dynamic_issued_read_unknown_widths=%" PRIu64
        " dynamic_issued_write_unknown_widths=%" PRIu64,
        summary->exact_dynamic_packet_count,
        summary->unknown_dynamic_packet_count, summary->dynamic_packet_count,
        summary->dynamic_source_byte_count, summary->dynamic_read_byte_count,
        summary->dynamic_write_byte_count,
        summary->dynamic_issued_read_byte_count,
        summary->dynamic_issued_write_byte_count,
        summary->dynamic_issued_read_unknown_width_count,
        summary->dynamic_issued_write_unknown_width_count));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
    loom_target_compile_report_dispatch_memory_bytes_t dispatch_source = {0};
    if (loom_target_compile_report_source_low_memory_dispatch_source_bytes(
            summary, workload, &dispatch_source)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " dispatch_source_read_bytes=%" PRIu64
          " dispatch_source_write_bytes=%" PRIu64
          " dispatch_source_total_bytes=%" PRIu64,
          dispatch_source.read_byte_count, dispatch_source.write_byte_count,
          dispatch_source.total_byte_count));
    }
    loom_target_compile_report_dispatch_memory_bytes_t dispatch_issued = {0};
    if (loom_target_compile_report_source_low_memory_dispatch_issued_bytes(
            summary, workload, &dispatch_issued)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " dispatch_issued_read_bytes=%" PRIu64
          " dispatch_issued_write_bytes=%" PRIu64
          " dispatch_issued_total_bytes=%" PRIu64,
          dispatch_issued.read_byte_count, dispatch_issued.write_byte_count,
          dispatch_issued.total_byte_count));
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_append_memory_interval_summary_text(
          "interval_envelope", &summary->interval_envelope, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_append_memory_interval_summary_text(
          "read_interval_envelope", &summary->read_interval_envelope, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_append_memory_interval_summary_text(
          "write_interval_envelope", &summary->write_interval_envelope,
          builder));
  return iree_ok_status();
}

static bool
loom_target_compile_report_source_low_selection_summary_has_dynamic_delta(
    const loom_target_compile_report_source_low_selection_summary_t* row) {
  return row->unknown_dynamic_op_count != 0 ||
         row->dynamic_selected_op_count != row->selected_op_count ||
         row->dynamic_emitted_low_op_count != row->emitted_low_op_count;
}

static iree_status_t loom_target_compile_report_append_source_low_descriptor(
    iree_string_view_t descriptor_key,
    iree_string_view_t descriptor_semantic_tag,
    iree_string_builder_t* builder) {
  if (!iree_string_view_is_empty(descriptor_key)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " descriptor_key=%.*s", (int)descriptor_key.size,
        descriptor_key.data));
  }
  if (!iree_string_view_is_empty(descriptor_semantic_tag)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " descriptor_semantic_tag=%.*s",
        (int)descriptor_semantic_tag.size, descriptor_semantic_tag.data));
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_append_source_low_descriptor_fields(
    const loom_target_compile_report_source_low_row_t* row,
    iree_string_builder_t* builder) {
  return loom_target_compile_report_append_source_low_descriptor(
      row->descriptor_key, row->descriptor_semantic_tag, builder);
}

static iree_status_t
loom_target_compile_report_append_source_low_selection_summary_descriptor(
    const loom_target_compile_report_source_low_selection_summary_t* row,
    iree_string_builder_t* builder) {
  return loom_target_compile_report_append_source_low_descriptor(
      row->descriptor_key, row->descriptor_semantic_tag, builder);
}

static iree_status_t
loom_target_compile_report_format_source_low_selection_summary_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_selection_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_selection_summary_t* rows =
        (const loom_target_compile_report_source_low_selection_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_selection_summary_t* row =
          &rows[i];
      if (!loom_target_compile_report_source_low_selection_summary_has_dynamic_delta(
              row)) {
        continue;
      }
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_text_non_empty(row->source_op_name);
      const iree_string_view_t selection_name =
          loom_target_compile_report_source_low_selection_name(
              row->selection_kind);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_selection[%" PRIhsz
          "] function=%.*s source_op=%.*s selection=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data,
          (int)selection_name.size, selection_name.data));
      if (!iree_string_view_is_empty(row->plan_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " plan_key=%.*s", (int)row->plan_key.size,
            row->plan_key.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_selection_summary_descriptor(
              row, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " selected_ops=%" PRIu64 " emitted_ops=%" PRIu64,
          row->selected_op_count, row->emitted_low_op_count));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " exact_dynamic_ops=%" PRIu64 " unknown_dynamic_ops=%" PRIu64
          " dynamic_selected_ops=%" PRIu64 " dynamic_emitted_ops=%" PRIu64,
          row->exact_dynamic_op_count, row->unknown_dynamic_op_count,
          row->dynamic_selected_op_count, row->dynamic_emitted_low_op_count));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_append_native_contraction_role_text(
    iree_string_view_t role_name,
    const loom_native_contraction_role_facts_t* facts,
    iree_string_builder_t* builder) {
  const iree_string_view_t evidence =
      loom_target_compile_report_native_layout_evidence_name(facts->evidence);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "%.*s:{evidence:%.*s,bits:%u,registers:%u,payload:%u,positions:%u,"
      "coordinates:%u",
      (int)role_name.size, role_name.data, (int)evidence.size, evidence.data,
      facts->element_bit_count, facts->register_count,
      facts->payload_element_count, facts->physical_position_count,
      facts->logical_coordinate_count));
  if (facts->evidence == LOOM_NATIVE_LAYOUT_EVIDENCE_EXACT) {
    if (facts->owner_multiplicity_minimum ==
        facts->owner_multiplicity_maximum) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ",owners:%u", facts->owner_multiplicity_minimum));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ",owners:%u..%u", facts->owner_multiplicity_minimum,
          facts->owner_multiplicity_maximum));
    }
  }
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t loom_target_compile_report_append_native_contraction_text(
    const loom_native_contraction_facts_t* facts,
    iree_string_builder_t* builder) {
  if (facts == NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " native_contraction={tile:%ux%ux%ux%u,participants:%u,",
      facts->shape.block_count, facts->shape.result_row_count,
      facts->shape.result_column_count, facts->shape.reduction_count,
      facts->participant_count));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_append_native_contraction_role_text(
          IREE_SV("lhs"), &facts->lhs, builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_append_native_contraction_role_text(
          IREE_SV("rhs"), &facts->rhs, builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_append_native_contraction_role_text(
          IREE_SV("accumulator"), &facts->accumulator, builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_append_native_contraction_role_text(
          IREE_SV("result"), &facts->result, builder));
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t loom_target_compile_report_append_native_transition_text(
    const loom_target_compile_report_source_low_row_t* row,
    iree_string_builder_t* builder) {
  const loom_native_transition_facts_t* facts = row->native_transition_facts;
  if (facts == NULL) {
    return iree_ok_status();
  }
  const iree_string_view_t source_role =
      loom_target_compile_report_native_contraction_role_name(
          facts->source_role);
  const iree_string_view_t destination_role =
      loom_target_compile_report_native_contraction_role_name(
          facts->destination_role);
  const iree_string_view_t source_type =
      loom_target_compile_report_scalar_type_name(
          row->native_transition_source_type);
  const iree_string_view_t destination_type =
      loom_target_compile_report_scalar_type_name(
          row->native_transition_destination_type);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " native_transition={%.*s:%.*s->%.*s:%.*s,positions:%u,"
      "participant_changes:%u,local_position_changes:%u,replication:",
      (int)source_role.size, source_role.data, (int)source_type.size,
      source_type.data, (int)destination_role.size, destination_role.data,
      (int)destination_type.size, destination_type.data,
      facts->destination_position_count, facts->participant_change_count,
      facts->local_position_change_count));
  if (facts->destination_positions_per_source_minimum ==
      facts->destination_positions_per_source_maximum) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%u", facts->destination_positions_per_source_minimum));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%u..%u", facts->destination_positions_per_source_minimum,
        facts->destination_positions_per_source_maximum));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",source_owner:["));
  for (uint8_t i = 0; i < facts->source_owner_factor_count; ++i) {
    const loom_native_transition_owner_factor_t* factor =
        &facts->source_owner_factors[i];
    const iree_string_view_t destination_dimension =
        loom_target_compile_report_native_physical_dimension_name(
            factor->destination_dimension);
    const iree_string_view_t source_owner_dimension =
        loom_target_compile_report_native_physical_dimension_name(
            factor->source_owner_dimension);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%s%.*s+=(%.*s/%u", i == 0 ? "" : ",",
        (int)source_owner_dimension.size, source_owner_dimension.data,
        (int)destination_dimension.size, destination_dimension.data,
        factor->destination_divisor));
    if (factor->destination_modulus != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "%%%u", factor->destination_modulus));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ")*%u", factor->source_owner_multiplier));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}

static iree_status_t loom_target_compile_report_format_source_low_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_row_t* rows =
        (const loom_target_compile_report_source_low_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_text_non_empty(row->source_op_name);
      const iree_string_view_t selection_name =
          loom_target_compile_report_source_low_selection_name(
              row->selection_kind);
      const iree_string_view_t plan_key = row->plan_key;
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low[%" PRIhsz
          "] function=%.*s source_op=%.*s selection=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data,
          (int)selection_name.size, selection_name.data));
      if (!iree_string_view_is_empty(plan_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " plan_key=%.*s", (int)plan_key.size, plan_key.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_descriptor_fields(
              row, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " emitted_ops=%u", row->emitted_low_op_count));
      if (row->execution_count_plus_one !=
              LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_EXECUTION_COUNT_PLUS_ONE_UNKNOWN &&
          row->execution_count_plus_one != 2) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " execution_count=%" PRIu64,
            row->execution_count_plus_one - 1));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
      if (row->native_contraction_facts != NULL ||
          row->native_transition_facts != NULL) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "COMPILE-REPORT: source_low_native[%" PRIhsz
            "] function=%.*s source_op=%.*s",
            row_index, (int)function_name.size, function_name.data,
            (int)source_op_name.size, source_op_name.data));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_native_contraction_text(
                row->native_contraction_facts, builder));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_native_transition_text(row,
                                                                     builder));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_source_low_target_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_target_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_target_row_t* rows =
        (const loom_target_compile_report_source_low_target_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_target_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t target_symbol_name =
          loom_target_compile_report_text_non_empty(row->target_symbol_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_text_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_text_non_empty(row->target_config_name);
      const iree_string_view_t target_source =
          loom_target_binding_source_name(row->target_source);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_target[%" PRIhsz
          "] function=%.*s source=%.*s target=%.*s bundle=%.*s config=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)target_source.size, target_source.data,
          (int)target_symbol_name.size, target_symbol_name.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_config_name.size, target_config_name.data));
      if (row->target_subgroup_size != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " subgroup_size=%u", row->target_subgroup_size));
      }
      if (row->candidate_target_count != 0) {
        const iree_string_view_t candidate_symbol_name =
            loom_target_compile_report_text_non_empty(
                row->candidate_target_symbol_name);
        const iree_string_view_t candidate_bundle_name =
            loom_target_compile_report_text_non_empty(
                row->candidate_target_bundle_name);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            " candidate_targets=%u first_candidate=%.*s "
            "first_candidate_bundle=%.*s",
            row->candidate_target_count, (int)candidate_symbol_name.size,
            candidate_symbol_name.data, (int)candidate_bundle_name.size,
            candidate_bundle_name.data));
        if (row->candidate_target_subgroup_size != 0) {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, " first_candidate_subgroup_size=%u",
              row->candidate_target_subgroup_size));
        }
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_transform_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  if (report->source_low_transform_rows.count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "COMPILE-REPORT: source_low_transforms rows=%" PRIhsz "\n",
      report->source_low_transform_rows.count));
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_transform_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_transform_row_t* rows =
        (const loom_target_compile_report_source_low_transform_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_transform_row_t* row =
          &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_text_non_empty(row->source_op_name);
      const iree_string_view_t transform_key =
          loom_target_compile_report_text_non_empty(row->transform_key);
      const iree_string_view_t outcome =
          loom_target_compile_report_text_non_empty(row->outcome);
      const iree_string_view_t reason =
          loom_target_compile_report_text_non_empty(row->reason);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_transform[%" PRIhsz
          "] function=%.*s source_op=%.*s transform=%.*s outcome=%.*s "
          "reason=%.*s candidates=%u selected=%u removed_loop_carried=%u",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data,
          (int)transform_key.size, transform_key.data, (int)outcome.size,
          outcome.data, (int)reason.size, reason.data,
          row->candidate_value_count, row->selected_value_count,
          row->removed_loop_carried_value_count));
      if (row->removed_loop_carried_payload_register_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " removed_payload_registers=%" PRIu64,
            row->removed_loop_carried_payload_register_count));
      }
      if (row->block_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " shape=%" PRIu64 "x%" PRIu64 "x%" PRIu64,
            row->block_count, row->row_count, row->column_count));
      } else if (row->row_count != 0 || row->column_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " shape=%" PRIu64 "x%" PRIu64, row->row_count,
            row->column_count));
      }
      if (row->workgroup_memory_byte_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " workgroup_bytes=%" PRIu64,
            row->workgroup_memory_byte_count));
      }
      if (row->inserted_load_op_count != 0 ||
          row->inserted_store_op_count != 0 ||
          row->inserted_barrier_op_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " inserted_loads=%u inserted_stores=%u barriers=%u",
            row->inserted_load_op_count, row->inserted_store_op_count,
            row->inserted_barrier_op_count));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_source_low_memory_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_row_t* rows =
        (const loom_target_compile_report_source_low_memory_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_text_non_empty(row->source_op_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_text_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_text_non_empty(row->memory_space);
      const iree_string_view_t operation_kind =
          loom_target_compile_report_text_non_empty(row->operation_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_text_non_empty(row->packet_key);
      const iree_string_view_t strategy_key =
          loom_target_compile_report_text_non_empty(row->strategy_key);
      const iree_string_view_t address_form =
          loom_target_compile_report_text_non_empty(row->address_form);
      const iree_string_view_t dynamic_term_kind =
          loom_target_compile_report_text_non_empty(row->dynamic_term_kind);
      const iree_string_view_t fallback_reason =
          loom_target_compile_report_text_non_empty(row->fallback_reason);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory[%" PRIhsz
          "] function=%.*s source_op=%.*s memory_space=%.*s operation=%.*s "
          "packet=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data, (int)memory_space.size,
          memory_space.data, (int)operation_kind.size, operation_kind.data,
          (int)packet_key.size, packet_key.data));
      if (!iree_string_view_is_empty(row->source_root_name)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root=%.*s", (int)source_root_name.size,
            source_root_name.data));
      }
      if (row->source_root_argument_index != UINT16_MAX) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root_argument_index=%u",
            row->source_root_argument_index));
      }
      if (!iree_string_view_is_empty(row->strategy_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " strategy=%.*s", (int)strategy_key.size,
            strategy_key.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_storage_text(
              row, builder));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_bank_service_text(
              &row->bank_service, builder));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_subgroup_access_text(
              &row->subgroup_access, builder));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_memory_interval_text(
              &row->source_interval, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " address_form=%.*s dynamic_term_kind=%.*s "
          "fallback_reason=%.*s static_offset_bytes=%" PRId64
          " element_bytes=%u "
          "vector_lanes=%u issued_read_bytes=%u issued_write_bytes=%u "
          "issued_read_unknown_widths=%u issued_write_unknown_widths=%u "
          "dynamic_stride_bytes=%u "
          "vector_lane_stride_bytes=%u\n",
          (int)address_form.size, address_form.data,
          (int)dynamic_term_kind.size, dynamic_term_kind.data,
          (int)fallback_reason.size, fallback_reason.data,
          row->static_offset_bytes, row->element_byte_count,
          row->vector_lane_count, row->issued_read_byte_count,
          row->issued_write_byte_count, row->issued_read_unknown_width_count,
          row->issued_write_unknown_width_count, row->dynamic_stride_bytes,
          row->vector_lane_stride_bytes));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_root_summaries(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_root_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_root_summary_t* rows =
        (const loom_target_compile_report_source_low_memory_root_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_root_summary_t* row =
          &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_text_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_text_non_empty(row->memory_space);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory_root[%" PRIhsz
          "] function=%.*s source_root=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)source_root_name.size, source_root_name.data));
      if (row->source_root_argument_index != UINT16_MAX) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root_argument_index=%u",
            row->source_root_argument_index));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " memory_space=%.*s", (int)memory_space.size,
          memory_space.data));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_text_source_low_memory_summary(
              &row->summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_argument_summaries(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_argument_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_argument_summary_t* rows =
        (const loom_target_compile_report_source_low_memory_argument_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_argument_summary_t*
          row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_text_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_text_non_empty(row->memory_space);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory_argument[%" PRIhsz
          "] function=%.*s",
          row_index, (int)function_name.size, function_name.data));
      if (!iree_string_view_is_empty(row->source_root_name)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root=%.*s", (int)source_root_name.size,
            source_root_name.data));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " source_root_argument_index=%u memory_space=%.*s",
          row->source_root_argument_index, (int)memory_space.size,
          memory_space.data));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_text_source_low_memory_summary(
              &row->summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_argument_packet_summaries(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_argument_packet_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        rows =
            (const loom_target_compile_report_source_low_memory_argument_packet_summary_t*)
                loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
          row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_text_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_text_non_empty(row->memory_space);
      const iree_string_view_t operation_kind =
          loom_target_compile_report_text_non_empty(row->operation_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_text_non_empty(row->packet_key);
      const iree_string_view_t strategy_key =
          loom_target_compile_report_text_non_empty(row->strategy_key);
      const iree_string_view_t fallback_reason =
          loom_target_compile_report_text_non_empty(row->fallback_reason);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory_argument_packet[%" PRIhsz
          "] function=%.*s",
          row_index, (int)function_name.size, function_name.data));
      if (!iree_string_view_is_empty(row->source_root_name)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root=%.*s", (int)source_root_name.size,
            source_root_name.data));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " source_root_argument_index=%u memory_space=%.*s operation=%.*s "
          "packet=%.*s",
          row->source_root_argument_index, (int)memory_space.size,
          memory_space.data, (int)operation_kind.size, operation_kind.data,
          (int)packet_key.size, packet_key.data));
      if (!iree_string_view_is_empty(row->strategy_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " strategy=%.*s", (int)strategy_key.size,
            strategy_key.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_argument_packet_storage_text(
              row, builder));
      if (!iree_string_view_is_empty(row->fallback_reason)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " fallback_reason=%.*s", (int)fallback_reason.size,
            fallback_reason.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_text_source_low_memory_summary(
              &row->summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_memory_strategy_summaries(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_memory_strategy_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_strategy_summary_t* rows =
        (const loom_target_compile_report_source_low_memory_strategy_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_memory_strategy_summary_t*
          row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_text_non_empty(row->memory_space);
      const iree_string_view_t operation_kind =
          loom_target_compile_report_text_non_empty(row->operation_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_text_non_empty(row->packet_key);
      const iree_string_view_t strategy_key =
          loom_target_compile_report_text_non_empty(row->strategy_key);
      const iree_string_view_t fallback_reason =
          loom_target_compile_report_text_non_empty(row->fallback_reason);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory_strategy[%" PRIhsz
          "] function=%.*s memory_space=%.*s operation=%.*s packet=%.*s "
          "strategy=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)memory_space.size, memory_space.data, (int)operation_kind.size,
          operation_kind.data, (int)packet_key.size, packet_key.data,
          (int)strategy_key.size, strategy_key.data));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_strategy_storage_text(
              row, builder));
      if (!iree_string_view_is_empty(row->fallback_reason)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " fallback_reason=%.*s", (int)fallback_reason.size,
            fallback_reason.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_text_source_low_memory_summary(
              &row->summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_bank_service_summaries(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_bank_service_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_bank_service_summary_t* rows =
        (const loom_target_compile_report_source_low_bank_service_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_bank_service_summary_t* row =
          &rows[i];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_bank_service[%" PRIhsz
          "] function=%.*s source_op=%.*s source_op_kind=%" PRIu32,
          row_index, (int)row->function_name.size, row->function_name.data,
          (int)row->source_op_name.size, row->source_op_name.data,
          row->source_op_kind));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("source_root"), row->source_root_name));
      if (row->source_root_argument_index != UINT16_MAX) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root_argument=%" PRIu16,
            row->source_root_argument_index));
      }
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("memory_space"), row->memory_space));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("operation"), row->operation_kind));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("packet"), row->packet_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("strategy"), row->strategy_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("model"), row->model_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("model_revision"), row->model_revision));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("model_evidence"), row->model_evidence));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("request_policy"), row->request_policy));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " wave_size=%" PRIu8 " banks=%" PRIu8 " bank_word_bytes=%" PRIu8
          " packet_bank_words=%" PRIu8,
          row->wave_size, row->bank_count, row->bank_word_byte_count,
          row->packet_word_count));
      if (row->summary.unknown_packet_count != 0) {
        const iree_string_view_t unknown_reason =
            row->has_mixed_unknown_reasons
                ? IREE_SV("mixed")
                : loom_target_compile_report_text_non_empty(
                      row->unknown_reason);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " unknown_reason=%.*s", (int)unknown_reason.size,
            unknown_reason.data));
      }
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_bank_service_summary_text_fields(
              &row->summary, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_compile_report_format_source_low_subgroup_access_summaries(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->source_low_subgroup_access_summaries.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_subgroup_access_summary_t* rows =
        (const loom_target_compile_report_source_low_subgroup_access_summary_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_source_low_subgroup_access_summary_t*
          row = &rows[i];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_subgroup_access[%" PRIhsz
          "] function=%.*s source_op=%.*s source_op_kind=%" PRIu32,
          row_index, (int)row->function_name.size, row->function_name.data,
          (int)row->source_op_name.size, row->source_op_name.data,
          row->source_op_kind));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("source_root"), row->source_root_name));
      if (row->source_root_argument_index != UINT16_MAX) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " source_root_argument=%" PRIu16,
            row->source_root_argument_index));
      }
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("memory_space"), row->memory_space));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("operation"), row->operation_kind));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("packet"), row->packet_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
          builder, IREE_SV("strategy"), row->strategy_key));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_subgroup_access_text(
              &row->access, builder));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_subgroup_access_summary_text_fields(
              &row->summary, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_math_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->math_legalization_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_math_row_t* rows =
        (const loom_target_compile_report_math_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_math_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_text_non_empty(row->source_op_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_text_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_text_non_empty(row->target_config_name);
      const iree_string_view_t policy_name =
          loom_target_compile_report_text_non_empty(row->policy_name);
      const iree_string_view_t constraint_key =
          loom_target_compile_report_text_non_empty(row->constraint_key);
      const iree_string_view_t action_name =
          loom_target_compile_report_math_action_name(row->action);
      const iree_string_view_t math_op_name =
          loom_target_math_op_name((loom_target_math_op_t)row->math_op);
      const iree_string_view_t lane_domain_name =
          loom_target_math_lane_domain_name(
              (loom_target_math_lane_domain_t)row->lane_domain);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t recipe_name =
          loom_target_math_recipe_name((loom_target_math_recipe_t)row->recipe);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: math_legalization[%" PRIhsz
          "] function=%.*s source_op=%.*s action=%.*s policy=%.*s "
          "math_op=%.*s domain=%.*s element=%.*s recipe=%.*s "
          "constraint=%.*s bundle=%.*s config=%.*s source_fastmath=0x%02x "
          "recipe_fastmath=0x%02x created_ops=%" PRIu64 " erased_ops=%" PRIu64
          "\n",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data, (int)action_name.size,
          action_name.data, (int)policy_name.size, policy_name.data,
          (int)math_op_name.size, math_op_name.data, (int)lane_domain_name.size,
          lane_domain_name.data, (int)element_type_name.size,
          element_type_name.data, (int)recipe_name.size, recipe_name.data,
          (int)constraint_key.size, constraint_key.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_config_name.size, target_config_name.data,
          (uint32_t)row->source_fastmath_flags,
          (uint32_t)row->recipe_fastmath_flags, row->created_op_count,
          row->erased_op_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_legalization_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->target_legalization_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_legalization_row_t* rows =
        (const loom_target_compile_report_legalization_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_legalization_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_text_non_empty(row->source_op_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_text_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_text_non_empty(row->target_config_name);
      const iree_string_view_t legalizer_name =
          loom_target_compile_report_text_non_empty(row->legalizer_name);
      const iree_string_view_t strategy_name =
          loom_target_compile_report_legalizer_strategy_name(
              row->legalizer_strategy);
      const iree_string_view_t mode_name =
          loom_target_compile_report_legalization_mode_name(row->mode);
      const iree_string_view_t policy_name =
          loom_target_compile_report_legalization_policy_name(row->policy);
      const iree_string_view_t action_name =
          loom_target_compile_report_legalization_action_name(row->action);
      const iree_string_view_t legalization_outcome_name =
          loom_target_compile_report_legalization_outcome_name(
              row->legalization_outcome);
      const iree_string_view_t outcome_name =
          loom_target_compile_report_contract_outcome_name(
              row->contract_outcome);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: target_legalization[%" PRIhsz
          "] function=%.*s source_op=%.*s mode=%.*s policy=%.*s "
          "action=%.*s outcome=%.*s contract=%.*s legalizer=%.*s strategy=%.*s "
          "bundle=%.*s config=%.*s",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data, (int)mode_name.size,
          mode_name.data, (int)policy_name.size, policy_name.data,
          (int)action_name.size, action_name.data,
          (int)legalization_outcome_name.size, legalization_outcome_name.data,
          (int)outcome_name.size, outcome_name.data, (int)legalizer_name.size,
          legalizer_name.data, (int)strategy_name.size, strategy_name.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_config_name.size, target_config_name.data));
      if (!iree_string_view_is_empty(row->descriptor_key)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " descriptor_key=%.*s", (int)row->descriptor_key.size,
            row->descriptor_key.data));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " source_rejections=0x%08" PRIx32 " source_rejection_detail=%" PRIu32
          " target_rejections=0x%08" PRIx32 " missing_features=0x%08" PRIx32
          " missing_facts=0x%08" PRIx32 " created_ops=%" PRIu64
          " erased_ops=%" PRIu64 "\n",
          row->source_rejection_bits, row->source_rejection_detail,
          row->target_rejection_bits, row->missing_feature_bits,
          row->missing_fact_bits, row->created_op_count, row->erased_op_count));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_target_compile_report_format_text_lowering_details(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_math_rows(report, builder));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_source_low_target_rows(
      report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_selection_summary_rows(
          report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_rows(report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_transform_rows(report,
                                                                  builder));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_source_low_memory_rows(
      report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_root_summaries(
          report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_argument_summaries(
          report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_argument_packet_summaries(
          report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_memory_strategy_summaries(
          report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_bank_service_summaries(
          report, builder));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_source_low_subgroup_access_summaries(
          report, builder));
  return loom_target_compile_report_format_legalization_rows(report, builder);
}
