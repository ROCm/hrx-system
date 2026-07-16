// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>

#include "loom/ir/scalar_type.h"
#include "loom/target/math_policy.h"
#include "loom/target/reporting/compile_report_format.h"
#include "loom/target/reporting/compile_report_planning_format.h"
#include "loom/target/reporting/compile_report_schema.h"

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

static iree_status_t
loom_target_compile_report_append_source_low_memory_summary_text(
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

static iree_string_view_t loom_target_compile_report_non_empty(
    iree_string_view_t value) {
  return iree_string_view_is_empty(value) ? IREE_SV("-") : value;
}

static iree_status_t loom_target_compile_report_append_optional_u32(
    iree_string_builder_t* builder, uint32_t value) {
  if (value == UINT32_MAX) {
    return iree_string_builder_append_string(builder, IREE_SV("-"));
  }
  return iree_string_builder_append_format(builder, "%u", value);
}

static void loom_target_compile_report_move_cause_totals(
    const loom_target_compile_report_t* report, uint64_t* out_kind_count,
    uint64_t* out_packet_count, uint64_t* out_unit_count) {
  loom_target_compile_report_move_cause_counts_totals(
      report->move_causes, out_kind_count, out_packet_count, out_unit_count);
}

static iree_status_t loom_target_compile_report_append_string_field(
    iree_string_builder_t* builder, iree_string_view_t name,
    iree_string_view_t value) {
  value = loom_target_compile_report_non_empty(value);
  return iree_string_builder_append_format(builder, " %.*s=%.*s",
                                           (int)name.size, name.data,
                                           (int)value.size, value.data);
}

static iree_status_t loom_target_compile_report_append_workload_fields(
    iree_string_builder_t* builder,
    const loom_target_compile_report_workload_t* workload) {
  if (iree_any_bit_set(workload->flags,
                       LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " workgroup_size=%" PRIu32 "x%" PRIu32 "x%" PRIu32,
        workload->workgroup_size.x, workload->workgroup_size.y,
        workload->workgroup_size.z));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " flat_workgroup_size=%" PRIu64,
        workload->flat_workgroup_size));
  }
  if (iree_any_bit_set(workload->flags,
                       LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_COUNT)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " workgroup_count=%" PRIu32 "x%" PRIu32 "x%" PRIu32,
        workload->workgroup_count.x, workload->workgroup_count.y,
        workload->workgroup_count.z));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKGROUP_COUNT)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " dispatch_workgroup_count=%" PRIu64,
        workload->dispatch_workgroup_count));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " dispatch_workitem_count=%" PRIu64,
        workload->dispatch_workitem_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_append_instruction_mix_fields(
    iree_string_builder_t* builder, iree_string_view_t name,
    const loom_target_compile_report_static_instruction_mix_t* mix) {
  return iree_string_builder_append_format(
      builder,
      "COMPILE-REPORT: %.*s descriptors=%" PRIu64 " unknown=%" PRIu64
      " scalar_alu=%" PRIu64 " vector_alu=%" PRIu64 " matrix=%" PRIu64
      " mfma=%" PRIu64 " smfmac=%" PRIu64 " wmma=%" PRIu64 " swmmac=%" PRIu64
      " dot=%" PRIu64 " global_memory=%" PRIu64 " global_load=%" PRIu64
      " global_store=%" PRIu64 " buffer_load=%" PRIu64 " buffer_store=%" PRIu64
      " flat_memory=%" PRIu64 " local_memory=%" PRIu64 " scalar_memory=%" PRIu64
      " private_memory=%" PRIu64 " generic_memory=%" PRIu64
      " memory_read_unknown_width=%" PRIu64
      " memory_write_unknown_width=%" PRIu64 " memory_read_bytes=%" PRIu64
      " memory_write_bytes=%" PRIu64 " global_load_bytes=%" PRIu64
      " global_store_bytes=%" PRIu64 " buffer_load_bytes=%" PRIu64
      " buffer_store_bytes=%" PRIu64 " flat_read_bytes=%" PRIu64
      " flat_write_bytes=%" PRIu64 " local_read_bytes=%" PRIu64
      " local_write_bytes=%" PRIu64 " scalar_read_bytes=%" PRIu64
      " scalar_write_bytes=%" PRIu64 " private_read_bytes=%" PRIu64
      " private_write_bytes=%" PRIu64 " unclassified_read_bytes=%" PRIu64
      " unclassified_write_bytes=%" PRIu64 " atomic=%" PRIu64 " branch=%" PRIu64
      " barrier=%" PRIu64 " control=%" PRIu64 " conversion=%" PRIu64
      " cache=%" PRIu64 " register_move=%" PRIu64 "\n",
      (int)name.size, name.data, mix->descriptor_count, mix->unknown_count,
      mix->scalar_alu_count, mix->vector_alu_count, mix->matrix_count,
      mix->mfma_count, mix->smfmac_count, mix->wmma_count, mix->swmmac_count,
      mix->dot_count, mix->global_memory_count, mix->global_load_count,
      mix->global_store_count, mix->buffer_load_count, mix->buffer_store_count,
      mix->flat_memory_count, mix->local_memory_count, mix->scalar_memory_count,
      mix->private_memory_count, mix->generic_memory_count,
      mix->memory_read_unknown_width_count,
      mix->memory_write_unknown_width_count, mix->memory_read_byte_count,
      mix->memory_write_byte_count, mix->global_load_byte_count,
      mix->global_store_byte_count, mix->buffer_load_byte_count,
      mix->buffer_store_byte_count, mix->flat_read_byte_count,
      mix->flat_write_byte_count, mix->local_read_byte_count,
      mix->local_write_byte_count, mix->scalar_read_byte_count,
      mix->scalar_write_byte_count, mix->private_read_byte_count,
      mix->private_write_byte_count, mix->unclassified_read_byte_count,
      mix->unclassified_write_byte_count, mix->atomic_count, mix->branch_count,
      mix->barrier_count, mix->control_count, mix->conversion_count,
      mix->cache_count, mix->register_move_count);
}

static iree_status_t loom_target_compile_report_append_economics_fields(
    iree_string_builder_t* builder,
    const loom_target_compile_report_static_instruction_mix_t* mix,
    const loom_target_compile_report_workload_t* workload) {
  uint64_t per_workitem_total = 0;
  if (loom_target_compile_report_checked_add_u64(mix->memory_read_byte_count,
                                                 mix->memory_write_byte_count,
                                                 &per_workitem_total)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        " per_workitem_issued_read_bytes=%" PRIu64
        " per_workitem_issued_write_bytes=%" PRIu64
        " per_workitem_issued_total_bytes=%" PRIu64,
        mix->memory_read_byte_count, mix->memory_write_byte_count,
        per_workitem_total));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_DISPATCH_WORKITEM_COUNT)) {
    uint64_t dispatch_read_bytes = 0;
    uint64_t dispatch_write_bytes = 0;
    uint64_t dispatch_total_bytes = 0;
    if (loom_target_compile_report_checked_mul_u64(
            mix->memory_read_byte_count, workload->dispatch_workitem_count,
            &dispatch_read_bytes) &&
        loom_target_compile_report_checked_mul_u64(
            mix->memory_write_byte_count, workload->dispatch_workitem_count,
            &dispatch_write_bytes) &&
        loom_target_compile_report_checked_add_u64(
            dispatch_read_bytes, dispatch_write_bytes, &dispatch_total_bytes)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " dispatch_read_bytes=%" PRIu64 " dispatch_write_bytes=%" PRIu64
          " dispatch_total_bytes=%" PRIu64,
          dispatch_read_bytes, dispatch_write_bytes, dispatch_total_bytes));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_append_target_resources_fields(
    iree_string_builder_t* builder,
    const loom_target_compile_report_target_resources_t* resources) {
  const iree_string_view_t scalar_register_class =
      loom_target_compile_report_non_empty(resources->scalar_register_class);
  const iree_string_view_t vector_register_class =
      loom_target_compile_report_non_empty(resources->vector_register_class);
  const iree_string_view_t limiting_resource =
      loom_target_compile_report_non_empty(resources->limiting_resource);
  return iree_string_builder_append_format(
      builder,
      "scalar_register_class=%.*s scalar_final_registers=%" PRIu64
      " scalar_scheduled_pressure_peak=%" PRIu64
      " scalar_final_overhead=%" PRIu64
      " vector_register_class=%.*s vector_final_registers=%" PRIu64
      " vector_scheduled_pressure_peak=%" PRIu64
      " vector_final_overhead=%" PRIu64 " subgroup_size=%" PRIu32
      " resident_subgroups_per_simd=%" PRIu32 " max_subgroups_per_simd=%" PRIu32
      " occupancy_percent=%" PRIu32 " limiting=%.*s",
      (int)scalar_register_class.size, scalar_register_class.data,
      resources->scalar_register_count,
      resources->scalar_pressure_peak_live_units,
      resources->scalar_register_overhead_units,
      (int)vector_register_class.size, vector_register_class.data,
      resources->vector_register_count,
      resources->vector_pressure_peak_live_units,
      resources->vector_register_overhead_units, resources->subgroup_size,
      resources->resident_subgroups_per_simd, resources->max_subgroups_per_simd,
      resources->occupancy_percent, (int)limiting_resource.size,
      limiting_resource.data);
}

static iree_status_t loom_target_compile_report_format_summary(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, IREE_SV("COMPILE-REPORT: summary")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("artifact"),
      loom_target_compile_report_artifact_kind_name(report->artifact_kind)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " status=%s", iree_status_code_string(report->status_code)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("function"), report->function_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("backend"), report->backend_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("bundle"), report->target_bundle_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("export"), report->target_export_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("export_symbol"), report->target_export_symbol));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("config"), report->target_config_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("lowered"), report->lowered_symbol));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " artifact_bytes=%" PRIu64, report->artifact_size));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("\n")));

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: schedule nodes=%" PRIu64 " scheduled=%" PRIu64
        " deps=%" PRIu64 " resources=%" PRIu64 " hazards=%" PRIu64
        " models=%" PRIu64 " pressure_classes=%" PRIu64
        " peak_live_units=%" PRIu64 "\n",
        report->schedule_node_count, report->scheduled_node_count,
        report->schedule_dependency_count, report->schedule_resource_use_count,
        report->schedule_hazard_gap_count, report->schedule_model_summary_count,
        report->register_pressure_summary_count,
        report->register_pressure_peak_live_units));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("COMPILE-REPORT: planning")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_append_low_planning_text_fields(
            &report->low_planning, builder));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("\n")));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("COMPILE-REPORT: workload")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_workload_fields(
        builder, &report->workload));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("\n")));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_append_instruction_mix_fields(
            builder, IREE_SV("static_instruction_mix"),
            &report->static_instruction_mix));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_append_instruction_mix_fields(
            builder, IREE_SV("dynamic_instruction_mix"),
            &report->dynamic_instruction_mix));
  }

  if (loom_target_compile_report_has_economics(report->detail_flags,
                                               &report->dynamic_instruction_mix,
                                               &report->workload)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("COMPILE-REPORT: economics memory")));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_economics_fields(
        builder, &report->dynamic_instruction_mix, &report->workload));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("\n")));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: allocation assignments=%" PRIu64 " spills=%" PRIu64
        " spill_plans=%" PRIu64 " coalesced_copies=%" PRIu64
        " materialized_copies=%" PRIu64 " materialized_spill_storage=%" PRIu64
        " materialized_spill_storage_bytes=%" PRIu64
        " materialized_spill_stores=%" PRIu64
        " materialized_spill_store_bytes=%" PRIu64
        " materialized_reloads=%" PRIu64 " materialized_reload_bytes=%" PRIu64
        " storage_leases=%" PRIu64 " storage_lease_instances=%" PRIu64
        " storage_release_actions=%" PRIu64 "\n",
        report->allocation_assignment_count, report->allocation_spill_count,
        report->allocation_spill_plan_count,
        report->allocation_coalesced_copy_count,
        report->allocation_materialized_copy_count,
        report->allocation_materialized_spill_storage_count,
        report->allocation_materialized_spill_storage_bytes,
        report->allocation_materialized_spill_store_count,
        report->allocation_materialized_spill_store_bytes,
        report->allocation_materialized_reload_count,
        report->allocation_materialized_reload_bytes,
        report->allocation_storage_lease_count,
        report->allocation_storage_lease_instance_count,
        report->allocation_storage_release_action_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
    const loom_target_compile_report_target_resources_t* resources =
        &report->target_resources;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, IREE_SV("COMPILE-REPORT: target_resources ")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_append_target_resources_fields(builder,
                                                                  resources));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV("\n")));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES)) {
    uint64_t kind_count = 0;
    uint64_t packet_count = 0;
    uint64_t unit_count = 0;
    loom_target_compile_report_move_cause_totals(report, &kind_count,
                                                 &packet_count, &unit_count);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: move_causes kinds=%" PRIu64 " packets=%" PRIu64
        " units=%" PRIu64 "\n",
        kind_count, packet_count, unit_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    const loom_target_compile_report_wait_plan_t* wait_plan =
        &report->wait_plan;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: wait_plan actions=%" PRIu64 " explicit=%" PRIu64
        " planned=%" PRIu64 " full_drains=%" PRIu64 " partial_waits=%" PRIu64
        " drained=%" PRIu64 " max_drained=%" PRIu64 " max_outstanding=%" PRIu64
        " max_full_drain_outstanding=%" PRIu64 " counter_rows=%" PRIhsz
        " reason_summary_rows=%" PRIhsz " action_rows=%" PRIhsz "\n",
        wait_plan->action_count, wait_plan->explicit_action_count,
        wait_plan->planned_action_count, wait_plan->full_drain_count,
        wait_plan->partial_wait_count, wait_plan->drained_count,
        wait_plan->max_drained_count, wait_plan->max_outstanding_before,
        wait_plan->max_full_drain_outstanding_before,
        report->wait_counter_rows.count, report->wait_reason_summary_rows.count,
        report->wait_action_rows.count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_CAPABILITY_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: target_capability_rows count=%" PRIhsz "\n",
        report->target_capability_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: emission instructions=%" PRIu64 " code_bytes=%" PRIu64
        " storage_bytes=%" PRIu64 "\n",
        report->emitted_instruction_count, report->emitted_code_byte_count,
        report->emitted_code_storage_byte_count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: math_legalization rewritten=%" PRIu64
        " rejected=%" PRIu64 " missing_policy=%" PRIu64
        " missing_recipe=%" PRIu64 " rows=%" PRIhsz "\n",
        report->math_legalization_rewritten_op_count,
        report->math_legalization_rejected_op_count,
        report->math_legalization_missing_policy_op_count,
        report->math_legalization_missing_recipe_op_count,
        report->math_legalization_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: source_low selected_ops=%" PRIu64
        " emitted_ops=%" PRIu64 " rows=%" PRIhsz "\n",
        report->source_low_selected_op_count,
        report->source_low_emitted_op_count, report->source_low_rows.count));
    if (report->source_low_memory_summary.packet_count != 0) {
      const loom_target_compile_report_source_low_memory_summary_t* summary =
          &report->source_low_memory_summary;
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low_memory roots=%" PRIhsz
          " arguments=%" PRIhsz " argument_packets=%" PRIhsz
          " strategies=%" PRIhsz,
          report->source_low_memory_root_summaries.count,
          report->source_low_memory_argument_summaries.count,
          report->source_low_memory_argument_packet_summaries.count,
          report->source_low_memory_strategy_summaries.count));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_source_low_memory_summary_text(
              summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: target_legalization legal=%" PRIu64
        " rewritten=%" PRIu64 " target_rewritten=%" PRIu64
        " reference_rewritten=%" PRIu64 " deferred=%" PRIu64
        " invalid_ir=%" PRIu64 " unsupported=%" PRIu64 " unhandled=%" PRIu64
        " rows=%" PRIhsz "\n",
        report->target_legalization_legal_op_count,
        report->target_legalization_rewritten_op_count,
        report->target_legalization_target_rewritten_op_count,
        report->target_legalization_reference_rewritten_op_count,
        report->target_legalization_deferred_op_count,
        report->target_legalization_invalid_ir_op_count,
        report->target_legalization_unsupported_op_count,
        report->target_legalization_unhandled_op_count,
        report->target_legalization_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: memory private_bytes=%" PRIu64 " local_bytes=%" PRIu64
        "\n",
        report->private_memory_bytes, report->local_memory_bytes));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: pressure_rows count=%" PRIhsz "\n",
        report->pressure_rows.count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: pressure_origin_rows count=%" PRIhsz "\n",
        report->pressure_origin_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: schedule_band_rows count=%" PRIhsz "\n",
        report->schedule_band_rows.count));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: schedule_band_summary_rows count=%" PRIhsz "\n",
        report->schedule_band_summary_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: spill_rows count=%" PRIhsz "\n",
        report->spill_rows.count));
  }

  if (options->diagnostic_count != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: diagnostics count=%" PRIhsz "\n",
        options->diagnostic_count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: allocation_failure_rows count=%" PRIhsz "\n",
        report->allocation_failure_rows.count));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: allocation_high_water_rows count=%" PRIhsz "\n",
        report->allocation_high_water_rows.count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ENTRIES)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: entries count=%" PRIhsz "\n",
        report->entry_rows.count));
  }

  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_entry_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec = report->entry_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_entry_t* rows =
        (const loom_target_compile_report_entry_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_entry_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_function_name =
          loom_target_compile_report_non_empty(row->source_function_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_export_name =
          loom_target_compile_report_non_empty(row->target_export_name);
      const iree_string_view_t target_export_symbol =
          loom_target_compile_report_non_empty(row->target_export_symbol);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      uint64_t move_kind_count = 0;
      uint64_t move_packet_count = 0;
      uint64_t move_unit_count = 0;
      loom_target_compile_report_move_cause_counts_totals(
          row->move_causes, &move_kind_count, &move_packet_count,
          &move_unit_count);
      const loom_target_compile_report_wait_plan_t* wait_plan = &row->wait_plan;
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: entry[%" PRIhsz
          "] function=%.*s source=%.*s bundle=%.*s export=%.*s "
          "export_symbol=%.*s config=%.*s schedule_nodes=%" PRIu64
          " scheduled=%" PRIu64 " resource_uses=%" PRIu64
          " hazard_gaps=%" PRIu64 " model_summaries=%" PRIu64
          " pressure_summaries=%" PRIu64 " peak_live=%" PRIu64
          " assignments=%" PRIu64 " spills=%" PRIu64 " spill_plans=%" PRIu64
          " coalesced_copies=%" PRIu64 " materialized_copies=%" PRIu64
          " materialized_spill_storage=%" PRIu64
          " materialized_spill_storage_bytes=%" PRIu64
          " materialized_spill_stores=%" PRIu64
          " materialized_spill_store_bytes=%" PRIu64
          " materialized_reloads=%" PRIu64 " materialized_reload_bytes=%" PRIu64
          " storage_leases=%" PRIu64 " storage_lease_instances=%" PRIu64
          " storage_release_actions=%" PRIu64 " move_kinds=%" PRIu64
          " move_packets=%" PRIu64 " move_units=%" PRIu64
          " wait_actions=%" PRIu64 " wait_explicit=%" PRIu64
          " wait_planned=%" PRIu64 " wait_full_drains=%" PRIu64
          " wait_partial=%" PRIu64 " wait_drained=%" PRIu64
          " wait_max_drained=%" PRIu64 " wait_max_outstanding=%" PRIu64
          " wait_max_full_drain_outstanding=%" PRIu64 " instructions=%" PRIu64
          " code_bytes=%" PRIu64 " storage_bytes=%" PRIu64
          " private_bytes=%" PRIu64 " local_bytes=%" PRIu64,
          row_index, (int)function_name.size, function_name.data,
          (int)source_function_name.size, source_function_name.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_export_name.size, target_export_name.data,
          (int)target_export_symbol.size, target_export_symbol.data,
          (int)target_config_name.size, target_config_name.data,
          row->schedule_node_count, row->scheduled_node_count,
          row->schedule_resource_use_count, row->schedule_hazard_gap_count,
          row->schedule_model_summary_count,
          row->register_pressure_summary_count,
          row->register_pressure_peak_live_units,
          row->allocation_assignment_count, row->allocation_spill_count,
          row->allocation_spill_plan_count,
          row->allocation_coalesced_copy_count,
          row->allocation_materialized_copy_count,
          row->allocation_materialized_spill_storage_count,
          row->allocation_materialized_spill_storage_bytes,
          row->allocation_materialized_spill_store_count,
          row->allocation_materialized_spill_store_bytes,
          row->allocation_materialized_reload_count,
          row->allocation_materialized_reload_bytes,
          row->allocation_storage_lease_count,
          row->allocation_storage_lease_instance_count,
          row->allocation_storage_release_action_count, move_kind_count,
          move_packet_count, move_unit_count, wait_plan->action_count,
          wait_plan->explicit_action_count, wait_plan->planned_action_count,
          wait_plan->full_drain_count, wait_plan->partial_wait_count,
          wait_plan->drained_count, wait_plan->max_drained_count,
          wait_plan->max_outstanding_before,
          wait_plan->max_full_drain_outstanding_before,
          row->emitted_instruction_count, row->emitted_code_byte_count,
          row->emitted_code_storage_byte_count, row->private_memory_bytes,
          row->local_memory_bytes));
      if (iree_any_bit_set(
              row->detail_flags,
              LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_RESOURCES)) {
        const loom_target_compile_report_target_resources_t* resources =
            &row->target_resources;
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV(" ")));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_target_resources_fields(
                builder, resources));
      }
      if (iree_any_bit_set(row->detail_flags,
                           LOOM_TARGET_COMPILE_REPORT_DETAIL_WORKLOAD)) {
        IREE_RETURN_IF_ERROR(loom_target_compile_report_append_workload_fields(
            builder, &row->workload));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " pressure_rows=%" PRIhsz " pressure_origin_rows=%" PRIhsz
          " schedule_band_rows=%" PRIhsz " schedule_band_summary_rows=%" PRIhsz
          " spill_rows=%" PRIhsz " allocation_high_water_rows=%" PRIhsz
          " wait_counter_rows=%" PRIhsz " wait_reason_summary_rows=%" PRIhsz
          " wait_action_rows=%" PRIhsz " target_capability_rows=%" PRIhsz "\n",
          row->pressure_row_count, row->pressure_origin_row_count,
          row->schedule_band_row_count, row->schedule_band_summary_row_count,
          row->spill_row_count, row->allocation_high_water_row_count,
          row->wait_counter_row_count, row->wait_reason_summary_row_count,
          row->wait_action_row_count, row->target_capability_row_count));
      if (iree_any_bit_set(row->detail_flags,
                           LOOM_TARGET_COMPILE_REPORT_DETAIL_LOW_PLANNING)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "COMPILE-REPORT: entry[%" PRIhsz "].planning", row_index));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_low_planning_text_fields(
                &row->low_planning, builder));
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(builder, IREE_SV("\n")));
      }
    }
  }
  return iree_ok_status();
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t target_family_name =
          loom_target_compile_report_non_empty(row->target_family_name);
      const iree_string_view_t namespace_name =
          loom_target_compile_report_non_empty(row->namespace_name);
      const iree_string_view_t key =
          loom_target_compile_report_non_empty(row->key);
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
              loom_target_compile_report_non_empty(row->value_string);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_non_empty(row->counter_name);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_non_empty(row->counter_name);
      const iree_string_view_t reason_name =
          loom_target_compile_report_non_empty(row->reason_name);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t counter_name =
          loom_target_compile_report_non_empty(row->counter_name);
      const iree_string_view_t action_name =
          loom_target_compile_report_non_empty(row->action_name);
      const iree_string_view_t reason_name =
          loom_target_compile_report_non_empty(row->reason_name);
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
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("producer_operation"),
          row->producer_operation_name));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("producer_descriptor_key"),
          row->producer_descriptor_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
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
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("consumer_operation"),
          row->consumer_operation_name));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
          builder, IREE_SV("consumer_descriptor_key"),
          row->consumer_descriptor_key));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(
              (loom_scalar_type_t)row->element_type);
      const iree_string_view_t peak_block_name =
          loom_target_compile_report_non_empty(row->peak_block_name);
      const iree_string_view_t peak_operation_name =
          loom_target_compile_report_non_empty(row->peak_operation_name);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t peak_block_name =
          loom_target_compile_report_non_empty(row->peak_block_name);
      const iree_string_view_t peak_operation_name =
          loom_target_compile_report_non_empty(row->peak_operation_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_non_empty(row->sample_value_name);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t block_name =
          loom_target_compile_report_non_empty(row->block_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_non_empty(row->sample_value_name);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t block_name =
          loom_target_compile_report_non_empty(row->block_name);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t sample_value_name =
          loom_target_compile_report_non_empty(row->sample_value_name);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t slot_space =
          loom_target_compile_report_non_empty(row->slot_space);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t failure_code =
          loom_target_compile_report_non_empty(row->failure_code);
      const iree_string_view_t blocking_kind =
          loom_target_compile_report_allocation_failure_blocking_kind_name(
              row->blocking_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t origin_block_name =
          loom_target_compile_report_non_empty(row->origin_block_name);
      const iree_string_view_t location_kind =
          loom_target_compile_report_non_empty(row->location_kind);
      const iree_string_view_t conflict_value_name =
          loom_target_compile_report_non_empty(row->conflict_value_name);
      const iree_string_view_t conflict_location_kind =
          loom_target_compile_report_non_empty(row->conflict_location_kind);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t value_name =
          loom_target_compile_report_non_empty(row->value_name);
      const iree_string_view_t register_class =
          loom_target_compile_report_non_empty(row->register_class);
      const iree_string_view_t type_kind_name =
          loom_target_compile_report_type_kind_name(row->type_kind);
      const iree_string_view_t element_type_name =
          loom_target_compile_report_scalar_type_name(row->element_type);
      const iree_string_view_t origin_kind =
          loom_target_compile_report_pressure_origin_kind_name(
              row->origin_kind);
      const iree_string_view_t origin_operation_name =
          loom_target_compile_report_non_empty(row->origin_operation_name);
      const iree_string_view_t semantic_tag =
          loom_target_compile_report_non_empty(row->semantic_tag);
      const iree_string_view_t location_kind =
          loom_target_compile_report_non_empty(row->location_kind);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t selection_name =
          loom_target_compile_report_source_low_selection_name(
              row->selection_kind);
      const iree_string_view_t plan_key = row->plan_key;
      if (row->selection_kind ==
          LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "COMPILE-REPORT: source_low[%" PRIhsz
            "] function=%.*s source_op=%.*s selection=%.*s rule_set=%u "
            "rule=%u",
            row_index, (int)function_name.size, function_name.data,
            (int)source_op_name.size, source_op_name.data,
            (int)selection_name.size, selection_name.data, row->rule_set_index,
            row->rule_index));
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
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low[%" PRIhsz
          "] function=%.*s source_op=%.*s selection=%.*s plan=%" PRIu64,
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data,
          (int)selection_name.size, selection_name.data, row->plan_id));
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t target_symbol_name =
          loom_target_compile_report_non_empty(row->target_symbol_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      const iree_string_view_t target_source =
          loom_target_selection_source_name(row->target_source);
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
            loom_target_compile_report_non_empty(
                row->candidate_target_symbol_name);
        const iree_string_view_t candidate_bundle_name =
            loom_target_compile_report_non_empty(
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t transform_key =
          loom_target_compile_report_non_empty(row->transform_key);
      const iree_string_view_t outcome =
          loom_target_compile_report_non_empty(row->outcome);
      const iree_string_view_t reason =
          loom_target_compile_report_non_empty(row->reason);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
      const iree_string_view_t operation_kind =
          loom_target_compile_report_non_empty(row->operation_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_non_empty(row->packet_key);
      const iree_string_view_t strategy_key =
          loom_target_compile_report_non_empty(row->strategy_key);
      const iree_string_view_t address_form =
          loom_target_compile_report_non_empty(row->address_form);
      const iree_string_view_t dynamic_term_kind =
          loom_target_compile_report_non_empty(row->dynamic_term_kind);
      const iree_string_view_t fallback_reason =
          loom_target_compile_report_non_empty(row->fallback_reason);
      const iree_string_view_t bank_conflict_kind =
          loom_target_compile_report_non_empty(row->bank_conflict_kind);
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
          "vector_lane_stride_bytes=%u bank_stride_words=%u "
          "bank_conflict_degree=%u bank_conflict_kind=%.*s\n",
          (int)address_form.size, address_form.data,
          (int)dynamic_term_kind.size, dynamic_term_kind.data,
          (int)fallback_reason.size, fallback_reason.data,
          row->static_offset_bytes, row->element_byte_count,
          row->vector_lane_count, row->issued_read_byte_count,
          row->issued_write_byte_count, row->issued_read_unknown_width_count,
          row->issued_write_unknown_width_count, row->dynamic_stride_bytes,
          row->vector_lane_stride_bytes, row->bank_stride_words,
          row->bank_conflict_degree, (int)bank_conflict_kind.size,
          bank_conflict_kind.data));
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
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
          loom_target_compile_report_append_source_low_memory_summary_text(
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
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
          loom_target_compile_report_append_source_low_memory_summary_text(
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_root_name =
          loom_target_compile_report_non_empty(row->source_root_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
      const iree_string_view_t operation_kind =
          loom_target_compile_report_non_empty(row->operation_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_non_empty(row->packet_key);
      const iree_string_view_t strategy_key =
          loom_target_compile_report_non_empty(row->strategy_key);
      const iree_string_view_t fallback_reason =
          loom_target_compile_report_non_empty(row->fallback_reason);
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
          loom_target_compile_report_append_source_low_memory_summary_text(
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t memory_space =
          loom_target_compile_report_non_empty(row->memory_space);
      const iree_string_view_t operation_kind =
          loom_target_compile_report_non_empty(row->operation_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_non_empty(row->packet_key);
      const iree_string_view_t strategy_key =
          loom_target_compile_report_non_empty(row->strategy_key);
      const iree_string_view_t fallback_reason =
          loom_target_compile_report_non_empty(row->fallback_reason);
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
          loom_target_compile_report_append_source_low_memory_summary_text(
              &row->summary, &report->workload, builder));
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      const iree_string_view_t policy_name =
          loom_target_compile_report_non_empty(row->policy_name);
      const iree_string_view_t constraint_key =
          loom_target_compile_report_non_empty(row->constraint_key);
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
          loom_target_compile_report_non_empty(row->function_name);
      const iree_string_view_t source_op_name =
          loom_target_compile_report_non_empty(row->source_op_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_non_empty(row->target_bundle_name);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_non_empty(row->target_config_name);
      const iree_string_view_t legalizer_name =
          loom_target_compile_report_non_empty(row->legalizer_name);
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
          "bundle=%.*s config=%.*s binding=%u case=%u rule_set=%u rule=%u "
          "diagnostic=%u",
          row_index, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data, (int)mode_name.size,
          mode_name.data, (int)policy_name.size, policy_name.data,
          (int)action_name.size, action_name.data,
          (int)legalization_outcome_name.size, legalization_outcome_name.data,
          (int)outcome_name.size, outcome_name.data, (int)legalizer_name.size,
          legalizer_name.data, (int)strategy_name.size, strategy_name.data,
          (int)target_bundle_name.size, target_bundle_name.data,
          (int)target_config_name.size, target_config_name.data,
          row->binding_index, row->case_index, row->rule_set_index,
          row->rule_index, row->diagnostic_index));
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

iree_status_t loom_target_compile_report_format_text(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    iree_string_builder_t* builder) {
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_summary(report, options, builder));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ENTRIES)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_entry_rows(report, builder));
  }
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_move_causes(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_counter_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_reason_summary_rows(report,
                                                                   builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_wait_action_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_target_capability_rows(report,
                                                                 builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_config_binding_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pressure_rows(report, builder));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pressure_origin_rows(
        report, builder));
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
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_allocation_high_water_rows(report,
                                                                     builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_math_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_target_rows(report,
                                                                 builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_selection_summary_rows(
            report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_transform_rows(report,
                                                                    builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_memory_rows(report,
                                                                 builder));
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
        loom_target_compile_report_format_legalization_rows(report, builder));
  }
  return iree_ok_status();
}
