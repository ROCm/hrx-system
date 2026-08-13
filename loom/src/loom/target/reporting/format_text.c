// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/format_text.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/target/reporting/format_planning.h"
#include "loom/target/reporting/schema.h"

iree_string_view_t loom_target_compile_report_text_non_empty(
    iree_string_view_t value) {
  return iree_string_view_is_empty(value) ? IREE_SV("-") : value;
}

static void loom_target_compile_report_move_cause_totals(
    const loom_target_compile_report_t* report, uint64_t* out_kind_count,
    uint64_t* out_packet_count, uint64_t* out_unit_count) {
  loom_target_compile_report_move_cause_counts_totals(
      report->move_causes, out_kind_count, out_packet_count, out_unit_count);
}

iree_status_t loom_target_compile_report_text_append_string_field(
    iree_string_builder_t* builder, iree_string_view_t name,
    iree_string_view_t value) {
  value = loom_target_compile_report_text_non_empty(value);
  return iree_string_builder_append_format(builder, " %.*s=%.*s",
                                           (int)name.size, name.data,
                                           (int)value.size, value.data);
}

static iree_status_t
loom_target_compile_report_append_target_insertion_summary_fields(
    iree_string_builder_t* builder,
    const loom_target_compile_report_target_insertion_summary_t* summary) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " static_packets=%" PRIu64 " exact_dynamic_packets=%" PRIu64
      " unknown_dynamic_packets=%" PRIu64,
      summary->static_packet_count, summary->exact_dynamic_packet_count,
      summary->unknown_dynamic_packet_count));
  if (summary->unknown_dynamic_packet_count != 0) {
    return iree_string_builder_append_cstring(builder,
                                              " dynamic_packets=unavailable");
  }
  return iree_string_builder_append_format(builder, " dynamic_packets=%" PRIu64,
                                           summary->dynamic_packet_count);
}

static iree_status_t
loom_target_compile_report_append_emission_breakdown_fields(
    iree_string_builder_t* builder,
    const loom_target_compile_report_emission_breakdown_t* breakdown) {
  return iree_string_builder_append_format(
      builder,
      " body_instructions=%" PRIu64 " entry_instructions=%" PRIu64
      " coissued_instructions=%" PRIu64 " coissued_components=%" PRIu64,
      breakdown->body_instruction_count, breakdown->entry_instruction_count,
      breakdown->coissued_instruction_count,
      breakdown->coissued_component_count);
}

static iree_status_t loom_target_compile_report_format_target_insertion_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  iree_host_size_t row_index = 0;
  for (const loom_target_compile_report_vec_t* vec =
           report->target_insertion_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_target_insertion_row_t* rows =
        (const loom_target_compile_report_target_insertion_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i, ++row_index) {
      const loom_target_compile_report_target_insertion_row_t* row = &rows[i];
      const iree_string_view_t function_name =
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t kind =
          loom_target_compile_report_target_insertion_kind_name(
              row->insertion_kind);
      const iree_string_view_t packet_key =
          loom_target_compile_report_text_non_empty(row->packet_key);
      const iree_string_view_t block_name =
          loom_target_compile_report_text_non_empty(row->block_name);
      const iree_string_view_t operation_name =
          loom_target_compile_report_text_non_empty(
              row->boundary_operation_name);
      const iree_string_view_t descriptor_key =
          loom_target_compile_report_text_non_empty(
              row->boundary_descriptor_key);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: target_insertion[%" PRIhsz
          "] function=%.*s kind=%.*s packet=%.*s block=%.*s"
          " block_index=%" PRIu32 " node_index=%" PRIu32
          " scheduled_ordinal=%" PRIu32
          " boundary_operation=%.*s"
          " boundary_descriptor=%.*s static_packets=%" PRIu64,
          row_index, (int)function_name.size, function_name.data,
          (int)kind.size, kind.data, (int)packet_key.size, packet_key.data,
          (int)block_name.size, block_name.data, row->block_index,
          row->node_index, row->scheduled_ordinal, (int)operation_name.size,
          operation_name.data, (int)descriptor_key.size, descriptor_key.data,
          row->static_packet_count));
      if (iree_any_bit_set(
              row->flags,
              LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_FLAG_DYNAMIC_PACKET_COUNT)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, " dynamic_packets=%" PRIu64 "\n",
            row->dynamic_packet_count));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
            builder, " dynamic_packets=unavailable\n"));
      }
    }
  }
  return iree_ok_status();
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
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_WORKGROUP_CLUSTER_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " cluster_size=%" PRIu32 "x%" PRIu32 "x%" PRIu32,
        workload->workgroup_cluster_size.x, workload->workgroup_cluster_size.y,
        workload->workgroup_cluster_size.z));
  }
  if (iree_any_bit_set(
          workload->flags,
          LOOM_TARGET_COMPILE_REPORT_WORKLOAD_FLAT_WORKGROUP_CLUSTER_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " flat_cluster_size=%" PRIu64,
        workload->flat_workgroup_cluster_size));
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

static iree_status_t
loom_target_compile_report_text_append_instruction_mix_fields(
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
  if (iree_checked_add_u64(mix->memory_read_byte_count,
                           mix->memory_write_byte_count, &per_workitem_total)) {
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
    if (iree_checked_mul_u64(mix->memory_read_byte_count,
                             workload->dispatch_workitem_count,
                             &dispatch_read_bytes) &&
        iree_checked_mul_u64(mix->memory_write_byte_count,
                             workload->dispatch_workitem_count,
                             &dispatch_write_bytes) &&
        iree_checked_add_u64(dispatch_read_bytes, dispatch_write_bytes,
                             &dispatch_total_bytes)) {
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
      loom_target_compile_report_text_non_empty(
          resources->scalar_register_class);
  const iree_string_view_t vector_register_class =
      loom_target_compile_report_text_non_empty(
          resources->vector_register_class);
  const iree_string_view_t limiting_resource =
      loom_target_compile_report_text_non_empty(resources->limiting_resource);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
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
      limiting_resource.data));
  const loom_target_residency_summary_t* summary =
      &resources->residency_summary;
  if (!loom_target_residency_summary_is_valid(summary)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " residency_best_tier=%" PRIu32 " residency_current_tier=%" PRIu32
      " residency_limiting_resource_count=%" PRIu32,
      summary->best_tier, summary->tier, summary->limiting_resource_count));
  if (iree_any_bit_set(
          summary->flags,
          LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " residency_next_better_tier=%" PRIu32,
        summary->next_better_tier));
  }
  if (!iree_any_bit_set(
          summary->flags,
          LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_UNIQUE_LIMITING_RESOURCE)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " residency_unique_limiting_resource=%.*s"
      " residency_limiting_resource_units=%" PRIu64,
      (int)summary->limiting_resource.size, summary->limiting_resource.data,
      summary->limiting_resource_units));
  if (iree_any_bit_set(
          summary->flags,
          LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_NEXT_BETTER_TIER)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " residency_reduction_units_to_next_better_tier=%" PRIu64,
        summary->limiting_resource_reduction_units_to_next_better_tier));
  }
  if (iree_any_bit_set(
          summary->flags,
          LOOM_TARGET_RESIDENCY_SUMMARY_FLAG_HAS_LIMITING_RESOURCE_NEXT_WORSE_TIER)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        " residency_limiting_resource_next_worse_tier=%" PRIu32
        " residency_limiting_resource_next_worse_cliff_units=%" PRIu64
        " residency_limiting_resource_additional_units_to_next_worse_tier="
        "%" PRIu64,
        summary->limiting_resource_next_worse_tier,
        summary->limiting_resource_next_worse_cliff_units,
        summary->limiting_resource_additional_units_to_next_worse_tier));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_summary(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, IREE_SV("COMPILE-REPORT: summary")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
      builder, IREE_SV("artifact"),
      loom_target_compile_report_artifact_kind_name(report->artifact_kind)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " status=%s", iree_status_code_string(report->status_code)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
      builder, IREE_SV("function"), report->function_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
      builder, IREE_SV("backend"), report->backend_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
      builder, IREE_SV("bundle"), report->target_bundle_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
      builder, IREE_SV("export"), report->target_export_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
      builder, IREE_SV("export_symbol"), report->target_export_symbol));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
      builder, IREE_SV("config"), report->target_config_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_text_append_string_field(
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
        loom_target_compile_report_text_append_instruction_mix_fields(
            builder, IREE_SV("static_instruction_mix"),
            &report->static_instruction_mix));
  }

  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_DYNAMIC_INSTRUCTION_MIX)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_text_append_instruction_mix_fields(
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
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder, "COMPILE-REPORT: target_insertions"));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_append_target_insertion_summary_fields(
            builder, &report->target_insertion_summary));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " rows=%" PRIhsz "\n", report->target_insertion_rows.count));
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
        " storage_bytes=%" PRIu64,
        report->emitted_instruction_count, report->emitted_code_byte_count,
        report->emitted_code_storage_byte_count));
    if (iree_any_bit_set(
            report->detail_flags,
            LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION_BREAKDOWN)) {
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_append_emission_breakdown_fields(
              builder, &report->emission_breakdown));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
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
          loom_target_compile_report_format_text_source_low_memory_summary(
              summary, &report->workload, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
      if (report->bank_service_summary.modeled_packet_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "COMPILE-REPORT: source_low_bank_service_summary groups=%" PRIhsz,
            report->source_low_bank_service_summaries.count));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_bank_service_summary_text_fields(
                &report->bank_service_summary, builder));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
      }
      if (report->subgroup_access_summary.modeled_packet_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "COMPILE-REPORT: source_low_subgroup_access_summary "
            "groups=%" PRIhsz,
            report->source_low_subgroup_access_summaries.count));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_subgroup_access_summary_text_fields(
                &report->subgroup_access_summary, builder));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
      }
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

  if (options->diagnostics.count != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: diagnostics count=%" PRIhsz "\n",
        options->diagnostics.count));
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
          loom_target_compile_report_text_non_empty(row->function_name);
      const iree_string_view_t source_function_name =
          loom_target_compile_report_text_non_empty(row->source_function_name);
      const iree_string_view_t target_bundle_name =
          loom_target_compile_report_text_non_empty(row->target_bundle_name);
      const iree_string_view_t target_export_name =
          loom_target_compile_report_text_non_empty(row->target_export_name);
      const iree_string_view_t target_export_symbol =
          loom_target_compile_report_text_non_empty(row->target_export_symbol);
      const iree_string_view_t target_config_name =
          loom_target_compile_report_text_non_empty(row->target_config_name);
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
              LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION_BREAKDOWN)) {
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_emission_breakdown_fields(
                builder, &row->emission_breakdown));
      }
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
      if (iree_any_bit_set(
              row->detail_flags,
              LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS)) {
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_target_insertion_summary_fields(
                builder, &row->target_insertion_summary));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " pressure_rows=%" PRIhsz " pressure_origin_rows=%" PRIhsz
          " schedule_band_rows=%" PRIhsz " schedule_band_summary_rows=%" PRIhsz
          " spill_rows=%" PRIhsz " allocation_high_water_rows=%" PRIhsz
          " wait_counter_rows=%" PRIhsz " wait_reason_summary_rows=%" PRIhsz
          " wait_action_rows=%" PRIhsz " target_capability_rows=%" PRIhsz
          " target_insertion_rows=%" PRIhsz "\n",
          row->pressure_row_count, row->pressure_origin_row_count,
          row->schedule_band_row_count, row->schedule_band_summary_row_count,
          row->spill_row_count, row->allocation_high_water_row_count,
          row->wait_counter_row_count, row->wait_reason_summary_row_count,
          row->wait_action_row_count, row->target_capability_row_count,
          row->target_insertion_row_count));
      if (row->bank_service_summary.modeled_packet_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "COMPILE-REPORT: entry_bank_service[%" PRIhsz "] function=%.*s",
            row_index, (int)function_name.size, function_name.data));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_bank_service_summary_text_fields(
                &row->bank_service_summary, builder));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
      }
      if (row->subgroup_access_summary.modeled_packet_count != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "COMPILE-REPORT: entry_subgroup_access[%" PRIhsz "] function=%.*s",
            row_index, (int)function_name.size, function_name.data));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_append_subgroup_access_summary_text_fields(
                &row->subgroup_access_summary, builder));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
      }
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
    if (iree_any_bit_set(
            report->detail_flags,
            LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS)) {
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_target_insertion_rows(report,
                                                                  builder));
    }
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_text_planning_details(report,
                                                                builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_text_lowering_details(report,
                                                                builder));
  }
  return iree_ok_status();
}
