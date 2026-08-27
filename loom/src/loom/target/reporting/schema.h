// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared compile-report schema names and derived values.

#ifndef LOOM_TARGET_REPORTING_SCHEMA_H_
#define LOOM_TARGET_REPORTING_SCHEMA_H_

#include "iree/base/api.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Named string field used by compact storage-format projections.
typedef struct loom_target_compile_report_string_field_t {
  // Stable field name.
  const char* name;
  // Field value, empty when the field does not apply.
  iree_string_view_t value;
} loom_target_compile_report_string_field_t;

enum {
  LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_MEMORY_STORAGE_FIELD_COUNT = 9,
};

// Dispatch-scaled memory byte totals.
typedef struct loom_target_compile_report_dispatch_memory_bytes_t {
  // Dispatch-scaled logical or issued read bytes.
  uint64_t read_byte_count;
  // Dispatch-scaled logical or issued write bytes.
  uint64_t write_byte_count;
  // Sum of dispatch-scaled read and write bytes.
  uint64_t total_byte_count;
} loom_target_compile_report_dispatch_memory_bytes_t;

// Stable name and counter index for a reportable residual move cause.
typedef struct loom_target_compile_report_move_cause_descriptor_t {
  // Residual move cause used as the report counter index.
  loom_target_compile_report_move_cause_t cause;
  // Stable text name emitted in compile reports.
  iree_string_view_t name;
} loom_target_compile_report_move_cause_descriptor_t;

// Table covering every reportable residual move cause.
extern const loom_target_compile_report_move_cause_descriptor_t
    loom_target_compile_report_move_cause_descriptors
        [LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT - 1];

iree_string_view_t loom_target_compile_report_artifact_kind_name(
    loom_target_compile_artifact_kind_t kind);
iree_string_view_t loom_target_compile_report_source_low_selection_name(
    loom_target_compile_report_source_low_selection_kind_t kind);

iree_host_size_t loom_target_compile_report_source_low_memory_storage_fields(
    const loom_target_compile_report_source_low_memory_row_t* row,
    loom_target_compile_report_string_field_t* fields);
iree_host_size_t
loom_target_compile_report_source_low_memory_argument_packet_storage_fields(
    const loom_target_compile_report_source_low_memory_argument_packet_summary_t*
        summary,
    loom_target_compile_report_string_field_t* fields);
iree_host_size_t
loom_target_compile_report_source_low_memory_strategy_storage_fields(
    const loom_target_compile_report_source_low_memory_strategy_summary_t*
        summary,
    loom_target_compile_report_string_field_t* fields);
iree_host_size_t loom_target_compile_report_first_non_empty_string_field(
    const loom_target_compile_report_string_field_t* fields,
    iree_host_size_t field_count);

bool loom_target_compile_report_memory_interval_has_range(
    const loom_target_compile_report_memory_interval_t* interval);
bool loom_target_compile_report_source_low_memory_has_dynamic_evidence(
    const loom_target_compile_report_source_low_memory_summary_t* summary);
bool loom_target_compile_report_source_low_memory_has_complete_dynamic_evidence(
    const loom_target_compile_report_source_low_memory_summary_t* summary);
bool loom_target_compile_report_source_low_memory_has_dynamic_delta(
    const loom_target_compile_report_source_low_memory_summary_t* summary);
bool loom_target_compile_report_source_low_memory_can_dispatch_scale(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload);
bool loom_target_compile_report_source_low_memory_should_print_dynamic(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload);
bool loom_target_compile_report_dispatch_memory_bytes(
    uint64_t read_byte_count, uint64_t write_byte_count,
    uint64_t dispatch_workitem_count,
    loom_target_compile_report_dispatch_memory_bytes_t* out_bytes);
bool loom_target_compile_report_source_low_memory_dispatch_source_bytes(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_target_compile_report_dispatch_memory_bytes_t* out_bytes);
bool loom_target_compile_report_source_low_memory_dispatch_issued_bytes(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    loom_target_compile_report_dispatch_memory_bytes_t* out_bytes);

iree_string_view_t
loom_target_compile_report_allocation_failure_blocking_kind_name(
    loom_target_compile_report_allocation_failure_blocking_kind_t kind);
iree_string_view_t loom_target_compile_report_pressure_origin_kind_name(
    loom_target_compile_report_pressure_origin_kind_t kind);
iree_string_view_t loom_target_compile_report_spill_row_kind_name(
    loom_target_compile_report_spill_row_kind_t kind);
iree_string_view_t loom_target_compile_report_legalization_mode_name(
    loom_target_compile_report_legalization_mode_t mode);
iree_string_view_t loom_target_compile_report_legalization_policy_name(
    loom_target_compile_report_legalization_policy_t policy);
iree_string_view_t loom_target_compile_report_legalization_action_name(
    loom_target_compile_report_legalization_action_t action);
iree_string_view_t loom_target_compile_report_legalization_outcome_name(
    loom_target_compile_report_legalization_outcome_t outcome);
iree_string_view_t loom_target_compile_report_contract_outcome_name(
    loom_target_compile_report_contract_outcome_t outcome);
iree_string_view_t loom_target_compile_report_capability_value_kind_name(
    loom_target_compile_report_capability_value_kind_t kind);
iree_string_view_t loom_target_compile_report_legalizer_strategy_name(
    loom_target_compile_report_legalizer_strategy_t strategy);
iree_string_view_t loom_target_compile_report_math_action_name(
    loom_target_compile_report_math_action_t action);
iree_string_view_t loom_target_compile_report_type_kind_name(
    uint32_t type_kind);
iree_string_view_t loom_target_compile_report_scalar_type_name(
    uint32_t element_type);
iree_string_view_t loom_target_compile_report_native_layout_evidence_name(
    loom_native_layout_evidence_t evidence);
iree_string_view_t loom_target_compile_report_native_contraction_role_name(
    loom_contract_operand_role_t role);
iree_string_view_t loom_target_compile_report_native_physical_dimension_name(
    loom_native_physical_dimension_t dimension);

void loom_target_compile_report_move_cause_counts_totals(
    const loom_target_compile_report_move_cause_counts_t* counts,
    uint64_t* out_kind_count, uint64_t* out_packet_count,
    uint64_t* out_unit_count);

bool loom_target_compile_report_economics_has_memory(
    const loom_target_compile_report_static_instruction_mix_t* mix);
bool loom_target_compile_report_economics_has_operations(
    const loom_target_compile_report_static_instruction_mix_t* mix);
bool loom_target_compile_report_has_economics(
    loom_target_compile_report_detail_flags_t detail_flags,
    const loom_target_compile_report_static_instruction_mix_t* mix,
    const loom_target_compile_report_workload_t* workload);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_SCHEMA_H_
