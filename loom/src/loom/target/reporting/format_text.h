// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile report text formatting shared across package implementation files.

#ifndef LOOM_TARGET_REPORTING_FORMAT_TEXT_H_
#define LOOM_TARGET_REPORTING_FORMAT_TEXT_H_

#include "loom/target/reporting/format.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns |value| or the stable placeholder for an unavailable text value.
iree_string_view_t loom_target_compile_report_text_non_empty(
    iree_string_view_t value);

// Appends a named string field using the stable unavailable-value placeholder.
iree_status_t loom_target_compile_report_text_append_string_field(
    iree_string_builder_t* builder, iree_string_view_t name,
    iree_string_view_t value);

// Appends one source-low memory summary using its known workload scale.
iree_status_t loom_target_compile_report_format_text_source_low_memory_summary(
    const loom_target_compile_report_source_low_memory_summary_t* summary,
    const loom_target_compile_report_workload_t* workload,
    iree_string_builder_t* builder);

// Appends the fields of one structural bank-service summary.
iree_status_t
loom_target_compile_report_append_bank_service_summary_text_fields(
    const loom_target_compile_report_bank_service_summary_t* summary,
    iree_string_builder_t* builder);

// Appends structural subgroup-access summary fields.
iree_status_t
loom_target_compile_report_append_subgroup_access_summary_text_fields(
    const loom_target_compile_report_subgroup_access_summary_t* summary,
    iree_string_builder_t* builder);

// Appends detailed source-low, math, and target legalization evidence.
iree_status_t loom_target_compile_report_format_text_lowering_details(
    const loom_target_compile_report_t* report, iree_string_builder_t* builder);

// Appends detailed configuration, scheduling, and allocation evidence.
iree_status_t loom_target_compile_report_format_text_planning_details(
    const loom_target_compile_report_t* report, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_FORMAT_TEXT_H_
