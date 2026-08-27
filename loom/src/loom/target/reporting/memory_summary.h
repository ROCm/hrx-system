// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_REPORTING_MEMORY_SUMMARY_H_
#define LOOM_TARGET_REPORTING_MEMORY_SUMMARY_H_

#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Accumulates structural bank-service counters from |source| into |target|.
void loom_target_compile_report_accumulate_bank_service_summaries(
    loom_target_compile_report_bank_service_summary_t* target,
    const loom_target_compile_report_bank_service_summary_t* source);

// Accumulates subgroup-access counters from |source| into |target|.
void loom_target_compile_report_accumulate_subgroup_access_summaries(
    loom_target_compile_report_subgroup_access_summary_t* target,
    const loom_target_compile_report_subgroup_access_summary_t* source);

// Accumulates source-memory summary counters from |source| into |target|.
void loom_target_compile_report_accumulate_source_low_memory_summaries(
    loom_target_compile_report_source_low_memory_summary_t* target,
    const loom_target_compile_report_source_low_memory_summary_t* source);

// Merges source-memory detail rows from |source| into |target|.
iree_status_t loom_target_compile_report_merge_source_low_memory_details(
    loom_target_compile_report_t* target,
    const loom_target_compile_report_t* source);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TARGET_REPORTING_MEMORY_SUMMARY_H_
