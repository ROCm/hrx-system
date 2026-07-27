// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile-report adapters for target-low emission frames.

#ifndef LOOM_TARGET_REPORTING_LOW_H_
#define LOOM_TARGET_REPORTING_LOW_H_

#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Records schedule, allocation, pressure-row, and spill-row summaries for one
// packetized low function.
iree_status_t loom_target_compile_report_record_low_emission_frame(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame);

// Records coarse planning work and memory statistics for one low function.
// This accepts partial frames returned with user-facing diagnostics.
void loom_target_compile_report_record_low_planning(
    loom_target_compile_report_t* report,
    const loom_low_planning_statistics_t* statistics);

// Records allocation, pressure-row, spill-row, and allocation-failure summaries
// for one low allocation table. This accepts partial tables produced when
// allocation emitted user-facing diagnostics.
iree_status_t loom_target_compile_report_record_low_allocation(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation);

// Records source-to-low selection and emission summaries for one lowered
// source function.
iree_status_t loom_target_compile_report_record_low_lowering(
    loom_target_compile_report_t* report,
    const loom_low_lower_result_t* lower_result);

// Records workload facts preserved on one target-low kernel function.
void loom_target_compile_report_record_low_kernel_workload(
    loom_target_compile_report_t* report, const loom_op_t* low_function_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_LOW_H_
