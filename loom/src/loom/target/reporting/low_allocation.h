// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_REPORTING_LOW_ALLOCATION_H_
#define LOOM_TARGET_REPORTING_LOW_ALLOCATION_H_

#include "loom/target/reporting/low.h"
#include "loom/target/reporting/low_mix.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Records allocation, pressure, schedule-band, spill, and failure evidence.
iree_status_t loom_target_compile_report_record_low_allocation_contents(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_table_t* schedule,
    const loom_target_compile_report_low_dynamic_context_t* dynamic_context);

// Records spill operations materialized in the final emission frame.
iree_status_t loom_target_compile_report_record_materialized_spill_rows(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TARGET_REPORTING_LOW_ALLOCATION_H_
