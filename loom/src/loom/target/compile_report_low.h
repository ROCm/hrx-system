// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile-report adapters for target-low emission frames.

#ifndef LOOM_TARGET_COMPILE_REPORT_LOW_H_
#define LOOM_TARGET_COMPILE_REPORT_LOW_H_

#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/compile_report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Records schedule, allocation, pressure-row, and spill-row summaries for one
// packetized low function.
iree_status_t loom_target_compile_report_record_low_emission_frame(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame);

// Records source-to-low selection and emission summaries for one lowered
// source function.
iree_status_t loom_target_compile_report_record_low_lowering(
    loom_target_compile_report_t* report,
    const loom_low_lower_result_t* lower_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_COMPILE_REPORT_LOW_H_
