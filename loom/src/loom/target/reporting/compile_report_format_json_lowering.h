// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile report lowering and economics JSON fields.

#ifndef LOOM_TARGET_REPORTING_COMPILE_REPORT_FORMAT_JSON_LOWERING_H_
#define LOOM_TARGET_REPORTING_COMPILE_REPORT_FORMAT_JSON_LOWERING_H_

#include "loom/target/reporting/compile_report_format_json.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when the report has economics evidence to serialize.
bool loom_target_compile_report_has_report_economics(
    const loom_target_compile_report_t* report);

// Writes the canonical report-level economics object.
iree_status_t loom_target_compile_report_format_report_economics_json(
    const loom_target_compile_report_t* report, loom_output_stream_t* stream);

// Writes detailed source-low, math, and target legalization fields.
iree_status_t loom_target_compile_report_format_json_lowering_details(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_json_object_writer_t* root_object, loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_COMPILE_REPORT_FORMAT_JSON_LOWERING_H_
