// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Detailed compile report planning JSON fields.

#ifndef LOOM_TARGET_REPORTING_FORMAT_JSON_PLANNING_H_
#define LOOM_TARGET_REPORTING_FORMAT_JSON_PLANNING_H_

#include "loom/target/reporting/format_json.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes detailed pressure, scheduling, spill, and allocation fields.
iree_status_t loom_target_compile_report_format_json_planning_details(
    const loom_target_compile_report_t* report,
    loom_target_compile_report_format_mode_t mode,
    loom_json_object_writer_t* root_object, loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_FORMAT_JSON_PLANNING_H_
