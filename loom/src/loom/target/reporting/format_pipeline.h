// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pipeline-plan compile report formatting.

#ifndef LOOM_TARGET_REPORTING_FORMAT_PIPELINE_H_
#define LOOM_TARGET_REPORTING_FORMAT_PIPELINE_H_

#include "loom/target/reporting/format.h"
#include "loom/util/json.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes the pipeline-plan object in JSON format.
iree_status_t loom_target_compile_report_format_pipeline_plan_json(
    const loom_target_compile_report_pipeline_plan_t* plan,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream);

// Appends the pipeline-plan summary and optional detail rows as text.
iree_status_t loom_target_compile_report_format_pipeline_plan_text(
    const loom_target_compile_report_pipeline_plan_t* plan,
    loom_target_compile_report_format_mode_t mode,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_FORMAT_PIPELINE_H_
