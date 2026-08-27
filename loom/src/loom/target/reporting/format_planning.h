// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared text and JSON formatting for target-low planning statistics.

#ifndef LOOM_TARGET_REPORTING_FORMAT_PLANNING_H_
#define LOOM_TARGET_REPORTING_FORMAT_PLANNING_H_

#include "iree/base/api.h"
#include "loom/codegen/low/planning_statistics.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends stable line-oriented planning fields without a leading label or
// trailing newline.
iree_status_t loom_target_compile_report_append_low_planning_text_fields(
    const loom_low_planning_statistics_t* statistics,
    iree_string_builder_t* builder);

// Formats one target-low planning statistics object as JSON.
iree_status_t loom_target_compile_report_format_low_planning_json(
    const loom_low_planning_statistics_t* statistics,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_FORMAT_PLANNING_H_
