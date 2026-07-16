// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Line-oriented compile report planning details.

#ifndef LOOM_TARGET_REPORTING_COMPILE_REPORT_FORMAT_TEXT_PLANNING_H_
#define LOOM_TARGET_REPORTING_COMPILE_REPORT_FORMAT_TEXT_PLANNING_H_

#include "loom/target/reporting/compile_report_format.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends detailed configuration, scheduling, and allocation evidence.
iree_status_t loom_target_compile_report_format_text_planning_details(
    const loom_target_compile_report_t* report, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_COMPILE_REPORT_FORMAT_TEXT_PLANNING_H_
