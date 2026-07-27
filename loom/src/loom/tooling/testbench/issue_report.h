// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reporting helpers for testbench planning issues.

#ifndef LOOM_TOOLING_TESTBENCH_ISSUE_REPORT_H_
#define LOOM_TOOLING_TESTBENCH_ISSUE_REPORT_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the stable JSON spelling for |kind|.
iree_string_view_t loom_testbench_issue_kind_name(
    loom_testbench_issue_kind_t kind);

// Writes one planning issue as a JSON object.
iree_status_t loom_testbench_issue_write_json(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_issue_t* issue, loom_output_stream_t* stream);

// Writes |issues| as a JSON array.
iree_status_t loom_testbench_issue_array_write_json(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_issue_t* issues, iree_host_size_t issue_count,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_ISSUE_REPORT_H_
