// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Terminal output helpers for loom-check.

#ifndef LOOM_TOOLS_LOOM_CHECK_OUTPUT_H_
#define LOOM_TOOLS_LOOM_CHECK_OUTPUT_H_

#include "iree/base/api.h"
#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prints the case header: filename :: case N [mode] OUTCOME.
void loom_check_print_case_header(iree_string_view_t filename,
                                  iree_host_size_t case_index,
                                  const loom_check_case_t* test_case,
                                  const loom_check_result_t* result);

// Prints the terminal summary for all processed cases.
void loom_check_print_summary(iree_host_size_t pass_count,
                              iree_host_size_t fail_count,
                              iree_host_size_t skip_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_OUTPUT_H_
