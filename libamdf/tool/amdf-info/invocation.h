// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_TOOL_INFO_INVOCATION_H_
#define AMDF_TOOL_INFO_INVOCATION_H_

#include "libamdf/tool/amdf-info/selection.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runs one process-lifetime diagnostic through the public library entry point.
// The options and selector have already passed parsing. Handles are explicitly
// torn down before the final result; native cleanup failure retains borrowed
// parents for process exit and fails the invocation. No retry, background work,
// implicit queue creation, or fallback discovery path is used.
amdf_info_exit_code_t amdf_info_invoke(const amdf_info_options_t* options,
                                       amdf_info_selector_t selector,
                                       amdf_query_api_fn_t query_api,
                                       FILE* output_file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_TOOL_INFO_INVOCATION_H_
