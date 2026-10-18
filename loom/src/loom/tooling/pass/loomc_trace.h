// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLING_PASS_LOOMC_TRACE_H_
#define LOOM_TOOLING_PASS_LOOMC_TRACE_H_

#include "loom/tooling/pass/trace_cli.h"
#include "loomc/pass_trace.h"

#ifdef __cplusplus
extern "C" {
#endif

// Owns command-line trace output and its public invocation descriptor.
// The object must remain at a stable address until close.
typedef struct loom_tooling_loomc_pass_trace_t {
  // CLI output routing and artifact storage.
  loom_tooling_pass_trace_t output;
  // Public options borrowing this object's output and filter storage. The
  // caller may set next to compose other invocation extensions.
  loomc_pass_trace_options_t options;
  // Owned public string views referencing the CLI's before/after filter text.
  loomc_string_view_t* filters;
  // Allocator used for the filter array.
  iree_allocator_t allocator;
} loom_tooling_loomc_pass_trace_t;

// Opens trace output and adapts shared CLI flags to LoomC invocation options.
// Output routing must already have been validated. No storage is allocated
// when tracing is disabled. Call close after any open attempt.
iree_status_t loom_tooling_loomc_pass_trace_open_from_flags(
    const loom_tooling_pass_trace_open_options_t* options,
    iree_allocator_t allocator, loom_tooling_loomc_pass_trace_t* out_trace);

// Closes output and frees filters. Also accepts a zero-initialized object.
iree_status_t loom_tooling_loomc_pass_trace_close(
    loom_tooling_loomc_pass_trace_t* trace);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_PASS_LOOMC_TRACE_H_
