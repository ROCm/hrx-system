// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Interleaved dispatch_complete benchmark comparison execution.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_COMPARISON_EXECUTION_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_COMPARISON_EXECUTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/iree-benchmark-loom/context.h"
#include "loom/tools/iree-benchmark-loom/event.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/options.h"
#include "loom/tools/iree-benchmark-loom/work_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_benchmark_loom_comparison_execution_options_t {
  // Stable run identity emitted into lifecycle events.
  const iree_benchmark_loom_run_identity_t* run;
  // Planned testbench module containing cases and benchmarks.
  const loom_testbench_module_plan_t* module_plan;
  // Selected benchmarks and deduplicated setup work for the comparison.
  const iree_benchmark_loom_work_plan_t* work_plan;
  // Parsed benchmark options controlling compilation and measurement.
  const iree_benchmark_loom_options_t* benchmark_options;
  // Shared HAL context used by dispatch_complete work items.
  iree_benchmark_loom_hal_context_t* hal_context;
  // Shared Loom run session used for candidate compilation.
  loom_run_session_t* session;
  // Source filename used in diagnostics and compiled artifacts.
  iree_string_view_t filename;
  // Full source text used by candidate compilation.
  iree_string_view_t source;
  // Structured compile-report capture policy.
  const loom_run_compile_report_capture_options_t* compile_report_options;
  // Structured artifact-manifest sidecar policy.
  const loom_run_candidate_artifact_manifest_options_t*
      artifact_manifest_options;
  // Base case execution options for correctness checks.
  const loom_testbench_case_execution_options_t* case_execution_options;
  // Arena reused for prepared correctness executors.
  iree_arena_allocator_t* execution_arena;
  // Host allocator used for execution-owned scratch storage.
  iree_allocator_t host_allocator;
  // Structured lifecycle event sink receiving execution records.
  const iree_benchmark_loom_event_sink_t* event_sink;
} iree_benchmark_loom_comparison_execution_options_t;

// Returns the number of timing samples a candidate receives from a schedule.
iree_host_size_t iree_benchmark_loom_dispatch_comparison_sample_capacity(
    iree_benchmark_loom_interleave_mode_t interleave_mode,
    iree_host_size_t candidate_count, iree_host_size_t candidate_index,
    iree_host_size_t repetitions);

// Executes an interleaved comparison and emits repetition/comparison events.
iree_status_t iree_benchmark_loom_run_dispatch_comparison(
    const iree_benchmark_loom_comparison_execution_options_t* options,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_COMPARISON_EXECUTION_H_
