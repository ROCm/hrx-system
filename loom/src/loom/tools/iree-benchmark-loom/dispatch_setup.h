// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared dispatch_complete compile and correctness setup.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_DISPATCH_SETUP_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_DISPATCH_SETUP_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/iree-benchmark-loom/context.h"
#include "loom/tools/iree-benchmark-loom/event.h"
#include "loom/tools/iree-benchmark-loom/hal_actual.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/options.h"
#include "loom/tools/iree-benchmark-loom/testbench.h"
#include "loom/tools/iree-benchmark-loom/work_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_benchmark_loom_dispatch_setup_options_t {
  // Stable run identity emitted into lifecycle events.
  const iree_benchmark_loom_run_identity_t* run;
  // Planned testbench module containing cases and benchmarks.
  const loom_testbench_module_plan_t* module_plan;
  // Selected and deduplicated physical work items to prepare.
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
} iree_benchmark_loom_dispatch_setup_options_t;

typedef struct iree_benchmark_loom_dispatch_compile_context_t {
  // True once this compile item has been initialized or skipped.
  bool initialized;
  // True when target requirements skip every work item using this compile item.
  bool skipped;
  // True when this compile item executes a multi-actual sequence.
  bool uses_sequence;
  // Compiled multi-actual sequence reused by all work items in the compile
  // item.
  loom_run_hal_testbench_actual_sequence_t hal_sequence;
  // True when |hal_sequence| owns initialized state.
  bool hal_sequence_initialized;
  // Compiled single-actual provider reused by all work items in the compile
  // item.
  iree_benchmark_loom_hal_actual_provider_t hal_provider;
  // True when |hal_provider| owns initialized state.
  bool hal_provider_initialized;
  // First sequence provider that rejected compilation, or NULL when runnable.
  const loom_run_hal_testbench_actual_provider_t* rejected_sequence_provider;
  // Execution options with HAL actual and reference providers wired in.
  loom_testbench_case_execution_options_t execution_options;
  // Materializer options used by HAL benchmark timing batches.
  loom_testbench_value_materializer_options_t benchmark_materializer;
  // Reference oracle storage borrowed by |execution_options|.
  iree_benchmark_loom_reference_oracles_t reference_oracles;
} iree_benchmark_loom_dispatch_compile_context_t;

typedef struct iree_benchmark_loom_dispatch_work_item_state_t {
  // True when compile and correctness succeeded and timing windows can run.
  bool runnable;
  // Number of correctness samples executed while preparing this work item.
  iree_host_size_t correctness_sample_count;
  // Number of failed correctness samples observed while preparing this work
  // item.
  iree_host_size_t correctness_failed_sample_count;
} iree_benchmark_loom_dispatch_work_item_state_t;

// Initializes or reuses the compile context for |compile_item|.
iree_status_t iree_benchmark_loom_dispatch_compile_context_initialize(
    const iree_benchmark_loom_dispatch_setup_options_t* options,
    const iree_benchmark_loom_dispatch_compile_item_t* compile_item,
    iree_benchmark_loom_dispatch_compile_context_t* context);

// Releases resources owned by |context|.
void iree_benchmark_loom_dispatch_compile_context_deinitialize(
    iree_benchmark_loom_dispatch_compile_context_t* context);

// Compiles and correctness-checks one physical dispatch work item.
iree_status_t iree_benchmark_loom_prepare_dispatch_work_item(
    const iree_benchmark_loom_dispatch_setup_options_t* options,
    const iree_benchmark_loom_work_item_t* work_item,
    iree_benchmark_loom_dispatch_compile_context_t* compile_context,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count,
    iree_benchmark_loom_dispatch_work_item_state_t* out_state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_DISPATCH_SETUP_H_
