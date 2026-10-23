// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared HAL candidate compile and correctness setup.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_HAL_SETUP_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_HAL_SETUP_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/loom-benchmark/context.h"
#include "loom/tools/loom-benchmark/event.h"
#include "loom/tools/loom-benchmark/hal_actual.h"
#include "loom/tools/loom-benchmark/launch_evidence.h"
#include "loom/tools/loom-benchmark/model.h"
#include "loom/tools/loom-benchmark/options.h"
#include "loom/tools/loom-benchmark/testbench.h"
#include "loom/tools/loom-benchmark/work_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_benchmark_hal_setup_options_t {
  // Stable run identity emitted into lifecycle events.
  const loom_benchmark_run_identity_t* run;
  // Planned testbench module containing cases and benchmarks.
  const loom_testbench_module_plan_t* module_plan;
  // Selected and deduplicated physical work items to prepare.
  const loom_benchmark_work_plan_t* work_plan;
  // Parsed benchmark options controlling compilation and measurement.
  const loom_benchmark_options_t* benchmark_options;
  // Shared HAL context used by work items with kernel launches.
  loom_benchmark_hal_context_t* hal_context;
  // Public compiler inputs shared by all HAL candidates in this run.
  const loom_benchmark_hal_compilation_options_t* compilation;
  // Base case execution options for correctness checks.
  const loom_testbench_case_execution_options_t* case_execution_options;
  // Arena reused for prepared correctness executors.
  iree_arena_allocator_t* execution_arena;
  // Host allocator used for execution-owned scratch storage.
  iree_allocator_t host_allocator;
  // Structured lifecycle event sink receiving execution records.
  const loom_benchmark_event_sink_t* event_sink;
} loom_benchmark_hal_setup_options_t;

typedef struct loom_benchmark_hal_compile_context_t {
  // Planned compilation identity shared by all providers and work results.
  const loom_benchmark_hal_compile_item_t* compile_item;
  // True once this compile item has been initialized.
  bool initialized;
  // True when this compile item executes a multi-actual sequence.
  bool uses_sequence;
  // Compiled multi-actual sequence reused by all work items in the compile
  // item.
  loom_benchmark_hal_actual_sequence_t hal_sequence;
  // True when |hal_sequence| owns initialized state.
  bool hal_sequence_initialized;
  // Compiled single-actual provider reused by all work items in the compile
  // item.
  loom_benchmark_hal_actual_provider_t hal_provider;
  // True when |hal_provider| owns initialized state.
  bool hal_provider_initialized;
  // First sequence provider that rejected compilation, or NULL when runnable.
  const loom_benchmark_hal_actual_provider_t* rejected_sequence_provider;
  // Execution options with HAL actual and reference providers wired in.
  loom_testbench_case_execution_options_t execution_options;
  // Execution options used to prepare HAL benchmark timing batches. Host-side
  // invocations remain enabled while shaped values target device-local storage.
  loom_testbench_case_execution_options_t benchmark_execution_options;
  // Reference oracle storage borrowed by both execution option sets.
  loom_benchmark_reference_oracles_t reference_oracles;
} loom_benchmark_hal_compile_context_t;

typedef struct loom_benchmark_hal_work_item_state_t {
  // True when compile and correctness succeeded and timing windows can run.
  bool runnable;
  // Number of correctness samples executed while preparing this work item.
  iree_host_size_t correctness_sample_count;
  // Number of failed correctness samples observed while preparing this work
  // item.
  iree_host_size_t correctness_failed_sample_count;
  // Exact workloads and resolved launch configurations observed during the
  // correctness pass for this physical work item.
  loom_benchmark_launch_evidence_t launch_evidence;
} loom_benchmark_hal_work_item_state_t;

// Releases launch evidence owned by one prepared HAL work item.
void loom_benchmark_hal_work_item_state_deinitialize(
    loom_benchmark_hal_work_item_state_t* state);

// Initializes or reuses the compile context for |compile_item|.
iree_status_t loom_benchmark_hal_compile_context_initialize(
    const loom_benchmark_hal_setup_options_t* options,
    const loom_benchmark_hal_compile_item_t* compile_item,
    loom_benchmark_hal_compile_context_t* context);

// Releases resources owned by |context|.
void loom_benchmark_hal_compile_context_deinitialize(
    loom_benchmark_hal_compile_context_t* context);

// Projects compilation identity and single-provider artifact paths into
// |result|.
void loom_benchmark_hal_compile_context_set_result_artifacts(
    const loom_benchmark_hal_compile_context_t* context,
    loom_benchmark_benchmark_result_t* result);

// Compiles and correctness-checks one physical HAL-backed work item.
iree_status_t loom_benchmark_prepare_hal_work_item(
    const loom_benchmark_hal_setup_options_t* options,
    const loom_benchmark_work_item_t* work_item,
    loom_benchmark_hal_compile_context_t* compile_context,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count,
    loom_benchmark_hal_work_item_state_t* out_state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_HAL_SETUP_H_
