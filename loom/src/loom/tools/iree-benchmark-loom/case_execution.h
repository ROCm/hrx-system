// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Correctness-gated check.case benchmark execution.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_CASE_EXECUTION_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_CASE_EXECUTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/iree-benchmark-loom/event.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/work_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps a benchmark-local sample ordinal to its underlying case sample ordinal.
iree_host_size_t iree_benchmark_loom_case_sample_from_benchmark_sample(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t benchmark_sample_ordinal);

// Emits result/profile events for every logical benchmark alias of |work_item|.
iree_status_t iree_benchmark_loom_emit_work_item_result_aliases(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* inout_failed_benchmark_count);

// Runs a benchmark sample range directly and emits sample events for
// |candidate|.
iree_status_t iree_benchmark_loom_run_case_correctness_range(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_string_view_t sample_compilation, iree_host_size_t begin_sample,
    iree_host_size_t end_sample, iree_arena_allocator_t* arena,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count);

// Runs correctness samples for |work_item| and emits aliased sample events.
iree_status_t iree_benchmark_loom_run_work_item_correctness_range(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* arena,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count);

// Runs one case_end_to_end physical work item and emits logical alias events.
iree_status_t iree_benchmark_loom_run_case_end_to_end_work_item(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_CASE_EXECUTION_H_
