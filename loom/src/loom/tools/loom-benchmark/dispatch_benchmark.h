// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL dispatch benchmark measurement for dispatch_complete work items.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_DISPATCH_BENCHMARK_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_DISPATCH_BENCHMARK_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/loom-benchmark/hal_actual.h"
#include "loom/tools/loom-benchmark/model.h"
#include "loom/tools/loom-benchmark/options.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runs one dispatch_complete sample using a compiled single-actual provider.
iree_status_t loom_benchmark_run_hal_benchmark_sample(
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_benchmark_benchmark_policy_t* policy,
    const loom_benchmark_options_t* options,
    loom_benchmark_hal_actual_provider_t* provider,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena,
    iree_host_size_t benchmark_sample_ordinal, iree_allocator_t allocator,
    loom_benchmark_benchmark_result_t* out_result);

// Runs one dispatch_complete sample using a compiled multi-actual sequence.
iree_status_t loom_benchmark_run_hal_sequence_benchmark_sample(
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_benchmark_benchmark_policy_t* policy,
    const loom_benchmark_options_t* options,
    loom_benchmark_hal_context_t* hal_context,
    loom_benchmark_hal_actual_sequence_t* sequence,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena,
    iree_host_size_t benchmark_sample_ordinal, iree_allocator_t allocator,
    loom_benchmark_benchmark_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_DISPATCH_BENCHMARK_H_
