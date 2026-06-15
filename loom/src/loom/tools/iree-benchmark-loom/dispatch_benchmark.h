// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL dispatch benchmark measurement for dispatch_complete work items.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_DISPATCH_BENCHMARK_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_DISPATCH_BENCHMARK_H_

#include "iree/base/api.h"
#include "loom/tools/iree-benchmark-loom/hal_actual.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/options.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runs one dispatch_complete sample using a compiled single-actual provider.
iree_status_t iree_benchmark_loom_run_hal_benchmark_sample(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_actual_provider_t* provider,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t benchmark_sample_ordinal, iree_allocator_t allocator,
    iree_benchmark_loom_benchmark_result_t* out_result);

// Runs one dispatch_complete sample using a compiled multi-actual sequence.
iree_status_t iree_benchmark_loom_run_hal_sequence_benchmark_sample(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_context_t* hal_context,
    loom_run_hal_testbench_actual_sequence_t* sequence,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_string_view_t sample_compilation,
    iree_host_size_t benchmark_sample_ordinal, iree_allocator_t allocator,
    iree_benchmark_loom_benchmark_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_DISPATCH_BENCHMARK_H_
