// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL actual provider lifecycle for benchmark candidates.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_HAL_ACTUAL_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_HAL_ACTUAL_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tools/iree-benchmark-loom/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes a benchmark-owned HAL actual provider for one candidate.
iree_status_t iree_benchmark_loom_hal_actual_provider_initialize(
    iree_benchmark_loom_hal_context_t* context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    iree_string_view_t pipeline, const loom_module_t* test_module,
    const loom_testbench_invocation_plan_t* actual_invocation,
    iree_string_view_t sample_compilation,
    const loom_testbench_case_plan_t* sample_constant_case_plan,
    iree_host_size_t sample_constant_ordinal, bool has_sample_constant_ordinal,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_run_candidate_artifact_manifest_options_t*
        artifact_manifest_options,
    iree_benchmark_loom_hal_actual_provider_t* out_provider);

// Releases provider-owned compile, diagnostic, and artifact path state.
void iree_benchmark_loom_hal_actual_provider_deinitialize(
    iree_benchmark_loom_hal_actual_provider_t* provider);

// Compiles the candidate owned by |provider|.
iree_status_t iree_benchmark_loom_hal_actual_provider_compile(
    iree_benchmark_loom_hal_actual_provider_t* provider);

// Projects a single-provider compile rejection into a benchmark result.
void iree_benchmark_loom_benchmark_result_set_compile_rejection(
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_benchmark_loom_benchmark_result_t* out_result);

// Compiles every provider in a multi-actual sequence.
iree_status_t iree_benchmark_loom_hal_actual_sequence_compile(
    loom_run_hal_testbench_actual_sequence_t* sequence);

// Returns the first rejected provider in |sequence|, or NULL when none failed.
const loom_run_hal_testbench_actual_provider_t*
iree_benchmark_loom_hal_actual_sequence_first_rejection(
    const loom_run_hal_testbench_actual_sequence_t* sequence);

// Projects a sequence-provider compile rejection into a benchmark result.
void iree_benchmark_loom_benchmark_result_set_sequence_compile_rejection(
    const loom_run_hal_testbench_actual_provider_t* provider,
    iree_string_view_t sample_compilation,
    iree_benchmark_loom_benchmark_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_HAL_ACTUAL_H_
