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
#include "loom/sanitizer/options.h"
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
    const loom_run_module_t* run_module, iree_string_view_t pipeline,
    loom_sanitizer_options_t sanitizer,
    const loom_testbench_invocation_plan_t* kernel_launch,
    iree_string_view_t artifact_path_suffix,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_run_candidate_artifact_manifest_options_t*
        artifact_manifest_options,
    iree_benchmark_loom_hal_actual_provider_t* out_provider);

// Releases provider-owned compile, diagnostic, and artifact path state.
void iree_benchmark_loom_hal_actual_provider_deinitialize(
    iree_benchmark_loom_hal_actual_provider_t* provider);

// Initializes benchmark-owned HAL actual providers for a multi-actual case.
iree_status_t iree_benchmark_loom_hal_actual_sequence_initialize(
    iree_benchmark_loom_hal_context_t* context, loom_run_session_t* session,
    const loom_run_module_t* run_module, iree_string_view_t pipeline,
    loom_sanitizer_options_t sanitizer,
    const loom_testbench_case_plan_t* case_plan,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_run_candidate_artifact_manifest_options_t*
        artifact_manifest_options,
    iree_benchmark_loom_hal_actual_sequence_t* out_sequence);

// Releases storage owned by |sequence|.
void iree_benchmark_loom_hal_actual_sequence_deinitialize(
    iree_benchmark_loom_hal_actual_sequence_t* sequence);

// Compiles the candidate owned by |provider|.
iree_status_t iree_benchmark_loom_hal_actual_provider_compile(
    iree_benchmark_loom_hal_actual_provider_t* provider);

// Projects a single-provider compile rejection into a benchmark result.
void iree_benchmark_loom_benchmark_result_set_compile_rejection(
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_benchmark_loom_benchmark_result_t* out_result);

// Compiles every provider in a multi-actual sequence.
iree_status_t iree_benchmark_loom_hal_actual_sequence_compile(
    iree_benchmark_loom_hal_actual_sequence_t* sequence);

// Returns the first rejected provider in |sequence|, or NULL when none failed.
const iree_benchmark_loom_hal_actual_provider_t*
iree_benchmark_loom_hal_actual_sequence_first_rejection(
    const iree_benchmark_loom_hal_actual_sequence_t* sequence);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_HAL_ACTUAL_H_
