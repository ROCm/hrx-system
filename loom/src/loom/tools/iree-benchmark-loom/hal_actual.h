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
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/testbench/compiled_provider.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tools/iree-benchmark-loom/model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_benchmark_loom_options_t iree_benchmark_loom_options_t;

typedef struct iree_benchmark_loom_hal_compilation_options_t {
  // Public compiler services, module, and invocation configuration.
  const loom_testbench_compilation_t* compilation;
  // Exact-version native view of |compilation->module| used by testbench plans.
  const loom_module_t* native_module;
  // Exact source snapshots for native planning and scenario diagnostics.
  const loom_source_table_resolver_t* source_table;
  // Optional prepared pass program. NULL selects the target default.
  const loomc_pass_program_t* pass_program;
  // Optional caller-selected profile validated against the live HAL device.
  loomc_target_profile_t* requested_target_profile;
  // Optional sanitizer policy for the target default pass program.
  const loomc_sanitizer_options_t* sanitizer;
  // Observer for products without provider-local diagnostic capture.
  loom_testbench_compile_result_callback_t result_callback;
  // Public structured compile-report request.
  loomc_compile_report_options_t compile_report;
  // Public artifact-manifest request.
  loomc_artifact_manifest_options_t artifact_manifest;
} iree_benchmark_loom_hal_compilation_options_t;

// Initializes shared HAL provider options for benchmark-owned compilation.
// |invocation| may be NULL when a scenario profile selects subjects lazily.
void iree_benchmark_loom_hal_provider_options_initialize(
    iree_benchmark_loom_hal_context_t* context,
    const iree_benchmark_loom_hal_compilation_options_t* compilation_options,
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_compile_result_callback_t result_callback,
    loom_run_hal_testbench_actual_provider_options_t* out_options);

// Initializes a benchmark-owned HAL actual provider for one candidate.
void iree_benchmark_loom_hal_actual_provider_initialize(
    iree_benchmark_loom_hal_context_t* context,
    const iree_benchmark_loom_hal_compilation_options_t* compilation_options,
    const loom_testbench_invocation_plan_t* kernel_launch,
    iree_string_view_t artifact_path_suffix,
    iree_benchmark_loom_hal_actual_provider_t* out_provider);

// Releases provider-owned compile, diagnostic, and artifact path state.
void iree_benchmark_loom_hal_actual_provider_deinitialize(
    iree_benchmark_loom_hal_actual_provider_t* provider);

// Initializes benchmark-owned HAL actual providers for a multi-actual case.
iree_status_t iree_benchmark_loom_hal_actual_sequence_initialize(
    iree_benchmark_loom_hal_context_t* context,
    const iree_benchmark_loom_hal_compilation_options_t* compilation_options,
    const loom_testbench_case_plan_t* case_plan,
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
