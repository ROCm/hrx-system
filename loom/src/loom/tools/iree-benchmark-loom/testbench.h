// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Testbench integration helpers for iree-benchmark-loom.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_TESTBENCH_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_TESTBENCH_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tooling/testbench/reference.h"
#include "loom/tooling/testbench/requirements.h"
#include "loom/tools/iree-benchmark-loom/configuration.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Reference oracles linked into HAL-backed tuning correctness checks.
  IREE_BENCHMARK_LOOM_REFERENCE_ORACLE_PROVIDER_COUNT = 2,
};

typedef struct iree_benchmark_loom_reference_oracles_t {
  // Shared options borrowed by every provider in |providers|.
  loom_testbench_reference_matmul_oracle_options_t options;
  // Reference oracle providers exposed to check.oracle.call.
  loom_testbench_oracle_provider_t
      providers[IREE_BENCHMARK_LOOM_REFERENCE_ORACLE_PROVIDER_COUNT];
} iree_benchmark_loom_reference_oracles_t;

// Evaluates check.requires/check.skip_if predicates for |case_plan|.
iree_status_t iree_benchmark_loom_evaluate_case_requirements(
    const iree_benchmark_loom_configuration_t* configuration,
    loom_run_hal_testbench_context_t* hal_context,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    loom_testbench_requirement_result_t* out_result);

// Configures generic host reference oracles used by check.oracle.call.
void iree_benchmark_loom_configure_reference_oracles(
    loom_run_hal_testbench_context_t* context, iree_allocator_t host_allocator,
    iree_benchmark_loom_reference_oracles_t* out_oracles,
    loom_testbench_case_execution_options_t* inout_execution_options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_TESTBENCH_H_
