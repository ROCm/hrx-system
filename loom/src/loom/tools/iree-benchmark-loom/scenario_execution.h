// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared check.scenario benchmark execution.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_SCENARIO_EXECUTION_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_SCENARIO_EXECUTION_H_

#include "iree/base/api.h"
#include "loom/tools/iree-benchmark-loom/work_execution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_benchmark_loom_scenario_execution_t
    iree_benchmark_loom_scenario_execution_t;

// Creates lazily prepared scenario benchmark execution state.
//
// When |entropy_domain| is non-empty, independently named scenarios derive
// generated values from that shared domain. Ordinary execution passes an empty
// domain and retains scenario-name isolation.
iree_status_t iree_benchmark_loom_scenario_execution_allocate(
    const iree_benchmark_loom_work_plan_execution_options_t* options,
    iree_string_view_t entropy_domain,
    iree_benchmark_loom_scenario_execution_t** out_execution);

// Releases all prepared products and materialized values owned by |execution|.
void iree_benchmark_loom_scenario_execution_free(
    iree_benchmark_loom_scenario_execution_t* execution);

// Prepares one concrete scenario trial and selects its physical input-ring
// shape. Values and device bindings are materialized outside each timed window.
iree_status_t iree_benchmark_loom_scenario_execution_prepare(
    iree_benchmark_loom_scenario_execution_t* execution,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    iree_host_size_t benchmark_sample_ordinal,
    loom_testbench_scenario_sample_coordinate_t coordinate);

// Verifies that two prepared trials have identical invocation layouts, values,
// and buffer alias topology.
iree_status_t iree_benchmark_loom_scenario_execution_validate_equivalent(
    iree_benchmark_loom_scenario_execution_t* baseline,
    iree_benchmark_loom_scenario_execution_t* candidate);

// Measures one prepared trial with |policy| and returns complete HAL evidence.
iree_status_t iree_benchmark_loom_scenario_execution_measure(
    iree_benchmark_loom_scenario_execution_t* execution,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    iree_benchmark_loom_benchmark_result_t* out_result);

// Measures one concrete scenario trial and emits every logical result alias.
iree_status_t iree_benchmark_loom_run_scenario_work_item(
    iree_benchmark_loom_scenario_execution_t* execution,
    const iree_benchmark_loom_work_item_t* work_item,
    iree_host_size_t* inout_failed_benchmark_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_SCENARIO_EXECUTION_H_
