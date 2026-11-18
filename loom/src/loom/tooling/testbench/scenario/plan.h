// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Planning for finite check.scenario configuration and trial domains.

#ifndef LOOM_TOOLING_TESTBENCH_SCENARIO_PLAN_H_
#define LOOM_TOOLING_TESTBENCH_SCENARIO_PLAN_H_

#include "loom/tooling/testbench/testbench.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_testbench_scenario_plan_counts_t {
  // Number of check.scenario records in the module.
  iree_host_size_t scenario_count;
  // Number of check.trial domains across all scenarios.
  iree_host_size_t trial_count;
  // Number of configuration value source operations.
  iree_host_size_t value_source_count;
  // Number of value-source and generator steps across trial recipes.
  iree_host_size_t recipe_step_count;
  // Number of expectations nested under check.compare actions.
  iree_host_size_t expectation_count;
  // Number of file outputs nested under check.compare actions.
  iree_host_size_t file_write_count;
  // Maximum number of structured issues scenario planning can emit.
  iree_host_size_t issue_capacity;
} loom_testbench_scenario_plan_counts_t;

// Counts arena storage and issue capacity required to plan all scenarios.
void loom_testbench_count_scenario_plans(
    const loom_module_t* module,
    loom_testbench_scenario_plan_counts_t* out_counts);

// Plans all scenario records into |arena| and appends planning issues to the
// module-owned flat issue array.
iree_status_t loom_testbench_plan_scenarios(
    const loom_module_t* module,
    const loom_testbench_scenario_plan_counts_t* counts,
    iree_arena_allocator_t* arena, loom_testbench_issue_t* issues,
    iree_host_size_t issue_capacity, iree_host_size_t* inout_issue_count,
    const loom_testbench_scenario_plan_t** out_scenarios,
    iree_host_size_t* out_scenario_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_SCENARIO_PLAN_H_
