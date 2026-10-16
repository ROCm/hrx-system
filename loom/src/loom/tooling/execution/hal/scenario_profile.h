// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared HAL execution profile for check.scenario kernel and function
// subjects.

#ifndef LOOM_TOOLING_EXECUTION_HAL_SCENARIO_PROFILE_H_
#define LOOM_TOOLING_EXECUTION_HAL_SCENARIO_PROFILE_H_

#include "loom/error/diagnostic.h"
#include "loom/error/source.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/testbench/scenario/executor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiler and runtime inputs shared by every product prepared through one
// externally selected HAL scenario profile.
typedef struct loom_run_hal_testbench_scenario_profile_t {
  // Stable profile name surfaced in scenario results and diagnostics.
  iree_string_view_t name;
  // Provider options copied into each unique prepared HAL product. The module
  // and kernel launch may be replaced by a target-only function adapter during
  // product preparation.
  loom_run_hal_testbench_actual_provider_options_t provider_options;
  // Exact source snapshots used by tooling-generated diagnostics.
  const loom_source_table_resolver_t* source_table;
  // Diagnostic sink for scenario adapter errors outside public compilation.
  loom_diagnostic_sink_t diagnostic_sink;
} loom_run_hal_testbench_scenario_profile_t;

// Initializes a borrowing HAL scenario profile without compiling a product.
void loom_run_hal_testbench_scenario_profile_initialize(
    iree_string_view_t name,
    const loom_run_hal_testbench_actual_provider_options_t* provider_options,
    const loom_source_table_resolver_t* source_table,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_run_hal_testbench_scenario_profile_t* out_profile);

// Returns the external execution profile backed by |profile|.
//
// Product preparation compiles and loads one HAL artifact for each unique
// subject before trial-local values exist. Trial domains calling the same
// subject share that product while retaining their own value IDs and source
// identity. An ordinary function with scalar and buffer arguments and scalar
// results is privately adapted to a fixed 1x1 kernel; the authored module
// remains unchanged for oracle execution. Returned values occupy typed slots
// in a private output binding, separate from mutable input buffers.
// Correctness stages and executes bounded batches, reads back once, and decodes
// each disjoint result record. Product benchmarking stages a bounded batch once
// and replays one command sequence without allocation or readback.
loom_testbench_execution_profile_t
loom_run_hal_testbench_scenario_execution_profile(
    loom_run_hal_testbench_scenario_profile_t* profile);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_SCENARIO_PROFILE_H_
