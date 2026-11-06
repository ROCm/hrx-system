// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared HAL execution profile for check.scenario kernel, function, pipeline,
// and command-program subjects.

#ifndef LOOM_TOOLING_EXECUTION_HAL_SCENARIO_PROFILE_H_
#define LOOM_TOOLING_EXECUTION_HAL_SCENARIO_PROFILE_H_

#include "loom/error/diagnostic.h"
#include "loom/error/source.h"
#include "loom/tooling/execution/hal/benchmark.h"
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

typedef struct loom_run_hal_testbench_scenario_benchmark_result_t {
  // HAL timing, input-ring, command-buffer, and final-profile evidence.
  loom_run_hal_benchmark_result_t benchmark;
  // Number of physical kernel dispatches in one logical product call.
  iree_host_size_t dispatch_count_per_call;
  // Number of HAL bindings materialized for each physical call.
  iree_host_size_t binding_count_per_call;
  // Sum of logical binding byte lengths in one physical call.
  uint64_t binding_set_bytes;
  // Sum of logical binding byte lengths across the physical input ring.
  uint64_t binding_ring_bytes;
  // Number of disjoint reachable binding ranges in one physical call.
  iree_host_size_t binding_range_count_per_call;
  // Sum of reachable binding ranges in one physical call.
  uint64_t binding_range_set_bytes;
  // Sum of reachable binding ranges across the physical input ring.
  uint64_t binding_range_ring_bytes;
  // Number of unique logical input allocations in one physical call.
  iree_host_size_t materialized_storage_count_per_call;
  // Sum of unique logical input-allocation bytes in one physical call.
  uint64_t materialized_storage_set_bytes;
  // Sum of unique logical input-allocation bytes across the physical ring.
  uint64_t materialized_storage_ring_bytes;
} loom_run_hal_testbench_scenario_benchmark_result_t;

typedef struct loom_run_hal_testbench_scenario_input_layout_t {
  // Number of HAL bindings materialized for one physical call.
  iree_host_size_t binding_count;
  // Sum of logical binding byte lengths in the physical call.
  uint64_t binding_set_bytes;
  // Number of disjoint reachable binding ranges in the physical call.
  iree_host_size_t binding_range_count;
  // Sum of reachable binding ranges in the physical call.
  uint64_t binding_range_set_bytes;
  // Number of unique logical input allocations in the physical call.
  iree_host_size_t materialized_storage_count;
  // Sum of unique logical input-allocation bytes in the physical call.
  uint64_t materialized_storage_set_bytes;
} loom_run_hal_testbench_scenario_input_layout_t;

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
// remains unchanged for oracle execution. A command program is privately
// wrapped with configuration specializations frozen to exact scalar values,
// and its published kernel requests are compiled for the same live target.
// Returned function values occupy typed slots in a private output binding,
// separate from mutable input buffers.
// Correctness stages and executes bounded batches, reads back once, and decodes
// each disjoint result record. Product benchmarking stages a bounded batch once
// and replays one command sequence without allocation or readback.
loom_testbench_execution_profile_t
loom_run_hal_testbench_scenario_execution_profile(
    loom_run_hal_testbench_scenario_profile_t* profile);

// Benchmarks materialized calls from a prepared HAL scenario product.
//
// |calls| are independent physical realizations of one concrete semantic
// trial. The logical batch size comes from |options| and may differ from
// |call_count|. Target-local staging and replay preparation occur outside the
// measured timing window.
// |invocation| describes the current trial; the prepared product may have been
// shared from an earlier trial calling the same subject.
iree_status_t loom_run_hal_testbench_benchmark_scenario_product(
    const loom_testbench_prepared_product_t* prepared_product,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls,
    const loom_run_hal_benchmark_options_t* options,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_scenario_benchmark_result_t* out_result);

// Materializes one physical call and reports its HAL input layout without
// target-local staging or execution.
iree_status_t loom_run_hal_testbench_query_scenario_product_input_layout(
    const loom_testbench_prepared_product_t* prepared_product,
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_product_call_t* call,
    loom_run_hal_testbench_scenario_input_layout_t* out_layout);

// Returns the single compiled HAL provider owned by |prepared_product|.
// Composite command-program products and other execution profiles return NULL.
const loom_run_hal_testbench_actual_provider_t*
loom_run_hal_testbench_scenario_product_provider(
    const loom_testbench_prepared_product_t* prepared_product);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_SCENARIO_PROFILE_H_
