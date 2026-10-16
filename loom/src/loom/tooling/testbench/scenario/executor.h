// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared-product execution for finite check.scenario domains.

#ifndef LOOM_TOOLING_TESTBENCH_SCENARIO_EXECUTOR_H_
#define LOOM_TOOLING_TESTBENCH_SCENARIO_EXECUTOR_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/benchmark.h"
#include "loom/tooling/testbench/device_event.h"
#include "loom/tooling/testbench/expectation.h"
#include "loom/tooling/testbench/scenario/values.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_testbench_product_call_t {
  // Exact scenario coordinate represented by this call.
  const loom_testbench_scenario_trial_identity_t* identity;
  // Retained workload or specialization values in signature order.
  loom_testbench_value_t* call_parameters;
  // Retained ordinary input values in signature order.
  loom_testbench_value_t* arguments;
  // Uninitialized result storage populated by the prepared product.
  loom_testbench_value_t* results;
} loom_testbench_product_call_t;

// Executes a bounded batch against one prepared product.
//
// Every call has the arities declared by |invocation|. Implementations may
// execute calls serially or submit them together, but must complete every call
// before returning successfully and populate every declared result.
typedef iree_status_t(IREE_API_PTR* loom_testbench_product_execute_fn_t)(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls);

// Benchmarks a prepared product over target-local calls.
//
// Every call represents one independent physical realization of the same
// semantic trial. Implementations prepare target-local staging before entering
// their timing loop and execute exactly |options->batch_size| calls per timed
// batch. Oracle execution, result comparison, and readback are excluded.
typedef iree_status_t(IREE_API_PTR* loom_testbench_product_benchmark_fn_t)(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls,
    const loom_run_benchmark_options_t* options,
    iree_allocator_t host_allocator, loom_run_benchmark_result_t* out_result);

typedef void(IREE_API_PTR* loom_testbench_product_destroy_fn_t)(
    void* user_data);

// One ordinary executable product prepared for a static subject invocation.
typedef struct loom_testbench_prepared_product_t {
  // External profile that prepared this product.
  iree_string_view_t profile;
  // Representative static invocation compiled into this product.
  const loom_testbench_invocation_plan_t* invocation;
  // Executes one bounded batch.
  loom_testbench_product_execute_fn_t execute;
  // Benchmarks one target-only batch, or NULL when unsupported.
  loom_testbench_product_benchmark_fn_t benchmark;
  // Optional teardown for |user_data|, cleared on borrowed product handles.
  loom_testbench_product_destroy_fn_t destroy;
  // Product-owned state passed to callbacks.
  void* user_data;
} loom_testbench_prepared_product_t;

// Prepares one executable product from static IR and configuration values.
//
// The configuration table contains only values defined outside check.trial.
// Runtime trial values do not exist while this callback runs. The product is
// reusable by every invocation of the same subject with the same arities in
// this concrete configuration; trial-local SSA IDs and source locations are
// supplied separately during execution.
typedef iree_status_t(IREE_API_PTR* loom_testbench_product_prepare_fn_t)(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    iree_allocator_t host_allocator,
    loom_testbench_prepared_product_t* out_product);

// Externally selected compilation and execution profile.
typedef struct loom_testbench_execution_profile_t {
  // Stable profile name used in diagnostics and reports.
  iree_string_view_t name;
  // Product preparation callback.
  loom_testbench_product_prepare_fn_t prepare;
  // Profile-owned state passed to |prepare|.
  void* user_data;
} loom_testbench_execution_profile_t;

typedef uint8_t loom_testbench_scenario_execution_mode_t;
enum loom_testbench_scenario_execution_mode_e {
  // Executes target and oracle products and evaluates authored expectations.
  LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS = 0u,
  // Realizes and benchmarks only the target product.
  LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK = 1u,
};

typedef struct loom_testbench_scenario_execution_options_t {
  // Profile used for the product under test.
  loom_testbench_execution_profile_t target;
  // Profile used to produce comparison results and state.
  loom_testbench_execution_profile_t oracle;
  // Provider used to execute authored check.generate functions.
  loom_testbench_invocation_provider_t function_call;
  // Shared capture receiving device events during correctness runs.
  loom_testbench_device_event_capture_t* device_event_capture;
  // Allocator for prepared scenario bookkeeping and provider products.
  iree_allocator_t host_allocator;
} loom_testbench_scenario_execution_options_t;

// Initializes scenario execution options with no profiles and the system
// allocator.
void loom_testbench_scenario_execution_options_initialize(
    loom_testbench_scenario_execution_options_t* out_options);

typedef struct loom_testbench_prepared_scenario_trial_t {
  // Static trial domain represented by these products.
  const loom_testbench_trial_plan_t* trial_plan;
  // Prepared target product.
  loom_testbench_prepared_product_t target;
  // Prepared oracle product, empty for check.invoke.
  loom_testbench_prepared_product_t oracle;
} loom_testbench_prepared_scenario_trial_t;

// Products prepared for one concrete scenario configuration.
typedef struct loom_testbench_prepared_scenario_configuration_t {
  // Materialized configuration visible during product preparation.
  const loom_testbench_scenario_configuration_values_t* configuration;
  // Execution semantics selected while preparing every product.
  loom_testbench_scenario_execution_mode_t mode;
  // Provider used to execute authored check.generate functions.
  loom_testbench_invocation_provider_t function_call;
  // Shared device-event capture for correctness execution.
  loom_testbench_device_event_capture_t* device_event_capture;
  // Allocator owning |trials| and passed to product teardown.
  iree_allocator_t host_allocator;
  // Prepared trial domains in source order.
  loom_testbench_prepared_scenario_trial_t* trials;
  // Number of entries in |trials|.
  iree_host_size_t trial_count;
} loom_testbench_prepared_scenario_configuration_t;

// Prepares target and oracle products for one materialized configuration.
// All products are prepared before any trial executor can materialize inputs.
iree_status_t loom_testbench_prepare_scenario_configuration(
    const loom_testbench_scenario_execution_options_t* options,
    const loom_testbench_scenario_configuration_values_t* configuration,
    loom_testbench_scenario_execution_mode_t mode,
    loom_testbench_prepared_scenario_configuration_t* out_prepared);

// Releases all products and bookkeeping owned by |prepared|.
void loom_testbench_prepared_scenario_configuration_deinitialize(
    loom_testbench_prepared_scenario_configuration_t* prepared);

typedef struct loom_testbench_scenario_trial_result_t {
  // Exact coordinate needed to replay this trial.
  loom_testbench_scenario_trial_identity_t identity;
  // Static scenario that owns this trial result.
  const loom_testbench_scenario_plan_t* scenario_plan;
  // Static trial domain that produced this result.
  const loom_testbench_trial_plan_t* trial_plan;
  // True when execution completed without unhandled outcomes and every
  // authored expectation passed.
  bool passed;
  // Borrowed report owned by the executor, or NULL for check.invoke.
  const loom_testbench_expectation_report_t* expectation_report;
  // Borrowed target device events owned by the executor.
  const loom_testbench_device_event_list_t* device_events;
  // Borrowed event flags set by positive authored expectations.
  const uint8_t* expected_device_events;
  // Number of error-severity events with no positive expectation.
  iree_host_size_t unhandled_device_event_count;
} loom_testbench_scenario_trial_result_t;

typedef struct loom_testbench_scenario_trial_result_list_t {
  // Borrowed results owned by the executor until the next run.
  const loom_testbench_scenario_trial_result_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_testbench_scenario_trial_result_list_t;

// Reusable bounded-batch executor for one prepared trial domain.
typedef struct loom_testbench_scenario_trial_executor_t {
  // Prepared trial domain being executed.
  const loom_testbench_prepared_scenario_trial_t* prepared_trial;
  // Materialized configuration shared into each trial recipe.
  const loom_testbench_scenario_configuration_values_t* configuration;
  // Execution semantics inherited from the prepared configuration.
  loom_testbench_scenario_execution_mode_t mode;
  // Runtime dependencies used while materializing trial values.
  loom_testbench_value_materializer_options_t materializer_options;
  // Host allocator owning all arrays below.
  iree_allocator_t host_allocator;
  // Prepared authored generators in recipe order.
  loom_testbench_prepared_invocation_t* prepared_generators;
  // Static schedule describing |prepared_generators|.
  loom_testbench_invocation_schedule_t generator_schedule;
  // Reusable scratch used to execute authored generators.
  loom_testbench_invocation_executor_t generator_executor;
  // Reusable independent target/oracle value graphs per batch slot.
  loom_testbench_scenario_trial_values_t* trial_values;
  // Reusable result records per batch slot.
  loom_testbench_scenario_trial_result_t* results;
  // Reusable expectation reports per batch slot.
  loom_testbench_expectation_report_t* expectation_reports;
  // Owned target device-event snapshots per batch slot.
  loom_testbench_device_event_snapshot_t* device_event_snapshots;
  // Flat positive-expectation event flags per batch slot.
  uint8_t* expected_device_events;
  // Number of event flags reserved for each batch slot.
  iree_host_size_t expected_device_event_capacity;
  // Shared capture isolating target events and validating oracle execution.
  loom_testbench_device_event_capture_t* device_event_capture;
  // Target product calls per batch slot.
  loom_testbench_product_call_t* target_calls;
  // Oracle product calls per batch slot, or NULL for check.invoke.
  loom_testbench_product_call_t* oracle_calls;
  // Flat retained target call-parameter storage.
  loom_testbench_value_t* target_call_parameters;
  // Flat retained target argument storage.
  loom_testbench_value_t* target_arguments;
  // Flat move-owned target result storage.
  loom_testbench_value_t* target_results;
  // Flat retained oracle call-parameter storage.
  loom_testbench_value_t* oracle_call_parameters;
  // Flat retained oracle argument storage.
  loom_testbench_value_t* oracle_arguments;
  // Flat move-owned oracle result storage.
  loom_testbench_value_t* oracle_results;
  // Maximum number of trials accepted by one run call.
  iree_host_size_t batch_capacity;
  // Number of initialized entries in |trial_values| and expectation reports.
  iree_host_size_t initialized_slot_count;
} loom_testbench_scenario_trial_executor_t;

// Initializes reusable storage after every product has been prepared.
iree_status_t loom_testbench_scenario_trial_executor_initialize(
    const loom_testbench_prepared_scenario_configuration_t* prepared,
    iree_host_size_t trial_index,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t batch_capacity,
    loom_testbench_scenario_trial_executor_t* out_executor);

// Releases all storage owned by |executor|.
void loom_testbench_scenario_trial_executor_deinitialize(
    loom_testbench_scenario_trial_executor_t* executor);

// Materializes and executes a contiguous range within one trial domain.
//
// A non-OK status means the batch could not execute. Ordinary comparison
// mismatches are returned as failed results with source-located expectation
// reports.
iree_status_t loom_testbench_run_scenario_trial_batch(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t first_trial_ordinal, iree_host_size_t trial_count,
    loom_testbench_scenario_trial_result_list_t* out_results);

// Benchmarks one semantic trial using a batch of independent target graphs.
//
// |options->batch_size| determines both the number of physical target
// realizations and the number of logical operations normalized into one timed
// batch. Configuration and trial recipes run before product-local timing.
iree_status_t loom_testbench_benchmark_scenario_trial(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t trial_ordinal, const loom_run_benchmark_options_t* options,
    loom_run_benchmark_result_t* out_result);

// Writes one deterministic scenario trial result with its complete replay
// coordinate and any source-located expectation failures.
iree_status_t loom_testbench_scenario_trial_result_write_json(
    const loom_testbench_scenario_trial_result_t* result,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_SCENARIO_EXECUTOR_H_
