// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reusable execution loop for prepared check testbench cases.
//
// This layer binds planning, value materialization, invocation dispatch, and
// expectation reporting into the production case-execution primitive shared by
// run, test, benchmark, tuning, and custom harnesses. It stays target-free:
// callers inject actual/oracle/expectation/file providers.

#ifndef LOOM_TOOLING_TESTBENCH_EXECUTOR_H_
#define LOOM_TOOLING_TESTBENCH_EXECUTOR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/tooling/testbench/expectation.h"
#include "loom/tooling/testbench/invocation.h"
#include "loom/tooling/testbench/value_materializer.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_testbench_case_execution_options_t {
  // Runtime dependencies used while materializing parameters, generated values,
  // fixture file values, and reusable executor scratch storage.
  loom_testbench_value_materializer_options_t materializer;
  // Actual and oracle invocation providers visible while preparing the case.
  loom_testbench_invocation_options_t invocation;
  // Custom expectation providers visible while preparing the case.
  loom_testbench_expectation_options_t expectation;
} loom_testbench_case_execution_options_t;

// Initializes case execution options with no providers and the system host
// allocator.
void loom_testbench_case_execution_options_initialize(
    loom_testbench_case_execution_options_t* out_options);

typedef struct loom_testbench_prepared_case_t {
  // Module that owns all IR referenced by this prepared case.
  const loom_module_t* module;
  // Static case plan being executed.
  const loom_testbench_case_plan_t* case_plan;
  // Prepared invocation schedule with provider callbacks resolved.
  loom_testbench_invocation_schedule_t invocation_schedule;
  // Prepared expectation schedule with provider callbacks resolved.
  loom_testbench_expectation_schedule_t expectation_schedule;
} loom_testbench_prepared_case_t;

// Prepares one case from |module_plan| for repeated execution.
//
// Provider-name resolution happens here. The returned prepared case borrows
// schedules allocated from |arena| and remains valid until that arena resets.
iree_status_t loom_testbench_prepare_case_execution(
    const loom_testbench_case_execution_options_t* options,
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index, iree_arena_allocator_t* arena,
    loom_testbench_prepared_case_t* out_prepared_case);

typedef struct loom_testbench_case_sample_result_t {
  // Static case plan that produced this result.
  const loom_testbench_case_plan_t* case_plan;
  // Concrete sample ordinal executed for |case_plan|.
  iree_host_size_t sample_ordinal;
  // True when all expectations passed.
  bool passed;
  // Borrowed report owned by the executor until the next run or deinitialize.
  const loom_testbench_expectation_report_t* expectation_report;
} loom_testbench_case_sample_result_t;

typedef struct loom_testbench_case_executor_t {
  // Prepared case being executed.
  const loom_testbench_prepared_case_t* prepared_case;
  // Materializer options captured from initialization.
  loom_testbench_value_materializer_options_t materializer_options;
  // Reusable materialized value table.
  loom_testbench_value_table_t value_table;
  // Reusable invocation scratch storage.
  loom_testbench_invocation_executor_t invocation_executor;
  // Reusable expectation result report.
  loom_testbench_expectation_report_t expectation_report;
} loom_testbench_case_executor_t;

// Initializes reusable scratch storage for one prepared case.
iree_status_t loom_testbench_case_executor_initialize(
    const loom_testbench_prepared_case_t* prepared_case,
    const loom_testbench_case_execution_options_t* options,
    loom_testbench_case_executor_t* out_executor);

// Releases all storage owned by |executor|.
void loom_testbench_case_executor_deinitialize(
    loom_testbench_case_executor_t* executor);

// Executes one concrete sample.
//
// A non-OK status means the sample could not execute. Ordinary expectation
// mismatches are recorded in |out_result| with |passed| set to false.
iree_status_t loom_testbench_run_case_sample(
    loom_testbench_case_executor_t* executor, iree_host_size_t sample_ordinal,
    loom_testbench_case_sample_result_t* out_result);

// Writes a deterministic JSON object for |result|. The schema is production
// evidence for check runners, reproducers, and tuning workflows.
iree_status_t loom_testbench_case_sample_result_write_json(
    const loom_testbench_case_sample_result_t* result,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_EXECUTOR_H_
