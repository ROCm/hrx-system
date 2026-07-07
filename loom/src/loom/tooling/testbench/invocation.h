// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared invocation execution for check testbench cases.
//
// This layer resolves target/oracle providers once per planned case and then
// executes the resulting direct callbacks against materialized values. It stays
// target-free: VM, HAL, native, or reference execution providers are injected
// by callers instead of linked into the core testbench library.

#ifndef LOOM_TOOLING_TESTBENCH_INVOCATION_H_
#define LOOM_TOOLING_TESTBENCH_INVOCATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/tooling/testbench/value_materializer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t(IREE_API_PTR* loom_testbench_invocation_fn_t)(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results);

typedef struct loom_testbench_invocation_callback_t {
  // Callback function, or NULL when this provider is unavailable.
  loom_testbench_invocation_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_testbench_invocation_callback_t;

typedef enum loom_testbench_sample_issue_category_e {
  LOOM_TESTBENCH_SAMPLE_ISSUE_NONE = 0,
  LOOM_TESTBENCH_SAMPLE_ISSUE_COMPILE_REJECTED = 1,
} loom_testbench_sample_issue_category_t;

typedef struct loom_testbench_sample_issue_t {
  // Stable issue category.
  loom_testbench_sample_issue_category_t category;
  // Provider surface that produced the issue.
  iree_string_view_t provider;
  // Product stage that produced the issue.
  iree_string_view_t stage;
  // Stable provider-specific issue kind.
  iree_string_view_t kind;
  // Optional human-facing explanation.
  iree_string_view_t message;
} loom_testbench_sample_issue_t;

typedef iree_status_t(IREE_API_PTR* loom_testbench_invocation_issue_query_fn_t)(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_sample_issue_t* out_issue);

typedef struct loom_testbench_invocation_issue_query_t {
  // Query function, or NULL when the provider has no sample issues.
  loom_testbench_invocation_issue_query_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_testbench_invocation_issue_query_t;

typedef struct loom_testbench_oracle_provider_t {
  // Stable provider name referenced by check.oracle.call.
  iree_string_view_t name;
  // Callback used to produce expected results.
  loom_testbench_invocation_callback_t invoke;
} loom_testbench_oracle_provider_t;

typedef struct loom_testbench_oracle_provider_list_t {
  // Borrowed provider entries.
  const loom_testbench_oracle_provider_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_testbench_oracle_provider_list_t;

// Returns an empty oracle provider list.
static inline loom_testbench_oracle_provider_list_t
loom_testbench_oracle_provider_list_empty(void) {
  loom_testbench_oracle_provider_list_t list = {0};
  return list;
}

// Returns a borrowed oracle provider list.
static inline loom_testbench_oracle_provider_list_t
loom_make_testbench_oracle_provider_list(
    const loom_testbench_oracle_provider_t* values, iree_host_size_t count) {
  loom_testbench_oracle_provider_list_t list = {count > 0 ? values : NULL,
                                                count};
  return list;
}

// Returns true when |list| has no oracle providers.
static inline bool loom_testbench_oracle_provider_list_is_empty(
    loom_testbench_oracle_provider_list_t list) {
  return list.count == 0;
}

typedef struct loom_testbench_invocation_options_t {
  // Callback used for semantic call-like invocations of the function under
  // test.
  loom_testbench_invocation_callback_t invoke_actual;
  // Optional query for provider-owned sample issues after each invocation.
  loom_testbench_invocation_issue_query_t query_issue;
  // Named oracle providers visible to check.oracle.call.
  loom_testbench_oracle_provider_list_t oracle_providers;
} loom_testbench_invocation_options_t;

// Initializes invocation options with no providers.
void loom_testbench_invocation_options_initialize(
    loom_testbench_invocation_options_t* out_options);

typedef struct loom_testbench_prepared_invocation_t {
  // Static case invocation plan.
  const loom_testbench_invocation_plan_t* plan;
  // Resolved callback for |plan|.
  loom_testbench_invocation_callback_t invoke;
} loom_testbench_prepared_invocation_t;

typedef struct loom_testbench_invocation_schedule_t {
  // Prepared invocations in source order.
  const loom_testbench_prepared_invocation_t* invocations;
  // Optional query for provider-owned sample issues after each invocation.
  loom_testbench_invocation_issue_query_t query_issue;
  // Number of entries in |invocations|.
  iree_host_size_t invocation_count;
  // Maximum input arity across prepared invocations.
  iree_host_size_t max_input_count;
  // Maximum result arity across prepared invocations.
  iree_host_size_t max_result_count;
} loom_testbench_invocation_schedule_t;

// Resolves provider names and direct callbacks for all invocations in
// |case_plan|. String provider lookup happens here, not in the per-sample
// execution loop.
iree_status_t loom_testbench_prepare_case_invocations(
    const loom_testbench_invocation_options_t* options,
    const loom_testbench_case_plan_t* case_plan, iree_arena_allocator_t* arena,
    loom_testbench_invocation_schedule_t* out_schedule);

typedef struct loom_testbench_invocation_executor_t {
  // Prepared invocation schedule to execute.
  const loom_testbench_invocation_schedule_t* schedule;
  // Host allocator that owns |inputs| and |results|.
  iree_allocator_t host_allocator;
  // Retained input values reused for each invocation.
  loom_testbench_value_t* inputs;
  // Move-owned result values reused for each invocation.
  loom_testbench_value_t* results;
  // Provider-owned issues recorded during the current sample.
  loom_testbench_sample_issue_t* issues;
  // Number of entries in |inputs|.
  iree_host_size_t input_capacity;
  // Number of entries in |results|.
  iree_host_size_t result_capacity;
  // Number of entries available in |issues|.
  iree_host_size_t issue_capacity;
  // Number of entries populated in |issues| for the current sample.
  iree_host_size_t issue_count;
} loom_testbench_invocation_executor_t;

// Initializes reusable scratch storage for one prepared invocation schedule.
iree_status_t loom_testbench_invocation_executor_initialize(
    const loom_testbench_invocation_schedule_t* schedule,
    iree_allocator_t host_allocator,
    loom_testbench_invocation_executor_t* out_executor);

// Releases all scratch storage owned by |executor|.
void loom_testbench_invocation_executor_deinitialize(
    loom_testbench_invocation_executor_t* executor);

// Executes all prepared invocations in source order against |table|.
iree_status_t loom_testbench_run_case_invocations(
    loom_testbench_invocation_executor_t* executor,
    loom_testbench_value_table_t* table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_INVOCATION_H_
