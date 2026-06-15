// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared expectation evaluation for check testbench cases.
//
// This layer evaluates check.expect.* operations over materialized VM/HAL
// values and records structured failure data. It is target-free: builtin
// expectations handle VM scalars and HAL buffer views, while custom validators
// are injected by name during schedule preparation.

#ifndef LOOM_TOOLING_TESTBENCH_EXPECTATION_H_
#define LOOM_TOOLING_TESTBENCH_EXPECTATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/vm/api.h"
#include "loom/tooling/testbench/value_materializer.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t(IREE_API_PTR* loom_testbench_expectation_fn_t)(
    void* user_data, const loom_testbench_expectation_plan_t* expectation,
    const iree_vm_variant_t* actual, const iree_vm_variant_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched);

typedef struct loom_testbench_expectation_callback_t {
  // Callback function, or NULL when this provider is unavailable.
  loom_testbench_expectation_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_testbench_expectation_callback_t;

typedef struct loom_testbench_expectation_provider_t {
  // Stable provider name referenced by check.expect<provider>.
  iree_string_view_t name;
  // Callback used to evaluate the custom expectation.
  loom_testbench_expectation_callback_t evaluate;
} loom_testbench_expectation_provider_t;

typedef struct loom_testbench_expectation_provider_list_t {
  // Borrowed provider entries.
  const loom_testbench_expectation_provider_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_testbench_expectation_provider_list_t;

// Returns an empty expectation provider list.
static inline loom_testbench_expectation_provider_list_t
loom_testbench_expectation_provider_list_empty(void) {
  loom_testbench_expectation_provider_list_t list = {0};
  return list;
}

// Returns a borrowed expectation provider list.
static inline loom_testbench_expectation_provider_list_t
loom_make_testbench_expectation_provider_list(
    const loom_testbench_expectation_provider_t* values,
    iree_host_size_t count) {
  loom_testbench_expectation_provider_list_t list = {count > 0 ? values : NULL,
                                                     count};
  return list;
}

// Returns true when |list| has no expectation providers.
static inline bool loom_testbench_expectation_provider_list_is_empty(
    loom_testbench_expectation_provider_list_t list) {
  return list.count == 0;
}

typedef struct loom_testbench_expectation_options_t {
  // Named custom validators visible to check.expect<provider>.
  loom_testbench_expectation_provider_list_t providers;
} loom_testbench_expectation_options_t;

// Returns the stable lowercase name for |kind|.
const char* loom_testbench_expectation_kind_name(
    loom_testbench_expectation_kind_t kind);

// Initializes expectation options with no custom providers.
void loom_testbench_expectation_options_initialize(
    loom_testbench_expectation_options_t* out_options);

typedef struct loom_testbench_prepared_expectation_t {
  // Static case expectation plan.
  const loom_testbench_expectation_plan_t* plan;
  // Resolved custom callback for CUSTOM expectations, or NULL for builtins.
  loom_testbench_expectation_callback_t custom_evaluate;
} loom_testbench_prepared_expectation_t;

typedef struct loom_testbench_expectation_schedule_t {
  // Prepared expectations in source order.
  const loom_testbench_prepared_expectation_t* expectations;
  // Number of entries in |expectations|.
  iree_host_size_t expectation_count;
} loom_testbench_expectation_schedule_t;

// Resolves custom provider names for all expectations in |case_plan|.
//
// String provider lookup happens here, not in the per-sample evaluation loop.
iree_status_t loom_testbench_prepare_case_expectations(
    const loom_testbench_expectation_options_t* options,
    const loom_testbench_case_plan_t* case_plan, iree_arena_allocator_t* arena,
    loom_testbench_expectation_schedule_t* out_schedule);

typedef struct loom_testbench_expectation_failure_t {
  // Source-order expectation ordinal.
  iree_host_size_t expectation_index;
  // Static expectation plan that failed.
  const loom_testbench_expectation_plan_t* expectation;
  // Kind copied from |expectation| for report consumers.
  loom_testbench_expectation_kind_t kind;
  // Actual value ID compared by the expectation.
  loom_value_id_t actual_value_id;
  // Expected value ID, or INVALID when not applicable.
  loom_value_id_t expected_value_id;
  // Byte offset of the detail message in the owning report string storage.
  iree_host_size_t detail_offset;
  // Byte length of the detail message in the owning report string storage.
  iree_host_size_t detail_length;
} loom_testbench_expectation_failure_t;

typedef struct loom_testbench_expectation_report_t {
  // Host allocator that owns |failures| and |detail_builder|.
  iree_allocator_t host_allocator;
  // Failure storage with capacity for one entry per expectation.
  loom_testbench_expectation_failure_t* failures;
  // Number of entries allocated in |failures|.
  iree_host_size_t failure_capacity;
  // Number of expectations evaluated in the last run.
  iree_host_size_t expectation_count;
  // Number of expectations that passed in the last run.
  iree_host_size_t passed_count;
  // Number of entries populated in |failures|.
  iree_host_size_t failure_count;
  // Stable storage for failure detail messages.
  iree_string_builder_t detail_builder;
} loom_testbench_expectation_report_t;

// Initializes a reusable expectation report.
iree_status_t loom_testbench_expectation_report_initialize(
    iree_host_size_t failure_capacity, iree_allocator_t host_allocator,
    loom_testbench_expectation_report_t* out_report);

// Clears all recorded results while retaining allocated storage.
void loom_testbench_expectation_report_reset(
    loom_testbench_expectation_report_t* report);

// Releases storage owned by |report|.
void loom_testbench_expectation_report_deinitialize(
    loom_testbench_expectation_report_t* report);

// Returns the detail string for |failure|.
iree_string_view_t loom_testbench_expectation_failure_detail(
    const loom_testbench_expectation_report_t* report,
    const loom_testbench_expectation_failure_t* failure);

// Evaluates all prepared expectations against |table| and records failures in
// |report|. A non-OK status means the evaluator or custom provider could not
// run; ordinary expectation mismatches are recorded in |report|.
iree_status_t loom_testbench_evaluate_case_expectations(
    const loom_testbench_expectation_schedule_t* schedule,
    const loom_testbench_value_table_t* table,
    loom_testbench_expectation_report_t* report);

// Writes a deterministic JSON object for |report|. The schema is stable
// production evidence for loom-check, reproducers, and tuning workflows.
iree_status_t loom_testbench_expectation_report_write_json(
    const loom_testbench_expectation_report_t* report,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_EXPECTATION_H_
