// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared expectation evaluation for check testbench cases.
//
// This layer evaluates check.expect.* operations over materialized values and
// records structured failure data. It is target-free: expectations handle
// scalars, HAL buffer views, and structured sample observations.

#ifndef LOOM_TOOLING_TESTBENCH_EXPECTATION_H_
#define LOOM_TOOLING_TESTBENCH_EXPECTATION_H_

#include "iree/base/api.h"
#include "loom/error/error_defs.h"
#include "loom/tooling/testbench/device_event.h"
#include "loom/tooling/testbench/value_materializer.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_testbench_sample_observations_t {
  // Device events captured while executing the sample's kernel launches.
  const loom_testbench_device_event_list_t* device_events;
  // Mutable byte flags set for events matched by positive event expectations.
  uint8_t* expected_device_events;
  // Number of entries available in |expected_device_events|.
  iree_host_size_t expected_device_event_capacity;
} loom_testbench_sample_observations_t;

// Returns empty sample observations.
static inline loom_testbench_sample_observations_t
loom_testbench_sample_observations_empty(void) {
  loom_testbench_sample_observations_t observations = {0};
  return observations;
}

// Returns the stable lowercase name for |kind|.
const char* loom_testbench_expectation_kind_name(
    loom_check_expectation_kind_t kind);

typedef struct loom_testbench_expectation_failure_t {
  // Source-order expectation ordinal.
  iree_host_size_t expectation_index;
  // Static expectation plan that failed.
  const loom_testbench_expectation_plan_t* expectation;
  // Kind copied from |expectation| for report consumers.
  loom_check_expectation_kind_t kind;
  // Stable EXPECT-domain diagnostic identity for this mismatch kind.
  loom_error_ref_t diagnostic_ref;
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
  // Borrowed module owning recorded failure operations; live until reset.
  const loom_module_t* module;
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

// Evaluates all planned expectations against |table| and records failures in
// |report|. |observations| supplies side-channel sample observations for event
// expectations and may be NULL when the case has none. A non-OK status means
// the evaluator could not run; ordinary expectation mismatches are recorded in
// |report|.
// Numerical comparisons decode each operand from its independently planned
// source type. Equal infinities match; other infinite pairs fail independently
// of tolerance. NaNs follow the expectation's explicit NaN policy, and integer
// distances remain exact across widths.
iree_status_t loom_testbench_evaluate_case_expectations(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_value_table_t* table,
    const loom_testbench_sample_observations_t* observations,
    loom_testbench_expectation_report_t* report);

// Evaluates a comparison action against independent target and oracle value
// tables. Actual operands resolve in |target_table| and expected operands
// resolve in |oracle_table|; lexical captures therefore select the matching
// realization even when both operands carry the same Loom SSA value ID.
iree_status_t loom_testbench_evaluate_scenario_action_expectations(
    const loom_testbench_scenario_action_plan_t* action,
    const loom_testbench_value_table_t* target_table,
    const loom_testbench_value_table_t* oracle_table,
    const loom_testbench_sample_observations_t* observations,
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
