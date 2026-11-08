// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured loom-test execution below the command-line adapter.

#ifndef LOOM_TOOLS_LOOM_TEST_RUN_H_
#define LOOM_TOOLS_LOOM_TEST_RUN_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "loom/tools/loom-test/main.h"
#include "loom/tools/loom-test/options.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_test_file_run_options_t {
  // Borrowed target-linked runner configuration.
  const loom_test_configuration_t* configuration;
  // Borrowed structured test options.
  const loom_test_options_t* test_options;
  // Input module path, empty for stdin, or "-" for stdin.
  iree_string_view_t input_path;
  // Caller-owned sink receiving the complete loom.test.v0 report.
  loom_output_stream_t* report_output;
  // Host allocator used for run-owned storage.
  iree_allocator_t host_allocator;
} loom_test_file_run_options_t;

typedef struct loom_test_run_result_t {
  // Process-style exit code implied by completed test outcomes.
  int exit_code;
  // Number of selected check.case records.
  iree_host_size_t case_count;
  // Number of selected check.scenario records.
  iree_host_size_t scenario_count;
  // Number of executed case samples.
  iree_host_size_t sample_count;
  // Number of raw failed case samples.
  iree_host_size_t failed_sample_count;
  // Number of executed scenario trials.
  iree_host_size_t trial_count;
  // Number of raw failed scenario trials.
  iree_host_size_t failed_trial_count;
  // Number of selected cases skipped by requirements.
  iree_host_size_t skipped_case_count;
  // Number of selected scenarios skipped by requirements.
  iree_host_size_t skipped_scenario_count;
  // Number of static planning issues across selected records.
  iree_host_size_t planning_issue_count;
  // Number of records that failed with an expected diagnostic.
  iree_host_size_t xfail_count;
  // Number of strict expected failures that unexpectedly passed.
  iree_host_size_t xpass_count;
  // Number of expected failures with missing or mismatched diagnostics.
  iree_host_size_t xfail_mismatch_count;
} loom_test_run_result_t;

// Runs one input file or stdin using structured test options.
iree_status_t loom_test_run_file(const loom_test_file_run_options_t* options,
                                 loom_test_run_result_t* out_result);

// Prints the compact command documentation requested by --agents_md.
void loom_test_print_agents_markdown(FILE* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_TEST_RUN_H_
