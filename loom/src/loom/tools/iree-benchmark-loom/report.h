// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON row emission and artifact report writing for iree-benchmark-loom.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_REPORT_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_REPORT_H_

#include "iree/base/api.h"
#include "loom/sanitizer/options.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_benchmark_loom_summary_counts_t {
  // Number of planned check.case records.
  iree_host_size_t planned_case_count;
  // Number of planned check.benchmark records.
  iree_host_size_t planned_benchmark_count;
  // Number of selected benchmark candidates.
  iree_host_size_t selected_benchmark_count;
  // Number of logical benchmark samples selected for reporting.
  iree_host_size_t logical_sample_count;
  // Number of unique physical work items selected for execution.
  iree_host_size_t work_item_count;
  // Number of top-level failure rows emitted.
  iree_host_size_t failure_count;
  // Number of benchmark rows that failed or did not execute successfully.
  iree_host_size_t failed_benchmark_count;
  // Number of correctness samples executed.
  iree_host_size_t correctness_sample_count;
  // Number of correctness samples that failed expectations.
  iree_host_size_t correctness_failed_sample_count;
  // True when an artifact bundle was active.
  bool artifact_bundle_enabled;
  // Number of fixture-read files observed.
  iree_host_size_t fixture_read_count;
  // Number of file-output files observed.
  iree_host_size_t file_output_count;
  // Number of profile artifact files observed.
  iree_host_size_t profile_count;
  // Number of compile-report artifact files observed.
  iree_host_size_t compile_report_count;
  // Number of artifact-manifest files observed.
  iree_host_size_t artifact_manifest_count;
  // Number of target artifact files observed.
  iree_host_size_t target_artifact_count;
  // Number of target listing files observed.
  iree_host_size_t target_listing_count;
  // Number of HAL executable artifact files observed.
  iree_host_size_t hal_executable_count;
} iree_benchmark_loom_summary_counts_t;

// Writes a "status" field containing an IREE status-code object.
iree_status_t iree_benchmark_loom_write_status_field_json(
    iree_status_code_t code, iree_string_view_t message,
    loom_json_object_writer_t* object);

// Writes the run identifier field shared by every JSONL row.
iree_status_t iree_benchmark_loom_write_run_id_field_json(
    const iree_benchmark_loom_run_identity_t* run,
    loom_json_object_writer_t* object);

// Writes stable benchmark candidate identity fields.
iree_status_t iree_benchmark_loom_write_candidate_identity_json(
    const iree_benchmark_loom_candidate_identity_t* candidate,
    loom_json_object_writer_t* object);

// Writes sanitizer compiler/runtime policy as a compact JSON object.
iree_status_t iree_benchmark_loom_write_sanitizer_options_json(
    const loom_sanitizer_options_t* sanitizer, loom_output_stream_t* stream);

// Writes fields for one concrete parameterized case sample.
iree_status_t iree_benchmark_loom_write_sample_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_json_object_writer_t* object);

// Writes sample-plan fields for a parameterized case.
iree_status_t iree_benchmark_loom_write_case_sample_plan_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    loom_json_object_writer_t* object);

// Writes selected HAL context identity fields into an open JSON object.
iree_status_t iree_benchmark_loom_write_hal_context_identity_fields_json(
    const iree_benchmark_loom_hal_context_t* context,
    loom_json_object_writer_t* object);

// Writes a HAL profile summary object.
iree_status_t iree_benchmark_loom_write_hal_profile_summary_json(
    const loom_run_hal_profile_summary_t* profile,
    loom_output_stream_t* stream);

// Writes a final profile replay and its relationship to the scored measurement
// without the surrounding field name.
iree_status_t iree_benchmark_loom_write_hal_profile_replay_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream);

// Writes host-side timing summary statistics as a compact JSON object.
iree_status_t iree_benchmark_loom_write_timing_stats_json(
    const iree_benchmark_loom_timing_stats_t* stats,
    loom_output_stream_t* stream);

// Writes HAL benchmark timing summary statistics as a compact JSON object.
iree_status_t iree_benchmark_loom_write_benchmark_timing_stats_json(
    const loom_run_benchmark_timing_stats_t* stats,
    loom_output_stream_t* stream);

// Writes the effective benchmark policy as a compact JSON object.
iree_status_t iree_benchmark_loom_write_benchmark_policy_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    loom_output_stream_t* stream);

// Writes correctness sample counts as a compact JSON object.
iree_status_t iree_benchmark_loom_write_benchmark_correctness_json(
    iree_host_size_t sample_count, iree_host_size_t failed_sample_count,
    loom_output_stream_t* stream);

// Writes benchmark measurement evidence as a compact JSON object.
iree_status_t iree_benchmark_loom_write_benchmark_measurement_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream);

// Writes shared benchmark evidence fields into an open JSON object.
iree_status_t iree_benchmark_loom_write_benchmark_evidence_fields_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    loom_json_object_writer_t* object);

// Writes final aggregate run counts as a compact JSON object.
iree_status_t iree_benchmark_loom_write_summary_counts_json(
    const iree_benchmark_loom_summary_counts_t* counts,
    loom_output_stream_t* stream);

// Writes diagnostic capture count fields and diagnostics array fields into an
// open JSON object.
iree_status_t iree_benchmark_loom_write_diagnostic_capture_fields_json(
    const iree_benchmark_loom_diagnostic_capture_t* diagnostics,
    loom_json_object_writer_t* object);

// Writes testbench planning issue fields into an open JSON object.
iree_status_t iree_benchmark_loom_write_planning_issue_fields_json(
    const loom_testbench_module_plan_t* testbench_plan,
    const loom_testbench_issue_t* planning_issues,
    iree_host_size_t planning_issue_count, loom_json_object_writer_t* object);

// Returns the number of physical dispatches recorded in one measured HAL batch.
iree_status_t iree_benchmark_loom_hal_physical_dispatches_per_batch(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t* out_dispatch_count);

// Returns the number of physical dispatches represented by one logical
// operation timing sample.
iree_status_t iree_benchmark_loom_hal_physical_dispatches_per_logical_operation(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t* out_dispatch_count);

// Returns the number of physical dispatches covered by measured HAL batches.
iree_status_t iree_benchmark_loom_hal_measured_physical_dispatch_count(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t* out_dispatch_count);

// Returns measured HAL duration normalized by measured physical dispatches.
iree_status_t iree_benchmark_loom_hal_mean_physical_dispatch_duration_ns(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    double* out_duration_ns);

// Writes timing interpretation metadata for a HAL benchmark result object.
iree_status_t iree_benchmark_loom_write_hal_timing_interpretation_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream);

// Writes a benchmark failure object without the surrounding field name.
iree_status_t iree_benchmark_loom_write_benchmark_failure_json(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream);

// Writes a benchmark result object without the surrounding JSONL row wrapper.
iree_status_t iree_benchmark_loom_write_benchmark_result_json(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    loom_output_stream_t* stream);

// Appends one correctness sample result row.
iree_status_t iree_benchmark_loom_append_sample_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t benchmark_sample_ordinal,
    iree_host_size_t case_sample_ordinal,
    const loom_testbench_case_sample_result_t* sample_result,
    iree_string_builder_t* sample_output);

// Appends the initial run row.
iree_status_t iree_benchmark_loom_append_run_row(
    const iree_benchmark_loom_run_identity_t* run, bool dry_run,
    const loom_sanitizer_options_t* sanitizer, iree_string_builder_t* output);

// Appends the selected HAL device row once per run.
iree_status_t iree_benchmark_loom_append_device_row(
    const iree_benchmark_loom_run_identity_t* run,
    iree_benchmark_loom_hal_context_t* context,
    iree_benchmark_loom_device_row_state_t* state,
    iree_string_builder_t* output);

// Appends a selected benchmark plan row.
iree_status_t iree_benchmark_loom_append_plan_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options, iree_allocator_t allocator,
    iree_string_builder_t* plan_output);

// Writes target, listing, and HAL executable artifacts for a candidate.
iree_status_t iree_benchmark_loom_write_compiled_artifacts(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_allocator_t allocator);

// Writes a compile-report sidecar for a candidate when the bundle wants one.
iree_status_t iree_benchmark_loom_write_compile_report_artifact(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_allocator_t allocator);

// Appends a candidate compile row.
iree_status_t iree_benchmark_loom_append_compile_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* compile_output);

// Appends a benchmark result row.
iree_status_t iree_benchmark_loom_append_benchmark_result(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    iree_string_builder_t* benchmark_output);

// Appends one interleaved comparison benchmark repetition row.
iree_status_t iree_benchmark_loom_append_benchmark_repetition_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    const iree_benchmark_loom_candidate_identity_t* baseline,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, bool profile_suppressed,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_string_builder_t* benchmark_output);

// Appends one aggregate interleaved comparison row.
iree_status_t iree_benchmark_loom_append_comparison_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* baseline,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_string_builder_t* benchmark_output);

// Appends a parse, verify, planning, or infrastructure failure row.
iree_status_t iree_benchmark_loom_append_failure_row(
    const iree_benchmark_loom_run_identity_t* run, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message,
    const iree_benchmark_loom_diagnostic_capture_t* diagnostics,
    const loom_testbench_module_plan_t* testbench_plan,
    const loom_testbench_issue_t* planning_issues,
    iree_host_size_t planning_issue_count,
    iree_string_builder_t* failure_output);

// Appends the final summary row.
iree_status_t iree_benchmark_loom_append_summary_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count,
    iree_host_size_t logical_sample_count, iree_host_size_t work_item_count,
    iree_host_size_t failure_count, iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_REPORT_H_
