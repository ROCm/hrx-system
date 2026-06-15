// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON row emission and artifact report writing for iree-benchmark-loom.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_REPORT_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_REPORT_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes a borrowed status as a structured JSON object.
iree_status_t iree_benchmark_loom_write_status_object_json(
    const iree_status_t status, loom_output_stream_t* stream);

// Writes the run identifier field shared by every JSONL row.
iree_status_t iree_benchmark_loom_write_run_id_field_json(
    const iree_benchmark_loom_run_identity_t* run,
    loom_output_stream_t* stream);

// Writes stable benchmark candidate identity fields.
iree_status_t iree_benchmark_loom_write_candidate_identity_json(
    const iree_benchmark_loom_candidate_identity_t* candidate,
    loom_output_stream_t* stream);

// Writes a sample-compilation field when |sample_compilation| is non-empty.
iree_status_t iree_benchmark_loom_write_sample_compilation_field_json(
    iree_string_view_t sample_compilation, loom_output_stream_t* stream);

// Writes fields for one concrete parameterized case sample.
iree_status_t iree_benchmark_loom_write_sample_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_output_stream_t* stream);

// Writes sample-plan fields for a parameterized case.
iree_status_t iree_benchmark_loom_write_case_sample_plan_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    loom_output_stream_t* stream);

// Writes selected HAL context identity fields into an open JSON object.
iree_status_t iree_benchmark_loom_write_hal_context_identity_fields_json(
    const iree_benchmark_loom_hal_context_t* context,
    loom_output_stream_t* stream);

// Writes a HAL profile summary object.
iree_status_t iree_benchmark_loom_write_hal_profile_summary_json(
    const loom_run_hal_profile_summary_t* profile,
    loom_output_stream_t* stream);

// Writes one field separator/name pair inside a JSON object.
iree_status_t iree_benchmark_loom_write_json_object_field_name(
    loom_output_stream_t* stream, bool* first_field, const char* name);

// Writes a required string field inside a JSON object.
iree_status_t iree_benchmark_loom_write_json_string_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_string_view_t value);

// Writes a string field only when |value| is non-empty.
iree_status_t iree_benchmark_loom_write_json_optional_string_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_string_view_t value);

// Writes a required uint32 field inside a JSON object.
iree_status_t iree_benchmark_loom_write_json_u32_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    uint32_t value);

// Writes a required uint64 field inside a JSON object.
iree_status_t iree_benchmark_loom_write_json_u64_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    uint64_t value);

// Writes a required host-size field inside a JSON object.
iree_status_t iree_benchmark_loom_write_json_size_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_host_size_t value);

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
    iree_string_view_t sample_compilation,
    iree_host_size_t benchmark_sample_ordinal,
    iree_host_size_t case_sample_ordinal,
    const loom_testbench_case_sample_result_t* sample_result,
    iree_string_builder_t* sample_output);

// Appends the initial run row.
iree_status_t iree_benchmark_loom_append_run_row(
    const iree_benchmark_loom_run_identity_t* run, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode,
    iree_string_builder_t* output);

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
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode,
    iree_allocator_t allocator, iree_string_builder_t* plan_output);

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
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_REPORT_H_
