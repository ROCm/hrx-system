// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/snapshot.h"

#include <inttypes.h>
#include <string.h>

#include "loom/tooling/execution/benchmark.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/io/file.h"
#include "loom/tools/iree-benchmark-loom/report.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

typedef struct iree_benchmark_loom_snapshot_state_t {
  // Host allocator used for copied strings, builders, and index storage.
  iree_allocator_t host_allocator;
  // True after the run metadata event has been received.
  bool run_seen;
  // True after the terminal summary event has been received.
  bool summary_seen;
  // True when the run stopped after planning.
  bool dry_run;
  // Requested sample-compilation mode for this run.
  iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode;
  // Owned storage backing |run_id|.
  char* run_id_storage;
  // Copied run identifier.
  iree_string_view_t run_id;
  // Owned storage backing |source|.
  char* source_storage;
  // Copied input source path.
  iree_string_view_t source;
  // Owned storage backing |results_path|.
  char* results_path_storage;
  // Copied result output path.
  iree_string_view_t results_path;
  // Owned storage backing |file_output_dir|.
  char* file_output_dir_storage;
  // Copied file-output directory.
  iree_string_view_t file_output_dir;
  // Owned storage backing |profile_artifacts_dir|.
  char* profile_artifacts_dir_storage;
  // Copied profile-artifact directory.
  iree_string_view_t profile_artifacts_dir;
  // Owned storage backing |artifact_bundle_dir|.
  char* artifact_bundle_dir_storage;
  // Copied artifact bundle directory.
  iree_string_view_t artifact_bundle_dir;
  // Owned storage backing |artifact_bundle_policy|.
  char* artifact_bundle_policy_storage;
  // Copied artifact bundle policy name.
  iree_string_view_t artifact_bundle_policy;
  // Number of check.case records planned.
  iree_host_size_t planned_case_count;
  // Number of check.benchmark records planned.
  iree_host_size_t planned_benchmark_count;
  // Number of selected benchmark candidates.
  iree_host_size_t selected_benchmark_count;
  // Number of logical samples selected for reporting.
  iree_host_size_t logical_sample_count;
  // Number of physical work items selected for execution.
  iree_host_size_t work_item_count;
  // Number of top-level failure records.
  iree_host_size_t failure_count;
  // Number of failed benchmark records.
  iree_host_size_t failed_benchmark_count;
  // Number of correctness samples executed.
  iree_host_size_t correctness_sample_count;
  // Number of correctness samples that failed.
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
  // True after |device_json| receives a device object.
  bool has_device_json;
  // JSON object for the selected device.
  iree_string_builder_t device_json;
  // True until the first benchmark entry is appended.
  bool first_benchmark;
  // JSON array entries for logical benchmark rows.
  iree_string_builder_t benchmarks_json;
  // True until the first physical work entry is appended.
  bool first_work_item;
  // JSON array entries for deduplicated physical work results.
  iree_string_builder_t work_items_json;
  // True until the first failed correctness sample is appended.
  bool first_failed_sample;
  // JSON array entries for failed correctness samples.
  iree_string_builder_t failed_samples_json;
  // True until the first top-level failure is appended.
  bool first_failure;
  // JSON array entries for parse, verify, and infrastructure failures.
  iree_string_builder_t failures_json;
  // True until the first comparison repetition is appended.
  bool first_repetition;
  // JSON array entries for interleaved comparison repetitions.
  iree_string_builder_t repetitions_json;
  // True until the first comparison summary is appended.
  bool first_comparison;
  // JSON array entries for interleaved comparison summaries.
  iree_string_builder_t comparisons_json;
  // Physical work item indexes already emitted into |work_items_json|.
  iree_host_size_t* emitted_work_item_indexes;
  // Number of populated entries in |emitted_work_item_indexes|.
  iree_host_size_t emitted_work_item_count;
  // Allocated capacity of |emitted_work_item_indexes|.
  iree_host_size_t emitted_work_item_capacity;
} iree_benchmark_loom_snapshot_state_t;

static iree_benchmark_loom_snapshot_state_t* iree_benchmark_loom_snapshot_state(
    const iree_benchmark_loom_snapshot_sink_t* snapshot) {
  return (iree_benchmark_loom_snapshot_state_t*)snapshot->state;
}

static iree_status_t iree_benchmark_loom_snapshot_copy_string(
    iree_benchmark_loom_snapshot_state_t* state, iree_string_view_t value,
    char** storage, iree_string_view_t* out_value) {
  iree_allocator_free(state->host_allocator, *storage);
  *storage = NULL;
  *out_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_dup_string_view(
      value, state->host_allocator, storage));
  *out_value = iree_make_cstring_view(*storage);
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_snapshot_begin_array_entry(
    iree_string_builder_t* builder, bool* first_entry) {
  if (!*first_entry) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  }
  *first_entry = false;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_snapshot_write_candidate_fields(
    const iree_benchmark_loom_candidate_identity_t* candidate,
    loom_output_stream_t* stream, bool* first_field) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, first_field, "candidate_id", candidate->candidate_id));
  return iree_benchmark_loom_write_json_size_field(
      stream, first_field, "candidate_index", candidate->candidate_index);
}

static iree_status_t iree_benchmark_loom_snapshot_write_work_item_field(
    iree_host_size_t work_item_index, loom_output_stream_t* stream,
    bool* first_field) {
  if (work_item_index == IREE_BENCHMARK_LOOM_INDEX_INVALID) {
    return iree_ok_status();
  }
  return iree_benchmark_loom_write_json_size_field(
      stream, first_field, "work_item_index", work_item_index);
}

static iree_status_t iree_benchmark_loom_snapshot_write_benchmark_fields(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan, loom_output_stream_t* stream,
    bool* first_field) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, first_field, "benchmark", benchmark_plan->name));
  return iree_benchmark_loom_write_json_string_field(stream, first_field,
                                                     "case", case_plan->name);
}

static iree_string_view_t iree_benchmark_loom_snapshot_result_status(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  if (!iree_string_view_is_empty(benchmark_result->status)) {
    return benchmark_result->status;
  }
  if (benchmark_result->executed) {
    return benchmark_result->passed ? IREE_SV("ok") : IREE_SV("failed");
  }
  return IREE_SV("skipped");
}

static iree_status_t iree_benchmark_loom_snapshot_write_case_timing_json(
    const iree_benchmark_loom_timing_stats_t* timing,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"count\":%" PRIhsz, timing->count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"total\":%" PRIi64, timing->total_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"min\":%" PRIi64, timing->minimum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"max\":%" PRIi64, timing->maximum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, ",\"mean\":%.3f",
                                                       timing->mean_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p50\":%" PRIi64, timing->p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p90\":%" PRIi64, timing->p90_ns));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_write_hal_timing_json(
    const loom_run_benchmark_timing_stats_t* timing,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"count\":%" PRIhsz, timing->count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"total\":%" PRIi64, timing->total_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"min\":%" PRIi64, timing->minimum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"max\":%" PRIi64, timing->maximum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, ",\"mean\":%.3f",
                                                       timing->mean_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p50\":%" PRIi64, timing->p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p90\":%" PRIi64, timing->p90_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p90_to_p50_delta_ppm\":%" PRIu64,
      timing->p90_to_p50_delta_ppm));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_write_profile_json(
    const loom_run_hal_profile_summary_t* profile,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "status",
      profile->executed ? IREE_SV("ok") : IREE_SV("requested")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "row_count", profile->row_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, &first_field, "captured_row_count", profile->captured_row_count));
  if (profile->truncated_row_count != 0) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_field, "truncated_row_count",
        profile->truncated_row_count));
  }
  if (profile->dropped_record_count != 0) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
        stream, &first_field, "dropped_record_count",
        profile->dropped_record_count));
  }
  if (profile->has_artifact_path) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, &first_field, "artifact_path",
        iree_make_string_view(profile->artifact_path,
                              profile->artifact_path_length)));
  }
  if (profile->has_error) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, &first_field, "error"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
    bool first_error_field = true;
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, &first_error_field, "status",
        iree_make_cstring_view(iree_status_code_string(profile->error_code))));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, &first_error_field, "message",
        iree_make_string_view(profile->error_message,
                              profile->error_message_length)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_write_compile_report_json(
    const loom_run_compile_report_capture_t* compile_report_capture,
    loom_output_stream_t* stream, bool* first_field) {
  if (!loom_run_compile_report_capture_is_enabled(compile_report_capture)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "compile_report"));
  return loom_run_compile_report_capture_append_json(compile_report_capture,
                                                     stream);
}

static iree_status_t iree_benchmark_loom_snapshot_write_failure_fields_json(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream, bool* first_field) {
  if (!benchmark_result->has_failure) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "failure"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_failure_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_failure_field, "stage", benchmark_result->failure_stage));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_failure_field, "kind", benchmark_result->failure_kind));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, &first_failure_field, "message",
      benchmark_result->failure_message));
  if (benchmark_result->diagnostic_error_count != 0) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_failure_field, "diagnostic_error_count",
        benchmark_result->diagnostic_error_count));
  }
  if (benchmark_result->diagnostic_warning_count != 0) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_failure_field, "diagnostic_warning_count",
        benchmark_result->diagnostic_warning_count));
  }
  if (benchmark_result->diagnostic_remark_count != 0) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, &first_failure_field, "diagnostic_remark_count",
        benchmark_result->diagnostic_remark_count));
  }
  if (!iree_string_view_is_empty(benchmark_result->diagnostic_json)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, &first_failure_field, "diagnostics"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write(stream, benchmark_result->diagnostic_json));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_write_measurement_fields(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream, bool* first_field) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, first_field, "measure", policy->measure));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, first_field, "samples_per_iteration",
      benchmark_result->samples_per_iteration));
  if (benchmark_result->failed_sample_count != 0) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, first_field, "benchmark_failed_sample_count",
        benchmark_result->failed_sample_count));
  }
  if (benchmark_result->executed && !benchmark_result->has_hal_benchmark) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, first_field, "timing_ns"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_case_timing_json(
        &benchmark_result->timing, stream));
  }
  if (benchmark_result->has_hal_benchmark) {
    const loom_run_benchmark_result_t* timing =
        &benchmark_result->hal_benchmark.timing;
    iree_host_size_t physical_dispatches_per_batch = 0;
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_hal_physical_dispatches_per_batch(
        benchmark_result, &physical_dispatches_per_batch));
    iree_host_size_t physical_dispatches_per_logical_operation = 0;
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_hal_physical_dispatches_per_logical_operation(
            benchmark_result, &physical_dispatches_per_logical_operation));
    iree_host_size_t measured_physical_dispatch_count = 0;
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_hal_measured_physical_dispatch_count(
            benchmark_result, &measured_physical_dispatch_count));
    double mean_physical_dispatch_duration_ns = 0.0;
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_hal_mean_physical_dispatch_duration_ns(
            benchmark_result, &mean_physical_dispatch_duration_ns));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, first_field, "timing_ns"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_hal_timing_json(
        &timing->operation_timing, stream));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, first_field, "batch_timing_ns"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_hal_timing_json(
        &timing->batch_timing, stream));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, first_field, "operation_timing_ns"));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_hal_timing_json(
        &timing->operation_timing, stream));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, first_field, "logical_operations_per_batch",
        timing->batch_size));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, first_field, "physical_dispatches_per_batch",
        physical_dispatches_per_batch));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, first_field, "physical_dispatches_per_logical_operation",
        physical_dispatches_per_logical_operation));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, first_field, "measured_batch_count",
        timing->measured_batch_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, first_field, "measured_logical_operation_count",
        timing->measured_operation_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, first_field, "measured_physical_dispatch_count",
        measured_physical_dispatch_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, first_field, "mean_physical_dispatch_duration_ns"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "%.3f", mean_physical_dispatch_duration_ns));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, first_field, "stop_reason",
        loom_run_benchmark_stop_reason_name(timing->stop_reason)));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, first_field, "timing_interpretation"));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_hal_timing_interpretation_json(
            policy, benchmark_result, stream));
    if (benchmark_result->hal_benchmark.profile.requested ||
        benchmark_result->hal_benchmark.profile.executed ||
        benchmark_result->hal_benchmark.profile.has_error) {
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
          stream, first_field, "profile"));
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_profile_json(
          &benchmark_result->hal_benchmark.profile, stream));
    }
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_compile_report_json(
      benchmark_result->compile_report_capture, stream, first_field));
  return iree_benchmark_loom_snapshot_write_failure_fields_json(
      benchmark_result, stream, first_field);
}

static iree_status_t iree_benchmark_loom_snapshot_work_item_already_emitted(
    iree_benchmark_loom_snapshot_state_t* state,
    iree_host_size_t work_item_index, bool* out_already_emitted) {
  *out_already_emitted = false;
  if (work_item_index == IREE_BENCHMARK_LOOM_INDEX_INVALID) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < state->emitted_work_item_count; ++i) {
    if (state->emitted_work_item_indexes[i] == work_item_index) {
      *out_already_emitted = true;
      return iree_ok_status();
    }
  }
  if (state->emitted_work_item_count == state->emitted_work_item_capacity) {
    iree_host_size_t new_capacity = 8;
    if (state->emitted_work_item_capacity != 0 &&
        !iree_host_size_checked_mul(state->emitted_work_item_capacity, 2,
                                    &new_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "too many benchmark work item indexes");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        state->host_allocator, new_capacity,
        sizeof(state->emitted_work_item_indexes[0]),
        (void**)&state->emitted_work_item_indexes));
    state->emitted_work_item_capacity = new_capacity;
  }
  state->emitted_work_item_indexes[state->emitted_work_item_count++] =
      work_item_index;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_snapshot_append_run(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_run_event_t* event) {
  state->run_seen = true;
  state->dry_run = event->dry_run;
  state->sample_compilation_mode = event->sample_compilation_mode;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_copy_string(
      state, event->run->run_id, &state->run_id_storage, &state->run_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_copy_string(
      state, event->run->source, &state->source_storage, &state->source));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_copy_string(
      state, event->run->results_path, &state->results_path_storage,
      &state->results_path));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_copy_string(
      state, event->run->file_output_dir, &state->file_output_dir_storage,
      &state->file_output_dir));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_copy_string(
      state, event->run->profile_artifacts_dir,
      &state->profile_artifacts_dir_storage, &state->profile_artifacts_dir));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_copy_string(
      state, event->run->artifact_bundle_dir,
      &state->artifact_bundle_dir_storage, &state->artifact_bundle_dir));
  return iree_benchmark_loom_snapshot_copy_string(
      state, event->run->artifact_bundle_policy,
      &state->artifact_bundle_policy_storage, &state->artifact_bundle_policy);
}

static iree_status_t iree_benchmark_loom_snapshot_append_plan(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_plan_event_t* event) {
  (void)state;
  (void)event;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_snapshot_append_summary(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_summary_event_t* event) {
  state->summary_seen = true;
  state->planned_case_count = event->planned_case_count;
  state->planned_benchmark_count = event->planned_benchmark_count;
  state->selected_benchmark_count = event->selected_benchmark_count;
  state->logical_sample_count = event->logical_sample_count;
  state->work_item_count = event->work_item_count;
  state->failure_count = event->failure_count;
  state->failed_benchmark_count = event->failed_benchmark_count;
  state->correctness_sample_count = event->correctness_sample_count;
  state->correctness_failed_sample_count =
      event->correctness_failed_sample_count;
  state->dry_run = event->dry_run;
  state->sample_compilation_mode = event->sample_compilation_mode;
  state->artifact_bundle_enabled =
      event->artifact_bundle != NULL && event->artifact_bundle->enabled;
  state->fixture_read_count = iree_benchmark_loom_artifact_bundle_file_count(
      event->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_FIXTURE_READ);
  state->file_output_count = iree_benchmark_loom_artifact_bundle_file_count(
      event->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_OUTPUT);
  state->profile_count = iree_benchmark_loom_artifact_bundle_file_count(
      event->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE);
  state->compile_report_count = iree_benchmark_loom_artifact_bundle_file_count(
      event->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_COMPILE_REPORT);
  state->artifact_manifest_count =
      iree_benchmark_loom_artifact_bundle_file_count(
          event->artifact_bundle,
          IREE_BENCHMARK_LOOM_BUNDLE_FILE_ARTIFACT_MANIFEST);
  state->target_artifact_count = iree_benchmark_loom_artifact_bundle_file_count(
      event->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_ARTIFACT);
  state->target_listing_count = iree_benchmark_loom_artifact_bundle_file_count(
      event->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_LISTING);
  state->hal_executable_count = iree_benchmark_loom_artifact_bundle_file_count(
      event->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_HAL_EXECUTABLE);
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_snapshot_append_device(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_device_event_t* event) {
  iree_string_builder_reset(&state->device_json);
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_context_ensure_runtime(
      &event->context->execution));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&state->device_json, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_hal_context_identity_fields_json(event->context,
                                                                 &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  state->has_device_json = true;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_snapshot_append_sample(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_sample_event_t* event) {
  if (event->sample_result->passed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_begin_array_entry(
      &state->failed_samples_json, &state->first_failed_sample));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&state->failed_samples_json, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_candidate_fields(
      event->candidate, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_work_item_field(
      event->work_item_index, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_benchmark_fields(
      event->benchmark_plan, event->case_plan, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      &stream, &first_field, "sample_compilation", event->sample_compilation));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      &stream, &first_field, "benchmark_sample_index",
      event->benchmark_sample_ordinal));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      &stream, &first_field, "case_sample_index", event->case_sample_ordinal));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
      event->module, event->case_plan, event->case_sample_ordinal, &stream));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      &stream, &first_field, "sample_result"));
  IREE_RETURN_IF_ERROR(loom_testbench_case_sample_result_write_json(
      event->sample_result, &stream));
  return loom_output_stream_write_cstring(&stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_append_work_item(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_benchmark_result_event_t* event) {
  bool already_emitted = false;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_work_item_already_emitted(
      state, event->work_item_index, &already_emitted));
  if (already_emitted ||
      event->work_item_index == IREE_BENCHMARK_LOOM_INDEX_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_begin_array_entry(
      &state->work_items_json, &state->first_work_item));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&state->work_items_json, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      &stream, &first_field, "work_item_index", event->work_item_index));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "status",
      iree_benchmark_loom_snapshot_result_status(event->benchmark_result)));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      &stream, &first_field, "sample_compilation",
      event->benchmark_result->sample_compilation));
  if (event->benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
        event->module, event->case_plan,
        event->benchmark_result->sample_ordinal, &stream));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      &stream, &first_field, "correctness"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      "{\"sample_count\":%" PRIhsz ",\"failed_sample_count\":%" PRIhsz "}",
      event->correctness_sample_count, event->correctness_failed_sample_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_measurement_fields(
      event->policy, event->benchmark_result, &stream, &first_field));
  return loom_output_stream_write_cstring(&stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_append_benchmark(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_benchmark_result_event_t* event) {
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_snapshot_append_work_item(state, event));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_begin_array_entry(
      &state->benchmarks_json, &state->first_benchmark));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&state->benchmarks_json, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_candidate_fields(
      event->candidate, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_work_item_field(
      event->work_item_index, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_benchmark_fields(
      event->benchmark_plan, event->case_plan, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "status",
      iree_benchmark_loom_snapshot_result_status(event->benchmark_result)));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      &stream, &first_field, "sample_compilation",
      event->benchmark_result->sample_compilation));
  if (event->benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        &stream, &first_field, "sample_index",
        event->benchmark_result->sample_ordinal));
  }
  if (event->work_item_index == IREE_BENCHMARK_LOOM_INDEX_INVALID) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        &stream, &first_field, "result"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
    bool first_result_field = true;
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_measurement_fields(
        event->policy, event->benchmark_result, &stream, &first_result_field));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  }
  return loom_output_stream_write_cstring(&stream, "}");
}

static iree_string_view_t iree_benchmark_loom_snapshot_work_item_kind_name(
    iree_benchmark_loom_work_item_kind_t kind) {
  switch (kind) {
    case IREE_BENCHMARK_LOOM_WORK_ITEM_CASE_END_TO_END:
      return IREE_SV("case_end_to_end");
    case IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE:
      return IREE_SV("dispatch_sample");
    case IREE_BENCHMARK_LOOM_WORK_ITEM_NONE:
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t iree_benchmark_loom_snapshot_write_sample_range_fields(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t begin_sample, iree_host_size_t end_sample,
    bool has_case_sample_ordinal, iree_host_size_t case_sample_ordinal,
    loom_output_stream_t* stream, bool* first_field) {
  if (has_case_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, first_field, "benchmark_sample_index", begin_sample));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
        stream, first_field, "case_sample_index", case_sample_ordinal));
    return iree_benchmark_loom_write_sample_fields_json(
        module, case_plan, case_sample_ordinal, stream);
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, first_field, "benchmark_sample_begin", begin_sample));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      stream, first_field, "benchmark_sample_end", end_sample));
  return iree_benchmark_loom_write_json_size_field(
      stream, first_field, "sample_count", end_sample - begin_sample);
}

static iree_status_t iree_benchmark_loom_snapshot_append_planned_work_item(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_work_plan_event_t* event,
    const iree_benchmark_loom_work_item_t* work_item) {
  const iree_benchmark_loom_work_plan_t* work_plan = event->work_plan;
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_begin_array_entry(
      &state->work_items_json, &state->first_work_item));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&state->work_items_json, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      &stream, &first_field, "work_item_index", work_item->work_item_index));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "status", IREE_SV("planned")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "kind",
      iree_benchmark_loom_snapshot_work_item_kind_name(work_item->kind)));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      &stream, &first_field, "representative_candidate_index",
      selection->identity.candidate_index));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "representative_candidate_id",
      selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_benchmark_fields(
      selection->benchmark_plan, selection->case_plan, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "measure", selection->policy.measure));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      &stream, &first_field, "sample_compilation",
      work_item->sample_compilation));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_sample_range_fields(
      event->module, selection->case_plan, work_item->begin_benchmark_sample,
      work_item->end_benchmark_sample, work_item->has_case_sample_ordinal,
      work_item->case_sample_ordinal, &stream, &first_field));
  return loom_output_stream_write_cstring(&stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_append_planned_benchmark(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_work_plan_event_t* event,
    const iree_benchmark_loom_logical_sample_t* logical_sample) {
  const iree_benchmark_loom_work_plan_t* work_plan = event->work_plan;
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan->selected_benchmarks[logical_sample->selection_index];

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_begin_array_entry(
      &state->benchmarks_json, &state->first_benchmark));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&state->benchmarks_json, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_candidate_fields(
      &selection->identity, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_work_item_field(
      logical_sample->work_item_index, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_benchmark_fields(
      selection->benchmark_plan, selection->case_plan, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "status", IREE_SV("planned")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "measure", selection->policy.measure));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      &stream, &first_field, "sample_compilation",
      logical_sample->sample_compilation));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_sample_range_fields(
      event->module, selection->case_plan,
      logical_sample->begin_benchmark_sample,
      logical_sample->end_benchmark_sample,
      logical_sample->has_case_sample_ordinal,
      logical_sample->case_sample_ordinal, &stream, &first_field));
  return loom_output_stream_write_cstring(&stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_append_work_plan(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_work_plan_event_t* event) {
  for (iree_host_size_t i = 0; i < event->work_plan->work_item_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_planned_work_item(
        state, event, &event->work_plan->work_items[i]));
  }
  for (iree_host_size_t i = 0; i < event->work_plan->logical_sample_count;
       ++i) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_planned_benchmark(
        state, event, &event->work_plan->logical_samples[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_snapshot_append_failure(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_failure_event_t* event) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_begin_array_entry(
      &state->failures_json, &state->first_failure));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&state->failures_json, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "stage", event->stage));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "kind", event->kind));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      &stream, &first_field, "message", event->message));
  if (event->diagnostics != NULL) {
    if (event->diagnostics->error_count != 0) {
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
          &stream, &first_field, "diagnostic_error_count",
          event->diagnostics->error_count));
    }
    if (event->diagnostics->warning_count != 0) {
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
          &stream, &first_field, "diagnostic_warning_count",
          event->diagnostics->warning_count));
    }
    if (event->diagnostics->remark_count != 0) {
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
          &stream, &first_field, "diagnostic_remark_count",
          event->diagnostics->remark_count));
    }
    iree_string_view_t diagnostics_json =
        iree_string_builder_view(&event->diagnostics->output);
    if (!iree_string_view_is_empty(diagnostics_json)) {
      IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
          &stream, &first_field, "diagnostics"));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "["));
      IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, diagnostics_json));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]"));
    }
  }
  return loom_output_stream_write_cstring(&stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_append_repetition(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_benchmark_repetition_event_t* event) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_begin_array_entry(
      &state->repetitions_json, &state->first_repetition));
  const iree_benchmark_loom_selected_benchmark_t* selection =
      event->candidate->selection;
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&state->repetitions_json, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_candidate_fields(
      &selection->identity, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "baseline_candidate_id",
      event->baseline->candidate_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "comparison_group", event->comparison_group));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "method", event->method));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      &stream, &first_field, "order_index", event->order_index));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      &stream, &first_field, "repetition_index", event->repetition_index));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      &stream, &first_field, "schedule_token"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(&stream, "\"%c\"",
                                                       event->schedule_token));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "status",
      iree_benchmark_loom_snapshot_result_status(event->benchmark_result)));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_measurement_fields(
      &selection->policy, event->benchmark_result, &stream, &first_field));
  return loom_output_stream_write_cstring(&stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_append_comparison(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_comparison_event_t* event) {
  if (event->baseline->sample_count == 0 ||
      event->candidate->sample_count == 0) {
    return iree_ok_status();
  }

  loom_run_benchmark_timing_stats_t baseline_p50 = {0};
  loom_run_benchmark_timing_stats_t candidate_p50 = {0};
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      event->baseline->p50_samples, event->baseline->sample_count,
      &baseline_p50));
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      event->candidate->p50_samples, event->candidate->sample_count,
      &candidate_p50));

  const double baseline_p50_ns = (double)baseline_p50.p50_ns;
  const double candidate_p50_ns = (double)candidate_p50.p50_ns;
  const double ratio_p50 =
      baseline_p50_ns == 0.0 ? 0.0 : candidate_p50_ns / baseline_p50_ns;
  const double speedup_p50 =
      candidate_p50_ns == 0.0 ? 0.0 : baseline_p50_ns / candidate_p50_ns;

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_begin_array_entry(
      &state->comparisons_json, &state->first_comparison));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&state->comparisons_json, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "comparison_group", event->comparison_group));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "method", event->method));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "baseline_candidate_id",
      event->baseline->selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      &stream, &first_field, "candidate_id",
      event->candidate->selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      &stream, &first_field, "baseline_repetition_count",
      event->baseline->sample_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_size_field(
      &stream, &first_field, "candidate_repetition_count",
      event->candidate->sample_count));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      &stream, &first_field, "p50_ns"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      "{\"baseline\":%" PRIi64 ",\"candidate\":%" PRIi64
      ",\"ratio\":%.6f,\"speedup\":%.6f}",
      baseline_p50.p50_ns, candidate_p50.p50_ns, ratio_p50, speedup_p50));
  return loom_output_stream_write_cstring(&stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_emit_event(
    void* user_data, const iree_benchmark_loom_event_t* event) {
  iree_benchmark_loom_snapshot_state_t* state =
      (iree_benchmark_loom_snapshot_state_t*)user_data;
  switch (event->kind) {
    case IREE_BENCHMARK_LOOM_EVENT_RUN:
      return iree_benchmark_loom_snapshot_append_run(state, &event->run);
    case IREE_BENCHMARK_LOOM_EVENT_PLAN:
      return iree_benchmark_loom_snapshot_append_plan(state, &event->plan);
    case IREE_BENCHMARK_LOOM_EVENT_SUMMARY:
      return iree_benchmark_loom_snapshot_append_summary(state,
                                                         &event->summary);
    case IREE_BENCHMARK_LOOM_EVENT_DEVICE:
      return iree_benchmark_loom_snapshot_append_device(state, &event->device);
    case IREE_BENCHMARK_LOOM_EVENT_COMPILE:
      return iree_ok_status();
    case IREE_BENCHMARK_LOOM_EVENT_SAMPLE:
      return iree_benchmark_loom_snapshot_append_sample(state, &event->sample);
    case IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT:
      return iree_benchmark_loom_snapshot_append_benchmark(
          state, &event->benchmark_result);
    case IREE_BENCHMARK_LOOM_EVENT_PROFILE:
      return iree_ok_status();
    case IREE_BENCHMARK_LOOM_EVENT_FAILURE:
      return iree_benchmark_loom_snapshot_append_failure(state,
                                                         &event->failure);
    case IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_REPETITION:
      return iree_benchmark_loom_snapshot_append_repetition(
          state, &event->benchmark_repetition);
    case IREE_BENCHMARK_LOOM_EVENT_COMPARISON:
      return iree_benchmark_loom_snapshot_append_comparison(state,
                                                            &event->comparison);
    case IREE_BENCHMARK_LOOM_EVENT_WORK_PLAN:
      return iree_benchmark_loom_snapshot_append_work_plan(state,
                                                           &event->work_plan);
    case IREE_BENCHMARK_LOOM_EVENT_NONE:
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported benchmark event kind %d",
                              (int)event->kind);
  }
}

iree_status_t iree_benchmark_loom_snapshot_sink_initialize(
    iree_allocator_t allocator,
    iree_benchmark_loom_snapshot_sink_t* out_snapshot) {
  IREE_ASSERT_ARGUMENT(out_snapshot);
  memset(out_snapshot, 0, sizeof(*out_snapshot));

  iree_benchmark_loom_snapshot_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = allocator;
  state->sample_compilation_mode = IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE;
  state->first_benchmark = true;
  state->first_work_item = true;
  state->first_failed_sample = true;
  state->first_failure = true;
  state->first_repetition = true;
  state->first_comparison = true;
  iree_string_builder_initialize(allocator, &state->device_json);
  iree_string_builder_initialize(allocator, &state->benchmarks_json);
  iree_string_builder_initialize(allocator, &state->work_items_json);
  iree_string_builder_initialize(allocator, &state->failed_samples_json);
  iree_string_builder_initialize(allocator, &state->failures_json);
  iree_string_builder_initialize(allocator, &state->repetitions_json);
  iree_string_builder_initialize(allocator, &state->comparisons_json);
  out_snapshot->state = state;
  return iree_ok_status();
}

void iree_benchmark_loom_snapshot_sink_deinitialize(
    iree_benchmark_loom_snapshot_sink_t* snapshot) {
  if (snapshot == NULL || snapshot->state == NULL) {
    return;
  }
  iree_benchmark_loom_snapshot_state_t* state =
      iree_benchmark_loom_snapshot_state(snapshot);
  iree_allocator_free(state->host_allocator, state->emitted_work_item_indexes);
  iree_string_builder_deinitialize(&state->comparisons_json);
  iree_string_builder_deinitialize(&state->repetitions_json);
  iree_string_builder_deinitialize(&state->failures_json);
  iree_string_builder_deinitialize(&state->failed_samples_json);
  iree_string_builder_deinitialize(&state->work_items_json);
  iree_string_builder_deinitialize(&state->benchmarks_json);
  iree_string_builder_deinitialize(&state->device_json);
  iree_allocator_free(state->host_allocator,
                      state->artifact_bundle_policy_storage);
  iree_allocator_free(state->host_allocator,
                      state->artifact_bundle_dir_storage);
  iree_allocator_free(state->host_allocator,
                      state->profile_artifacts_dir_storage);
  iree_allocator_free(state->host_allocator, state->file_output_dir_storage);
  iree_allocator_free(state->host_allocator, state->results_path_storage);
  iree_allocator_free(state->host_allocator, state->source_storage);
  iree_allocator_free(state->host_allocator, state->run_id_storage);
  iree_allocator_t host_allocator = state->host_allocator;
  iree_allocator_free(host_allocator, state);
  snapshot->state = NULL;
}

void iree_benchmark_loom_snapshot_event_sink_initialize(
    iree_benchmark_loom_snapshot_sink_t* snapshot,
    iree_benchmark_loom_event_sink_t* out_sink) {
  IREE_ASSERT_ARGUMENT(snapshot);
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = (iree_benchmark_loom_event_sink_t){
      .emit = iree_benchmark_loom_snapshot_emit_event,
      .user_data = iree_benchmark_loom_snapshot_state(snapshot),
  };
}

static iree_status_t iree_benchmark_loom_snapshot_append_run_json(
    const iree_benchmark_loom_snapshot_state_t* state,
    loom_output_stream_t* stream, bool* first_field) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, first_field, "tool", IREE_SV("iree-benchmark-loom")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, first_field, "run_id", state->run_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, first_field, "source", state->source));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, first_field, "results_path", state->results_path));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, first_field, "file_output_dir", state->file_output_dir));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_optional_string_field(
      stream, first_field, "profile_artifacts_dir",
      state->profile_artifacts_dir));
  if (!iree_string_view_is_empty(state->artifact_bundle_dir)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        stream, first_field, "artifact_bundle"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
    bool first_bundle_field = true;
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, &first_bundle_field, "dir", state->artifact_bundle_dir));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, &first_bundle_field, "policy", state->artifact_bundle_policy));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, first_field, "output_format", IREE_SV("snapshot")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "dry_run"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, state->dry_run ? "true" : "false"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, first_field, "sample_compilation",
      iree_benchmark_loom_sample_compilation_mode_name(
          state->sample_compilation_mode)));
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_snapshot_append_summary_json(
    const iree_benchmark_loom_snapshot_state_t* state,
    loom_output_stream_t* stream, bool* first_field) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, "summary"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      "\"planned\":{\"case_count\":%" PRIhsz ",\"benchmark_count\":%" PRIhsz
      ",\"selected_benchmark_count\":%" PRIhsz
      ",\"logical_sample_count\":%" PRIhsz ",\"work_item_count\":%" PRIhsz "}",
      state->planned_case_count, state->planned_benchmark_count,
      state->selected_benchmark_count, state->logical_sample_count,
      state->work_item_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"failures\":{\"row_count\":%" PRIhsz
      ",\"failed_benchmark_count\":%" PRIhsz "}",
      state->failure_count, state->failed_benchmark_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"correctness\":{\"sample_count\":%" PRIhsz
      ",\"failed_sample_count\":%" PRIhsz "}",
      state->correctness_sample_count, state->correctness_failed_sample_count));
  if (state->artifact_bundle_enabled || state->fixture_read_count != 0 ||
      state->file_output_count != 0 || state->profile_count != 0 ||
      state->compile_report_count != 0 || state->artifact_manifest_count != 0 ||
      state->target_artifact_count != 0 || state->target_listing_count != 0 ||
      state->hal_executable_count != 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"artifacts\":{\"bundle_enabled\":%s,\"fixture_read_count\":%" PRIhsz
        ",\"file_output_count\":%" PRIhsz ",\"profile_count\":%" PRIhsz
        ",\"compile_report_count\":%" PRIhsz
        ",\"artifact_manifest_count\":%" PRIhsz
        ",\"target_artifact_count\":%" PRIhsz
        ",\"target_listing_count\":%" PRIhsz
        ",\"hal_executable_count\":%" PRIhsz "}",
        state->artifact_bundle_enabled ? "true" : "false",
        state->fixture_read_count, state->file_output_count,
        state->profile_count, state->compile_report_count,
        state->artifact_manifest_count, state->target_artifact_count,
        state->target_listing_count, state->hal_executable_count));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_snapshot_append_builder_array(
    const char* name, const iree_string_builder_t* array_entries,
    loom_output_stream_t* stream, bool* first_field) {
  iree_string_view_t entries = iree_string_builder_view(array_entries);
  if (iree_string_view_is_empty(entries)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, first_field, name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, entries));
  return loom_output_stream_write_cstring(stream, "]");
}

iree_status_t iree_benchmark_loom_snapshot_sink_append_json(
    const iree_benchmark_loom_snapshot_sink_t* snapshot,
    iree_string_builder_t* output) {
  IREE_ASSERT_ARGUMENT(snapshot);
  IREE_ASSERT_ARGUMENT(output);
  const iree_benchmark_loom_snapshot_state_t* state =
      iree_benchmark_loom_snapshot_state(snapshot);
  if (state == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "benchmark snapshot sink is not initialized");
  }
  if (!state->run_seen) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "benchmark snapshot has no run metadata");
  }
  if (!state->summary_seen) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "benchmark snapshot has no terminal summary");
  }

  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_run_json(
      state, &stream, &first_field));
  if (state->has_device_json) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
        &stream, &first_field, "device"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        &stream, iree_string_builder_view(&state->device_json)));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_summary_json(
      state, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_builder_array(
      "work_items", &state->work_items_json, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_builder_array(
      "benchmarks", &state->benchmarks_json, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_builder_array(
      "failed_samples", &state->failed_samples_json, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_builder_array(
      "failures", &state->failures_json, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_builder_array(
      "repetitions", &state->repetitions_json, &stream, &first_field));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_builder_array(
      "comparisons", &state->comparisons_json, &stream, &first_field));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_snapshot_sink_write(
    const iree_benchmark_loom_snapshot_sink_t* snapshot,
    iree_string_view_t path) {
  IREE_ASSERT_ARGUMENT(snapshot);
  iree_benchmark_loom_snapshot_state_t* state =
      iree_benchmark_loom_snapshot_state(snapshot);
  if (state == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "benchmark snapshot sink is not initialized");
  }
  iree_string_builder_t output;
  iree_string_builder_initialize(state->host_allocator, &output);
  iree_status_t status =
      iree_benchmark_loom_snapshot_sink_append_json(snapshot, &output);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_create_parent_directory(
        iree_string_view_trim(path), state->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        path, iree_string_builder_view(&output), state->host_allocator);
  }
  iree_string_builder_deinitialize(&output);
  return status;
}
