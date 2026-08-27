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
  // Sanitizer checks and reporting mode used for compiler-backed work.
  loom_sanitizer_options_t sanitizer;
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
  // Final aggregate run counts emitted under the summary object.
  iree_benchmark_loom_summary_counts_t summary;
  // True after |device_json| receives a device object.
  bool has_device_json;
  // JSON object for the selected device.
  iree_string_builder_t device_json;
  // JSON array entries for logical benchmark rows.
  loom_json_value_list_t benchmarks;
  // JSON array entries for deduplicated physical work results.
  loom_json_value_list_t work_items;
  // JSON array entries for failed correctness samples.
  loom_json_value_list_t failed_samples;
  // JSON array entries for parse, verify, and infrastructure failures.
  loom_json_value_list_t failures;
  // JSON array entries for interleaved comparison repetitions.
  loom_json_value_list_t repetitions;
  // JSON array entries for interleaved comparison summaries.
  loom_json_value_list_t comparisons;
  // Physical work item indexes already emitted into |work_items|.
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

static iree_status_t iree_benchmark_loom_snapshot_write_candidate_fields(
    const iree_benchmark_loom_candidate_identity_t* candidate,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("candidate_id"), candidate->candidate_id));
  return loom_json_object_write_host_size_field(
      object, IREE_SV("candidate_index"), candidate->candidate_index);
}

static iree_status_t iree_benchmark_loom_snapshot_write_work_item_field(
    iree_host_size_t work_item_index, loom_json_object_writer_t* object) {
  if (work_item_index == IREE_BENCHMARK_LOOM_INDEX_INVALID) {
    return iree_ok_status();
  }
  return loom_json_object_write_host_size_field(
      object, IREE_SV("work_item_index"), work_item_index);
}

static iree_status_t iree_benchmark_loom_snapshot_write_benchmark_fields(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("benchmark"), benchmark_plan->name));
  return loom_json_object_write_string_field(object, IREE_SV("case"),
                                             case_plan->name);
}

static iree_string_view_t iree_benchmark_loom_snapshot_result_state(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  if (!iree_string_view_is_empty(benchmark_result->state)) {
    return benchmark_result->state;
  }
  if (benchmark_result->executed) {
    return benchmark_result->passed ? IREE_SV("ok") : IREE_SV("failed");
  }
  return IREE_SV("skipped");
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
  state->sanitizer = event->sanitizer;
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
  state->summary.planned_case_count = event->planned_case_count;
  state->summary.planned_benchmark_count = event->planned_benchmark_count;
  state->summary.selected_benchmark_count = event->selected_benchmark_count;
  state->summary.logical_sample_count = event->logical_sample_count;
  state->summary.work_item_count = event->work_item_count;
  state->summary.failure_count = event->failure_count;
  state->summary.failed_benchmark_count = event->failed_benchmark_count;
  state->summary.correctness_sample_count = event->correctness_sample_count;
  state->summary.correctness_failed_sample_count =
      event->correctness_failed_sample_count;
  state->dry_run = event->dry_run;
  state->summary.artifact_bundle_enabled =
      event->artifact_bundle != NULL && event->artifact_bundle->enabled;
  state->summary.fixture_read_count =
      iree_benchmark_loom_artifact_bundle_file_count(
          event->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_FIXTURE_READ);
  state->summary.file_output_count =
      iree_benchmark_loom_artifact_bundle_file_count(
          event->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_OUTPUT);
  state->summary.profile_count = iree_benchmark_loom_artifact_bundle_file_count(
      event->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE);
  state->summary.compile_report_count =
      iree_benchmark_loom_artifact_bundle_file_count(
          event->artifact_bundle,
          IREE_BENCHMARK_LOOM_BUNDLE_FILE_COMPILE_REPORT);
  state->summary.artifact_manifest_count =
      iree_benchmark_loom_artifact_bundle_file_count(
          event->artifact_bundle,
          IREE_BENCHMARK_LOOM_BUNDLE_FILE_ARTIFACT_MANIFEST);
  state->summary.target_artifact_count =
      iree_benchmark_loom_artifact_bundle_file_count(
          event->artifact_bundle,
          IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_ARTIFACT);
  state->summary.target_listing_count =
      iree_benchmark_loom_artifact_bundle_file_count(
          event->artifact_bundle,
          IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_LISTING);
  state->summary.hal_executable_count =
      iree_benchmark_loom_artifact_bundle_file_count(
          event->artifact_bundle,
          IREE_BENCHMARK_LOOM_BUNDLE_FILE_HAL_EXECUTABLE);
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_hal_context_identity_fields_json(event->context,
                                                                 &object));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  state->has_device_json = true;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_snapshot_append_sample(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_sample_event_t* event) {
  if (event->sample_result->passed) {
    return iree_ok_status();
  }
  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&state->failed_samples, &stream));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_candidate_fields(
      event->candidate, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_work_item_field(
      event->work_item_index, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_benchmark_fields(
      event->benchmark_plan, event->case_plan, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("benchmark_sample_index"),
      event->benchmark_sample_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("case_sample_index"), event->case_sample_ordinal));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
      event->module, event->case_plan, event->case_sample_ordinal, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("sample_result")));
  IREE_RETURN_IF_ERROR(loom_testbench_case_sample_result_write_json(
      event->sample_result, &stream));
  return loom_json_object_end(&object);
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
  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&state->work_items, &stream));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("work_item_index"), event->work_item_index));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_benchmark_fields(
      event->benchmark_plan, event->case_plan, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"),
      iree_benchmark_loom_snapshot_result_state(event->benchmark_result)));
  if (event->benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
        event->module, event->case_plan,
        event->benchmark_result->sample_ordinal, &object));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_evidence_fields_json(
      event->policy, event->benchmark_result, event->correctness_sample_count,
      event->correctness_failed_sample_count, &object));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_snapshot_append_benchmark(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_benchmark_result_event_t* event) {
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_snapshot_append_work_item(state, event));
  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&state->benchmarks, &stream));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_candidate_fields(
      event->candidate, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_work_item_field(
      event->work_item_index, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_benchmark_fields(
      event->benchmark_plan, event->case_plan, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"),
      iree_benchmark_loom_snapshot_result_state(event->benchmark_result)));
  if (event->benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("sample_index"),
        event->benchmark_result->sample_ordinal));
  }
  if (event->work_item_index == IREE_BENCHMARK_LOOM_INDEX_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("result")));
    loom_json_object_writer_t result_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &result_object));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_benchmark_evidence_fields_json(
            event->policy, event->benchmark_result,
            event->correctness_sample_count,
            event->correctness_failed_sample_count, &result_object));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&result_object));
  }
  return loom_json_object_end(&object);
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
    loom_json_object_writer_t* object) {
  if (has_case_sample_ordinal) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        object, IREE_SV("benchmark_sample_index"), begin_sample));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        object, IREE_SV("case_sample_index"), case_sample_ordinal));
    return iree_benchmark_loom_write_sample_fields_json(
        module, case_plan, case_sample_ordinal, object);
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("benchmark_sample_begin"), begin_sample));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("benchmark_sample_end"), end_sample));
  return loom_json_object_write_host_size_field(object, IREE_SV("sample_count"),
                                                end_sample - begin_sample);
}

static iree_status_t iree_benchmark_loom_snapshot_append_planned_work_item(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_work_plan_event_t* event,
    const iree_benchmark_loom_work_item_t* work_item) {
  const iree_benchmark_loom_work_plan_t* work_plan = event->work_plan;
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];

  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&state->work_items, &stream));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("work_item_index"), work_item->work_item_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"), IREE_SV("planned")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      iree_benchmark_loom_snapshot_work_item_kind_name(work_item->kind)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("representative_candidate_index"),
      selection->identity.candidate_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("representative_candidate_id"),
      selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_benchmark_fields(
      selection->benchmark_plan, selection->case_plan, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("measure"), selection->policy.measure));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_sample_range_fields(
      event->module, selection->case_plan, work_item->begin_benchmark_sample,
      work_item->end_benchmark_sample, work_item->has_case_sample_ordinal,
      work_item->case_sample_ordinal, &object));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_snapshot_append_planned_benchmark(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_work_plan_event_t* event,
    const iree_benchmark_loom_logical_sample_t* logical_sample) {
  const iree_benchmark_loom_work_plan_t* work_plan = event->work_plan;
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan->selected_benchmarks[logical_sample->selection_index];

  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&state->benchmarks, &stream));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_candidate_fields(
      &selection->identity, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_work_item_field(
      logical_sample->work_item_index, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_benchmark_fields(
      selection->benchmark_plan, selection->case_plan, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"), IREE_SV("planned")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("measure"), selection->policy.measure));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_sample_range_fields(
      event->module, selection->case_plan,
      logical_sample->begin_benchmark_sample,
      logical_sample->end_benchmark_sample,
      logical_sample->has_case_sample_ordinal,
      logical_sample->case_sample_ordinal, &object));
  return loom_json_object_end(&object);
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
  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&state->failures, &stream));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("stage"), event->stage));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"), event->kind));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &object, IREE_SV("message"), event->message));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_diagnostic_capture_fields_json(
      event->diagnostics, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_planning_issue_fields_json(
      event->testbench_plan, event->planning_issues,
      event->planning_issue_count, &object));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_snapshot_append_repetition(
    iree_benchmark_loom_snapshot_state_t* state,
    const iree_benchmark_loom_benchmark_repetition_event_t* event) {
  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&state->repetitions, &stream));
  const iree_benchmark_loom_selected_benchmark_t* selection =
      event->candidate->selection;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_write_candidate_fields(
      &selection->identity, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("baseline_candidate_id"),
      event->baseline->candidate_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("comparison_group"), event->comparison_group));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("method"), event->method));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("order_index"), event->order_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("repetition_index"), event->repetition_index));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("schedule_token")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(&stream, "\"%c\"",
                                                       event->schedule_token));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"),
      iree_benchmark_loom_snapshot_result_state(event->benchmark_result)));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_benchmark_evidence_fields_json(
      &selection->policy, event->benchmark_result,
      event->candidate->correctness_sample_count,
      event->candidate->correctness_failed_sample_count, &object));
  return loom_json_object_end(&object);
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

  loom_output_stream_t stream;
  IREE_RETURN_IF_ERROR(
      loom_json_value_list_begin_value(&state->comparisons, &stream));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("comparison_group"), event->comparison_group));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("method"), event->method));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("baseline_candidate_id"),
      event->baseline->selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("candidate_id"),
      event->candidate->selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("baseline_repetition_count"),
      event->baseline->sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("candidate_repetition_count"),
      event->candidate->sample_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("p50_ns")));
  loom_json_object_writer_t p50_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &p50_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &p50_object, IREE_SV("baseline"), baseline_p50.p50_ns));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &p50_object, IREE_SV("candidate"), candidate_p50.p50_ns));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&p50_object, IREE_SV("ratio")));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(&stream, "%.6f", ratio_p50));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&p50_object, IREE_SV("speedup")));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(&stream, "%.6f", speedup_p50));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&p50_object));
  return loom_json_object_end(&object);
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
    case IREE_BENCHMARK_LOOM_EVENT_PROFILE_REPLAY:
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
  iree_string_builder_initialize(allocator, &state->device_json);
  loom_json_value_list_t* value_lists[] = {
      &state->benchmarks, &state->work_items,  &state->failed_samples,
      &state->failures,   &state->repetitions, &state->comparisons,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(value_lists); ++i) {
    loom_json_value_list_initialize(allocator, value_lists[i]);
  }
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
  loom_json_value_list_deinitialize(&state->comparisons);
  loom_json_value_list_deinitialize(&state->repetitions);
  loom_json_value_list_deinitialize(&state->failures);
  loom_json_value_list_deinitialize(&state->failed_samples);
  loom_json_value_list_deinitialize(&state->work_items);
  loom_json_value_list_deinitialize(&state->benchmarks);
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
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("tool"), IREE_SV("iree-benchmark-loom")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("run_id"), state->run_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("source"), state->source));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("results_path"), state->results_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("file_output_dir"), state->file_output_dir));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      object, IREE_SV("profile_artifacts_dir"), state->profile_artifacts_dir));
  if (!iree_string_view_is_empty(state->artifact_bundle_dir)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(object, IREE_SV("artifact_bundle")));
    loom_json_object_writer_t bundle_object;
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin(object->stream, &bundle_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &bundle_object, IREE_SV("dir"), state->artifact_bundle_dir));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &bundle_object, IREE_SV("policy"), state->artifact_bundle_policy));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&bundle_object));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("output_format"), IREE_SV("snapshot")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      object, IREE_SV("dry_run"), state->dry_run));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("sanitizer")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sanitizer_options_json(
      &state->sanitizer, object->stream));
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_snapshot_append_summary_json(
    const iree_benchmark_loom_snapshot_state_t* state,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("summary")));
  return iree_benchmark_loom_write_summary_counts_json(&state->summary,
                                                       object->stream);
}

static iree_status_t iree_benchmark_loom_snapshot_append_array(
    iree_string_view_t name, const loom_json_value_list_t* value_list,
    loom_json_object_writer_t* object) {
  if (value_list->count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, name));
  return loom_json_value_list_write_array(value_list, object->stream);
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
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_snapshot_append_run_json(state, &object));
  if (state->has_device_json) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("device")));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        &stream, iree_string_builder_view(&state->device_json)));
  }
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_snapshot_append_summary_json(state, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_array(
      IREE_SV("work_items"), &state->work_items, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_array(
      IREE_SV("benchmarks"), &state->benchmarks, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_array(
      IREE_SV("failed_samples"), &state->failed_samples, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_array(
      IREE_SV("failures"), &state->failures, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_array(
      IREE_SV("repetitions"), &state->repetitions, &object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_snapshot_append_array(
      IREE_SV("comparisons"), &state->comparisons, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
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
