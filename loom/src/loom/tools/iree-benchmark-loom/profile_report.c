// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/profile_report.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/tooling/profile/counter.h"
#include "iree/tooling/profile/summary.h"
#include "loom/tools/iree-benchmark-loom/options.h"
#include "loom/tools/iree-benchmark-loom/report.h"
#include "loom/util/json.h"

typedef struct iree_benchmark_loom_profile_counter_decode_summary_t {
  // Total non-empty JSONL rows decoded from the raw profile bundle.
  iree_host_size_t total_count;
  // Number of counter_summary rows decoded from the raw profile bundle.
  iree_host_size_t summary_count;
  // Number of counter_set metadata rows decoded from the raw profile bundle.
  iree_host_size_t counter_set_count;
  // Number of counter metadata rows decoded from the raw profile bundle.
  iree_host_size_t counter_count;
  // Number of aggregate counter_group rows decoded from the raw profile bundle.
  iree_host_size_t group_count;
  // Number of raw counter_sample rows decoded from the raw profile bundle.
  iree_host_size_t sample_count;
  // Number of decoded rows whose type is not known by this tuner build.
  iree_host_size_t unknown_count;
} iree_benchmark_loom_profile_counter_decode_summary_t;

typedef struct iree_benchmark_loom_profile_summary_decode_summary_t {
  // Total non-empty JSONL rows decoded from the raw profile bundle.
  iree_host_size_t total_count;
  // Number of bundle-level summary rows decoded from the raw profile bundle.
  iree_host_size_t summary_count;
  // Number of per-device summary rows decoded from the raw profile bundle.
  iree_host_size_t device_summary_count;
  // Number of decoded rows whose type is not known by this tuner build.
  iree_host_size_t unknown_count;
} iree_benchmark_loom_profile_summary_decode_summary_t;

static iree_status_t iree_benchmark_loom_read_file_into_builder(
    FILE* file, iree_string_builder_t* output) {
  if (fflush(file) != 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to flush temporary report stream: %s",
                            strerror(errno));
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to rewind temporary report stream: %s",
                            strerror(errno));
  }

  char buffer[4096];
  while (true) {
    const size_t read_count = fread(buffer, 1, sizeof(buffer), file);
    if (read_count != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          output, iree_make_string_view(buffer, read_count)));
    }
    if (read_count < sizeof(buffer)) {
      break;
    }
  }
  if (ferror(file)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to read temporary report stream: %s",
                            strerror(errno));
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_write_profile_artifact_identity_json(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, object));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, object));
  if (work_item_index != IREE_BENCHMARK_LOOM_INDEX_INVALID) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        object, IREE_SV("work_item_index"), work_item_index));
  }
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
        module, case_plan, benchmark_result->sample_ordinal, object));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("benchmark"), benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("case"), case_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("batch_size"),
      benchmark_result->hal_benchmark.timing.batch_size));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      object, IREE_SV("measurement_relationship"),
      IREE_SV("distinct_execution")));
  const loom_run_hal_profile_summary_t* profile =
      &benchmark_result->hal_benchmark.profile_replay;
  if (profile->has_artifact_path) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        object, IREE_SV("artifact_path"),
        iree_make_string_view(profile->artifact_path,
                              profile->artifact_path_length)));
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_summarize_profile_summary_line(
    iree_string_view_t line,
    iree_benchmark_loom_profile_summary_decode_summary_t* summary) {
  ++summary->total_count;

  char type_storage[64] = {0};
  iree_host_size_t type_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_string(
      line, IREE_SV("type"), IREE_SV("unknown"),
      iree_make_mutable_string_view(type_storage, sizeof(type_storage)),
      &type_length));
  const iree_string_view_t type =
      iree_make_string_view(type_storage, type_length);
  if (iree_string_view_equal(type, IREE_SV("summary"))) {
    ++summary->summary_count;
  } else if (iree_string_view_equal(type, IREE_SV("device_summary"))) {
    ++summary->device_summary_count;
  } else {
    ++summary->unknown_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_summarize_profile_summary_rows(
    iree_string_view_t summary_rows,
    iree_benchmark_loom_profile_summary_decode_summary_t* summary) {
  memset(summary, 0, sizeof(*summary));
  iree_string_view_t remaining = summary_rows;
  iree_status_t status = iree_ok_status();
  while (!iree_string_view_is_empty(remaining) && iree_status_is_ok(status)) {
    iree_string_view_t line = iree_string_view_empty();
    iree_string_view_split(remaining, '\n', &line, &remaining);
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line)) {
      continue;
    }
    status = iree_benchmark_loom_summarize_profile_summary_line(line, summary);
  }
  return status;
}

static iree_status_t iree_benchmark_loom_summarize_profile_counter_line(
    iree_string_view_t line,
    iree_benchmark_loom_profile_counter_decode_summary_t* summary) {
  ++summary->total_count;

  char type_storage[64] = {0};
  iree_host_size_t type_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_string(
      line, IREE_SV("type"), IREE_SV("unknown"),
      iree_make_mutable_string_view(type_storage, sizeof(type_storage)),
      &type_length));
  const iree_string_view_t type =
      iree_make_string_view(type_storage, type_length);
  if (iree_string_view_equal(type, IREE_SV("counter_summary"))) {
    ++summary->summary_count;
  } else if (iree_string_view_equal(type, IREE_SV("counter_set"))) {
    ++summary->counter_set_count;
  } else if (iree_string_view_equal(type, IREE_SV("counter"))) {
    ++summary->counter_count;
  } else if (iree_string_view_equal(type, IREE_SV("counter_group"))) {
    ++summary->group_count;
  } else if (iree_string_view_equal(type, IREE_SV("counter_sample"))) {
    ++summary->sample_count;
  } else {
    ++summary->unknown_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_summarize_profile_counter_rows(
    iree_string_view_t counter_rows,
    iree_benchmark_loom_profile_counter_decode_summary_t* summary) {
  memset(summary, 0, sizeof(*summary));
  iree_string_view_t remaining = counter_rows;
  iree_status_t status = iree_ok_status();
  while (!iree_string_view_is_empty(remaining) && iree_status_is_ok(status)) {
    iree_string_view_t line = iree_string_view_empty();
    iree_string_view_split(remaining, '\n', &line, &remaining);
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line)) {
      continue;
    }
    status = iree_benchmark_loom_summarize_profile_counter_line(line, summary);
  }
  return status;
}

iree_status_t iree_benchmark_loom_write_profile_counter_request_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("data_family_names")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_family_names_json(
      policy->hal_options.profile_data_families, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("counter_set_count"),
      policy->hal_options.profile_counter_set_count));
  if (policy->hal_options.profile_counter_set_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("counter_sets")));
    loom_json_array_writer_t counter_sets;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &counter_sets));
    for (iree_host_size_t i = 0;
         i < policy->hal_options.profile_counter_set_count; ++i) {
      const iree_hal_profile_counter_set_selection_t* counter_set =
          &policy->hal_options.profile_counter_sets[i];
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&counter_sets));
      loom_json_object_writer_t counter_set_object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &counter_set_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &counter_set_object, IREE_SV("name"), counter_set->name));
      IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
          &counter_set_object, IREE_SV("counter_count"),
          counter_set->counter_name_count));
      if (counter_set->counter_name_count != 0) {
        IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
            &counter_set_object, IREE_SV("counter_names")));
        loom_json_array_writer_t counter_names;
        IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &counter_names));
        for (iree_host_size_t j = 0; j < counter_set->counter_name_count; ++j) {
          IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
              &counter_names, counter_set->counter_names[j]));
        }
        IREE_RETURN_IF_ERROR(loom_json_array_end(&counter_names));
      }
      IREE_RETURN_IF_ERROR(loom_json_object_end(&counter_set_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&counter_sets));
  }
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_profile_summary_request_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("data_family_names")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_family_names_json(
      policy->hal_options.profile_data_families, stream));
  return loom_json_object_end(&object);
}

static iree_status_t
iree_benchmark_loom_write_profile_summary_decode_summary_json(
    const iree_benchmark_loom_profile_summary_decode_summary_t* summary,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("total"), summary->total_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("summary"), summary->summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("device_summary"), summary->device_summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("unknown"), summary->unknown_count));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_profile_summary_state_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_profile_summary_decode_summary_t* decode_summary,
    iree_string_view_t state, iree_string_view_t state_code,
    iree_status_code_t error_code, iree_string_view_t error_message,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"), IREE_SV("profile_summary_state")));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("state"), state));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state_code"), state_code));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("request")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_profile_summary_request_json(policy, stream));
  if (decode_summary != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("decoded_rows")));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_profile_summary_decode_summary_json(
            decode_summary, stream));
  }
  if (error_code != IREE_STATUS_OK) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_status_field_json(
        error_code, error_message, &object));
  }
  return loom_json_object_end(&object);
}

static iree_status_t
iree_benchmark_loom_write_profile_counter_decode_summary_json(
    const iree_benchmark_loom_profile_counter_decode_summary_t* summary,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("total"), summary->total_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("summary"), summary->summary_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("counter_sets"), summary->counter_set_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("counters"), summary->counter_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("groups"), summary->group_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("samples"), summary->sample_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("unknown"), summary->unknown_count));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_profile_counter_state_json(
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_profile_counter_decode_summary_t* decode_summary,
    iree_string_view_t state, iree_string_view_t state_code,
    iree_status_code_t error_code, iree_string_view_t error_message,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("type"), IREE_SV("counter_decode_state")));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("state"), state));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state_code"), state_code));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("request")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_profile_counter_request_json(policy, stream));
  if (decode_summary != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("decoded_rows")));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_profile_counter_decode_summary_json(
            decode_summary, stream));
  }
  if (error_code != IREE_STATUS_OK) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_status_field_json(
        error_code, error_message, &object));
  }
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_append_profile_summary_state_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    const iree_benchmark_loom_profile_summary_decode_summary_t* decode_summary,
    iree_string_view_t state, iree_string_view_t state_code,
    iree_status_code_t error_code, iree_string_view_t error_message,
    iree_string_builder_t* profile_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(profile_output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("profile_summary")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_artifact_identity_json(
      run, candidate, work_item_index, module, benchmark_plan, case_plan,
      benchmark_result, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("profile_summary")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_summary_state_json(
      policy, decode_summary, state, state_code, error_code, error_message,
      &stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
}

static iree_status_t
iree_benchmark_loom_append_profile_summary_state_from_error(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_string_view_t state, iree_string_view_t state_code,
    iree_status_t error_status, iree_string_builder_t* profile_output) {
  const iree_status_code_t error_code = iree_status_code(error_status);
  const iree_string_view_t error_message = iree_status_message(error_status);
  iree_status_t append_status =
      iree_benchmark_loom_append_profile_summary_state_row(
          run, candidate, work_item_index, module, benchmark_plan, case_plan,
          policy, benchmark_result, /*decode_summary=*/NULL, state, state_code,
          error_code, error_message, profile_output);
  iree_status_free(error_status);
  return append_status;
}

static iree_status_t iree_benchmark_loom_append_profile_summary_payload_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_string_view_t summary_json, iree_string_builder_t* profile_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(profile_output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("profile_summary")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_artifact_identity_json(
      run, candidate, work_item_index, module, benchmark_plan, case_plan,
      benchmark_result, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("profile_summary")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, summary_json));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
}

static iree_status_t iree_benchmark_loom_append_profile_summary_rows(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_allocator_t allocator, iree_string_builder_t* profile_output) {
  const loom_run_hal_profile_summary_t* profile =
      &benchmark_result->hal_benchmark.profile_replay;
  if (!benchmark_result->has_hal_benchmark || !profile->requested) {
    return iree_ok_status();
  }
  const bool summary_requested =
      profile->has_artifact_path ||
      iree_benchmark_loom_profile_data_needs_artifact_data(
          profile->data_families);
  if (!summary_requested) {
    return iree_ok_status();
  }
  if (!profile->executed) {
    if (profile->has_error) {
      return iree_benchmark_loom_append_profile_summary_state_row(
          run, candidate, work_item_index, module, benchmark_plan, case_plan,
          policy, benchmark_result, /*decode_summary=*/NULL,
          IREE_SV("unavailable"), IREE_SV("profile_error"), profile->error_code,
          iree_make_string_view(profile->error_message,
                                profile->error_message_length),
          profile_output);
    }
    return iree_benchmark_loom_append_profile_summary_state_row(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        policy, benchmark_result, /*decode_summary=*/NULL,
        IREE_SV("unavailable"), IREE_SV("profile_not_executed"), IREE_STATUS_OK,
        iree_string_view_empty(), profile_output);
  }
  if (profile->has_error) {
    return iree_benchmark_loom_append_profile_summary_state_row(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        policy, benchmark_result, /*decode_summary=*/NULL,
        IREE_SV("unavailable"), IREE_SV("profile_error"), profile->error_code,
        iree_make_string_view(profile->error_message,
                              profile->error_message_length),
        profile_output);
  }
  if (!profile->has_artifact_path) {
    return iree_benchmark_loom_append_profile_summary_state_row(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        policy, benchmark_result, /*decode_summary=*/NULL,
        IREE_SV("unavailable"), IREE_SV("no_profile_artifact_path"),
        IREE_STATUS_OK, iree_string_view_empty(), profile_output);
  }

  FILE* summary_report_file = tmpfile();
  if (summary_report_file == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to create temporary profile summary file: "
                            "%s",
                            strerror(errno));
  }

  const iree_string_view_t artifact_path = iree_make_string_view(
      profile->artifact_path, profile->artifact_path_length);
  iree_status_t decode_status = iree_profile_summary_file(
      artifact_path, IREE_SV("jsonl"), summary_report_file, allocator);
  iree_string_builder_t summary_rows;
  iree_string_builder_initialize(allocator, &summary_rows);
  if (!iree_status_is_ok(decode_status)) {
    fclose(summary_report_file);
    iree_string_builder_deinitialize(&summary_rows);
    return iree_benchmark_loom_append_profile_summary_state_from_error(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        policy, benchmark_result, IREE_SV("raw_artifact_only"),
        IREE_SV("decode_failed"), decode_status, profile_output);
  }
  iree_status_t status = iree_benchmark_loom_read_file_into_builder(
      summary_report_file, &summary_rows);
  fclose(summary_report_file);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&summary_rows);
    return status;
  }

  const iree_string_view_t summary_rows_view =
      iree_string_builder_view(&summary_rows);
  iree_benchmark_loom_profile_summary_decode_summary_t decode_summary;
  status = iree_benchmark_loom_summarize_profile_summary_rows(summary_rows_view,
                                                              &decode_summary);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&summary_rows);
    return iree_benchmark_loom_append_profile_summary_state_from_error(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        policy, benchmark_result, IREE_SV("raw_artifact_only"),
        IREE_SV("decode_summary_failed"), status, profile_output);
  }
  const iree_string_view_t state_code =
      decode_summary.summary_count == 0 &&
              decode_summary.device_summary_count == 0
          ? IREE_SV("decoded_empty_profile_summary")
          : IREE_SV("decoded_profile_summary");
  status = iree_benchmark_loom_append_profile_summary_state_row(
      run, candidate, work_item_index, module, benchmark_plan, case_plan,
      policy, benchmark_result, &decode_summary, IREE_SV("decoded"), state_code,
      IREE_STATUS_OK, iree_string_view_empty(), profile_output);

  iree_string_view_t remaining = iree_string_builder_view(&summary_rows);
  while (!iree_string_view_is_empty(remaining) && iree_status_is_ok(status)) {
    iree_string_view_t line = iree_string_view_empty();
    iree_string_view_split(remaining, '\n', &line, &remaining);
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line)) {
      continue;
    }
    status = iree_benchmark_loom_append_profile_summary_payload_row(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        benchmark_result, line, profile_output);
  }
  iree_string_builder_deinitialize(&summary_rows);
  return status;
}

static iree_status_t iree_benchmark_loom_append_profile_counter_state_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    const iree_benchmark_loom_profile_counter_decode_summary_t* decode_summary,
    iree_string_view_t state, iree_string_view_t state_code,
    iree_status_code_t error_code, iree_string_view_t error_message,
    iree_string_builder_t* profile_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(profile_output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("profile_counter")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_artifact_identity_json(
      run, candidate, work_item_index, module, benchmark_plan, case_plan,
      benchmark_result, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("counter")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_counter_state_json(
      policy, decode_summary, state, state_code, error_code, error_message,
      &stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
}

static iree_status_t
iree_benchmark_loom_append_profile_counter_state_from_error(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_string_view_t state, iree_string_view_t state_code,
    iree_status_t error_status, iree_string_builder_t* profile_output) {
  const iree_status_code_t error_code = iree_status_code(error_status);
  const iree_string_view_t error_message = iree_status_message(error_status);
  iree_status_t append_status =
      iree_benchmark_loom_append_profile_counter_state_row(
          run, candidate, work_item_index, module, benchmark_plan, case_plan,
          policy, benchmark_result, /*decode_summary=*/NULL, state, state_code,
          error_code, error_message, profile_output);
  iree_status_free(error_status);
  return append_status;
}

static iree_status_t iree_benchmark_loom_append_profile_counter_payload_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_string_view_t counter_json, iree_string_builder_t* profile_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(profile_output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("profile_counter")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_profile_artifact_identity_json(
      run, candidate, work_item_index, module, benchmark_plan, case_plan,
      benchmark_result, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("counter")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, counter_json));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
}

static iree_status_t iree_benchmark_loom_append_profile_counter_rows(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_allocator_t allocator, iree_string_builder_t* profile_output) {
  const loom_run_hal_profile_summary_t* profile =
      &benchmark_result->hal_benchmark.profile_replay;
  if (!benchmark_result->has_hal_benchmark || !profile->requested ||
      !iree_benchmark_loom_profile_data_has_counter_data(
          profile->data_families)) {
    return iree_ok_status();
  }
  if (!profile->executed) {
    if (profile->has_error) {
      return iree_benchmark_loom_append_profile_counter_state_row(
          run, candidate, work_item_index, module, benchmark_plan, case_plan,
          policy, benchmark_result, /*decode_summary=*/NULL,
          IREE_SV("unavailable"), IREE_SV("profile_error"), profile->error_code,
          iree_make_string_view(profile->error_message,
                                profile->error_message_length),
          profile_output);
    }
    return iree_benchmark_loom_append_profile_counter_state_row(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        policy, benchmark_result, /*decode_summary=*/NULL,
        IREE_SV("unavailable"), IREE_SV("profile_not_executed"), IREE_STATUS_OK,
        iree_string_view_empty(), profile_output);
  }
  if (profile->has_error) {
    return iree_benchmark_loom_append_profile_counter_state_row(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        policy, benchmark_result, /*decode_summary=*/NULL,
        IREE_SV("unavailable"), IREE_SV("profile_error"), profile->error_code,
        iree_make_string_view(profile->error_message,
                              profile->error_message_length),
        profile_output);
  }
  if (!profile->has_artifact_path) {
    return iree_benchmark_loom_append_profile_counter_state_row(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        policy, benchmark_result, /*decode_summary=*/NULL,
        IREE_SV("unavailable"), IREE_SV("no_profile_artifact_path"),
        IREE_STATUS_OK, iree_string_view_empty(), profile_output);
  }

  FILE* counter_report_file = tmpfile();
  if (counter_report_file == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to create temporary counter report file: "
                            "%s",
                            strerror(errno));
  }

  const iree_string_view_t artifact_path = iree_make_string_view(
      profile->artifact_path, profile->artifact_path_length);
  iree_status_t decode_status = iree_profile_counter_file(
      artifact_path, IREE_SV("jsonl"), iree_string_view_empty(),
      /*id_filter=*/-1,
      /*emit_samples=*/false, counter_report_file, allocator);
  iree_string_builder_t counter_rows;
  iree_string_builder_initialize(allocator, &counter_rows);
  if (!iree_status_is_ok(decode_status)) {
    fclose(counter_report_file);
    iree_string_builder_deinitialize(&counter_rows);
    return iree_benchmark_loom_append_profile_counter_state_from_error(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        policy, benchmark_result, IREE_SV("raw_artifact_only"),
        IREE_SV("decode_failed"), decode_status, profile_output);
  }
  iree_status_t status = iree_benchmark_loom_read_file_into_builder(
      counter_report_file, &counter_rows);
  fclose(counter_report_file);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&counter_rows);
    return status;
  }

  const iree_string_view_t counter_rows_view =
      iree_string_builder_view(&counter_rows);
  iree_benchmark_loom_profile_counter_decode_summary_t decode_summary;
  status = iree_benchmark_loom_summarize_profile_counter_rows(counter_rows_view,
                                                              &decode_summary);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&counter_rows);
    return iree_benchmark_loom_append_profile_counter_state_from_error(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        policy, benchmark_result, IREE_SV("raw_artifact_only"),
        IREE_SV("decode_summary_failed"), status, profile_output);
  }
  const iree_string_view_t state_code =
      decode_summary.counter_count == 0 && decode_summary.group_count == 0 &&
              decode_summary.sample_count == 0
          ? IREE_SV("decoded_empty_counter_report")
          : IREE_SV("decoded_counter_report");
  status = iree_benchmark_loom_append_profile_counter_state_row(
      run, candidate, work_item_index, module, benchmark_plan, case_plan,
      policy, benchmark_result, &decode_summary, IREE_SV("decoded"), state_code,
      IREE_STATUS_OK, iree_string_view_empty(), profile_output);

  iree_string_view_t remaining = iree_string_builder_view(&counter_rows);
  while (!iree_string_view_is_empty(remaining) && iree_status_is_ok(status)) {
    iree_string_view_t line = iree_string_view_empty();
    iree_string_view_split(remaining, '\n', &line, &remaining);
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line)) {
      continue;
    }
    status = iree_benchmark_loom_append_profile_counter_payload_row(
        run, candidate, work_item_index, module, benchmark_plan, case_plan,
        benchmark_result, line, profile_output);
  }
  iree_string_builder_deinitialize(&counter_rows);
  return status;
}

iree_status_t iree_benchmark_loom_append_profile_replay_row(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_allocator_t allocator, iree_string_builder_t* profile_output) {
  if (!benchmark_result->has_hal_benchmark ||
      !benchmark_result->hal_benchmark.profile_replay.requested) {
    return iree_ok_status();
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(profile_output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("row"), IREE_SV("profile_replay")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_run_id_field_json(run, &object));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_candidate_identity_json(candidate, &object));
  if (work_item_index != IREE_BENCHMARK_LOOM_INDEX_INVALID) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("work_item_index"), work_item_index));
  }
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_sample_fields_json(
        module, case_plan, benchmark_result->sample_ordinal, &object));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("benchmark"), benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("case"), case_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("batch_size"),
      benchmark_result->hal_benchmark.timing.batch_size));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("profile_replay")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_hal_profile_replay_json(
      policy, benchmark_result, &stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, '\n'));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_append_profile_summary_rows(
      run, candidate, work_item_index, module, benchmark_plan, case_plan,
      policy, benchmark_result, allocator, profile_output));
  return iree_benchmark_loom_append_profile_counter_rows(
      run, candidate, work_item_index, module, benchmark_plan, case_plan,
      policy, benchmark_result, allocator, profile_output);
}
