// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/json_output.h"

#include "loom/util/json.h"

static const char* loom_check_outcome_string(loom_check_outcome_t outcome) {
  switch (outcome) {
    case LOOM_CHECK_PASS:
      return "pass";
    case LOOM_CHECK_FAIL:
      return "fail";
    case LOOM_CHECK_SKIP:
      return "skip";
  }
  return "unknown";
}

static bool loom_check_json_output_mode_is_valid(
    loom_check_json_output_mode_t output_mode) {
  switch (output_mode) {
    case LOOM_CHECK_JSON_OUTPUT_FAILURES:
    case LOOM_CHECK_JSON_OUTPUT_SUMMARY:
    case LOOM_CHECK_JSON_OUTPUT_ALL:
      return true;
  }
  return false;
}

static bool loom_check_json_should_write_case(
    loom_check_json_output_mode_t output_mode,
    const loom_check_result_t* result) {
  switch (output_mode) {
    case LOOM_CHECK_JSON_OUTPUT_FAILURES:
      return result->final_outcome == LOOM_CHECK_FAIL;
    case LOOM_CHECK_JSON_OUTPUT_SUMMARY:
      return false;
    case LOOM_CHECK_JSON_OUTPUT_ALL:
      return true;
  }
  return false;
}

static iree_status_t loom_check_json_write_source_range(
    loom_test_source_range_t source_range, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("start_byte"), source_range.start_byte));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("end_byte"), source_range.end_byte));
  return loom_json_object_end(&object);
}

static iree_status_t loom_check_json_write_optional_source_range_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    loom_test_source_range_t source_range) {
  if (loom_test_source_range_is_empty(source_range)) {
    return loom_json_object_write_null_field(object, name);
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, name));
  return loom_check_json_write_source_range(source_range, object->stream);
}

static iree_status_t loom_check_json_write_optional_string_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    iree_string_view_t string) {
  if (iree_string_view_is_empty(string)) {
    return loom_json_object_write_null_field(object, name);
  }
  return loom_json_object_write_string_field(object, name, string);
}

static iree_status_t loom_check_json_write_string_array_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    const iree_string_view_t* strings, iree_host_size_t string_count) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, name));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (iree_host_size_t i = 0; i < string_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_json_array_write_string_element(&array, strings[i]));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_check_json_write_object_list_field(
    loom_json_object_writer_t* object, iree_string_view_t name,
    const loom_json_value_list_t* list) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, name));
  return loom_json_value_list_write_array(list, object->stream);
}

static iree_status_t loom_check_json_write_update_edit_field(
    loom_json_object_writer_t* object, const loom_check_result_t* result) {
  if (!result->update_edit.present) {
    return loom_json_object_write_null_field(object, IREE_SV("update_edit"));
  }

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("update_edit")));
  loom_json_object_writer_t edit_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &edit_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &edit_object, IREE_SV("kind"),
      iree_make_cstring_view(
          loom_check_update_edit_kind_name(result->update_edit.value.kind))));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&edit_object, IREE_SV("range")));
  IREE_RETURN_IF_ERROR(loom_check_json_write_source_range(
      result->update_edit.value.range, object->stream));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &edit_object, IREE_SV("text"),
      iree_string_builder_view(&result->update_edit.text)));
  return loom_json_object_end(&edit_object);
}

static iree_status_t loom_check_json_write_annotation(
    const loom_test_annotation_t* annotation, bool matched,
    iree_host_size_t annotation_index, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), annotation_index + 1));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("source_range")));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_source_range(annotation->source_range, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("target_line"), annotation->target_line));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("expected")));
  loom_json_object_writer_t expected_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &expected_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &expected_object, IREE_SV("severity"),
      iree_make_cstring_view(
          loom_diagnostic_severity_name(annotation->severity))));
  if (annotation->domain == LOOM_ERROR_DOMAIN_COUNT_) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&expected_object, IREE_SV("domain")));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &expected_object, IREE_SV("domain"),
        iree_make_cstring_view(loom_error_domain_name(annotation->domain))));
  }
  if (annotation->code == 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&expected_object, IREE_SV("code")));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &expected_object, IREE_SV("code"), annotation->code));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
      &expected_object, IREE_SV("message_substrings")));
  loom_json_array_writer_t message_substrings;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &message_substrings));
  for (uint8_t i = 0; i < annotation->message_substring_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &message_substrings, annotation->message_substrings[i]));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&message_substrings));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&expected_object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_bool_field(&object, IREE_SV("matched"), matched));
  return loom_json_object_end(&object);
}

static iree_status_t loom_check_json_write_annotations(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    const loom_check_file_report_t* report,
    loom_json_object_writer_t* case_object) {
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(case_object, IREE_SV("annotations")));
  loom_json_array_writer_t annotations;
  IREE_RETURN_IF_ERROR(
      loom_json_array_begin(case_object->stream, &annotations));
  for (iree_host_size_t i = 0; i < test_case->annotation_count; ++i) {
    bool matched = false;
    IREE_RETURN_IF_ERROR(loom_check_file_report_annotation_matched(
        report, case_index, i, &matched));
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&annotations));
    IREE_RETURN_IF_ERROR(loom_check_json_write_annotation(
        &test_case->annotations[i], matched, i, case_object->stream));
  }
  return loom_json_array_end(&annotations);
}

static iree_status_t loom_check_json_write_case(
    const loom_test_case_t* test_case, const loom_check_result_t* result,
    const loom_check_file_report_t* report, iree_host_size_t case_index,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), case_index + 1));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("mode"),
      iree_make_cstring_view(loom_test_mode_name(test_case->mode))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("has_run_directive"), test_case->has_run_directive));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("has_requires_directive"),
      test_case->has_requires_directive));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range_field(
      &object, IREE_SV("requires_directive_range"),
      test_case->requires_directive_range));
  IREE_RETURN_IF_ERROR(loom_check_json_write_string_array_field(
      &object, IREE_SV("requirements"), test_case->requirements,
      test_case->requirement_count));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_string_field(
      &object, IREE_SV("pipeline"), test_case->pipeline));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_string_field(
      &object, IREE_SV("format_target"), test_case->format_target));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_string_field(
      &object, IREE_SV("emit_target"), test_case->emit_target));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("source_range")));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_source_range(test_case->source_range, stream));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range_field(
      &object, IREE_SV("separator_range"), test_case->separator_range));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range_field(
      &object, IREE_SV("run_directive_range"), test_case->run_directive_range));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("xfail"), test_case->xfail));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range_field(
      &object, IREE_SV("xfail_directive_range"),
      test_case->xfail_directive_range));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_string_field(
      &object, IREE_SV("xfail_reason"), test_case->xfail_reason));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("raw_outcome"),
      iree_make_cstring_view(loom_check_outcome_string(result->raw_outcome))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("final_outcome"),
      iree_make_cstring_view(
          loom_check_outcome_string(result->final_outcome))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("detail"), iree_string_builder_view(&result->detail)));

  if (result->diff_hunks.count == 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_null_field(&object, IREE_SV("diff")));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("diff")));
    loom_json_object_writer_t diff_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &diff_object));
    IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range_field(
        &diff_object, IREE_SV("expected_range"), test_case->expected_range));
    IREE_RETURN_IF_ERROR(loom_check_json_write_object_list_field(
        &diff_object, IREE_SV("hunks"), &result->diff_hunks));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&diff_object));
  }
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_update_edit_field(&object, result));
  IREE_RETURN_IF_ERROR(loom_check_json_write_object_list_field(
      &object, IREE_SV("annotation_edits"), &result->annotation_edits));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("input_range")));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_source_range(test_case->input_range, stream));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range_field(
      &object, IREE_SV("expected_separator_range"),
      test_case->expected_separator_range));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range_field(
      &object, IREE_SV("expected_range"), test_case->expected_range));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("has_expected_section"),
      test_case->has_expected_section));
  IREE_RETURN_IF_ERROR(loom_check_json_write_annotations(test_case, case_index,
                                                         report, &object));
  IREE_RETURN_IF_ERROR(loom_check_json_write_object_list_field(
      &object, IREE_SV("diagnostics"), &result->diagnostics));
  return loom_json_object_end(&object);
}

iree_status_t loom_check_json_write_file_result(
    iree_string_view_t filename, const loom_test_file_t* file,
    const loom_check_file_report_t* report, const loom_check_result_t* results,
    iree_host_size_t pass_count, iree_host_size_t fail_count,
    iree_host_size_t skip_count, loom_check_json_output_mode_t output_mode,
    loom_output_stream_t* stream) {
  if (!loom_check_json_output_mode_is_valid(output_mode)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid loom-check JSON output mode %d",
                            (int)output_mode);
  }
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("file"), filename));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("default_mode"),
      iree_make_cstring_view(loom_test_mode_name(file->default_mode))));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_string_field(
      &object, IREE_SV("default_pipeline"), file->default_pipeline));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_string_field(
      &object, IREE_SV("default_format_target"), file->default_format_target));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_string_field(
      &object, IREE_SV("default_emit_target"), file->default_emit_target));
  IREE_RETURN_IF_ERROR(loom_check_json_write_string_array_field(
      &object, IREE_SV("default_requirements"), file->default_requirements,
      file->default_requirement_count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("cases")));
  loom_json_array_writer_t cases;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &cases));
  for (iree_host_size_t i = 0; i < file->case_count; ++i) {
    const loom_test_case_t* test_case = &file->cases[i];
    const loom_check_result_t* result = &results[i];
    if (loom_check_json_should_write_case(output_mode, result)) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&cases));
      IREE_RETURN_IF_ERROR(
          loom_check_json_write_case(test_case, result, report, i, stream));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&cases));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("summary")));
  loom_json_object_writer_t summary;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &summary));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &summary, IREE_SV("total"), pass_count + fail_count + skip_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &summary, IREE_SV("passed"), pass_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &summary, IREE_SV("failed"), fail_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &summary, IREE_SV("skipped"), skip_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&summary));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(stream, '\n');
}
