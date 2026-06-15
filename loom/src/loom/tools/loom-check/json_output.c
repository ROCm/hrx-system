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
    loom_check_source_range_t source_range, loom_output_stream_t* stream) {
  return loom_output_stream_write_format(
      stream, "{\"start_byte\": %zu, \"end_byte\": %zu}",
      source_range.start_byte, source_range.end_byte);
}

static iree_status_t loom_check_json_write_optional_source_range(
    loom_check_source_range_t source_range, loom_output_stream_t* stream) {
  if (loom_check_source_range_is_empty(source_range)) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_check_json_write_source_range(source_range, stream);
}

static iree_status_t loom_check_json_write_optional_string(
    iree_string_view_t string, loom_output_stream_t* stream) {
  if (iree_string_view_is_empty(string)) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream, string);
}

static iree_status_t loom_check_json_write_string_array(
    const iree_string_view_t* strings, iree_host_size_t string_count,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  for (iree_host_size_t i = 0; i < string_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, strings[i]));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return iree_ok_status();
}

static iree_status_t loom_check_json_write_update_edit(
    const loom_check_result_t* result, loom_output_stream_t* stream) {
  if (!result->update_edit.present) {
    return loom_output_stream_write_cstring(stream, "null");
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\n"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "        \"kind\": "));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream,
      loom_check_update_edit_kind_name(result->update_edit.value.kind)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\n        \"range\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_source_range(
      result->update_edit.value.range, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\n        \"text\": "));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_string_builder_view(&result->update_edit.text)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n      }"));
  return iree_ok_status();
}

static iree_status_t loom_check_json_write_annotation(
    const loom_check_annotation_t* annotation, bool matched,
    iree_host_size_t annotation_index, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "        {\n"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "          \"index\": %zu,\n", annotation_index + 1));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "          \"source_range\": "));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_source_range(annotation->source_range, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "          \"target_line\": %zu,\n", annotation->target_line));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "          \"expected\": {\n"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "            \"severity\": "));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_diagnostic_severity_name(annotation->severity)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "            \"domain\": "));
  if (annotation->domain == LOOM_ERROR_DOMAIN_COUNT_) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null,\n"));
  } else {
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        stream, loom_error_domain_name(annotation->domain)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "            \"code\": "));
  if (annotation->code == 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null,\n"));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(stream, "%u,\n", annotation->code));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, "            \"message_substrings\": ["));
  for (uint8_t i = 0; i < annotation->message_substring_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, annotation->message_substrings[i]));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "]\n          },\n"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "          \"matched\": %s\n", matched ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "        }"));
  return iree_ok_status();
}

static iree_status_t loom_check_json_write_annotations(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    const loom_check_file_report_t* report, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"annotations\": ["));
  if (test_case->annotation_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n"));
    for (iree_host_size_t i = 0; i < test_case->annotation_count; ++i) {
      bool matched = false;
      IREE_RETURN_IF_ERROR(loom_check_file_report_annotation_matched(
          report, case_index, i, &matched));
      IREE_RETURN_IF_ERROR(loom_check_json_write_annotation(
          &test_case->annotations[i], matched, i, stream));
      if (i + 1 < test_case->annotation_count) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));
      } else {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n"));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "      "));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "],\n"));
  return iree_ok_status();
}

static iree_status_t loom_check_json_write_case(
    const loom_check_case_t* test_case, const loom_check_result_t* result,
    const loom_check_file_report_t* report, iree_host_size_t case_index,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "    {\n"));

  // "index": N (1-based for human readability)
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "      \"index\": %zu,\n", case_index + 1));

  // "mode": "<mode>"
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"mode\": \""));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, loom_check_mode_name(test_case->mode)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\",\n"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "      \"has_run_directive\": %s,\n",
      test_case->has_run_directive ? "true" : "false"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "      \"has_requires_directive\": %s,\n",
      test_case->has_requires_directive ? "true" : "false"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, "      \"requires_directive_range\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range(
      test_case->requires_directive_range, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"requirements\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_string_array(
      test_case->requirements, test_case->requirement_count, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"pipeline\": "));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_optional_string(test_case->pipeline, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"format_target\": "));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_optional_string(test_case->format_target, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"emit_target\": "));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_optional_string(test_case->emit_target, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"source_range\": "));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_source_range(test_case->source_range, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"separator_range\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range(
      test_case->separator_range, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, "      \"run_directive_range\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range(
      test_case->run_directive_range, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  // "xfail": true/false
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "      \"xfail\": %s,\n", test_case->xfail ? "true" : "false"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, "      \"xfail_directive_range\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range(
      test_case->xfail_directive_range, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"xfail_reason\": "));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_optional_string(test_case->xfail_reason, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  // "raw_outcome": "pass"/"fail"/"skip"
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "      \"raw_outcome\": \"%s\",\n",
      loom_check_outcome_string(result->raw_outcome)));

  // "final_outcome": "pass"/"fail"/"skip"
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "      \"final_outcome\": \"%s\",\n",
      loom_check_outcome_string(result->final_outcome)));

  // "detail": "<escaped detail>"
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"detail\": "));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_string_builder_view(&result->detail)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"diff\": "));
  if (result->diff_hunk_count == 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null,\n"));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream, "{\n        \"expected_range\": "));
    IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range(
        test_case->expected_range, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream, ",\n        \"hunks\": [\n          "));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        stream, iree_string_builder_view(&result->diff_hunk_json)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "\n        ]\n      },\n"));
  }

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"update_edit\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_update_edit(result, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, "      \"annotation_edits\": ["));
  if (result->annotation_edits.count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "\n        "));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        stream, iree_string_builder_view(&result->annotation_edits.json)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n      "));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "],\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"input_range\": "));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_source_range(test_case->input_range, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, "      \"expected_separator_range\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range(
      test_case->expected_separator_range, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"expected_range\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_source_range(
      test_case->expected_range, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "      \"has_expected_section\": %s,\n",
      test_case->has_expected_section ? "true" : "false"));

  IREE_RETURN_IF_ERROR(
      loom_check_json_write_annotations(test_case, case_index, report, stream));

  // "diagnostics": [<shared diagnostic JSON objects>]
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "      \"diagnostics\": ["));
  if (result->diagnostic_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "\n        "));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        stream, iree_string_builder_view(&result->diagnostic_json)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n      "));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]\n"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "    }"));
  return iree_ok_status();
}

iree_status_t loom_check_json_write_file_result(
    iree_string_view_t filename, const loom_check_file_t* file,
    const loom_check_file_report_t* report, const loom_check_result_t* results,
    iree_host_size_t pass_count, iree_host_size_t fail_count,
    iree_host_size_t skip_count, loom_check_json_output_mode_t output_mode,
    loom_output_stream_t* stream) {
  if (file->case_count > 0) {
  }
  if (!loom_check_json_output_mode_is_valid(output_mode)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid loom-check JSON output mode %d",
                            (int)output_mode);
  }

  // Open the root object.
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\n"));

  // "file": "<filename>"
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"file\": "));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, filename));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  // "default_mode": "<mode>"
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"default_mode\": \""));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, loom_check_mode_name(file->default_mode)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"default_pipeline\": "));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_optional_string(file->default_pipeline, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, "  \"default_format_target\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_optional_string(
      file->default_format_target, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"default_emit_target\": "));
  IREE_RETURN_IF_ERROR(
      loom_check_json_write_optional_string(file->default_emit_target, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"default_requirements\": "));
  IREE_RETURN_IF_ERROR(loom_check_json_write_string_array(
      file->default_requirements, file->default_requirement_count, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));

  // "cases": [...]
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"cases\": [\n"));

  bool has_written_case = false;
  for (iree_host_size_t i = 0; i < file->case_count; ++i) {
    const loom_check_case_t* test_case = &file->cases[i];
    const loom_check_result_t* result = &results[i];
    if (loom_check_json_should_write_case(output_mode, result)) {
      if (has_written_case) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\n"));
      }
      IREE_RETURN_IF_ERROR(
          loom_check_json_write_case(test_case, result, report, i, stream));
      has_written_case = true;
    }
  }
  if (has_written_case) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n"));
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "  ],\n"));

  // "summary": {...}
  iree_host_size_t total = pass_count + fail_count + skip_count;
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "  \"summary\": {\n"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, "    \"total\": %zu,\n", total));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "    \"passed\": %zu,\n", pass_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "    \"failed\": %zu,\n", fail_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "    \"skipped\": %zu\n", skip_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "  }\n"));

  // Close root object.
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}\n"));
  return iree_ok_status();
}
