// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/json_output.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/report.h"
#include "loom/util/stream.h"

namespace {

static iree_string_view_t ParseJsonDocument(const std::string& json) {
  iree_string_view_t cursor = iree_make_string_view(json.data(), json.size());
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_consume_value(&cursor, &value));
  IREE_EXPECT_OK(iree_json_consume_insignificant(&cursor));
  EXPECT_TRUE(iree_string_view_is_empty(cursor));
  return value;
}

static iree_string_view_t LookupObject(iree_string_view_t object,
                                       iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(object, key, &value));
  return value;
}

static iree_string_view_t LookupArrayElement(iree_string_view_t array,
                                             iree_host_size_t index) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_array_get(array, index, &value));
  return value;
}

static void ExpectArrayLength(iree_string_view_t array,
                              iree_host_size_t expected) {
  iree_host_size_t actual = 0;
  IREE_EXPECT_OK(iree_json_array_length(array, &actual));
  EXPECT_EQ(actual, expected);
}

static void ExpectObjectValueEquals(iree_string_view_t object,
                                    iree_string_view_t key,
                                    iree_string_view_t expected) {
  EXPECT_TRUE(iree_string_view_equal(LookupObject(object, key), expected));
}

static void ExpectObjectUint64Equals(iree_string_view_t object,
                                     iree_string_view_t key,
                                     uint64_t expected) {
  uint64_t actual = 0;
  IREE_EXPECT_OK(iree_json_parse_uint64(LookupObject(object, key), &actual));
  EXPECT_EQ(actual, expected);
}

static void ExpectObjectBoolEquals(iree_string_view_t object,
                                   iree_string_view_t key, bool expected) {
  bool actual = false;
  IREE_EXPECT_OK(iree_json_parse_bool(LookupObject(object, key), &actual));
  EXPECT_EQ(actual, expected);
}

class JsonOutputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    iree_string_builder_initialize(iree_allocator_system(), &output_);
  }

  void TearDown() override {
    iree_string_builder_deinitialize(&output_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Parses a check file from source. Arena-allocated.
  loom_check_file_t Parse(const char* source) {
    iree_string_view_t source_view = iree_make_cstring_view(source);
    loom_check_file_t file = {0};
    IREE_EXPECT_OK(loom_check_parse(source_view, &arena_, &file));
    return file;
  }

  // Creates a result with the given outcomes and optional detail text.
  loom_check_result_t MakeResult(loom_check_outcome_t raw,
                                 loom_check_outcome_t final_outcome,
                                 const char* detail = "") {
    loom_check_result_t result;
    loom_check_result_initialize(iree_allocator_system(), &result);
    result.raw_outcome = raw;
    result.final_outcome = final_outcome;
    if (detail[0] != '\0') {
      IREE_EXPECT_OK(
          iree_string_builder_append_cstring(&result.detail, detail));
    }
    return result;
  }

  loom_check_file_report_t MakeReport(const loom_check_file_t& file) {
    loom_check_file_report_t report = {0};
    IREE_EXPECT_OK(loom_check_file_report_initialize(&file, &arena_, &report));
    return report;
  }

  // Writes JSON for the given file and results, returns the output string.
  std::string WriteJson(
      iree_string_view_t filename, const loom_check_file_t& file,
      const loom_check_file_report_t& report,
      const loom_check_result_t* results, iree_host_size_t pass_count,
      iree_host_size_t fail_count, iree_host_size_t skip_count,
      loom_check_json_output_mode_t output_mode = LOOM_CHECK_JSON_OUTPUT_ALL) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&output_, &stream);
    IREE_EXPECT_OK(loom_check_json_write_file_result(
        filename, &file, &report, results, pass_count, fail_count, skip_count,
        output_mode, &stream));
    return std::string(iree_string_builder_buffer(&output_),
                       iree_string_builder_size(&output_));
  }

  std::string WriteJson(
      iree_string_view_t filename, const loom_check_file_t& file,
      const loom_check_result_t* results, iree_host_size_t pass_count,
      iree_host_size_t fail_count, iree_host_size_t skip_count,
      loom_check_json_output_mode_t output_mode = LOOM_CHECK_JSON_OUTPUT_ALL) {
    loom_check_file_report_t report = MakeReport(file);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&output_, &stream);
    IREE_EXPECT_OK(loom_check_json_write_file_result(
        filename, &file, &report, results, pass_count, fail_count, skip_count,
        output_mode, &stream));
    return std::string(iree_string_builder_buffer(&output_),
                       iree_string_builder_size(&output_));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  iree_string_builder_t output_;
};

TEST_F(JsonOutputTest, SinglePassingCase) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
  };

  std::string json = WriteJson(iree_make_cstring_view("test.loom-test"), file,
                               results, 1, 0, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  ExpectObjectValueEquals(root, IREE_SV("file"), IREE_SV("test.loom-test"));
  ExpectObjectValueEquals(root, IREE_SV("default_mode"), IREE_SV("roundtrip"));
  ExpectObjectValueEquals(root, IREE_SV("default_pipeline"), IREE_SV("null"));
  ExpectObjectValueEquals(root, IREE_SV("default_format_target"),
                          IREE_SV("null"));
  ExpectObjectValueEquals(root, IREE_SV("default_emit_target"),
                          IREE_SV("null"));
  ExpectArrayLength(LookupObject(root, IREE_SV("default_requirements")), 0);

  const iree_string_view_t cases = LookupObject(root, IREE_SV("cases"));
  ExpectArrayLength(cases, 1);
  const iree_string_view_t test_case = LookupArrayElement(cases, 0);
  ExpectObjectUint64Equals(test_case, IREE_SV("index"), 1);
  ExpectObjectValueEquals(test_case, IREE_SV("mode"), IREE_SV("roundtrip"));
  ExpectObjectBoolEquals(test_case, IREE_SV("has_run_directive"), true);
  ExpectObjectBoolEquals(test_case, IREE_SV("has_requires_directive"), false);
  ExpectObjectValueEquals(test_case, IREE_SV("requires_directive_range"),
                          IREE_SV("null"));
  ExpectArrayLength(LookupObject(test_case, IREE_SV("requirements")), 0);
  ExpectObjectValueEquals(test_case, IREE_SV("pipeline"), IREE_SV("null"));
  ExpectObjectValueEquals(test_case, IREE_SV("format_target"), IREE_SV("null"));
  ExpectObjectValueEquals(test_case, IREE_SV("emit_target"), IREE_SV("null"));
  ExpectObjectUint64Equals(LookupObject(test_case, IREE_SV("source_range")),
                           IREE_SV("start_byte"), 0);
  ExpectObjectValueEquals(test_case, IREE_SV("separator_range"),
                          IREE_SV("null"));
  const iree_string_view_t run_range =
      LookupObject(test_case, IREE_SV("run_directive_range"));
  ExpectObjectUint64Equals(run_range, IREE_SV("start_byte"), 0);
  ExpectObjectUint64Equals(run_range, IREE_SV("end_byte"), 17);
  ExpectObjectBoolEquals(test_case, IREE_SV("xfail"), false);
  ExpectObjectValueEquals(test_case, IREE_SV("xfail_directive_range"),
                          IREE_SV("null"));
  ExpectObjectValueEquals(test_case, IREE_SV("xfail_reason"), IREE_SV("null"));
  ExpectObjectValueEquals(test_case, IREE_SV("raw_outcome"), IREE_SV("pass"));
  ExpectObjectValueEquals(test_case, IREE_SV("final_outcome"), IREE_SV("pass"));
  ExpectObjectValueEquals(test_case, IREE_SV("detail"),
                          iree_string_view_empty());
  ExpectObjectValueEquals(test_case, IREE_SV("diff"), IREE_SV("null"));
  ExpectObjectValueEquals(test_case, IREE_SV("update_edit"), IREE_SV("null"));
  ExpectArrayLength(LookupObject(test_case, IREE_SV("annotation_edits")), 0);
  ExpectObjectUint64Equals(LookupObject(test_case, IREE_SV("input_range")),
                           IREE_SV("start_byte"), 18);
  ExpectObjectValueEquals(test_case, IREE_SV("expected_separator_range"),
                          IREE_SV("null"));
  ExpectObjectValueEquals(test_case, IREE_SV("expected_range"),
                          IREE_SV("null"));
  ExpectObjectBoolEquals(test_case, IREE_SV("has_expected_section"), false);
  ExpectArrayLength(LookupObject(test_case, IREE_SV("annotations")), 0);
  ExpectArrayLength(LookupObject(test_case, IREE_SV("diagnostics")), 0);

  const iree_string_view_t summary = LookupObject(root, IREE_SV("summary"));
  ExpectObjectUint64Equals(summary, IREE_SV("total"), 1);
  ExpectObjectUint64Equals(summary, IREE_SV("passed"), 1);
  ExpectObjectUint64Equals(summary, IREE_SV("failed"), 0);
  ExpectObjectUint64Equals(summary, IREE_SV("skipped"), 0);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, SkippedCase) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "// REQUIRES: loom-check-test-unavailable\n"
      "func.def @f() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_SKIP, LOOM_CHECK_SKIP, "skipped\n"),
  };

  std::string json = WriteJson(iree_make_cstring_view("skip.loom-test"), file,
                               results, 0, 0, 1);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t test_case =
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 0);
  ExpectObjectBoolEquals(test_case, IREE_SV("has_requires_directive"), true);
  const iree_string_view_t requirements =
      LookupObject(test_case, IREE_SV("requirements"));
  ExpectArrayLength(requirements, 1);
  EXPECT_TRUE(iree_string_view_equal(LookupArrayElement(requirements, 0),
                                     IREE_SV("loom-check-test-unavailable")));
  ExpectObjectValueEquals(test_case, IREE_SV("raw_outcome"), IREE_SV("skip"));
  ExpectObjectValueEquals(test_case, IREE_SV("final_outcome"), IREE_SV("skip"));
  ExpectObjectUint64Equals(LookupObject(root, IREE_SV("summary")),
                           IREE_SV("skipped"), 1);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, SingleFailingCase) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "mismatch on line 2\n"),
  };

  std::string json = WriteJson(iree_make_cstring_view("fail.loom-test"), file,
                               results, 0, 1, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t test_case =
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 0);
  ExpectObjectValueEquals(test_case, IREE_SV("final_outcome"), IREE_SV("fail"));
  ExpectObjectValueEquals(test_case, IREE_SV("detail"),
                          IREE_SV("mismatch on line 2\\n"));
  const iree_string_view_t summary = LookupObject(root, IREE_SV("summary"));
  ExpectObjectUint64Equals(summary, IREE_SV("total"), 1);
  ExpectObjectUint64Equals(summary, IREE_SV("failed"), 1);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, EmbedsStructuredDiagnosticsArray) {
  auto file = Parse(
      "// RUN: verify\n"
      "bogus.nonexistent\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "unexpected diagnostic\n"),
  };
  loom_output_stream_t diagnostic_stream;
  IREE_ASSERT_OK(loom_json_value_list_begin_value(&results[0].diagnostics,
                                                  &diagnostic_stream));
  IREE_ASSERT_OK(loom_output_stream_write_cstring(
      &diagnostic_stream,
      "{\"severity\":\"error\",\"error_id\":\"ERR_PARSE_006\"}"));

  std::string json = WriteJson(iree_make_cstring_view("diagnostic.loom-test"),
                               file, results, 0, 1, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t test_case =
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 0);
  const iree_string_view_t diagnostics =
      LookupObject(test_case, IREE_SV("diagnostics"));
  ExpectArrayLength(diagnostics, 1);
  const iree_string_view_t diagnostic = LookupArrayElement(diagnostics, 0);
  ExpectObjectValueEquals(diagnostic, IREE_SV("severity"), IREE_SV("error"));
  ExpectObjectValueEquals(diagnostic, IREE_SV("error_id"),
                          IREE_SV("ERR_PARSE_006"));

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, EmbedsStructuredDiffHunks) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n"
      "// ----\n"
      "func.def @different() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "diff output\n"),
  };
  IREE_ASSERT_OK(
      loom_check_result_record_diff(IREE_SV("old\n"), IREE_SV("new\n"),
                                    iree_allocator_system(), &results[0]));

  std::string json = WriteJson(iree_make_cstring_view("diff.loom-test"), file,
                               results, 0, 1, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t test_case =
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 0);
  const iree_string_view_t diff = LookupObject(test_case, IREE_SV("diff"));
  const iree_string_view_t expected_range =
      LookupObject(diff, IREE_SV("expected_range"));
  ExpectObjectUint64Equals(expected_range, IREE_SV("start_byte"),
                           file.cases[0].expected_range.start_byte);
  const iree_string_view_t hunks = LookupObject(diff, IREE_SV("hunks"));
  ExpectArrayLength(hunks, 1);
  const iree_string_view_t hunk = LookupArrayElement(hunks, 0);
  ExpectObjectUint64Equals(hunk, IREE_SV("expected_start_line"), 1);
  const iree_string_view_t lines = LookupObject(hunk, IREE_SV("lines"));
  ExpectArrayLength(lines, 2);
  ExpectObjectValueEquals(LookupArrayElement(lines, 0), IREE_SV("kind"),
                          IREE_SV("delete"));
  ExpectObjectValueEquals(LookupArrayElement(lines, 1), IREE_SV("kind"),
                          IREE_SV("insert"));

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, EmbedsUpdateEdit) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n"
      "// ----\n"
      "func.def @different() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "diff output\n"),
  };
  results[0].update_edit.present = true;
  results[0].update_edit.value = (loom_check_update_edit_t){
      /*.kind=*/LOOM_CHECK_UPDATE_EDIT_REPLACE_EXPECTED_OUTPUT,
      /*.range=*/file.cases[0].expected_range,
  };
  IREE_ASSERT_OK(iree_string_builder_append_cstring(
      &results[0].update_edit.text, "func.def @f() {\n}\n"));

  std::string json = WriteJson(iree_make_cstring_view("edit.loom-test"), file,
                               results, 0, 1, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t test_case =
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 0);
  const iree_string_view_t update_edit =
      LookupObject(test_case, IREE_SV("update_edit"));
  ExpectObjectValueEquals(update_edit, IREE_SV("kind"),
                          IREE_SV("replace_expected_output"));
  ExpectObjectUint64Equals(LookupObject(update_edit, IREE_SV("range")),
                           IREE_SV("start_byte"),
                           file.cases[0].expected_range.start_byte);
  ExpectObjectValueEquals(update_edit, IREE_SV("text"),
                          IREE_SV("func.def @f() {\\n}\\n"));

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, EmbedsAnnotationEdits) {
  auto file = Parse(
      "// RUN: verify\n"
      "bogus.nonexistent\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "unexpected diagnostic\n"),
  };
  IREE_ASSERT_OK(loom_check_result_append_annotation_edit(
      &results[0], LOOM_CHECK_UPDATE_EDIT_INSERT_DIAGNOSTIC_ANNOTATIONS,
      (loom_check_source_range_t){/*start_byte=*/15, /*end_byte=*/15},
      /*target_line=*/1, IREE_SV("// ERROR@+1: PARSE/006\n")));

  std::string json = WriteJson(iree_make_cstring_view("annotation.loom-test"),
                               file, results, 0, 1, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t test_case =
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 0);
  const iree_string_view_t edits =
      LookupObject(test_case, IREE_SV("annotation_edits"));
  ExpectArrayLength(edits, 1);
  const iree_string_view_t edit = LookupArrayElement(edits, 0);
  ExpectObjectValueEquals(edit, IREE_SV("kind"),
                          IREE_SV("insert_diagnostic_annotations"));
  ExpectObjectValueEquals(edit, IREE_SV("text"),
                          IREE_SV("// ERROR@+1: PARSE/006\\n"));

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, EmitsVerifyAnnotationsWithMatchedState) {
  auto file = Parse(
      "// RUN: verify\n"
      "// ERROR@+1: PARSE/006 \"unknown\" \"operation\"\n"
      "bogus.nonexistent\n"
      "// WARNING: \"left unmatched\"\n");

  ASSERT_EQ(file.case_count, 1u);
  ASSERT_EQ(file.cases[0].annotation_count, 2u);

  loom_check_file_report_t report = MakeReport(file);
  IREE_ASSERT_OK(loom_check_file_report_mark_annotation_matched(&report, 0, 0));

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "unmatched annotation\n"),
  };

  std::string json = WriteJson(iree_make_cstring_view("annotations.loom-test"),
                               file, report, results, 0, 1, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t test_case =
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 0);
  const iree_string_view_t annotations =
      LookupObject(test_case, IREE_SV("annotations"));
  ExpectArrayLength(annotations, 2);

  const iree_string_view_t first = LookupArrayElement(annotations, 0);
  ExpectObjectUint64Equals(first, IREE_SV("index"), 1);
  ExpectObjectUint64Equals(LookupObject(first, IREE_SV("source_range")),
                           IREE_SV("start_byte"), 15);
  ExpectObjectUint64Equals(first, IREE_SV("target_line"), 2);
  const iree_string_view_t first_expected =
      LookupObject(first, IREE_SV("expected"));
  ExpectObjectValueEquals(first_expected, IREE_SV("severity"),
                          IREE_SV("error"));
  ExpectObjectValueEquals(first_expected, IREE_SV("domain"), IREE_SV("PARSE"));
  ExpectObjectUint64Equals(first_expected, IREE_SV("code"), 6);
  const iree_string_view_t first_messages =
      LookupObject(first_expected, IREE_SV("message_substrings"));
  ExpectArrayLength(first_messages, 2);
  EXPECT_TRUE(iree_string_view_equal(LookupArrayElement(first_messages, 0),
                                     IREE_SV("unknown")));
  EXPECT_TRUE(iree_string_view_equal(LookupArrayElement(first_messages, 1),
                                     IREE_SV("operation")));
  ExpectObjectBoolEquals(first, IREE_SV("matched"), true);

  const iree_string_view_t second = LookupArrayElement(annotations, 1);
  const iree_string_view_t second_expected =
      LookupObject(second, IREE_SV("expected"));
  ExpectObjectValueEquals(second_expected, IREE_SV("severity"),
                          IREE_SV("warning"));
  ExpectObjectValueEquals(second_expected, IREE_SV("domain"), IREE_SV("null"));
  ExpectObjectValueEquals(second_expected, IREE_SV("code"), IREE_SV("null"));
  const iree_string_view_t second_messages =
      LookupObject(second_expected, IREE_SV("message_substrings"));
  ExpectArrayLength(second_messages, 1);
  EXPECT_TRUE(iree_string_view_equal(LookupArrayElement(second_messages, 0),
                                     IREE_SV("left unmatched")));
  ExpectObjectBoolEquals(second, IREE_SV("matched"), false);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, XfailCase) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "// XFAIL: known bug\n"
      "func.def @f() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 1u);
  ASSERT_TRUE(file.cases[0].xfail);

  // Raw outcome is fail, but final is pass due to xfail inversion.
  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_PASS),
  };

  std::string json = WriteJson(iree_make_cstring_view("xfail.loom-test"), file,
                               results, 1, 0, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t test_case =
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 0);
  ExpectObjectBoolEquals(test_case, IREE_SV("xfail"), true);
  ExpectObjectValueEquals(test_case, IREE_SV("xfail_reason"),
                          IREE_SV("known bug"));
  ExpectObjectValueEquals(test_case, IREE_SV("raw_outcome"), IREE_SV("fail"));
  ExpectObjectValueEquals(test_case, IREE_SV("final_outcome"), IREE_SV("pass"));

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, VerifyMode) {
  auto file = Parse(
      "// RUN: verify\n"
      "func.def @f() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
  };

  std::string json = WriteJson(iree_make_cstring_view("verify.loom-test"), file,
                               results, 1, 0, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  ExpectObjectValueEquals(root, IREE_SV("default_mode"), IREE_SV("verify"));
  ExpectObjectValueEquals(
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 0),
      IREE_SV("mode"), IREE_SV("verify"));

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, EmitsInheritedRunMetadata) {
  auto file = Parse(
      "// RUN: pass dce,cse\n"
      "func.def @default_owner() {\n"
      "}\n"
      "// ====\n"
      "func.def @f() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 2u);
  ASSERT_TRUE(file.cases[0].has_run_directive);
  ASSERT_FALSE(file.cases[1].has_run_directive);

  loom_check_result_t results[2] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
  };

  std::string json = WriteJson(iree_make_cstring_view("inherited.loom-test"),
                               file, results, 2, 0, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  ExpectObjectValueEquals(root, IREE_SV("default_mode"), IREE_SV("pass"));
  ExpectObjectValueEquals(root, IREE_SV("default_pipeline"),
                          IREE_SV("dce,cse"));
  const iree_string_view_t inherited_case =
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 1);
  ExpectObjectValueEquals(inherited_case, IREE_SV("mode"), IREE_SV("pass"));
  ExpectObjectBoolEquals(inherited_case, IREE_SV("has_run_directive"), false);
  ExpectObjectValueEquals(inherited_case, IREE_SV("pipeline"),
                          IREE_SV("dce,cse"));
  ExpectObjectValueEquals(inherited_case, IREE_SV("run_directive_range"),
                          IREE_SV("null"));

  loom_check_result_deinitialize(&results[0]);
  loom_check_result_deinitialize(&results[1]);
}

TEST_F(JsonOutputTest, EmitsInheritedRequiresMetadata) {
  auto file = Parse(
      "// REQUIRES: tool-dis\n"
      "func.def @default_owner() {\n"
      "}\n"
      "// ====\n"
      "func.def @f() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 2u);
  ASSERT_TRUE(file.cases[0].has_requires_directive);
  ASSERT_FALSE(file.cases[1].has_requires_directive);

  loom_check_result_t results[2] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
  };

  std::string json = WriteJson(iree_make_cstring_view("requires.loom-test"),
                               file, results, 2, 0, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t default_requirements =
      LookupObject(root, IREE_SV("default_requirements"));
  ExpectArrayLength(default_requirements, 1);
  EXPECT_TRUE(iree_string_view_equal(
      LookupArrayElement(default_requirements, 0), IREE_SV("tool-dis")));
  const iree_string_view_t inherited_case =
      LookupArrayElement(LookupObject(root, IREE_SV("cases")), 1);
  ExpectObjectBoolEquals(inherited_case, IREE_SV("has_requires_directive"),
                         false);
  const iree_string_view_t requirements =
      LookupObject(inherited_case, IREE_SV("requirements"));
  ExpectArrayLength(requirements, 1);
  EXPECT_TRUE(iree_string_view_equal(LookupArrayElement(requirements, 0),
                                     IREE_SV("tool-dis")));
  ExpectObjectValueEquals(inherited_case, IREE_SV("requires_directive_range"),
                          IREE_SV("null"));

  loom_check_result_deinitialize(&results[0]);
  loom_check_result_deinitialize(&results[1]);
}

TEST_F(JsonOutputTest, MultipleCases) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @a() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @b() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 2u);

  loom_check_result_t results[2] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "diff output\n"),
  };

  std::string json = WriteJson(iree_make_cstring_view("multi.loom-test"), file,
                               results, 1, 1, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t cases = LookupObject(root, IREE_SV("cases"));
  ExpectArrayLength(cases, 2);
  ExpectObjectUint64Equals(LookupArrayElement(cases, 0), IREE_SV("index"), 1);
  ExpectObjectUint64Equals(LookupArrayElement(cases, 1), IREE_SV("index"), 2);
  const iree_string_view_t summary = LookupObject(root, IREE_SV("summary"));
  ExpectObjectUint64Equals(summary, IREE_SV("total"), 2);
  ExpectObjectUint64Equals(summary, IREE_SV("passed"), 1);
  ExpectObjectUint64Equals(summary, IREE_SV("failed"), 1);

  loom_check_result_deinitialize(&results[0]);
  loom_check_result_deinitialize(&results[1]);
}

TEST_F(JsonOutputTest, FailuresModeOnlyEmitsFailingCases) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @a() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @b() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 2u);

  loom_check_result_t results[2] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "diff output\n"),
  };

  std::string json =
      WriteJson(iree_make_cstring_view("filtered.loom-test"), file, results, 1,
                1, 0, LOOM_CHECK_JSON_OUTPUT_FAILURES);

  const iree_string_view_t root = ParseJsonDocument(json);
  const iree_string_view_t cases = LookupObject(root, IREE_SV("cases"));
  ExpectArrayLength(cases, 1);
  const iree_string_view_t failed_case = LookupArrayElement(cases, 0);
  ExpectObjectUint64Equals(failed_case, IREE_SV("index"), 2);
  ExpectObjectValueEquals(failed_case, IREE_SV("final_outcome"),
                          IREE_SV("fail"));
  const iree_string_view_t summary = LookupObject(root, IREE_SV("summary"));
  ExpectObjectUint64Equals(summary, IREE_SV("total"), 2);
  ExpectObjectUint64Equals(summary, IREE_SV("passed"), 1);
  ExpectObjectUint64Equals(summary, IREE_SV("failed"), 1);

  loom_check_result_deinitialize(&results[0]);
  loom_check_result_deinitialize(&results[1]);
}

TEST_F(JsonOutputTest, SummaryModeOmitsCases) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @a() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @b() {\n"
      "}\n");

  ASSERT_EQ(file.case_count, 2u);

  loom_check_result_t results[2] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "diff output\n"),
  };

  std::string json =
      WriteJson(iree_make_cstring_view("summary.loom-test"), file, results, 1,
                1, 0, LOOM_CHECK_JSON_OUTPUT_SUMMARY);

  const iree_string_view_t root = ParseJsonDocument(json);
  ExpectArrayLength(LookupObject(root, IREE_SV("cases")), 0);
  const iree_string_view_t summary = LookupObject(root, IREE_SV("summary"));
  ExpectObjectUint64Equals(summary, IREE_SV("total"), 2);
  ExpectObjectUint64Equals(summary, IREE_SV("passed"), 1);
  ExpectObjectUint64Equals(summary, IREE_SV("failed"), 1);

  loom_check_result_deinitialize(&results[0]);
  loom_check_result_deinitialize(&results[1]);
}

TEST_F(JsonOutputTest, EscapedStrings) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  // Detail contains characters that need JSON escaping.
  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL,
                 "line 1: \"expected\" vs \"actual\"\nline 2: tab\there\n"),
  };

  std::string json = WriteJson(iree_make_cstring_view("escape.loom-test"), file,
                               results, 0, 1, 0);

  // Quotes and newlines should be escaped in the JSON.
  EXPECT_NE(json.find("\\\"expected\\\""), std::string::npos);
  EXPECT_NE(json.find("\\n"), std::string::npos);
  EXPECT_NE(json.find("\\t"), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, FilenameWithSpecialCharacters) {
  auto file = Parse(
      "// RUN: roundtrip\n"
      "func.def @f() {\n"
      "}\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_PASS, LOOM_CHECK_PASS),
  };

  // Path with backslash (Windows-style) should be escaped.
  std::string json =
      WriteJson(iree_make_cstring_view("C:\\tests\\test.loom-test"), file,
                results, 1, 0, 0);

  EXPECT_NE(json.find("C:\\\\tests\\\\test.loom-test"), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, EmptyCaseArray) {
  // A file with only a preamble and no cases produces zero-count output.
  loom_check_file_t file = {0};
  file.default_mode = LOOM_CHECK_MODE_ROUNDTRIP;

  std::string json = WriteJson(iree_make_cstring_view("empty.loom-test"), file,
                               /*results=*/nullptr, 0, 0, 0);

  const iree_string_view_t root = ParseJsonDocument(json);
  ExpectArrayLength(LookupObject(root, IREE_SV("cases")), 0);
  ExpectObjectUint64Equals(LookupObject(root, IREE_SV("summary")),
                           IREE_SV("total"), 0);
}

}  // namespace
