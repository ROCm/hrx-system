// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/json_output.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/report.h"
#include "loom/util/stream.h"

namespace {

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

  EXPECT_NE(json.find("\"file\": \"test.loom-test\""), std::string::npos);
  EXPECT_NE(json.find("\"default_mode\": \"roundtrip\""), std::string::npos);
  EXPECT_NE(json.find("\"default_pipeline\": null"), std::string::npos);
  EXPECT_NE(json.find("\"default_format_target\": null"), std::string::npos);
  EXPECT_NE(json.find("\"default_emit_target\": null"), std::string::npos);
  EXPECT_NE(json.find("\"default_requirements\": []"), std::string::npos);
  EXPECT_NE(json.find("\"index\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"mode\": \"roundtrip\""), std::string::npos);
  EXPECT_NE(json.find("\"has_run_directive\": true"), std::string::npos);
  EXPECT_NE(json.find("\"has_requires_directive\": false"), std::string::npos);
  EXPECT_NE(json.find("\"requires_directive_range\": null"), std::string::npos);
  EXPECT_NE(json.find("\"requirements\": []"), std::string::npos);
  EXPECT_NE(json.find("\"pipeline\": null"), std::string::npos);
  EXPECT_NE(json.find("\"format_target\": null"), std::string::npos);
  EXPECT_NE(json.find("\"emit_target\": null"), std::string::npos);
  EXPECT_NE(json.find("\"source_range\": {\"start_byte\": 0"),
            std::string::npos);
  EXPECT_NE(json.find("\"separator_range\": null"), std::string::npos);
  EXPECT_NE(json.find("\"run_directive_range\": {\"start_byte\": 0, "
                      "\"end_byte\": 17}"),
            std::string::npos);
  EXPECT_NE(json.find("\"xfail\": false"), std::string::npos);
  EXPECT_NE(json.find("\"xfail_directive_range\": null"), std::string::npos);
  EXPECT_NE(json.find("\"xfail_reason\": null"), std::string::npos);
  EXPECT_NE(json.find("\"raw_outcome\": \"pass\""), std::string::npos);
  EXPECT_NE(json.find("\"final_outcome\": \"pass\""), std::string::npos);
  EXPECT_NE(json.find("\"detail\": \"\""), std::string::npos);
  EXPECT_NE(json.find("\"diff\": null"), std::string::npos);
  EXPECT_NE(json.find("\"update_edit\": null"), std::string::npos);
  EXPECT_NE(json.find("\"annotation_edits\": []"), std::string::npos);
  EXPECT_NE(json.find("\"input_range\": {\"start_byte\": 18"),
            std::string::npos);
  EXPECT_NE(json.find("\"expected_separator_range\": null"), std::string::npos);
  EXPECT_NE(json.find("\"expected_range\": null"), std::string::npos);
  EXPECT_NE(json.find("\"has_expected_section\": false"), std::string::npos);
  EXPECT_NE(json.find("\"total\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"passed\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"failed\": 0"), std::string::npos);
  EXPECT_NE(json.find("\"skipped\": 0"), std::string::npos);

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

  EXPECT_NE(json.find("\"has_requires_directive\": true"), std::string::npos);
  EXPECT_NE(json.find("\"requirements\": [\"loom-check-test-unavailable\"]"),
            std::string::npos);
  EXPECT_NE(json.find("\"raw_outcome\": \"skip\""), std::string::npos);
  EXPECT_NE(json.find("\"final_outcome\": \"skip\""), std::string::npos);
  EXPECT_NE(json.find("\"skipped\": 1"), std::string::npos);

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

  EXPECT_NE(json.find("\"final_outcome\": \"fail\""), std::string::npos);
  EXPECT_NE(json.find("\"total\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"failed\": 1"), std::string::npos);
  // Detail should contain the message (escaped).
  EXPECT_NE(json.find("mismatch on line 2"), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, EmbedsStructuredDiagnosticsArray) {
  auto file = Parse(
      "// RUN: verify\n"
      "bogus.nonexistent\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "unexpected diagnostic\n"),
  };
  IREE_ASSERT_OK(iree_string_builder_append_cstring(
      &results[0].diagnostic_json,
      "{\"severity\":\"error\",\"error_id\":\"ERR_PARSE_006\"}"));
  results[0].diagnostic_count = 1;

  std::string json = WriteJson(iree_make_cstring_view("diagnostic.loom-test"),
                               file, results, 0, 1, 0);

  EXPECT_NE(json.find("\"diagnostics\": [\n        {\"severity\":\"error\","
                      "\"error_id\":\"ERR_PARSE_006\"}\n      ]"),
            std::string::npos);

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
  IREE_ASSERT_OK(iree_string_builder_append_cstring(
      &results[0].diff_hunk_json,
      "{\"expected_start_line\": 1, \"expected_line_count\": 1, "
      "\"actual_start_line\": 1, \"actual_line_count\": 1, \"lines\": "
      "[{\"kind\": \"delete\", \"text\": \"old\"}, {\"kind\": \"insert\", "
      "\"text\": \"new\"}]}"));
  results[0].diff_hunk_count = 1;

  std::string json = WriteJson(iree_make_cstring_view("diff.loom-test"), file,
                               results, 0, 1, 0);

  EXPECT_NE(json.find("\"diff\": {"), std::string::npos);
  EXPECT_NE(json.find("\"expected_range\": {\"start_byte\""),
            std::string::npos);
  EXPECT_NE(json.find("\"hunks\": ["), std::string::npos);
  EXPECT_NE(json.find("\"expected_start_line\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"kind\": \"delete\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\": \"insert\""), std::string::npos);

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

  EXPECT_NE(json.find("\"update_edit\": {"), std::string::npos);
  EXPECT_NE(json.find("\"kind\": \"replace_expected_output\""),
            std::string::npos);
  EXPECT_NE(json.find("\"range\": {\"start_byte\""), std::string::npos);
  EXPECT_NE(json.find("\"text\": \"func.def @f()"), std::string::npos);

  loom_check_result_deinitialize(&results[0]);
}

TEST_F(JsonOutputTest, EmbedsAnnotationEdits) {
  auto file = Parse(
      "// RUN: verify\n"
      "bogus.nonexistent\n");

  loom_check_result_t results[1] = {
      MakeResult(LOOM_CHECK_FAIL, LOOM_CHECK_FAIL, "unexpected diagnostic\n"),
  };
  IREE_ASSERT_OK(iree_string_builder_append_cstring(
      &results[0].annotation_edits.json,
      "{\"kind\": \"insert_diagnostic_annotations\", \"range\": "
      "{\"start_byte\": 15, \"end_byte\": 15}, \"target_line\": 1, "
      "\"text\": \"// ERROR@+1: PARSE/006\\n\"}"));
  results[0].annotation_edits.count = 1;

  std::string json = WriteJson(iree_make_cstring_view("annotation.loom-test"),
                               file, results, 0, 1, 0);

  EXPECT_NE(json.find("\"annotation_edits\": ["), std::string::npos);
  EXPECT_NE(json.find("\"kind\": \"insert_diagnostic_annotations\""),
            std::string::npos);
  EXPECT_NE(json.find("\"text\": \"// ERROR@+1: PARSE/006\\n\""),
            std::string::npos);

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

  EXPECT_NE(json.find("\"annotations\": ["), std::string::npos);
  EXPECT_NE(json.find("\"index\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"source_range\": {\"start_byte\": 15"),
            std::string::npos);
  EXPECT_NE(json.find("\"target_line\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"severity\": \"error\""), std::string::npos);
  EXPECT_NE(json.find("\"domain\": \"PARSE\""), std::string::npos);
  EXPECT_NE(json.find("\"code\": 6"), std::string::npos);
  EXPECT_NE(json.find("\"message_substrings\": [\"unknown\", \"operation\"]"),
            std::string::npos);
  EXPECT_NE(json.find("\"matched\": true"), std::string::npos);
  EXPECT_NE(json.find("\"severity\": \"warning\""), std::string::npos);
  EXPECT_NE(json.find("\"domain\": null"), std::string::npos);
  EXPECT_NE(json.find("\"code\": null"), std::string::npos);
  EXPECT_NE(json.find("\"message_substrings\": [\"left unmatched\"]"),
            std::string::npos);
  EXPECT_NE(json.find("\"matched\": false"), std::string::npos);

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

  EXPECT_NE(json.find("\"xfail\": true"), std::string::npos);
  EXPECT_NE(json.find("\"xfail_reason\": \"known bug\""), std::string::npos);
  EXPECT_NE(json.find("\"raw_outcome\": \"fail\""), std::string::npos);
  EXPECT_NE(json.find("\"final_outcome\": \"pass\""), std::string::npos);

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

  EXPECT_NE(json.find("\"default_mode\": \"verify\""), std::string::npos);
  EXPECT_NE(json.find("\"mode\": \"verify\""), std::string::npos);

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

  EXPECT_NE(json.find("\"default_mode\": \"pass\""), std::string::npos);
  EXPECT_NE(json.find("\"default_pipeline\": \"dce,cse\""), std::string::npos);
  EXPECT_NE(json.find("\"mode\": \"pass\""), std::string::npos);
  EXPECT_NE(json.find("\"has_run_directive\": false"), std::string::npos);
  EXPECT_NE(json.find("\"pipeline\": \"dce,cse\""), std::string::npos);
  EXPECT_NE(json.find("\"run_directive_range\": null"), std::string::npos);

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

  EXPECT_NE(json.find("\"default_requirements\": [\"tool-dis\"]"),
            std::string::npos);
  EXPECT_NE(json.find("\"has_requires_directive\": false"), std::string::npos);
  EXPECT_NE(json.find("\"requirements\": [\"tool-dis\"]"), std::string::npos);
  EXPECT_NE(json.find("\"requires_directive_range\": null"), std::string::npos);

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

  EXPECT_NE(json.find("\"index\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"index\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"total\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"passed\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"failed\": 1"), std::string::npos);

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

  EXPECT_EQ(json.find("      \"index\": 1"), std::string::npos);
  EXPECT_NE(json.find("      \"index\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"final_outcome\": \"fail\""), std::string::npos);
  EXPECT_NE(json.find("\"total\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"passed\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"failed\": 1"), std::string::npos);

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

  EXPECT_NE(json.find("\"cases\": [\n  ]"), std::string::npos);
  EXPECT_EQ(json.find("\"final_outcome\""), std::string::npos);
  EXPECT_EQ(json.find("\"detail\""), std::string::npos);
  EXPECT_NE(json.find("\"total\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"passed\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"failed\": 1"), std::string::npos);

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

  EXPECT_NE(json.find("\"cases\": [\n  ]"), std::string::npos);
  EXPECT_NE(json.find("\"total\": 0"), std::string::npos);
}

}  // namespace
