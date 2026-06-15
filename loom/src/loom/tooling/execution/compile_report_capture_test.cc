// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/compile_report_capture.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(CompileReportCaptureTest, ConfiguresDetailedReportRequest) {
  loom_run_compile_report_capture_options_t options = {};
  loom_run_compile_report_capture_options_initialize(&options);
  options.sink_format = LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON;
  options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;

  loom_run_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_run_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));

  loom_run_candidate_compile_options_t compile_options = {};
  loom_run_candidate_compile_options_initialize(&compile_options);
  loom_run_compile_report_capture_configure_compile_options(&capture,
                                                            &compile_options);
  EXPECT_EQ(compile_options.report, &capture.report);
  EXPECT_TRUE(iree_all_bits_set(
      capture.report.requested_detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS));

  loom_run_compile_report_capture_deinitialize(&capture);
}

TEST(CompileReportCaptureTest, AppendsWithSeparator) {
  loom_run_compile_report_capture_options_t options = {};
  loom_run_compile_report_capture_options_initialize(&options);
  options.sink_format = LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT;
  options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;

  loom_run_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_run_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));
  capture.report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      iree_string_builder_append_string(&builder, IREE_SV("output")));
  IREE_ASSERT_OK(
      loom_run_compile_report_capture_append_text(&capture, &builder));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("output\nCOMPILE-REPORT"), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_run_compile_report_capture_deinitialize(&capture);
}

TEST(CompileReportCaptureTest, AppendsJsonObject) {
  loom_run_compile_report_capture_options_t options = {};
  loom_run_compile_report_capture_options_initialize(&options);
  options.sink_format = LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON;
  options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;

  loom_run_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_run_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));
  capture.report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(
      loom_run_compile_report_capture_append_json(&capture, &stream));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"artifact_kind\":\"vm-archive\""), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
  loom_run_compile_report_capture_deinitialize(&capture);
}

TEST(CompileReportCaptureTest, ParsesStructuredRequestsByDefault) {
  loom_run_compile_report_capture_options_t options = {};
  loom_run_compile_report_capture_options_initialize(&options);

  IREE_ASSERT_OK(loom_run_compile_report_capture_options_parse_request(
      IREE_SV("summary"), &options));
  EXPECT_EQ(options.sink_format, LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON);
  EXPECT_EQ(options.detail_mode,
            LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY);

  IREE_ASSERT_OK(loom_run_compile_report_capture_options_parse_request(
      IREE_SV("details"), &options));
  EXPECT_EQ(options.sink_format, LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_JSON);
  EXPECT_EQ(options.detail_mode,
            LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS);

  IREE_ASSERT_OK(loom_run_compile_report_capture_options_parse_request(
      IREE_SV("text-details"), &options));
  EXPECT_EQ(options.sink_format, LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT);
  EXPECT_EQ(options.detail_mode,
            LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS);

  IREE_ASSERT_OK(loom_run_compile_report_capture_options_parse_request(
      IREE_SV("none"), &options));
  EXPECT_FALSE(loom_run_compile_report_capture_options_is_enabled(&options));
}

TEST(CompileReportCaptureTest, AppendsConfiguredJsonOutput) {
  loom_run_compile_report_capture_options_t options = {};
  loom_run_compile_report_capture_options_initialize(&options);
  IREE_ASSERT_OK(loom_run_compile_report_capture_options_parse_request(
      IREE_SV("summary"), &options));

  loom_run_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_run_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));
  capture.report.artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      iree_string_builder_append_string(&builder, IREE_SV("output")));
  IREE_ASSERT_OK(
      loom_run_compile_report_capture_append_output(&capture, &builder));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(
      iree_string_view_find(output, IREE_SV("output\n{\"artifact_kind\""), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"artifact_kind\":\"vm-archive\""), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_run_compile_report_capture_deinitialize(&capture);
}

}  // namespace
}  // namespace loom
