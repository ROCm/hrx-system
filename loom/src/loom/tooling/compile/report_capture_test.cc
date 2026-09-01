// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/report_capture.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/error/json_sink.h"

namespace loom {
namespace {

static std::string EmitDiagnosticJsonObject(
    const loom_diagnostic_t* diagnostic,
    loom_type_formatter_t type_formatter = {loom_type_format_minimal,
                                            nullptr}) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_EXPECT_OK(
      loom_diagnostic_json_write_object(&stream, diagnostic, type_formatter));
  std::string result(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return result;
}

TEST(CompileReportCaptureTest, ConfiguresDetailedReportRequest) {
  loom_compile_report_capture_options_t options = {};
  loom_compile_report_capture_options_initialize(&options);
  options.sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_JSON;
  options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;

  loom_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));

  loom_compile_options_t compile_options = {};
  loom_compile_options_initialize(&compile_options);
  loom_compile_report_capture_configure_compile_options(&capture,
                                                        &compile_options);
  EXPECT_EQ(compile_options.report, &capture.report);
  EXPECT_TRUE(iree_all_bits_set(compile_options.target_pipeline_options
                                    .source_to_low_legality_diagnostic_flags,
                                LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL));
  EXPECT_TRUE(iree_all_bits_set(
      capture.report.requested_detail_flags,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_MATH_LEGALIZATION_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS |
          LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS));

  loom_compile_report_capture_deinitialize(&capture);
}

TEST(CompileReportCaptureTest,
     SummaryReportRequestsEconomicsWithoutSourceLowDiagnostics) {
  loom_compile_report_capture_options_t options = {};
  loom_compile_report_capture_options_initialize(&options);
  options.sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_JSON;
  options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;

  loom_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));

  loom_compile_options_t compile_options = {};
  loom_compile_options_initialize(&compile_options);
  loom_compile_report_capture_configure_compile_options(&capture,
                                                        &compile_options);
  EXPECT_EQ(compile_options.target_pipeline_options
                .source_to_low_legality_diagnostic_flags,
            0u);
  EXPECT_EQ(capture.report.requested_detail_flags,
            LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS |
                LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_SUMMARY_ROWS |
                LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS);

  loom_compile_report_capture_deinitialize(&capture);
}

TEST(CompileReportCaptureTest, AppendsWithSeparator) {
  loom_compile_report_capture_options_t options = {};
  loom_compile_report_capture_options_initialize(&options);
  options.sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_TEXT;
  options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;

  loom_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));
  capture.report.artifact_kind =
      LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      iree_string_builder_append_string(&builder, IREE_SV("output")));
  IREE_ASSERT_OK(loom_compile_report_capture_append_text(&capture, &builder));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(output, IREE_SV("output\nCOMPILE-REPORT"), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_compile_report_capture_deinitialize(&capture);
}

TEST(CompileReportCaptureTest, EmbedsCanonicalDiagnosticJson) {
  loom_compile_report_capture_options_t options = {};
  loom_compile_report_capture_options_initialize(&options);
  options.sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_JSON;
  options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;

  loom_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));

  const iree_string_view_t source =
      IREE_SV("  %chunk = vector.load %view[%origin] : vector<4xi32>\n");
  const loom_source_range_t range = {
      /*.provenance=*/LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
      /*.filename=*/IREE_SV("kernel.loom"),
      /*.source=*/source,
      /*.start=*/2,
      /*.end=*/8,
      /*.start_line=*/7,
      /*.start_column=*/3,
      /*.end_line=*/7,
      /*.end_column=*/9,
  };
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("vector.load")),
      loom_param_i64(0),
      loom_param_i64(0),
      loom_param_string(IREE_SV("%view")),
      loom_param_string(IREE_SV("%origin")),
      loom_param_string(IREE_SV("4")),
      loom_param_string(IREE_SV("8388608")),
      loom_param_string(IREE_SV("8388604")),
      loom_param_string(IREE_SV("vector_footprint.full_vector_upper_bound")),
  };
  const loom_diagnostic_t diagnostic = {
      /*.severity=*/LOOM_DIAGNOSTIC_ERROR,
      /*.error=*/loom_error_def_lookup(LOOM_ERROR_DOMAIN_SUBRANGE, 10),
      /*.params=*/params,
      /*.param_count=*/IREE_ARRAYSIZE(params),
      /*.emitter=*/LOOM_EMITTER_PASS,
      /*.origin=*/range,
      /*.source_location=*/range,
  };

  const loom_type_formatter_t type_formatter = {loom_type_format_minimal,
                                                nullptr};
  const std::string canonical_diagnostic_json =
      EmitDiagnosticJsonObject(&diagnostic, type_formatter);
  IREE_ASSERT_OK(loom_compile_report_capture_record_diagnostic(
      &capture, &diagnostic, type_formatter));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_compile_report_capture_append_json(&capture, &stream));

  const std::string report_json(iree_string_builder_buffer(&builder),
                                iree_string_builder_size(&builder));
  EXPECT_NE(report_json.find("\"diagnostic_count\":1"), std::string::npos);
  EXPECT_NE(
      report_json.find("\"diagnostics\":[" + canonical_diagnostic_json + "]"),
      std::string::npos);
  EXPECT_NE(report_json.find("\"constraint_key\":"
                             "\"vector_footprint.full_vector_upper_bound\""),
            std::string::npos);

  iree_string_builder_deinitialize(&builder);
  loom_compile_report_capture_deinitialize(&capture);
}

TEST(CompileReportCaptureTest, TextReportsDoNotSerializeDiagnosticJson) {
  loom_compile_report_capture_options_t options = {};
  loom_compile_report_capture_options_initialize(&options);
  options.sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_TEXT;
  options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;

  loom_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));

  loom_diagnostic_param_t params[] = {loom_param_string(IREE_SV("x"))};
  const loom_diagnostic_t diagnostic = {
      /*.severity=*/LOOM_DIAGNOSTIC_ERROR,
      /*.error=*/loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1),
      /*.params=*/params,
      /*.param_count=*/IREE_ARRAYSIZE(params),
      /*.emitter=*/LOOM_EMITTER_PARSER,
  };
  IREE_ASSERT_OK(loom_compile_report_capture_record_diagnostic(
      &capture, &diagnostic,
      (loom_type_formatter_t){loom_type_format_minimal, nullptr}));

  EXPECT_EQ(capture.diagnostics.count, 1u);
  EXPECT_TRUE(iree_string_view_is_empty(
      loom_json_value_list_body(&capture.diagnostics.json_values)));

  loom_compile_report_capture_deinitialize(&capture);
}

TEST(CompileReportCaptureTest, AppendsJsonObject) {
  loom_compile_report_capture_options_t options = {};
  loom_compile_report_capture_options_initialize(&options);
  options.sink_format = LOOM_COMPILE_REPORT_SINK_FORMAT_JSON;
  options.detail_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;

  loom_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));
  capture.report.artifact_kind =
      LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_compile_report_capture_append_json(&capture, &stream));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"artifact_kind\":\"target-artifact\""), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
  loom_compile_report_capture_deinitialize(&capture);
}

TEST(CompileReportCaptureTest, ParsesStructuredRequestsByDefault) {
  loom_compile_report_capture_options_t options = {};
  loom_compile_report_capture_options_initialize(&options);

  IREE_ASSERT_OK(loom_compile_report_capture_options_parse_request(
      IREE_SV("summary"), &options));
  EXPECT_EQ(options.sink_format, LOOM_COMPILE_REPORT_SINK_FORMAT_JSON);
  EXPECT_EQ(options.detail_mode,
            LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY);

  IREE_ASSERT_OK(loom_compile_report_capture_options_parse_request(
      IREE_SV("details"), &options));
  EXPECT_EQ(options.sink_format, LOOM_COMPILE_REPORT_SINK_FORMAT_JSON);
  EXPECT_EQ(options.detail_mode,
            LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS);

  IREE_ASSERT_OK(loom_compile_report_capture_options_parse_request(
      IREE_SV("text-details"), &options));
  EXPECT_EQ(options.sink_format, LOOM_COMPILE_REPORT_SINK_FORMAT_TEXT);
  EXPECT_EQ(options.detail_mode,
            LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS);

  IREE_ASSERT_OK(loom_compile_report_capture_options_parse_request(
      IREE_SV("none"), &options));
  EXPECT_FALSE(loom_compile_report_capture_options_is_enabled(&options));
}

TEST(CompileReportCaptureTest, AppendsConfiguredJsonOutput) {
  loom_compile_report_capture_options_t options = {};
  loom_compile_report_capture_options_initialize(&options);
  IREE_ASSERT_OK(loom_compile_report_capture_options_parse_request(
      IREE_SV("summary"), &options));

  loom_compile_report_capture_t capture = {};
  IREE_ASSERT_OK(loom_compile_report_capture_initialize(
      &options, iree_allocator_system(), &capture));
  capture.report.artifact_kind =
      LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      iree_string_builder_append_string(&builder, IREE_SV("output")));
  IREE_ASSERT_OK(loom_compile_report_capture_append_output(&capture, &builder));

  iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_NE(
      iree_string_view_find(
          output, IREE_SV("output\n{\"kind\":\"loom.compile_report\""), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(
      iree_string_view_find(
          output, IREE_SV("\"schema_version\":0,\"mode\":\"summary\""), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                output, IREE_SV("\"artifact_kind\":\"target-artifact\""), 0),
            IREE_STRING_VIEW_NPOS);

  iree_string_builder_deinitialize(&builder);
  loom_compile_report_capture_deinitialize(&capture);
}

}  // namespace
}  // namespace loom
