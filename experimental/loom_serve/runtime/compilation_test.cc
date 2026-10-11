// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/compilation.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::string Format(const loom_serve_compilation_report_t& report) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t output;
  loom_output_stream_for_builder(&builder, &output);
  IREE_EXPECT_OK(loom_serve_compilation_write(&report, &output));
  const auto view = iree_string_builder_view(&builder);
  std::string text(view.data ? view.data : "", view.size);
  iree_string_builder_deinitialize(&builder);
  return text;
}

TEST(CompilationTest, EmptyReportHasNoOutput) { EXPECT_EQ(Format({}), ""); }

TEST(CompilationTest, DirectStageWorkHasNoCommandProduct) {
  EXPECT_EQ(Format({.kernels = 1,
                    .reused_kernels = 2,
                    .entries = 3,
                    .workers = 1,
                    .duration_ns = 456}),
            "{\"event\":\"compilation\",\"commands\":0,\"kernels\":1,"
            "\"reused_kernels\":2,\"entries\":3,\"workers\":1,"
            "\"duration_ns\":456,\"notes\":0,\"warnings\":0,\"errors\":0,"
            "\"omitted_diagnostics\":0,\"diagnostics\":[]}\n");
}

TEST(CompilationTest, OwnedDiagnosticAndWorkHaveExactJson) {
  const loom_serve_compilation_report_t report = {
      .severities = {0, 1, 0},
      .diagnostic_count = 1,
      .diagnostics = {{.severity = LOOMC_DIAGNOSTIC_SEVERITY_WARNING,
                       .line = 9,
                       .column = 2,
                       .omitted_locations = 3,
                       .omitted_parameters = 1,
                       .source_length = 7,
                       .code_length = 5,
                       .message_length = 4,
                       .source = "a\".loom",
                       .code = "W/123",
                       .message = "a\n\"b"}},
      .commands = 2,
      .kernels = 8,
      .reused_kernels = 4,
      .entries = 16,
      .workers = 4,
      .duration_ns = 123};
  EXPECT_EQ(Format(report),
            "{\"event\":\"compilation\",\"commands\":2,\"kernels\":8,"
            "\"reused_kernels\":4,\"entries\":16,\"workers\":4,"
            "\"duration_ns\":123,\"notes\":0,"
            "\"warnings\":1,\"errors\":0,\"omitted_diagnostics\":0,"
            "\"diagnostics\":[{\"severity\":1,\"line\":9,\"column\":2,"
            "\"omitted_locations\":3,\"omitted_parameters\":1,"
            "\"source_length\":7,\"source\":\"a\\\".loom\","
            "\"code_length\":5,\"code\":\"W/123\","
            "\"message_length\":4,\"message\":\"a\\n\\\"b\"}]}\n");
}

TEST(CompilationTest, MergeRetainsBoundedEvidenceAndAllCounters) {
  loom_serve_compilation_report_t report = {};
  for (uint64_t i = 0; i < 12; ++i) {
    loom_serve_compilation_report_t source = {};
    source.severities[i % 3] = 1;
    source.diagnostic_count = 1;
    source.diagnostics[0].severity = i % 3;
    source.diagnostics[0].line = i + 1;
    source.commands = 1;
    source.kernels = 2;
    source.reused_kernels = 4;
    source.entries = 3;
    source.workers = 1 + i % 4;
    source.duration_ns = 100;
    loom_serve_compilation_merge(&report, &source);
  }
  EXPECT_EQ(report.diagnostic_count, IREE_ARRAYSIZE(report.diagnostics));
  for (size_t i = 0; i < report.diagnostic_count; ++i) {
    EXPECT_EQ(report.diagnostics[i].line, i + 1);
  }
  for (const auto count : report.severities) {
    EXPECT_EQ(count, 4u);
  }
  EXPECT_EQ(report.commands, 12u);
  EXPECT_EQ(report.kernels, 24u);
  EXPECT_EQ(report.reused_kernels, 48u);
  EXPECT_EQ(report.entries, 36u);
  EXPECT_EQ(report.workers, 4u);
  EXPECT_EQ(report.duration_ns, 1200u);
  EXPECT_NE(Format(report).find("\"omitted_diagnostics\":4"),
            std::string::npos);
}

TEST(CompilationTest, OriginalExtentsExposeTruncatedStrings) {
  loom_serve_compilation_report_t report = {};
  report.severities[LOOMC_DIAGNOSTIC_SEVERITY_ERROR] = 1;
  report.diagnostic_count = 1;
  auto& diagnostic = report.diagnostics[0];
  diagnostic.severity = LOOMC_DIAGNOSTIC_SEVERITY_ERROR;
  diagnostic.source_length = sizeof(diagnostic.source) + 10;
  diagnostic.code_length = sizeof(diagnostic.code) + 20;
  diagnostic.message_length = sizeof(diagnostic.message) + 30;
  memset(diagnostic.source, 's', sizeof(diagnostic.source));
  memset(diagnostic.code, 'c', sizeof(diagnostic.code));
  memset(diagnostic.message, 'm', sizeof(diagnostic.message));
  const std::string text = Format(report);
  EXPECT_NE(text.find("\"source_length\":266,\"source\":\"" +
                      std::string(256, 's') + "\""),
            std::string::npos);
  EXPECT_NE(
      text.find("\"code_length\":84,\"code\":\"" + std::string(64, 'c') + "\""),
      std::string::npos);
  EXPECT_NE(text.find("\"message_length\":542,\"message\":\"" +
                      std::string(512, 'm') + "\""),
            std::string::npos);
}

}  // namespace
