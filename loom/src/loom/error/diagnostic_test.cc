// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/diagnostic.h"

#include <string.h>

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/error/renderer.h"

namespace loom {
namespace {

// Helper to format a structured diagnostic and return the output string.
std::string FormatStructured(
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count, loom_diagnostic_severity_t severity,
    const char* source_text = nullptr, iree_host_size_t start = 0,
    iree_host_size_t end = 0, uint32_t line = 0, uint32_t column = 0) {
  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = severity;
  diagnostic.error = error;
  diagnostic.params = params;
  diagnostic.param_count = param_count;
  if (source_text) {
    iree_string_view_t source = iree_make_cstring_view(source_text);
    diagnostic.origin.filename = IREE_SV("test.loom");
    diagnostic.origin.source = source;
    diagnostic.origin.start = start;
    diagnostic.origin.end = end;
    diagnostic.origin.start_line = line;
    diagnostic.origin.start_column = column;
    diagnostic.origin.end_line = line;
    diagnostic.origin.end_column = column + (uint32_t)(end - start);
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status = loom_diagnostic_format(&diagnostic, &stream);
  std::string result;
  if (iree_status_is_ok(status)) {
    result = std::string(iree_string_builder_buffer(&builder),
                         iree_string_builder_size(&builder));
  }
  IREE_EXPECT_OK(status);
  iree_string_builder_deinitialize(&builder);
  return result;
}

//===----------------------------------------------------------------------===//
// Caret formatting (using PARSE errors for simple message templates)
//===----------------------------------------------------------------------===//

TEST(Diagnostic, ErrorWithCaret) {
  //                 0         1         2
  //                 0123456789012345678901234567
  const char* src = "%r = test.addi %x, %y : i32";
  //                                    ^^ %y at offset 19-21

  // ERR_PARSE_001: "undefined SSA value '%{value_name}'"
  // The param is the bare name (no '%' prefix); the template adds it.
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("y")),
  };
  std::string output = FormatStructured(
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1), params,
      IREE_ARRAYSIZE(params), LOOM_DIAGNOSTIC_ERROR, src, 19, 21, 1, 20);

  // Clang-style: file:line:col: severity [DOMAIN/CODE]: message
  EXPECT_NE(output.find("test.loom:1:20: error [PARSE/001]"), std::string::npos)
      << "output: " << output;
  EXPECT_NE(output.find("undefined SSA value '%y'"), std::string::npos)
      << "output: " << output;

  // Verify exact alignment: ^^ must be directly under %y.
  size_t source_line_pos = output.find(" 1 | %r");
  ASSERT_NE(source_line_pos, std::string::npos);
  size_t source_line_end = output.find('\n', source_line_pos);
  std::string source_line =
      output.substr(source_line_pos, source_line_end - source_line_pos);
  size_t caret_line_start = source_line_end + 1;
  size_t caret_line_end = output.find('\n', caret_line_start);
  std::string caret_line =
      output.substr(caret_line_start, caret_line_end - caret_line_start);
  size_t percent_pos = source_line.rfind("%y");
  ASSERT_NE(percent_pos, std::string::npos);
  size_t caret_pos = caret_line.find('^');
  ASSERT_NE(caret_pos, std::string::npos);
  EXPECT_EQ(percent_pos, caret_pos) << "source: '" << source_line << "'\n"
                                    << "carets: '" << caret_line << "'";
}

TEST(Diagnostic, MultiTokenUnderline) {
  // ERR_PARSE_003: "expected {expected}, got '{actual}'"
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("an operand")),
      loom_param_string(IREE_SV("test.addi")),
  };
  std::string output =
      FormatStructured(loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 3),
                       params, IREE_ARRAYSIZE(params), LOOM_DIAGNOSTIC_ERROR,
                       "%r = test.addi %x, %y : i32", 5, 15, 1, 6);
  EXPECT_NE(output.find("^^^^^^^^^^"), std::string::npos);
}

TEST(Diagnostic, OmittedHighlightsFormatAsNote) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("x")),
  };
  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.highlight_omitted_count = 3;
  diagnostic.origin.filename = IREE_SV("test.loom");
  diagnostic.origin.source = IREE_SV("%x = test.produce : i32");
  diagnostic.origin.start = 0;
  diagnostic.origin.end = 2;
  diagnostic.origin.start_line = 1;
  diagnostic.origin.start_column = 1;
  diagnostic.origin.end_line = 1;
  diagnostic.origin.end_column = 3;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_diagnostic_format(&diagnostic, &stream));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(output.find("3 additional highlights omitted"), std::string::npos)
      << "output: " << output;
}

TEST(Diagnostic, WarningFormat) {
  // ERR_PARSE_003 with WARNING severity (severity comes from the diagnostic,
  // not the error def).
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("a result")),
      loom_param_string(IREE_SV("test.yield")),
  };
  std::string output =
      FormatStructured(loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 3),
                       params, IREE_ARRAYSIZE(params), LOOM_DIAGNOSTIC_WARNING,
                       "test.yield %x : f32", 0, 10, 5, 1);
  EXPECT_NE(output.find("test.loom:5:1: warning"), std::string::npos);
}

TEST(Diagnostic, RemarkFormat) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("input")),
  };
  std::string output =
      FormatStructured(loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1),
                       params, IREE_ARRAYSIZE(params), LOOM_DIAGNOSTIC_REMARK,
                       "%x = test.neg %input : f32", 0, 2, 10, 1);
  EXPECT_NE(output.find("remark"), std::string::npos);
}

TEST(Diagnostic, SingleCharCaret) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("')'")),
      loom_param_string(IREE_SV("]")),
  };
  std::string output = FormatStructured(
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 3), params,
      IREE_ARRAYSIZE(params), LOOM_DIAGNOSTIC_ERROR, "( ] )", 2, 3, 1, 3);
  EXPECT_NE(output.find("^"), std::string::npos);
}

TEST(Diagnostic, MultiLineSource) {
  // Source with multiple lines. Error on line 2.
  const char* source = "%a = test.neg %b : f32\n%c = test.addi %a, %z : i32";
  //                      line 2 starts at offset 23
  //                      %z is at offset 43-45
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("z")),
  };
  std::string output = FormatStructured(
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1), params,
      IREE_ARRAYSIZE(params), LOOM_DIAGNOSTIC_ERROR, source, 43, 45, 2, 21);
  EXPECT_NE(output.find("test.loom:2:21: error"), std::string::npos);
  EXPECT_NE(output.find("test.addi"), std::string::npos);
  EXPECT_NE(output.find("^^"), std::string::npos);
}

TEST(Diagnostic, NoSource) {
  // Diagnostic without source — just severity + message, no caret.
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("missing")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.origin.filename = IREE_SV("test.loom");
  diagnostic.origin.source = iree_string_view_empty();

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_diagnostic_format(&diagnostic, &stream));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(output.find("error"), std::string::npos);
  EXPECT_NE(output.find("undefined SSA value '%missing'"), std::string::npos);
  // No caret or source line.
  EXPECT_EQ(output.find(" | "), std::string::npos);
}

TEST(Diagnostic, RelatedLocationsFormatAsNotes) {
  const char* source_text =
      "%next = test.invoke @callee(%arg) : (f32) -> (%arg as f32)\n"
      "test.use %arg : f32\n";
  iree_string_view_t source = iree_make_cstring_view(source_text);

  const char* consume_text = strstr(source_text, "%next = test.invoke");
  ASSERT_NE(consume_text, nullptr);
  iree_host_size_t consume_start =
      (iree_host_size_t)(consume_text - source_text);
  iree_host_size_t consume_length = strcspn(consume_text, "\n");
  loom_diagnostic_related_location_t related_locations[] = {{
      /*.label=*/IREE_SV("consumed here"),
      /*.source_location=*/
      {
          /*.provenance=*/LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
          /*.filename=*/IREE_SV("test.loom"),
          /*.source=*/source,
          /*.start=*/consume_start,
          /*.end=*/consume_start + consume_length,
          /*.start_line=*/1,
          /*.start_column=*/1,
          /*.end_line=*/1,
          /*.end_column=*/1 + (uint32_t)consume_length,
      },
  }};

  const char* use_text = strstr(source_text, "test.use %arg : f32");
  ASSERT_NE(use_text, nullptr);
  iree_host_size_t use_start = (iree_host_size_t)(use_text - source_text);
  iree_host_size_t use_length = strcspn(use_text, "\n");
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("arg")),
      loom_param_string(IREE_SV("test.invoke")),
  };
  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 2);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.origin = {
      /*.provenance=*/LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
      /*.filename=*/IREE_SV("test.loom"),
      /*.source=*/source,
      /*.start=*/use_start,
      /*.end=*/use_start + use_length,
      /*.start_line=*/2,
      /*.start_column=*/1,
      /*.end_line=*/2,
      /*.end_column=*/1 + (uint32_t)use_length,
  };
  diagnostic.related_locations = related_locations;
  diagnostic.related_location_count = IREE_ARRAYSIZE(related_locations);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_diagnostic_format(&diagnostic, &stream));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(output.find("test.loom:2:1: error [DOMINANCE/002]"),
            std::string::npos)
      << "output: " << output;
  EXPECT_NE(output.find("use of consumed value 'arg' (consumed by tied "
                        "result of 'test.invoke')"),
            std::string::npos)
      << "output: " << output;
  EXPECT_NE(output.find("  = note[consumed here]: test.loom:1:1"),
            std::string::npos)
      << "output: " << output;
  EXPECT_NE(output.find("%next = test.invoke @callee(%arg)"), std::string::npos)
      << "output: " << output;
}

TEST(Diagnostic, OmittedRelatedLocationsFormatAsNote) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("arg")),
      loom_param_string(IREE_SV("test.invoke")),
  };
  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 2);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.related_location_omitted_count = 3;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_diagnostic_format(&diagnostic, &stream));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(output.find("  = note: 3 additional related locations omitted"),
            std::string::npos)
      << "output: " << output;
}

//===----------------------------------------------------------------------===//
// Structured diagnostic formatting
//===----------------------------------------------------------------------===//

TEST(Diagnostic, StructuredErrorCodeInOutput) {
  // A structured diagnostic should include the error code in the output.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_diagnostic_param_t params[4] = {
      loom_param_string(IREE_SV("lhs")),
      loom_param_type(i32_type),
      loom_param_string(IREE_SV("rhs")),
      loom_param_type(f32_type),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1);
  diagnostic.params = params;
  diagnostic.param_count = 4;
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_diagnostic_format(&diagnostic, &stream));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  // Error code should be present.
  EXPECT_NE(output.find("[TYPE/001]"), std::string::npos)
      << "output: " << output;
  // Fix hint should be present.
  EXPECT_NE(output.find("= help:"), std::string::npos) << "output: " << output;
  EXPECT_NE(output.find("Ensure 'lhs' and 'rhs' have the same type"),
            std::string::npos)
      << "output: " << output;
}

TEST(Diagnostic, StructuredWithSourceRange) {
  const char* src = "%r = test.addi %x, %y : i32";
  loom_diagnostic_param_t params[3] = {
      loom_param_string(IREE_SV("test.addi")),
      loom_param_u32(3),
      loom_param_u32(2),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1);
  diagnostic.params = params;
  diagnostic.param_count = 3;
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;
  diagnostic.origin.filename = IREE_SV("test.loom");
  diagnostic.origin.source = iree_make_cstring_view(src);
  diagnostic.origin.start = 5;
  diagnostic.origin.end = 15;
  diagnostic.origin.start_line = 1;
  diagnostic.origin.start_column = 6;
  diagnostic.origin.end_line = 1;
  diagnostic.origin.end_column = 16;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_diagnostic_format(&diagnostic, &stream));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  // Clang-style location on first line with error code.
  EXPECT_NE(output.find("test.loom:1:6: error [STRUCTURE/001]:"),
            std::string::npos)
      << "output: " << output;
  // Source line and caret should be present.
  EXPECT_NE(output.find("test.addi"), std::string::npos);
  EXPECT_NE(output.find("^^^^^^^^^^"), std::string::npos);
  // STRUCTURE_001 has no fix hint.
  EXPECT_EQ(output.find("= help:"), std::string::npos);
}

TEST(Diagnostic, NoFixHintForNullTemplate) {
  loom_diagnostic_param_t params[3] = {
      loom_param_string(IREE_SV("test.addi")),
      loom_param_u32(0),
      loom_param_u32(1),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 2);
  diagnostic.params = params;
  diagnostic.param_count = 3;

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_diagnostic_format(&diagnostic, &stream));
  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_EQ(output.find("= help:"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Source range from token
//===----------------------------------------------------------------------===//

TEST(Diagnostic, SourceRangeFromToken) {
  iree_string_view_t source = IREE_SV("%r = test.addi %x, %y : i32");
  iree_string_view_t token_text =
      iree_make_string_view(source.data + 20, 2);  // "%y"
  loom_source_range_t range = loom_source_range_from_token(
      IREE_SV("test.loom"), source, token_text, 1, 21, 23);
  EXPECT_EQ(range.provenance, LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_EQ(range.start, 20u);
  EXPECT_EQ(range.end, 22u);
  EXPECT_EQ(range.start_line, 1u);
  EXPECT_EQ(range.start_column, 21u);
  EXPECT_EQ(range.end_column, 23u);
}

//===----------------------------------------------------------------------===//
// Sink
//===----------------------------------------------------------------------===//

// Test sink that collects diagnostics into a vector.
struct CollectedDiagnostic {
  loom_diagnostic_severity_t severity;
  const loom_error_def_t* error;
};

static iree_status_t collecting_sink(void* user_data,
                                     const loom_diagnostic_t* diagnostic) {
  auto* collected = static_cast<std::vector<CollectedDiagnostic>*>(user_data);
  collected->push_back({diagnostic->severity, diagnostic->error});
  return iree_ok_status();
}

TEST(Diagnostic, SinkCollects) {
  std::vector<CollectedDiagnostic> collected;
  loom_diagnostic_sink_t sink = {collecting_sink, &collected};

  loom_diagnostic_param_t params1[] = {
      loom_param_string(IREE_SV("x")),
  };
  loom_diagnostic_t d1 = {};
  d1.severity = LOOM_DIAGNOSTIC_ERROR;
  d1.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  d1.params = params1;
  d1.param_count = IREE_ARRAYSIZE(params1);
  IREE_ASSERT_OK(loom_diagnostic_emit(&sink, &d1));

  loom_diagnostic_param_t params2[] = {
      loom_param_string(IREE_SV("y")),
  };
  loom_diagnostic_t d2 = {};
  d2.severity = LOOM_DIAGNOSTIC_WARNING;
  d2.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  d2.params = params2;
  d2.param_count = IREE_ARRAYSIZE(params2);
  IREE_ASSERT_OK(loom_diagnostic_emit(&sink, &d2));

  EXPECT_EQ(collected.size(), 2u);
  EXPECT_EQ(collected[0].severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_EQ(collected[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1));
  EXPECT_EQ(collected[1].severity, LOOM_DIAGNOSTIC_WARNING);
  EXPECT_EQ(collected[1].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1));
}

TEST(Diagnostic, NullSinkDoesNotCrash) {
  loom_diagnostic_sink_t null_sink = {NULL, NULL};
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("dropped")),
  };
  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  IREE_ASSERT_OK(loom_diagnostic_emit(&null_sink, &diagnostic));
  IREE_ASSERT_OK(loom_diagnostic_emit(NULL, &diagnostic));
}

}  // namespace
}  // namespace loom
