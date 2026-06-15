// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/json_sink.h"

#include <string.h>

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"

namespace loom {
namespace {

static iree_status_t EmitJsonStatus(const loom_diagnostic_t* diagnostic,
                                    std::string* out_json,
                                    loom_type_formatter_t type_formatter = {
                                        nullptr, nullptr}) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_json_sink_options_t options = {
      /*.stream=*/&stream,
      /*.type_formatter=*/type_formatter,
  };
  iree_status_t status = loom_diagnostic_json_sink(&options, diagnostic);
  out_json->clear();
  if (iree_status_is_ok(status)) {
    *out_json = std::string(iree_string_builder_buffer(&builder),
                            iree_string_builder_size(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

// Helper: emit a diagnostic through the JSON sink and return the output.
std::string EmitJson(const loom_diagnostic_t* diagnostic,
                     loom_type_formatter_t type_formatter = {nullptr,
                                                             nullptr}) {
  std::string result;
  IREE_EXPECT_OK(EmitJsonStatus(diagnostic, &result, type_formatter));
  return result;
}

std::string EmitJsonObject(const loom_diagnostic_t* diagnostic,
                           loom_type_formatter_t type_formatter = {nullptr,
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

//===----------------------------------------------------------------------===//
// Basic structured diagnostics
//===----------------------------------------------------------------------===//

TEST(JsonSink, SimpleStructuredError) {
  // ERR_PARSE_001: "undefined SSA value '%{value_name}'"
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("x")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_PARSER;

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"severity\":\"error\""), std::string::npos);
  EXPECT_NE(json.find("\"error_id\":\"ERR_PARSE_001\""), std::string::npos);
  EXPECT_NE(json.find("\"domain\":\"PARSE\""), std::string::npos);
  EXPECT_NE(json.find("\"code\":1"), std::string::npos);
  EXPECT_NE(json.find("\"summary\":\"Undefined SSA value.\""),
            std::string::npos);
  EXPECT_NE(json.find("\"emitter\":\"parser\""), std::string::npos);
  EXPECT_NE(json.find("\"message\":\"undefined SSA value '%x'\""),
            std::string::npos);
  // PARSE_001 has no fix hint.
  EXPECT_EQ(json.find("\"fix_hint\""), std::string::npos);
  // Params object.
  EXPECT_NE(json.find("\"params\":{"), std::string::npos);
  EXPECT_NE(json.find("\"value_name\":\"x\""), std::string::npos);
  // Ends with newline.
  EXPECT_EQ(json.back(), '\n');
}

TEST(JsonSink, ObjectWriterOmitsTrailingNewline) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("x")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_PARSER;

  std::string json = EmitJsonObject(&diagnostic);
  EXPECT_NE(json.find("\"error_id\":\"ERR_PARSE_001\""), std::string::npos);
  ASSERT_FALSE(json.empty());
  EXPECT_EQ(json.back(), '}');
}

TEST(JsonSink, WarningFormat) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("y")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_WARNING;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"severity\":\"warning\""), std::string::npos);
}

TEST(JsonSink, BackendPressureRemarkIsStructured) {
  iree_string_view_t contributors[] = {
      IREE_SV("%acc0"),
      IREE_SV("%acc1"),
      IREE_SV("%a_frag0"),
      IREE_SV("%b_frag0"),
  };
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("amdgpu.gfx950")),
      loom_param_string(IREE_SV("matmul_q4")),
      loom_param_string(IREE_SV("wg128_mfma_i8_k64_unroll4")),
      loom_param_string(IREE_SV("matmul_q4_low")),
      loom_param_string(IREE_SV("vgpr")),
      loom_param_u32(96),
      loom_param_u32(132),
      loom_param_string(IREE_SV("^k_loop")),
      loom_param_string(IREE_SV("%acc8")),
      loom_param_string_list(contributors, IREE_ARRAYSIZE(contributors)),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_REMARK;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 3);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_PASS;

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"severity\":\"remark\""), std::string::npos);
  EXPECT_NE(json.find("\"error_id\":\"ERR_BACKEND_003\""), std::string::npos);
  EXPECT_NE(json.find("\"domain\":\"BACKEND\""), std::string::npos);
  EXPECT_NE(json.find("\"emitter\":\"pass\""), std::string::npos);
  EXPECT_NE(json.find("\"target_key\":\"amdgpu.gfx950\""), std::string::npos);
  EXPECT_NE(json.find("\"config_key\":\"wg128_mfma_i8_k64_unroll4\""),
            std::string::npos);
  EXPECT_NE(json.find("\"budget\":96"), std::string::npos);
  EXPECT_NE(json.find("\"peak\":132"), std::string::npos);
  EXPECT_NE(json.find("peak vgpr pressure is 132 unit(s) against budget 96"),
            std::string::npos);
  EXPECT_NE(json.find("\"contributors\":[\"%acc0\",\"%acc1\",\"%a_frag0\","
                      "\"%b_frag0\"]"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Structured diagnostics with type params
//===----------------------------------------------------------------------===//

TEST(JsonSink, StructuredSameType) {
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

  std::string json = EmitJson(&diagnostic, {loom_type_format_minimal, nullptr});

  // Check structured fields.
  EXPECT_NE(json.find("\"error_id\":\"ERR_TYPE_001\""), std::string::npos);
  EXPECT_NE(json.find("\"domain\":\"TYPE\""), std::string::npos);
  EXPECT_NE(json.find("\"code\":1"), std::string::npos);
  EXPECT_NE(json.find("\"summary\":\"SameType constraint violated.\""),
            std::string::npos);
  EXPECT_NE(json.find("\"emitter\":\"verifier\""), std::string::npos);

  // Check rendered message.
  EXPECT_NE(json.find("'lhs' type i32 does not match 'rhs' type f32"),
            std::string::npos);

  // Check fix_hint.
  EXPECT_NE(json.find("\"fix_hint\":"), std::string::npos);
  EXPECT_NE(json.find("Ensure 'lhs' and 'rhs' have the same type"),
            std::string::npos);

  // Check params object.
  EXPECT_NE(json.find("\"params\":{"), std::string::npos);
  EXPECT_NE(json.find("\"field_a\":\"lhs\""), std::string::npos);
  EXPECT_NE(json.find("\"type_a\":\"i32\""), std::string::npos);
  EXPECT_NE(json.find("\"field_b\":\"rhs\""), std::string::npos);
  EXPECT_NE(json.find("\"type_b\":\"f32\""), std::string::npos);
}

TEST(JsonSink, StructuredStructureError) {
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

  std::string json = EmitJson(&diagnostic);

  EXPECT_NE(json.find("\"domain\":\"STRUCTURE\""), std::string::npos);
  EXPECT_NE(json.find("\"code\":1"), std::string::npos);
  // STRUCTURE_001 has no fix_hint.
  EXPECT_EQ(json.find("\"fix_hint\""), std::string::npos);
  // U32 params render as numbers.
  EXPECT_NE(json.find("\"actual_count\":3"), std::string::npos);
  EXPECT_NE(json.find("\"expected_count\":2"), std::string::npos);
}

TEST(JsonSink, StructuredBytecodeRangeUsesU64Params) {
  loom_diagnostic_param_t params[4] = {
      loom_param_string(IREE_SV("module[0]")),
      loom_param_u64(UINT64_C(4294967296)),
      loom_param_u64(64),
      loom_param_u64(UINT64_C(4294967297)),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_BYTECODE, 7);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_BYTECODE_READER;

  std::string json = EmitJson(&diagnostic);

  EXPECT_NE(json.find("\"domain\":\"BYTECODE\""), std::string::npos);
  EXPECT_NE(json.find("\"emitter\":\"bytecode_reader\""), std::string::npos);
  EXPECT_NE(json.find("\"offset\":4294967296"), std::string::npos);
  EXPECT_NE(json.find("\"length\":64"), std::string::npos);
  EXPECT_NE(json.find("\"container_length\":4294967297"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// JSON escaping
//===----------------------------------------------------------------------===//

TEST(JsonSink, EscapesSpecialCharacters) {
  // ERR_PARSE_001 with a name containing special characters.
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("line1\nline2\ttab\"quote\\backslash")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("line1\\nline2\\ttab\\\"quote\\\\backslash"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Source locations and highlights
//===----------------------------------------------------------------------===//

TEST(JsonSink, SerializesSourceRangesAndHighlights) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("x")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0)),
  };
  const char source[] = "%x = test.constant 0 : i32";
  loom_highlight_range_t highlights[] = {
      {
          /*.start=*/0,
          /*.end=*/2,
          /*.field_ref=*/
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0),
          /*.param_index=*/0,
      },
      {5, 18},
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;
  diagnostic.origin.provenance = LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK;
  diagnostic.origin.filename = IREE_SV("<verifier>");
  diagnostic.origin.source = iree_make_cstring_view(source);
  diagnostic.origin.start = 0;
  diagnostic.origin.end = 26;
  diagnostic.origin.start_line = 1;
  diagnostic.origin.start_column = 1;
  diagnostic.origin.end_line = 1;
  diagnostic.origin.end_column = 27;
  diagnostic.source_location.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  diagnostic.source_location.filename = IREE_SV("model.loom");
  diagnostic.source_location.source = iree_make_cstring_view(source);
  diagnostic.source_location.start = 0;
  diagnostic.source_location.end = 26;
  diagnostic.source_location.start_line = 7;
  diagnostic.source_location.start_column = 3;
  diagnostic.source_location.end_line = 7;
  diagnostic.source_location.end_column = 29;
  diagnostic.highlights = highlights;
  diagnostic.highlight_count = IREE_ARRAYSIZE(highlights);

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"origin\":{\"provenance\":\"printed_ir_fallback\","
                      "\"filename\":\"<verifier>\",\"start_line\":1,"
                      "\"start_column\":1,\"end_line\":1,\"end_column\":27,"
                      "\"start_byte\":0,\"end_byte\":26,"
                      "\"excerpt\":{\"start_byte\":0,\"end_byte\":26,"
                      "\"truncated_prefix\":false,\"truncated_suffix\":false,"
                      "\"text\":\"%x = test.constant 0 : i32\"}}"),
            std::string::npos);
  EXPECT_NE(json.find("\"source_location\":{\"provenance\":\"exact_source\","
                      "\"filename\":\"model.loom\","
                      "\"start_line\":7,\"start_column\":3,\"end_line\":7,"
                      "\"end_column\":29,\"start_byte\":0,\"end_byte\":26,"
                      "\"excerpt\":{\"start_byte\":0,\"end_byte\":26,"
                      "\"truncated_prefix\":false,\"truncated_suffix\":false,"
                      "\"text\":\"%x = test.constant 0 : i32\"}}"),
            std::string::npos);
  EXPECT_NE(json.find("\"highlights\":[{\"start_byte\":0,\"end_byte\":2,"
                      "\"field\":{\"kind\":\"operand\",\"index\":0,"
                      "\"occurrence\":0},\"param\":\"value_name\"},"
                      "{\"start_byte\":5,\"end_byte\":18}]"),
            std::string::npos);
  EXPECT_NE(json.find("\"param_fields\":{\"value_name\":{"
                      "\"kind\":\"operand\",\"index\":0,"
                      "\"occurrence\":0}}"),
            std::string::npos);
}

TEST(JsonSink, SerializesSuccessorFieldRefs) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("test.br")),
      loom_param_with_field_ref(
          loom_param_u32(0),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_SUCCESSOR, 0)),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 23);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"param_fields\":{\"successor_index\":{"
                      "\"kind\":\"successor\",\"index\":0,"
                      "\"occurrence\":0}}"),
            std::string::npos);
}

TEST(JsonSink, SerializesClippedSourceExcerpt) {
  std::string source(320, 'a');
  source.replace(160, 3, "xyz");
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("x")),
  };
  loom_highlight_range_t highlights[] = {{
      /*.start=*/160,
      /*.end=*/163,
  }};
  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_PARSER;
  diagnostic.origin.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  diagnostic.origin.filename = IREE_SV("model.loom");
  diagnostic.origin.source =
      iree_make_string_view(source.data(), source.size());
  diagnostic.origin.start = 160;
  diagnostic.origin.end = 163;
  diagnostic.origin.start_line = 1;
  diagnostic.origin.start_column = 161;
  diagnostic.origin.end_line = 1;
  diagnostic.origin.end_column = 164;
  diagnostic.source_location = diagnostic.origin;
  diagnostic.highlights = highlights;
  diagnostic.highlight_count = IREE_ARRAYSIZE(highlights);

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"excerpt\":{\"start_byte\":128,\"end_byte\":320,"
                      "\"truncated_prefix\":true,\"truncated_suffix\":false,"),
            std::string::npos);
  EXPECT_NE(json.find("xyz"), std::string::npos);
  EXPECT_NE(json.find("\"highlights\":[{\"start_byte\":160,\"end_byte\":163}]"),
            std::string::npos);
}

TEST(JsonSink, SerializesUnavailableSourceProvenance) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("x")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;
  diagnostic.origin.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;
  diagnostic.source_location.provenance =
      LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"origin\":{\"provenance\":\"unavailable_source\","
                      "\"filename\":\"\",\"start_line\":0,\"start_column\":0,"
                      "\"end_line\":0,\"end_column\":0,"
                      "\"start_byte\":0,\"end_byte\":0,"
                      "\"excerpt\":{\"start_byte\":0,\"end_byte\":0,"
                      "\"truncated_prefix\":false,"
                      "\"truncated_suffix\":false}}"),
            std::string::npos);
  EXPECT_NE(
      json.find("\"source_location\":{\"provenance\":\"unavailable_source\","
                "\"filename\":\"\",\"start_line\":0,\"start_column\":0,"
                "\"end_line\":0,\"end_column\":0,"
                "\"start_byte\":0,\"end_byte\":0,"
                "\"excerpt\":{\"start_byte\":0,\"end_byte\":0,"
                "\"truncated_prefix\":false,\"truncated_suffix\":false}}"),
      std::string::npos);
}

TEST(JsonSink, SerializesRelatedLocations) {
  const char source_text[] =
      "%next = test.invoke @callee(%arg) : (f32) -> (%arg as f32)\n"
      "test.use %arg : f32\n";
  iree_host_size_t consume_length = strcspn(source_text, "\n");
  loom_diagnostic_related_location_t related_locations[] = {{
      /*.label=*/IREE_SV("consumed here"),
      /*.source_location=*/
      {
          /*.provenance=*/LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
          /*.filename=*/IREE_SV("model.loom"),
          /*.source=*/iree_make_cstring_view(source_text),
          /*.start=*/0,
          /*.end=*/consume_length,
          /*.start_line=*/3,
          /*.start_column=*/3,
          /*.end_line=*/3,
          /*.end_column=*/3 + (uint32_t)consume_length,
      },
  }};
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("arg")),
      loom_param_string(IREE_SV("test.invoke")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 2);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;
  diagnostic.related_locations = related_locations;
  diagnostic.related_location_count = IREE_ARRAYSIZE(related_locations);

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"related_locations\":[{\"label\":\"consumed here\","
                      "\"source_location\":{\"provenance\":\"exact_source\","
                      "\"filename\":\"model.loom\",\"start_line\":3,"
                      "\"start_column\":3,\"end_line\":3,\"end_column\":61,"
                      "\"start_byte\":0,\"end_byte\":58,"
                      "\"excerpt\":{\"start_byte\":0,\"end_byte\":58,"
                      "\"truncated_prefix\":false,\"truncated_suffix\":false,"
                      "\"text\":\"%next = test.invoke @callee(%arg) : (f32)"
                      " -> (%arg as f32)\"}}}]"),
            std::string::npos)
      << "json: " << json;
}

TEST(JsonSink, SerializesOmittedRelatedLocationCount) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("arg")),
      loom_param_string(IREE_SV("test.invoke")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 2);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;
  diagnostic.related_location_omitted_count = 5;

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"related_location_omitted_count\":5"),
            std::string::npos)
      << "json: " << json;
}

TEST(JsonSink, SerializesOmittedHighlightCount) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("test.invoke")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 2);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;
  diagnostic.highlight_omitted_count = 6;

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"highlight_omitted_count\":6"), std::string::npos)
      << "json: " << json;
}

TEST(JsonSink, RejectsInvalidParamFieldRefKind) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("x")),
          loom_diagnostic_field_ref((loom_diagnostic_field_kind_t)99, 0)),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);

  std::string json;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        EmitJsonStatus(&diagnostic, &json));
}

TEST(JsonSink, RejectsHighlightWithInvalidParamIndex) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("x")),
  };
  loom_highlight_range_t highlights[] = {{
      /*.start=*/0,
      /*.end=*/2,
      /*.field_ref=*/
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0),
      /*.param_index=*/1,
  }};

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.highlights = highlights;
  diagnostic.highlight_count = IREE_ARRAYSIZE(highlights);

  std::string json;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        EmitJsonStatus(&diagnostic, &json));
}

//===----------------------------------------------------------------------===//
// Multiple diagnostics (JSONL)
//===----------------------------------------------------------------------===//

TEST(JsonSink, MultipleDiagnostics) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_json_sink_options_t options = {
      /*.stream=*/&stream,
      /*.type_formatter=*/{nullptr, nullptr},
  };

  loom_diagnostic_param_t params1[] = {
      loom_param_string(IREE_SV("first")),
  };
  loom_diagnostic_t d1 = {};
  d1.severity = LOOM_DIAGNOSTIC_ERROR;
  d1.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  d1.params = params1;
  d1.param_count = IREE_ARRAYSIZE(params1);
  IREE_ASSERT_OK(loom_diagnostic_json_sink(&options, &d1));

  loom_diagnostic_param_t params2[] = {
      loom_param_string(IREE_SV("second")),
  };
  loom_diagnostic_t d2 = {};
  d2.severity = LOOM_DIAGNOSTIC_WARNING;
  d2.error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_PARSE, 1);
  d2.params = params2;
  d2.param_count = IREE_ARRAYSIZE(params2);
  IREE_ASSERT_OK(loom_diagnostic_json_sink(&options, &d2));

  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  // Two newline-terminated lines.
  size_t first_newline = output.find('\n');
  ASSERT_NE(first_newline, std::string::npos);
  size_t second_newline = output.find('\n', first_newline + 1);
  ASSERT_NE(second_newline, std::string::npos);

  std::string line1 = output.substr(0, first_newline);
  std::string line2 =
      output.substr(first_newline + 1, second_newline - first_newline - 1);

  EXPECT_NE(line1.find("%first"), std::string::npos);
  EXPECT_NE(line2.find("%second"), std::string::npos);
}

}  // namespace
}  // namespace loom
