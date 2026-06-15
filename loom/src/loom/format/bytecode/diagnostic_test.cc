// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/diagnostic.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/json_sink.h"
#include "loom/error/renderer.h"

namespace loom {
namespace {

struct CapturedDiagnostic {
  loom_emitter_t emitter;         // Subsystem tag captured from the diagnostic.
  const loom_error_def_t* error;  // Structured error definition pointer.
  loom_source_range_t origin;     // Bytecode origin range captured by value.
  std::string text;               // Text-rendered diagnostic message.
  std::string json;               // JSON-rendered diagnostic object.
};

static iree_status_t FormatDiagnosticText(const loom_diagnostic_t* diagnostic,
                                          std::string* out_text) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status = loom_diagnostic_format(diagnostic, &stream);
  if (iree_status_is_ok(status)) {
    *out_text = std::string(iree_string_builder_buffer(&builder),
                            iree_string_builder_size(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t FormatDiagnosticJson(const loom_diagnostic_t* diagnostic,
                                          std::string* out_json) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status = loom_diagnostic_json_write_object(
      &stream, diagnostic, {loom_type_format_minimal, nullptr});
  if (iree_status_is_ok(status)) {
    *out_json = std::string(iree_string_builder_buffer(&builder),
                            iree_string_builder_size(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t CaptureDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  CapturedDiagnostic* captured = static_cast<CapturedDiagnostic*>(user_data);
  captured->emitter = diagnostic->emitter;
  captured->error = diagnostic->error;
  captured->origin = diagnostic->origin;
  IREE_RETURN_IF_ERROR(FormatDiagnosticText(diagnostic, &captured->text));
  return FormatDiagnosticJson(diagnostic, &captured->json);
}

TEST(BytecodeDiagnostic, InvalidRecordFieldRendersPathAndOffset) {
  CapturedDiagnostic captured = {};
  loom_bytecode_reader_diagnostic_context_t context = {
      /*.sink=*/{CaptureDiagnostic, &captured},
      /*.filename=*/IREE_SV("model.loombc"),
  };
  uint64_t offset = UINT64_C(4294967296);

  IREE_ASSERT_OK(loom_bytecode_reader_emit_invalid_record_field(
      &context, IREE_SV("SECTIONS"), IREE_SV("directory"), 3, IREE_SV("offset"),
      offset, IREE_SV("overlapping_range")));

  EXPECT_EQ(captured.emitter, LOOM_EMITTER_BYTECODE_READER);
  EXPECT_EQ(captured.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_BYTECODE, 6));
  EXPECT_EQ(captured.origin.provenance,
            LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE);
  EXPECT_EQ(captured.origin.start, (iree_host_size_t)offset);
  EXPECT_EQ(captured.origin.end, (iree_host_size_t)(offset + 1));
  EXPECT_NE(captured.text.find("error [BYTECODE/006]: invalid field 'offset' "
                               "in SECTIONS/directory[3] at offset "
                               "4294967296: overlapping_range"),
            std::string::npos)
      << captured.text;
  EXPECT_NE(captured.json.find("\"emitter\":\"bytecode_reader\""),
            std::string::npos)
      << captured.json;
  EXPECT_NE(captured.json.find("\"error_id\":\"ERR_BYTECODE_006\""),
            std::string::npos)
      << captured.json;
  EXPECT_NE(captured.json.find("\"filename\":\"model.loombc\""),
            std::string::npos)
      << captured.json;
  EXPECT_NE(captured.json.find("\"start_byte\":4294967296"), std::string::npos)
      << captured.json;
  EXPECT_NE(captured.json.find("\"section_name\":\"SECTIONS\""),
            std::string::npos)
      << captured.json;
  EXPECT_NE(captured.json.find("\"record_index\":3"), std::string::npos)
      << captured.json;
  EXPECT_NE(captured.json.find("\"offset\":4294967296"), std::string::npos)
      << captured.json;
  EXPECT_NE(captured.json.find("\"failure_code\":\"overlapping_range\""),
            std::string::npos)
      << captured.json;
}

TEST(BytecodeDiagnostic, InvalidRangeRendersStructuredBounds) {
  CapturedDiagnostic captured = {};
  loom_bytecode_reader_diagnostic_context_t context = {
      /*.sink=*/{CaptureDiagnostic, &captured},
      /*.filename=*/IREE_SV("model.loombc"),
  };

  IREE_ASSERT_OK(loom_bytecode_reader_emit_invalid_range(
      &context, IREE_SV("module[0].sections"), 120, 48, 128));

  EXPECT_EQ(captured.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_BYTECODE, 7));
  EXPECT_EQ(captured.origin.start, 120u);
  EXPECT_EQ(captured.origin.end, 168u);
  EXPECT_NE(captured.text.find("invalid range 'module[0].sections' at offset "
                               "120 with length 48; container length is 128"),
            std::string::npos)
      << captured.text;
  EXPECT_NE(captured.json.find("\"range_name\":\"module[0].sections\""),
            std::string::npos)
      << captured.json;
  EXPECT_NE(captured.json.find("\"length\":48"), std::string::npos)
      << captured.json;
  EXPECT_NE(captured.json.find("\"container_length\":128"), std::string::npos)
      << captured.json;
}

TEST(BytecodeDiagnostic, NullSinkDropsDiagnostic) {
  IREE_ASSERT_OK(loom_bytecode_reader_emit_invalid_range(
      nullptr, IREE_SV("file"), 0, 16, 8));
}

}  // namespace
}  // namespace loom
