// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "src/result.h"

#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ByteSequencePtr =
    HandlePtr<loomc_byte_sequence_t, loomc_byte_sequence_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ToString(loomc_byte_span_t value) {
  return std::string(reinterpret_cast<const char*>(value.data),
                     value.data_length);
}

std::string ToString(const loomc_byte_sequence_t* value) {
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_EXPECT_OK(
      loomc_byte_sequence_clone(value, loomc_allocator_system(), &contents));
  std::string result = ToString(contents);
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
  return result;
}

TEST(ResultTest, OwnsDiagnosticsAndArtifacts) {
  char source_text[] = "bad";
  loomc_source_options_t source_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("bad.loom"),
      /*.contents=*/loomc_make_byte_span(source_text, sizeof(source_text) - 1),
      /*.storage=*/LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source = nullptr;
  LOOMC_ASSERT_OK(
      loomc_source_create(&source_options, loomc_allocator_system(), &source));

  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_result_create(LOOMC_RESULT_STATE_FAILED,
                                      loomc_allocator_system(), &result));

  char code[] = "PARSE/001";
  char message[] = "expected a thing";
  loomc_diagnostic_t diagnostic = {
      /*.severity=*/LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      /*.code=*/loomc_make_string_view(code, sizeof(code) - 1),
      /*.message=*/loomc_make_string_view(message, sizeof(message) - 1),
      /*.range=*/
      {
          /*.source=*/source,
          /*.start=*/0,
          /*.end=*/3,
          /*.start_line=*/1,
          /*.start_column=*/1,
          /*.end_line=*/1,
          /*.end_column=*/4,
      },
  };
  char label[] = "defined here";
  loomc_diagnostic_related_location_t related = {};
  related.label = loomc_make_cstring_view(label);
  related.range = diagnostic.range;
  diagnostic.related_locations = &related;
  diagnostic.related_location_count = 1;
  diagnostic.related_location_omitted_count = 2;
  LOOMC_ASSERT_OK(loomc_result_add_diagnostic(result, &diagnostic));
  const auto* retained_related =
      loomc_result_diagnostic_at(result, 0)->related_locations;
  // Growing the result array must not move the owned note payload.
  for (int i = 0; i < 8; ++i) {
    LOOMC_ASSERT_OK(loomc_result_add_diagnostic(result, &diagnostic));
  }

  char format[] = "text";
  char identifier[] = "report";
  char contents[] = "hello";
  loomc_byte_sequence_t* contents_sequence = nullptr;
  LOOMC_ASSERT_OK(loomc_byte_sequence_create_copy(
      loomc_make_byte_span(contents, sizeof(contents) - 1),
      loomc_allocator_system(), &contents_sequence));
  ByteSequencePtr contents_owner(contents_sequence);
  loomc_artifact_t artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_REPORT,
      /*.format=*/loomc_make_string_view(format, sizeof(format) - 1),
      /*.identifier=*/
      loomc_make_string_view(identifier, sizeof(identifier) - 1),
      /*.contents=*/contents_sequence,
  };
  LOOMC_ASSERT_OK(loomc_result_add_artifact(result, &artifact));
  contents_owner.reset();

  code[0] = 'X';
  message[0] = 'X';
  label[0] = 'X';
  related.range = {};
  format[0] = 'X';
  identifier[0] = 'X';
  contents[0] = 'X';
  loomc_source_release(source);

  EXPECT_FALSE(loomc_result_succeeded(result));
  EXPECT_EQ(loomc_result_state(result), LOOMC_RESULT_STATE_FAILED);
  ASSERT_EQ(loomc_result_diagnostic_count(result), 9u);
  const loomc_diagnostic_t* stored_diagnostic =
      loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(stored_diagnostic, nullptr);
  EXPECT_EQ(stored_diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_EQ(ToString(stored_diagnostic->code), "PARSE/001");
  EXPECT_EQ(ToString(stored_diagnostic->message), "expected a thing");
  EXPECT_EQ(ToString(loomc_source_identifier(stored_diagnostic->range.source)),
            "bad.loom");
  ASSERT_EQ(stored_diagnostic->related_location_count, 1u);
  EXPECT_EQ(stored_diagnostic->related_location_omitted_count, 2u);
  EXPECT_EQ(stored_diagnostic->related_locations, retained_related);
  EXPECT_EQ(ToString(retained_related->label), "defined here");
  EXPECT_EQ(retained_related->range.source, stored_diagnostic->range.source);
  EXPECT_EQ(retained_related->range.end_column, 4u);
  EXPECT_EQ(ToString(loomc_source_contents(retained_related->range.source)),
            "bad");

  ASSERT_EQ(loomc_result_artifact_count(result), 1u);
  const loomc_artifact_t* stored_artifact = loomc_result_artifact_at(result, 0);
  ASSERT_NE(stored_artifact, nullptr);
  EXPECT_EQ(stored_artifact->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(ToString(stored_artifact->format), "text");
  EXPECT_EQ(ToString(stored_artifact->identifier), "report");
  EXPECT_EQ(ToString(stored_artifact->contents), "hello");

  loomc_result_release(result);
}

TEST(ResultTest, EmptyDiagnosticNeedsNoMetadataPayload) {
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_result_create(LOOMC_RESULT_STATE_FAILED,
                                      loomc_allocator_system(), &result));
  ResultPtr owner(result);
  const loomc_diagnostic_t diagnostic = {};
  LOOMC_ASSERT_OK(loomc_result_add_diagnostic(result, &diagnostic));
  const auto* stored = loomc_result_diagnostic_at(result, 0);
  EXPECT_EQ(stored->code.size, 0u);
  EXPECT_EQ(stored->message.size, 0u);
  EXPECT_EQ(stored->related_locations, nullptr);
  EXPECT_EQ(stored->related_location_count, 0u);
}

TEST(ResultTest, RejectsMalformedArtifact) {
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                                      loomc_allocator_system(), &result));

  loomc_artifact_t artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_REPORT,
      /*.format=*/loomc_make_cstring_view("text"),
      /*.identifier=*/loomc_make_cstring_view("broken"),
      /*.contents=*/nullptr,
  };
  loomc_status_t status = loomc_result_add_artifact(result, &artifact);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  loomc_result_release(result);
}

}  // namespace
