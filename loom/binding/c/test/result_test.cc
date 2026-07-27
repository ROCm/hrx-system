// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "src/result.h"

#include <string>

#include "iree/testing/gtest.h"
#include "test/util.h"

namespace {

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ToString(loomc_byte_span_t value) {
  return std::string(reinterpret_cast<const char*>(value.data),
                     value.data_length);
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
  LOOMC_ASSERT_OK(loomc_result_add_diagnostic(result, &diagnostic));

  char format[] = "text";
  char identifier[] = "report";
  char contents[] = "hello";
  loomc_artifact_t artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_REPORT,
      /*.format=*/loomc_make_string_view(format, sizeof(format) - 1),
      /*.identifier=*/
      loomc_make_string_view(identifier, sizeof(identifier) - 1),
      /*.contents=*/loomc_make_byte_span(contents, sizeof(contents) - 1),
  };
  LOOMC_ASSERT_OK(loomc_result_add_artifact(result, &artifact));

  code[0] = 'X';
  message[0] = 'X';
  format[0] = 'X';
  identifier[0] = 'X';
  contents[0] = 'X';
  loomc_source_release(source);

  EXPECT_FALSE(loomc_result_succeeded(result));
  EXPECT_EQ(loomc_result_state(result), LOOMC_RESULT_STATE_FAILED);
  ASSERT_EQ(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* stored_diagnostic =
      loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(stored_diagnostic, nullptr);
  EXPECT_EQ(stored_diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_EQ(ToString(stored_diagnostic->code), "PARSE/001");
  EXPECT_EQ(ToString(stored_diagnostic->message), "expected a thing");
  EXPECT_EQ(ToString(loomc_source_identifier(stored_diagnostic->range.source)),
            "bad.loom");

  ASSERT_EQ(loomc_result_artifact_count(result), 1u);
  const loomc_artifact_t* stored_artifact = loomc_result_artifact_at(result, 0);
  ASSERT_NE(stored_artifact, nullptr);
  EXPECT_EQ(stored_artifact->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(ToString(stored_artifact->format), "text");
  EXPECT_EQ(ToString(stored_artifact->identifier), "report");
  EXPECT_EQ(ToString(stored_artifact->contents), "hello");

  loomc_result_release(result);
}

TEST(ResultTest, RejectsMalformedArtifact) {
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                                      loomc_allocator_system(), &result));

  loomc_artifact_t artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_REPORT,
      /*.format=*/loomc_make_cstring_view("text"),
      /*.identifier=*/loomc_make_cstring_view("broken"),
      /*.contents=*/
      {
          /*.data=*/nullptr,
          /*.data_length=*/1,
      },
  };
  loomc_status_t status = loomc_result_add_artifact(result, &artifact);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  loomc_result_release(result);
}

}  // namespace
