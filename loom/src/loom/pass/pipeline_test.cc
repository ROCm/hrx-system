// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/pipeline.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(PassPipelineParseTest, ConsumesNamesAndOptions) {
  iree_string_view_t pipeline = IREE_SV("canonicalize{max-iterations=20}, dce");
  loom_pass_pipeline_entry_spec_t entry = {};
  bool has_entry = false;

  IREE_ASSERT_OK(
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));
  EXPECT_TRUE(has_entry);
  EXPECT_TRUE(iree_string_view_equal(entry.name, IREE_SV("canonicalize")));
  EXPECT_TRUE(
      iree_string_view_equal(entry.options, IREE_SV("max-iterations=20")));

  IREE_ASSERT_OK(
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));
  EXPECT_TRUE(has_entry);
  EXPECT_TRUE(iree_string_view_equal(entry.name, IREE_SV("dce")));
  EXPECT_TRUE(iree_string_view_is_empty(entry.options));

  IREE_ASSERT_OK(
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));
  EXPECT_FALSE(has_entry);
}

typedef struct parsed_option_t {
  // Last parsed option name.
  iree_string_view_t name;
  // Last parsed option value.
  iree_string_view_t value;
  // Number of parsed assignments.
  int count;
} parsed_option_t;

static iree_status_t capture_option(void* user_data, iree_string_view_t name,
                                    iree_string_view_t value) {
  parsed_option_t* parsed_option = (parsed_option_t*)user_data;
  parsed_option->name = name;
  parsed_option->value = value;
  ++parsed_option->count;
  return iree_ok_status();
}

TEST(PassPipelineParseTest, ParsesOptionAssignments) {
  parsed_option_t parsed_option = {};
  IREE_ASSERT_OK(loom_pass_options_parse(IREE_SV("canonicalize"),
                                         IREE_SV("max-iterations = 7"),
                                         (loom_pass_option_parse_callback_t){
                                             .fn = capture_option,
                                             .user_data = &parsed_option,
                                         }));
  EXPECT_EQ(parsed_option.count, 1);
  EXPECT_TRUE(
      iree_string_view_equal(parsed_option.name, IREE_SV("max-iterations")));
  EXPECT_TRUE(iree_string_view_equal(parsed_option.value, IREE_SV("7")));
}

TEST(PassPipelineParseTest, RejectsMalformedPipelineEntries) {
  iree_string_view_t pipeline = IREE_SV("canonicalize{max-iterations=1");
  loom_pass_pipeline_entry_spec_t entry = {};
  bool has_entry = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));

  pipeline = IREE_SV(",dce");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));

  pipeline = IREE_SV("dce,");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));
}

TEST(PassPipelineParseTest, RejectsMalformedOptionAssignments) {
  parsed_option_t parsed_option = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_options_parse(
                            IREE_SV("canonicalize"), IREE_SV("max-iterations"),
                            (loom_pass_option_parse_callback_t){
                                .fn = capture_option,
                                .user_data = &parsed_option,
                            }));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_options_parse(IREE_SV("canonicalize"), IREE_SV("=7"),
                              (loom_pass_option_parse_callback_t){
                                  .fn = capture_option,
                                  .user_data = &parsed_option,
                              }));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_options_parse(IREE_SV("canonicalize"),
                              IREE_SV("max-iterations=7,"),
                              (loom_pass_option_parse_callback_t){
                                  .fn = capture_option,
                                  .user_data = &parsed_option,
                              }));
}

TEST(PassPipelineParseTest, ParsesUint32OptionValues) {
  uint32_t value = 0;
  IREE_ASSERT_OK(loom_pass_option_parse_uint32(IREE_SV("canonicalize"),
                                               IREE_SV("max-iterations"),
                                               IREE_SV("42"), &value));
  EXPECT_EQ(value, 42u);
  IREE_ASSERT_OK(loom_pass_option_parse_uint32(IREE_SV("canonicalize"),
                                               IREE_SV("max-iterations"),
                                               IREE_SV("0"), &value));
  EXPECT_EQ(value, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_option_parse_uint32(IREE_SV("canonicalize"),
                                                      IREE_SV("max-iterations"),
                                                      IREE_SV("many"), &value));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_option_parse_uint32(IREE_SV("canonicalize"),
                                                      IREE_SV("max-iterations"),
                                                      IREE_SV("-1"), &value));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_option_parse_uint32(
                            IREE_SV("canonicalize"), IREE_SV("max-iterations"),
                            IREE_SV("4294967296"), &value));
}

}  // namespace
}  // namespace loom
