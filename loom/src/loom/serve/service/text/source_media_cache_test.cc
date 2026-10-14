// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/image_service_test_fixture.h"

namespace {

using loom_serve::testing::text::ImageTextServiceTest;
using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

TEST_F(ImageTextServiceTest,
       ProjectsPixelsOnceAndKeepsRetainedHistoryIndependent) {
  ASSERT_EQ(interface_.capabilities.input_modalities.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(
      interface_.capabilities.input_modalities.values[0], IREE_SV("text")));
  EXPECT_TRUE(iree_string_view_equal(
      interface_.capabilities.input_modalities.values[1], IREE_SV("image")));
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_),
            LOOM_SERVE_FINITE_COMPONENT_DECLARED);

  const std::string image = ImageMessage("a", "image/png", kPng, "low", "b");
  Oracle oracle;
  oracle.Consume(Oracle::Id('a'));
  oracle.Consume(255);
  oracle.Consume(Oracle::Id('b'));
  const std::string first_expected = oracle.Generate(4);
  const std::string first = Content(CompleteRequest(
      Request(image, 4), "X-Loom-Session: projected-history\r\n"));
  EXPECT_EQ(first, first_expected);
  EXPECT_NE(first, Oracle::Complete("axb", 4));
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_),
            LOOM_SERVE_FINITE_COMPONENT_READY);

  loom_serve_finite_model_t* image_model = nullptr;
  IREE_ASSERT_OK(
      loom_serve_finite_component_prepare(image_component_, &image_model));
  ASSERT_NE(image_model, nullptr);
  const bool elastic = loom_serve_residency_snapshot(
                           loom_serve_finite_model_residency(image_model))
                           .elastic;
  const uint64_t compilation_commands =
      loom_serve_finite_component_report(image_component_)->commands;
  ASSERT_NO_FATAL_FAILURE(TrimImageComponent());
  const auto trimmed_phase =
      loom_serve_finite_component_phase(image_component_);
  EXPECT_EQ(trimmed_phase, elastic ? LOOM_SERVE_FINITE_COMPONENT_READY
                                   : LOOM_SERVE_FINITE_COMPONENT_DECLARED);
  if (elastic) {
    EXPECT_FALSE(loom_serve_residency_snapshot(
                     loom_serve_finite_model_residency(image_model))
                     .active);
  }

  oracle.Consume("#b");
  const std::string second_expected = oracle.Generate(3);
  const std::string continuation =
      image + "," + Message("assistant", first) + "," + Message("user", "b");
  const std::string second_response = CompleteRequest(
      Request(continuation, 3), "X-Loom-Session: projected-history\r\n");
  EXPECT_EQ(Content(second_response), second_expected);
  EXPECT_NE(second_response.find(R"("cached_tokens":6})"), std::string::npos)
      << second_response;
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_), trimmed_phase);
  EXPECT_EQ(loom_serve_finite_component_report(image_component_)->commands,
            compilation_commands);
  if (elastic) {
    EXPECT_FALSE(loom_serve_residency_snapshot(
                     loom_serve_finite_model_residency(image_model))
                     .active);
  }

  EXPECT_EQ(Content(CompleteRequest(Request(Message("user", "abxa"), 2))),
            Oracle::Complete("abxa", 2));
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_), trimmed_phase);
}

TEST_F(ImageTextServiceTest, SeparatesTokenIdenticalMediaPrefixes) {
  const std::string prompt = "abxxabb";
  const auto run = [&](const std::string& before, const std::string& mime_type,
                       const std::string& base64, const std::string& detail,
                       const std::string& after, size_t cached) {
    const std::string message =
        ImageMessage(before, mime_type, base64, detail, after);
    const std::string response = CompleteRequest(Request(message, 2));
    Oracle oracle;
    oracle.Consume(before);
    oracle.Consume(255);
    oracle.Consume(after);
    EXPECT_EQ(Content(response), oracle.Generate(2));
    EXPECT_NE(
        response.find("\"cached_tokens\":" + std::to_string(cached) + "}"),
        std::string::npos)
        << response;
    EXPECT_NE(response.find(
                  "\"prompt_tokens\":" + std::to_string(prompt.size()) + ","),
              std::string::npos)
        << response;
  };
  run("abx", "image/png", kPng, "low", "abb", 0);
  run("ab", "image/png", kPng, "low", "xabb", 0);
  run("abx", "image/png", kPng, "low", "abb", 4);
  run("abx", "image/gif", kGif, "low", "abb", 0);
  run("abx", "image/png", kPng, "low", "abb", 4);
  run("abx", "image/png", kPng, "high", "abb", 0);
  run("abx", "image/png", kPng, "low", "abb", 4);
}

TEST_F(SourceTextServiceTest, RetainedHistoryRequiresExactOrderedMedia) {
  const auto* policy = loom_serve_text_model_chat_policy(model_);
  const std::string first_body =
      Request(ImageMessage("x", "image/png", kPng, "low", ""), 4);
  const auto continuation =
      [&](const std::string& before, const std::string& mime_type,
          const std::string& base64, const std::string& detail,
          const std::string& after) {
        return Request(ImageMessage(before, mime_type, base64, detail, after) +
                           "," + Message("assistant", "b") + "," +
                           Message("user", "x"),
                       4);
      };
  loom_serve_text_chat_t first = {};
  loom_serve_text_chat_t matching = {};
  loom_serve_text_chat_t different_image = {};
  loom_serve_text_chat_t different_detail = {};
  loom_serve_text_chat_t different_placement = {};
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(first_body.c_str()), 10, 4,
      host_allocator_, &first));
  const std::string matching_body =
      continuation("x", "image/png", kPng, "low", "");
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(matching_body.c_str()), 11,
      4, host_allocator_, &matching));
  const std::string different_image_body =
      continuation("x", "image/gif", kGif, "low", "");
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name,
      iree_make_cstring_view(different_image_body.c_str()), 12, 4,
      host_allocator_, &different_image));
  const std::string different_detail_body =
      continuation("x", "image/png", kPng, "high", "");
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name,
      iree_make_cstring_view(different_detail_body.c_str()), 13, 4,
      host_allocator_, &different_detail));
  const std::string different_placement_body =
      continuation("", "image/png", kPng, "low", "x");
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name,
      iree_make_cstring_view(different_placement_body.c_str()), 14, 4,
      host_allocator_, &different_placement));

  iree_string_builder_t tools;
  iree_string_builder_initialize(host_allocator_, &tools);
  loom_serve_text_chat_completion_t completion = {};
  IREE_ASSERT_OK(loom_serve_text_chat_complete(&first, IREE_SV("b"), 10, &tools,
                                               &completion));
  loom_serve_text_chat_completion_t retained = {};
  loom_serve_text_chat_completion_retain(&completion, &retained);
  loom_serve_text_chat_completion_deinitialize(&completion);
  const loom_serve_text_input_prefix_t prefix = {
      .completion = &retained,
      .position = 1,
      .token = Oracle::Id('b'),
      .boundary = LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN,
  };
  loom_serve_text_input_t input = {};
  int32_t scratch[96];
  IREE_ASSERT_OK(loom_serve_text_input_prepare(&matching, prefix, true,
                                               IREE_ARRAYSIZE(scratch), scratch,
                                               &input, host_allocator_));
  EXPECT_EQ(input.retained_count, 1u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_input_prepare(&different_image, prefix, true,
                                    IREE_ARRAYSIZE(scratch), scratch, &input,
                                    host_allocator_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_input_prepare(&different_detail, prefix, true,
                                    IREE_ARRAYSIZE(scratch), scratch, &input,
                                    host_allocator_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_input_prepare(&different_placement, prefix, true,
                                    IREE_ARRAYSIZE(scratch), scratch, &input,
                                    host_allocator_));

  loom_serve_text_input_deinitialize(&input, host_allocator_);
  loom_serve_text_chat_completion_deinitialize(&retained);
  iree_string_builder_deinitialize(&tools);
  loom_serve_text_chat_deinitialize(&different_placement);
  loom_serve_text_chat_deinitialize(&different_detail);
  loom_serve_text_chat_deinitialize(&different_image);
  loom_serve_text_chat_deinitialize(&matching);
  loom_serve_text_chat_deinitialize(&first);
}

}  // namespace
