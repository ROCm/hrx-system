// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/chat.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ChatResponseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &output_);
    loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls_);
  }
  void TearDown() override {
    loom_serve_text_tool_calls_deinitialize(&calls_);
    iree_string_builder_deinitialize(&output_);
  }
  std::string Output() const {
    const auto view = iree_string_builder_view(&output_);
    return std::string(view.data, view.size);
  }

  // Wire metadata is borrowed only while each encoder runs.
  loom_serve_text_chat_response_t response_ = {
      .model_name = IREE_SV("local-λ"),
      .request_id = 7,
      .created_unix_seconds = 12,
      .usage = {.input_tokens = 120, .cached_tokens = 100, .output_tokens = 4},
  };
  // Owned output from the production JSON/SSE encoder.
  iree_string_builder_t output_;
  // Validated generated records consumed directly by the production encoder.
  loom_serve_text_tool_calls_t calls_;
};

TEST_F(ChatResponseTest, CompleteTextIncludesUsageAndEscapesContent) {
  IREE_ASSERT_OK(loom_serve_text_chat_response(
      &response_, IREE_SV("a\n\"λ\""), iree_string_view_empty(), nullptr,
      IREE_SV("stop"), &output_));
  EXPECT_EQ(
      Output(),
      R"({"id":"loom-7","object":"chat.completion","created":12,"model":"local-λ","choices":[{"index":0,"message":{"role":"assistant","content":"a\n\"λ\""},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":120,"completion_tokens":4,"total_tokens":124,"prompt_tokens_details":{"cached_tokens":100}}})");
}

TEST_F(ChatResponseTest, ToolOnlyMessageHasNullContent) {
  IREE_ASSERT_OK(loom_serve_text_tool_calls_append(&calls_, 7, IREE_SV("read"),
                                                   IREE_SV(R"({"path":"a"})")));
  IREE_ASSERT_OK(loom_serve_text_chat_response(
      &response_, iree_string_view_empty(), iree_string_view_empty(), &calls_,
      IREE_SV("tool_calls"), &output_));
  EXPECT_EQ(
      Output(),
      R"({"id":"loom-7","object":"chat.completion","created":12,"model":"local-λ","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_7_0","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]},"finish_reason":"tool_calls","logprobs":null}],"usage":{"prompt_tokens":120,"completion_tokens":4,"total_tokens":124,"prompt_tokens_details":{"cached_tokens":100}}})");
}

TEST_F(ChatResponseTest, ReasoningIsSeparateFromAnswerAndToolFields) {
  IREE_ASSERT_OK(loom_serve_text_chat_response(
      &response_, IREE_SV("4"), IREE_SV("check \"λ\"\n"), nullptr,
      IREE_SV("stop"), &output_));
  EXPECT_NE(
      Output().find(R"("content":"4","reasoning_content":"check \"λ\"\n")"),
      std::string::npos);
}

TEST_F(ChatResponseTest, StreamingUsageIsNullUntilTheUsageOnlyChunk) {
  response_.flags = LOOM_SERVE_TEXT_CHAT_FLAG_STREAM |
                    LOOM_SERVE_TEXT_CHAT_FLAG_INCLUDE_USAGE;
  IREE_ASSERT_OK(
      loom_serve_text_chat_event(&response_, IREE_SV(R"({"content":"a"})"),
                                 iree_string_view_empty(), &output_));
  IREE_ASSERT_OK(loom_serve_text_chat_event(&response_, IREE_SV("{}"),
                                            IREE_SV("length"), &output_));
  IREE_ASSERT_OK(loom_serve_text_chat_usage(&response_, &output_));
  EXPECT_EQ(
      Output(),
      "data: "
      R"({"id":"loom-7","object":"chat.completion.chunk","created":12,"model":"local-λ","choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null,"logprobs":null}],"usage":null})"
      "\n\ndata: "
      R"({"id":"loom-7","object":"chat.completion.chunk","created":12,"model":"local-λ","choices":[{"index":0,"delta":{},"finish_reason":"length","logprobs":null}],"usage":null})"
      "\n\ndata: "
      R"({"id":"loom-7","object":"chat.completion.chunk","created":12,"model":"local-λ","choices":[],"usage":{"prompt_tokens":120,"completion_tokens":4,"total_tokens":124,"prompt_tokens_details":{"cached_tokens":100}}})"
      "\n\n");
}

TEST_F(ChatResponseTest, StreamingWithoutUsageOmitsTheField) {
  response_.flags = LOOM_SERVE_TEXT_CHAT_FLAG_STREAM;
  IREE_ASSERT_OK(
      loom_serve_text_chat_event(&response_, IREE_SV(R"({"role":"assistant"})"),
                                 iree_string_view_empty(), &output_));
  EXPECT_EQ(
      Output(),
      "data: "
      R"({"id":"loom-7","object":"chat.completion.chunk","created":12,"model":"local-λ","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null,"logprobs":null}]})"
      "\n\n");
}

TEST_F(ChatResponseTest, UnixSecondsAreNotScaledOrNarrowed) {
  for (int64_t timestamp :
       {INT64_C(0), INT64_C(2147483648), INT64_C(4294967297)}) {
    SCOPED_TRACE(timestamp);
    response_.created_unix_seconds = timestamp;
    iree_string_builder_reset(&output_);
    IREE_ASSERT_OK(loom_serve_text_chat_event(&response_, IREE_SV("{}"),
                                              IREE_SV("stop"), &output_));
    EXPECT_EQ(
        Output(),
        "data: "
        R"({"id":"loom-7","object":"chat.completion.chunk","created":)" +
            std::to_string(timestamp) +
            R"(,"model":"local-λ","choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":null}]})"
            "\n\n");
  }
}

}  // namespace
