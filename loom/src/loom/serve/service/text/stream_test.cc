// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/stream.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::string String(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class TextStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(allocator_, &content_);
    iree_string_builder_initialize(allocator_, &reasoning_);
    iree_string_builder_initialize(allocator_, &scratch_);
    loom_serve_text_tool_calls_initialize(allocator_, &calls_);
    loom_serve_text_request_t request = {};
    iree_string_builder_initialize(allocator_, &request.body);
    IREE_ASSERT_OK(iree_string_builder_append_cstring(&request.body, "{}"));
    const char* body = iree_string_builder_buffer(&request.body);
    std::string name = "conversation";
    std::string model = "text";
    request.session = iree_make_cstring_view(name.c_str());
    auto status = loom_serve_text_stream_allocate(
        iree_make_cstring_view(model.c_str()), &request, allocator_, &stream_,
        &consumer_);
    iree_string_builder_deinitialize(&request.body);
    IREE_ASSERT_OK(status);
    const auto* owned = loom_serve_text_stream_request(stream_);
    EXPECT_EQ(iree_string_builder_buffer(&owned->body), body);
    EXPECT_EQ(request.session.data, nullptr);
    name.assign(name.size(), 'x');
    model.assign(model.size(), 'x');
    EXPECT_EQ(String(owned->session), "conversation");
    IREE_ASSERT_OK(consumer_.vtable->begin(consumer_.self, &response_));
  }

  void TearDown() override {
    if (consumer_.self) {
      consumer_.vtable->end(consumer_.self, false, IREE_STATUS_CANCELLED);
    }
    if (stream_) {
      loom_serve_text_stream_close(stream_);
    }
    loom_serve_text_tool_calls_deinitialize(&calls_);
    iree_string_builder_deinitialize(&scratch_);
    iree_string_builder_deinitialize(&reasoning_);
    iree_string_builder_deinitialize(&content_);
  }

  void Append(loom_serve_text_chat_output_kind_t kind,
              const std::string& text) {
    const loom_serve_text_chat_output_t output = {
        .advanced = true,
        .kind = kind,
        .text = iree_make_string_view(text.data(), text.size()),
    };
    IREE_ASSERT_OK(consumer_.vtable->append(consumer_.self, &response_, &output,
                                            &scratch_));
  }

  void Publish(bool expected_ready, bool expected_progress) {
    bool ready = false, progress = false;
    IREE_ASSERT_OK(
        consumer_.vtable->publish(consumer_.self, &ready, &progress));
    EXPECT_EQ(ready, expected_ready);
    EXPECT_EQ(progress, expected_progress);
  }

  void End(bool complete, iree_status_code_t code) {
    consumer_.vtable->end(consumer_.self, complete, code);
    consumer_ = {};
  }

  // Caller-owned allocator, independent of a service or source VM.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Caller receipt, retained through final output inspection.
  loom_serve_text_stream_t* stream_ = nullptr;
  // Execution receipt, consumed only after its simulated finite loan retires.
  loom_serve_text_consumer_t consumer_ = {};
  // Semantic metadata supplied by the source execution owner.
  loom_serve_text_chat_response_t response_ = {
      .model_name = IREE_SV("text"),
      .request_id = 7,
      .created_unix_seconds = 12,
      .flags = LOOM_SERVE_TEXT_CHAT_FLAG_STREAM,
      .usage = {.input_tokens = 6, .cached_tokens = 4, .output_tokens = 2},
  };
  // Owned source answer moved by successful completion.
  iree_string_builder_t content_ = {};
  // Owned source reasoning moved by successful completion.
  iree_string_builder_t reasoning_ = {};
  // Shared finite-loan scratch; no borrow may escape encoding.
  iree_string_builder_t scratch_ = {};
  // Complete validated transaction moved into the native result.
  loom_serve_text_tool_calls_t calls_ = {};
};

TEST_F(TextStreamTest, OrderedCreditAndMovedFinalOutputOutliveExecution) {
  ASSERT_NO_FATAL_FAILURE(Publish(true, false));
  for (size_t i = 0; i < 6; ++i) {
    std::string span(100 * (i + 1), 'a' + i);
    ASSERT_NO_FATAL_FAILURE(Append(i % 2
                                       ? LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT
                                       : LOOM_SERVE_TEXT_CHAT_OUTPUT_REASONING,
                                   span));
    span.assign(span.size(), 'x');
  }
  ASSERT_NO_FATAL_FAILURE(Publish(true, true));
  const auto* held = loom_serve_text_stream_peek(stream_);
  ASSERT_NE(held, nullptr);
  ASSERT_EQ(loom_serve_text_stream_block_count(held), 6u);
  ASSERT_NO_FATAL_FAILURE(Append(LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT, "last"));
  EXPECT_TRUE(consumer_.vtable->is_blocked(consumer_.self));
  ASSERT_NO_FATAL_FAILURE(Publish(false, false));
  for (size_t i = 0; i < 6; ++i) {
    const auto span = loom_serve_text_stream_block_at(held, i);
    EXPECT_EQ(span.kind, i % 2 ? LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT
                               : LOOM_SERVE_TEXT_CHAT_OUTPUT_REASONING);
    EXPECT_EQ(String(span.text), std::string(100 * (i + 1), 'a' + i));
  }
  IREE_ASSERT_OK(iree_string_builder_append_cstring(&content_, "answer λ"));
  IREE_ASSERT_OK(iree_string_builder_append_cstring(&reasoning_, "checked"));
  IREE_ASSERT_OK(loom_serve_text_tool_calls_append(
      &calls_, 7, IREE_SV("lookup"), IREE_SV(R"({"term":"λ"})")));
  const char* answer = iree_string_builder_buffer(&content_);
  IREE_ASSERT_OK(consumer_.vtable->complete(consumer_.self, &response_,
                                            &content_, &reasoning_, &calls_,
                                            IREE_SV("tool_calls"), &scratch_));
  EXPECT_EQ(iree_string_builder_size(&content_), 0u);
  EXPECT_EQ(iree_string_builder_size(&reasoning_), 0u);
  EXPECT_EQ(calls_.count, 0u);
  EXPECT_EQ(loom_serve_text_stream_result(stream_), nullptr);
  loom_serve_text_stream_consume(stream_);
  ASSERT_NO_FATAL_FAILURE(Publish(true, true));
  End(true, IREE_STATUS_OK);
  const auto* result = loom_serve_text_stream_result(stream_);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->content.data, answer);
  EXPECT_EQ(String(result->content), "answer λ");
  EXPECT_EQ(String(result->reasoning), "checked");
  EXPECT_EQ(String(result->response.model_name), "text");
  EXPECT_EQ(result->response.request_id, 7u);
  EXPECT_EQ(result->response.usage.cached_tokens, 4u);
  EXPECT_EQ(String(result->finish_reason), "tool_calls");
  ASSERT_EQ(result->calls->count, 1u);
  const auto call = loom_serve_text_tool_calls_at(result->calls, 0);
  EXPECT_EQ(String(call.id), "call_7_0");
  EXPECT_EQ(String(call.name), "lookup");
  EXPECT_EQ(String(call.arguments), R"({"term":"λ"})");
  EXPECT_FALSE(loom_serve_text_stream_is_complete(stream_));
  ASSERT_NE(loom_serve_text_stream_peek(stream_), nullptr);
  EXPECT_EQ(String(loom_serve_text_stream_block_at(
                       loom_serve_text_stream_peek(stream_), 0)
                       .text),
            "last");
  loom_serve_text_stream_consume(stream_);
  EXPECT_TRUE(loom_serve_text_stream_is_complete(stream_));
  IREE_EXPECT_OK(loom_serve_text_stream_status(stream_));
}

TEST_F(TextStreamTest, FailureDiscardsStageAndPublishesOwnedDiagnostic) {
  ASSERT_NO_FATAL_FAILURE(
      Append(LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT, "unpublished"));
  IREE_ASSERT_OK(consumer_.vtable->fail(
      consumer_.self, response_.flags,
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid later call")));
  // The CPU loan has not published into owner-visible status yet.
  IREE_EXPECT_OK(loom_serve_text_stream_status(stream_));
  ASSERT_NO_FATAL_FAILURE(Publish(true, true));
  End(true, IREE_STATUS_INVALID_ARGUMENT);
  const auto* block = loom_serve_text_stream_peek(stream_);
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(loom_serve_text_stream_block_count(block), 0u);
  loom_serve_text_stream_consume(stream_);
  EXPECT_TRUE(loom_serve_text_stream_is_complete(stream_));
  EXPECT_EQ(loom_serve_text_stream_result(stream_), nullptr);
  auto status = loom_serve_text_stream_status(stream_);
  EXPECT_EQ(String(iree_status_message(status)), "invalid later call");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(TextStreamTest, CancellationRetainsBorrowedBytesUntilCallerClose) {
  ASSERT_NO_FATAL_FAILURE(Append(LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT, "held"));
  ASSERT_NO_FATAL_FAILURE(Publish(true, true));
  const auto* block = loom_serve_text_stream_peek(stream_);
  ASSERT_NE(block, nullptr);
  const auto held = loom_serve_text_stream_block_at(block, 0);
  End(false, IREE_STATUS_CANCELLED);
  EXPECT_EQ(loom_serve_text_stream_peek(stream_), nullptr);
  EXPECT_TRUE(loom_serve_text_stream_is_complete(stream_));
  EXPECT_EQ(String(held.text), "held");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_serve_text_stream_status(stream_));
}

TEST_F(TextStreamTest, AcceptedCpuLoanOutlivesCallerClose) {
  loom_serve_text_stream_close(stream_);
  stream_ = nullptr;
  EXPECT_FALSE(consumer_.vtable->is_open(consumer_.self));
  ASSERT_NO_FATAL_FAILURE(
      Append(LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT, "already accepted"));
  End(false, IREE_STATUS_CANCELLED);
}

TEST(TextStreamAdmissionTest, BorrowedInputStorageCannotTransferOwnership) {
  char storage[] = "{}";
  loom_serve_text_request_t request = {};
  iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                              &request.body);
  IREE_ASSERT_OK(iree_string_builder_append_cstring(&request.body, "{}"));
  loom_serve_text_stream_t* stream = nullptr;
  loom_serve_text_consumer_t consumer = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_stream_allocate(
                            IREE_SV("text"), &request, iree_allocator_system(),
                            &stream, &consumer));
  EXPECT_EQ(stream, nullptr);
  EXPECT_EQ(consumer.self, nullptr);
  EXPECT_EQ(iree_string_builder_buffer(&request.body), storage);
  EXPECT_EQ(String(iree_string_builder_view(&request.body)), "{}");
  iree_string_builder_deinitialize(&request.body);
}

}  // namespace
