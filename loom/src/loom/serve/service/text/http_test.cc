// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/http.h"

#include <string>

#include "iree/async/util/proactor_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/testing/http_harness.h"

namespace {

class TextHttpTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &content_);
    iree_string_builder_initialize(iree_allocator_system(), &reasoning_);
    iree_string_builder_initialize(iree_allocator_system(), &scratch_);
    loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls_);
    IREE_ASSERT_OK(loom_serve_text_telemetry_initialize(
        nullptr, iree_allocator_system(), &telemetry_));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    const auto options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        pool_, &options, iree_allocator_system(), &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(
        server_, &listener_options, iree_allocator_system(), &listener_));
    IREE_ASSERT_OK(http_.Initialize(server_, listener_));
    IREE_ASSERT_OK(
        http_.Send("POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\n"
                   "Content-Length: 2\r\n\r\n{}",
                   client_.put()));
    loom_serve_http_connection_t* connection = nullptr;
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(http_.TakeRequest(client_.get(), &connection, &request));
    const auto receipt = loom_serve_text_telemetry_begin(&telemetry_);
    auto status = loom_serve_text_http_consumer_allocate(
        connection, &telemetry_, receipt, iree_allocator_system(), &consumer_);
    if (!iree_status_is_ok(status)) {
      loom_serve_http_connection_abort(connection);
    }
    IREE_ASSERT_OK(status);
    response_.flags = GetParam() ? LOOM_SERVE_TEXT_CHAT_FLAG_STREAM : 0;
    IREE_ASSERT_OK(consumer_.vtable->begin(consumer_.self, &response_));
    ASSERT_NO_FATAL_FAILURE(Publish());
  }

  void TearDown() override {
    if (consumer_.self) {
      consumer_.vtable->end(consumer_.self, false, IREE_STATUS_CANCELLED);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    iree_async_proactor_pool_release(pool_);
    loom_serve_telemetry_stream_free(telemetry_.stream);
    loom_serve_text_tool_calls_deinitialize(&calls_);
    iree_string_builder_deinitialize(&scratch_);
    iree_string_builder_deinitialize(&reasoning_);
    iree_string_builder_deinitialize(&content_);
  }

  void Publish() {
    auto* notification = loom_serve_http_server_notification(server_);
    bool ready = false;
    while (!ready) {
      const auto token = iree_notification_prepare_wait(notification);
      bool progress = false;
      auto status =
          consumer_.vtable->publish(consumer_.self, &ready, &progress);
      if (ready || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, token, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
    }
  }

  void Append(loom_serve_text_chat_output_kind_t kind,
              iree_string_view_t text) {
    if (GetParam()) {
      const loom_serve_text_chat_output_t output = {
          .advanced = true,
          .kind = kind,
          .text = text,
      };
      IREE_ASSERT_OK(consumer_.vtable->append(consumer_.self, &response_,
                                              &output, &scratch_));
    }
  }

  std::string Finish(iree_status_code_t code = IREE_STATUS_OK) {
    consumer_.vtable->end(consumer_.self, true, code);
    consumer_ = {};
    std::string response;
    IREE_EXPECT_OK(http_.Read(client_.release(), false, &response));
    return response;
  }

  // Real transport and manually serialized service-side publication.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Transport outlives its consumer claim and listener.
  loom_serve_http_server_t* server_ = nullptr;
  // Loopback listener bound to an ephemeral address.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Client readiness without a private service loop or timing deadline.
  loom_serve::testing::HttpHarness http_;
  // Client descriptor closed even if an assertion exits early.
  loom_serve::testing::HttpClient client_;
  // Owner accounting borrowed by the production adapter.
  loom_serve_text_telemetry_t telemetry_ = {};
  // Owned production receipt; teardown cancels an unfinished transaction.
  loom_serve_text_consumer_t consumer_ = {};
  // Deterministic wire metadata independent of wall time.
  loom_serve_text_chat_response_t response_ = {
      .model_name = IREE_SV("text"),
      .request_id = 7,
      .created_unix_seconds = 12,
      .usage = {.input_tokens = 6, .cached_tokens = 4, .output_tokens = 2},
  };
  // Completed answer storage borrowed by the adapter.
  iree_string_builder_t content_ = {};
  // Independent completed reasoning channel.
  iree_string_builder_t reasoning_ = {};
  // Finite output-loan scratch, reused between semantic spans.
  iree_string_builder_t scratch_ = {};
  // Validated complete call table borrowed through final encoding.
  loom_serve_text_tool_calls_t calls_ = {};
};

TEST_P(TextHttpTest, SemanticChannelsAndCallsReachTheSameClaim) {
  IREE_ASSERT_OK(iree_string_builder_append_cstring(&content_, "yes"));
  IREE_ASSERT_OK(iree_string_builder_append_cstring(&reasoning_, "check λ"));
  ASSERT_NO_FATAL_FAILURE(
      Append(LOOM_SERVE_TEXT_CHAT_OUTPUT_REASONING, IREE_SV("check λ")));
  ASSERT_NO_FATAL_FAILURE(
      Append(LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT, IREE_SV("yes")));
  IREE_ASSERT_OK(loom_serve_text_tool_calls_append(&calls_, 7, IREE_SV("read"),
                                                   IREE_SV(R"({"path":"a"})")));
  IREE_ASSERT_OK(consumer_.vtable->complete(consumer_.self, &response_,
                                            &content_, &reasoning_, &calls_,
                                            IREE_SV("tool_calls"), &scratch_));
  // The adapter encoded its own stage before these shared scratch owners reset.
  iree_string_builder_reset(&content_);
  iree_string_builder_reset(&reasoning_);
  loom_serve_text_tool_calls_reset(&calls_);
  iree_string_builder_reset(&scratch_);
  ASSERT_NO_FATAL_FAILURE(Publish());
  const auto response = Finish();
  EXPECT_EQ(response.find("HTTP/1.1 200 OK\r\n"), 0u) << response;
  EXPECT_NE(response.find(R"("content":"yes")"), std::string::npos);
  EXPECT_NE(response.find(R"("reasoning_content":"check λ")"),
            std::string::npos);
  EXPECT_NE(
      response.find(
          R"("id":"call_7_0","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"})"),
      std::string::npos);
  EXPECT_NE(response.find(R"("finish_reason":"tool_calls")"),
            std::string::npos);
  EXPECT_EQ(response.find("data: [DONE]\n\n") != std::string::npos, GetParam());
}

TEST_P(TextHttpTest, TranslationFailureDiscardsUnpublishedOutput) {
  ASSERT_NO_FATAL_FAILURE(
      Append(LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT, IREE_SV("unpublished")));
  IREE_ASSERT_OK(consumer_.vtable->fail(
      consumer_.self, response_.flags,
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid \"tool\"")));
  ASSERT_NO_FATAL_FAILURE(Publish());
  const auto response = Finish(IREE_STATUS_INVALID_ARGUMENT);
  const std::string error =
      R"({"error":{"message":"invalid \"tool\"","type":"loom_request_error"}})";
  const auto headers_end = response.find("\r\n\r\n");
  ASSERT_NE(headers_end, std::string::npos) << response;
  if (GetParam()) {
    EXPECT_EQ(
        response.substr(headers_end + 4),
        "data: "
        R"({"id":"loom-7","object":"chat.completion.chunk","created":12,"model":"text","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null,"logprobs":null}]})"
        "\n\ndata: " +
            error + "\n\ndata: [DONE]\n\n");
  } else {
    EXPECT_EQ(response.find("HTTP/1.1 500 Internal Server Error\r\n"), 0u);
    EXPECT_EQ(response.substr(headers_end + 4), error);
  }
}

INSTANTIATE_TEST_SUITE_P(Delivery, TextHttpTest, ::testing::Bool());

}  // namespace
