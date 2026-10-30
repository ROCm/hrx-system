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
#include "loom/serve/service/text/checkpoint_http.h"
#include "loom/serve/testing/http_harness.h"

namespace {

class TextHttpFixture : public ::testing::Test {
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
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(http_.TakeRequest(client_.get(), &connection_, &request));
    receipt_ = loom_serve_text_telemetry_begin(&telemetry_);
  }

  void AllocateConsumer() {
    auto status = loom_serve_text_http_consumer_allocate(
        connection_, &telemetry_, receipt_, false, iree_allocator_system(),
        &consumer_);
    IREE_ASSERT_OK(status);
    connection_ = nullptr;
  }

  void TearDown() override {
    if (consumer_.self) {
      consumer_.vtable->end(consumer_.self, false, IREE_STATUS_CANCELLED);
    }
    if (connection_) {
      loom_serve_http_connection_abort(connection_);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    iree_async_proactor_pool_release(pool_);
    loom_serve_text_telemetry_deinitialize(&telemetry_);
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
    if (iree_any_bit_set(response_.flags, LOOM_SERVE_TEXT_CHAT_FLAG_STREAM)) {
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
  // Undelivered claim, transferred to the selected consumer on allocation.
  loom_serve_http_connection_t* connection_ = nullptr;
  // Admission identity shared by the transport and telemetry.
  loom_serve_text_receipt_t receipt_ = {};
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

class TextHttpTest : public TextHttpFixture,
                     public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(TextHttpFixture::SetUp());
    ASSERT_NO_FATAL_FAILURE(AllocateConsumer());
    response_.flags = GetParam() ? LOOM_SERVE_TEXT_CHAT_FLAG_STREAM : 0;
    IREE_ASSERT_OK(consumer_.vtable->begin(consumer_.self, &response_));
    ASSERT_NO_FATAL_FAILURE(Publish());
  }
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
  IREE_ASSERT_OK(loom_serve_text_tool_calls_append(
      &calls_, 7, IREE_SV("read"), IREE_SV(R"({"q":"a","q":"b"})")));
  IREE_ASSERT_OK(loom_serve_text_tool_calls_append(
      &calls_, 7, IREE_SV("unknown_tool"), IREE_SV(R"({"path":)")));
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
  EXPECT_NE(
      response.find(
          R"("id":"call_7_1","type":"function","function":{"name":"read","arguments":"{\"q\":\"a\",\"q\":\"b\"}"})"),
      std::string::npos);
  EXPECT_NE(
      response.find(
          R"("id":"call_7_2","type":"function","function":{"name":"unknown_tool","arguments":"{\"path\":"})"),
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
      R"({"error":{"message":"invalid \"tool\"","type":"loom_output_error","code":"INVALID_ARGUMENT"}})";
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

TEST_P(TextHttpTest, LargeEscapedCallsSurvivePacketAndTransportGrowth) {
  const size_t lengths[] = {0, 1023, 8191, 16383, 32767, 65535};
  std::string expected_calls = "[";
  for (size_t i = 0; i < IREE_ARRAYSIZE(lengths); ++i) {
    if (i) {
      expected_calls += ',';
    }
    expected_calls += '{';
    if (GetParam()) {
      expected_calls += "\"index\":" + std::to_string(i) + ',';
    }
    expected_calls +=
        "\"id\":\"call_7_" + std::to_string(i) +
        R"(","type":"function","function":{"name":"task","arguments":"{\"prompt\":\")";
    // Put an escape across a 10,000-byte boundary. A cut after the first
    // backslash would leave an invalid escape before the SSE suffix.
    const size_t padding_length =
        i == 0
            ? 9999 - IREE_SV(R"({"tool_calls":)").size - expected_calls.size()
            : lengths[i];
    const std::string padding(padding_length, static_cast<char>('a' + i));
    const std::string arguments =
        R"({"prompt":")" + padding + R"(\n quote \" slash \\ tab \t λ"})";
    IREE_ASSERT_OK(loom_serve_text_tool_calls_append(
        &calls_, 7, IREE_SV("task"),
        iree_make_string_view(arguments.data(), arguments.size())));
    expected_calls += padding + R"(\\n quote \\\" slash \\\\ tab \\t λ\"}"}})";
  }
  expected_calls += ']';
  IREE_ASSERT_OK(consumer_.vtable->complete(consumer_.self, &response_,
                                            &content_, &reasoning_, &calls_,
                                            IREE_SV("tool_calls"), &scratch_));
  loom_serve_text_tool_calls_reset(&calls_);
  iree_string_builder_reset(&scratch_);
  ASSERT_NO_FATAL_FAILURE(Publish());
  const auto response = Finish();
  const auto calls_begin = response.find("\"tool_calls\":");
  ASSERT_NE(calls_begin, std::string::npos);
  const auto actual_calls =
      response.substr(calls_begin + 13, expected_calls.size());
  size_t matching = 0;
  while (matching < actual_calls.size() && matching < expected_calls.size() &&
         actual_calls[matching] == expected_calls[matching]) {
    ++matching;
  }
  EXPECT_EQ(matching, expected_calls.size())
      << "actual length " << actual_calls.size() << " actual suffix "
      << actual_calls.substr(matching, 100) << " expected suffix "
      << expected_calls.substr(matching, 100);
  EXPECT_EQ(response.find("data: [DONE]\n\n") != std::string::npos, GetParam());
}

INSTANTIATE_TEST_SUITE_P(Delivery, TextHttpTest, ::testing::Bool());

struct RejectionCase {
  // Owned failure received from admission or retired checkpoint work.
  iree_status_code_t status;
  // Public status line prefix, independent of the status diagnostic string.
  const char* http;
};

class TextHttpRejectionTest
    : public TextHttpFixture,
      public ::testing::WithParamInterface<RejectionCase> {};

TEST_P(TextHttpRejectionTest, PreservesFailureCodeAndAvailability) {
  ASSERT_NO_FATAL_FAILURE(AllocateConsumer());
  const auto test = GetParam();
  auto consumer = consumer_;
  consumer_ = {};
  IREE_ASSERT_OK(consumer.vtable->reject(consumer.self,
                                         iree_status_from_code(test.status)));
  std::string response;
  IREE_ASSERT_OK(http_.Read(client_.release(), false, &response));
  EXPECT_EQ(response.find(test.http), 0u) << response;
  const std::string code = iree_status_code_string(test.status);
  EXPECT_NE(response.find("\"code\":\"" + code + "\""), std::string::npos);
  EXPECT_NE(response.find("\"message\":\"" + code + "\""), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    Admission, TextHttpRejectionTest,
    ::testing::Values(
        RejectionCase{IREE_STATUS_INVALID_ARGUMENT, "HTTP/1.1 400 "},
        RejectionCase{IREE_STATUS_OUT_OF_RANGE, "HTTP/1.1 400 "},
        RejectionCase{IREE_STATUS_NOT_FOUND, "HTTP/1.1 404 "},
        RejectionCase{IREE_STATUS_FAILED_PRECONDITION, "HTTP/1.1 409 "},
        RejectionCase{IREE_STATUS_UNIMPLEMENTED, "HTTP/1.1 501 "},
        RejectionCase{IREE_STATUS_RESOURCE_EXHAUSTED, "HTTP/1.1 503 "},
        RejectionCase{IREE_STATUS_UNAVAILABLE, "HTTP/1.1 503 "},
        RejectionCase{IREE_STATUS_CANCELLED, "HTTP/1.1 503 "},
        RejectionCase{IREE_STATUS_UNKNOWN, "HTTP/1.1 500 "},
        RejectionCase{IREE_STATUS_INTERNAL, "HTTP/1.1 500 "},
        RejectionCase{IREE_STATUS_DATA_LOSS, "HTTP/1.1 500 "}));

class TextCheckpointHttpRejectionTest
    : public TextHttpFixture,
      public ::testing::WithParamInterface<RejectionCase> {};

TEST_P(TextCheckpointHttpRejectionTest, PreservesFailureCodeAndAvailability) {
  auto consumer = loom_serve_text_checkpoint_http_consumer(connection_);
  connection_ = nullptr;
  const loom_serve_text_checkpoint_request_t request = {
      .action = LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
      .name = IREE_SV("checkpoint"),
      .session = IREE_SV("session"),
  };
  const auto test = GetParam();
  IREE_ASSERT_OK(consumer.end(consumer.self, &request, nullptr,
                              iree_status_from_code(test.status), &telemetry_,
                              &receipt_, &scratch_));
  std::string response;
  IREE_ASSERT_OK(http_.Read(client_.release(), false, &response));
  EXPECT_EQ(response.find(test.http), 0u) << response;
  const std::string code = iree_status_code_string(test.status);
  EXPECT_NE(response.find("\"code\":\"" + code + "\""), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    Checkpoint, TextCheckpointHttpRejectionTest,
    ::testing::Values(
        RejectionCase{IREE_STATUS_NOT_FOUND, "HTTP/1.1 404 "},
        RejectionCase{IREE_STATUS_ALREADY_EXISTS, "HTTP/1.1 409 "},
        RejectionCase{IREE_STATUS_FAILED_PRECONDITION, "HTTP/1.1 409 "},
        RejectionCase{IREE_STATUS_RESOURCE_EXHAUSTED, "HTTP/1.1 503 "},
        RejectionCase{IREE_STATUS_UNAVAILABLE, "HTTP/1.1 503 "},
        RejectionCase{IREE_STATUS_CANCELLED, "HTTP/1.1 503 "},
        RejectionCase{IREE_STATUS_INTERNAL, "HTTP/1.1 500 "},
        RejectionCase{IREE_STATUS_DATA_LOSS, "HTTP/1.1 500 "}));

}  // namespace
