// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/test_allocator.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/service/text/session.h"
#include "loom/serve/service/text/stream.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class TextResponseServiceTest : public SourceTextServiceTest,
                                public ::testing::WithParamInterface<size_t> {
 protected:
  bool NeedsStandaloneModel() const override { return false; }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    // The numerical model and prompt IDs are unchanged. Only the spelling of
    // predicted ID zero grows, so a few native predictions exercise complete
    // large responses without a long-running generation workload.
    spelling_.assign(GetParam(), 'a');
    const size_t repetitions = GetParam() > 8193 ? 16 : 1;
    const auto path = iree::testing::MakeTempFilePath("text_response", ".json");
    const std::string decoder =
        repetitions == 1
            ? R"({"type":"ByteFallback"})"
            : R"({"type":"Sequence","decoders":[{"type":"ByteFallback"},
                  {"type":"Fuse"},{"type":"Replace",
                   "pattern":{"String":"a"},"content":")" +
                  std::string(repetitions, 'a') + R"("}]})";
    std::ofstream file(path);
    file << R"({"added_tokens":[{"id":3,"content":"#","special":true,
                "single_word":false,"lstrip":false,"rstrip":false,
                "normalized":false}],"decoder":)"
         << decoder << R"(,"model":{"type":"BPE","vocab":{")" << spelling_
         << R"(":0,"b":1,"x":2,"#":3},"merges":[]}})";
    file.close();
    ASSERT_TRUE(file.good());
    auto options = ModelOptions();
    options.tokenizer_path = iree_make_cstring_view(path.c_str());
    IREE_ASSERT_OK(loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &model_));
    for (const char token : Oracle::Complete("b", 3)) {
      expected_ += token == 'a'
                       ? std::string(spelling_.size() * repetitions, 'a')
                       : std::string(1, token);
    }
  }

  void TearDown() override {
    const auto consumer = session_.request.consumer;
    loom_serve_text_session_deinitialize(&session_);
    if (consumer.self) {
      consumer.vtable->end(consumer.self, false, IREE_STATUS_CANCELLED);
    }
    if (stream_) {
      loom_serve_text_stream_close(stream_);
    }
    loom_serve_text_chat_scratch_deinitialize(&completion_scratch_);
    loom_serve_text_tool_calls_deinitialize(&calls_);
    iree_string_builder_deinitialize(&scratch_);
    if (service_) {
      AdvanceUntil(
          interface_, [&] { return Idle(); },
          LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
      interface_.runtime.vtable->set_observer(interface_.runtime.self, {});
      loom_serve_text_service_free(service_, IREE_STATUS_OK);
    }
    SourceTextServiceTest::TearDown();
  }

  bool Idle() const {
    return !interface_.runtime.vtable->is_pending(interface_.runtime.self) &&
           !interface_.runtime.vtable->is_active(interface_.runtime.self) &&
           !interface_.snapshot(interface_.runtime.self).queued_requests;
  }

  std::string Body(bool streaming) const {
    return std::string(R"({"model":"fixture-text","stream":)") +
           (streaming ? "true" : "false") +
           R"(,"max_tokens":3,"messages":[{"role":"user","content":"b"}]})";
  }

  // Real model source and sampling use the fixture's independent ID oracle.
  std::string spelling_;
  // Expected bytes after decoding those IDs through the declared vocabulary.
  std::string expected_;
  // Native output tests serialize this service exactly like the HTTP router.
  loom_serve_text_service_t* service_ = nullptr;
  // Borrowed callbacks valid through service retirement.
  loom_serve_service_t interface_ = {};
  // Direct translation coverage keeps the real model row, policy, and receipt.
  loom_serve_text_session_t session_ = {};
  // Independent caller receipt, retained after failed output translation.
  loom_serve_text_stream_t* stream_ = nullptr;
  // Reusable finite-loan storage, as owned by the service output stage.
  iree_string_builder_t scratch_ = {};
  // Validated calls and final source scratch share that same loan.
  loom_serve_text_tool_calls_t calls_ = {};
  // Completion scratch outlives each source invocation but not the test.
  loom_serve_text_chat_scratch_t completion_scratch_ = {};
  // Fails only growth of the session's already-initialized response builder.
  iree::vm::testing::CountingAllocator response_allocator_{1};
  // Tracks prepared input ownership through terminal output failure.
  iree::vm::testing::CountingAllocator input_allocator_;
};

TEST_P(TextResponseServiceTest, NativeOutputPreservesEveryDecodedByte) {
  const auto options = ServiceOptions();
  IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                  host_allocator_, &service_));
  interface_ = loom_serve_text_service_interface(service_);
  Observe(interface_);
  for (const bool streaming : {false, true}) {
    SCOPED_TRACE(streaming);
    loom_serve_text_request_t request = {};
    iree_string_builder_initialize(host_allocator_, &request.body);
    IREE_ASSERT_OK(iree_string_builder_append_cstring(&request.body,
                                                      Body(streaming).c_str()));
    loom_serve_text_stream_t* value = nullptr;
    auto status = loom_serve_text_service_try_open(service_, &request, &value);
    iree_string_builder_deinitialize(&request.body);
    IREE_ASSERT_OK(status);
    const std::unique_ptr<loom_serve_text_stream_t,
                          decltype(&loom_serve_text_stream_close)>
        stream(value, loom_serve_text_stream_close);
    ASSERT_NE(stream, nullptr);
    std::string spans;
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
      if (const auto* block = loom_serve_text_stream_peek(stream.get())) {
        for (iree_host_size_t i = 0;
             i < loom_serve_text_stream_block_count(block); ++i) {
          const auto span = loom_serve_text_stream_block_at(block, i);
          EXPECT_EQ(span.kind, LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT);
          spans.append(span.text.data, span.text.size);
        }
        loom_serve_text_stream_consume(stream.get());
      }
      return loom_serve_text_stream_is_complete(stream.get());
    }));
    IREE_ASSERT_OK(loom_serve_text_stream_status(stream.get()));
    const auto* result = loom_serve_text_stream_result(stream.get());
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(std::string(result->content.data, result->content.size),
              expected_);
    EXPECT_EQ(spans, streaming ? expected_ : "");
    EXPECT_EQ(result->response.usage.output_tokens, 3u);
    EXPECT_TRUE(
        iree_string_view_equal(result->finish_reason, IREE_SV("length")));
  }
}

TEST_P(TextResponseServiceTest, HttpOutputPreservesEveryDecodedByte) {
  ASSERT_NO_FATAL_FAILURE(Start());
  for (const bool streaming : {false, true}) {
    SCOPED_TRACE(streaming);
    const auto response = Read(Send(Body(streaming)));
    ASSERT_EQ(response.find("HTTP/1.1 200"), 0u) << response.substr(0, 1024);
    if (streaming) {
      EXPECT_EQ(Content(response), expected_);
    } else {
      EXPECT_NE(response.find("\"content\":\"" + expected_ + "\""),
                std::string::npos);
    }
    EXPECT_NE(response.find(R"("finish_reason":"length")"), std::string::npos);
  }
}

TEST_P(TextResponseServiceTest, GrowthFailureCannotPublishPartialSuccess) {
  const auto* tokenizer = loom_serve_text_model_tokenizer(model_);
  IREE_ASSERT_OK(loom_serve_text_session_initialize(
      LOOM_SERVE_TEXT_MODE_GENERATE, tokenizer, host_allocator_, &session_));
  session_.row = loom_serve_text_model_row(model_, 0);
  session_.lifetime.serial = 1;
  iree_string_builder_initialize(host_allocator_, &scratch_);
  loom_serve_text_tool_calls_initialize(host_allocator_, &calls_);
  loom_serve_text_chat_scratch_initialize(host_allocator_,
                                          &completion_scratch_);
  loom_serve_text_request_t request = {};
  iree_string_builder_initialize(host_allocator_, &request.body);
  IREE_ASSERT_OK(iree_string_builder_append_cstring(
      &request.body, Request(Message("user", "b"), 9).c_str()));
  auto status = loom_serve_text_stream_allocate(
      IREE_SV("fixture-text"), &request, host_allocator_, &stream_,
      &session_.request.consumer);
  iree_string_builder_deinitialize(&request.body);
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      loom_serve_text_model_chat_policy(model_), IREE_SV("fixture-text"),
      iree_string_builder_view(&loom_serve_text_stream_request(stream_)->body),
      1, 9, host_allocator_, &session_.request.chat));
  loom_serve_text_input_t input = {};
  IREE_ASSERT_OK(loom_serve_text_input_prepare(
      &session_.request.chat, {}, false,
      loom_serve_text_model_context_capacity(model_), &input,
      input_allocator_.allocator()));
  session_.request.input.storage = input.token_storage;
  session_.request.input.count = input.count;
  input.token_storage = nullptr;
  loom_serve_text_input_deinitialize(&input, input_allocator_.allocator());
  EXPECT_EQ(input_allocator_.free_count() + 1,
            input_allocator_.allocation_count());
  IREE_ASSERT_OK(loom_serve_text_session_begin(&session_, tokenizer,
                                               IREE_SV("fixture-text")));
  // Only the output storage allocator fails. The source policy and consumer
  // remain usable to translate and publish the actual terminal failure.
  session_.response.allocator = response_allocator_.allocator();
  const int32_t tokens[9] = {};
  IREE_ASSERT_OK(loom_serve_text_session_append_tokens(
      &session_, IREE_SV("fixture-text"), IREE_ARRAYSIZE(tokens), tokens,
      &scratch_, &calls_, &completion_scratch_, host_allocator_));
  EXPECT_EQ(response_allocator_.allocation_count(), 1u);
  EXPECT_GT(iree_string_builder_size(&session_.response), 0u);
  EXPECT_FALSE(session_.completion.reusable);
  EXPECT_FALSE(session_.request.publication.pending);
  EXPECT_STREQ(session_.request.finish_reason, "error");
  bool ready = false, progress = false;
  const auto consumer = session_.request.consumer;
  IREE_ASSERT_OK(consumer.vtable->publish(consumer.self, &ready, &progress));
  loom_serve_text_session_release_request(&session_);
  EXPECT_EQ(session_.request.input.storage, nullptr);
  EXPECT_EQ(input_allocator_.free_count(), input_allocator_.allocation_count());
  consumer.vtable->end(consumer.self, true, IREE_STATUS_OK);
  const auto* block = loom_serve_text_stream_peek(stream_);
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(loom_serve_text_stream_block_count(block), 0u);
  loom_serve_text_stream_consume(stream_);
  EXPECT_TRUE(loom_serve_text_stream_is_complete(stream_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_text_stream_status(stream_));
  EXPECT_EQ(loom_serve_text_stream_result(stream_), nullptr);
}

INSTANTIATE_TEST_SUITE_P(DecodedBytes, TextResponseServiceTest,
                         ::testing::Values(8193, 65535));

}  // namespace
