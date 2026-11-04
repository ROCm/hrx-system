// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/test_allocator.h"
#include "loom/serve/media/image_codec.h"
#include "loom/serve/service/text/chat.h"
#include "loom/serve/service/text/input.h"
#include "loom/serve/testing/text_policy.h"

namespace {

class SourceChatPolicyTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const char* control = std::getenv("LOOM_SERVE_TEST_CHAT_CONTROL");
    const char* tokenizer = std::getenv("LOOM_SERVE_TEST_CHAT_TOKENIZER");
    ASSERT_NE(control, nullptr);
    ASSERT_NE(tokenizer, nullptr);
    const loom_serve_sources_t sources = {
        .base_directory = IREE_SV("."),
        .control = iree_make_cstring_view(control),
    };
    loom_serve_compilation_report_t report = {};
    auto status = loom_serve_test_text_policy_allocate(
        &sources, iree_make_cstring_view(tokenizer), IREE_SV("{}"), &report,
        iree_allocator_system(), &policy_owner_);
    loom_output_stream_t diagnostics;
    loom_output_stream_for_file(stderr, &diagnostics);
    IREE_EXPECT_OK(loom_serve_compilation_write(&report, &diagnostics));
    IREE_ASSERT_OK(status);
  }

  static void TearDownTestSuite() {
    loom_serve_test_text_policy_free(policy_owner_);
    policy_owner_ = nullptr;
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(64 * 1024, host_allocator_, &pool_);
  }

  void TearDown() override { iree_arena_block_pool_deinitialize(&pool_); }

  static std::string Message(const std::string& role,
                             const std::string& content) {
    return "{\"role\":\"" + role + "\",\"content\":\"" + content + "\"}";
  }

  static std::string Request(const std::string& messages, size_t limit) {
    return "{\"model\":\"fixture-text\",\"stream\":true,\"stream_options\":{"
           "\"include_usage\":true},\"max_tokens\":" +
           std::to_string(limit) + ",\"messages\":[" + messages + "]}";
  }

  static std::string ImageMessage(const std::string& before,
                                  const std::string& mime_type,
                                  const std::string& base64,
                                  const std::string& detail,
                                  const std::string& after) {
    return "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"" +
           before +
           "\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:" +
           mime_type + ";base64," + base64 + "\",\"detail\":\"" + detail +
           "\"}},{\"type\":\"text\",\"text\":\"" + after + "\"}]}";
  }

  static std::string VideoMessage(const std::string& frames) {
    return "{\"role\":\"user\",\"content\":[{\"type\":\"video_url\","
           "\"video_url\":{\"url\":\"data:video/jpeg;base64," +
           frames + "\"}}]}";
  }

  static std::string VideoRequest(const std::string& messages,
                                  const std::string& options) {
    return "{\"model\":\"fixture-text\",\"messages\":[" + messages +
           "],\"media_io_kwargs\":{\"video\":" + options + "}}";
  }

  // A real 2x2 RGB JPEG (solid red); video/jpeg frames use the JPEG decoder.
  static constexpr char kJpeg[] =
      "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYI"
      "DAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkF"
      "BQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU"
      "FBQUFBT/wAARCAACAAIDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAj/xAAU"
      "EAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAABwn/xAAUEQEAAAAA"
      "AAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdAAYqm//Z";

  // Real encoded image inputs observed by the source policy's media plan.
  static constexpr char kPam[] =
      "UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCAzCk1BWFZBTCAyNTUKVFVQTFRZUEUg"
      "UkdCCkVOREhEUgr/AAAA/wAAAP////8=";
  // Same geometry with distinct pixels and alpha metadata.
  static constexpr char kAlphaPam[] =
      "UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCA0Ck1BWFZBTCAyNTUKVFVQTFRZUEUg"
      "UkdCX0FMUEhBCkVOREhEUgoKFB4AKDI8gEZQWv9kbnjI";

  // One source program/tokenizer, shared across serialized policy callers.
  inline static loom_serve_test_text_policy_t* policy_owner_ = nullptr;
  // Ordinary owner allocator; individual tests inject scratch allocation
  // errors.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // Request batches borrow/reset arenas against this shared block owner.
  iree_arena_block_pool_t pool_;
};

TEST_F(SourceChatPolicyTest,
       ColdSourceDeclaresReasoningWithoutRetainedStorage) {
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  EXPECT_EQ(policy->reasoning.support, LOOM_SERVE_REASONING_UNSUPPORTED);
  EXPECT_EQ(policy->reasoning.efforts, 0);
  EXPECT_EQ(policy->reasoning.default_effort, 0);
}

TEST_F(SourceChatPolicyTest, OutputBudgetsUseSignedTokenCoordinates) {
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  for (const char* field : {"max_tokens", "max_completion_tokens"}) {
    SCOPED_TRACE(field);
    const std::string prefix =
        R"({"model":"fixture-text","messages":[{"role":"user","content":"a"}],")" +
        std::string(field) + "\":";
    for (uint32_t capacity : {16385u, 1048576u, uint32_t{INT32_MAX}}) {
      const std::string body = prefix + std::to_string(capacity) + "}";
      loom_serve_text_chat_t chat = {};
      IREE_ASSERT_OK(loom_serve_text_chat_initialize(
          policy, policy->name, iree_make_cstring_view(body.c_str()), 1, 32,
          host_allocator_, &chat));
      EXPECT_EQ(chat.max_tokens, capacity);
      loom_serve_text_chat_deinitialize(&chat);
    }
    for (const char* value : {"0", "-1", "2147483648", "4294967296", "1.5"}) {
      const std::string body = prefix + value + "}";
      loom_serve_text_chat_t chat = {};
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          loom_serve_text_chat_initialize(policy, policy->name,
                                          iree_make_cstring_view(body.c_str()),
                                          1, 32, host_allocator_, &chat));
    }
  }
}

TEST_F(SourceChatPolicyTest, StreamingBatchesReuseScratchWithoutHeapChurn) {
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  const std::string body = Request(Message("user", "a"), 32);
  iree::vm::testing::CountingAllocator allocator;
  loom_serve_text_chat_t chats[8] = {};
  for (auto& chat : chats) {
    IREE_ASSERT_OK(loom_serve_text_chat_initialize(
        policy, policy->name, iree_make_cstring_view(body.c_str()), 1, 32,
        allocator.allocator(), &chat));
    IREE_ASSERT_OK(iree_string_builder_reserve(&chat.output.content, 128));
  }
  const auto allocation_count = allocator.allocation_count();
  const auto free_count = allocator.free_count();
  const std::string response(32, 'b');
  for (size_t length = 1; length <= response.size(); ++length) {
    // Like the service, one batch owns an arena shared by all its rows. The
    // next batch may run on a different worker and borrow the returned blocks.
    iree_arena_allocator_t arena;
    iree_arena_initialize(&pool_, &arena);
    for (auto& chat : chats) {
      loom_serve_text_chat_output_t output;
      IREE_ASSERT_OK(loom_serve_text_chat_output_next(
          &chat, iree_make_string_view(response.data(), length),
          LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING, iree_arena_allocator(&arena),
          &output));
      EXPECT_TRUE(output.advanced);
      EXPECT_EQ(output.kind, LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT);
      EXPECT_TRUE(iree_string_view_equal(output.text, IREE_SV("b")));
    }
    iree_arena_deinitialize(&arena);
  }
  EXPECT_EQ(allocator.allocation_count(), allocation_count);
  EXPECT_EQ(allocator.free_count(), free_count);
  for (auto& chat : chats) {
    EXPECT_TRUE(iree_string_view_equal(
        iree_string_builder_view(&chat.output.content),
        iree_make_string_view(response.data(), response.size())));
    loom_serve_text_chat_deinitialize(&chat);
  }
}

TEST_F(SourceChatPolicyTest, CompletionAliasSurvivesBatchAndRequestRetirement) {
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  const std::string body = Request(Message("user", "abx"), 4);
  loom_serve_text_chat_t chat = {};
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(body.c_str()), 1, 4,
      host_allocator_, &chat));
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool_, &arena);
  loom_serve_text_tool_calls_t tools;
  loom_serve_text_tool_calls_initialize(host_allocator_, &tools);
  loom_serve_text_chat_scratch_t scratch;
  loom_serve_text_chat_scratch_initialize(host_allocator_, &scratch);
  loom_serve_text_chat_completion_t completion = {};
  // The fixture returns the prompt itself for an empty response. That source
  // alias must keep owned storage, not retain a temporary argument wrapper.
  IREE_ASSERT_OK(loom_serve_text_chat_complete(
      &chat, iree_string_view_empty(), false, 1, &tools, &scratch,
      iree_arena_allocator(&arena), &completion));
  loom_serve_text_chat_deinitialize(&chat);
  iree_arena_reset(&arena);
  void* overwritten = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, 4096, &overwritten));
  std::memset(overwritten, 0x5a, 4096);
  EXPECT_TRUE(iree_string_view_equal(completion.transcript, IREE_SV("abx")));
  EXPECT_TRUE(completion.reusable);
  loom_serve_text_chat_completion_deinitialize(&completion);
  loom_serve_text_chat_scratch_deinitialize(&scratch);
  loom_serve_text_tool_calls_deinitialize(&tools);
  iree_arena_deinitialize(&arena);
}

TEST_F(SourceChatPolicyTest, OutputScratchFailureLeavesParserUnadvanced) {
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  const std::string body = Request(Message("user", "a"), 4);
  loom_serve_text_chat_t chat = {};
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(body.c_str()), 1, 4,
      host_allocator_, &chat));
  iree::vm::testing::CountingAllocator allocator(1);
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(64 * 1024, allocator.allocator(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  loom_serve_text_chat_output_t output;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_text_chat_output_next(&chat, IREE_SV("b"),
                                       LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING,
                                       iree_arena_allocator(&arena), &output));
  EXPECT_EQ(chat.output.cursor, 0u);
  EXPECT_EQ(iree_string_builder_size(&chat.output.content), 0u);
  EXPECT_FALSE(output.advanced);
  iree_arena_reset(&arena);
  IREE_ASSERT_OK(loom_serve_text_chat_output_next(
      &chat, IREE_SV("b"), LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING,
      iree_arena_allocator(&arena), &output));
  EXPECT_TRUE(output.advanced);
  EXPECT_TRUE(iree_string_view_equal(output.text, IREE_SV("b")));
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
  loom_serve_text_chat_deinitialize(&chat);
}

TEST_F(SourceChatPolicyTest, CompletionFormattingReusesCapacityAcrossRows) {
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  const std::string body = Request(Message("user", "a"), 4);
  iree::vm::testing::CountingAllocator allocator;
  loom_serve_text_chat_scratch_t scratch;
  loom_serve_text_chat_scratch_initialize(allocator.allocator(), &scratch);
  loom_serve_text_tool_calls_t tools;
  loom_serve_text_tool_calls_initialize(allocator.allocator(), &tools);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool_, &arena);
  iree_host_size_t warm_allocations = 0;
  for (size_t row = 0; row < 8; ++row) {
    loom_serve_text_chat_t chat = {};
    IREE_ASSERT_OK(loom_serve_text_chat_initialize(
        policy, policy->name, iree_make_cstring_view(body.c_str()), row + 1, 4,
        host_allocator_, &chat));
    // Prior-row formatting must not leak into the next row, even when its
    // source parser returns no calls and never touches these builders itself.
    IREE_ASSERT_OK(iree_string_builder_append_cstring(&scratch.arguments,
                                                      "prior arguments"));
    IREE_ASSERT_OK(iree_string_builder_append_cstring(&scratch.rendered,
                                                      "prior tool calls"));
    IREE_ASSERT_OK(loom_serve_text_tool_calls_append(
        &tools, row + 1, IREE_SV("prior"), IREE_SV("{}")));
    loom_serve_text_chat_completion_t completion = {};
    IREE_ASSERT_OK(loom_serve_text_chat_complete(
        &chat, IREE_SV("b"), false, row + 1, &tools, &scratch,
        iree_arena_allocator(&arena), &completion));
    if (row == 0) {
      warm_allocations = allocator.allocation_count();
    }
    EXPECT_EQ(allocator.allocation_count(), warm_allocations);
    EXPECT_EQ(allocator.free_count(), 0u);
    EXPECT_EQ(tools.count, 0u);
    EXPECT_TRUE(iree_string_view_equal(
        iree_string_builder_view(&scratch.parsed), IREE_SV("[]")));
    EXPECT_EQ(iree_string_builder_size(&scratch.rendered), 0u);
    EXPECT_EQ(iree_string_builder_size(&scratch.arguments), 0u);
    loom_serve_text_chat_deinitialize(&chat);
    iree_arena_reset(&arena);
    EXPECT_TRUE(iree_string_view_equal(completion.transcript, IREE_SV("ab")));
    loom_serve_text_chat_completion_deinitialize(&completion);
  }
  loom_serve_text_chat_scratch_deinitialize(&scratch);
  loom_serve_text_tool_calls_deinitialize(&tools);
  iree_arena_deinitialize(&arena);
}

TEST_F(SourceChatPolicyTest,
       PreparedInputSurvivesScratchReuseAndHistoryEviction) {
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  const std::string first_body = Request(Message("user", "a"), 4);
  const std::string next_body =
      Request(Message("user", "a") + "," + Message("assistant", "b") + "," +
                  Message("user", "x"),
              4);
  loom_serve_text_chat_t first = {}, next = {};
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(first_body.c_str()), 1, 4,
      host_allocator_, &first));
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(next_body.c_str()), 2, 4,
      host_allocator_, &next));
  loom_serve_text_tool_calls_t tools;
  loom_serve_text_tool_calls_initialize(host_allocator_, &tools);
  loom_serve_text_chat_scratch_t completion_scratch;
  loom_serve_text_chat_scratch_initialize(host_allocator_, &completion_scratch);
  loom_serve_text_chat_completion_t completion = {};
  IREE_ASSERT_OK(loom_serve_text_chat_complete(&first, IREE_SV("b"), false, 1,
                                               &tools, &completion_scratch,
                                               host_allocator_, &completion));
  const loom_serve_text_input_prefix_t prefix = {
      .completion = &completion,
      .position = 1,
      .token = 1,
      .boundary = LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN,
  };
  loom_serve_text_input_t input = {};
  int32_t scratch[96];
  IREE_ASSERT_OK(loom_serve_text_input_prepare(&next, prefix, true,
                                               IREE_ARRAYSIZE(scratch), scratch,
                                               &input, host_allocator_));
  EXPECT_EQ(input.retained_count, 1u);
  EXPECT_EQ(std::vector<int32_t>(input.tokens, input.tokens + input.count),
            (std::vector<int32_t>{1, 3, 2}));
  const int32_t* prepared = input.tokens;
  std::fill(std::begin(scratch), std::end(scratch), -7);
  IREE_ASSERT_OK(loom_serve_text_input_prepare(&next, prefix, true,
                                               IREE_ARRAYSIZE(scratch), scratch,
                                               &input, host_allocator_));
  EXPECT_EQ(input.tokens, prepared);
  EXPECT_TRUE(std::all_of(std::begin(scratch), std::end(scratch),
                          [](int32_t value) { return value == -7; }));
  // The same full prompt exceeds this smaller context with or without reuse.
  // Diagnostics count retained input too, not merely the newly encoded suffix.
  for (const auto limited_prefix : {prefix, loom_serve_text_input_prefix_t{}}) {
    loom_serve_text_input_t limited = {};
    auto failure = loom_serve_text_input_prepare(
        &next, limited_prefix, false, 6, scratch, &limited, host_allocator_);
    EXPECT_EQ(iree_status_code(failure), IREE_STATUS_OUT_OF_RANGE);
    const auto message = iree_status_message(failure);
    EXPECT_EQ(std::string(message.data, message.size),
              "request exceeds the available context size of 6 tokens: "
              "4 input tokens and max_tokens=4 require 7 stored tokens; "
              "compact history or lower max_tokens");
    iree_status_free(failure);
    EXPECT_EQ(limited.tokens, nullptr);
    EXPECT_EQ(limited.history, nullptr);
    loom_serve_text_input_deinitialize(&limited, host_allocator_);
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_input_prepare(&next, {}, true, IREE_ARRAYSIZE(scratch),
                                    scratch, &input, host_allocator_));
  EXPECT_EQ(input.tokens, prepared);
  completion.reusable = false;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_input_prepare(
                            &next, prefix, true, IREE_ARRAYSIZE(scratch),
                            scratch, &input, host_allocator_));
  EXPECT_FALSE(loom_serve_text_input_is_current(&input, &next, prefix));
  completion.reusable = true;
  loom_serve_text_chat_completion_deinitialize(&completion);
  // Eviction releases native history independently. The prepared input's host
  // identity remains valid, but its suffix cannot be replayed as a fresh row.
  EXPECT_NE(input.history, nullptr);
  IREE_ASSERT_OK(loom_serve_text_input_prepare(&next, {}, false,
                                               IREE_ARRAYSIZE(scratch), scratch,
                                               &input, host_allocator_));
  EXPECT_EQ(input.retained_count, 0u);
  EXPECT_EQ(input.history, nullptr);
  EXPECT_EQ(std::vector<int32_t>(input.tokens, input.tokens + input.count),
            (std::vector<int32_t>{0, 1, 3, 2}));
  loom_serve_text_input_deinitialize(&input, host_allocator_);
  loom_serve_text_chat_deinitialize(&next);
  loom_serve_text_chat_deinitialize(&first);
  loom_serve_text_tool_calls_deinitialize(&tools);
  loom_serve_text_chat_scratch_deinitialize(&completion_scratch);
}

TEST_F(SourceChatPolicyTest, OrderedImagesReachSourceDefinedInputSpans) {
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  const std::string body = Request(
      ImageMessage("a", "image/x-portable-arbitrarymap", kPam, "high", "b") +
          "," +
          ImageMessage("", "image/x-portable-arbitrarymap", kPam, "low", ""),
      4);
  loom_serve_text_chat_t chat = {};
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNIMPLEMENTED,
        loom_serve_text_chat_initialize(policy, policy->name,
                                        iree_make_cstring_view(body.c_str()),
                                        10, 4, host_allocator_, &chat));
    EXPECT_EQ(chat.images.count, 0u);
    return;
  }
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(body.c_str()), 10, 4,
      host_allocator_, &chat));
  EXPECT_TRUE(iree_string_view_equal(iree_string_builder_view(&chat.prompt),
                                     IREE_SV("axbx")));
  loom_serve_text_input_t input = {};
  int32_t scratch[96];
  IREE_ASSERT_OK(loom_serve_text_input_prepare(&chat, {}, false,
                                               IREE_ARRAYSIZE(scratch), scratch,
                                               &input, host_allocator_));
  EXPECT_EQ(std::vector<int32_t>(input.tokens, input.tokens + input.count),
            (std::vector<int32_t>{0, 2, 1, 2}));
  ASSERT_EQ(input.media_span_count, 2u);
  EXPECT_EQ(input.media_spans[0].projection_index, 0u);
  EXPECT_EQ(input.media_spans[0].input_offset, 1u);
  EXPECT_EQ(input.media_spans[0].input_count, 1u);
  EXPECT_EQ(input.media_spans[1].projection_index, 1u);
  EXPECT_EQ(input.media_spans[1].input_offset, 3u);
  EXPECT_EQ(input.media_spans[1].input_count, 1u);
  loom_serve_text_input_deinitialize(&input, host_allocator_);
  loom_serve_text_chat_deinitialize(&chat);
}

TEST_F(SourceChatPolicyTest, TimestampedFramesReachSourceAsOneOrderedPart) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG)) {
    GTEST_SKIP() << "image input is disabled in this build";
  }
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  const std::string image =
      ImageMessage("", "image/x-portable-arbitrarymap", kPam, "high", "");
  const std::string video = VideoMessage(std::string(kJpeg) + "," + kJpeg);
  const std::string body = VideoRequest(
      image + "," + video,
      R"({"fps":8,"frames_indices":[1,14],"total_num_frames":16,"duration":2,"num_frames":-1,"do_sample_frames":false})");
  loom_serve_text_chat_t chat = {};
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(body.c_str()), 10, 4,
      host_allocator_, &chat));
  ASSERT_EQ(chat.images.count, 3u);
  ASSERT_EQ(chat.projections.count, 2u);
  EXPECT_EQ(chat.projections.values[1].image_begin, 1u);
  EXPECT_EQ(chat.projections.values[1].image_count, 2u);
  EXPECT_EQ(chat.images.values[1].kind, LOOM_SERVE_TEXT_CHAT_CONTENT_VIDEO);
  EXPECT_EQ(chat.images.values[2].timestamp, 1.75);
  const auto& projection = chat.projections.values[1];
  ASSERT_EQ(projection.metadata_length, 32u);
  const char* metadata =
      iree_string_builder_buffer(&chat.projections.metadata) +
      projection.metadata_offset;
  EXPECT_EQ(iree_unaligned_load_le_u32(metadata), 2u);
  EXPECT_EQ(iree_unaligned_load_le_u32(metadata + 4), 2u);
  EXPECT_DOUBLE_EQ(iree_unaligned_load_le_f64(metadata + 8), 0.125);
  EXPECT_DOUBLE_EQ(iree_unaligned_load_le_f64(metadata + 24), 1.75);
  loom_serve_media_image_input_t decoded = {};
  IREE_ASSERT_OK(loom_serve_media_image_input_initialize(
      chat.images.values[1].source, &policy->images.limits, host_allocator_,
      &decoded));
  EXPECT_EQ(decoded.info.format, LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG);
  EXPECT_EQ(decoded.info.width, 2u);
  std::vector<uint8_t> pixels(decoded.info.pixel_length);
  const auto destination = iree_make_byte_span(pixels.data(), pixels.size());
  const loom_serve_media_image_input_t* inputs[] = {&decoded};
  iree_status_t result = iree_ok_status();
  IREE_ASSERT_OK(loom_serve_media_image_decode_batch(1, inputs, &destination,
                                                     &result, host_allocator_));
  IREE_ASSERT_OK(result);
  EXPECT_GE(pixels[0], 250u);
  EXPECT_LE(pixels[1], 5u);
  EXPECT_LE(pixels[2], 5u);
  loom_serve_media_image_input_deinitialize(&decoded);
  loom_serve_text_input_t input = {};
  int32_t scratch[16];
  IREE_ASSERT_OK(loom_serve_text_input_prepare(&chat, {}, false,
                                               IREE_ARRAYSIZE(scratch), scratch,
                                               &input, host_allocator_));
  ASSERT_EQ(input.media_span_count, 2u);
  EXPECT_EQ(input.media_spans[1].projection_index, 1u);
  EXPECT_EQ(input.media_spans[1].input_offset, 1u);
  EXPECT_EQ(input.media_spans[1].input_count, 2u);
  loom_serve_text_input_deinitialize(&input, host_allocator_);

  loom_serve_text_chat_media_identity_t identity = {};
  IREE_ASSERT_OK(loom_serve_text_chat_media_identity_initialize(
      &chat, host_allocator_, &identity));
  loom_serve_text_chat_deinitialize(&chat);
  // Timestamps can change while encoded pixels, text, and model rows stay
  // equal.
  const std::string changed =
      VideoRequest(image + "," + video, R"({"fps":8,"frames_indices":[1,1]})");
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(changed.c_str()), 11, 4,
      host_allocator_, &chat));
  EXPECT_FALSE(
      loom_serve_text_chat_media_identity_equal_chat(&identity, &chat));
  loom_serve_text_chat_media_identity_deinitialize(&identity);
  loom_serve_text_chat_deinitialize(&chat);
}

TEST_F(SourceChatPolicyTest, JpegSequencesUseDefaultTimingAndJsonEscapes) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG)) {
    GTEST_SKIP() << "JPEG input is disabled in this build";
  }
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  std::string video = VideoMessage(std::string(kJpeg) + "," + kJpeg);
  // Escaped commas delimit frames just like literal commas. Escaped slashes
  // remain inside each borrowed base64 slice until image transport decoding.
  for (size_t i = video.find("data:"); i < video.size(); ++i) {
    if (video[i] == ',') {
      video.replace(i, 1, "\\u002C");
      i += 5;
    } else if (video[i] == '/') {
      video.insert(i, 1, '\\');
      ++i;
    }
  }
  const std::string body = Request(video, 4);
  loom_serve_text_chat_t chat = {};
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(body.c_str()), 10, 4,
      host_allocator_, &chat));
  ASSERT_EQ(chat.images.count, 2u);
  EXPECT_DOUBLE_EQ(chat.images.values[0].timestamp, 0.0);
  EXPECT_DOUBLE_EQ(chat.images.values[1].timestamp, 1.0);
  loom_serve_text_chat_deinitialize(&chat);
}

TEST_F(SourceChatPolicyTest, RejectsInvalidVideoTransportAndTiming) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG)) {
    GTEST_SKIP() << "JPEG input is disabled in this build";
  }
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  const std::string video = VideoMessage(std::string(kJpeg) + "," + kJpeg);
  const std::vector<std::string> invalid_options = {
      R"({"fps":0})",
      R"({"fps":-1})",
      R"({"fps":1e400})",
      R"({"fps":1e-320})",
      R"({"frames_indices":[]})",
      R"({"frames_indices":[0]})",
      R"({"frames_indices":[0,1,2]})",
      R"({"frames_indices":[0,-1]})",
      R"({"frames_indices":[1,0]})",
      R"({"frames_indices":[0,1.5]})",
      R"({"frames_indices":[0,"1"]})",
      R"({"total_num_frames":0})",
      R"({"total_num_frames":1})",
      R"({"duration":-1})",
      R"({"duration":1e400})",
      R"({"num_frames":1})",
      R"({"do_sample_frames":true})"};
  std::vector<std::string> invalid;
  for (const auto& options : invalid_options) {
    invalid.push_back(VideoRequest(video, options));
  }
  for (const auto& frames : {std::string(), std::string(kPam),
                             std::string(kJpeg) + ",", "," + std::string(kJpeg),
                             std::string(kJpeg) + "," + kJpeg + "," + kJpeg +
                                 "," + kJpeg + "," + kJpeg}) {
    invalid.push_back(Request(VideoMessage(frames), 4));
  }
  invalid.push_back(Request(
      R"({"role":"user","content":[{"type":"video_url","video_url":{"url":"https://example.com/clip.mp4"}}]})",
      4));
  for (const auto& body : invalid) {
    SCOPED_TRACE(body);
    loom_serve_text_chat_t chat = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_text_chat_initialize(policy, policy->name,
                                        iree_make_cstring_view(body.c_str()),
                                        10, 4, host_allocator_, &chat));
    EXPECT_EQ(chat.images.count, 0u);
    EXPECT_EQ(chat.projections.count, 0u);
    loom_serve_text_chat_deinitialize(&chat);
  }
}

TEST_F(SourceChatPolicyTest, ExplicitAllFramesDoesNotTruncateAtDefaultLimit) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_JPEG)) {
    GTEST_SKIP() << "JPEG input is disabled in this build";
  }
  auto policy = *loom_serve_test_text_policy_chat(policy_owner_);
  policy.images.maximum_count = 64;
  std::string frames = kJpeg;
  for (size_t i = 1; i < 32; ++i) {
    frames += "," + std::string(kJpeg);
  }
  std::string body = Request(VideoMessage(frames), 4);
  loom_serve_text_chat_t chat = {};
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      &policy, policy.name, iree_make_cstring_view(body.c_str()), 10, 4,
      host_allocator_, &chat));
  EXPECT_EQ(chat.images.count, 32u);
  loom_serve_text_chat_deinitialize(&chat);

  frames += "," + std::string(kJpeg);
  body = Request(VideoMessage(frames), 4);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_chat_initialize(&policy, policy.name,
                                      iree_make_cstring_view(body.c_str()), 10,
                                      4, host_allocator_, &chat));
  EXPECT_EQ(chat.images.count, 0u);
  body = VideoRequest(VideoMessage(frames), R"({"num_frames":-1})");
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      &policy, policy.name, iree_make_cstring_view(body.c_str()), 10, 4,
      host_allocator_, &chat));
  EXPECT_EQ(chat.images.count, 33u);
  EXPECT_DOUBLE_EQ(chat.images.values[32].timestamp, 32.0);
  loom_serve_text_chat_deinitialize(&chat);
}

TEST_F(SourceChatPolicyTest, RetainedHistoryRequiresExactOrderedMedia) {
  const auto* policy = loom_serve_test_text_policy_chat(policy_owner_);
  const std::string first_body = Request(
      ImageMessage("x", "image/x-portable-arbitrarymap", kPam, "low", ""), 4);
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
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNIMPLEMENTED,
        loom_serve_text_chat_initialize(
            policy, policy->name, iree_make_cstring_view(first_body.c_str()),
            10, 4, host_allocator_, &first));
    return;
  }
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(first_body.c_str()), 10, 4,
      host_allocator_, &first));
  const std::string matching_body =
      continuation("x", "image/x-portable-arbitrarymap", kPam, "low", "");
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(matching_body.c_str()), 11,
      4, host_allocator_, &matching));
  const std::string different_image_body =
      continuation("x", "image/x-portable-arbitrarymap", kAlphaPam, "low", "");
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name,
      iree_make_cstring_view(different_image_body.c_str()), 12, 4,
      host_allocator_, &different_image));
  const std::string different_detail_body =
      continuation("x", "image/x-portable-arbitrarymap", kPam, "high", "");
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name,
      iree_make_cstring_view(different_detail_body.c_str()), 13, 4,
      host_allocator_, &different_detail));
  const std::string different_placement_body =
      continuation("", "image/x-portable-arbitrarymap", kPam, "low", "x");
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name,
      iree_make_cstring_view(different_placement_body.c_str()), 14, 4,
      host_allocator_, &different_placement));

  loom_serve_text_tool_calls_t tools;
  loom_serve_text_tool_calls_initialize(host_allocator_, &tools);
  loom_serve_text_chat_scratch_t completion_scratch;
  loom_serve_text_chat_scratch_initialize(host_allocator_, &completion_scratch);
  loom_serve_text_chat_completion_t completion = {};
  IREE_ASSERT_OK(loom_serve_text_chat_complete(&first, IREE_SV("b"), false, 10,
                                               &tools, &completion_scratch,
                                               host_allocator_, &completion));
  loom_serve_text_chat_completion_t retained = {};
  loom_serve_text_chat_completion_retain(&completion, &retained);
  loom_serve_text_chat_completion_deinitialize(&completion);
  const loom_serve_text_input_prefix_t prefix = {
      .completion = &retained,
      .position = 1,
      .token = 1,
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
  loom_serve_text_tool_calls_deinitialize(&tools);
  loom_serve_text_chat_scratch_deinitialize(&completion_scratch);
  loom_serve_text_chat_deinitialize(&different_placement);
  loom_serve_text_chat_deinitialize(&different_detail);
  loom_serve_text_chat_deinitialize(&different_image);
  loom_serve_text_chat_deinitialize(&matching);
  loom_serve_text_chat_deinitialize(&first);
}

}  // namespace
