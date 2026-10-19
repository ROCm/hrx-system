// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cerrno>

#include "iree/base/internal/arena.h"
#include "iree/vm/test_allocator.h"
#include "loom/serve/media/image_codec.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::RegisteredSourceTextServiceTest;
using loom_serve::testing::text::SourceTextServiceTest;
using loom_serve::testing::text::StalledClient;

class UntaggedTextServiceTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.checkpoint_capacity = 0;
    return options;
  }
};

TEST_F(UntaggedTextServiceTest,
       InterleavedContinuationsReuseRowsWithoutCheckpointHeadroom) {
  Start();
  const std::vector<std::string> prompts = {"abxa", std::string(60, 'b')};
  std::vector<std::string> replies;
  for (const auto& prompt : prompts) {
    const auto response = Read(Send(Request(Message("user", prompt), 2)));
    replies.push_back(Content(response));
    EXPECT_EQ(replies.back(), Oracle::Complete(prompt, 2));
    EXPECT_NE(response.find(R"("cached_tokens":0})"), std::string::npos);
  }
  // Two never-used rows are older than either retained history. Continuations
  // must find their exact rows, without requiring a session header or a pin.
  for (size_t i : {1u, 0u}) {
    const auto response = Read(Send(Request(
        Message("user", prompts[i]) + "," + Message("assistant", replies[i]) +
            "," + Message("user", "b"),
        2)));
    EXPECT_EQ(Content(response),
              Oracle::Complete(prompts[i] + replies[i] + "#b", 2));
    EXPECT_NE(response.find("\"cached_tokens\":" +
                            std::to_string(prompts[i].size() + 1) + "}"),
              std::string::npos)
        << response;
  }
}

TEST_F(UntaggedTextServiceTest, NamedAndUntaggedHistoriesStayIndependent) {
  Start();
  const std::string named_prompt = "abxa";
  const auto named_reply = Content(Read(Send(
      Request(Message("user", named_prompt), 2), "X-Loom-Session: named\r\n")));
  const auto named_history =
      Message("user", named_prompt) + "," + Message("assistant", named_reply);
  const auto anonymous =
      Read(Send(Request(named_history + "," + Message("user", "b"), 2)));
  EXPECT_EQ(Content(anonymous),
            Oracle::Complete(named_prompt + named_reply + "#b", 2));
  EXPECT_NE(anonymous.find(R"("cached_tokens":0})"), std::string::npos);

  const std::string prompt = "bxxb";
  const auto reply = Content(Read(Send(Request(Message("user", prompt), 2))));
  const auto adopting = Read(
      Send(Request(Message("user", prompt) + "," + Message("assistant", reply) +
                       "," + Message("user", "b"),
                   2),
           "X-Loom-Session: new\r\n"));
  EXPECT_EQ(Content(adopting), Oracle::Complete(prompt + reply + "#b", 2));
  EXPECT_NE(adopting.find(R"("cached_tokens":0})"), std::string::npos);

  const auto continued =
      Read(Send(Request(named_history + "," + Message("user", "x"), 2),
                "X-Loom-Session: named\r\n"));
  EXPECT_EQ(Content(continued),
            Oracle::Complete(named_prompt + named_reply + "#x", 2));
  EXPECT_NE(continued.find(R"("cached_tokens":5})"), std::string::npos);
}

TEST_F(UntaggedTextServiceTest, ToolPermissionChangesPreserveRetainedHistory) {
  Start();
  const auto request = [](const std::string& messages, const char* choice) {
    std::string body = Request(messages, 2);
    body.pop_back();
    body +=
        R"(,"tools":[{"type":"function","function":{"name":"inspect","parameters":{"type":"object","properties":{}}}}],"tool_choice":")";
    body += choice;
    body += R"("})";
    return body;
  };
  std::string history = Message("user", "abxa");
  // The fixture renders one prefix marker per declared tool. Removing the
  // catalog changes both the arithmetic result and the reusable token prefix.
  std::string prompt = "#abxa";
  auto response = Read(Send(request(history, "auto")));
  std::string reply = Content(response);
  EXPECT_EQ(reply, Oracle::Complete(prompt, 2));
  EXPECT_NE(response.find(R"("cached_tokens":0})"), std::string::npos);

  for (const char* choice : {"none", "auto"}) {
    SCOPED_TRACE(choice);
    const auto retained_count = prompt.size() + reply.size() - 1;
    history += "," + Message("assistant", reply) + "," + Message("user", "b");
    prompt += reply + "#b";
    response = Read(Send(request(history, choice)));
    reply = Content(response);
    EXPECT_EQ(reply, Oracle::Complete(prompt, 2));
    EXPECT_NE(response.find(
                  "\"cached_tokens\":" + std::to_string(retained_count) + "}"),
              std::string::npos)
        << response;
  }
}

TEST_F(UntaggedTextServiceTest,
       RejectedContinuationPreservesHistoryAndEditedHistoryReplays) {
  Start();
  const std::string prompt = "abxa";
  const auto reply = Content(Read(Send(Request(Message("user", prompt), 2))));
  const auto history =
      Message("user", prompt) + "," + Message("assistant", reply);
  const auto rejected =
      Read(Send(Request(history + "," + Message("user", "?"), 2)));
  EXPECT_EQ(rejected.find("HTTP/1.1 400"), 0u) << rejected;
  const auto edited = Read(
      Send(Request(Message("user", "xbxa") + "," + Message("assistant", reply) +
                       "," + Message("user", "b"),
                   2)));
  EXPECT_EQ(Content(edited), Oracle::Complete("xbxa" + reply + "#b", 2));
  EXPECT_NE(edited.find(R"("cached_tokens":0})"), std::string::npos);
  const auto continued =
      Read(Send(Request(history + "," + Message("user", "b"), 2)));
  EXPECT_EQ(Content(continued), Oracle::Complete(prompt + reply + "#b", 2));
  EXPECT_NE(continued.find(R"("cached_tokens":5})"), std::string::npos);

  // Advancing an idle row is not an implicit checkpoint. A sibling branch at
  // the old endpoint replays once the previous native state has advanced.
  const auto rewind =
      Read(Send(Request(history + "," + Message("user", "x"), 2)));
  EXPECT_EQ(Content(rewind), Oracle::Complete(prompt + reply + "#x", 2));
  EXPECT_NE(rewind.find(R"("cached_tokens":0})"), std::string::npos);
}

TEST_F(UntaggedTextServiceTest, ActiveAndCancelledCompletionsCannotBeReused) {
  Start();
  loom_serve::testing::StalledHttpClient active;
  const std::string prompt = "abxa";
  const auto reply = Oracle::Complete(prompt, 1);
  ASSERT_NO_FATAL_FAILURE(active.Send(server_, "/v1/chat/completions",
                                      Request(Message("user", prompt), 1)));
  WaitModel("fixture-text",
            R"("phase":"idle","queued_requests":0,"active_requests":1)");
  const auto history =
      Message("user", prompt) + "," + Message("assistant", reply);
  const auto peer =
      Read(Send(Request(history + "," + Message("user", "b"), 2)));
  EXPECT_EQ(Content(peer), Oracle::Complete(prompt + reply + "#b", 2));
  EXPECT_NE(peer.find(R"("cached_tokens":0})"), std::string::npos);
  ASSERT_NO_FATAL_FAILURE(active.Close());
  WaitModel("fixture-text",
            R"("phase":"idle","queued_requests":0,"active_requests":0)");
  const auto retry =
      Read(Send(Request(history + "," + Message("user", "x"), 2)));
  EXPECT_EQ(Content(retry), Oracle::Complete(prompt + reply + "#x", 2));
  EXPECT_NE(retry.find(R"("cached_tokens":0})"), std::string::npos);
}

TEST_F(UntaggedTextServiceTest, LongestCompletedHistoryWins) {
  Start();
  const std::string prompt = "abxa";
  loom_serve::testing::StalledHttpClient shorter;
  ASSERT_NO_FATAL_FAILURE(shorter.Send(server_, "/v1/chat/completions",
                                       Request(Message("user", prompt), 1)));
  WaitModel("fixture-text",
            R"("phase":"idle","queued_requests":0,"active_requests":1)");
  const auto longer = Content(Read(Send(Request(Message("user", prompt), 2))));
  EXPECT_EQ(longer, Oracle::Complete(prompt, 2));
  EXPECT_EQ(Content(shorter.Read()), Oracle::Complete(prompt, 1));
  ASSERT_NO_FATAL_FAILURE(shorter.Close());
  WaitModel("fixture-text",
            R"("phase":"idle","queued_requests":0,"active_requests":0)");
  const auto response = Read(Send(Request(Message("user", prompt) + "," +
                                              Message("assistant", longer) +
                                              "," + Message("user", "b"),
                                          2)));
  EXPECT_EQ(Content(response), Oracle::Complete(prompt + longer + "#b", 2));
  EXPECT_NE(response.find(R"("cached_tokens":5})"), std::string::npos);
}

TEST_F(UntaggedTextServiceTest, DisplacedHistoryReplaysWithoutExtraCache) {
  Start();
  const std::string prompt = "abxa";
  const auto reply = Content(Read(Send(Request(Message("user", prompt), 2))));
  for (const auto& other : {"baab", "xbxa", "babb", "xxab"}) {
    EXPECT_EQ(Content(Read(Send(Request(Message("user", other), 2)))),
              Oracle::Complete(other, 2));
  }
  const auto replay = Read(
      Send(Request(Message("user", prompt) + "," + Message("assistant", reply) +
                       "," + Message("user", "b"),
                   2)));
  EXPECT_EQ(Content(replay), Oracle::Complete(prompt + reply + "#b", 2));
  EXPECT_NE(replay.find(R"("cached_tokens":0})"), std::string::npos);
}

TEST_F(SourceTextServiceTest, StreamingBatchesReuseScratchWithoutHeapChurn) {
  const auto* policy = loom_serve_text_model_chat_policy(model_);
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
    iree_arena_initialize(loom_serve_host_block_pool(host_), &arena);
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

TEST_F(SourceTextServiceTest,
       CompletionAliasSurvivesBatchAndRequestRetirement) {
  const auto* policy = loom_serve_text_model_chat_policy(model_);
  const std::string body = Request(Message("user", "abx"), 4);
  loom_serve_text_chat_t chat = {};
  IREE_ASSERT_OK(loom_serve_text_chat_initialize(
      policy, policy->name, iree_make_cstring_view(body.c_str()), 1, 4,
      host_allocator_, &chat));
  iree_arena_allocator_t arena;
  iree_arena_initialize(loom_serve_host_block_pool(host_), &arena);
  loom_serve_text_tool_calls_t tools;
  loom_serve_text_tool_calls_initialize(host_allocator_, &tools);
  loom_serve_text_chat_scratch_t scratch;
  loom_serve_text_chat_scratch_initialize(host_allocator_, &scratch);
  loom_serve_text_chat_completion_t completion = {};
  // The fixture returns the prompt itself for an empty response. That source
  // alias must keep owned storage, not retain a temporary argument wrapper.
  IREE_ASSERT_OK(loom_serve_text_chat_complete(
      &chat, iree_string_view_empty(), 1, &tools, &scratch,
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

TEST_F(SourceTextServiceTest, OutputScratchFailureLeavesParserUnadvanced) {
  const auto* policy = loom_serve_text_model_chat_policy(model_);
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

TEST_F(SourceTextServiceTest, CompletionFormattingReusesCapacityAcrossRows) {
  const auto* policy = loom_serve_text_model_chat_policy(model_);
  const std::string body = Request(Message("user", "a"), 4);
  iree::vm::testing::CountingAllocator allocator;
  loom_serve_text_chat_scratch_t scratch;
  loom_serve_text_chat_scratch_initialize(allocator.allocator(), &scratch);
  loom_serve_text_tool_calls_t tools;
  loom_serve_text_tool_calls_initialize(allocator.allocator(), &tools);
  iree_arena_allocator_t arena;
  iree_arena_initialize(loom_serve_host_block_pool(host_), &arena);
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
        &chat, IREE_SV("b"), row + 1, &tools, &scratch,
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

TEST_F(SourceTextServiceTest,
       PreparedInputSurvivesScratchReuseAndHistoryEviction) {
  const auto* policy = loom_serve_text_model_chat_policy(model_);
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
  IREE_ASSERT_OK(loom_serve_text_chat_complete(&first, IREE_SV("b"), 1, &tools,
                                               &completion_scratch,
                                               host_allocator_, &completion));
  const loom_serve_text_input_prefix_t prefix = {
      .completion = &completion,
      .position = 1,
      .token = Oracle::Id('b'),
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

TEST_F(RegisteredSourceTextServiceTest,
       CheckpointDeletionRetiresWhileInferenceAndCaptureAreHeld) {
  StartManaged();
  const std::string headers = "X-Loom-Model: fixture-text\r\n";
  const std::string selector = R"({"model":"fixture-text"})";
  EXPECT_EQ(Read(Send(Declaration("fixture-text", directory_, "drained"), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Read(Send("", headers, "/v1/checkpoints/absent", "DELETE"))
                .find("HTTP/1.1 404"),
            0u);
  EXPECT_NE(WaitModel("fixture-text", R"("registration":"registered")")
                .find(R"("admission":"drained")"),
            std::string::npos);
  EXPECT_EQ(Read(Send(selector, "", "/v1/models/undrain")).find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(Send(Request(Message("user", "abxa"), 2),
                      "X-Loom-Session: base\r\n"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(Send("", headers + "X-Loom-Session: base\r\n",
                      "/v1/checkpoints/retained"))
                .find("HTTP/1.1 201"),
            0u);
  EXPECT_EQ(Read(Send(selector, "", "/v1/models/drain")).find("HTTP/1.1 200"),
            0u);

  const int inference =
      Send(Request(Message("user", "b"), 2), "X-Loom-Session: waiting\r\n");
  const int capture =
      Send("", headers + "X-Loom-Session: base\r\n", "/v1/checkpoints/another");
  WaitModel("fixture-text", R"("held_requests":2)");
  EXPECT_EQ(Read(Send("", headers, "/v1/checkpoints/retained", "DELETE"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(Send("", headers, "/v1/checkpoints/retained", "DELETE"))
                .find("HTTP/1.1 404"),
            0u);
  EXPECT_EQ(Read(Send("", headers, "/v1/checkpoints/another/suspend", "DELETE"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_NE(WaitModel("fixture-text", R"("held_requests":2)")
                .find(R"("admission":"drained")"),
            std::string::npos);
  EXPECT_EQ(Read(Send(selector, "", "/v1/models/undrain")).find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(inference).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Read(capture).find("HTTP/1.1 201"), 0u);

  EXPECT_EQ(Read(Send(selector, "", "/v1/models/drain")).find("HTTP/1.1 200"),
            0u);
  const int suspension = Send("", headers, "/v1/checkpoints/another/suspend");
  WaitModel("fixture-text", R"("held_requests":1)");
  EXPECT_EQ(Read(Send("", headers, "/v1/checkpoints/another", "DELETE"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_NE(WaitModel("fixture-text", R"("held_requests":1)")
                .find(R"("admission":"drained")"),
            std::string::npos);
  EXPECT_EQ(Read(Send(selector, "", "/v1/models/undrain")).find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(suspension).find("HTTP/1.1 404"), 0u);
}

TEST_F(RegisteredSourceTextServiceTest,
       DynamicTextAdmissionCheckpointRetirementAndReuse) {
  StartManaged();
  EXPECT_NE(Read(Send("", "", "/v1/models", "GET")).find(R"("data":[])"),
            std::string::npos);
  loom_serve::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(Read(Send(Declaration("fixture-text", directory_, "drained"), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  const auto declared = WaitModel("fixture-text", R"("admission":"drained")");
  EXPECT_NE(declared.find(R"("registration":"registered")"), std::string::npos);
  const auto initial_log =
      Read(Send("", "X-Loom-Model: fixture-text\r\n", "/logz", "GET"));
  EXPECT_NE(initial_log.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  const std::string prompt = "abxa";
  const int held =
      Send(Request(Message("user", prompt), 2), "X-Loom-Session: retained\r\n");
  const int tokenizer =
      Send(R"({"model":"fixture-text","content":"abx"})", "", "/tokenize");
  WaitModel("fixture-text", R"("held_requests":2)");
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Content(Read(held)), Oracle::Complete(prompt, 2));
  EXPECT_NE(Read(tokenizer).find(R"({"tokens":[0,1,2]})"), std::string::npos);
  const auto first = WaitModel("fixture-text", R"("registration":"ready")");
  const auto pinned = Read(
      Send("", "X-Loom-Model: fixture-text\r\nX-Loom-Session: retained\r\n",
           "/v1/checkpoints/base"));
  EXPECT_EQ(pinned.find("HTTP/1.1 201"), 0u) << pinned;
  EXPECT_EQ(
      Read(Send(Declaration("peer", directory_), "", "/v1/models/register"))
          .find("HTTP/1.1 202"),
      0u);
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/drain"))
                .find("HTTP/1.1 200"),
            0u);
  const int removed = Send(Request(Message("user", "b")));
  WaitModel("fixture-text", R"("held_requests":1)");
  std::string peer = Request(Message("user", "ab"), 2);
  peer.replace(peer.find("fixture-text"), 12, "peer");
  EXPECT_EQ(Content(Read(Send(peer))), Oracle::Complete("ab", 2));
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text","mode":"drain"})", "",
                      "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Read(removed).find("HTTP/1.1 410"), 0u);
  WaitModel("fixture-text", {});
  EXPECT_EQ(Read(Send(Declaration("fixture-text", directory_), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", prompt), 2)))),
            Oracle::Complete(prompt, 2));
  const auto replacement =
      WaitModel("fixture-text", R"("registration":"ready")");
  const auto incarnation = [](const std::string& state) {
    const auto start = state.find("\"incarnation\":");
    EXPECT_NE(start, std::string::npos);
    return state.substr(start, state.find(',', start) - start);
  };
  EXPECT_NE(incarnation(first), incarnation(replacement));
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text","mode":"abort"})", "",
                      "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("fixture-text", {});
  EXPECT_EQ(Content(Read(Send(peer))), Oracle::Complete("ab", 2));
}

TEST_F(RegisteredSourceTextServiceTest,
       IdleCacheExchangesFullRowsAndSharesExplicitPins) {
  StartManaged();
  std::string declaration = Declaration("fixture-text", directory_, "open",
                                        R"(,"session_cache_capacity":2)");
  ASSERT_EQ(
      Read(Send(declaration, "", "/v1/models/register")).find("HTTP/1.1 202"),
      0u);
  std::vector<std::string> prompts, replies;
  for (size_t i = 0; i < 6; ++i) {
    prompts.push_back("abxa" + std::string(i, 'b'));
    replies.push_back(Content(Read(
        Send(Request(Message("user", prompts.back()), 2),
             "X-Loom-Session: conversation" + std::to_string(i) + "\r\n"))));
    EXPECT_EQ(replies.back(), Oracle::Complete(prompts.back(), 2));
  }
  // The first conversation no longer has a row. Both explicit names retain its
  // existing immutable endpoint, independently of automatic cache ownership.
  for (const char* name : {"one", "two"}) {
    const auto pinned = Read(Send("", "X-Loom-Session: conversation0\r\n",
                                  std::string("/v1/checkpoints/") + name));
    ASSERT_EQ(pinned.find("HTTP/1.1 201"), 0u) << pinned;
  }
  EXPECT_EQ(Read(Send("", "X-Loom-Session: conversation1\r\n",
                      "/v1/checkpoints/full"))
                .find("HTTP/1.1 503"),
            0u);
  EXPECT_EQ(
      Read(Send("", "", "/v1/checkpoints/one", "DELETE")).find("HTTP/1.1 200"),
      0u);
  for (size_t i = 0; i < 6; ++i) {
    const auto response =
        Read(Send(Request(Message("user", prompts[i]) + "," +
                              Message("assistant", replies[i]) + "," +
                              Message("user", "b"),
                          2),
                  "X-Loom-Session: conversation" + std::to_string(i) + "\r\n"));
    EXPECT_EQ(Content(response),
              Oracle::Complete(prompts[i] + replies[i] + "#b", 2));
    EXPECT_NE(response.find("\"cached_tokens\":" +
                            std::to_string(prompts[i].size() + 1) + "}"),
              std::string::npos)
        << response;
  }
  // The automatic owner has been consumed and the resumed row has advanced.
  // The surviving explicit alias still represents the original endpoint.
  const auto rewind = Read(Send(
      Request(Message("user", prompts[0]) + "," +
                  Message("assistant", replies[0]) + "," + Message("user", "x"),
              2),
      "X-Loom-Checkpoint: two\r\n"));
  EXPECT_EQ(Content(rewind),
            Oracle::Complete(prompts[0] + replies[0] + "#x", 2));
  EXPECT_NE(rewind.find("\"cached_tokens\":5}"), std::string::npos);
  EXPECT_EQ(
      Read(Send("", "", "/v1/checkpoints/two", "DELETE")).find("HTTP/1.1 200"),
      0u);
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text","mode":"drain"})", "",
                      "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("fixture-text", {});
  // All automatic and explicit ownership retires with the registration.
  EXPECT_EQ(
      Read(Send(declaration, "", "/v1/models/register")).find("HTTP/1.1 202"),
      0u);
  const auto fresh = Read(Send(Request(Message("user", "abxa"), 2),
                               "X-Loom-Session: conversation0\r\n"));
  EXPECT_EQ(Content(fresh), Oracle::Complete("abxa", 2));
  EXPECT_NE(fresh.find("\"cached_tokens\":0}"), std::string::npos);
}

TEST_F(SourceTextServiceTest, CancelledRowExchangePreservesDetachedHistory) {
  CheckRetentionCancellation(OverlapAction::kDisconnect);
}

TEST_F(RegisteredSourceTextServiceTest,
       InventoryTracksRetainedLifetimesAcrossRowReuse) {
  StartManaged();
  const std::string headers = "X-Loom-Model: fixture-text\r\n";
  const auto inspect = [&](const char* target, const std::string& extra = "") {
    return Read(Send("", headers + extra, target, "GET"));
  };
  EXPECT_EQ(inspect("/v1/sessions").find("HTTP/1.1 404"), 0u);
  auto declaration = Declaration("fixture-text", directory_, "drained",
                                 R"(,"session_cache_capacity":64)");
  ASSERT_EQ(
      Read(Send(declaration, "", "/v1/models/register")).find("HTTP/1.1 202"),
      0u);
  EXPECT_EQ(inspect("/v1/sessions").find("HTTP/1.1 409"), 0u);
  ASSERT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  std::string initial;
  for (size_t i = 0; i < 6; ++i) {
    const auto reply =
        Read(Send(Request(Message("user", "abxa"), 2),
                  "X-Loom-Session: conversation" + std::to_string(i) + "\r\n"));
    EXPECT_EQ(Content(reply), Oracle::Complete("abxa", 2));
    if (!i) {
      initial = inspect("/v1/sessions");
    }
  }
  EXPECT_NE(
      initial.find(
          R"("id":1,"name":"conversation0","request":null,"storage":"row",)"),
      std::string::npos)
      << initial;
  const auto retained = inspect("/v1/sessions");
  EXPECT_NE(
      retained.find(
          R"("id":1,"name":"conversation0","request":null,"storage":"cache",)"),
      std::string::npos)
      << retained;
  EXPECT_NE(
      retained.find(R"("position":5,"resident_tokens":8,"reserved_tokens":0,)"),
      std::string::npos)
      << retained;
  EXPECT_NE(retained.find(R"("next_offset":64)"), std::string::npos);
  EXPECT_NE(inspect("/v1/sessions", "X-Loom-Offset: 64\r\n")
                .find(R"("data":[],"next_offset":null)"),
            std::string::npos);
  EXPECT_NE(inspect("/v1/requests").find(R"("data":[],"next_offset":null)"),
            std::string::npos);
  for (const char* value : {"-1", "+1", "1x", "18446744073709551616"}) {
    EXPECT_EQ(
        inspect("/v1/sessions", std::string("X-Loom-Offset: ") + value + "\r\n")
            .find("HTTP/1.1 400"),
        0u);
  }
  EXPECT_EQ(Read(Send("", "", "/v1/requests", "GET")).find("HTTP/1.1 400"), 0u);
  EXPECT_EQ(
      Read(Send("{}", headers, "/v1/requests", "GET")).find("HTTP/1.1 400"),
      0u);
  EXPECT_EQ(retained.find("abxa"), std::string::npos);
  StalledClient inspector;
  ASSERT_NO_FATAL_FAILURE(inspector.Send(server_, "", "/v1/sessions", "GET"));
  inspector.WaitUntilSendCreditExhausted();
  // Copied HTTP bytes may outlive the entire registration. The inspector has
  // not polled its carrier and therefore has not consumed the pending send.
  ASSERT_EQ(
      Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/unregister"))
          .find("HTTP/1.1 202"),
      0u);
  WaitModel("fixture-text", {});
  EXPECT_EQ(inspect("/v1/sessions").find("HTTP/1.1 404"), 0u);
  ASSERT_EQ(Read(Send(Declaration("fixture-text", directory_), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("fixture-text", R"("registration":"ready")");
  const auto replacement = inspect("/v1/sessions");
  EXPECT_NE(replacement.find(R"("incarnation":2,)"), std::string::npos)
      << replacement;
  EXPECT_NE(replacement.find(R"("data":[],"next_offset":null)"),
            std::string::npos);
  const auto old_inventory = inspector.Read();
  EXPECT_NE(old_inventory.find(R"("incarnation":1,)"), std::string::npos);
  EXPECT_NE(old_inventory.find(R"("name":"conversation0",)"),
            std::string::npos);
  EXPECT_EQ(old_inventory.find(R"("incarnation":2,)"), std::string::npos);
}

TEST_F(SourceTextServiceTest, ShutdownRetiresAcceptedRowExchange) {
  CheckRetentionCancellation(OverlapAction::kStop);
}

TEST_F(RegisteredSourceTextServiceTest,
       IdleCacheEvictsLruButPreservesRejectedHistory) {
  StartManaged();
  std::string declaration = Declaration("fixture-text", directory_, "open",
                                        R"(,"session_cache_capacity":1)");
  ASSERT_EQ(
      Read(Send(declaration, "", "/v1/models/register")).find("HTTP/1.1 202"),
      0u);
  const auto first = Oracle::Complete("abxa", 2);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(Content(Read(Send(Request(Message("user", "abxa"), 2),
                                "X-Loom-Session: conversation" +
                                    std::to_string(i) + "\r\n"))),
              first);
  }
  EXPECT_EQ(Read(Send("", "X-Loom-Session: conversation0\r\n",
                      "/v1/checkpoints/evicted"))
                .find("HTTP/1.1 404"),
            0u);
  const auto history =
      Message("user", "abxa") + "," + Message("assistant", first);
  EXPECT_EQ(Read(Send(Request(history + "," + Message("user", "?"), 2),
                      "X-Loom-Session: conversation1\r\n"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("", "X-Loom-Session: conversation1\r\n",
                      "/v1/checkpoints/retained"))
                .find("HTTP/1.1 201"),
            0u);
  const auto continued =
      Read(Send(Request(history + "," + Message("user", "b"), 2),
                "X-Loom-Session: conversation1\r\n"));
  EXPECT_EQ(Content(continued), Oracle::Complete("abxa" + first + "#b", 2));
  EXPECT_NE(continued.find("\"cached_tokens\":5}"), std::string::npos);
  // A changed automatic history is an ordinary replay, not an explicit-pin
  // validation error. The independent explicit endpoint remains usable.
  const auto changed = Read(Send(Request(Message("user", "baxx"), 2),
                                 "X-Loom-Session: conversation2\r\n"));
  EXPECT_EQ(Content(changed), Oracle::Complete("baxx", 2));
  EXPECT_NE(changed.find("\"cached_tokens\":0}"), std::string::npos);
  const auto rewind =
      Read(Send(Request(history + "," + Message("user", "x"), 2),
                "X-Loom-Checkpoint: retained\r\n"));
  EXPECT_EQ(Content(rewind), Oracle::Complete("abxa" + first + "#x", 2));
  EXPECT_NE(rewind.find("\"cached_tokens\":5}"), std::string::npos);
}

TEST_F(RegisteredSourceTextServiceTest,
       IdleCacheYieldsLogicalBlocksToAdmission) {
  StartManaged();
  std::string declaration = Declaration("fixture-text", directory_, "open",
                                        R"(,"session_cache_capacity":2)");
  const auto pool = declaration.find("\"pool_capacity\":256");
  ASSERT_NE(pool, std::string::npos);
  declaration.replace(pool, strlen("\"pool_capacity\":256"),
                      "\"pool_capacity\":32");
  ASSERT_EQ(
      Read(Send(declaration, "", "/v1/models/register")).find("HTTP/1.1 202"),
      0u);
  const auto first = Oracle::Complete("abxa", 2);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(Content(Read(Send(Request(Message("user", "abxa"), 2),
                                "X-Loom-Session: conversation" +
                                    std::to_string(i) + "\r\n"))),
              first);
  }
  const auto replay = Read(
      Send(Request(Message("user", "abxa") + "," + Message("assistant", first) +
                       "," + Message("user", "b"),
                   2),
           "X-Loom-Session: conversation0\r\n"));
  EXPECT_EQ(Content(replay), Oracle::Complete("abxa" + first + "#b", 2));
  EXPECT_NE(replay.find("\"cached_tokens\":0}"), std::string::npos);
}

TEST_F(SourceTextServiceTest, OrderedImagesReachSourceDefinedInputSpans) {
  const auto* policy = loom_serve_text_model_chat_policy(model_);
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
  EXPECT_EQ(input.media_spans[0].image_index, 0u);
  EXPECT_EQ(input.media_spans[0].input_offset, 1u);
  EXPECT_EQ(input.media_spans[0].input_count, 1u);
  EXPECT_EQ(input.media_spans[1].image_index, 1u);
  EXPECT_EQ(input.media_spans[1].input_offset, 3u);
  EXPECT_EQ(input.media_spans[1].input_count, 1u);
  loom_serve_text_input_deinitialize(&input, host_allocator_);
  loom_serve_text_chat_deinitialize(&chat);
}

TEST_F(SourceTextServiceTest, RetainedHistoryRequiresExactOrderedMedia) {
  const auto* policy = loom_serve_text_model_chat_policy(model_);
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
  IREE_ASSERT_OK(loom_serve_text_chat_complete(&first, IREE_SV("b"), 10, &tools,
                                               &completion_scratch,
                                               host_allocator_, &completion));
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
  loom_serve_text_tool_calls_deinitialize(&tools);
  loom_serve_text_chat_scratch_deinitialize(&completion_scratch);
  loom_serve_text_chat_deinitialize(&different_placement);
  loom_serve_text_chat_deinitialize(&different_detail);
  loom_serve_text_chat_deinitialize(&different_image);
  loom_serve_text_chat_deinitialize(&matching);
  loom_serve_text_chat_deinitialize(&first);
}

}  // namespace
