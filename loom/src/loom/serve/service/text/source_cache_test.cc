// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cerrno>

#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::RegisteredSourceTextServiceTest;
using loom_serve::testing::text::SourceTextServiceTest;
using loom_serve::testing::text::StalledClient;

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
  iree_string_builder_t tools;
  iree_string_builder_initialize(host_allocator_, &tools);
  loom_serve_text_chat_completion_t completion = {};
  IREE_ASSERT_OK(loom_serve_text_chat_complete(&first, IREE_SV("b"), 1, &tools,
                                               &completion));
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
  iree_string_builder_deinitialize(&tools);
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

}  // namespace
