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

class AutomaticCacheTextServiceTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.checkpoint_capacity = 0;
    options.cache = {65536, LOOM_SERVE_TEXT_CACHE_AUTOMATIC,
                     LOOM_SERVE_TEXT_CACHE_AUTOMATIC};
    return options;
  }

  loom_serve_text_service_options_t ServiceOptions() const override {
    auto options = SourceTextServiceTest::ServiceOptions();
    const auto selected = loom_serve_text_model_cache_selection(model_);
    options.row_count = loom_serve_text_model_row_count(model_);
    options.session_cache_capacity = selected.session_capacity;
    options.prefix_cache_capacity = selected.prefix_capacity;
    return options;
  }
};

TEST_F(AutomaticCacheTextServiceTest,
       SelectedSourceStateServesAndRestoresHistory) {
  const auto selected = loom_serve_text_model_cache_selection(model_);
  EXPECT_EQ(selected.prefix_capacity, 2u);
  EXPECT_GT(selected.session_capacity, 8u);
  EXPECT_LE(selected.session_capacity, 64u);
  EXPECT_EQ(selected.checkpoint_capacity, selected.session_capacity + 3);
  EXPECT_GT(selected.additional_bytes, 0u);
  EXPECT_LE(selected.additional_bytes, 65536u);
  EXPECT_EQ(loom_serve_text_model_checkpoint_capacity(model_),
            selected.checkpoint_capacity);
  loom_serve_text_resize_t* resize = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_prepare_resize(model_, 8, &resize));
  ASSERT_NE(resize, nullptr);
  loom_serve_execution_frontier_t frontier;
  auto status = loom_serve_text_model_submit_resize(model_, resize, &frontier);
  loom_serve_text_resize_free(resize);
  IREE_ASSERT_OK(status);
  for (size_t i = 0; i < IREE_ARRAYSIZE(frontier.semaphores); ++i) {
    IREE_ASSERT_OK(
        loom_serve_workspace_wait(loom_serve_device_workspace(device_),
                                  frontier.semaphores[i], frontier.values[i]));
  }
  loom_serve_text_model_complete_resize(model_);
  EXPECT_EQ(loom_serve_text_model_row_count(model_), 8u);
  EXPECT_EQ(loom_serve_text_model_checkpoint_capacity(model_),
            selected.checkpoint_capacity);
  Start();
  std::vector<std::string> replies;
  for (int i = 0; i < 12; ++i) {
    const auto response =
        Read(Send(Request(Message("user", "abxa"), 2),
                  "X-Loom-Session: agent-" + std::to_string(i) + "\r\n"));
    replies.push_back(Content(response));
    EXPECT_EQ(replies.back(), Oracle::Complete("abxa", 2));
  }
  const auto response = Read(Send(
      Request(Message("user", "abxa") + "," + Message("assistant", replies[0]) +
                  "," + Message("user", "b"),
              2),
      "X-Loom-Session: agent-0\r\n"));
  EXPECT_EQ(Content(response), Oracle::Complete("abxa" + replies[0] + "#b", 2));
  EXPECT_NE(response.find(R"("cached_tokens":5})"), std::string::npos)
      << response;
}

TEST_F(RegisteredSourceTextServiceTest,
       AutomaticCacheDefaultsAndExplicitDisablement) {
  StartManaged();
  for (bool automatic : {true, false}) {
    const std::string fields =
        std::string(R"(,"max_sessions":1)") +
        (automatic
             ? ""
             : R"(,"session_cache_capacity":0,"prefix_cache_capacity":0)");
    ASSERT_EQ(Read(Send(Declaration("fixture-text", directory_, "open", fields),
                        "", "/v1/models/register"))
                  .find("HTTP/1.1 202"),
              0u);
    const auto first = Content(Read(Send(Request(Message("user", "abxa"), 2),
                                         "X-Loom-Session: first\r\n")));
    EXPECT_EQ(first, Oracle::Complete("abxa", 2));
    EXPECT_EQ(Content(Read(Send(Request(Message("user", "xbbx"), 2),
                                "X-Loom-Session: peer\r\n"))),
              Oracle::Complete("xbbx", 2));
    const auto response = Read(Send(
        Request(Message("user", "abxa") + "," + Message("assistant", first) +
                    "," + Message("user", "b"),
                2),
        "X-Loom-Session: first\r\n"));
    EXPECT_EQ(Content(response), Oracle::Complete("abxa" + first + "#b", 2));
    EXPECT_NE(response.find(automatic ? R"("cached_tokens":5})"
                                      : R"("cached_tokens":0})"),
              std::string::npos)
        << response;
    ASSERT_EQ(
        Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/unregister"))
            .find("HTTP/1.1 202"),
        0u);
    WaitModelAbsent("fixture-text");
  }
}

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
  WaitModelState("fixture-text",
                 R"("phase":"idle","queued_requests":0,"active_requests":1)");
  const auto history =
      Message("user", prompt) + "," + Message("assistant", reply);
  const auto peer =
      Read(Send(Request(history + "," + Message("user", "b"), 2)));
  EXPECT_EQ(Content(peer), Oracle::Complete(prompt + reply + "#b", 2));
  EXPECT_NE(peer.find(R"("cached_tokens":0})"), std::string::npos);
  ASSERT_NO_FATAL_FAILURE(active.Close());
  WaitModelState("fixture-text",
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
  WaitModelState("fixture-text",
                 R"("phase":"idle","queued_requests":0,"active_requests":1)");
  const auto longer = Content(Read(Send(Request(Message("user", prompt), 2))));
  EXPECT_EQ(longer, Oracle::Complete(prompt, 2));
  EXPECT_EQ(Content(shorter.Read()), Oracle::Complete(prompt, 1));
  ASSERT_NO_FATAL_FAILURE(shorter.Close());
  WaitModelState("fixture-text",
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
  EXPECT_NE(WaitModelState("fixture-text", R"("registration":"registered")")
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
  WaitModelState("fixture-text", R"("held_requests":2)");
  EXPECT_EQ(Read(Send("", headers, "/v1/checkpoints/retained", "DELETE"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(Send("", headers, "/v1/checkpoints/retained", "DELETE"))
                .find("HTTP/1.1 404"),
            0u);
  EXPECT_EQ(Read(Send("", headers, "/v1/checkpoints/another/suspend", "DELETE"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_NE(WaitModelState("fixture-text", R"("held_requests":2)")
                .find(R"("admission":"drained")"),
            std::string::npos);
  EXPECT_EQ(Read(Send(selector, "", "/v1/models/undrain")).find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(inference).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Read(capture).find("HTTP/1.1 201"), 0u);

  EXPECT_EQ(Read(Send(selector, "", "/v1/models/drain")).find("HTTP/1.1 200"),
            0u);
  const int suspension = Send("", headers, "/v1/checkpoints/another/suspend");
  WaitModelState("fixture-text", R"("held_requests":1)");
  EXPECT_EQ(Read(Send("", headers, "/v1/checkpoints/another", "DELETE"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_NE(WaitModelState("fixture-text", R"("held_requests":1)")
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
  const auto declared =
      WaitModelState("fixture-text", R"("admission":"drained")");
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
  WaitModelState("fixture-text", R"("held_requests":2)");
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Content(Read(held)), Oracle::Complete(prompt, 2));
  EXPECT_NE(Read(tokenizer).find(R"({"tokens":[0,1,2]})"), std::string::npos);
  const auto first =
      WaitModelState("fixture-text", R"("registration":"ready")");
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
  WaitModelState("fixture-text", R"("held_requests":1)");
  std::string peer = Request(Message("user", "ab"), 2);
  peer.replace(peer.find("fixture-text"), 12, "peer");
  EXPECT_EQ(Content(Read(Send(peer))), Oracle::Complete("ab", 2));
  EXPECT_EQ(Read(Send(R"({"model":"fixture-text","mode":"drain"})", "",
                      "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Read(removed).find("HTTP/1.1 410"), 0u);
  WaitModelAbsent("fixture-text");
  EXPECT_EQ(Read(Send(Declaration("fixture-text", directory_), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", prompt), 2)))),
            Oracle::Complete(prompt, 2));
  const auto replacement =
      WaitModelState("fixture-text", R"("registration":"ready")");
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
  WaitModelAbsent("fixture-text");
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
  WaitModelAbsent("fixture-text");
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

class BpeHistoryServiceTest : public RegisteredSourceTextServiceTest,
                              public ::testing::WithParamInterface<bool> {};

TEST_P(BpeHistoryServiceTest, RetainedHistoriesPreserveGeneratedIds) {
  const bool end_on_eos = GetParam();
  const size_t output_limit = end_on_eos ? 16 : 5;
  tokenizer_ = directory_ + "/tokenizer_merging.json";
  StartManaged();
  const auto declaration = Declaration("fixture-text", directory_, "open",
                                       R"(,"session_cache_capacity":8)");
  ASSERT_EQ(
      Read(Send(declaration, "", "/v1/models/register")).find("HTTP/1.1 202"),
      0u);

  const auto generate = [](const std::vector<int32_t>& input, size_t count) {
    Oracle oracle;
    for (int32_t token : input) {
      oracle.Consume(token);
    }
    std::vector<int32_t> result;
    for (size_t i = 0; i < count; ++i) {
      const int32_t token = oracle.Predict();
      result.push_back(token);
      if (token == 3) {
        break;
      }
      oracle.Consume(token);
    }
    return result;
  };
  const auto decode = [](const std::vector<int32_t>& tokens) {
    const char* spellings[] = {"a", "b", "ab", ""};
    std::string text;
    for (int32_t token : tokens) {
      text += spellings[token];
    }
    return text;
  };

  // Sampling need not produce a canonical BPE encoding of its decoded text.
  // The first answer contains separate a,b IDs; re-encoding merges them, and
  // also merges the original prompt's a with the answer's initial b.
  const auto generated = generate({0}, output_limit);
  ASSERT_EQ(generated,
            end_on_eos ? (std::vector<int32_t>{1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 3})
                       : (std::vector<int32_t>{1, 1, 0, 0, 1}));
  const auto reply = decode(generated);
  ASSERT_EQ(reply, end_on_eos ? "bbaabbabbb" : "bbaab");
  std::vector<int32_t> retained_input = {0};
  retained_input.insert(retained_input.end(), generated.begin(),
                        generated.end());
  if (!end_on_eos) {
    retained_input.push_back(3);
  }
  retained_input.push_back(1);
  const auto replay_input =
      end_on_eos ? (std::vector<int32_t>{2, 1, 0, 2, 1, 2, 1, 1, 3, 1})
                 : (std::vector<int32_t>{2, 1, 0, 2, 3, 1});
  const auto continued = decode(generate(retained_input, 2));
  const auto replayed = decode(generate(replay_input, 2));
  ASSERT_EQ(continued, end_on_eos ? "aab" : "ab");
  ASSERT_EQ(replayed, end_on_eos ? "a" : "aa");
  ASSERT_NE(continued, replayed);
  const auto tokenized =
      Read(Send("{\"model\":\"fixture-text\",\"content\":\"a" + reply + "#b\"}",
                "", "/tokenize"));
  ASSERT_NE(tokenized.find(end_on_eos ? R"({"tokens":[2,1,0,2,1,2,1,1,3,1]})"
                                      : R"({"tokens":[2,1,0,2,3,1]})"),
            std::string::npos)
      << tokenized;

  for (const char* name : {"hot", "cold"}) {
    const auto response =
        Read(Send(Request(Message("user", "a"), output_limit),
                  std::string("X-Loom-Session: ") + name + "\r\n"));
    ASSERT_EQ(Content(response), reply);
    EXPECT_NE(response.find(end_on_eos ? R"("finish_reason":"stop")"
                                       : R"("finish_reason":"length")"),
              std::string::npos);
  }
  const auto pinned =
      Read(Send("", "X-Loom-Session: cold\r\n", "/v1/checkpoints/original"));
  ASSERT_EQ(pinned.find("HTTP/1.1 201"), 0u) << pinned;
  const auto history = Message("user", "a") + "," +
                       Message("assistant", reply) + "," + Message("user", "b");
  const auto check_continuation = [&](const std::string& headers) {
    const auto response = Read(Send(Request(history, 2), headers));
    EXPECT_EQ(Content(response), continued);
    // The prompt and all but the final prediction have been consumed. That
    // exact pending ID must arrive once, whether ordinary text or EOS.
    EXPECT_NE(response.find("\"cached_tokens\":" +
                            std::to_string(generated.size()) + "}"),
              std::string::npos)
        << response;
  };
  check_continuation("X-Loom-Session: hot\r\n");

  // Force row reuse through ordinary requests, then require inventory evidence
  // that the untouched conversation actually moved to the detached cache.
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(Content(Read(Send(
                  Request(Message("user", "a"), 1),
                  "X-Loom-Session: pressure" + std::to_string(i) + "\r\n"))),
              "b");
  }
  const auto inventory =
      Read(Send("", "X-Loom-Model: fixture-text\r\n", "/v1/sessions", "GET"));
  ASSERT_NE(
      inventory.find(R"("name":"cold","request":null,"storage":"cache",)"),
      std::string::npos)
      << inventory;
  check_continuation("X-Loom-Session: cold\r\n");
  // The immutable pin still names the original endpoint after its automatic
  // owner has resumed and advanced, so this must fork the old raw history.
  check_continuation("X-Loom-Checkpoint: original\r\n");

  const auto fresh =
      Read(Send(Request(history, 2), "X-Loom-Session: fresh\r\n"));
  EXPECT_EQ(Content(fresh), replayed);
  EXPECT_NE(fresh.find(R"("cached_tokens":0})"), std::string::npos) << fresh;
}

INSTANTIATE_TEST_SUITE_P(CompletionBoundaries, BpeHistoryServiceTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "EndOfSequence" : "OutputLimit";
                         });

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
  WaitModelAbsent("fixture-text");
  EXPECT_EQ(inspect("/v1/sessions").find("HTTP/1.1 404"), 0u);
  ASSERT_EQ(Read(Send(Declaration("fixture-text", directory_), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModelState("fixture-text", R"("registration":"ready")");
  const auto replacement = inspect("/v1/sessions");
  EXPECT_NE(replacement.find(R"("incarnation":2,)"), std::string::npos)
      << replacement;
  // Pagination addresses the retained directory, including empty cache slots.
  // A fresh incarnation can have several empty pages without any old history.
  std::string page = replacement;
  uint64_t offset = 0;
  const std::string next_key = R"("next_offset":)";
  while (true) {
    EXPECT_NE(page.find(R"("data":[])"), std::string::npos) << page;
    const auto begin = page.find(next_key);
    ASSERT_NE(begin, std::string::npos) << page;
    const auto next = page.substr(begin + next_key.size());
    if (next.starts_with("null")) {
      break;
    }
    const auto next_offset = std::stoull(next);
    ASSERT_GT(next_offset, offset);
    offset = next_offset;
    page = inspect("/v1/sessions",
                   "X-Loom-Offset: " + std::to_string(offset) + "\r\n");
  }
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
