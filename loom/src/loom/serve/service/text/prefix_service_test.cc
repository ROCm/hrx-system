// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;
using loom_serve::testing::text::StalledClient;

class PrefixServiceTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_service_options_t ServiceOptions() const override {
    auto options = SourceTextServiceTest::ServiceOptions();
    options.prefix_cache_capacity = 2;
    return options;
  }

  void ExpectResponse(const std::string& response, const std::string& prompt,
                      size_t limit, size_t cached) {
    EXPECT_EQ(Content(response), Oracle::Complete(prompt, limit));
    EXPECT_NE(
        response.find("\"cached_tokens\":" + std::to_string(cached) + "}"),
        std::string::npos)
        << response;
    EXPECT_NE(response.find(
                  "\"prompt_tokens\":" + std::to_string(prompt.size()) + ","),
              std::string::npos)
        << response;
  }
};

TEST_F(PrefixServiceTest, SharesActiveSourceWithIndependentDivergentRequests) {
  Start(4);
  StalledClient source;
  ASSERT_NO_FATAL_FAILURE(
      source.Send(server_, Request(Message("user", "abxabbxa"), 25)));
  const std::string held =
      R"("phase":"idle","queued_requests":0,"active_requests":1)";
  WaitModel("fixture-text", held);

  // A real carrier-held source remains active with an outstanding completion
  // reservation. The cached first page is already usable by unrelated peers.
  for (const auto& [prompt, cached] :
       std::vector<std::pair<std::string, size_t>>{
           {"abxaxbb", 4}, {"abxa", 0}, {"abxax", 4}, {"axb", 0}}) {
    ExpectResponse(Read(Send(Request(Message("user", prompt), 3))), prompt, 3,
                   cached);
    WaitModel("fixture-text", held);
  }

  // Cancelling the original row does not cancel an independently retained
  // immutable endpoint or change the tokens selected from another branch.
  ASSERT_NO_FATAL_FAILURE(source.Close());
  WaitModel("fixture-text",
            R"("phase":"idle","queued_requests":0,"active_requests":0)");
  ExpectResponse(Read(Send(Request(Message("user", "abxabx"), 3))), "abxabx", 3,
                 4);
  const auto log =
      Read(Send("", "X-Loom-Model: fixture-text\r\n", "/logz", "GET"));
  EXPECT_NE(log.find(R"("event":"prefix_capture")"), std::string::npos) << log;
}

TEST_F(PrefixServiceTest, SeparatesTokenIdenticalMediaPrefixes) {
  Start(4);
  const std::string prompt = "abxxabb";
  const auto run = [&](const std::string& before, const std::string& mime_type,
                       const std::string& base64, const std::string& detail,
                       const std::string& after, size_t cached) {
    const std::string message =
        ImageMessage(before, mime_type, base64, detail, after);
    ExpectResponse(Read(Send(Request(message, 2))), prompt, 2, cached);
  };
  run("abx", "image/png", kPng, "low", "abb", 0);
  run("ab", "image/png", kPng, "low", "xabb", 0);
  run("abx", "image/png", kPng, "low", "abb", 4);
  run("abx", "image/gif", kGif, "low", "abb", 0);
  run("abx", "image/png", kPng, "low", "abb", 4);
  run("abx", "image/png", kPng, "high", "abb", 0);
  run("abx", "image/png", kPng, "low", "abb", 4);
}

TEST_F(PrefixServiceTest, PreservesNamedCompletedHistoryAfterAutomaticReuse) {
  Start(4);
  ExpectResponse(Read(Send(Request(Message("user", "abxabb"), 2))), "abxabb", 2,
                 0);
  const std::string prompt = "abxaxa";
  const auto first = Read(
      Send(Request(Message("user", prompt), 3), "X-Loom-Session: branch\r\n"));
  ExpectResponse(first, prompt, 3, 4);
  const std::string output = Content(first);
  const auto next = Read(Send(
      Request(Message("user", prompt) + "," + Message("assistant", output) +
                  "," + Message("user", "b"),
              3),
      "X-Loom-Session: branch\r\n"));
  ExpectResponse(next, prompt + output + "#b", 3, prompt.size() + 2);
}

TEST_F(PrefixServiceTest, LruEvictionKeepsRecentlyUsedAncestor) {
  Start(4);
  auto run = [&](const std::string& prompt, size_t cached) {
    ExpectResponse(Read(Send(Request(Message("user", prompt), 2))), prompt, 2,
                   cached);
  };
  run("abxabb", 0);
  run("baxaxx", 0);
  run("abxaxx", 4);
  run("xabaax", 0);
  run("abxaab", 4);
  run("baxaxb", 0);
}

TEST_F(PrefixServiceTest, PressureReplaysWhenSelectedPrefixCannotFit) {
  IREE_ASSERT_OK(loom_serve_text_model_free(model_));
  model_ = nullptr;
  auto options = ModelOptions();
  options.pool_capacity = 16;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                host_allocator_, &model_));
  // A partial two-page endpoint consumes six tokens. Restoring a full-pool
  // request would need its shared partial tail plus a private replacement.
  // Dropping the optional prefix makes all sixteen positions available.
  Start(3);
  ExpectResponse(Read(Send(Request(Message("user", "abxabxabb"), 1))),
                 "abxabxabb", 1, 0);
  ExpectResponse(Read(Send(Request(Message("user", "abxabxabab"), 7))),
                 "abxabxabab", 7, 0);
}

TEST_F(PrefixServiceTest, DynamicDeclarationReservesIndependentEndpointQuotas) {
  StartManaged();
  std::string declaration =
      Declaration("fixture-text", directory_, "open",
                  R"(,"prefix_cache_capacity":2,"session_cache_capacity":2)");
  ASSERT_EQ(
      Read(Send(declaration, "", "/v1/models/register")).find("HTTP/1.1 202"),
      0u);
  const std::string prefix(32, 'a');
  ExpectResponse(Read(Send(Request(Message("user", prefix + "bb"), 2),
                           "X-Loom-Session: first\r\n")),
                 prefix + "bb", 2, 0);
  ExpectResponse(Read(Send(Request(Message("user", prefix + "xx"), 2),
                           "X-Loom-Session: second\r\n")),
                 prefix + "xx", 2, 32);
  for (const auto& name : {"first", "second"}) {
    const auto pinned =
        Read(Send("", "X-Loom-Session: " + std::string(name) + "\r\n",
                  "/v1/checkpoints/" + std::string(name)));
    EXPECT_EQ(pinned.find("HTTP/1.1 201"), 0u) << pinned;
  }
}

TEST_F(PrefixServiceTest, StopRetiresAcceptedCaptureBeforeReleasingOwners) {
  auto options = ServiceOptions();
  options.chunk_size = 4;
  loom_serve_text_service_t* service = nullptr;
  IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                  host_allocator_, &service));
  const auto interface = loom_serve_text_service_interface(service);
  Observe(interface);
  const int client = Send(Request(Message("user", "abxabbxaxa"), 12));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client, interface, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
    return std::string(interface.snapshot(interface.self).phase) ==
           "prefix_capture";
  }));
  EXPECT_TRUE(interface.is_pending(interface.self));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      interface,
      [&] {
        return !interface.is_pending(interface.self) &&
               !interface.is_active(interface.self);
      },
      LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING));
  EXPECT_EQ(interface.telemetry.counters->cancelled, 1u);
  EXPECT_EQ(interface.telemetry.counters->input_tokens, 4u);
  EXPECT_EQ(interface.telemetry.counters->output_tokens, 0u);
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).pending, 0u);
  interface.set_observer(interface.self, {});
  loom_serve_text_service_free(service, IREE_STATUS_OK);
  Read(client, /*allow_empty=*/true);
  for (iree_host_size_t i = 0; i < options.row_count; ++i) {
    loom_serve_text_row_reset_retired(loom_serve_text_model_row(model_, i));
  }
  const auto pool = loom_serve_text_model_pool_usage(model_);
  EXPECT_EQ(pool.available, pool.capacity);
}

}  // namespace
