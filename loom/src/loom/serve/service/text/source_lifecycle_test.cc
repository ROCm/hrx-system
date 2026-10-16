// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cerrno>

#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/telemetry/latency.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::RegisteredSourceTextServiceTest;
using loom_serve::testing::text::SourceTextServiceTest;
using loom_serve::testing::text::StalledClient;

uint64_t FirstOutputCount(const loom_serve_http_service_t& interface) {
  uint64_t count = 0;
  for (const auto bucket : interface.telemetry.latency->first_output.buckets) {
    count += bucket;
  }
  return count;
}

TEST_F(SourceTextServiceTest, FirstOutputExcludesFramingAndEmptyCompletions) {
  for (const bool streaming : {false, true}) {
    SCOPED_TRACE(streaming ? "streaming" : "nonstreaming");
    const auto options = ServiceOptions();
    loom_serve_text_service_t* value = nullptr;
    IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                    host_allocator_, &value));
    auto destroy = [](loom_serve_text_service_t* service) {
      loom_serve_text_service_free(service, IREE_STATUS_OK);
    };
    std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
        value, destroy);
    const auto interface = loom_serve_text_service_interface(value);
    Observe(interface);
    const auto idle = [&] {
      return !interface.is_pending(interface.self) &&
             !interface.is_active(interface.self) &&
             !interface.snapshot(interface.self).queued_requests;
    };
    const auto request = [streaming](const char* prompt) {
      return std::string(R"({"model":"fixture-text","stream":)") +
             (streaming ? "true" : "false") +
             R"(,"max_tokens":4,"messages":[{"role":"user","content":")" +
             prompt + R"("}]})";
    };
    loom_serve_http_connection_t* connection = nullptr;
    const int first = Send(request("abxa"));
    ASSERT_NO_FATAL_FAILURE(Accept(first, interface, &connection));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
      return std::string(interface.snapshot(interface.self).phase) == "execute";
    }));
    // Streaming headers and the assistant-role event have carrier acceptance,
    // but the owner has not consumed any model result yet.
    EXPECT_EQ(FirstOutputCount(interface), 0u);
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        interface, [&] { return FirstOutputCount(interface) == 1; }));
    if (streaming) {
      EXPECT_EQ(interface.telemetry.counters->completed, 0u);
    }
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
    const auto first_response = Read(first);
    if (streaming) {
      EXPECT_EQ(Content(first_response), Oracle::Complete("abxa", 4));
    } else {
      EXPECT_NE(first_response.find("\"content\":\"" +
                                    Oracle::Complete("abxa", 4) + "\""),
                std::string::npos);
    }
    EXPECT_EQ(FirstOutputCount(interface), 1u);

    // Eleven consumed positions select EOS immediately, producing a successful
    // empty response. Neither its stop event nor usage is meaningful output.
    const int empty = Send(request("aaaaaaaaaaa"));
    ASSERT_NO_FATAL_FAILURE(Accept(empty, interface, &connection));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
    const auto empty_response = Read(empty);
    if (streaming) {
      EXPECT_TRUE(Content(empty_response).empty());
    } else {
      EXPECT_NE(empty_response.find("\"content\":\"\""), std::string::npos);
    }
    EXPECT_EQ(FirstOutputCount(interface), 1u);
    EXPECT_EQ(interface.telemetry.counters->completed, 2u);
    if (streaming) {
      const int abandoned = Send(request("abxa"));
      ASSERT_NO_FATAL_FAILURE(Accept(abandoned, interface, &connection));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
          interface, [&] { return FirstOutputCount(interface) == 2; }));
      const linger reset = {1, 0};
      ASSERT_EQ(
          setsockopt(abandoned, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
          0);
      ASSERT_EQ(close(abandoned), 0);
      ASSERT_TRUE(iree_notification_await(
          loom_serve_http_server_notification(server_),
          [](void* self) {
            return loom_serve_http_connection_failed(
                static_cast<loom_serve_http_connection_t*>(self));
          },
          connection, iree_infinite_timeout()));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
      EXPECT_EQ(interface.telemetry.counters->cancelled, 1u);
      // Accepted output remains an observation even when its request later
      // cancels; it cannot become a completed-only latency sample.
      EXPECT_EQ(FirstOutputCount(interface), 2u);
    }
    interface.set_observer(interface.self, {});
  }
}

TEST_F(SourceTextServiceTest, DiscoveryReportsConfiguredSessionCapacity) {
  Start();
  const auto models = Read(Send("", "", "/v1/models", "GET"));
  EXPECT_NE(
      models.find(
          R"("input_modalities":["text"],"output_modalities":["text"],"context_length":96,"limits":{"max_output_tokens":96})"),
      std::string::npos)
      << models;
}

TEST_F(SourceTextServiceTest, NonstreamingDisconnectRetiresAnAcceptedEpoch) {
  const auto options = ServiceOptions();
  loom_serve_text_service_t* value = nullptr;
  IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                  host_allocator_, &value));
  auto destroy = [](loom_serve_text_service_t* service) {
    loom_serve_text_service_free(service, IREE_STATUS_OK);
  };
  std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
      value, destroy);
  const auto interface = loom_serve_text_service_interface(value);
  Observe(interface);
  const std::string request =
      R"({"model":"fixture-text","messages":[{"role":"user","content":"abxa"}],"max_tokens":2})";
  const int client = Send(request);
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client, interface, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
    return std::string(interface.snapshot(interface.self).phase) == "execute";
  }));
  // The owner has submitted a real epoch but cannot consume its completion
  // until it runs again. Nonstreaming output must not have published headers.
  char byte = 0;
  EXPECT_EQ(recv(client, &byte, 1, MSG_DONTWAIT), -1);
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(client), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connection, iree_infinite_timeout());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
    return !interface.is_pending(interface.self) &&
           !interface.is_active(interface.self) &&
           !interface.snapshot(interface.self).queued_requests;
  }));
  EXPECT_EQ(interface.telemetry.counters->completed, 0u);
  EXPECT_EQ(interface.telemetry.counters->cancelled, 1u);
  EXPECT_EQ(FirstOutputCount(interface), 0u);
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).pending, 0u);

  const int next = Send(request);
  ASSERT_NO_FATAL_FAILURE(Accept(next, interface, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
    return !interface.is_pending(interface.self) &&
           !interface.is_active(interface.self) &&
           !interface.snapshot(interface.self).queued_requests;
  }));
  const std::string response = Read(next);
  EXPECT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
  EXPECT_NE(response.find("Content-Type: application/json"), std::string::npos);
  EXPECT_NE(response.find(R"("message":{"role":"assistant","content":"bx"})"),
            std::string::npos);
  EXPECT_EQ(response.find("data:"), std::string::npos);
  EXPECT_EQ(interface.telemetry.counters->completed, 1u);
  EXPECT_EQ(interface.telemetry.counters->cancelled, 1u);
  EXPECT_EQ(interface.telemetry.counters->failed, 0u);
  EXPECT_EQ(FirstOutputCount(interface), 1u);
  interface.set_observer(interface.self, {});
}

TEST_F(SourceTextServiceTest, CheckpointClaimSurvivesDisconnectAndStopping) {
  for (const bool stopping : {false, true}) {
    SCOPED_TRACE(stopping ? "stopping" : "disconnected");
    const auto options = ServiceOptions();
    loom_serve_text_service_t* value = nullptr;
    IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                    host_allocator_, &value));
    auto destroy = [](loom_serve_text_service_t* service) {
      loom_serve_text_service_free(service, IREE_STATUS_OK);
    };
    std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
        value, destroy);
    const auto interface = loom_serve_text_service_interface(value);
    Observe(interface);
    const auto idle = [&] {
      return !interface.is_pending(interface.self) &&
             !interface.is_active(interface.self) &&
             !interface.snapshot(interface.self).queued_requests;
    };
    loom_serve_http_connection_t* connection = nullptr;
    const int generation =
        Send(Request(Message("user", "abxa"), 2), "X-Loom-Session: root\r\n");
    ASSERT_NO_FATAL_FAILURE(Accept(generation, interface, &connection));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
    const auto first = Content(Read(generation));
    ASSERT_EQ(first, Oracle::Complete("abxa", 2));

    const int pin =
        Send("", "X-Loom-Session: root\r\n", "/v1/checkpoints/base");
    ASSERT_NO_FATAL_FAILURE(Accept(pin, interface, &connection));
    // Pin ownership has been accepted. The owner has not consumed readiness,
    // even if the device/proactor has already completed its work.
    // Request-backed names must survive peer failure until the endpoint can be
    // published.
    ASSERT_TRUE(interface.is_pending(interface.self));
    EXPECT_EQ(interface.snapshot(interface.self).active_requests, 1u);
    char byte = 0;
    EXPECT_EQ(recv(pin, &byte, 1, MSG_DONTWAIT), -1);
    EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
    if (!stopping) {
      const linger reset = {1, 0};
      ASSERT_EQ(setsockopt(pin, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
                0);
      ASSERT_EQ(close(pin), 0);
      ASSERT_TRUE(iree_notification_await(
          loom_serve_http_server_notification(server_),
          [](void* self) {
            return loom_serve_http_connection_failed(
                static_cast<loom_serve_http_connection_t*>(self));
          },
          connection, iree_infinite_timeout()));
    }
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        interface, idle, stopping ? LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING : 0));
    EXPECT_EQ(interface.telemetry.counters->completed, 1u);
    EXPECT_EQ(interface.telemetry.counters->cancelled, 1u);
    EXPECT_EQ(interface.telemetry.counters->failed, 0u);
    if (stopping) {
      EXPECT_TRUE(Read(pin, /*allow_empty=*/true).empty());
    } else {
      const int resumed = Send(
          Request(Message("user", "abxa") + "," + Message("assistant", first) +
                      "," + Message("user", "b"),
                  2),
          "X-Loom-Session: fork\r\nX-Loom-Checkpoint: base\r\n");
      ASSERT_NO_FATAL_FAILURE(Accept(resumed, interface, &connection));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
      const auto response = Read(resumed);
      EXPECT_EQ(Content(response), Oracle::Complete("abxa" + first + "#b", 2));
      EXPECT_NE(response.find(R"("cached_tokens":5)"), std::string::npos);
      const int released = Send("", "", "/v1/checkpoints/base", "DELETE");
      ASSERT_NO_FATAL_FAILURE(Accept(released, interface, &connection));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
      EXPECT_EQ(Read(released).find("HTTP/1.1 200"), 0u);
      EXPECT_EQ(interface.telemetry.counters->completed, 3u);
      EXPECT_EQ(interface.telemetry.counters->cancelled, 1u);
    }
    interface.set_observer(interface.self, {});
  }
}

TEST_F(RegisteredSourceTextServiceTest,
       FailedTextSourceLeavesPeerAndDiagnosticsUsable) {
  const auto package = iree::testing::MakeTempFilePath("failed_text_source");
  ASSERT_TRUE(std::filesystem::create_directory(package));
  for (const char* file :
       {"prepare.loom", "control.loom", "model.json", "weights.loom"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           package + "/" + file));
  }
  {
    std::ofstream source(package + "/advance.loom");
    source << "this is not a source program\n";
    ASSERT_TRUE(source.good());
  }
  StartManaged();
  loom_serve::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(Read(Send(Declaration("fixture-text", directory_), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Read(Send(Declaration("broken", package, "drained"), "",
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  const int held =
      Send(R"({"model":"broken","content":"ab"})", "", "/tokenize");
  WaitModel("broken", R"("held_requests":1)");
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(held).find("HTTP/1.1 503"), 0u);
  const auto failed = WaitModel("broken", R"("registration":"failed")");
  EXPECT_NE(failed.find("advance.loom"), std::string::npos);
  EXPECT_NE(failed.find(R"("diagnostics":[{)"), std::string::npos);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "ab"), 2)))),
            Oracle::Complete("ab", 2));
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("broken", {});
}

TEST_F(SourceTextServiceTest, PreparationOverlapsAcceptedEpoch) {
  CheckPreparationOverlap(OverlapAction::kComplete);
}

TEST_F(SourceTextServiceTest, DisconnectRetiresOverlappingPreparation) {
  CheckPreparationOverlap(OverlapAction::kDisconnect);
}

TEST_F(SourceTextServiceTest, StoppingRetiresOverlappingPreparation) {
  CheckPreparationOverlap(OverlapAction::kStop);
}

TEST_F(SourceTextServiceTest, StalledOutputPreservesPeerProgressAndResumes) {
  CheckOutputBackpressure(OverlapAction::kComplete);
}

TEST_F(SourceTextServiceTest, DisconnectReclaimsStalledOutput) {
  CheckOutputBackpressure(OverlapAction::kDisconnect);
}

TEST_F(SourceTextServiceTest, StoppingRetiresStalledOutput) {
  CheckOutputBackpressure(OverlapAction::kStop);
}

TEST_F(SourceTextServiceTest, RejectedInputBacklogYieldsToReadyDecode) {
  const auto options = ServiceOptions();
  loom_serve_text_service_t* value = nullptr;
  IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                  host_allocator_, &value));
  auto destroy = [](loom_serve_text_service_t* service) {
    loom_serve_text_service_free(service, IREE_STATUS_OK);
  };
  std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
      value, destroy);
  const auto interface = loom_serve_text_service_interface(value);
  Observe(interface);

  StalledClient stalled;
  ASSERT_NO_FATAL_FAILURE(
      stalled.Send(server_, Request(Message("user", "abxa"), 4)));
  loom_serve_http_connection_t* active = nullptr;
  ASSERT_NO_FATAL_FAILURE(AcceptLoopback(interface, &active));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
    const auto state = interface.snapshot(interface.self);
    return state.active_requests == 1 &&
           !interface.is_pending(interface.self) &&
           std::string(state.phase) == "idle";
  }));
  ASSERT_EQ(interface.telemetry.counters->output_tokens, 1u);
  ASSERT_FALSE(loom_serve_http_connection_can_send(active));

  // Every malformed/oversized input is claimed before another owner turn. Its
  // CPU job can publish readiness, but cannot consume the rejection itself.
  std::vector<int> rejected;
  for (size_t i = 0; i < options.pending_capacity; ++i) {
    rejected.push_back(
        Send(i % 2 ? Request(Message("user", std::string(97, 'a'))) : "{"));
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(rejected.back(), interface, &connection));
  }
  EXPECT_EQ(interface.telemetry.counters->rejected, 0u);
  EXPECT_EQ(interface.snapshot(interface.self).queued_requests,
            rejected.size());

  // Return actual network credit before allowing the owner to run. There is
  // now a ready decode alongside the pending CPU rejection, independent of
  // device speed or callback timing.
  std::string response;
  std::thread reader([&] {
    response = stalled.Read();
    stalled.Close();
  });
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_can_send(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      active, iree_infinite_timeout());
  EXPECT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
    return std::string(interface.snapshot(interface.self).phase) == "execute";
  }));
  EXPECT_LT(interface.telemetry.counters->rejected, rejected.size());
  EXPECT_GT(interface.snapshot(interface.self).queued_requests, 0u);
  EXPECT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
    return !interface.is_pending(interface.self) &&
           !interface.is_active(interface.self) &&
           !interface.snapshot(interface.self).queued_requests;
  }));
  reader.join();
  EXPECT_EQ(Content(response), Oracle::Complete("abxa", 4));
  for (const int client : rejected) {
    EXPECT_EQ(Read(client).find("HTTP/1.1 400"), 0u);
  }
  EXPECT_EQ(interface.telemetry.counters->completed, 1u);
  EXPECT_EQ(interface.telemetry.counters->rejected, rejected.size());
  EXPECT_EQ(interface.telemetry.counters->cancelled, 0u);
  EXPECT_EQ(interface.telemetry.counters->failed, 0u);
  EXPECT_EQ(interface.telemetry.counters->output_tokens, 4u);

  // An idle model has no epoch to yield to. Another complete error backlog
  // must still drain without relying on a device or a new request to wake it.
  rejected.clear();
  for (size_t i = 0; i < options.pending_capacity; ++i) {
    rejected.push_back(Send("{"));
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(rejected.back(), interface, &connection));
  }
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
    return !interface.is_pending(interface.self) &&
           !interface.snapshot(interface.self).queued_requests;
  }));
  for (const int client : rejected) {
    EXPECT_EQ(Read(client).find("HTTP/1.1 400"), 0u);
  }
  EXPECT_EQ(interface.telemetry.counters->completed, 1u);
  EXPECT_EQ(interface.telemetry.counters->rejected,
            2 * options.pending_capacity);
  interface.set_observer(interface.self, {});
}

TEST_F(SourceTextServiceTest, StoppingAccountsForQueuedControlAndCpuBorrowers) {
  auto options = ServiceOptions();
  loom_serve_text_service_t* service = nullptr;
  IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                  host_allocator_, &service));
  auto destroy = [](loom_serve_text_service_t* value) {
    loom_serve_text_service_free(value, IREE_STATUS_OK);
  };
  std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> owner(service,
                                                                      destroy);
  const auto interface = loom_serve_text_service_interface(service);
  auto* notification = loom_serve_http_server_notification(server_);
  interface.set_observer(interface.self,
                         {[](void* payload) {
                            iree_notification_post(
                                static_cast<iree_notification_t*>(payload),
                                IREE_ALL_WAITERS);
                          },
                          notification});
  const auto accept = [&](int client) {
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(client, interface, &connection));
  };
  const int generation =
      Send(Request(Message("user", "ab")), "X-Loom-Session: first\r\n");
  ASSERT_NO_FATAL_FAILURE(accept(generation));
  // CPU completion may already be posted, but only the owner can consume it.
  // Checkpoint and generation claims queue behind that real accepted borrower.
  const int control =
      Send("", "X-Loom-Session: first\r\n", "/v1/checkpoints/base");
  ASSERT_NO_FATAL_FAILURE(accept(control));
  const int queued =
      Send(Request(Message("user", "ab")), "X-Loom-Session: second\r\n");
  ASSERT_NO_FATAL_FAILURE(accept(queued));
  EXPECT_EQ(interface.snapshot(interface.self).queued_requests, 3u);
  bool complete = false;
  while (!complete) {
    const auto ticket = iree_notification_prepare_wait(notification);
    bool progress = false;
    IREE_ASSERT_OK(interface.advance(
        interface.self, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING, &progress));
    const auto state = interface.snapshot(interface.self);
    complete = !interface.is_pending(interface.self) &&
               !state.queued_requests && !state.active_requests;
    if (complete || progress) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  EXPECT_EQ(interface.telemetry.counters->requests, 3u);
  EXPECT_EQ(interface.telemetry.counters->cancelled, 3u);
  EXPECT_EQ(interface.telemetry.counters->completed, 0u);
  EXPECT_TRUE(Read(generation, /*allow_empty=*/true).empty());
  EXPECT_TRUE(Read(control, /*allow_empty=*/true).empty());
  EXPECT_TRUE(Read(queued, /*allow_empty=*/true).empty());
  EXPECT_FALSE(interface.is_active(interface.self));
}

TEST_F(SourceTextServiceTest, TelemetryWrapDoesNotBlockTextOrChangeCounters) {
  Start();
  auto poll = [&](const std::string& cursor) {
    return Read(Send("",
                     "X-Loom-Model: fixture-text\r\n" +
                         (cursor.empty() ? std::string()
                                         : "X-Loom-Cursor: " + cursor + "\r\n"),
                     "/logz", "GET"));
  };
  auto cursor_of = [](const std::string& response) {
    const std::string marker = R"({"event":"cursor","value":")";
    const size_t start = response.find(marker);
    EXPECT_NE(start, std::string::npos) << response;
    if (start == std::string::npos) {
      return std::string();
    }
    const size_t first = start + marker.size();
    return response.substr(first, response.find('"', first) - first);
  };
  const std::string initial = poll("");
  ASSERT_EQ(initial.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(initial.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  const std::string slow = cursor_of(initial);
  std::string fast = slow;
  std::string events;
  auto drain = [&] {
    for (;;) {
      const std::string response = poll(fast);
      if (HasFatalFailure()) {
        return response;
      }
      EXPECT_EQ(response.find(R"("event":"gap")"), std::string::npos);
      events += response;
      const std::string next = cursor_of(response);
      if (next == fast) {
        return response;
      }
      fast = next;
    }
  };
  loom_serve::testing::StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  uint64_t input_tokens = 0;
  for (size_t wave = 0; wave < 10; ++wave) {
    std::vector<int> clients;
    std::vector<std::string> prompts;
    for (size_t row = 0; row < 4; ++row) {
      prompts.push_back("abxa" + std::string((wave * 4 + row) % 3, 'b'));
      clients.push_back(
          Send(Request(Message("user", prompts.back()), 2),
               "X-Loom-Session: row" + std::to_string(row) + "\r\n"));
      input_tokens += prompts.back().size() + 1;
    }
    for (size_t row = 0; row < clients.size(); ++row) {
      EXPECT_EQ(Content(Read(clients[row])), Oracle::Complete(prompts[row], 2));
    }
    drain();
  }
  // Retained continuation counts only appended work, not the shared prefix.
  const std::string first = Content(Read(
      Send(Request(Message("user", "abxa"), 2), "X-Loom-Session: root\r\n")));
  EXPECT_EQ(first, Oracle::Complete("abxa", 2));
  const std::string continued = Read(
      Send(Request(Message("user", "abxa") + "," + Message("assistant", first) +
                       "," + Message("user", "b"),
                   2),
           "X-Loom-Session: root\r\n"));
  EXPECT_EQ(Content(continued), Oracle::Complete("abxa" + first + "#b", 2));
  EXPECT_NE(continued.find(R"("cached_tokens":5)"), std::string::npos);
  input_tokens += 5 + 4;
  EXPECT_EQ(Read(Send("", "X-Loom-Session: root\r\n", "/v1/checkpoints/pin"))
                .find("HTTP/1.1 201"),
            0u);
  EXPECT_EQ(
      Read(Send("", "", "/v1/checkpoints/pin", "DELETE")).find("HTTP/1.1 200"),
      0u);
  EXPECT_EQ(
      Read(Send(Request(Message("user", "a")), "X-Loom-Session: bad space\r\n"))
          .find("HTTP/1.1 400"),
      0u);
  // Recycle every row. The retained root's final lifetime includes both turns,
  // while its old execution slot starts a distinct identity for the newcomer.
  for (size_t row = 0; row < 4; ++row) {
    EXPECT_EQ(Content(Read(Send(Request(Message("user", "abxa"), 2),
                                "X-Loom-Session: replacement" +
                                    std::to_string(row) + "\r\n"))),
              Oracle::Complete("abxa", 2));
    input_tokens += 5;
    drain();
  }
  // HTTP completion precedes asynchronous state trimming. Join maintenance
  // through the owner status before asserting that no further events arrive.
  const std::string idle =
      R"("status":{"phase":"idle","queued_requests":0,"active_requests":0,)";
  const std::string status = WaitModel("fixture-text", idle);
  ASSERT_NE(status.find(idle), std::string::npos) << status;
  const std::string latest = drain();
  const std::string totals =
      R"("requests":49,"completed":48,"rejected":1,"cancelled":0,"failed":0,"input_tokens":)" +
      std::to_string(input_tokens) + R"(,"output_tokens":92,)";
  EXPECT_NE(latest.find(totals), std::string::npos) << latest;
  const size_t endpoint = events.find(R"("session":41,"position":9,)");
  ASSERT_NE(endpoint, std::string::npos) << events;
  const std::string lifetime =
      events.substr(endpoint, events.find('\n', endpoint) - endpoint);
  EXPECT_NE(
      lifetime.find(
          R"("counters":{"requests":2,"completed":2,"rejected":0,"cancelled":0,"failed":0,"input_tokens":9,"output_tokens":4,)"),
      std::string::npos)
      << lifetime;
  for (const char* event :
       {"prepared", "admit", "epoch", "span", "request_stats", "session_begin",
        "session_end", "checkpoint_pin", "checkpoint_release"}) {
    EXPECT_NE(events.find(std::string("\"event\":\"") + event + "\""),
              std::string::npos)
        << event;
  }
  // No prompt text enters observation records. Slow and new readers resume
  // independently after overwrite while current totals remain exact.
  EXPECT_EQ(events.find("abxa"), std::string::npos);
  const std::string resumed = poll(slow);
  EXPECT_NE(resumed.find(R"({"event":"gap","from":1,)"), std::string::npos);
  EXPECT_NE(resumed.find(totals), std::string::npos);
  EXPECT_NE(poll("").find(R"("partial":true)"), std::string::npos);
  const std::string final_poll = poll(fast);
  EXPECT_EQ(cursor_of(final_poll), fast) << final_poll;
  loom_serve_http_server_request_stop(server_);
  IREE_EXPECT_OK(http_.Join());
}

TEST_F(SourceTextServiceTest, IsolatedEpochsPreserveRetainedHttpHistory) {
  CheckScheduledResponses(LOOM_SERVE_TEXT_SCHEDULE_ISOLATED);
}

TEST_F(SourceTextServiceTest, MatchedEpochsPreserveRetainedHttpHistory) {
  CheckScheduledResponses(LOOM_SERVE_TEXT_SCHEDULE_MATCHED);
}

TEST_F(SourceTextServiceTest, CancelledPreparationRetainsPartialOwnership) {
  const auto before = loom_serve_device_memory_statistics(device_);
  auto options = ModelOptions();
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_load_t* load = nullptr;
  IREE_ASSERT_OK(loom_serve_text_load_allocate(
      device_, &options,
      [](void* payload) {
        iree_notification_post(static_cast<iree_notification_t*>(payload),
                               IREE_ALL_WAITERS);
      },
      &notification, host_allocator_, &load));
  bool complete = false;
  IREE_ASSERT_OK(loom_serve_text_load_advance(load, 0, &complete));
  ASSERT_FALSE(complete);
  iree_status_t status = iree_ok_status();
  while (!complete) {
    const auto ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_text_load_advance(
        load, LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL, &complete);
    if (complete) {
      iree_notification_cancel_wait(&notification);
    } else {
      IREE_ASSERT_OK(status);
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
  auto* partial = loom_serve_text_load_take_model(load);
  ASSERT_NE(partial, nullptr);
  loom_serve_text_load_free(load);
  loom_serve_text_model_retire(partial, nullptr, nullptr);
  ASSERT_TRUE(loom_serve_text_model_is_retired(partial));
  IREE_ASSERT_OK(loom_serve_text_model_free_retired(partial));
  iree_notification_deinitialize(&notification);
  const auto after = loom_serve_device_memory_statistics(device_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
}

TEST_F(SourceTextServiceTest, TokenizerFailureRetiresCompiledModel) {
  const auto before = loom_serve_device_memory_statistics(device_);
  auto options = ModelOptions();
  options.tokenizer_path = IREE_SV("absent-tokenizer.json");
  loom_serve_text_model_t* model = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_serve_text_model_allocate(
                            device_, &options, host_allocator_, &model));
  EXPECT_EQ(model, nullptr);
  const auto after = loom_serve_device_memory_statistics(device_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);

  // The failed tokenizer belongs to one load, not the application task pool or
  // another model's activation and execution timeline.
  auto* row = loom_serve_text_model_row(model_, 0);
  const int32_t token = Oracle::Id('a');
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
  Oracle reference;
  reference.Consume("a");
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
}

TEST_F(SourceTextServiceTest, FailedLinkRetiresAcceptedInitialization) {
  const auto package = iree::testing::MakeTempFilePath("failed_text_link");
  ASSERT_TRUE(std::filesystem::create_directory(package));
  for (const char* file : {"prepare.loom", "advance.loom", "model.json",
                           "weights.loom", "tokenizer.json"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           package + "/" + file));
  }
  {
    std::ifstream input(directory_ + "/control.loom");
    std::string source((std::istreambuf_iterator<char>(input)), {});
    const auto offset = source.find("import(\"runner\")");
    ASSERT_NE(offset, std::string::npos);
    source.replace(offset, std::string("import(\"runner\")").size(),
                   "import(\"missing\")");
    std::ofstream output(package + "/control.loom");
    output << source;
    ASSERT_TRUE(output.good());
  }
  const auto before = loom_serve_device_memory_statistics(device_);
  auto options = ModelOptions();
  options.sources.base_directory = iree_make_cstring_view(package.c_str());
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  const auto wake = [](void* payload) {
    iree_notification_post(static_cast<iree_notification_t*>(payload),
                           IREE_ALL_WAITERS);
  };
  loom_serve_text_load_t* load = nullptr;
  IREE_ASSERT_OK(loom_serve_text_load_allocate(
      device_, &options, wake, &notification, host_allocator_, &load));
  bool complete = false;
  iree_status_t status = iree_ok_status();
  while (!complete) {
    const auto ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_text_load_advance(load, 0, &complete);
    if (complete) {
      iree_notification_cancel_wait(&notification);
    } else {
      IREE_ASSERT_OK(status);
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  EXPECT_NE(
      iree::Status::ToString(status).find("requires missing module 'missing'"),
      std::string::npos);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  auto* partial = loom_serve_text_load_take_model(load);
  loom_serve_text_load_free(load);
  loom_serve_text_model_retire(partial, wake, &notification);
  EXPECT_TRUE(iree_notification_await(
      &notification,
      [](void* payload) {
        return loom_serve_text_model_is_retired(
            static_cast<loom_serve_text_model_t*>(payload));
      },
      partial, iree_infinite_timeout()));
  IREE_ASSERT_OK(loom_serve_text_model_free_retired(partial));
  iree_notification_deinitialize(&notification);
  const auto after = loom_serve_device_memory_statistics(device_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
  auto* row = loom_serve_text_model_row(model_, 0);
  const int32_t token = Oracle::Id('a');
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
  Oracle reference;
  reference.Consume("a");
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
}

TEST_F(SourceTextServiceTest,
       HttpRetainsRewindsAndRejectsWithoutLosingHistory) {
  Start();
  const std::string base = "abxa";
  const std::string first_response =
      Read(Send(Request(Message("user", base), 2), "X-Loom-Session: root\r\n"));
  const std::string first = Content(first_response);
  ASSERT_EQ(first, Oracle::Complete(base, 2));
  EXPECT_NE(first_response.find("\"finish_reason\":\"length\""),
            std::string::npos);
  const std::string pinned =
      Read(Send("", "X-Loom-Session: root\r\n", "/v1/checkpoints/base"));
  ASSERT_EQ(pinned.find("HTTP/1.1 201"), 0u) << pinned;
  const std::string history =
      Message("user", base) + "," + Message("assistant", first);
  std::vector<int> clients;
  const std::vector<std::string> suffixes = {"a", "bx", "xaba"};
  for (size_t i = 0; i < suffixes.size(); ++i) {
    clients.push_back(
        Send(Request(history + "," + Message("user", suffixes[i])),
             "X-Loom-Session: fork" + std::to_string(i) +
                 "\r\nX-Loom-Checkpoint: base\r\n"));
  }
  for (size_t i = 0; i < clients.size(); ++i) {
    const std::string response = Read(clients[i]);
    EXPECT_EQ(Content(response),
              Oracle::Complete(base + first + "#" + suffixes[i], 16));
    EXPECT_NE(response.find("\"cached_tokens\":5"), std::string::npos)
        << response;
  }
  const std::string bad =
      Read(Send(Request(history + "," + Message("user", "?")),
                "X-Loom-Session: root\r\n"));
  EXPECT_EQ(bad.find("HTTP/1.1 400"), 0u) << bad;
  const std::string resumed_request =
      Request(history + "," + Message("user", "bax"));
  const std::string resumed =
      Read(Send(resumed_request, "X-Loom-Session: root\r\n"));
  EXPECT_EQ(Content(resumed), Oracle::Complete(base + first + "#bax", 16));
  EXPECT_NE(resumed.find("\"cached_tokens\":5"), std::string::npos) << resumed;
  const std::string replay = Read(Send(resumed_request));
  EXPECT_EQ(Content(replay), Content(resumed));
  const std::string rewind =
      Read(Send(Request(history + "," + Message("user", "xx")),
                "X-Loom-Session: root\r\nX-Loom-Checkpoint: base\r\n"));
  EXPECT_EQ(Content(rewind), Oracle::Complete(base + first + "#xx", 16));
  const std::string deleted =
      Read(Send("", "", "/v1/checkpoints/base", "DELETE"));
  EXPECT_EQ(deleted.find("HTTP/1.1 200"), 0u) << deleted;

  const std::string terminal_response = Read(
      Send(Request(Message("user", "abx")), "X-Loom-Session: terminal\r\n"));
  const std::string terminal = Content(terminal_response);
  EXPECT_EQ(terminal, Oracle::Complete("abx", 16));
  EXPECT_NE(terminal_response.find("\"finish_reason\":\"stop\""),
            std::string::npos);
  const std::string next =
      Request(Message("user", "abx") + "," + Message("assistant", terminal) +
              "," + Message("user", "ba"));
  const std::string continued =
      Read(Send(next, "X-Loom-Session: terminal\r\n"));
  EXPECT_EQ(Content(continued), Oracle::Complete("abx" + terminal + "#ba", 16));
  EXPECT_NE(continued.find("\"cached_tokens\":11"), std::string::npos)
      << continued;
  EXPECT_EQ(Content(Read(Send(next))), Content(continued));
}

TEST_F(SourceTextServiceTest,
       CancelledQueuedRequestReturnsClaimAndSessionName) {
  auto options = ServiceOptions();
  options.heartbeat_interval = iree_make_duration_ms(1);
  loom_serve_text_service_t* value = nullptr;
  IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                  host_allocator_, &value));
  auto destroy = [](loom_serve_text_service_t* service) {
    loom_serve_text_service_free(service, IREE_STATUS_OK);
  };
  std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
      value, destroy);
  auto interface = loom_serve_text_service_interface(value);
  EXPECT_EQ(interface.capacity_domain,
            loom_serve_device_residency_cache(device_));
  Observe(interface);
  auto* notification = loom_serve_http_server_notification(server_);
  uint64_t cursor = 0;
  uint64_t reported_queued = UINT64_MAX;
  auto read_snapshots = [&] {
    const uint64_t head =
        loom_serve_telemetry_stream_head(interface.telemetry.stream);
    while (cursor < head) {
      loom_serve_telemetry_record_t record;
      ASSERT_EQ(loom_serve_telemetry_stream_read(interface.telemetry.stream,
                                                 &cursor, &record),
                LOOM_SERVE_TELEMETRY_READ_RECORD);
      if (record.event == &LOOM_SERVE_TELEMETRY_SNAPSHOT) {
        reported_queued = record.data.snapshot.queued;
      }
    }
  };
  struct Claim {
    // Actual HTTP claim held through service admission or cancellation.
    loom_serve_http_connection_t* connection = nullptr;
    // Immutable request storage borrowed from that claim.
    const loom_serve_http_request_t* request = nullptr;
  } claims[3];
  for (size_t i = 0; i < IREE_ARRAYSIZE(claims); ++i) {
    const std::string name = i ? "tail" + std::to_string(i) : "reusable";
    const int client = Send(Request(Message("user", "abx")),
                            "X-Loom-Session: " + name + "\r\n");
    auto& claim = claims[i];
    IREE_ASSERT_OK(
        http_.TakeRequest(client, &claim.connection, &claim.request));
    const linger reset = {1, 0};
    ASSERT_EQ(setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
              0);
    ASSERT_EQ(close(client), 0);
    iree_notification_await(
        notification,
        [](void* self) {
          return loom_serve_http_connection_failed(
              static_cast<loom_serve_http_connection_t*>(self));
        },
        claim.connection, iree_infinite_timeout());
  }
  // Failure can arrive after the router claims a request but before the model
  // accepts it. A single-worker turn cannot run the preparation it just queued:
  // that borrowed head must survive, while both unborrowed tails retire now.
  loom_serve::testing::text::OwnerTurn::Run(
      loom_serve_host_executor(host_), [&] {
        for (const auto& claim : claims) {
          IREE_ASSERT_OK(interface.accept(interface.self, claim.connection,
                                          claim.request, 0));
        }
        EXPECT_EQ(interface.snapshot(interface.self).queued_requests, 3u);
        bool progress = false;
        IREE_ASSERT_OK(interface.advance(interface.self, 0, &progress));
        // The first heartbeat precedes borrower collection. It must include
        // arrivals after preparation began, not just the initial FIFO head.
        read_snapshots();
        EXPECT_EQ(reported_queued, 3u);
        EXPECT_TRUE(progress);
        EXPECT_GE(interface.telemetry.counters->cancelled, 2u);
        EXPECT_EQ(interface.telemetry.counters->cancelled, 2u);
        EXPECT_EQ(interface.snapshot(interface.self).queued_requests, 1u);
        EXPECT_TRUE(interface.is_pending(interface.self));
      });
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, [&] {
    read_snapshots();
    return reported_queued == 0 &&
           !interface.snapshot(interface.self).queued_requests;
  }));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      interface, [&] { return !interface.is_pending(interface.self); },
      LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING));
  const auto cancelled = interface.snapshot(interface.self);
  EXPECT_EQ(cancelled.queued_requests, 0u);
  EXPECT_EQ(cancelled.oldest_queued_age, 0);
  EXPECT_EQ(cancelled.oldest_active_age, 0);
  EXPECT_FALSE(interface.is_pending(interface.self));
  EXPECT_FALSE(interface.is_active(interface.self));
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 256u);
  interface.set_observer(interface.self, {});
  const auto counters = *interface.telemetry.counters;
  EXPECT_EQ(counters.requests, 3u);
  EXPECT_EQ(counters.cancelled, 3u);
  EXPECT_EQ(counters.completed + counters.rejected + counters.failed, 0u);
  EXPECT_EQ(counters.input_tokens + counters.output_tokens + counters.batches,
            0u);
  service.reset();
  Start();
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "abx")),
                              "X-Loom-Session: reusable\r\n"))),
            Oracle::Complete("abx", 16));
}

}  // namespace
