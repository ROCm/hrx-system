// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SERVICE_TEST_FIXTURE_H_
#define LOOM_SERVE_SERVICE_TEXT_SERVICE_TEST_FIXTURE_H_

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/router.h"
#include "loom/serve/service/text/input.h"
#include "loom/serve/service/text/owner_turn_test_util.h"
#include "loom/serve/service/text/service.h"
#include "loom/serve/service/text/source_test_oracle.h"
#include "loom/serve/telemetry/stream.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/stalled_http_client.h"
#include "loom/serve/testing/stalled_stderr.h"
#include "loom/serve/transport/http/listener.h"

namespace loom_serve::testing::text {

// Credit-controlled client carrying the ordinary text routing headers.
class StalledClient : public loom_serve::testing::StalledHttpClient {
 public:
  void Send(loom_serve_http_server_t* server, const std::string& body,
            const char* target = "/v1/chat/completions",
            const char* method = "POST") {
    StalledHttpClient::Send(
        server, target, body, method,
        "X-Loom-Session: stalled\r\nX-Loom-Model: fixture-text\r\n");
  }
};

class SourceTextServiceTest : public ::testing::Test {
 protected:
  // Whether this fixture owns a standalone model before the test body runs.
  // Registry-owned tests override this because their production path creates
  // the model from the submitted declaration.
  virtual bool NeedsStandaloneModel() const { return true; }

  // Document ownership covers every asynchronous native load in this fixture.
  loom_serve::testing::ModelDocument document_;

  void SetUp() override;

  void TearDown() override;

  void WriteWeights(uint32_t bias) {
    std::string header =
        R"({"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream file(weights_, std::ios::binary);
    uint8_t bytes[8];
    iree_unaligned_store_le_u64(bytes, header.size());
    file.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    file.write(header.data(), header.size());
    iree_unaligned_store_le_u32(bytes, bias);
    file.write(reinterpret_cast<const char*>(bytes), 4);
    file.close();
    ASSERT_TRUE(file.good());
  }

  virtual loom_serve_text_options_t ModelOptions() const {
    return {
        .sources = document_.value().sources,
        .prefill_capacity = 32,
        .context_capacity = 96,
        .pool_capacity = 256,
        .epoch_count = IREE_ARRAYSIZE(shapes_),
        .epoch_shapes = shapes_,
        .kernel_sanitizer = {.type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
                             .structure_size =
                                 sizeof(loomc_sanitizer_options_t)},
        .weights_path = iree_make_cstring_view(weights_.c_str()),
        .tokenizer_path = IREE_SV("tokenizer.json"),
        .row_count = 4,
        .checkpoint_capacity = 2,
    };
  }

  virtual loom_serve_text_service_options_t ServiceOptions() const {
    return {
        .name = IREE_SV("fixture-text"),
        .row_count = 4,
        .chunk_size = 3,
        .default_max_tokens = 16,
        .pending_capacity = 16,
        .continuation_epochs = 1,
    };
  }

  void Start(
      iree_host_size_t chunk_size = 3,
      loom_serve_text_schedule_mode_t mode = LOOM_SERVE_TEXT_SCHEDULE_PACKED) {
    http_.Start([this, chunk_size, mode] {
      auto options = ServiceOptions();
      options.chunk_size = chunk_size;
      options.schedule_mode = mode;
      loom_serve_text_service_t* service = nullptr;
      iree_status_t status = loom_serve_text_service_allocate(
          model_, &options, nullptr, host_allocator_, &service);
      if (iree_status_is_ok(status)) {
        const auto interface = loom_serve_text_service_interface(service);
        loom_serve_registry_t* registry = nullptr;
        status = loom_serve_registry_allocate(1, &interface, 0, {},
                                              host_allocator_, &registry);
        if (iree_status_is_ok(status)) {
          status = loom_serve_router_run(
              loom_serve_host_executor(host_), server_, registry,
              {
                  .request_quota = options.row_count,
                  .holding_capacity =
                      loom_serve_http_server_connection_capacity(server_) / 2,
                  .resources = {.device_group = group_},
              },
              host_allocator_);
        }
        status = iree_status_join(status, loom_serve_registry_free(registry));
      }
      loom_serve_text_service_free(service, iree_status_code(status));
      return status;
    });
  }

  void StartManaged(loom_serve_device_idle_policy_t idle = {}) {
    ASSERT_EQ(model_, nullptr);
    http_.Start([this, idle]() -> iree_status_t {
      const loom_serve_model_registration_options_t options = {
          .device_group = group_, .heartbeat_interval = 0};
      loom_serve_registry_t* registry = nullptr;
      IREE_RETURN_IF_ERROR(loom_serve_registry_allocate(
          0, nullptr, 3, loom_serve_model_registration_factory(&options),
          host_allocator_, &registry));
      iree_status_t status = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {
              .request_quota = 16,
              .holding_capacity = 8,
              .resources = {.device_group = group_},
              .idle = idle,
          },
          host_allocator_);
      return iree_status_join(status, loom_serve_registry_free(registry));
    });
  }

  std::string Declaration(const std::string& name, const std::string& source,
                          const char* admission = "open",
                          const std::string& definition_fields = "") {
    return "{\"name\":\"" + name + "\",\"base_directory\":\"" +
           std::filesystem::absolute(source).string() + "\",\"admission\":\"" +
           admission +
           "\",\"definition\":{\"kind\":\"text\",\"prepare\":\"prepare.loom\","
           "\"control\":\"control.loom\",\"sources\":[\"advance.loom\"],"
           "\"weights\":\"" +
           weights_ + "\",\"tokenizer\":\"" +
           std::filesystem::absolute(tokenizer_).string() +
           "\",\"rows\":4,\"context_capacity\":96,"
           "\"pool_capacity\":256,\"prefill_capacity\":32,"
           "\"checkpoint_capacity\":2,\"draft_depth\":0" +
           definition_fields + "}}";
  }

  std::string WaitModel(const std::string& name, const std::string& marker) {
    while (!http_.failed() && !HasFatalFailure()) {
      const auto listing = Read(Send("", "", "/v1/status", "GET"));
      const size_t begin = listing.find("\"id\":\"" + name + "\"");
      const size_t end = listing.find("{\"object\":\"model\"", begin);
      const auto state =
          begin == std::string::npos
              ? std::string()
              : listing.substr(begin,
                               end == std::string::npos ? end : end - begin);
      if (marker.empty() ? state.empty()
                         : state.find(marker) != std::string::npos) {
        return state;
      }
    }
    return {};
  }

  int Send(const std::string& body, const std::string& headers = "",
           const std::string& target = "/v1/chat/completions",
           const std::string& method = "POST") {
    if (http_.failed() || HasFatalFailure()) {
      return -1;
    }
    const std::string request =
        method + " " + target +
        " HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\n" +
        headers + "Content-Length: " + std::to_string(body.size()) +
        "\r\n\r\n" + body;
    int client = -1;
    [&] { IREE_ASSERT_OK(http_.Send(request, &client)); }();
    return client;
  }

  std::string Read(int client, bool allow_empty = false) {
    std::string response;
    if (http_.failed() || HasFatalFailure()) {
      if (client >= 0) {
        EXPECT_EQ(close(client), 0);
      }
      return response;
    }
    [&] { IREE_ASSERT_OK(http_.Read(client, allow_empty, &response)); }();
    return response;
  }

  static std::string Message(const std::string& role,
                             const std::string& content) {
    return "{\"role\":\"" + role + "\",\"content\":\"" + content + "\"}";
  }

  static std::string Request(const std::string& messages, size_t limit = 16) {
    return "{\"model\":\"fixture-text\",\"stream\":true,\"stream_options\":{"
           "\"include_usage\":true},\"max_tokens\":" +
           std::to_string(limit) + ",\"messages\":[" + messages + "]}";
  }

  static std::string Content(const std::string& response) {
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
    EXPECT_NE(response.find("data: [DONE]"), std::string::npos) << response;
    std::string result;
    const std::string marker = "\"content\":\"";
    size_t cursor = 0;
    while ((cursor = response.find(marker, cursor)) != std::string::npos) {
      cursor += marker.size();
      const size_t end = response.find('"', cursor);
      EXPECT_NE(end, std::string::npos);
      if (end == std::string::npos) {
        break;
      }
      result.append(response, cursor, end - cursor);
      cursor = end + 1;
    }
    EXPECT_EQ(result.find_first_not_of("abx"), std::string::npos);
    return result;
  }

  void CheckScheduledResponses(loom_serve_text_schedule_mode_t mode) {
    Start(3, mode);
    const std::vector<std::string> prompts = {"abxa", "abxabba", "xab"};
    std::vector<int> clients;
    for (size_t i = 0; i < prompts.size(); ++i) {
      clients.push_back(
          Send(Request(Message("user", prompts[i]), 4),
               "X-Loom-Session: row" + std::to_string(i) + "\r\n"));
    }
    for (size_t i = 0; i < clients.size(); ++i) {
      const std::string first = Content(Read(clients[i]));
      EXPECT_EQ(first, Oracle::Complete(prompts[i], 4));
      const std::string next = Request(Message("user", prompts[i]) + "," +
                                           Message("assistant", first) + "," +
                                           Message("user", "bax"),
                                       4);
      EXPECT_EQ(Content(Read(Send(
                    next, "X-Loom-Session: row" + std::to_string(i) + "\r\n"))),
                Oracle::Complete(prompts[i] + first + "#bax", 4));
    }
    const auto log =
        Read(Send("", "X-Loom-Model: fixture-text\r\n", "/logz", "GET"));
    EXPECT_NE(log.find(R"("token_slots":null)"), std::string::npos) << log;
  }

  void Observe(const loom_serve_http_service_t& interface) {
    interface.set_observer(interface.self,
                           {[](void* payload) {
                              iree_notification_post(
                                  static_cast<iree_notification_t*>(payload),
                                  IREE_ALL_WAITERS);
                            },
                            loom_serve_http_server_notification(server_)});
  }

  void Accept(int client, const loom_serve_http_service_t& interface,
              loom_serve_http_connection_t** out_connection) {
    ASSERT_FALSE(HasFatalFailure());
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(http_.TakeRequest(client, out_connection, &request));
    IREE_ASSERT_OK(
        interface.accept(interface.self, *out_connection, request, 0));
  }

  void AcceptLoopback(const loom_serve_http_service_t& interface,
                      loom_serve_http_connection_t** out_connection) {
    auto* notification = loom_serve_http_server_notification(server_);
    for (;;) {
      const auto ticket = iree_notification_prepare_wait(notification);
      const loom_serve_http_request_t* request = nullptr;
      auto* connection = loom_serve_http_server_take_request(server_, &request);
      if (connection) {
        iree_notification_cancel_wait(notification);
        *out_connection = connection;
        IREE_ASSERT_OK(
            interface.accept(interface.self, connection, request, 0));
        return;
      }
      iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }

  template <typename Predicate>
  void AdvanceUntil(const loom_serve_http_service_t& interface, Predicate ready,
                    loom_serve_http_service_flags_t flags = 0) {
    auto* notification = loom_serve_http_server_notification(server_);
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      IREE_ASSERT_OK(interface.advance(interface.self, flags, &progress));
      if (ready() || progress) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }

  enum class OverlapAction { kComplete, kDisconnect, kStop };

  void CheckOutputBackpressure(OverlapAction action) {
    Start();
    StalledClient stalled;
    ASSERT_NO_FATAL_FAILURE(
        stalled.Send(server_, Request(Message("user", "abxa"), 4)));
    const std::string held_state =
        R"("phase":"idle","queued_requests":0,"active_requests":1)";
    WaitModel("fixture-text", held_state);
    auto expect_counts = [&](uint64_t completed, uint64_t cancelled,
                             uint64_t outputs) {
      const std::string metrics = Read(Send("", "", "/metrics", "GET"));
      for (const auto& [name, value] :
           std::vector<std::pair<std::string, uint64_t>>{
               {"completed", completed},
               {"cancelled", cancelled},
               {"failed", 0},
               {"output_tokens", outputs}}) {
        EXPECT_NE(metrics.find("\nloom_serve_server_" + name + "_total " +
                               std::to_string(value) + "\n"),
                  std::string::npos)
            << metrics;
      }
    };
    // The initial SSE header occupies the single carrier slot. The model can
    // still fill its one row packet, but cannot generate unbounded output.
    expect_counts(0, 0, 1);
    const auto active =
        Read(Send("", "X-Loom-Model: fixture-text\r\n", "/v1/requests", "GET"));
    EXPECT_NE(active.find(R"("data":[{"id":1,"phase":"active","session":1,)"),
              std::string::npos)
        << active;
    EXPECT_NE(active.find(R"("output_tokens":1,)"), std::string::npos);
    const auto sessions =
        Read(Send("", "X-Loom-Model: fixture-text\r\n", "/v1/sessions", "GET"));
    EXPECT_NE(sessions.find(R"("name":"stalled","request":1,"storage":"row",)"),
              std::string::npos)
        << sessions;
    const std::string prompt = "baxab";
    const std::string first = Content(Read(
        Send(Request(Message("user", prompt), 2), "X-Loom-Session: peer\r\n")));
    EXPECT_EQ(first, Oracle::Complete(prompt, 2));
    const std::string followup = Read(Send(
        Request(Message("user", prompt) + "," + Message("assistant", first) +
                    "," + Message("user", "b"),
                2),
        "X-Loom-Session: peer\r\n"));
    EXPECT_EQ(Content(followup), Oracle::Complete(prompt + first + "#b", 2));
    EXPECT_NE(followup.find(R"("cached_tokens":6)"), std::string::npos);
    EXPECT_EQ(Content(Read(Send(Request(Message("user", "abxa"), 2),
                                "X-Loom-Session: third\r\n"))),
              Oracle::Complete("abxa", 2));
    WaitModel("fixture-text", held_state);
    // All six new outputs belong to the independent TCP peers. The held row
    // has not advanced while they execute, retire and retain their histories.
    expect_counts(3, 0, 7);

    if (action == OverlapAction::kStop) {
      loom_serve_http_server_request_stop(server_);
      IREE_EXPECT_OK(http_.Join());
      ASSERT_NO_FATAL_FAILURE(stalled.Close());
      // The service releases unused reservations; the outer model owner still
      // owns consumed row storage until its own retirement.
      EXPECT_EQ(loom_serve_text_model_pool_usage(model_).pending, 0u);
      IREE_ASSERT_OK(loom_serve_text_model_free(model_));
      model_ = nullptr;
      return;
    }
    std::string continuation;
    if (action == OverlapAction::kComplete) {
      const std::string response = Content(stalled.Read());
      EXPECT_EQ(response, Oracle::Complete("abxa", 4));
      continuation = Message("user", "abxa") + "," +
                     Message("assistant", response) + "," +
                     Message("user", "b");
    }
    ASSERT_NO_FATAL_FAILURE(stalled.Close());
    WaitModel("fixture-text",
              R"("phase":"idle","queued_requests":0,"active_requests":0)");
    expect_counts(action == OverlapAction::kComplete ? 4 : 3,
                  action == OverlapAction::kDisconnect ? 1 : 0,
                  action == OverlapAction::kComplete ? 10 : 7);
    const bool retained = action == OverlapAction::kComplete;
    const std::string response =
        Read(Send(Request(retained ? continuation : Message("user", "abxa"), 2),
                  "X-Loom-Session: stalled\r\n"));
    EXPECT_EQ(
        Content(response),
        Oracle::Complete(
            retained ? "abxa" + Oracle::Complete("abxa", 4) + "#b" : "abxa",
            2));
    EXPECT_NE(response.find(retained ? R"("cached_tokens":7)"
                                     : R"("cached_tokens":0)"),
              std::string::npos);
  }

  void CheckPreparationOverlap(OverlapAction action) {
    auto options = ServiceOptions();
    loom_serve_text_service_t* value = nullptr;
    IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                    host_allocator_, &value));
    auto destroy = [](loom_serve_text_service_t* service) {
      loom_serve_text_service_free(service, IREE_STATUS_OK);
    };
    std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
        value, destroy);
    const auto interface = loom_serve_text_service_interface(value);
    auto* notification = loom_serve_http_server_notification(server_);
    Observe(interface);
    const std::string first_prompt = "abxabbax";
    const int first = Send(Request(Message("user", first_prompt), 4),
                           "X-Loom-Session: first\r\n");
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(first, interface, &connection));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        interface,
        [&] {
          return std::string(interface.snapshot(interface.self).phase) ==
                 "execute";
        },
        0));
    // Keep the first accepted epoch owned even if the hardware has already
    // signaled readiness. The second prompt must acquire a separate CPU borrow
    // before the owner consumes that epoch; no timing or queue FIFO is assumed.
    const std::string second_prompt = "baxab";
    const int second = Send(Request(Message("user", second_prompt), 4),
                            "X-Loom-Session: second\r\n");
    ASSERT_NO_FATAL_FAILURE(Accept(second, interface, &connection));
    EXPECT_EQ(interface.telemetry.counters->batches, 0u);
    EXPECT_EQ(interface.snapshot(interface.self).queued_requests, 1u);
    // Metadata stays owner-readable while preparation borrows the second
    // prompt and the device still owns the first epoch's rows.
    loom_serve_http_request_snapshot_t queued_snapshot = {};
    ASSERT_TRUE(
        interface.inspection.request_at(interface.self, 0, &queued_snapshot));
    EXPECT_EQ(queued_snapshot.id, 2u);
    EXPECT_STREQ(queued_snapshot.phase, "queued");
    uint64_t active_requests = 0;
    for (iree_host_size_t i = 1;; ++i) {
      loom_serve_http_request_snapshot_t snapshot = {};
      if (!interface.inspection.request_at(interface.self, i, &snapshot)) {
        break;
      }
      if (!snapshot.id) {
        continue;
      }
      ++active_requests;
      EXPECT_EQ(snapshot.id, 1u);
      EXPECT_EQ(snapshot.session, 1u);
      EXPECT_EQ(snapshot.counters.batches, 0u);
    }
    EXPECT_EQ(active_requests, 1u);
    if (action == OverlapAction::kDisconnect) {
      const linger reset = {1, 0};
      ASSERT_EQ(
          setsockopt(second, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)), 0);
      ASSERT_EQ(close(second), 0);
      iree_notification_await(
          notification,
          [](void* self) {
            return loom_serve_http_connection_failed(
                static_cast<loom_serve_http_connection_t*>(self));
          },
          connection, iree_infinite_timeout());
    }
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        interface,
        [&] {
          return !interface.is_pending(interface.self) &&
                 !interface.is_active(interface.self) &&
                 !interface.snapshot(interface.self).queued_requests;
        },
        action == OverlapAction::kStop ? LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING
                                       : 0));
    EXPECT_EQ(interface.telemetry.counters->requests, 2u);
    if (action == OverlapAction::kStop) {
      EXPECT_EQ(interface.telemetry.counters->cancelled, 2u);
      EXPECT_EQ(Read(first, /*allow_empty=*/true).find("data: [DONE]"),
                std::string::npos);
      EXPECT_TRUE(Read(second, /*allow_empty=*/true).empty());
    } else {
      EXPECT_EQ(Content(Read(first)), Oracle::Complete(first_prompt, 4));
      EXPECT_EQ(interface.telemetry.counters->completed,
                action == OverlapAction::kComplete ? 2u : 1u);
      if (action == OverlapAction::kDisconnect) {
        EXPECT_EQ(interface.telemetry.counters->cancelled, 1u);
      } else {
        EXPECT_EQ(Content(Read(second)), Oracle::Complete(second_prompt, 4));
        uint64_t cursor = 0;
        uint64_t prepared = UINT64_MAX, first_epoch = UINT64_MAX;
        const auto head =
            loom_serve_telemetry_stream_head(interface.telemetry.stream);
        while (cursor < head) {
          loom_serve_telemetry_record_t record;
          ASSERT_EQ(loom_serve_telemetry_stream_read(interface.telemetry.stream,
                                                     &cursor, &record),
                    LOOM_SERVE_TELEMETRY_READ_RECORD);
          if (std::string(record.event->name) == "prepared" &&
              record.request == 2) {
            prepared = cursor;
          }
          if (std::string(record.event->name) == "epoch") {
            first_epoch = std::min(first_epoch, cursor);
          }
        }
        ASSERT_NE(first_epoch, UINT64_MAX);
        EXPECT_LT(prepared, first_epoch);
      }
    }
    interface.set_observer(interface.self, {});
  }

  void CheckRetentionCancellation(OverlapAction action) {
    auto options = ServiceOptions();
    options.session_cache_capacity = 1;
    loom_serve_text_service_t* value = nullptr;
    IREE_ASSERT_OK(loom_serve_text_service_allocate(model_, &options, nullptr,
                                                    host_allocator_, &value));
    auto destroy = [](loom_serve_text_service_t* service) {
      loom_serve_text_service_free(service, IREE_STATUS_OK);
    };
    std::unique_ptr<loom_serve_text_service_t, decltype(destroy)> service(
        value, destroy);
    const auto interface = loom_serve_text_service_interface(value);
    auto* notification = loom_serve_http_server_notification(server_);
    Observe(interface);
    auto idle = [&] {
      return !interface.is_pending(interface.self) &&
             !interface.is_active(interface.self) &&
             !interface.snapshot(interface.self).queued_requests;
    };
    loom_serve_http_connection_t* connection = nullptr;
    const auto first = Oracle::Complete("abxa", 2);
    for (size_t i = 0; i < 4; ++i) {
      const int client =
          Send(Request(Message("user", "abxa"), 2),
               "X-Loom-Session: conversation" + std::to_string(i) + "\r\n");
      ASSERT_NO_FATAL_FAILURE(Accept(client, interface, &connection));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
      EXPECT_EQ(Content(Read(client)), first);
    }
    const int replacement = Send(Request(Message("user", "bax"), 2),
                                 "X-Loom-Session: replacement\r\n");
    ASSERT_NO_FATAL_FAILURE(Accept(replacement, interface, &connection));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        interface,
        [&] {
          return std::string(interface.snapshot(interface.self).phase) ==
                 "retain";
        },
        0));
    // Pin submission has accepted ownership, even if hardware has already
    // signaled it. Cancellation happens before the owner consumes readiness.
    if (action == OverlapAction::kDisconnect) {
      const linger reset = {1, 0};
      ASSERT_EQ(
          setsockopt(replacement, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
          0);
      ASSERT_EQ(close(replacement), 0);
      iree_notification_await(
          notification,
          [](void* self) {
            return loom_serve_http_connection_failed(
                static_cast<loom_serve_http_connection_t*>(self));
          },
          connection, iree_infinite_timeout());
    }
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        interface, idle,
        action == OverlapAction::kStop ? LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING
                                       : 0));
    EXPECT_EQ(interface.telemetry.counters->cancelled, 1u);
    if (action == OverlapAction::kStop) {
      EXPECT_TRUE(Read(replacement, /*allow_empty=*/true).empty());
    } else {
      const int resumed = Send(
          Request(Message("user", "abxa") + "," + Message("assistant", first) +
                      "," + Message("user", "b"),
                  2),
          "X-Loom-Session: conversation0\r\n");
      ASSERT_NO_FATAL_FAILURE(Accept(resumed, interface, &connection));
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface, idle));
      const auto response = Read(resumed);
      EXPECT_EQ(Content(response), Oracle::Complete("abxa" + first + "#b", 2));
      EXPECT_NE(response.find("\"cached_tokens\":5}"), std::string::npos);
      EXPECT_EQ(interface.telemetry.counters->completed, 5u);
    }
    interface.set_observer(interface.self, {});
  }

  // Shared allocator for all production owners.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // Host geometry buckets share one command in the source bootstrap.
  const loom_serve_packing_shape_t shapes_[3] = {{4, 1}, {16, 4}, {32, 4}};
  // External source package path.
  std::string directory_;
  // Real BPE tokenizer asset.
  std::string tokenizer_;
  // Tiny generated checkpoint owned by the test runner's temp directory.
  std::string weights_;
  // Borrowed process-wide task/JIT owner outliving every test model.
  loom_serve_host_t* host_ = nullptr;
  // Borrowed process-wide application group enclosing all test models.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed device record within the process-wide application group.
  loom_serve_device_t* device_ = nullptr;
  // Actual source-defined text model, with no native mock implementation.
  loom_serve_text_model_t* model_ = nullptr;
  // Real loopback HTTP carrier.
  loom_serve_http_server_t* server_ = nullptr;
  // Independent binding borrowing the HTTP owner and shared host pool.
  loom_serve_http_listener_t* listener_ = nullptr;
  // HTTP readiness and optional outer owner, joined before model destruction.
  loom_serve::testing::HttpHarness http_;
};

// Source-service fixture whose model is created by the registration owner.
class RegisteredSourceTextServiceTest : public SourceTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }
};

}  // namespace loom_serve::testing::text

#endif  // LOOM_SERVE_SERVICE_TEXT_SERVICE_TEST_FIXTURE_H_
