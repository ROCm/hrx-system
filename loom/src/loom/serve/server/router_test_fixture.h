// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_ROUTER_TEST_FIXTURE_H_
#define LOOM_SERVE_SERVER_ROUTER_TEST_FIXTURE_H_

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/semaphore.h"
#include "iree/net/carrier/loopback/carrier.h"
#include "iree/task/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "loom/runtime/telemetry/sample.h"
#include "loom/runtime/wake.h"
#include "loom/serve/server/router.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/service/telemetry/record.h"

namespace loom_serve::testing {

// A real carrier with one send credit. Only the client's progress is paused;
// the server continues on its shared proactor and task workers.
class ObservationClient {
 public:
  ~ObservationClient() {
    if (carrier_) {
      bool drained = false;
      iree_net_carrier_deactivate(
          carrier_, [](void* self) { *static_cast<bool*>(self) = true; },
          &drained);
      while (!drained) {
        Poll();
      }
      iree_net_carrier_release(carrier_);
    }
    iree_net_carrier_release(peer_);
    iree_async_proactor_release(proactor_);
  }

  void Open(loom_serve_http_server_t* server, const std::string& headers = "",
            const std::function<void()>& capacity_wait = {},
            const std::string& target = "/logz/stream") {
    auto options = iree_async_proactor_options_default();
    options.threading_mode = IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor_));
    iree_async_proactor_t* server_proactor = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_pool_get(
        loom_serve_http_server_proactor_pool(server), 0, &server_proactor));
    auto carrier_options = iree_net_loopback_carrier_options_default();
    carrier_options.max_send_operations = 1;
    IREE_ASSERT_OK(iree_net_loopback_carrier_create_pair(
        proactor_, server_proactor, &carrier_options, iree_allocator_system(),
        &carrier_, &peer_));
    iree_net_carrier_retain(peer_);
    uint64_t wait_value = 0;
    auto* capacity = loom_serve_http_server_capacity_semaphore(server);
    while (!loom_serve_http_server_try_accept(
        server, peer_, LOOM_SERVE_HTTP_ORIGIN_DIRECT, {}, &wait_value)) {
      // Like the production listener, retain the unadmitted carrier until a
      // capacity generation advances. Client EOF does not join reclamation of
      // the previous HTTP connection's server-side slot.
      if (capacity_wait) {
        capacity_wait();
      }
      iree_status_t status = iree_async_semaphore_multi_wait(
          IREE_ASYNC_WAIT_MODE_ALL, &capacity, &wait_value, 1,
          iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE,
          iree_allocator_system());
      if (!iree_status_is_ok(status)) {
        iree_net_carrier_release(peer_);
        IREE_ASSERT_OK(status);
      }
    }
    IREE_ASSERT_OK(iree_net_carrier_set_handlers(
        carrier_,
        {[](void* self, iree_async_span_t bytes,
            iree_async_buffer_lease_t* lease) -> iree_status_t {
           static_cast<ObservationClient*>(self)->received_.append(
               reinterpret_cast<const char*>(iree_async_span_ptr(bytes)),
               bytes.length);
           return iree_ok_status();
         },
         [](void* self, iree_status_t status) {
           IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
           static_cast<ObservationClient*>(self)->departed_ = true;
         },
         this}));
    IREE_ASSERT_OK(iree_net_carrier_activate(carrier_));
    const std::string request =
        "GET " + target + " HTTP/1.1\r\nHost: localhost\r\n" + headers + "\r\n";
    const iree_net_send_params_t params = {
        .generated_prefix = iree_net_send_prefix_from_bytes(
            iree_make_const_byte_span(request.data(), request.size())),
        .completion_callback = {
            [](void* user_data, iree_status_t status,
               iree_host_size_t bytes_sent) { IREE_EXPECT_OK(status); },
            nullptr}};
    IREE_ASSERT_OK(iree_net_carrier_send(carrier_, &params));
  }

  void WaitCredit(bool available) {
    while (iree_net_carrier_state(peer_) != IREE_NET_CARRIER_STATE_ACTIVE ||
           (iree_net_carrier_query_send_budget(peer_).slots != 0) !=
               available) {
      std::this_thread::yield();
    }
  }
  void OpenLive(loom_serve_http_server_t* server,
                const std::string& headers = "") {
    Open(server,
         "Connection: Upgrade\r\nUpgrade: websocket\r\n"
         "Sec-WebSocket-Version: 13\r\n"
         "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
         "Sec-WebSocket-Protocol: loom.observe.v1\r\n" +
             headers,
         {}, "/v1/observe");
  }
  void ReadUpgrade() {
    while (received_.find("\r\n\r\n") == std::string::npos && !departed_) {
      Poll();
    }
    const auto end = received_.find("\r\n\r\n");
    ASSERT_NE(end, std::string::npos);
    EXPECT_EQ(received_.find("HTTP/1.1 101 "), 0u);
    received_.erase(0, end + 4);
  }
  std::string ReadBytes(size_t count) {
    while (received_.size() < count && !departed_) {
      Poll();
    }
    EXPECT_GE(received_.size(), count);
    std::string bytes = received_.substr(0, count);
    received_.erase(0, bytes.size());
    return bytes;
  }
  std::string WebSocketFrame(uint8_t expected_type = 1) {
    const auto header = ReadBytes(2);
    if (header.size() != 2) {
      return {};
    }
    EXPECT_EQ(static_cast<uint8_t>(header[0]), 0x80 | expected_type);
    EXPECT_EQ(header[1] & 0x80, 0);
    size_t count = header[1] & 127;
    if (count == 126) {
      const auto length = ReadBytes(2);
      if (length.size() != 2) {
        return {};
      }
      count = (static_cast<uint8_t>(length[0]) << 8) |
              static_cast<uint8_t>(length[1]);
    }
    EXPECT_LE(count, 16368u);
    return ReadBytes(count);
  }
  void WriteFrame(uint8_t type, const std::string& payload) {
    ASSERT_LT(payload.size(), 126u);
    const uint8_t mask[] = {1, 2, 3, 4};
    std::string bytes{static_cast<char>(0x80 | type),
                      static_cast<char>(0x80 | payload.size())};
    bytes.append(reinterpret_cast<const char*>(mask), sizeof(mask));
    for (size_t i = 0; i < payload.size(); ++i) {
      bytes.push_back(payload[i] ^ mask[i % 4]);
    }
    while (!iree_net_carrier_query_send_budget(carrier_).slots) {
      Poll();
    }
    bool completed = false;
    const iree_net_send_params_t params = {
        .generated_prefix = iree_net_send_prefix_from_bytes(
            iree_make_const_byte_span(bytes.data(), bytes.size())),
        .completion_callback = {
            [](void* self, iree_status_t status, iree_host_size_t) {
              IREE_EXPECT_OK(status);
              *static_cast<bool*>(self) = true;
            },
            &completed}};
    IREE_ASSERT_OK(iree_net_carrier_send(carrier_, &params));
    // Join accepted send ownership before a final ACK can be cancelled by
    // this client's destructor. Receiving server pages remains independent.
    while (!completed) {
      Poll();
    }
  }
  void Ack(const std::string& frame) {
    const auto key = frame.find("\"sequence\":\"");
    ASSERT_NE(key, std::string::npos);
    const auto start = key + 12;
    WriteFrame(1, frame.substr(start, frame.find('"', start) - start));
  }
  std::string Frame() {
    while (received_.find("\n\n") == std::string::npos && !departed_) {
      Poll();
    }
    const auto end = received_.find("\n\n");
    EXPECT_NE(end, std::string::npos);
    if (end == std::string::npos) {
      return {};
    }
    auto frame = received_.substr(0, end + 2);
    received_.erase(0, end + 2);
    return frame;
  }
  void Join() {
    while (!departed_) {
      Poll();
    }
  }
  std::string Response() {
    Join();
    return std::exchange(received_, {});
  }
  static std::string Cursor(const std::string& frame) {
    const auto start = frame.find("\nid: ");
    EXPECT_NE(start, std::string::npos);
    if (start == std::string::npos) {
      return {};
    }
    return frame.substr(start + 5, frame.find('\n', start + 1) - start - 5);
  }

 private:
  void Poll() {
    IREE_CHECK_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  // Only the test thread polls this client owner.
  iree_async_proactor_t* proactor_ = nullptr;
  // Owned client endpoint, deactivated before releasing the proactor.
  iree_net_carrier_t* carrier_ = nullptr;
  // Retained endpoint for observing actual credit, not HTTP/model ownership.
  iree_net_carrier_t* peer_ = nullptr;
  // Received bytes not yet consumed as complete SSE or WebSocket frames.
  std::string received_;
  // Peer retirement publication on the client poll owner.
  bool departed_ = false;
};

// The dependency is a bounded model service. Routing, HTTP claims, copied
// output, task scheduling, readiness and shutdown use production
// infrastructure.
struct loom_serve_test_model_service_t {
  // Deployment identifier, borrowed for the router lifetime.
  const char* name;
  // Accepted connection/request pairs owned only by serialized router turns.
  std::deque<std::pair<loom_serve_http_connection_t*,
                       const loom_serve_http_request_t*>>
      pending;
  // This model waits for another service's request/maintenance to retire.
  bool wait_for_shared_activity = false;
  // The test owns this model dependency's device-readiness boundary.
  bool hold_completion = false;
  // Protects completion publication and observer detachment.
  std::mutex mutex;
  // Publishes explicit submission and stopping phases to the test.
  std::condition_variable condition;
  // Borrowed owner wake target, detached only after accepted work retires.
  loom_runtime_wake_callback_t observer = {};
  // An accepted completion retains this service until Advance consumes it.
  bool waiting = false;
  // The test has released the accepted completion.
  bool ready = false;
  // Simulated other ready model work continues while this request is held.
  bool advance_ready_work = false;
  // Bounded model units performed on owner turns before CPU readiness.
  uint64_t ready_units = 0;
  // The router has requested wind-down while completion is still held.
  bool stopping = false;
  // Terminal completion result, transferred to the router exactly once.
  iree_status_t result = iree_ok_status();
  // Accepted claims published to the outer test without exposing the FIFO.
  uint64_t accepted = 0;
  // First pending request's owner-side admission observation.
  struct {
    // The owner has considered a request, not merely accepted its transport.
    bool observed = false;
    // Copied router inputs, independent of later service/registry lifetimes.
    loom_runtime_service_flags_t flags = 0;
  } admission;
  // Optional immutable historical work, published before owner startup.
  const loom_service_telemetry_counters_t* counters = nullptr;
  // Optional immutable duration distributions published before owner startup.
  const loom_service_telemetry_latency_t* latency = nullptr;
  // Immutable source facts configured before the service is published.
  loom_serve_reasoning_t reasoning = {};

  static void SetObserver(void* self, loom_runtime_wake_callback_t observer) {
    auto* service = static_cast<loom_serve_test_model_service_t*>(self);
    std::lock_guard<std::mutex> lock(service->mutex);
    EXPECT_FALSE(service->waiting);
    service->observer = observer;
  }
  static bool IsPending(void* self) {
    auto* service = static_cast<loom_serve_test_model_service_t*>(self);
    std::lock_guard<std::mutex> lock(service->mutex);
    return service->waiting;
  }

  static bool HasRequests(void* self) {
    auto* service = static_cast<loom_serve_test_model_service_t*>(self);
    std::lock_guard<std::mutex> lock(service->mutex);
    return !service->pending.empty();
  }

  void WaitSubmitted() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return waiting; });
  }
  void WaitAccepted(uint64_t count) {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this, count] { return accepted >= count; });
  }
  loom_runtime_service_flags_t WaitAdmission() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return admission.observed; });
    return admission.flags;
  }
  void WaitStopping() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return stopping; });
  }
  void Release(iree_status_t status = iree_ok_status()) {
    std::lock_guard<std::mutex> lock(mutex);
    EXPECT_TRUE(waiting);
    ready = true;
    result = status;
    observer.wake(observer.user_data);
  }

  static bool IsActive(void* self) {
    const auto* service = static_cast<loom_serve_test_model_service_t*>(self);
    return !service->wait_for_shared_activity && !service->pending.empty();
  }

  static loom_service_snapshot_t Snapshot(void* self) {
    auto* service = static_cast<loom_serve_test_model_service_t*>(self);
    std::lock_guard<std::mutex> lock(service->mutex);
    const size_t queued = service->pending.size() - service->waiting;
    return {.phase = service->waiting ? "held" : "idle",
            .queued_requests = queued,
            .active_requests = service->waiting ? 1u : 0u,
            .oldest_queued_age = queued ? 12500000000ll : 0,
            .oldest_active_age = service->waiting ? 2250000000ll : 0};
  }

  static iree_status_t Accept(void* self,
                              loom_serve_http_connection_t* connection,
                              const loom_serve_http_request_t* request,
                              loom_runtime_service_flags_t flags) {
    auto* service = static_cast<loom_serve_test_model_service_t*>(self);
    std::lock_guard<std::mutex> lock(service->mutex);
    service->pending.emplace_back(connection, request);
    ++service->accepted;
    service->condition.notify_all();
    return iree_ok_status();
  }
  static iree_status_t Advance(void* self, loom_runtime_service_flags_t flags,
                               bool* progress) {
    auto* service = static_cast<loom_serve_test_model_service_t*>(self);
    *progress = false;
    if (iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING)) {
      std::lock_guard<std::mutex> lock(service->mutex);
      service->stopping = true;
      service->condition.notify_all();
      if (service->waiting && !service->ready) {
        return iree_ok_status();
      }
      *progress = service->waiting || !service->pending.empty();
      service->waiting = false;
      service->ready = false;
      while (!service->pending.empty()) {
        loom_serve_http_connection_abort(service->pending.front().first);
        service->pending.pop_front();
      }
      return std::exchange(service->result, iree_ok_status());
    }
    if (service->hold_completion && !service->pending.empty()) {
      std::lock_guard<std::mutex> lock(service->mutex);
      if (!service->waiting) {
        service->waiting = true;
        *progress = true;
        service->condition.notify_all();
        return iree_ok_status();
      }
      service->stopping =
          iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
      service->condition.notify_all();
      if (!service->ready) {
        if (service->advance_ready_work && !service->stopping) {
          ++service->ready_units;
          *progress = true;
        }
        return iree_ok_status();
      }
      service->waiting = false;
      service->ready = false;
      *progress = true;
      iree_status_t status = service->result;
      service->result = iree_ok_status();
      if (service->stopping || !iree_status_is_ok(status)) {
        return status;
      }
    }
    if (!service->pending.empty()) {
      std::lock_guard<std::mutex> lock(service->mutex);
      if (!service->admission.observed) {
        service->admission = {.observed = true, .flags = flags};
        service->condition.notify_all();
      }
    }
    if (service->wait_for_shared_activity &&
        iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_SHARED_ACTIVITY)) {
      *progress = false;
      return iree_ok_status();
    }
    *progress = !service->pending.empty();
    if (!*progress) {
      return iree_ok_status();
    }
    auto [connection, request] = service->pending.front();
    service->pending.pop_front();
    // The borrowed body is still usable after later claims were routed.
    const std::string response =
        "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n" +
        std::string(service->name) + ":" +
        std::string(request->target.data, request->target.size) + ":" +
        std::string(request->body.data ? request->body.data : "",
                    request->body.size);
    iree_status_t status = loom_serve_http_connection_send(
        connection, iree_make_string_view(response.data(), response.size()));
    if (iree_status_is_ok(status)) {
      loom_serve_http_connection_finish(connection);
    } else {
      loom_serve_http_connection_abort(connection);
    }
    return status;
  }

  static iree_status_t Sample(void* self,
                              loom_runtime_telemetry_sample_sink_t sink) {
    const auto* service = static_cast<loom_serve_test_model_service_t*>(self);
    const loom_runtime_telemetry_metric_t field = {
        "consumed", "tokens", LOOM_RUNTIME_TELEMETRY_COUNTER};
    const loom_runtime_telemetry_topic_t topic = {"work", 1, 1, &field};
    return sink.emit(sink.self, &topic, &service->counters->input_tokens);
  }

  loom_serve_service_t Interface(const iree_tokenizer_t* tokenizer,
                                 const void* capacity_domain) {
    static const loom_runtime_service_vtable_t vtable = {
        .set_observer = SetObserver,
        .has_requests = HasRequests,
        .is_pending = IsPending,
        .is_active = IsActive,
        .advance = Advance,
    };
    return {.name = iree_make_cstring_view(name),
            .runtime = {.self = this,
                        .vtable = &vtable,
                        .capacity_domain = capacity_domain},
            .accept = Accept,
            .tokenizer = tokenizer,
            .kind = LOOM_SERVE_MODEL_KIND_EMBEDDING,
            .endpoint = "/v1/test",
            .capabilities = {.reasoning = reasoning},
            .snapshot = Snapshot,
            .telemetry = {.counters = counters,
                          .latency = latency,
                          .sample = counters ? Sample : nullptr}};
  }
};

class HttpRouterTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(GetParam(), &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    // Match the serving host's worker stack budget for tokenizer and other
    // application CPU work.
    executor_options.worker_stack_size = 2 * 1024 * 1024;
    iree_status_t status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    auto options = loom_serve_http_server_options_default();
    options.connection_capacity = connection_capacity_;
    options.allowed_origins = allowed_origins_;
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        pool_, &options, iree_allocator_system(), &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(
        server_, &listener_options, iree_allocator_system(), &listener_));
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        IREE_SV(R"({"model":{"type":"BPE","vocab":{"a":0,"b":1},
                              "merges":[]},"decoder":{"type":"ByteFallback"}})"),
        iree_allocator_system(), &tokenizer_a_));
    iree_tokenizer_builder_t builder;
    iree_tokenizer_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t tokenizer_status = iree_tokenizer_parse_huggingface_json(
        IREE_SV(R"({"model":{"type":"BPE","vocab":{"a":1,"b":0},
                              "merges":[]}})"),
        &builder);
    if (iree_status_is_ok(tokenizer_status)) {
      // Native builders explicitly disable decoding; an absent HF decoder
      // selects the format's space-separated default instead.
      iree_tokenizer_builder_set_decoder(&builder, nullptr);
      tokenizer_status = iree_tokenizer_builder_build(&builder, &tokenizer_b_);
    }
    iree_tokenizer_builder_deinitialize(&builder);
    IREE_ASSERT_OK(tokenizer_status);
    owner_ = std::thread([this] {
      const loom_serve_service_t services[] = {
          a_.Interface(tokenizer_a_, &a_), b_.Interface(tokenizer_b_, &a_),
          held_.Interface(tokenizer_a_, &a_),
          other_.Interface(nullptr, nullptr),
          escaped_.Interface(nullptr, nullptr)};
      loom_serve_registry_t* registry = nullptr;
      IREE_ASSERT_OK(loom_serve_registry_allocate(
          IREE_ARRAYSIZE(services), services, managed_capacity_, factory_,
          iree_allocator_system(), &registry));
      status_ = loom_serve_router_run(
          executor_, server_, registry,
          {.request_quota = request_quota_,
           .holding_capacity = holding_capacity_,
           .resources = {.listener_count = 1, .listeners = &listener_}},
          iree_allocator_system());
      IREE_ASSERT_OK(loom_serve_registry_free(registry));
      returned_.store(true);
    });
  }
  void TearDown() override {
    if (owner_.joinable()) {
      loom_serve_http_server_request_stop(server_);
      owner_.join();
    }
    IREE_EXPECT_STATUS_IS(expected_code_, status_);
    for (auto* service : {&a_, &b_, &held_, &other_, &escaped_}) {
      for (auto& [connection, request] : service->pending) {
        loom_serve_http_connection_abort(connection);
      }
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    iree_async_proactor_pool_release(pool_);
    iree_tokenizer_free(tokenizer_b_);
    iree_tokenizer_free(tokenizer_a_);
    iree_task_executor_release(executor_);
  }
  int Send(const std::string& method, const std::string& target,
           const std::string& body = "", const std::string& headers = "",
           const std::string& content_type = "application/json") {
    const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    const auto* address = loom_serve_http_listener_address(listener_);
    EXPECT_EQ(
        connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                address->length),
        0);
    const std::string request =
        method + " " + target +
        " HTTP/1.1\r\nHost: local\r\nContent-Type: " + content_type + "\r\n" +
        headers + "Content-Length: " + std::to_string(body.size()) +
        "\r\n\r\n" + body;
    size_t offset = 0;
    while (offset < request.size()) {
      const ssize_t count = send(client, request.data() + offset,
                                 request.size() - offset, MSG_NOSIGNAL);
      EXPECT_GT(count, 0);
      if (count <= 0) {
        break;
      }
      offset += count;
    }
    return client;
  }
  std::string Read(int client) {
    std::string result;
    char buffer[4096];
    ssize_t count = 0;
    while ((count = recv(client, buffer, sizeof(buffer), 0)) > 0) {
      result.append(buffer, count);
    }
    EXPECT_EQ(count, 0);
    EXPECT_EQ(close(client), 0);
    return result;
  }
  std::string Control(const char* operation, const char* name) {
    return Read(Send("POST", std::string("/v1/models/") + operation,
                     std::string("{\"model\":\"") + name + "\"}"));
  }
  // Repeated drain is idempotent. Its copied response acknowledges the
  // serialized owner and observes admission without reading private state.
  std::string WaitHeld(const char* name, size_t count) {
    const std::string expected =
        "\"held_requests\":" + std::to_string(count) + ",";
    for (;;) {
      const auto response = Control("drain", name);
      EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
      if (response.find(expected) != std::string::npos ||
          response.find("HTTP/1.1 200") != 0) {
        return response;
      }
    }
  }
  // Tight-capacity cases change this before the production server starts.
  iree_host_size_t connection_capacity_ = 64;
  // Maximum ordinary requests or observation steps in one owner turn.
  iree_host_size_t request_quota_ = 4;
  // Zero-budget cases reject instead of retaining requests behind the gate.
  iree_host_size_t holding_capacity_ = 2;
  // Managed slots supplied only by registration-lifecycle cases.
  iree_host_size_t managed_capacity_ = 0;
  // Additional browser origins, borrowed until the HTTP owner retires.
  iree_string_view_list_t allowed_origins_ = {};
  // Model dependency construction, never a replacement for registry behavior.
  loom_serve_registration_factory_t factory_ = {};
  // HTTP connection owner with no private polling thread.
  loom_serve_http_server_t* server_ = nullptr;
  // Independent binding to the shared connection owner.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Shared persistent I/O infrastructure.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Shared task workers, joined only after the router's final release.
  iree_task_executor_t* executor_ = nullptr;
  // Immutable tokenizers outliving accepted jobs, with intentionally distinct
  // IDs.
  iree_tokenizer_t* tokenizer_a_ = nullptr;
  // Second model's vocabulary proves utilities use the selected deployment.
  iree_tokenizer_t* tokenizer_b_ = nullptr;
  // First model dependency used only by serialized router turns.
  loom_serve_test_model_service_t a_{"a"};
  // Second model dependency used only by serialized router turns.
  loom_serve_test_model_service_t b_{"b", {}, true};
  // Explicitly held device-readiness dependencies, independent of a and b.
  loom_serve_test_model_service_t held_{"held", {}, false, true};
  // A second held dependency proves failure drains every accepted callback.
  loom_serve_test_model_service_t other_{"other", {}, false, true};
  // Valid deployment identifier requiring distinct metric label escaping.
  loom_serve_test_model_service_t escaped_{R"(quoted"\name)", {}, true};
  // Outer application caller joining the production router's final release.
  std::thread owner_;
  // Terminal result published by joining owner_.
  iree_status_t status_ = iree_ok_status();
  // Owner return observation independent of its joined terminal status.
  std::atomic<bool> returned_{false};
  // Expected terminal result; service errors are not hidden by shutdown.
  iree_status_code_t expected_code_ = IREE_STATUS_OK;
};

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_SERVER_ROUTER_TEST_FIXTURE_H_
