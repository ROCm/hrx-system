// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/serving/router.h"

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "experimental/loom_serve/http/listener.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/net/carrier/loopback/carrier.h"
#include "iree/task/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

namespace {

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

  void Open(loom_serve_http_server_t* server, const std::string& headers = "") {
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
    loom_serve_http_server_accept(server, peer_);
    IREE_ASSERT_OK(iree_net_carrier_set_handlers(
        carrier_,
        {[](void* self, iree_async_span_t bytes,
            iree_async_buffer_lease_t*) -> iree_status_t {
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
        "GET /logz/stream HTTP/1.1\r\nHost: local\r\n" + headers + "\r\n";
    const iree_net_send_params_t params = {
        .generated_prefix = iree_net_send_prefix_from_bytes(
            iree_make_const_byte_span(request.data(), request.size())),
        .completion_callback = {
            [](void*, iree_status_t status, iree_host_size_t) {
              IREE_EXPECT_OK(status);
            },
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
  // Received bytes not yet consumed as complete SSE frames.
  std::string received_;
  // Peer retirement publication on the client poll owner.
  bool departed_ = false;
};

// The dependency is a bounded model service. Routing, HTTP claims, copied
// output, task scheduling, readiness and shutdown use production
// infrastructure.
struct ModelService {
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
  loom_serve_http_observer_t observer = {};
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
  // Optional immutable historical work, published before owner startup.
  const loom_serve_telemetry_counters_t* counters = nullptr;

  static void SetObserver(void* self, loom_serve_http_observer_t observer) {
    auto* service = static_cast<ModelService*>(self);
    std::lock_guard<std::mutex> lock(service->mutex);
    EXPECT_FALSE(service->waiting);
    service->observer = observer;
  }
  static bool IsPending(void* self) {
    auto* service = static_cast<ModelService*>(self);
    std::lock_guard<std::mutex> lock(service->mutex);
    return service->waiting;
  }

  void WaitSubmitted() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return waiting; });
  }
  void WaitAccepted(uint64_t count) {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this, count] { return accepted >= count; });
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
    const auto* service = static_cast<ModelService*>(self);
    return !service->wait_for_shared_activity && !service->pending.empty();
  }

  static loom_serve_http_service_snapshot_t Snapshot(void* self) {
    auto* service = static_cast<ModelService*>(self);
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
                              loom_serve_http_service_flags_t flags) {
    (void)flags;
    auto* service = static_cast<ModelService*>(self);
    std::lock_guard<std::mutex> lock(service->mutex);
    service->pending.emplace_back(connection, request);
    ++service->accepted;
    service->condition.notify_all();
    return iree_ok_status();
  }
  static iree_status_t Advance(void* self,
                               loom_serve_http_service_flags_t flags,
                               bool* progress) {
    auto* service = static_cast<ModelService*>(self);
    *progress = false;
    if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
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
          iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING);
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
    if (service->wait_for_shared_activity &&
        iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY)) {
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

  loom_serve_http_service_t Interface(const iree_tokenizer_t* tokenizer,
                                      const void* capacity_domain) {
    return {.name = iree_make_cstring_view(name),
            .self = this,
            .set_observer = SetObserver,
            .is_pending = IsPending,
            .is_active = IsActive,
            .accept = Accept,
            .advance = Advance,
            .tokenizer = tokenizer,
            .kind = "fixture",
            .endpoint = "/v1/test",
            .snapshot = Snapshot,
            .capacity_domain = capacity_domain,
            .telemetry = {.counters = counters}};
  }
};

// A model dependency with independently controlled service and final device
// ownership. HTTP admission, the registry, task owner and retirement policy are
// the production implementations under test.
struct RegistrationDependency {
  // Requests reach the same held-completion service used by the routing tests.
  ModelService service = {.name = "dynamic", .hold_completion = true};
  // Model-protocol counters supplied by the dependency, not synthesized by
  // routing.
  loom_serve_telemetry_counters_t counters = {};
  // Serializes retirement readiness and wake detachment with the owner.
  std::mutex mutex;
  // Outer-test synchronization on actual retirement entry.
  std::condition_variable condition;
  // Final device-borrower wake, independent of the service observer.
  loom_serve_http_observer_t observer = {};
  // Registry-owned history borrowed by the prepared service.
  loom_serve_telemetry_stream_t* stream = nullptr;
  // Retire has been called after all service and utility borrowers completed.
  bool retiring = false;
  // The test has returned final model ownership.
  bool retired = false;
  // Preparation allocation count, observable without reading owner-only state.
  std::atomic<unsigned> created{0};
  // Operation destruction count, independent of name/slot reuse.
  std::atomic<unsigned> destroyed{0};
  // Injects one quiescent platform release failure, preserving owned storage.
  std::atomic<bool> fail_release{false};
  // Counts final release attempts, not asynchronous ownership polling.
  std::atomic<unsigned> release_attempts{0};
  // Parameter state belongs to the model dependency and changes on owner turns.
  loom_serve_residency_snapshot_t parameters = {
      .active = true,
      .elastic = true,
      .memory = {.reserved_bytes = 4096, .committed_bytes = 4096},
      .activity = {.activations = 1}};
  // Injects an explicit native parameter release failure after quiescence.
  std::atomic<bool> fail_deactivation{false};
  // Optional shared pressure domain, configured before factory publication.
  const void* capacity_domain = nullptr;
  // Independently controlled parameter-readiness dependency.
  struct {
    // An accepted warmup still borrows the model until readiness is consumed.
    bool submitted = false;
    // The test has published terminal readiness.
    bool ready = false;
    // Unregister has requested cancellation while the callback is outstanding.
    bool cancelling = false;
    // Number of accepted operations, not repeated owner polls.
    unsigned starts = 0;
    // Terminal readiness result transferred to the production registry.
    iree_status_t status = iree_ok_status();
  } activation;

  struct Operation {
    // Test-owned model dependency, live beyond the owning registry.
    RegistrationDependency* dependency;
  };

  static iree_status_t Prepare(void* self, bool cancel, bool* complete,
                               loom_serve_http_service_t* service,
                               loom_serve_compilation_report_t*) {
    auto* dependency = static_cast<Operation*>(self)->dependency;
    *complete = true;
    if (cancel) {
      return iree_make_status(IREE_STATUS_CANCELLED, "preparation cancelled");
    }
    *service =
        dependency->service.Interface(nullptr, dependency->capacity_domain);
    service->telemetry.stream = dependency->stream;
    return iree_ok_status();
  }
  static iree_status_t Retire(void* self, bool* complete) {
    auto* dependency = static_cast<Operation*>(self)->dependency;
    EXPECT_FALSE(ModelService::IsPending(&dependency->service));
    EXPECT_TRUE(dependency->service.pending.empty());
    std::lock_guard<std::mutex> lock(dependency->mutex);
    EXPECT_FALSE(dependency->activation.submitted);
    dependency->retiring = true;
    *complete = dependency->retired;
    if (*complete) {
      dependency->observer = {};
      ++dependency->release_attempts;
      if (dependency->fail_release.exchange(false)) {
        return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "platform release failed");
      }
    }
    dependency->condition.notify_all();
    return iree_ok_status();
  }
  static void Destroy(void* self) {
    auto* operation = static_cast<Operation*>(self);
    ++operation->dependency->destroyed;
    delete operation;
  }
  static loom_serve_residency_snapshot_t Parameters(void* self) {
    return static_cast<Operation*>(self)->dependency->parameters;
  }
  static iree_status_t Deactivate(void* self) {
    auto* dependency = static_cast<Operation*>(self)->dependency;
    EXPECT_FALSE(ModelService::IsPending(&dependency->service));
    EXPECT_TRUE(dependency->service.pending.empty());
    if (dependency->fail_deactivation.exchange(false)) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "parameter release failed");
    }
    auto& parameters = dependency->parameters;
    parameters.activity.deactivations += parameters.active;
    parameters.active = false;
    parameters.memory.committed_bytes = 0;
    return iree_ok_status();
  }
  static iree_status_t Activate(void* self, loom_serve_activation_flags_t flags,
                                loom_serve_activation_phase_t* phase) {
    auto* dependency = static_cast<Operation*>(self)->dependency;
    std::lock_guard<std::mutex> lock(dependency->mutex);
    auto& activation = dependency->activation;
    const bool cancel =
        iree_any_bit_set(flags, LOOM_SERVE_ACTIVATION_FLAG_CANCEL);
    if (!activation.submitted && cancel) {
      *phase = LOOM_SERVE_ACTIVATION_COMPLETE;
      return iree_ok_status();
    }
    if (!activation.submitted) {
      activation.submitted = true;
      ++activation.starts;
    }
    activation.cancelling |= cancel;
    dependency->condition.notify_all();
    if (!activation.ready) {
      *phase = LOOM_SERVE_ACTIVATION_LOADING;
      return iree_ok_status();
    }
    activation.submitted = false;
    activation.ready = false;
    *phase = LOOM_SERVE_ACTIVATION_COMPLETE;
    if (iree_status_is_ok(activation.status)) {
      auto& parameters = dependency->parameters;
      parameters.activity.activations += !parameters.active;
      parameters.active = true;
      parameters.memory.committed_bytes = parameters.memory.reserved_bytes;
    }
    return std::exchange(activation.status, iree_ok_status());
  }
  static iree_status_t Create(
      void* self, const loom_serve_model_configuration_t* configuration,
      loom_serve_http_observer_t observer,
      loom_serve_telemetry_stream_t* stream,
      loom_serve_registration_t* registration, iree_allocator_t) {
    auto* dependency = static_cast<RegistrationDependency*>(self);
    // Import fixture configuration before transferring it to the model owner.
    std::lock_guard<std::mutex> lock(dependency->mutex);
    dependency->service.name = configuration->name.data;
    dependency->service.counters = &dependency->counters;
    dependency->observer = observer;
    dependency->stream = stream;
    ++dependency->created;
    *registration = {.self = new Operation{dependency},
                     .prepare = Prepare,
                     .retire = Retire,
                     .parameters = Parameters,
                     .activate = Activate,
                     .deactivate = Deactivate,
                     .destroy = Destroy};
    return iree_ok_status();
  }
  void WaitRetiring() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return retiring; });
  }
  void WaitActivating() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return activation.submitted; });
  }
  void WaitActivationCancellation() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return activation.cancelling; });
  }
  void ReleaseActivation(iree_status_t status = iree_ok_status()) {
    std::lock_guard<std::mutex> lock(mutex);
    EXPECT_TRUE(activation.submitted);
    activation.ready = true;
    activation.status = status;
    observer.wake(observer.user_data);
  }
  void ReleaseRetirement() {
    std::lock_guard<std::mutex> lock(mutex);
    retired = true;
    observer.wake(observer.user_data);
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
    iree_status_t status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    auto options = loom_serve_http_server_options_default();
    options.connection_capacity = connection_capacity_;
    IREE_ASSERT_OK(loom_serve_http_server_create(pool_, &options, &server_,
                                                 iree_allocator_system()));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_create(
        server_, &listener_options, &listener_, iree_allocator_system()));
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        IREE_SV(R"({"model":{"type":"BPE","vocab":{"a":0,"b":1},
                              "merges":[]},"decoder":{"type":"ByteFallback"}})"),
        iree_allocator_system(), &tokenizer_a_));
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        IREE_SV(R"({"model":{"type":"BPE","vocab":{"a":1,"b":0},
                              "merges":[]}})"),
        iree_allocator_system(), &tokenizer_b_));
    owner_ = std::thread([this] {
      const loom_serve_http_service_t services[] = {
          a_.Interface(tokenizer_a_, &a_), b_.Interface(tokenizer_b_, &a_),
          held_.Interface(tokenizer_a_, &a_),
          other_.Interface(nullptr, nullptr),
          escaped_.Interface(nullptr, nullptr)};
      loom_serve_registry_t* registry = nullptr;
      IREE_ASSERT_OK(loom_serve_registry_create(
          IREE_ARRAYSIZE(services), services, managed_capacity_, factory_,
          &registry, iree_allocator_system()));
      status_ = loom_serve_router_run(
          executor_, server_, registry,
          {.request_quota = 4, .holding_capacity = holding_capacity_},
          iree_allocator_system());
      IREE_ASSERT_OK(loom_serve_registry_destroy(registry));
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
        (void)request;
        loom_serve_http_connection_abort(connection);
      }
    }
    IREE_EXPECT_OK(loom_serve_http_listener_destroy(listener_));
    loom_serve_http_server_destroy(server_);
    iree_async_proactor_pool_release(pool_);
    iree_tokenizer_free(tokenizer_b_);
    iree_tokenizer_free(tokenizer_a_);
    iree_task_executor_release(executor_);
  }
  int Send(const std::string& method, const std::string& target,
           const std::string& body = "", const std::string& headers = "") {
    const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    const auto* address = loom_serve_http_listener_address(listener_);
    EXPECT_EQ(
        connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                address->length),
        0);
    const std::string request =
        method + " " + target +
        " HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\n" +
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
  // Zero-budget cases reject instead of retaining requests behind the gate.
  iree_host_size_t holding_capacity_ = 2;
  // Managed slots supplied only by registration-lifecycle cases.
  iree_host_size_t managed_capacity_ = 0;
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
  ModelService a_{"a"};
  // Second model dependency used only by serialized router turns.
  ModelService b_{"b", {}, true};
  // Explicitly held device-readiness dependencies, independent of a and b.
  ModelService held_{"held", {}, false, true};
  // A second held dependency proves failure drains every accepted callback.
  ModelService other_{"other", {}, false, true};
  // Valid deployment identifier requiring distinct metric label escaping.
  ModelService escaped_{R"(quoted"\name)", {}, true};
  // Outer application caller joining the production router's final release.
  std::thread owner_;
  // Terminal result published by joining owner_.
  iree_status_t status_ = iree_ok_status();
  // Owner return observation independent of its joined terminal status.
  std::atomic<bool> returned_{false};
  // Expected terminal result; service errors are not hidden by shutdown.
  iree_status_code_t expected_code_ = IREE_STATUS_OK;
};

class HttpRouterRegistrationTest : public HttpRouterTest {
 protected:
  void SetUp() override {
    managed_capacity_ = 1;
    factory_ = {.self = &managed_,
                .kinds = 1u << LOOM_SERVE_MODEL_KIND_EMBEDDING,
                .create = RegistrationDependency::Create};
    HttpRouterTest::SetUp();
  }
  void Declare() {
    EXPECT_EQ(
        Read(
            Send(
                "POST", "/v1/models/register",
                R"({"kind":"embedding","name":"dynamic","source":"model","assets":"assets","admission":"drained"})"))
            .find("HTTP/1.1 202"),
        0u);
    EXPECT_EQ(managed_.created.load(), 0u);
  }
  void Register() {
    Declare();
    EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
    while (Read(Send("GET", "/v1/status")).find(R"("registration":"ready")") ==
           std::string::npos) {
    }
  }
  void Remove() {
    EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
    managed_.WaitRetiring();
    managed_.ReleaseRetirement();
    while (Read(Send("GET", "/v1/models")).find(R"("id":"dynamic")") !=
           std::string::npos) {
    }
  }
  std::string WaitParameterOperation(const char* phase) {
    const std::string expected =
        std::string("\"parameter_operation\":\"") + phase + "\"";
    for (;;) {
      const auto state = Read(Send("GET", "/v1/status"));
      if (state.find(expected) != std::string::npos) {
        return state;
      }
    }
  }
  // The model alone is a dependency; every control/transport path is
  // production.
  RegistrationDependency managed_;
};

TEST_P(HttpRouterRegistrationTest, StalledObserverDoesNotRetainRemovedModel) {
  Register();
  ObservationClient observer;
  observer.Open(server_, "X-Loom-Model: dynamic\r\n");
  observer.WaitCredit(false);
  // The first response is already copied, but its only send credit remains
  // occupied throughout native model retirement and independent peer output.
  Remove();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_EQ(Read(Send("GET", "/logz/stream", "", "X-Loom-Model: dynamic\r\n"))
                .find("HTTP/1.1 404"),
            0u);
  // Reuse the registration slot before returning the old reader's credit.
  EXPECT_EQ(
      Read(
          Send(
              "POST", "/v1/models/register",
              R"({"kind":"embedding","name":"dynamic","source":"model","assets":"assets","admission":"drained"})"))
          .find("HTTP/1.1 202"),
      0u);
  const auto first = observer.Frame();
  const auto cursor = ObservationClient::Cursor(first);
  EXPECT_NE(first.find("\"model\":\"dynamic\""), std::string::npos);
  const auto ended = observer.Frame();
  EXPECT_EQ(ended.find("event: stream_end\n"), 0u);
  EXPECT_NE(ended.find("\"reason\":\"model_unregistered\""), std::string::npos);
  observer.Join();
  // The replacement registration never accepts the old incarnation's cursor.
  EXPECT_EQ(
      Read(Send("GET", "/logz/stream", "",
                "X-Loom-Model: dynamic\r\nLast-Event-ID: " + cursor + "\r\n"))
          .find("HTTP/1.1 409"),
      0u);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
}

TEST_P(HttpRouterTest,
       LiveReadersResumeAndYieldToPeersWhenTheirRingOverwrites) {
  ObservationClient slow;
  slow.Open(server_);
  const auto initial = slow.Frame();
  const auto cursor = ObservationClient::Cursor(initial);
  EXPECT_NE(initial.find("Content-Type: text/event-stream\r\n"),
            std::string::npos);
  slow.WaitCredit(true);
  // An independent diagnostic publisher, not an inference completion, must
  // wake an otherwise idle owner. This error has one explicit diagnostic sink.
  iree_status_t error =
      iree_make_status(IREE_STATUS_UNAVAILABLE, "stream probe");
  loom_serve_http_server_record_error(server_, error);
  slow.WaitCredit(false);
  const auto* stream = loom_serve_http_server_diagnostics(server_);
  for (size_t i = 0; i < loom_serve_telemetry_stream_capacity(stream) + 8;
       ++i) {
    loom_serve_http_server_record_error(server_, error);
  }
  iree_status_free(error);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  ObservationClient resumed;
  resumed.Open(server_, "Last-Event-ID: " + cursor + "\r\n");
  EXPECT_NE(resumed.Frame().find("\"event\":\"gap\""), std::string::npos);
  EXPECT_NE(slow.Frame().find("stream probe"), std::string::npos);
  EXPECT_NE(slow.Frame().find("\"event\":\"gap\""), std::string::npos);
  EXPECT_EQ(Read(Send("GET", "/logz/stream", "", "Last-Event-ID: bad\r\n"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("GET", "/logz/stream", "",
                      "Last-Event-ID: " + cursor +
                          "\r\nX-Loom-Cursor: different\r\n"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("GET", "/logz/stream", "", "X-Loom-Model: a\r\n"))
                .find("HTTP/1.1 501"),
            0u);
  // Shutdown cancels both claimed streams without waiting for their credit.
  loom_serve_http_server_request_stop(server_);
  owner_.join();
  slow.Join();
  resumed.Join();
}

TEST_P(HttpRouterTest, IdleObserverDoesNotProduceCompletionDrivenEmptyFrames) {
  ObservationClient observer;
  observer.Open(server_);
  const auto first = observer.Frame();
  observer.WaitCredit(true);
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
  ObservationClient resumed;
  resumed.Open(server_,
               "X-Loom-Cursor: " + ObservationClient::Cursor(first) + "\r\n");
  EXPECT_EQ(ObservationClient::Cursor(resumed.Frame()),
            ObservationClient::Cursor(first));
  resumed.WaitCredit(true);
  observer.WaitCredit(true);
  // No additional publication is needed to release either reader.
  loom_serve_http_server_request_stop(server_);
  owner_.join();
  observer.Join();
  resumed.Join();
}

TEST_P(HttpRouterRegistrationTest,
       DrainedActivationSurvivesDisconnectAndPreservesPeerProgress) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.parameters.active = false;
    managed_.parameters.memory.committed_bytes = 0;
    managed_.capacity_domain = &a_;
  }
  Declare();
  const int operation =
      Send("POST", "/v1/models/activate", R"({"model":"dynamic"})");
  managed_.WaitActivating();
  EXPECT_EQ(close(operation), 0);
  EXPECT_EQ(managed_.created.load(), 1u);
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 409"), 0u);
  const int request = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  WaitHeld("dynamic", 1);
  const int peer = Send("POST", "/v1/test", R"({"model":"b"})");
  b_.WaitAccepted(1);
  // This peer treats accepted activation as releasable shared pressure. It
  // remains queued, while independent requests and health still complete.
  EXPECT_NE(Control("drain", "b").find(R"("queued_requests":1)"),
            std::string::npos);
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_NE(
      WaitParameterOperation("activating").find(R"("admission":"drained")"),
      std::string::npos);
  managed_.ReleaseActivation();
  const auto ready = WaitParameterOperation("idle");
  EXPECT_NE(ready.find(R"("active":true,"backing":"elastic")"),
            std::string::npos);
  EXPECT_NE(ready.find(R"("held_requests":1)"), std::string::npos);
  EXPECT_EQ(Read(peer).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  managed_.service.WaitSubmitted();
  managed_.service.Release();
  EXPECT_EQ(Read(request).find("HTTP/1.1 200"), 0u);
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    EXPECT_EQ(managed_.activation.starts, 1u);
  }
  Remove();
}

TEST_P(HttpRouterRegistrationTest, UnregisterJoinsAcceptedActivationFirst) {
  Declare();
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitActivating();
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitActivationCancellation();
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    EXPECT_FALSE(managed_.retiring);
  }
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  loom_serve_http_server_request_stop(server_);
  EXPECT_FALSE(returned_.load());
  managed_.ReleaseActivation();
  managed_.WaitRetiring();
  managed_.ReleaseRetirement();
  owner_.join();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterRegistrationTest,
       RemovalWithdrawsWarmupBeforeAdmittedWorkEnds) {
  Register();
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 409"), 0u);
  const int request = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  managed_.service.WaitSubmitted();
  Control("drain", "dynamic");
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  WaitParameterOperation("waiting");
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"dynamic","mode":"abort"})"))
                .find("HTTP/1.1 202"),
            0u);
  managed_.service.WaitStopping();
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    EXPECT_EQ(managed_.activation.starts, 0u);
  }
  managed_.service.Release();
  EXPECT_TRUE(Read(request).empty());
  Remove();
}

TEST_P(HttpRouterRegistrationTest, ActivationFailureRetainsTheClosedOwner) {
  Declare();
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitActivating();
  managed_.ReleaseActivation(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "parameter read failed"));
  const auto state = WaitParameterOperation("idle");
  EXPECT_NE(state.find("parameter read failed"), std::string::npos);
  EXPECT_NE(state.find(R"("admission":"drained")"), std::string::npos);
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  Remove();
}

TEST_P(HttpRouterRegistrationTest,
       DeactivationFinishesAdmittedWorkAndSurvivesClientDisconnect) {
  Register();
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 409"), 0u);
  const int first =
      Send("POST", "/v1/test", R"({"model":"dynamic","value":1})");
  managed_.service.WaitSubmitted();
  const int second =
      Send("POST", "/v1/test", R"({"model":"dynamic","value":2})");
  managed_.service.WaitAccepted(2);
  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  const int held = Send("POST", "/v1/test", R"({"model":"dynamic","value":3})");
  WaitHeld("dynamic", 1);
  const int operation =
      Send("POST", "/v1/models/deactivate", R"({"model":"dynamic"})");
  while (Read(Send("GET", "/v1/status"))
             .find(R"("parameter_operation":"deactivating")") ==
         std::string::npos) {
  }
  EXPECT_EQ(close(operation), 0);
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  managed_.service.Release();
  EXPECT_NE(Read(first).find(R"("value":1)"), std::string::npos);
  managed_.service.WaitSubmitted();
  const auto waiting = Read(Send("GET", "/v1/status"));
  EXPECT_NE(waiting.find(R"("parameter_operation":"deactivating")"),
            std::string::npos);
  EXPECT_NE(waiting.find(R"("deactivations":0)"), std::string::npos);
  managed_.service.Release();
  EXPECT_NE(Read(second).find(R"("value":2)"), std::string::npos);
  std::string cold;
  do {
    cold = Read(Send("GET", "/v1/status"));
  } while (cold.find(R"("active":false,"backing":"elastic")") ==
           std::string::npos);
  EXPECT_NE(cold.find(R"("committed_bytes":0)"), std::string::npos);
  EXPECT_NE(cold.find(R"("deactivations":1)"), std::string::npos);
  EXPECT_NE(cold.find(R"("held_requests":1)"), std::string::npos);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  managed_.service.WaitSubmitted();
  managed_.service.Release();
  EXPECT_NE(Read(held).find(R"("value":3)"), std::string::npos);
  Remove();
}

TEST_P(HttpRouterRegistrationTest,
       FixedDeactivationRejectionPreservesTheModel) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.parameters.elastic = false;
  }
  Register();
  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  const auto rejected = Control("deactivate", "dynamic");
  EXPECT_EQ(rejected.find("HTTP/1.1 409"), 0u);
  EXPECT_NE(rejected.find("fixed parameter backing"), std::string::npos);
  const auto state = Read(Send("GET", "/v1/status"));
  EXPECT_NE(state.find(R"("active":true,"backing":"fixed")"),
            std::string::npos);
  EXPECT_EQ(state.find("committed_bytes"), std::string::npos);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  const int request = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  managed_.service.WaitSubmitted();
  managed_.service.Release();
  EXPECT_EQ(Read(request).find("HTTP/1.1 200"), 0u);
  Remove();
}

TEST_P(HttpRouterRegistrationTest, DeactivationFailureRetainsTheClosedOwner) {
  managed_.fail_deactivation = true;
  Register();
  Control("drain", "dynamic");
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 202"), 0u);
  std::string state;
  do {
    state = Read(Send("GET", "/v1/status"));
  } while (state.find("parameter release failed") == std::string::npos);
  EXPECT_NE(state.find(R"("parameter_operation":"idle")"), std::string::npos);
  EXPECT_NE(state.find(R"("admission":"drained")"), std::string::npos);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  Remove();
}

TEST_P(HttpRouterRegistrationTest, RemovalSupersedesPendingDeactivation) {
  Register();
  const int request = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  managed_.service.WaitSubmitted();
  Control("drain", "dynamic");
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"dynamic","mode":"abort"})"))
                .find("HTTP/1.1 202"),
            0u);
  managed_.service.WaitStopping();
  const auto state = Read(Send("GET", "/v1/status"));
  EXPECT_NE(state.find(R"("parameter_operation":"idle")"), std::string::npos);
  EXPECT_NE(state.find(R"("deactivations":0)"), std::string::npos);
  managed_.service.Release();
  EXPECT_TRUE(Read(request).empty());
  Remove();
}

TEST_P(HttpRouterRegistrationTest, EmptyHeaderCannotSelectAnUnusedSlot) {
  EXPECT_EQ(
      Read(Send("POST", "/v1/checkpoints/empty", "", "X-Loom-Model: \r\n"))
          .find("HTTP/1.1 404"),
      0u);
  EXPECT_EQ(
      Read(Send("GET", "/logz", "", "X-Loom-Model: \r\n")).find("HTTP/1.1 404"),
      0u);
}

TEST_P(HttpRouterRegistrationTest,
       ParameterOperationsValidateTheirSelectorAndOwnership) {
  for (const auto* operation : {"activate", "deactivate"}) {
    for (const char* body :
         {R"({})", R"({"model":""})", R"({"model":"dynamic","extra":true})",
          R"({"model":"dynamic","model":"a"})"}) {
      EXPECT_EQ(Read(Send("POST", std::string("/v1/models/") + operation, body))
                    .find("HTTP/1.1 400"),
                0u);
    }
    EXPECT_EQ(Control(operation, "absent").find("HTTP/1.1 404"), 0u);
    EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
    EXPECT_EQ(Control(operation, "a").find("HTTP/1.1 409"), 0u);
    EXPECT_EQ(Control("undrain", "a").find("HTTP/1.1 200"), 0u);
  }
}

TEST_P(HttpRouterRegistrationTest,
       GracefulRemovalFinishesAdmittedQueuesBeforeRetirement) {
  Register();
  const int first =
      Send("POST", "/v1/test", R"({"model":"dynamic","value":1})");
  managed_.service.WaitSubmitted();
  const int second =
      Send("POST", "/v1/test", R"({"model":"dynamic","value":2})");
  managed_.service.WaitAccepted(2);
  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  const int held = Send("POST", "/v1/test", R"({"model":"dynamic","value":3})");
  WaitHeld("dynamic", 1);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(Read(held).find("HTTP/1.1 410"), 0u);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  managed_.service.Release();
  EXPECT_NE(Read(first).find(R"("value":1)"), std::string::npos);
  managed_.service.WaitSubmitted();
  managed_.service.Release();
  EXPECT_NE(Read(second).find(R"("value":2)"), std::string::npos);
  managed_.WaitRetiring();
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(
      Read(
          Send(
              "POST", "/v1/models/register",
              R"({"kind":"embedding","name":"dynamic","source":"model","assets":"assets"})"))
          .find("HTTP/1.1 409"),
      0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  managed_.ReleaseRetirement();
  while (Read(Send("GET", "/v1/models")).find(R"("id":"dynamic")") !=
         std::string::npos) {
  }
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterRegistrationTest,
       AbortAndServerStopStillWaitForActualModelOwnership) {
  Register();
  const int first = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  managed_.service.WaitSubmitted();
  const int queued = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  managed_.service.WaitAccepted(2);
  Control("drain", "dynamic");
  const int held = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  WaitHeld("dynamic", 1);
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"dynamic","mode":"abort"})"))
                .find("HTTP/1.1 202"),
            0u);
  managed_.service.WaitStopping();
  EXPECT_EQ(Read(held).find("HTTP/1.1 410"), 0u);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  managed_.service.Release();
  EXPECT_TRUE(Read(first).empty());
  EXPECT_TRUE(Read(queued).empty());
  managed_.WaitRetiring();
  loom_serve_http_server_request_stop(server_);
  EXPECT_FALSE(returned_.load());
  managed_.ReleaseRetirement();
  owner_.join();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterRegistrationTest,
       ReleaseFailureRetainsOwnershipUntilExplicitRetry) {
  managed_.fail_release = true;
  Register();
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  managed_.ReleaseRetirement();
  std::string status;
  do {
    status = Read(Send("GET", "/v1/status"));
  } while (status.find(R"("registration":"release_failed")") ==
           std::string::npos);
  EXPECT_NE(status.find("platform release failed"), std::string::npos);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_EQ(managed_.release_attempts.load(), 1u);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  while (Read(Send("GET", "/v1/models")).find(R"("id":"dynamic")") !=
         std::string::npos) {
  }
  EXPECT_EQ(managed_.release_attempts.load(), 2u);
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterTest, MetricsCopyHeldAndQueuedOwnershipWithEscapedLabels) {
  const int held = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  const int queued = Send("POST", "/v1/test", R"({"model":"b"})");
  b_.WaitAccepted(1);
  const auto metrics = Read(Send("GET", "/metrics"));
  EXPECT_EQ(
      metrics,
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
      "Connection: close\r\n\r\n"
      R"(# HELP loom_serve_model_queued_requests Requests outside the active invocation or cohort.
# TYPE loom_serve_model_queued_requests gauge
loom_serve_model_queued_requests{model="a",kind="fixture"} 0
loom_serve_model_queued_requests{model="b",kind="fixture"} 1
loom_serve_model_queued_requests{model="held",kind="fixture"} 0
loom_serve_model_queued_requests{model="other",kind="fixture"} 0
loom_serve_model_queued_requests{model="quoted\"\\name",kind="fixture"} 0
# HELP loom_serve_model_active_requests Requests owned by the active invocation or cohort, not hardware utilization.
# TYPE loom_serve_model_active_requests gauge
loom_serve_model_active_requests{model="a",kind="fixture"} 0
loom_serve_model_active_requests{model="b",kind="fixture"} 0
loom_serve_model_active_requests{model="held",kind="fixture"} 1
loom_serve_model_active_requests{model="other",kind="fixture"} 0
loom_serve_model_active_requests{model="quoted\"\\name",kind="fixture"} 0
# HELP loom_serve_model_phase Current scheduler ownership phase.
# TYPE loom_serve_model_phase gauge
loom_serve_model_phase{model="a",kind="fixture",phase="idle"} 1
loom_serve_model_phase{model="b",kind="fixture",phase="idle"} 1
loom_serve_model_phase{model="held",kind="fixture",phase="held"} 1
loom_serve_model_phase{model="other",kind="fixture",phase="idle"} 1
loom_serve_model_phase{model="quoted\"\\name",kind="fixture",phase="idle"} 1
# HELP loom_serve_model_oldest_queued_age_seconds Time since the oldest queued request was accepted, or zero if empty.
# TYPE loom_serve_model_oldest_queued_age_seconds gauge
loom_serve_model_oldest_queued_age_seconds{model="a",kind="fixture"} 0
loom_serve_model_oldest_queued_age_seconds{model="b",kind="fixture"} 12.5
loom_serve_model_oldest_queued_age_seconds{model="held",kind="fixture"} 0
loom_serve_model_oldest_queued_age_seconds{model="other",kind="fixture"} 0
loom_serve_model_oldest_queued_age_seconds{model="quoted\"\\name",kind="fixture"} 0
# HELP loom_serve_model_oldest_active_age_seconds Time since the oldest active request was accepted, including prior queue residence, or zero if empty.
# TYPE loom_serve_model_oldest_active_age_seconds gauge
loom_serve_model_oldest_active_age_seconds{model="a",kind="fixture"} 0
loom_serve_model_oldest_active_age_seconds{model="b",kind="fixture"} 0
loom_serve_model_oldest_active_age_seconds{model="held",kind="fixture"} 2.25
loom_serve_model_oldest_active_age_seconds{model="other",kind="fixture"} 0
loom_serve_model_oldest_active_age_seconds{model="quoted\"\\name",kind="fixture"} 0
# HELP loom_serve_model_held_requests Requests held outside model admission.
# TYPE loom_serve_model_held_requests gauge
# HELP loom_serve_model_admission_open Whether new requests may enter the model.
# TYPE loom_serve_model_admission_open gauge
loom_serve_model_held_requests{model="a",kind="fixture"} 0
loom_serve_model_admission_open{model="a",kind="fixture"} 1
loom_serve_model_held_requests{model="b",kind="fixture"} 0
loom_serve_model_admission_open{model="b",kind="fixture"} 1
loom_serve_model_held_requests{model="held",kind="fixture"} 0
loom_serve_model_admission_open{model="held",kind="fixture"} 1
loom_serve_model_held_requests{model="other",kind="fixture"} 0
loom_serve_model_admission_open{model="other",kind="fixture"} 1
loom_serve_model_held_requests{model="quoted\"\\name",kind="fixture"} 0
loom_serve_model_admission_open{model="quoted\"\\name",kind="fixture"} 1
# HELP loom_serve_server_counters_available Every model publishes work counters.
# TYPE loom_serve_server_counters_available gauge
loom_serve_server_counters_available 0
# HELP loom_serve_model_requests_total Requests reaching a model protocol, including rejections.
# TYPE loom_serve_model_requests_total counter
# HELP loom_serve_model_completed_total Successful responses copied into the HTTP carrier.
# TYPE loom_serve_model_completed_total counter
# HELP loom_serve_model_rejected_total Protocol, source validation or capacity rejections.
# TYPE loom_serve_model_rejected_total counter
# HELP loom_serve_model_cancelled_total Requests abandoned by peers or orderly shutdown.
# TYPE loom_serve_model_cancelled_total counter
# HELP loom_serve_model_failed_total Requests terminated by transport or execution failures.
# TYPE loom_serve_model_failed_total counter
# HELP loom_serve_model_input_tokens_total Tokens in completed model work, including abandoned work.
# TYPE loom_serve_model_input_tokens_total counter
# HELP loom_serve_model_output_tokens_total Selected model output tokens, including EOS.
# TYPE loom_serve_model_output_tokens_total counter
# HELP loom_serve_model_batches_total Completed model invocations or cohorts, not HTTP requests.
# TYPE loom_serve_model_batches_total counter
)");
  const auto log = Read(Send("GET", "/logz"));
  EXPECT_NE(log.find("\"counters_available\":false"), std::string::npos);
  EXPECT_EQ(log.find("\"event\":\"counters\""), std::string::npos);
  // Scraping changes no model ownership and does not wait on the held work.
  EXPECT_NE(
      Read(Send("GET", "/v1/status"))
          .find(R"("phase":"held","queued_requests":0,"active_requests":1)"),
      std::string::npos);
  held_.Release();
  EXPECT_NE(Read(held).find("held:/v1/test:"), std::string::npos);
  EXPECT_NE(Read(queued).find("b:/v1/test:"), std::string::npos);
  const auto idle = Read(Send("GET", "/metrics"));
  EXPECT_NE(idle.find("loom_serve_model_active_requests{model=\"held\","
                      "kind=\"fixture\"} 0\n"),
            std::string::npos);
  EXPECT_NE(idle.find("loom_serve_model_queued_requests{model=\"b\","
                      "kind=\"fixture\"} 0\n"),
            std::string::npos);
  EXPECT_NE(idle.find("loom_serve_model_phase{model=\"held\","
                      "kind=\"fixture\",phase=\"idle\"} 1\n"),
            std::string::npos);
  EXPECT_EQ(idle.find("phase=\"held\""), std::string::npos);
  EXPECT_NE(
      idle.find("loom_serve_model_oldest_active_age_seconds{model=\"held\","
                "kind=\"fixture\"} 0\n"),
      std::string::npos);
  EXPECT_NE(idle.find("loom_serve_model_oldest_queued_age_seconds{model=\"b\","
                      "kind=\"fixture\"} 0\n"),
            std::string::npos);
}

TEST_P(HttpRouterTest, UnrelatedCapacityNeverWaitsForHeldCompletion) {
  const int held = Send("POST", "/v1/test", R"({"model":"other"})");
  other_.WaitSubmitted();
  // Both dependencies honor SHARED_ACTIVITY. The held service has no shared
  // domain: neither a named owner nor another NULL identity can wait on it.
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"b"})")).find("b:/v1/test:"),
      std::string::npos);
  EXPECT_NE(Read(Send("POST", "/v1/test", R"({"model":"quoted\"\\name"})"))
                .find(std::string(escaped_.name) + ":/v1/test:"),
            std::string::npos);
  const auto state = Read(Send("GET", "/v1/status"));
  EXPECT_NE(
      state.find(R"("phase":"held","queued_requests":0,"active_requests":1)"),
      std::string::npos);
  other_.Release();
  EXPECT_NE(Read(held).find("other:/v1/test:"), std::string::npos);
}

TEST_P(HttpRouterTest, RawUtilitiesProgressWhileTheSelectedModelIsHeld) {
  const int held = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  std::vector<int> clients;
  for (int i = 0; i < 12; ++i) {
    const char* name = i % 2 ? "b" : "held";
    clients.push_back(
        Send("POST", "/tokenize",
             std::string("{\"model\":\"") + name + "\",\"content\":\"aab\"}"));
  }
  EXPECT_NE(Read(Send("GET", "/healthz")).find("\"status\":\"ready\""),
            std::string::npos);
  EXPECT_NE(Read(Send("GET", "/readyz")).find(R"({"status":"ready"})"),
            std::string::npos);
  const auto snapshot = Read(Send("GET", "/v1/status"));
  EXPECT_NE(
      snapshot.find(
          R"({"object":"model","id":"held","loom":{"kind":"fixture","endpoint":"/v1/test","tokenize":true,"status":{"phase":"held","queued_requests":0,"active_requests":1,"oldest_queued_age_seconds":0.000000000,"oldest_active_age_seconds":2.250000000,"admission":"open","held_requests":0,"tokenizer_requests":)"),
      std::string::npos);
  for (int i = 0; i < 12; ++i) {
    const auto response = Read(clients[i]);
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
    EXPECT_NE(response.find(i % 2 ? R"({"tokens":[1,1,0]})"
                                  : R"({"tokens":[0,0,1]})"),
              std::string::npos);
  }
  const auto decoded =
      Read(Send("POST", "/detokenize", R"({"model":"held","tokens":[1,0,0]})"));
  // No utility reached the model dependency or required its device frontier.
  held_.Release();
  EXPECT_NE(decoded.find(R"({"content":"baa"})"), std::string::npos);
  if (decoded.find("HTTP/1.1 200") == 0) {
    EXPECT_NE(Read(held).find("held:/v1/test:"), std::string::npos);
  } else {
    // A broken utility may stop the owner; release its held claim in teardown
    // instead of circularly waiting for a response after that failure.
    EXPECT_EQ(close(held), 0);
  }
}

TEST_P(HttpRouterTest, UtilityRejectionsAndResetPeersPreserveTheOwner) {
  EXPECT_EQ(
      Read(Send("POST", "/tokenize", R"({"model":"other","content":"a"})"))
          .find("HTTP/1.1 404"),
      0u);
  // This deployment has an encoder but no decoder. Its CPU failure belongs
  // to this request and must leave every model and later utility usable.
  EXPECT_EQ(Read(Send("POST", "/detokenize", R"({"model":"b","tokens":[0]})"))
                .find("HTTP/1.1 500"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/tokenize",
                      R"({"model":"a","content":"a","with_pieces":true})"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/detokenize", R"({"model":"a","tokens":[2]})"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/tokenize",
                      "{\"model\":\"a\",\"content\":\"" +
                          std::string(256 * 1024 + 1, 'a') + "\"}"))
                .find("HTTP/1.1 413"),
            0u);
  for (int i = 0; i < 8; ++i) {
    const int client = Send("POST", "/tokenize",
                            "{\"model\":\"a\",\"content\":\"" +
                                std::string(64 * 1024, 'a') + "\"}");
    const linger reset = {1, 0};
    EXPECT_EQ(setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
              0);
    EXPECT_EQ(close(client), 0);
  }
  EXPECT_NE(Read(Send("POST", "/tokenize", R"({"model":"a","content":"ab"})"))
                .find(R"({"tokens":[0,1]})"),
            std::string::npos);
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"b"})")).find("b:/v1/test:"),
      std::string::npos);
}

TEST_P(HttpRouterTest, DiscoveryAndIndependentBorrowedRequests) {
  EXPECT_NE(Read(Send("GET", "/healthz")).find(R"({"status":"ready"})"),
            std::string::npos);
  const auto models = Read(Send("GET", "/v1/models"));
  EXPECT_NE(
      models.find(
          R"({"object":"model","id":"a","loom":{"kind":"fixture","endpoint":"/v1/test","tokenize":true}})"),
      std::string::npos);
  EXPECT_NE(
      models.find(
          R"({"object":"model","id":"other","loom":{"kind":"fixture","endpoint":"/v1/test","tokenize":false}})"),
      std::string::npos);
  EXPECT_EQ(models.find("\"status\":"), std::string::npos);
  const std::string body_a = R"({"model":"\u0061","messages":["first"]})";
  const std::string body_b = R"({"model":"b","messages":["second"]})";
  const int a = Send("POST", "/v1/chat/completions", body_a);
  const int b = Send("POST", "/v1/chat/completions", body_b);
  EXPECT_NE(Read(b).find("b:/v1/chat/completions:" + body_b),
            std::string::npos);
  EXPECT_NE(Read(a).find("a:/v1/chat/completions:" + body_a),
            std::string::npos);
  EXPECT_NE(
      Read(Send("POST", "/v1/checkpoints/same", "", "X-Loom-Model: b\r\n"))
          .find("b:/v1/checkpoints/same:"),
      std::string::npos);
  EXPECT_NE(
      Read(Send("DELETE", "/v1/checkpoints/same", "", "X-Loom-Model: a\r\n"))
          .find("a:/v1/checkpoints/same:"),
      std::string::npos);
}

TEST_P(HttpRouterTest, RejectsAmbiguousSelectorsWithoutInvokingModels) {
  for (const auto* body : {"{}", R"({"model":null})", R"({"model":4})",
                           R"({"model":""})", R"({"model":"a","model":"b"})",
                           R"({"model":"a"} trailing)", R"({"model":"\q"})"}) {
    SCOPED_TRACE(body);
    EXPECT_EQ(
        Read(Send("POST", "/v1/chat/completions", body)).find("HTTP/1.1 400"),
        0u);
  }
  EXPECT_EQ(Read(Send("POST", "/v1/chat/completions", R"({"model":"missing"})"))
                .find("HTTP/1.1 404"),
            0u);
  EXPECT_EQ(Read(Send("POST", "/v1/checkpoints/same")).find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("DELETE", "/v1/checkpoints/same", "",
                      "X-Loom-Model: missing\r\n"))
                .find("HTTP/1.1 404"),
            0u);
  // Rejections leave the listener and both services available.
  EXPECT_NE(Read(Send("POST", "/v1/chat/completions", R"({"model":"b"})"))
                .find("b:/v1/chat/completions:"),
            std::string::npos);
}

TEST_P(HttpRouterTest, HeldModelLeavesIndependentRequestsRunnable) {
  const std::string body = R"({"model":"held","messages":["retained"]})";
  const int held = Send("POST", "/v1/chat/completions", body);
  held_.WaitSubmitted();
  EXPECT_NE(Read(Send("POST", "/v1/chat/completions", R"({"model":"a"})"))
                .find("a:/v1/chat/completions:"),
            std::string::npos);
  EXPECT_NE(Read(Send("GET", "/healthz")).find(R"({"status":"ready"})"),
            std::string::npos);
  held_.Release();
  EXPECT_NE(Read(held).find("held:/v1/chat/completions:" + body),
            std::string::npos);
}

TEST_P(HttpRouterTest, DrainFinishesAdmittedQueueAndHoldsLaterModelBorrowers) {
  const int held = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  const int first = Send("POST", "/v1/test", R"({"model":"b","turn":1})");
  const int second = Send("POST", "/v1/test", R"({"model":"b","turn":2})");
  b_.WaitAccepted(2);
  const auto draining = Control("drain", "b");
  EXPECT_NE(draining.find(R"("queued_requests":2,"active_requests":0)"),
            std::string::npos);
  EXPECT_NE(draining.find(R"("admission":"draining")"), std::string::npos);

  const int later = Send("POST", "/v1/test", R"({"model":"b","turn":3})");
  const int tokens =
      Send("POST", "/tokenize", R"({"model":"b","content":"aab"})");
  EXPECT_NE(WaitHeld("b", 2).find(R"("tokenizer_requests":0)"),
            std::string::npos);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"b"})")).find("HTTP/1.1 429"),
      0u);
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("a:/v1/test:"),
      std::string::npos);
  const auto metrics = Read(Send("GET", "/metrics"));
  EXPECT_NE(metrics.find("loom_serve_model_queued_requests{model=\"b\","
                         "kind=\"fixture\"} 2\n"),
            std::string::npos);
  EXPECT_NE(metrics.find("loom_serve_model_held_requests{model=\"b\","
                         "kind=\"fixture\"} 2\n"),
            std::string::npos);
  EXPECT_NE(metrics.find("loom_serve_model_admission_open{model=\"b\","
                         "kind=\"fixture\"} 0\n"),
            std::string::npos);

  held_.Release();
  EXPECT_NE(Read(held).find("held:/v1/test:"), std::string::npos);
  EXPECT_NE(Read(first).find(R"(b:/v1/test:{"model":"b","turn":1})"),
            std::string::npos);
  EXPECT_NE(Read(second).find(R"(b:/v1/test:{"model":"b","turn":2})"),
            std::string::npos);
  const auto drained = Control("drain", "b");
  EXPECT_NE(drained.find(R"("queued_requests":0,"active_requests":0)"),
            std::string::npos);
  EXPECT_NE(drained.find(R"("admission":"drained","held_requests":2)"),
            std::string::npos);
  {
    std::lock_guard<std::mutex> lock(b_.mutex);
    EXPECT_EQ(b_.accepted, 2u);
  }
  EXPECT_NE(Control("undrain", "b").find(R"("admission":"open")"),
            std::string::npos);
  EXPECT_NE(Read(later).find(R"(b:/v1/test:{"model":"b","turn":3})"),
            std::string::npos);
  EXPECT_NE(Read(tokens).find(R"({"tokens":[1,1,0]})"), std::string::npos);
}

TEST_P(HttpRouterTest, AdmissionControlRejectsInvalidSelectorsWithoutClosing) {
  for (const auto* operation : {"drain", "undrain"}) {
    const std::string target = std::string("/v1/models/") + operation;
    for (const auto* body :
         {"{}", R"({"model":null})", R"({"model":""})",
          R"({"model":"a","model":"b"})", R"({"model":"a","extra":true})",
          R"({"model":"a"} trailing)"}) {
      SCOPED_TRACE(body);
      EXPECT_EQ(Read(Send("POST", target, body)).find("HTTP/1.1 400"), 0u);
    }
    EXPECT_EQ(
        Read(Send("GET", target, R"({"model":"a"})")).find("HTTP/1.1 404"), 0u);
    EXPECT_EQ(Control(operation, "missing").find("HTTP/1.1 404"), 0u);
  }
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("a:/v1/test:"),
      std::string::npos);
}

TEST_P(HttpRouterTest, DisconnectedHeldClaimReturnsItsAdmissionCredit) {
  EXPECT_NE(Control("drain", "a").find(R"("admission":"drained")"),
            std::string::npos);
  const int abandoned = Send("POST", "/v1/test", R"({"model":"a"})");
  WaitHeld("a", 1);
  const linger reset = {1, 0};
  EXPECT_EQ(setsockopt(abandoned, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  EXPECT_EQ(close(abandoned), 0);
  WaitHeld("a", 0);
  const int first = Send("POST", "/v1/test", R"({"model":"a"})");
  const int second =
      Send("POST", "/tokenize", R"({"model":"a","content":"ab"})");
  WaitHeld("a", 2);
  EXPECT_EQ(Control("undrain", "a").find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(first).find("a:/v1/test:"), std::string::npos);
  EXPECT_NE(Read(second).find(R"({"tokens":[0,1]})"), std::string::npos);
}

TEST_P(HttpRouterTest, ShutdownReleasesHeldClaimsWithoutAdmittingThem) {
  EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
  const int client = Send("POST", "/v1/test", R"({"model":"a"})");
  WaitHeld("a", 1);
  loom_serve_http_server_request_stop(server_);
  owner_.join();
  EXPECT_EQ(close(client), 0);
  EXPECT_EQ(a_.accepted, 0u);
}

TEST_P(HttpRouterTest, RejectsAQuotaThatCouldExcludeEveryControlConnection) {
  for (const auto options : {
           loom_serve_router_options_t{.request_quota = 0,
                                       .holding_capacity = 0},
           loom_serve_router_options_t{
               .request_quota = 4, .holding_capacity = connection_capacity_},
           loom_serve_router_options_t{
               .request_quota = 4,
               .holding_capacity = connection_capacity_ + 1},
       }) {
    // Validation happens before the owner installs any transport observer.
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_router_run(executor_, server_, nullptr, options,
                              iree_allocator_system()));
  }
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
}

TEST_P(HttpRouterTest, ShutdownDrainsAcceptedReadinessBeforeDetaching) {
  const int held = Send("POST", "/v1/chat/completions", R"({"model":"held"})");
  held_.WaitSubmitted();
  loom_serve_http_server_request_stop(server_);
  held_.WaitStopping();
  EXPECT_FALSE(returned_.load());
  held_.Release();
  owner_.join();
  EXPECT_EQ(close(held), 0);
}

TEST_P(HttpRouterTest, FailedReadinessStillDrainsOtherAcceptedCallbacks) {
  const int held = Send("POST", "/v1/chat/completions", R"({"model":"held"})");
  const int other =
      Send("POST", "/v1/chat/completions", R"({"model":"other"})");
  held_.WaitSubmitted();
  other_.WaitSubmitted();
  expected_code_ = IREE_STATUS_UNAVAILABLE;
  held_.Release(iree_make_status(IREE_STATUS_UNAVAILABLE, "device failed"));
  other_.WaitStopping();
  EXPECT_FALSE(returned_.load());
  other_.Release();
  owner_.join();
  EXPECT_EQ(close(held), 0);
  EXPECT_EQ(close(other), 0);
}

// Real finite CPU task on the router's executor. Release, not completion,
// returns its stack storage to the test owner.
struct CpuCompletion {
  // Borrowed dependency receiving this task's readiness publication.
  ModelService* service;
  // Intrusive executor ownership, outliving its final release callback.
  iree_task_process_t process = {};
  // Protects final release and joins its last access.
  std::mutex mutex;
  // Wakes only the test thread, never a task worker.
  std::condition_variable condition;
  // Executor has returned all intrusive process storage.
  bool released = false;

  static iree_status_t Run(iree_task_process_t* process,
                           const iree_task_worker_context_t*,
                           iree_task_process_drain_result_t* result) {
    auto* self = static_cast<CpuCompletion*>(process->user_data);
    self->service->Release();
    result->completed = true;
    result->did_work = true;
    return iree_ok_status();
  }
  static void Released(iree_task_process_t* process) {
    auto* self = static_cast<CpuCompletion*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->released = true;
    self->condition.notify_all();
  }
  void Submit(iree_task_executor_t* executor) {
    iree_task_process_initialize(Run, 0, 1, &process);
    process.user_data = this;
    process.release_fn = Released;
    iree_task_executor_schedule_process(executor, &process);
  }
  void AwaitRelease() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return released; });
  }
};

TEST_P(HttpRouterTest, ContinuouslyReadyOwnerYieldsToSharedCpuWork) {
  {
    std::lock_guard<std::mutex> lock(held_.mutex);
    held_.advance_ready_work = true;
  }
  const std::string body = R"({"model":"held"})";
  const int held = Send("POST", "/v1/chat/completions", body);
  held_.WaitSubmitted();
  EXPECT_NE(Read(Send("GET", "/healthz")).find(R"({"status":"ready"})"),
            std::string::npos);
  CpuCompletion cpu{&held_};
  cpu.Submit(executor_);
  EXPECT_NE(Read(held).find("held:/v1/chat/completions:" + body),
            std::string::npos);
  cpu.AwaitRelease();
  std::lock_guard<std::mutex> lock(held_.mutex);
  EXPECT_GT(held_.ready_units, 0u);
}

class HttpRouterCountersTest : public HttpRouterTest {
 protected:
  void SetUp() override {
    size_t index = 0;
    for (auto* service : {&a_, &b_, &held_, &other_, &escaped_}) {
      const uint64_t value = index + 1;
      counters_[index] = {.requests = value,
                          .completed = value,
                          .input_tokens = (uint64_t{1} << 53) + value,
                          .output_tokens = 7 * value,
                          .batches = value};
      service->counters = &counters_[index++];
    }
    HttpRouterTest::SetUp();
  }
  // Distinct service owners, unchanged by cold observation/utility requests.
  loom_serve_telemetry_counters_t counters_[5] = {};
};

class HttpRouterTightCapacityTest : public HttpRouterTest {
 protected:
  void SetUp() override {
    connection_capacity_ = 3;
    HttpRouterTest::SetUp();
  }
};

TEST_P(HttpRouterTightCapacityTest,
       FullHoldingBudgetLeavesAnUndrainConnection) {
  EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
  const int first = Send("POST", "/v1/test", R"({"model":"a"})");
  WaitHeld("a", 1);
  const int second =
      Send("POST", "/tokenize", R"({"model":"a","content":"ab"})");
  WaitHeld("a", 2);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 429"),
      0u);
  EXPECT_EQ(Control("undrain", "a").find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(first).find("a:/v1/test:"), std::string::npos);
  EXPECT_NE(Read(second).find(R"({"tokens":[0,1]})"), std::string::npos);
}

TEST_P(HttpRouterTightCapacityTest, ObservationAndHoldingShareControlCredit) {
  EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
  const int held = Send("POST", "/v1/test", R"({"model":"a"})");
  WaitHeld("a", 1);
  {
    ObservationClient observer;
    observer.Open(server_);
    observer.WaitCredit(false);
    EXPECT_EQ(Read(Send("GET", "/logz/stream")).find("HTTP/1.1 429"), 0u);
    EXPECT_EQ(
        Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 429"),
        0u);
    EXPECT_EQ(Control("undrain", "a").find("HTTP/1.1 200"), 0u);
    EXPECT_NE(Read(held).find("a:/v1/test:"), std::string::npos);
    // Client departure returns observation ownership independently of shutdown.
  }
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
}

class HttpRouterNoHoldingTest : public HttpRouterTest {
 protected:
  void SetUp() override {
    holding_capacity_ = 0;
    HttpRouterTest::SetUp();
  }
};

TEST_P(HttpRouterNoHoldingTest, ClosedAdmissionRejectsWithoutRetainingClaims) {
  EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 429"),
      0u);
  EXPECT_EQ(Read(Send("POST", "/tokenize", R"({"model":"a","content":"a"})"))
                .find("HTTP/1.1 429"),
            0u);
  EXPECT_NE(Control("undrain", "a").find(R"("held_requests":0)"),
            std::string::npos);
  EXPECT_NE(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("a:/v1/test:"),
      std::string::npos);
}

TEST_P(HttpRouterCountersTest, ServerTotalsPreserveIntegerPrecisionAndScope) {
  for (int i = 0; i < 2; ++i) {
    const auto metrics = Read(Send("GET", "/metrics"));
    EXPECT_NE(metrics.find("loom_serve_model_input_tokens_total{model=\"a\","
                           "kind=\"fixture\"} 9007199254740993\n"),
              std::string::npos);
    EXPECT_NE(metrics.find("loom_serve_server_requests_total 15\n"),
              std::string::npos);
    EXPECT_NE(metrics.find(
                  "loom_serve_server_input_tokens_total 45035996273704975\n"),
              std::string::npos);
    const auto log = Read(Send("GET", "/logz"));
    EXPECT_NE(log.find("\"counters_available\":true"), std::string::npos);
    EXPECT_NE(
        log.find("\"values\":{\"requests\":15,\"completed\":15,\"rejected\":0,"
                 "\"cancelled\":0,\"failed\":0,\"input_tokens\":"
                 "45035996273704975,\"output_tokens\":105,\"batches\":15}"),
        std::string::npos);
    EXPECT_NE(Read(Send("POST", "/tokenize", R"({"model":"a","content":"ab"})"))
                  .find("\"tokens\":[0,1]"),
              std::string::npos);
    EXPECT_NE(Read(Send("POST", "/v1/test", R"({"model":"missing"})"))
                  .find("404 Not Found"),
              std::string::npos);
  }
}

INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterTest, ::testing::Values(1, 4));
INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterRegistrationTest,
                         ::testing::Values(1, 4));
INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterCountersTest,
                         ::testing::Values(1, 4));
INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterTightCapacityTest,
                         ::testing::Values(1, 4));
INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterNoHoldingTest,
                         ::testing::Values(1, 4));

}  // namespace
