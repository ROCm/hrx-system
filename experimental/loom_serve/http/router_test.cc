// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/router.h"

#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/task/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

namespace {

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

  static iree_status_t Accept(void* self,
                              loom_serve_http_connection_t* connection,
                              const loom_serve_http_request_t* request,
                              loom_serve_http_service_flags_t flags) {
    (void)flags;
    static_cast<ModelService*>(self)->pending.emplace_back(connection, request);
    return iree_ok_status();
  }
  static iree_status_t Advance(void* self,
                               loom_serve_http_service_flags_t flags,
                               bool* progress) {
    auto* service = static_cast<ModelService*>(self);
    *progress = false;
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
};

class HttpRouterTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_signal_block_default());
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(GetParam(), &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    iree_status_t status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    auto options = loom_serve_http_server_options_default();
    options.port = 0;
    IREE_ASSERT_OK(loom_serve_http_server_create(&options, &server_,
                                                 iree_allocator_system()));
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
          {IREE_SV("a"), &a_, ModelService::SetObserver,
           ModelService::IsPending, ModelService::IsActive,
           ModelService::Accept, ModelService::Advance, tokenizer_a_},
          {IREE_SV("b"), &b_, ModelService::SetObserver,
           ModelService::IsPending, ModelService::IsActive,
           ModelService::Accept, ModelService::Advance, tokenizer_b_},
          {IREE_SV("held"), &held_, ModelService::SetObserver,
           ModelService::IsPending, ModelService::IsActive,
           ModelService::Accept, ModelService::Advance, tokenizer_a_},
          {IREE_SV("other"), &other_, ModelService::SetObserver,
           ModelService::IsPending, ModelService::IsActive,
           ModelService::Accept, ModelService::Advance, nullptr}};
      status_ = loom_serve_http_router_run(executor_, server_,
                                           IREE_ARRAYSIZE(services), services,
                                           4, iree_allocator_system());
      returned_.store(true);
    });
  }
  void TearDown() override {
    if (owner_.joinable()) {
      EXPECT_EQ(kill(getpid(), SIGTERM), 0);
      owner_.join();
    }
    IREE_EXPECT_STATUS_IS(expected_code_, status_);
    for (auto* service : {&a_, &b_, &held_, &other_}) {
      for (auto& [connection, request] : service->pending) {
        (void)request;
        loom_serve_http_connection_abort(connection);
      }
    }
    IREE_EXPECT_OK(loom_serve_http_server_destroy(server_));
    iree_tokenizer_free(tokenizer_b_);
    iree_tokenizer_free(tokenizer_a_);
    iree_task_executor_release(executor_);
  }
  int Send(const std::string& method, const std::string& target,
           const std::string& body = "", const std::string& headers = "") {
    const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    const auto* address = loom_serve_http_server_address(server_);
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
  // Real loopback transport and its separate proactor owner.
  loom_serve_http_server_t* server_ = nullptr;
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
  // Outer application caller joining the production router's final release.
  std::thread owner_;
  // Terminal result published by joining owner_.
  iree_status_t status_ = iree_ok_status();
  // Owner return observation independent of its joined terminal status.
  std::atomic<bool> returned_{false};
  // Expected terminal result; service errors are not hidden by shutdown.
  iree_status_code_t expected_code_ = IREE_STATUS_OK;
};

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
  EXPECT_NE(models.find(R"({"object":"model","id":"a"})"), std::string::npos);
  EXPECT_NE(models.find(R"({"object":"model","id":"b"})"), std::string::npos);
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

TEST_P(HttpRouterTest, ShutdownDrainsAcceptedReadinessBeforeDetaching) {
  const int held = Send("POST", "/v1/chat/completions", R"({"model":"held"})");
  held_.WaitSubmitted();
  EXPECT_EQ(kill(getpid(), SIGTERM), 0);
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

INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterTest, ::testing::Values(1, 4));

}  // namespace
