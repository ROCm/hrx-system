// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/finite/service.h"

#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "experimental/loom_serve/embedding/protocol.h"
#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

IREE_FLAG(string, model_source, "",
          "Deterministic embedding bootstrap source.");

namespace {

class SourceEmbeddingServiceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  virtual iree_host_size_t DeviceCount() const { return 1; }

  void SetUp() override {
    iree_notification_initialize(&notification_);
    IREE_ASSERT_OK(iree_async_signal_block_default());
    IREE_ASSERT_OK(loom_serve_host_create(GetParam(), &host_, allocator_));
    const loom_serve_device_options_t device_options[] = {
        {.uri = IREE_SV("amdgpu")}, {.uri = IREE_SV("amdgpu")}};
    IREE_ASSERT_OK(loom_serve_device_group_create(
        host_, DeviceCount(), device_options, &group_, allocator_));
    device_ = loom_serve_device_group_at(group_, DeviceCount() - 1);
    const std::string source = FLAG_model_source;
    directory_ = source.substr(0, source.find_last_of('/'));
    ASSERT_NO_FATAL_FAILURE(CreateModel(directory_));
    ASSERT_NO_FATAL_FAILURE(CreateService(0));
    auto options = loom_serve_http_server_options_default();
    options.port = 0;
    IREE_ASSERT_OK(
        loom_serve_http_server_create(&options, &server_, allocator_));
  }

  void CreateModel(const std::string& directory) {
    const loom_serve_finite_model_options_t model_options = {
        .source_directory = iree_make_cstring_view(directory.c_str()),
        .asset_directory = iree_make_cstring_view(directory.c_str()),
        .options = IREE_SV("{\"batch_units\":64}")};
    IREE_ASSERT_OK(loom_serve_finite_model_create(device_, &model_options,
                                                  &model_, allocator_));
    EXPECT_EQ(loom_serve_finite_model_batch_limit(model_), 64u);
  }

  void CreateService(iree_duration_t heartbeat_interval) {
    IREE_ASSERT_OK(loom_serve_finite_service_create(
        host_, model_, loom_serve_embedding_protocol(), IREE_SV("fixture"), 16,
        heartbeat_interval, &service_, allocator_));
    interface_ = loom_serve_finite_service_interface(service_);
    EXPECT_EQ(interface_.capacity_domain,
              loom_serve_device_residency_cache(device_));
    interface_.set_observer(
        interface_.self,
        {[](void* self) {
           auto* test = static_cast<SourceEmbeddingServiceTest*>(self);
           test->wakes_.fetch_add(1);
           iree_notification_post(&test->notification_, IREE_ALL_WAITERS);
         },
         this});
  }

  void TearDown() override {
    if (router_.joinable()) {
      EXPECT_EQ(kill(getpid(), SIGTERM), 0);
      router_.join();
      IREE_EXPECT_OK(status_);
    }
    if (service_) {
      interface_.set_observer(interface_.self, {});
      loom_serve_finite_service_destroy(service_);
    }
    IREE_EXPECT_OK(loom_serve_http_server_destroy(server_));
    IREE_EXPECT_OK(loom_serve_finite_model_destroy(model_));
    IREE_EXPECT_OK(loom_serve_finite_model_destroy(retained_));
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
    loom_serve_host_release(host_);
    iree_notification_deinitialize(&notification_);
  }

  int Send(const std::string& body, const char* target = "/v1/embeddings",
           const char* method = "POST") {
    const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    const auto* address = loom_serve_http_server_address(server_);
    EXPECT_EQ(
        connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                address->length),
        0);
    const std::string request = std::string(method) + " " + target +
                                " HTTP/1.1\r\nHost: local\r\nContent-Type: "
                                "application/json\r\nContent-Length: " +
                                std::to_string(body.size()) + "\r\n\r\n" + body;
    size_t offset = 0;
    while (offset < request.size()) {
      const ssize_t sent = send(client, request.data() + offset,
                                request.size() - offset, MSG_NOSIGNAL);
      EXPECT_GT(sent, 0);
      if (sent <= 0) {
        break;
      }
      offset += sent;
    }
    return client;
  }

  std::string Read(int client) {
    std::string response;
    char bytes[4096];
    ssize_t length;
    while ((length = recv(client, bytes, sizeof(bytes), 0)) > 0) {
      response.append(bytes, length);
    }
    EXPECT_EQ(length, 0);
    EXPECT_EQ(close(client), 0);
    return response;
  }

  // Admit already-arrived HTTP requests before the next scheduler turn. This
  // uses the production carrier/claim/service interfaces, with no fake model
  // or sleeps to manufacture coalescing.
  loom_serve_http_connection_t* Accept() {
    auto* notification = loom_serve_http_server_notification(server_);
    while (true) {
      const auto ticket = iree_notification_prepare_wait(notification);
      const loom_serve_http_request_t* request = nullptr;
      auto* connection = loom_serve_http_server_take_request(server_, &request);
      if (connection) {
        iree_notification_cancel_wait(notification);
        IREE_EXPECT_OK(
            interface_.accept(interface_.self, connection, request, 0));
        return connection;
      }
      iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }

  // One serialized turn followed by its explicit CPU-release/device wake.
  // This is an outer test driver, not a worker waiting on other worker work.
  bool Step(loom_serve_http_service_flags_t flags = 0) {
    const uint64_t previous = wakes_.load();
    bool progress = false;
    IREE_EXPECT_OK(interface_.advance(interface_.self, flags, &progress));
    if (interface_.is_active(interface_.self)) {
      struct Wait {
        // Shared release counter.
        std::atomic<uint64_t>* wakes;
        // Last event already consumed by this turn.
        uint64_t previous;
      } wait{&wakes_, previous};
      iree_notification_await(
          &notification_,
          [](void* data) {
            const auto* wait = static_cast<Wait*>(data);
            return wait->wakes->load() > wait->previous;
          },
          &wait, iree_infinite_timeout());
    }
    return progress;
  }

  void Drain() {
    while (Step()) {
    }
  }

  void StartRouter() {
    interface_.set_observer(interface_.self, {});
    router_ = std::thread([this] {
      status_ =
          loom_serve_http_router_run(loom_serve_host_executor(host_), server_,
                                     1, &interface_, 16, allocator_);
    });
  }

  // The replacement is ordinary model source and a tiny real checkpoint. Only
  // package files change; the same compiled native service loads both models.
  void WriteWeightedPackage(const std::string& target) {
    ASSERT_TRUE(std::filesystem::create_directory(target));
    for (const char* file :
         {"config.loom", "control.loom", "embed.loom", "tokenizer.json"}) {
      ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                             target + "/" + file));
    }
    for (const char* file : {"weights.loom", "policy.loom"}) {
      ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/variants/" + file,
                                             target + "/" + file));
    }
    std::ifstream input(directory_ + "/prepare.loom");
    ASSERT_TRUE(input.good());
    std::string prepare((std::istreambuf_iterator<char>(input)), {});
    const std::string root = "bytes(\"656d626564\")";
    const auto root_position = prepare.find(root);
    ASSERT_NE(root_position, std::string::npos);
    prepare.replace(root_position, root.size(),
                    "bytes(\"77656967687465645f656d626564\")");
    const auto result_position = prepare.find("  func.return %units,");
    ASSERT_NE(result_position, std::string::npos);
    prepare.insert(result_position, R"(
  %parameter_index = scalar.constant 0 : i32
  %parameter_path = global.load @checkpoint : buffer
  %parameter_policy = global.load @policy : buffer
  func.call @parameter(%stage, %parameter_index, %directory, %parameter_path, %parameter_policy) : (i32, i32, buffer, buffer, buffer) -> ()
)");
    const std::string declarations = R"(
func.decl import("prepare") @parameter(%stage: i32, %binding: i32, %directory: buffer, %path: buffer, %policy: buffer)
global.rodata.def @checkpoint = bytes("776569676874732e7361666574656e736f7273")
global.rodata.def @policy = bytes("706f6c6963792e6c6f6f6d")
)";
    std::ofstream bootstrap(target + "/prepare.loom");
    bootstrap << declarations << prepare;
    bootstrap.close();
    ASSERT_TRUE(bootstrap.good());
    std::ofstream manifest(target + "/sources.txt");
    manifest << "config.loom\nembed.loom\nweights.loom\n";
    manifest.close();
    ASSERT_TRUE(manifest.good());
    std::string header =
        R"({"factor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream weights(target + "/weights.safetensors", std::ios::binary);
    uint8_t length[8];
    iree_unaligned_store_le_u64(length, header.size());
    weights.write(reinterpret_cast<const char*>(length), sizeof(length));
    weights.write(header.data(), header.size());
    uint8_t factor[4];
    iree_unaligned_store_le_u32(factor, 0x40000000);  // F32 2.0.
    weights.write(reinterpret_cast<const char*>(factor), sizeof(factor));
    weights.close();
    ASSERT_TRUE(weights.good());
  }

  // Host allocator used by all production owners.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Shared CPU/JIT/task runtime.
  loom_serve_host_t* host_ = nullptr;
  // Application group enclosing every device and model resource.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed placement of the source-defined embedding model.
  loom_serve_device_t* device_ = nullptr;
  // Shared source-JIT finite model.
  loom_serve_finite_model_t* model_ = nullptr;
  // A second actual model retaining parameter capacity during the pressure
  // test.
  loom_serve_finite_model_t* retained_ = nullptr;
  // Source package directory, independent of temporary replacements.
  std::string directory_;
  // Production batched protocol service.
  loom_serve_finite_service_t* service_ = nullptr;
  // Serialized callbacks used by both direct turns and the shared router.
  loom_serve_http_service_t interface_ = {};
  // Production loopback carrier, independent of scheduler turns.
  loom_serve_http_server_t* server_ = nullptr;
  // Outer driver notification for actual task/queue readiness.
  iree_notification_t notification_;
  // Published callback generations.
  std::atomic<uint64_t> wakes_{0};
  // Optional outer shared-router thread.
  std::thread router_;
  // Router status read after join.
  iree_status_t status_ = iree_ok_status();
};

TEST_P(SourceEmbeddingServiceTest,
       CoalescesRaggedRequestsAndDropsOnlyResetPeer) {
  const int a = Send(R"({"model":"fixture","input":["a","bbb"]})");
  Accept();
  const int b = Send(R"({"model":"fixture","input":[2]})");
  auto* cancelled = Accept();
  const int c = Send(R"({"model":"fixture","input":[[2,0]],"dimensions":2})");
  Accept();
  const int bad = Send(R"({"model":"fixture","input":"a","dimensions":3})");
  Accept();
  EXPECT_TRUE(Step());  // All independent requests prepared.
  EXPECT_TRUE(Step());  // Rejection collected; valid cohort constructed.
  EXPECT_TRUE(Step());  // One command submitted, with compact feedback.
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(b, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)), 0);
  ASSERT_EQ(close(b), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* value) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(value));
      },
      cancelled, iree_infinite_timeout());
  Drain();
  const std::string result_a = Read(a);
  EXPECT_EQ(result_a.find("HTTP/1.1 200"), 0u);
  // Fourth component was computed on device from the *whole* cohort's four
  // rows, including the reset peer. Physical result rows were stored backwards.
  EXPECT_NE(result_a.find(R"("index":0,"embedding":[1,1,1,4])"),
            std::string::npos);
  EXPECT_NE(result_a.find(R"("index":1,"embedding":[6,3,2,4])"),
            std::string::npos);
  EXPECT_NE(result_a.find(R"("prompt_tokens":4,"total_tokens":4)"),
            std::string::npos);
  const std::string result_c = Read(c);
  EXPECT_NE(result_c.find(R"("index":0,"embedding":[4,2])"), std::string::npos);
  EXPECT_NE(result_c.find(R"("prompt_tokens":2,"total_tokens":2)"),
            std::string::npos);
  EXPECT_EQ(Read(bad).find("HTTP/1.1 400"), 0u);
  // The same source/queue/pool is reusable after feedback and CPU release.
  const int next = Send(R"({"model":"fixture","input":[0]})");
  Accept();
  Drain();
  EXPECT_NE(Read(next).find(R"("embedding":[1,1,1,1])"), std::string::npos);
}

TEST_P(SourceEmbeddingServiceTest,
       SourceCapacitySplitsWithoutChangingClientRows) {
  std::vector<int> clients;
  for (size_t i = 0; i < 4; ++i) {
    clients.push_back(Send("{\"model\":\"fixture\",\"input\":\"" +
                           std::string(31, 'b') + "\"}"));
    Accept();
  }
  Drain();
  for (int client : clients) {
    EXPECT_NE(Read(client).find(R"("embedding":[62,31,2,2])"),
              std::string::npos);
  }
}

TEST_P(SourceEmbeddingServiceTest,
       SharedRouterServesUtilitiesAndBothEncodings) {
  interface_.set_observer(interface_.self, {});
  loom_serve_finite_service_destroy(std::exchange(service_, nullptr));
  // Exercise heartbeat cancellation during normal router shutdown. The test
  // waits for ownership, not the deadline, so even an hour-long timer retires.
  ASSERT_NO_FATAL_FAILURE(CreateService(INT64_C(3600000000000)));
  StartRouter();
  EXPECT_NE(Read(Send("", "/v1/models", "GET")).find("fixture"),
            std::string::npos);
  EXPECT_NE(Read(Send(R"({"model":"fixture","content":"aba"})", "/tokenize"))
                .find(R"({"tokens":[0,1,0]})"),
            std::string::npos);
  const int a =
      Send(R"({"model":"fixture","input":["a","bbb"],"dimensions":2})");
  const int b = Send(
      R"({"model":"fixture","input":[[0],[1,1,1]],"dimensions":2,"encoding_format":"base64"})");
  EXPECT_NE(Read(Send("", "/healthz", "GET")).find("\"status\":\"ready\""),
            std::string::npos);
  const std::string floats = Read(a);
  EXPECT_NE(floats.find(R"("embedding":[1,1])"), std::string::npos);
  EXPECT_NE(floats.find(R"("embedding":[6,3])"), std::string::npos);
  const std::string encoded = Read(b);
  EXPECT_NE(encoded.find(R"("embedding":"AACAPwAAgD8=")"), std::string::npos);
  EXPECT_NE(encoded.find(R"("embedding":"AADAQAAAQEA=")"), std::string::npos);
  EXPECT_EQ(
      Read(Send(R"({"model":"fixture","input":[-1]})")).find("HTTP/1.1 400"),
      0u);
  EXPECT_EQ(Read(Send(R"({"model":"fixture","input":"a","dimensions":3})"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(
      Read(Send(R"({"model":"fixture","input":"a"})")).find("HTTP/1.1 200"),
      0u);
}

TEST_P(SourceEmbeddingServiceTest,
       ExternalWeightedPackageRetainsPreparedCohortUnderPressure) {
  const std::string package =
      iree::testing::MakeTempFilePath("weighted_embedding");
  ASSERT_NO_FATAL_FAILURE(WriteWeightedPackage(package));
  interface_.set_observer(interface_.self, {});
  loom_serve_finite_service_destroy(std::exchange(service_, nullptr));
  IREE_ASSERT_OK(
      loom_serve_finite_model_destroy(std::exchange(model_, nullptr)));
  device_ = nullptr;
  IREE_ASSERT_OK(
      loom_serve_device_group_destroy(std::exchange(group_, nullptr)));
  constexpr uint64_t kSlabBytes = 2 * 1024 * 1024;
  const loom_serve_device_options_t device_options = {
      .uri = IREE_SV("amdgpu"),
      .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
      .slab_size = kSlabBytes,
      .memory_limit = kSlabBytes};
  IREE_ASSERT_OK(loom_serve_device_group_create(host_, 1, &device_options,
                                                &group_, allocator_));
  device_ = loom_serve_device_group_at(group_, 0);
  ASSERT_NO_FATAL_FAILURE(CreateModel(package));
  retained_ = std::exchange(model_, nullptr);
  ASSERT_NO_FATAL_FAILURE(CreateModel(package));
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  auto* retained_residency = loom_serve_finite_model_residency(retained_);
  auto* residency = loom_serve_finite_model_residency(model_);
  IREE_ASSERT_OK(loom_serve_residency_acquire(retained_residency));
  IREE_ASSERT_OK(loom_serve_residency_await(retained_residency));
  EXPECT_EQ(loom_serve_residency_statistics(retained_residency).committed_bytes,
            kSlabBytes);
  const int cancelled = Send(R"({"model":"fixture","input":["a","bbb"]})");
  auto* connection = Accept();
  const auto queued = interface_.snapshot(interface_.self);
  EXPECT_EQ(queued.queued_requests, 1u);
  EXPECT_EQ(queued.active_requests, 0u);
  EXPECT_GE(queued.oldest_queued_age, 0);
  EXPECT_EQ(queued.oldest_active_age, 0);
  // The outer owner holds another model's pin and will release it. Queue the
  // cohort behind that activity instead of treating capacity as unreachable.
  constexpr auto kShared = LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY;
  EXPECT_TRUE(Step(kShared));  // Prepare requests.
  EXPECT_TRUE(Step(kShared));  // Prepare cohort.
  EXPECT_TRUE(Step(kShared));  // Consume preparation; admission denied.
  const auto waiting = interface_.snapshot(interface_.self);
  EXPECT_STREQ(waiting.phase, "waiting");
  EXPECT_EQ(waiting.queued_requests, 0u);
  EXPECT_EQ(waiting.oldest_queued_age, 0);
  EXPECT_EQ(waiting.active_requests, 1u);
  EXPECT_GT(waiting.oldest_active_age, 0);
  EXPECT_GE(waiting.oldest_active_age, queued.oldest_queued_age);
  const uint64_t wakes = wakes_.load();
  for (int i = 0; i < 3; ++i) {
    EXPECT_FALSE(Step(kShared));
    EXPECT_FALSE(interface_.is_active(interface_.self));
    EXPECT_GE(interface_.snapshot(interface_.self).oldest_active_age,
              waiting.oldest_active_age);
  }
  // No new CPU process or device submission occurred during those turns. The
  // source packet and result association were retained, not prepared again.
  EXPECT_EQ(wakes_.load(), wakes);
  EXPECT_EQ(loom_serve_residency_statistics(residency).committed_bytes, 0u);
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(
      loom_serve_residency_try_admit(retained_residency, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(cancelled, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(cancelled), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connection, iree_infinite_timeout());
  EXPECT_TRUE(Step(kShared));
  // Reset withdraws the older pressure interest. The still-pinned peer may
  // accept again without waiting for its existing invocation to finish.
  IREE_ASSERT_OK(
      loom_serve_residency_try_admit(retained_residency, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_residency_release(retained_residency);
  const auto cancelled_state = interface_.snapshot(interface_.self);
  EXPECT_EQ(cancelled_state.oldest_queued_age, 0);
  EXPECT_EQ(cancelled_state.oldest_active_age, 0);
  const int request = Send(R"({"model":"fixture","input":["a","bbb"]})");
  Accept();
  EXPECT_TRUE(Step(kShared));
  EXPECT_TRUE(Step(kShared));
  EXPECT_TRUE(Step(kShared));
  loom_serve_residency_release(retained_residency);
  Drain();
  const std::string result = Read(request);
  const auto idle = interface_.snapshot(interface_.self);
  EXPECT_EQ(idle.queued_requests, 0u);
  EXPECT_EQ(idle.active_requests, 0u);
  EXPECT_EQ(idle.oldest_queued_age, 0);
  EXPECT_EQ(idle.oldest_active_age, 0);
  EXPECT_EQ(result.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(result.find(R"("index":0,"embedding":[2,2,2,4])"),
            std::string::npos);
  EXPECT_NE(result.find(R"("index":1,"embedding":[12,6,4,4])"),
            std::string::npos);
  EXPECT_EQ(loom_serve_residency_statistics(retained_residency).committed_bytes,
            0u);
  EXPECT_EQ(loom_serve_residency_statistics(residency).committed_bytes,
            kSlabBytes);
}

class GroupEmbeddingServiceTest : public SourceEmbeddingServiceTest {
 protected:
  iree_host_size_t DeviceCount() const override { return 2; }

  void TearDown() override {
    // Release a held dependency even when an assertion interrupts the witness.
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 2, nullptr));
    }
    iree_hal_buffer_release(workspace_);
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    loom_serve_execution_release(execution_);
    iree_hal_semaphore_release(gate_);
    SourceEmbeddingServiceTest::TearDown();
  }

  // Work on the group's other device, independent of the embedding timeline.
  loom_serve_execution_t* execution_ = nullptr;
  // External predecessor held until the other device returns its HTTP result.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Queue-allocated root retained through submission of its final reader.
  iree_hal_buffer_t* workspace_ = nullptr;
  // Feedback storage outliving accepted work, including assertion cleanup.
  std::array<uint32_t, 16> output_ = {};
};

TEST_P(GroupEmbeddingServiceTest, EmbeddingsCompleteBeforeOtherDeviceRetires) {
  auto* first = loom_serve_device_group_at(group_, 0);
  EXPECT_EQ(loom_serve_device_group_count(group_), 2u);
  EXPECT_EQ(loom_serve_device_group(first), loom_serve_device_group(device_));
  EXPECT_EQ(iree_hal_device_group_device_count(loom_serve_device_group(first)),
            2u);
  EXPECT_NE(loom_serve_device_residency_cache(first),
            loom_serve_device_residency_cache(device_));
  IREE_ASSERT_OK(loom_serve_device_create_execution(first, &execution_));
  IREE_ASSERT_OK(loom_serve_execution_reserve_workspace(execution_, 64, 16));
  const iree_hal_pool_reservation_request_t request = {
      .params = {.usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                          IREE_HAL_BUFFER_USAGE_TRANSFER,
                 .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                 .min_alignment = 16},
      .allocation_size = 64};
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(execution_, &request, &workspace_));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(first), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
  uint64_t value = 1;
  IREE_ASSERT_OK(loom_serve_execution_barrier(execution_, {1, &gate_, &value}));
  const uint32_t pattern = 0x13579BDF;
  const iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill = {.target_buffer = workspace_,
               .length = sizeof(output_),
               .pattern = &pattern,
               .pattern_length = sizeof(pattern)}};
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution_, 1, &fill, &value));
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {.source_buffer = workspace_,
                   .target = output_.data(),
                   .length = sizeof(output_)}};
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &value));
  auto frontier = loom_serve_execution_frontier(execution_);
  // The feedback branch is a final reader, not an implicit FIFO dependency.
  IREE_ASSERT_OK(loom_serve_execution_barrier(
      execution_, {1, &frontier.semaphores[1], &frontier.values[1]}));
  IREE_ASSERT_OK(loom_serve_execution_dealloca(execution_, workspace_, &value));
  iree_hal_buffer_release(std::exchange(workspace_, nullptr));

  StartRouter();
  const auto response =
      Read(Send(R"({"model":"fixture","input":["a","bbb"]})"));
  EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(response.find(R"("index":0,"embedding":[1,1,1,2])"),
            std::string::npos);
  EXPECT_NE(response.find(R"("index":1,"embedding":[6,3,2,2])"),
            std::string::npos);
  uint64_t observed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(frontier.semaphores[1], &observed));
  EXPECT_LT(observed, frontier.values[1]);
  IREE_ASSERT_OK(loom_serve_device_trim(first, 0));
  EXPECT_EQ(
      loom_serve_device_memory_statistics(first).workspace.reservation_count,
      1u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  for (uint32_t word : output_) {
    EXPECT_EQ(word, pattern);
  }
  IREE_ASSERT_OK(loom_serve_device_trim(first, 0));
  EXPECT_EQ(
      loom_serve_device_memory_statistics(first).workspace.bytes_committed, 0u);
}

INSTANTIATE_TEST_SUITE_P(Workers, GroupEmbeddingServiceTest,
                         ::testing::Values(1u, 4u));

INSTANTIATE_TEST_SUITE_P(Workers, SourceEmbeddingServiceTest,
                         ::testing::Values(1u, 4u));

}  // namespace
