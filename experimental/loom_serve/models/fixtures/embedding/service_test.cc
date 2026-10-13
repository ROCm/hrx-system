// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/finite/service.h"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "experimental/loom_serve/embedding/protocol.h"
#include "experimental/loom_serve/models/fixtures/stalled_stderr.h"
#include "experimental/loom_serve/serving/model_registration.h"
#include "experimental/loom_serve/serving/router.h"
#include "experimental/loom_serve/testing/model.h"
#include "iree/async/proactor.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/transport/http/listener.h"

IREE_FLAG(string, model_source, "",
          "Deterministic embedding bootstrap source.");
IREE_FLAG(string, test_device, "amdgpu",
          "HAL device executing the source-defined embedding fixture.");
IREE_FLAG(string, direct_stage_source, "",
          "Resident source replacing the command fixture at construction.");

namespace {

using loom_serve::testing::StalledStderr;

// A serialized serving-owner process, sharing the exact host with compilation
// and an optional already-admitted service cohort. Only the outer test waits.
struct ModelLifecycle {
  static void Wake(void* user_data) {
    auto* self = static_cast<ModelLifecycle*>(user_data);
    iree_task_executor_schedule_process(self->executor, &self->process);
  }

  static iree_status_t Drain(iree_task_process_t* process,
                             const iree_task_worker_context_t*,
                             iree_task_process_drain_result_t* result) {
    auto* self = static_cast<ModelLifecycle*>(process->user_data);
    if (!self->complete && self->retiring) {
      if (!self->retirement_started) {
        self->retirement_started = true;
        loom_serve_finite_model_retire(self->model, Wake, self);
      }
      self->complete = loom_serve_finite_model_is_retired(self->model);
      if (self->complete) {
        self->status = loom_serve_finite_model_destroy_retired(self->model);
        if (iree_status_is_ok(self->status)) {
          self->model = nullptr;
        }
      }
    } else if (!self->complete) {
      self->status = iree_status_join(
          self->status,
          loom_serve_finite_load_advance(self->load, 0, &self->complete));
      if (self->cancel_after_start && !self->complete) {
        // The first CPU task has been accepted, even with only one worker.
        self->status = iree_status_join(
            self->status, loom_serve_finite_load_advance(
                              self->load, LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL,
                              &self->complete));
      }
      if (self->complete) {
        self->report = *loom_serve_finite_load_report(self->load);
        if (iree_status_is_ok(self->status)) {
          self->model = loom_serve_finite_load_take_model(self->load);
        }
        self->status = iree_status_join(
            self->status,
            loom_serve_finite_load_destroy(std::exchange(self->load, nullptr)));
      }
    }
    bool pending = false;
    if (self->service) {
      const auto flags =
          self->complete ? 0 : LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY;
      self->status = iree_status_join(
          self->status, self->service->advance(self->service->self, flags,
                                               &result->did_work));
      pending = self->service->is_pending(self->service->self);
    }
    result->completed = self->complete && !pending;
    return iree_ok_status();
  }

  static void Release(iree_task_process_t* process) {
    auto* self = static_cast<ModelLifecycle*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->released = true;
    self->condition.notify_one();
  }

  iree_status_t Run(loom_serve_device_t* device,
                    const loom_serve_finite_model_options_t& options) {
    IREE_RETURN_IF_ERROR(loom_serve_finite_load_create(
        device, &options, Wake, this, &load, iree_allocator_system()));
    return RunOwner(device);
  }

  iree_status_t Retire(loom_serve_device_t* device) {
    retiring = true;
    return RunOwner(device);
  }

  iree_status_t RunOwner(loom_serve_device_t* device) {
    executor = loom_serve_host_executor(loom_serve_device_host(device));
    iree_task_process_initialize(Drain, 0, 1, &process);
    process.user_data = this;
    process.release_fn = Release;
    if (service) {
      service->set_observer(service->self, {Wake, this});
    }
    Wake(this);
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return released; });
    if (service) {
      service->set_observer(service->self, {});
    }
    return status;
  }

  // Borrowed shared executor for both owner turns and private preparation.
  iree_task_executor_t* executor = nullptr;
  // Optional live service advanced by this same serialized pool owner.
  loom_serve_http_service_t* service = nullptr;
  // Owned in-progress load, consumed at its terminal owner turn.
  loom_serve_finite_load_t* load = nullptr;
  // Prepared model returned by loading, or owned model consumed by retirement.
  loom_serve_finite_model_t* model = nullptr;
  // Copied terminal evidence, surviving failed or cancelled preparation.
  loom_serve_compilation_report_t report = {};
  // Exclusive single-drainer process; asynchronous callbacks only wake it.
  iree_task_process_t process = {};
  // Publishes final release to the outer test, independent of owner turns.
  std::mutex mutex;
  // Outer-only notification of actual process retirement.
  std::condition_variable condition;
  // Exercises cancellation after actual task acceptance, not a fake callback.
  bool cancel_after_start = false;
  // This owner removes a model instead of preparing one.
  bool retiring = false;
  // Cached consumers were released once and ownership observation is armed.
  bool retirement_started = false;
  // Terminal preparation or retirement outcome has been consumed.
  bool complete = false;
  // Executor has made its final access to this process.
  bool released = false;
  // Terminal outcome, owned by the outer caller after Run.
  iree_status_t status = iree_ok_status();
};

class SourceEmbeddingServiceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  virtual iree_host_size_t DeviceCount() const { return 1; }

  void SetUp() override {
    iree_notification_initialize(&notification_);
    IREE_ASSERT_OK(loom_serve_host_create(GetParam(), &host_, allocator_));
    const auto uri = iree_make_cstring_view(FLAG_test_device);
    const loom_serve_device_options_t device_options[] = {{.uri = uri},
                                                          {.uri = uri}};
    IREE_ASSERT_OK(loom_serve_device_group_create(
        host_, DeviceCount(), device_options, &group_, allocator_));
    device_ = loom_serve_device_group_at(group_, DeviceCount() - 1);
    const std::string source = FLAG_model_source;
    directory_ = source.substr(0, source.find_last_of('/'));
    if (FLAG_direct_stage_source[0]) {
      ASSERT_NO_FATAL_FAILURE(UseDirectPackage());
    } else {
      ASSERT_NO_FATAL_FAILURE(CreateModel(directory_));
      ASSERT_NO_FATAL_FAILURE(CreateService(0));
    }
    auto options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_create(
        loom_serve_host_proactor_pool(host_), &options, &server_, allocator_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_create(server_, &listener_options,
                                                   &listener_, allocator_));
  }

  void CreateModel(const std::string& directory) {
    loom_serve::testing::ModelDocument document;
    IREE_ASSERT_OK(document.Load(directory + "/model.json"));
    const loom_serve_finite_model_options_t model_options = {
        .sources = document.value().sources,
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{\"batch_units\":64}")};
    ModelLifecycle load;
    IREE_ASSERT_OK(load.Run(device_, model_options));
    model_ = load.model;
    EXPECT_EQ(loom_serve_finite_model_batch_limit(model_), 64u);
  }

  void CreateService(iree_duration_t heartbeat_interval) {
    IREE_ASSERT_OK(loom_serve_finite_service_create(
        host_, model_, loom_serve_embedding_protocol(), IREE_SV("fixture"), 16,
        heartbeat_interval, nullptr, &service_, allocator_));
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
      loom_serve_http_server_request_stop(server_);
      router_.join();
      IREE_EXPECT_OK(status_);
    }
    if (service_) {
      interface_.set_observer(interface_.self, {});
      loom_serve_finite_service_destroy(service_);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_destroy(listener_));
    loom_serve_http_server_destroy(server_);
    IREE_EXPECT_OK(loom_serve_finite_model_destroy(model_));
    IREE_EXPECT_OK(loom_serve_finite_model_destroy(retained_));
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
    loom_serve_host_release(host_);
    iree_notification_deinitialize(&notification_);
  }

  int Send(const std::string& body, const char* target = "/v1/embeddings",
           const char* method = "POST", const std::string& headers = {}) {
    const int client = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    EXPECT_GE(client, 0);
    const auto* address = loom_serve_http_listener_address(listener_);
    EXPECT_EQ(
        connect(client, reinterpret_cast<const sockaddr*>(address->storage),
                address->length),
        0);
    const std::string request = std::string(method) + " " + target +
                                " HTTP/1.1\r\nHost: local\r\nContent-Type: "
                                "application/json\r\nContent-Length: " +
                                std::to_string(body.size()) + "\r\n" + headers +
                                "\r\n" + body;
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

  void CheckAbandonedCohort(loom_serve_http_service_flags_t flags) {
    const char* bodies[] = {R"({"model":"fixture","input":"a"})",
                            R"({"model":"fixture","input":"bbb"})"};
    int clients[IREE_ARRAYSIZE(bodies)] = {};
    loom_serve_http_connection_t* connections[IREE_ARRAYSIZE(bodies)] = {};
    for (size_t i = 0; i < IREE_ARRAYSIZE(bodies); ++i) {
      clients[i] = Send(bodies[i]);
      connections[i] = Accept();
    }
    ASSERT_TRUE(Step());  // CPU request preparation.
    ASSERT_TRUE(Step());  // Source batch preparation.
    ASSERT_TRUE(Step());  // Accepted device work and its actual readiness.
    ASSERT_STREQ(interface_.snapshot(interface_.self).phase, "running");

    if (!iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
      for (size_t i = 0; i < IREE_ARRAYSIZE(clients); ++i) {
        const linger reset = {1, 0};
        ASSERT_EQ(setsockopt(clients[i], SOL_SOCKET, SO_LINGER, &reset,
                             sizeof(reset)),
                  0);
        ASSERT_EQ(close(clients[i]), 0);
        iree_notification_await(
            loom_serve_http_server_notification(server_),
            [](void* value) {
              return loom_serve_http_connection_failed(
                  static_cast<loom_serve_http_connection_t*>(value));
            },
            connections[i], iree_infinite_timeout());
      }
    }

    // Successful completion still counts the whole device cohort. No response
    // consumer survives, so no CPU encoder should acquire these result views.
    EXPECT_TRUE(Step(flags));
    EXPECT_STREQ(interface_.snapshot(interface_.self).phase, "idle");
    EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 0u);
    EXPECT_FALSE(interface_.is_pending(interface_.self));
    EXPECT_EQ(interface_.telemetry.counters->cancelled, 2u);
    EXPECT_EQ(interface_.telemetry.counters->input_tokens, 4u);
    EXPECT_EQ(interface_.telemetry.counters->batches, 1u);
    Drain();
    if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
      for (int client : clients) {
        EXPECT_TRUE(Read(client).empty());
      }
      interface_.set_observer(interface_.self, {});
      loom_serve_finite_service_destroy(std::exchange(service_, nullptr));
      ASSERT_NO_FATAL_FAILURE(CreateService(0));
    }

    const int next = Send(R"({"model":"fixture","input":[0]})");
    Accept();
    Drain();
    EXPECT_NE(Read(next).find(R"("embedding":[1,1,1,1])"), std::string::npos);
    EXPECT_EQ(interface_.telemetry.counters->completed, 1u);
    EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
  }

  void StartRouter() {
    interface_.set_observer(interface_.self, {});
    router_ = std::thread([this] {
      loom_serve_registry_t* registry = nullptr;
      IREE_ASSERT_OK(loom_serve_registry_create(1, &interface_, 0, {},
                                                &registry, allocator_));
      status_ = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {.request_quota = 16, .holding_capacity = 8}, allocator_);
      IREE_ASSERT_OK(loom_serve_registry_destroy(registry));
    });
  }

  void StartManagedRouter(iree_host_size_t capacity = 2) {
    if (service_) {
      interface_.set_observer(interface_.self, {});
    }
    router_ = std::thread([this, capacity] {
      const loom_serve_model_registration_options_t options = {
          .device_group = group_, .heartbeat_interval = 1000000};
      loom_serve_registry_t* registry = nullptr;
      IREE_ASSERT_OK(loom_serve_registry_create(
          service_ ? 1 : 0, &interface_, capacity,
          loom_serve_model_registration_factory(&options), &registry,
          allocator_));
      status_ = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {.request_quota = 16, .holding_capacity = 8}, allocator_);
      EXPECT_TRUE(loom_serve_registry_counters(registry, &retired_counters_));
      status_ =
          iree_status_join(status_, loom_serve_registry_destroy(registry));
    });
  }

  std::string Declaration(const std::string& name, const std::string& source,
                          const char* admission = "open",
                          const std::string& base = "") {
    return "{\"name\":\"" + name + "\",\"base_directory\":\"" +
           std::filesystem::absolute(base.empty() ? source : base).string() +
           "\",\"admission\":\"" + admission +
           "\",\"definition\":" + loom_serve::testing::ReadDefinition(source) +
           "}";
  }

  std::string ModelStatus(const std::string& name) {
    const std::string listing = Read(Send({}, "/v1/status", "GET"));
    const size_t begin = listing.find("\"id\":\"" + name + "\"");
    if (begin == std::string::npos) {
      return {};
    }
    // Exact serialized object boundaries, not a readiness read of native state.
    const size_t end = listing.find("{\"object\":\"model\"", begin);
    return listing.substr(begin, end == std::string::npos ? end : end - begin);
  }

  std::string WaitModel(const std::string& name, const std::string& marker) {
    for (;;) {
      std::string state = ModelStatus(name);
      if (marker.empty() ? state.empty()
                         : state.find(marker) != std::string::npos) {
        return state;
      }
    }
  }

  uint64_t CompletedRequests() {
    const std::string metrics = Read(Send({}, "/metrics", "GET"));
    const std::string prefix = "\nloom_serve_server_completed_total ";
    const size_t position = metrics.find(prefix);
    EXPECT_NE(position, std::string::npos);
    return position == std::string::npos
               ? 0
               : std::stoull(metrics.substr(position + prefix.size()));
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
    std::ofstream manifest(target + "/model.json");
    manifest
        << R"({"name":"weighted","definition":{"kind":"embedding","prepare":"prepare.loom","control":"control.loom","sources":["config.loom","embed.loom","weights.loom"],"assets":"."}})";
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
    // F32 1.0; the source-selected preparation command doubles it on device.
    iree_unaligned_store_le_u32(factor, 0x3f800000);
    weights.write(reinterpret_cast<const char*>(factor), sizeof(factor));
    weights.close();
    ASSERT_TRUE(weights.good());
  }

  // Replace the command package with a native kernel package while retaining
  // the same source request packing, output encoding and device owner.
  void UseDirectPackage() {
    const std::string target =
        iree::testing::MakeTempFilePath("direct_embedding");
    ASSERT_TRUE(std::filesystem::create_directory(target));
    for (const char* file : {"config.loom", "control.loom", "tokenizer.json"}) {
      ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                             target + "/" + file));
    }
    const std::string stage = FLAG_direct_stage_source[0]
                                  ? FLAG_direct_stage_source
                                  : directory_ + "/variants/direct.loom";
    ASSERT_TRUE(std::filesystem::copy_file(stage, target + "/direct.loom"));
    std::ifstream input(directory_ + "/prepare.loom");
    ASSERT_TRUE(input.good());
    std::string prepare((std::istreambuf_iterator<char>(input)), {});
    const std::string root = "bytes(\"656d626564\")";
    const auto root_position = prepare.find(root);
    ASSERT_NE(root_position, std::string::npos);
    prepare.replace(root_position, root.size(),
                    "bytes(\"7265736964656e745f656d626564\")");
    size_t position = 0;
    while ((position = prepare.find("@stage(", position)) !=
           std::string::npos) {
      prepare.replace(position, 7, "@dispatch(");
      position += 10;
    }
    std::ofstream bootstrap(target + "/prepare.loom");
    bootstrap << prepare;
    bootstrap.close();
    ASSERT_TRUE(bootstrap.good());
    std::ofstream manifest(target + "/model.json");
    manifest
        << R"({"name":"direct","definition":{"kind":"embedding","prepare":"prepare.loom","control":"control.loom","sources":["config.loom","direct.loom"],"assets":"."}})";
    manifest.close();
    ASSERT_TRUE(manifest.good());
    if (service_) {
      interface_.set_observer(interface_.self, {});
      loom_serve_finite_service_destroy(std::exchange(service_, nullptr));
    }
    IREE_ASSERT_OK(
        loom_serve_finite_model_destroy(std::exchange(model_, nullptr)));
    ASSERT_NO_FATAL_FAILURE(CreateModel(target));
    ASSERT_NO_FATAL_FAILURE(CreateService(0));
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
  // Independent binding borrowing the HTTP owner and shared host pool.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Outer driver notification for actual task/queue readiness.
  iree_notification_t notification_;
  // Published callback generations.
  std::atomic<uint64_t> wakes_{0};
  // Optional outer shared-router thread.
  std::thread router_;
  // Router status read after join.
  iree_status_t status_ = iree_ok_status();
  // Copied after the router joins, including counters from removed
  // incarnations.
  loom_serve_telemetry_counters_t retired_counters_ = {};
};

TEST_P(SourceEmbeddingServiceTest,
       DynamicWeightedRegistrationOwnsItsOperationAndIncarnation) {
  const std::string package =
      iree::testing::MakeTempFilePath("dynamic_weighted");
  ASSERT_NO_FATAL_FAILURE(WriteWeightedPackage(package));
  StartManagedRouter();
  const std::string declaration = Declaration("weighted", package, "drained");
  const auto registered = Read(Send(declaration, "/v1/models/register"));
  EXPECT_EQ(registered.find("HTTP/1.1 202"), 0u);
  EXPECT_NE(registered.find(R"("registration":"registered")"),
            std::string::npos);
  EXPECT_NE(registered.find(R"("admission":"drained")"), std::string::npos);
  EXPECT_EQ(Read(Send(declaration, "/v1/models/register")).find("HTTP/1.1 409"),
            0u);
  EXPECT_EQ(
      Read(Send(Declaration("cold", package, "drained", package + "/absent"),
                "/v1/models/register"))
          .find("HTTP/1.1 202"),
      0u);
  EXPECT_EQ(Read(Send(Declaration("full", package), "/v1/models/register"))
                .find("HTTP/1.1 429"),
            0u);
  EXPECT_NE(ModelStatus("cold").find(R"("registration":"registered")"),
            std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"cold"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("cold", {});

  const int held = Send(R"({"model":"weighted","input":"a"})");
  WaitModel("weighted", R"("held_requests":1)");
  const int tokenizer =
      Send(R"({"model":"weighted","content":"ab"})", "/tokenize");
  WaitModel("weighted", R"("held_requests":2)");
  // The accepted operation belongs to the registry, not this unread response.
  const int admin = Send(R"({"model":"weighted"})", "/v1/models/undrain");
  WaitModel("weighted", R"("admission":"open")");
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(admin, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)), 0);
  ASSERT_EQ(close(admin), 0);
  EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  const std::string embedding_response = Read(held);
  const std::string tokenizer_response = Read(tokenizer);
  ASSERT_NE(embedding_response.find(R"("embedding":[2,2,2,2])"),
            std::string::npos)
      << embedding_response << ModelStatus("weighted");
  ASSERT_NE(tokenizer_response.find(R"({"tokens":[0,1]})"), std::string::npos)
      << tokenizer_response;
  const auto ready = WaitModel("weighted", R"("registration":"ready")");
  EXPECT_NE(ready.find("\"workers\":" + std::to_string(GetParam())),
            std::string::npos);
  const uint64_t completed = CompletedRequests();
  EXPECT_EQ(completed, 2u);
  EXPECT_NE(Read(Send({}, "/metrics", "GET"))
                .find("loom_serve_model_completed_total{model=\"weighted\","
                      "kind=\"embedding\",incarnation=\"1\"} 1\n"),
            std::string::npos);
  const std::string logs =
      Read(Send({}, "/logz", "GET", "X-Loom-Model: weighted\r\n"));
  const std::string prefix = R"({"event":"cursor","value":")";
  const size_t cursor_begin = logs.find(prefix);
  ASSERT_NE(cursor_begin, std::string::npos);
  const size_t value_begin = cursor_begin + prefix.size();
  const std::string cursor =
      logs.substr(value_begin, logs.find('"', value_begin) - value_begin);
  EXPECT_EQ(Read(Send(R"({"model":"weighted","mode":"drain"})",
                      "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("weighted", {});
  EXPECT_EQ(CompletedRequests(), completed);
  EXPECT_EQ(Read(Send(Declaration("weighted", package), "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_NE(Read(Send(R"({"model":"weighted","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  EXPECT_EQ(
      Read(Send({}, "/logz", "GET",
                "X-Loom-Model: weighted\r\nX-Loom-Cursor: " + cursor + "\r\n"))
          .find("HTTP/1.1 409"),
      0u);
  EXPECT_EQ(CompletedRequests(), completed + 1);
  const std::string metrics = Read(Send({}, "/metrics", "GET"));
  EXPECT_EQ(metrics.find("incarnation=\"1\""), std::string::npos);
  EXPECT_NE(metrics.find("loom_serve_model_completed_total{model=\"weighted\","
                         "kind=\"embedding\",incarnation=\"3\"} 1\n"),
            std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"weighted","mode":"abort"})",
                      "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("weighted", {});
}

TEST_P(SourceEmbeddingServiceTest,
       InvalidFeedbackAfterExecutionRetiresManagedRequests) {
  const std::string package =
      iree::testing::MakeTempFilePath("invalid_feedback");
  ASSERT_TRUE(std::filesystem::create_directory(package));
  for (const char* file : {"config.loom", "control.loom", "embed.loom",
                           "prepare.loom", "model.json", "tokenizer.json"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           package + "/" + file));
  }
  {
    const std::string path = package + "/control.loom";
    std::ifstream input(path);
    ASSERT_TRUE(input.good());
    std::string source((std::istreambuf_iterator<char>(input)), {});
    const std::string original =
        "  %done = func.call @feedback(%slot, %output, %offset, "
        "%feedback_bytes)";
    const size_t position = source.find(original);
    ASSERT_NE(position, std::string::npos);
    // The command has already been accepted when the source exceeds its
    // declared 1024-byte host destination. No invalid transfer is submitted.
    source.replace(position, original.size(),
                   "  %oversized = scalar.constant 4096 : i64\n"
                   "  %done = func.call @feedback(%slot, %output, %offset, "
                   "%oversized)");
    input.close();
    std::filesystem::permissions(path, std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
    std::ofstream output(path, std::ios::trunc);
    output << source;
    ASSERT_TRUE(output.good());
  }
  StartManagedRouter();
  EXPECT_EQ(Read(Send(Declaration("broken", package, "drained"),
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Read(Send(Declaration("cold", directory_, "drained"),
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  const int cold = Send(R"({"model":"cold","input":"a"})");
  WaitModel("cold", R"("held_requests":1)");
  const int broken = Send(R"({"model":"broken","input":"a"})");
  WaitModel("broken", R"("held_requests":1)");
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);

  // The runner rejects feedback after accepting real command work. Its
  // terminal error drains even an unrelated held registration.
  // Join the owner directly: polling HTTP after terminal shutdown would wait
  // on a consumer that has already retired.
  router_.join();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        std::exchange(status_, iree_ok_status()));
  EXPECT_TRUE(Read(broken).empty());
  EXPECT_TRUE(Read(cold).empty());
  // Registry counters cover work admitted to a model. The cold request never
  // left the router's holding set and is observed through its aborted claim.
  EXPECT_EQ(retired_counters_.requests, 1u);
  EXPECT_EQ(retired_counters_.failed, 1u);
  EXPECT_EQ(retired_counters_.cancelled, 0u);
  // Model consumers have retired, but raw queue allocation records may still
  // own workspace backing. Trim does not wait for those records; the outer
  // TearDown joins them when destroying the shared device group.
  IREE_ASSERT_OK(loom_serve_device_trim(device_, 0));
}

TEST_P(SourceEmbeddingServiceTest,
       FailedRegistrationPublishesEvidenceAndReleasesHeldClaims) {
  const std::string package = iree::testing::MakeTempFilePath("failed_dynamic");
  ASSERT_NO_FATAL_FAILURE(WriteWeightedPackage(package));
  {
    std::filesystem::permissions(package + "/embed.loom",
                                 std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
    std::ofstream source(package + "/embed.loom", std::ios::trunc);
    source << "this is not a source program\n";
    ASSERT_TRUE(source.good());
  }
  StartManagedRouter();
  StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(Read(Send(Declaration("broken", package, "drained"),
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  const int held = Send(R"({"model":"broken","input":"a"})");
  WaitModel("broken", R"("held_requests":1)");
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(held).find("HTTP/1.1 503"), 0u);
  const auto failed = WaitModel("broken", R"("registration":"failed")");
  EXPECT_NE(failed.find("embed.loom"), std::string::npos);
  EXPECT_NE(failed.find(R"("diagnostics":[{)"), std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "/v1/models/undrain"))
                .find("HTTP/1.1 409"),
            0u);
  EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("broken", {});
}

// Virtual backing is a separate device capability from source execution.
class ElasticEmbeddingServiceTest : public SourceEmbeddingServiceTest {};

TEST_P(ElasticEmbeddingServiceTest,
       EmptyServerReusesOneParameterSlabAcrossManagedLifetimes) {
  const std::string package =
      iree::testing::MakeTempFilePath("dynamic_pressure");
  ASSERT_NO_FATAL_FAILURE(WriteWeightedPackage(package));
  interface_.set_observer(interface_.self, {});
  loom_serve_finite_service_destroy(std::exchange(service_, nullptr));
  IREE_ASSERT_OK(
      loom_serve_finite_model_destroy(std::exchange(model_, nullptr)));
  IREE_ASSERT_OK(
      loom_serve_device_group_destroy(std::exchange(group_, nullptr)));
  const loom_serve_device_options_t options = {
      .uri = IREE_SV("amdgpu"),
      .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
      .slab_size = 2 * 1024 * 1024,
      .memory_limit = 2 * 1024 * 1024};
  IREE_ASSERT_OK(
      loom_serve_device_group_create(host_, 1, &options, &group_, allocator_));
  device_ = loom_serve_device_group_at(group_, 0);
  StartManagedRouter();
  EXPECT_NE(Read(Send({}, "/v1/models", "GET")).find(R"("data":[])"),
            std::string::npos);
  for (const char* name : {"first", "second"}) {
    EXPECT_EQ(Read(Send(Declaration(name, package), "/v1/models/register"))
                  .find("HTTP/1.1 202"),
              0u);
    EXPECT_NE(
        Read(Send(std::string("{\"model\":\"") + name + "\",\"input\":\"a\"}"))
            .find(R"("embedding":[2,2,2,2])"),
        std::string::npos);
  }
  EXPECT_EQ(Read(Send(R"({"model":"first"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("first", {});
  EXPECT_NE(Read(Send(R"({"model":"second","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"second"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("second", {});
  EXPECT_EQ(CompletedRequests(), 3u);
  EXPECT_NE(Read(Send({}, "/v1/models", "GET")).find(R"("data":[])"),
            std::string::npos);
  loom_serve_http_server_request_stop(server_);
  router_.join();
  IREE_ASSERT_OK(status_);
  const auto memory = loom_serve_device_memory_statistics(device_).retained;
  EXPECT_EQ(memory.reserved_bytes, 0u);
  EXPECT_EQ(memory.committed_bytes, 0u);
  EXPECT_EQ(retired_counters_.completed, 3u);
}

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
  for (iree_host_size_t i = 0; i < 4; ++i) {
    loom_serve_http_request_snapshot_t snapshot = {};
    ASSERT_TRUE(
        interface_.inspection.request_at(interface_.self, i, &snapshot));
    EXPECT_EQ(snapshot.id, i + 1);
    EXPECT_STREQ(snapshot.phase, "queued");
    EXPECT_EQ(snapshot.counters.requests, 1u);
    EXPECT_EQ(snapshot.counters.input_tokens, 0u);
  }
  EXPECT_TRUE(Step());  // All independent requests prepared.
  // The executor may still own prepared packets; only immutable receipt and
  // owner-published progress are inspected before that release is consumed.
  loom_serve_http_request_snapshot_t preparing = {};
  ASSERT_TRUE(interface_.inspection.request_at(interface_.self, 0, &preparing));
  EXPECT_EQ(preparing.id, 1u);
  EXPECT_STREQ(preparing.phase, "preparing");
  EXPECT_TRUE(Step());  // Rejection collected; valid cohort constructed.
  EXPECT_TRUE(Step());  // One command submitted, with compact feedback.
  for (iree_host_size_t i = 0; i < 3; ++i) {
    loom_serve_http_request_snapshot_t snapshot = {};
    ASSERT_TRUE(
        interface_.inspection.request_at(interface_.self, i, &snapshot));
    EXPECT_EQ(snapshot.id, i + 1);
    EXPECT_STREQ(snapshot.phase, "running");
    EXPECT_EQ(snapshot.counters.input_tokens, 0u);
  }
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
  EXPECT_TRUE(Step());  // Capture only live result views for CPU encoding.
  EXPECT_STREQ(interface_.snapshot(interface_.self).phase, "encoding");
  EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 2u);
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 1u);
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
  const auto& counters = *interface_.telemetry.counters;
  EXPECT_EQ(counters.requests, 5u);
  EXPECT_EQ(counters.completed, 3u);
  EXPECT_EQ(counters.rejected, 1u);
  EXPECT_EQ(counters.cancelled, 1u);
  EXPECT_EQ(counters.failed, 0u);
  EXPECT_EQ(counters.input_tokens, 8u);
  EXPECT_EQ(counters.batches, 2u);
}

TEST_P(SourceEmbeddingServiceTest, CancelledDeviceCohortSkipsEncoding) {
  CheckAbandonedCohort(0);
}

TEST_P(SourceEmbeddingServiceTest, DirectDispatchReturnsCompactRaggedVectors) {
  ASSERT_NO_FATAL_FAILURE(UseDirectPackage());
  const int a = Send(R"({"model":"fixture","input":["a","bbb"]})");
  Accept();
  const int b = Send(R"({"model":"fixture","input":[[2,0]],"dimensions":2})");
  Accept();
  Drain();
  const std::string first = Read(a);
  SCOPED_TRACE(first);
  EXPECT_EQ(first.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(first.find(R"("index":0,"embedding":[1,1,1,3])"),
            std::string::npos);
  EXPECT_NE(first.find(R"("index":1,"embedding":[6,3,2,3])"),
            std::string::npos);
  const std::string second = Read(b);
  EXPECT_NE(second.find(R"("embedding":[4,2])"), std::string::npos) << second;
  const int next = Send(R"({"model":"fixture","input":[1]})");
  Accept();
  Drain();
  const std::string last = Read(next);
  EXPECT_NE(last.find(R"("embedding":[2,1,2,1])"), std::string::npos) << last;
}

TEST_P(SourceEmbeddingServiceTest, DirectDispatchRetiresCancelledCohort) {
  ASSERT_NO_FATAL_FAILURE(UseDirectPackage());
  CheckAbandonedCohort(0);
}

TEST_P(SourceEmbeddingServiceTest, StoppingDeviceCohortSkipsEncoding) {
  CheckAbandonedCohort(LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING);
}

TEST_P(SourceEmbeddingServiceTest,
       LogReadersWrapIndependentlyWhileStderrIsFull) {
  StartRouter();
  auto poll = [&](const std::string& cursor) {
    return Read(
        Send("", "/logz", "GET",
             "X-Loom-Model: fixture\r\n" +
                 (cursor.empty() ? std::string()
                                 : "X-Loom-Cursor: " + cursor + "\r\n")));
  };
  auto cursor_of = [](const std::string& response) {
    const std::string marker = R"({"event":"cursor","value":")";
    const size_t start = response.find(marker);
    EXPECT_NE(start, std::string::npos);
    if (start == std::string::npos) {
      return std::string();
    }
    const size_t first = start + marker.size();
    return response.substr(first, response.find('"', first) - first);
  };
  const std::string initial = poll("");
  EXPECT_EQ(initial.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(initial.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  EXPECT_NE(initial.find(R"("partial":false)"), std::string::npos);
  const std::string slow = cursor_of(initial);
  std::string fast = slow;
  StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(Read(Send(R"({"model":7})")).find("HTTP/1.1 400"), 0u);
  EXPECT_EQ(Read(Send(R"({"model":"fixture","tokens":[-1]})", "/detokenize"))
                .find("HTTP/1.1 400"),
            0u);
  const std::string diagnostics = Read(Send("", "/logz", "GET"));
  EXPECT_NE(diagnostics.find(R"("scope":"server","counters_available":true)"),
            std::string::npos);
  EXPECT_NE(diagnostics.find(R"("event":"error","sequence":1)"),
            std::string::npos);
  EXPECT_NE(diagnostics.find(R"("event":"error","sequence":2)"),
            std::string::npos);
  EXPECT_NE(
      diagnostics.find(
          R"("values":{"requests":0,"completed":0,"rejected":0,"cancelled":0,"failed":0,"input_tokens":0,"output_tokens":0,"batches":0})"),
      std::string::npos);
  EXPECT_EQ(Read(Send("", "/logz", "GET", "X-Loom-Cursor: " + slow + "\r\n"))
                .find("HTTP/1.1 409"),
            0u);
  uint64_t next = 1;
  std::string latest;
  for (uint64_t i = 0; i < 40; ++i) {
    const std::string inference =
        Read(Send(R"({"model":"fixture","input":["a","bbb"],"dimensions":2})"));
    ASSERT_EQ(inference.find("HTTP/1.1 200"), 0u);
    EXPECT_NE(inference.find(R"("embedding":[1,1])"), std::string::npos);
    EXPECT_NE(inference.find(R"("embedding":[6,3])"), std::string::npos);
    latest = poll(fast);
    EXPECT_EQ(latest.find(R"("event":"gap")"), std::string::npos);
    // Each inference has a begin, submit, device end and request end. Counters
    // and HTTP polls are not inserted into the producer's stream.
    size_t offset = 0;
    for (size_t record = 0; record < 4; ++record) {
      const auto found =
          latest.find("\"sequence\":" + std::to_string(next++) + ",", offset);
      ASSERT_NE(found, std::string::npos);
      offset = found + 12;
    }
    EXPECT_EQ(latest.find("\"sequence\":", offset), std::string::npos);
    fast = cursor_of(latest);
    if (i % 8 == 0) {
      // Observation transport cancellation must not write a blocking diagnostic
      // or poison subsequent inference. The model stream remains unchanged.
      const int peer = Send("", "/logz", "GET");
      const linger reset = {1, 0};
      ASSERT_EQ(setsockopt(peer, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
                0);
      ASSERT_EQ(close(peer), 0);
    }
  }
  EXPECT_EQ(next, 161u);
  const std::string totals =
      R"("requests":40,"completed":40,"rejected":0,"cancelled":0,"failed":0,"input_tokens":160,"output_tokens":0,"batches":40)";
  EXPECT_NE(latest.find(totals), std::string::npos);
  EXPECT_NE(Read(Send("", "/logz", "GET")).find(totals), std::string::npos);
  const std::string resumed = poll(slow);
  EXPECT_NE(resumed.find(R"({"event":"gap","from":1,"to":33})"),
            std::string::npos);
  EXPECT_NE(resumed.find(totals), std::string::npos);
  const std::string joined = poll("");
  EXPECT_NE(joined.find(R"("partial":true)"), std::string::npos);
  EXPECT_NE(joined.find(R"({"event":"gap","from":0,"to":33})"),
            std::string::npos);
  const std::string caught_up = poll(fast);
  EXPECT_EQ(caught_up.find("\"sequence\":"), std::string::npos);
  EXPECT_EQ(cursor_of(caught_up), fast);
  EXPECT_NE(caught_up.find(totals), std::string::npos);
  EXPECT_EQ(poll("bad").find("HTTP/1.1 400"), 0u);
  EXPECT_EQ(poll("0:0:1").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(poll(fast.substr(0, fast.find_last_of(':') + 1) + "999")
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("", "/logz", "GET", "X-Loom-Model: missing\r\n"))
                .find("HTTP/1.1 404"),
            0u);
  // Shutdown itself must also retire while no stderr consumer is running.
  loom_serve_http_server_request_stop(server_);
  router_.join();
  IREE_EXPECT_OK(status_);
  status_ = iree_ok_status();
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
       InputLimitsRejectOnlyTheirRequestsAndPreserveFullCapacity) {
  std::string long_ids = "[0";
  for (size_t i = 1; i < 32; ++i) {
    long_ids += ",0";
  }
  long_ids += "]";
  std::string many_rows = "[\"a\"";
  for (size_t i = 1; i < 65; ++i) {
    many_rows += ",\"a\"";
  }
  many_rows += "]";
  const std::string costly_row = "\"" + std::string(22, 'a') + "\"";
  const std::string invalid_inputs[] = {
      "\"" + std::string(32, 'b') + "\"",
      "\"" + std::string(4096, 'a') + "\"",
      long_ids,
      "[3]",
      many_rows,
      "[" + costly_row + "," + costly_row + "," + costly_row + "]",
  };
  std::vector<int> rejected;
  for (const auto& input : invalid_inputs) {
    rejected.push_back(Send("{\"model\":\"fixture\",\"input\":" + input + "}"));
    Accept();
  }
  // Every row fits individually, and these two consume the entire 64-unit
  // cohort. Rejected preparation must retain neither units nor device rows.
  const int valid =
      Send("{\"model\":\"fixture\",\"input\":[\"" + std::string(31, 'b') +
           "\",\"" + std::string(31, 'a') + "\"]}");
  Accept();
  Drain();
  for (size_t i = 0; i < rejected.size(); ++i) {
    SCOPED_TRACE(i);
    const auto response = Read(rejected[i]);
    EXPECT_EQ(response.find("HTTP/1.1 400"), 0u);
    EXPECT_NE(response.find("fixture requires"), std::string::npos);
  }
  const auto response = Read(valid);
  EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(response.find(R"("index":0,"embedding":[62,31,2,2])"),
            std::string::npos);
  EXPECT_NE(response.find(R"("index":1,"embedding":[31,31,1,2])"),
            std::string::npos);
  EXPECT_NE(response.find(R"("prompt_tokens":62,"total_tokens":62)"),
            std::string::npos);
  EXPECT_FALSE(interface_.is_pending(interface_.self));

  const int next = Send(R"({"model":"fixture","input":[0]})");
  Accept();
  Drain();
  const auto next_response = Read(next);
  EXPECT_EQ(next_response.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(next_response.find(R"("embedding":[1,1,1,1])"), std::string::npos);
  const auto& counters = *interface_.telemetry.counters;
  EXPECT_EQ(counters.requests, 8u);
  EXPECT_EQ(counters.completed, 2u);
  EXPECT_EQ(counters.rejected, 6u);
  EXPECT_EQ(counters.cancelled, 0u);
  EXPECT_EQ(counters.failed, 0u);
  EXPECT_EQ(counters.input_tokens, 63u);
  EXPECT_EQ(counters.batches, 2u);
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
       AdmissionDrainPreservesPreparedWeightsAndResumesHeldHttpRequests) {
  const std::string package =
      iree::testing::MakeTempFilePath("drained_weighted_embedding");
  ASSERT_NO_FATAL_FAILURE(WriteWeightedPackage(package));
  retained_ = std::exchange(model_, nullptr);
  ASSERT_NO_FATAL_FAILURE(CreateModel(package));
  loom_serve_finite_service_destroy(std::exchange(service_, nullptr));
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  EXPECT_EQ(
      loom_serve_residency_activity(loom_serve_finite_model_residency(model_))
          .activations,
      0u);
  StartRouter();
  const auto control = [this](const char* target) {
    return Read(Send(R"({"model":"fixture"})", target));
  };
  EXPECT_NE(control("/v1/models/drain").find(R"("admission":"drained")"),
            std::string::npos);
  std::vector<int> clients;
  for (int i = 0; i < 7; ++i) {
    clients.push_back(Send(R"({"model":"fixture","input":"a"})"));
  }
  const int tokens =
      Send(R"({"model":"fixture","content":"aba"})", "/tokenize");
  // The control reply acknowledges the actual owner cutoff; transport
  // arrivals may precede or follow each other across independent connections.
  std::string state;
  do {
    state = control("/v1/models/drain");
    ASSERT_EQ(state.find("HTTP/1.1 200"), 0u);
  } while (state.find(R"("held_requests":8,)") == std::string::npos);
  EXPECT_NE(state.find(R"("queued_requests":0,"active_requests":0)"),
            std::string::npos);
  EXPECT_NE(state.find(R"("tokenizer_requests":0)"), std::string::npos);
  EXPECT_EQ(
      Read(Send(R"({"model":"fixture","input":"a"})")).find("HTTP/1.1 429"),
      0u);
  EXPECT_NE(Read(Send("", "/metrics", "GET"))
                .find("loom_serve_server_completed_total 0\n"),
            std::string::npos);
  EXPECT_EQ(control("/v1/models/undrain").find("HTTP/1.1 200"), 0u);
  for (const int client : clients) {
    const auto response = Read(client);
    // The fourth oracle component reports the live cohort width. All seven
    // parked inference requests enter the same model batch on undrain.
    EXPECT_NE(response.find(R"("embedding":[2,2,2,14])"), std::string::npos)
        << response;
  }
  EXPECT_NE(Read(tokens).find(R"({"tokens":[0,1,0]})"), std::string::npos);

  EXPECT_NE(control("/v1/models/drain").find(R"("admission":"drained")"),
            std::string::npos);
  const int resumed = Send(R"({"model":"fixture","input":"a"})");
  do {
    state = control("/v1/models/drain");
    ASSERT_EQ(state.find("HTTP/1.1 200"), 0u);
  } while (state.find(R"("held_requests":1,)") == std::string::npos);
  EXPECT_EQ(control("/v1/models/undrain").find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(resumed).find(R"("embedding":[2,2,2,2])"), std::string::npos);
  EXPECT_NE(Read(Send("", "/metrics", "GET"))
                .find("loom_serve_server_completed_total 8\n"),
            std::string::npos);
  loom_serve_http_server_request_stop(server_);
  router_.join();
  IREE_EXPECT_OK(status_);
  // Reversible admission never evicted/reloaded the in-place prepared
  // parameter. Applying preparation twice would also change the exact output.
  EXPECT_EQ(
      loom_serve_residency_activity(loom_serve_finite_model_residency(model_))
          .activations,
      1u);
}

TEST_P(SourceEmbeddingServiceTest,
       WeightedLoadSharesTheServingWorkerAndNeverWritesStderr) {
  const std::string package =
      iree::testing::MakeTempFilePath("live_weighted_embedding");
  ASSERT_NO_FATAL_FAILURE(WriteWeightedPackage(package));
  loom_serve::testing::ModelDocument document;
  IREE_ASSERT_OK(document.Load(package + "/model.json"));
  const loom_serve_finite_model_options_t options = {
      .sources = document.value().sources,
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{}")};
  const int resident = Send(R"({"model":"fixture","input":"a"})");
  Accept();
  ModelLifecycle load;
  load.service = &interface_;
  iree_status_t status;
  {
    StalledStderr stalled;
    ASSERT_NO_FATAL_FAILURE(stalled.Begin());
    status = load.Run(device_, options);
  }
  retained_ = load.model;
  IREE_ASSERT_OK(status);
  ASSERT_NE(retained_, nullptr);
  EXPECT_NE(Read(resident).find(R"("embedding":[1,1,1,1])"), std::string::npos);
  EXPECT_EQ(load.report.commands, 2u);  // Consumer and in-place preparer.
  EXPECT_EQ(load.report.workers, GetParam());
  EXPECT_EQ(load.report.diagnostic_count, 0u);

  // Preparation has not loaded weights. First use activates and transforms
  // canonical bytes in place; exact output distinguishes the prepared factor.
  EXPECT_EQ(loom_serve_residency_activity(
                loom_serve_finite_model_residency(retained_))
                .activations,
            0u);
  loom_serve_finite_service_destroy(std::exchange(service_, nullptr));
  std::swap(model_, retained_);
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  const int prepared = Send(R"({"model":"fixture","input":"a"})");
  Accept();
  Drain();
  EXPECT_NE(Read(prepared).find(R"("embedding":[2,2,2,2])"), std::string::npos);
}

TEST_P(SourceEmbeddingServiceTest,
       CancelledAndFailedLoadsKeepTheirResidentPeerUsable) {
  const std::string package =
      iree::testing::MakeTempFilePath("invalid_embedding");
  ASSERT_NO_FATAL_FAILURE(WriteWeightedPackage(package));
  // copy_file preserves read-only runfile permissions on this owned copy.
  std::filesystem::permissions(package + "/control.loom",
                               std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::add);
  std::ofstream invalid(package + "/control.loom");
  invalid << "this is not a Loom program\n";
  invalid.close();
  ASSERT_TRUE(invalid.good());
  for (bool cancel : {true, false}) {
    const auto& source = cancel ? directory_ : package;
    loom_serve::testing::ModelDocument document;
    IREE_ASSERT_OK(document.Load(source + "/model.json"));
    const loom_serve_finite_model_options_t options = {
        .sources = document.value().sources,
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{}")};
    const int resident = Send(R"({"model":"fixture","input":"a"})");
    Accept();
    ModelLifecycle load;
    load.service = &interface_;
    load.cancel_after_start = cancel;
    iree_status_t status;
    {
      StalledStderr stalled;
      ASSERT_NO_FATAL_FAILURE(stalled.Begin());
      status = load.Run(device_, options);
    }
    IREE_EXPECT_STATUS_IS(
        cancel ? IREE_STATUS_CANCELLED : IREE_STATUS_INVALID_ARGUMENT, status);
    EXPECT_EQ(load.model, nullptr);
    EXPECT_EQ(load.report.commands, 0u);
    if (!cancel) {
      EXPECT_GT(load.report.diagnostic_count, 0u);
      EXPECT_GT(load.report.diagnostics[0].message_length, 0u);
    }
    EXPECT_NE(Read(resident).find(R"("embedding":[1,1,1,1])"),
              std::string::npos);
  }
}

TEST_P(ElasticEmbeddingServiceTest,
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

  const uint64_t reservation_bytes =
      loom_serve_residency_statistics(residency).reserved_bytes;
  const auto before_removal =
      loom_serve_device_memory_statistics(device_).retained;
  // Retire the active weighted model on the same serialized owner that must
  // admit its peer. The one-slab budget cannot hold both sets of weights.
  interface_.set_observer(interface_.self, {});
  loom_serve_finite_service_destroy(std::exchange(service_, nullptr));
  std::swap(model_, retained_);
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  const int peer = Send(R"({"model":"fixture","input":"a"})");
  Accept();
  ModelLifecycle removal;
  removal.model = std::exchange(retained_, nullptr);
  removal.service = &interface_;
  iree_status_t removal_status;
  {
    StalledStderr stalled;
    ASSERT_NO_FATAL_FAILURE(stalled.Begin());
    removal_status = removal.Retire(device_);
  }
  retained_ = removal.model;
  IREE_ASSERT_OK(removal_status);
  EXPECT_EQ(retained_, nullptr);
  EXPECT_NE(Read(peer).find(R"("embedding":[2,2,2,2])"), std::string::npos);
  const auto after_removal =
      loom_serve_device_memory_statistics(device_).retained;
  EXPECT_EQ(after_removal.reserved_bytes,
            before_removal.reserved_bytes - reservation_bytes);
  EXPECT_EQ(after_removal.committed_bytes, kSlabBytes);

  // Final removal returns the shared model pool to empty without tearing down
  // its device or host executor. Only private command cache backing may remain.
  loom_serve_finite_service_destroy(std::exchange(service_, nullptr));
  ModelLifecycle final_removal;
  final_removal.model = std::exchange(model_, nullptr);
  removal_status = final_removal.Retire(device_);
  model_ = final_removal.model;
  IREE_ASSERT_OK(removal_status);
  EXPECT_EQ(model_, nullptr);
  const auto empty = loom_serve_device_memory_statistics(device_).retained;
  EXPECT_EQ(empty.reserved_bytes, 0u);
  EXPECT_EQ(empty.committed_bytes, 0u);
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
  const iree_hal_pool_reservation_request_t request = {.allocation_size = 64,
                                                       .min_alignment = 16};
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

INSTANTIATE_TEST_SUITE_P(Workers, ElasticEmbeddingServiceTest,
                         ::testing::Values(1u, 4u));

}  // namespace
