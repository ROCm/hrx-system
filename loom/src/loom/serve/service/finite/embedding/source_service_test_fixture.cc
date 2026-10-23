// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/embedding/source_service_test_fixture.h"

#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <utility>

#include "iree/base/internal/json.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/router.h"
#include "loom/serve/service/finite/embedding/protocol.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/package.h"
#include "loom/serve/testing/service_application.h"

namespace loom_serve::testing::embedding {

using loom_serve::testing::HttpClient;

const char* SourceServiceDirectStageSource() {
  const char* source = std::getenv("LOOM_SERVE_TEST_DIRECT_STAGE_SOURCE");
  return source ? source : "";
}

void loom_serve_test_embedding_model_lifecycle_t::Wake(void* user_data) {
  auto* self =
      static_cast<loom_serve_test_embedding_model_lifecycle_t*>(user_data);
  iree_task_executor_schedule_process(self->executor, &self->process);
}

iree_status_t loom_serve_test_embedding_model_lifecycle_t::Drain(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  auto* self = static_cast<loom_serve_test_embedding_model_lifecycle_t*>(
      process->user_data);
  if (!self->complete && self->retiring) {
    if (!self->retirement_started) {
      self->retirement_started = true;
      loom_serve_finite_model_retire(self->model, Wake, self);
    }
    self->complete = loom_serve_finite_model_is_retired(self->model);
    if (self->complete) {
      self->status = loom_serve_finite_model_free_retired(self->model);
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
          self->status,
          loom_serve_finite_load_advance(
              self->load, LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL, &self->complete));
    }
    if (self->complete) {
      self->report = *loom_serve_finite_load_report(self->load);
      if (iree_status_is_ok(self->status)) {
        self->model = loom_serve_finite_load_take_model(self->load);
      }
      self->status = iree_status_join(
          self->status,
          loom_serve_finite_load_free(std::exchange(self->load, nullptr)));
    }
  }
  bool pending = false;
  if (self->service) {
    const auto flags =
        self->complete ? 0 : LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY;
    self->status = iree_status_join(
        self->status,
        self->service->advance(self->service->self, flags, &result->did_work));
    pending = self->service->is_pending(self->service->self);
  }
  if (self->pool) {
    pending |= loom_serve_memory_pool_is_pending(self->pool);
  }
  result->completed = self->complete && !pending;
  if (result->completed && self->pool) {
    loom_serve_memory_pool_set_observer(self->pool, nullptr, nullptr);
  }
  return iree_ok_status();
}

void loom_serve_test_embedding_model_lifecycle_t::Release(
    iree_task_process_t* process) {
  auto* self = static_cast<loom_serve_test_embedding_model_lifecycle_t*>(
      process->user_data);
  std::lock_guard<std::mutex> lock(self->mutex);
  self->released = true;
  self->condition.notify_one();
}

iree_status_t loom_serve_test_embedding_model_lifecycle_t::Run(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t& options,
    iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(loom_serve_finite_load_allocate(
      device, &options, Wake, this, host_allocator, &load));
  return RunOwner(device);
}

iree_status_t loom_serve_test_embedding_model_lifecycle_t::Retire(
    loom_serve_device_t* device) {
  retiring = true;
  return RunOwner(device);
}

iree_status_t loom_serve_test_embedding_model_lifecycle_t::RunOwner(
    loom_serve_device_t* device) {
  executor = loom_serve_host_executor(loom_serve_device_host(device));
  iree_task_process_initialize(Drain, 0, 1, &process);
  process.user_data = this;
  process.release_fn = Release;
  if (service) {
    service->set_observer(service->self, {Wake, this});
  }
  pool = loom_serve_device_memory_pool(device);
  if (pool) {
    loom_serve_memory_pool_set_observer(pool, Wake, this);
  }
  Wake(this);
  std::unique_lock<std::mutex> lock(mutex);
  condition.wait(lock, [&] { return released; });
  if (service) {
    service->set_observer(service->self, {});
  }
  return status;
}

iree_host_size_t SourceEmbeddingServiceTest::DeviceCount() const { return 1; }

void SourceEmbeddingServiceTest::SetUp() {
  iree_notification_initialize(&notification_);
  auto application_options = loom_serve_test_application_options_default();
  application_options.device_count = DeviceCount();
  if (application_options.backing == LOOM_SERVE_DEVICE_BACKING_ELASTIC) {
    // One parameter page plus the fixture's 3344-byte invocation fits. Two
    // parameter pages do not: a peer must reuse backing after deactivation.
    application_options.slab_size = 4096;
    application_options.memory_limit = 7680;
  }
  loom_serve_test_application_t* application = nullptr;
  IREE_ASSERT_OK(loom_serve_test_application_acquire(
      &application_options, host_allocator_, &application));
  host_ = loom_serve_test_application_host(application);
  group_ = loom_serve_test_application_device_group(application);
  device_ =
      loom_serve_test_application_device_at(application, DeviceCount() - 1);
  const char* source_path = std::getenv("LOOM_SERVE_TEST_MODEL_SOURCE");
  ASSERT_NE(source_path, nullptr);
  const std::string source = source_path;
  directory_ = source.substr(0, source.find_last_of('/'));
  if (SourceServiceDirectStageSource()[0]) {
    ASSERT_NO_FATAL_FAILURE(UseDirectPackage());
  } else {
    ASSERT_NO_FATAL_FAILURE(CreateModel(directory_));
    ASSERT_NO_FATAL_FAILURE(CreateService(0));
  }
  auto options = loom_serve_http_server_options_default();
  IREE_ASSERT_OK(
      loom_serve_http_server_allocate(loom_serve_host_proactor_pool(host_),
                                      &options, host_allocator_, &server_));
  auto listener_options = loom_serve_http_listener_options_default();
  listener_options.address = IREE_SV("127.0.0.1:0");
  IREE_ASSERT_OK(loom_serve_http_listener_allocate(
      server_, &listener_options, {this, ListenerAllocatorControl},
      &listener_));
  IREE_ASSERT_OK(http_.Initialize(server_, listener_));
}

iree_status_t SourceEmbeddingServiceTest::ListenerAllocatorControl(
    void* self, iree_allocator_command_t command, const void* parameters,
    void** inout_pointer) {
  auto* test = static_cast<SourceEmbeddingServiceTest*>(self);
  if (command != IREE_ALLOCATOR_COMMAND_FREE &&
      test->fail_listener_allocations_.load()) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "registration transport witness");
  }
  return test->host_allocator_.ctl(test->host_allocator_.self, command,
                                   parameters, inout_pointer);
}

void SourceEmbeddingServiceTest::CreateModel(const std::string& directory) {
  loom_serve::testing::ModelDocument document;
  IREE_ASSERT_OK(document.Load(directory + "/model.json"));
  const loom_serve_finite_model_options_t model_options = {
      .sources = document.value().sources,
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{\"batch_units\":64}"),
  };
  loom_serve_test_embedding_model_lifecycle_t load;
  IREE_ASSERT_OK(load.Run(device_, model_options));
  model_ = load.model;
  EXPECT_EQ(loom_serve_finite_model_batch_limit(model_), 64u);
}

void SourceEmbeddingServiceTest::CreateService(
    iree_duration_t heartbeat_interval) {
  IREE_ASSERT_OK(loom_serve_finite_service_allocate(
      host_, model_, loom_serve_embedding_protocol(), IREE_SV("fixture"), 16,
      16, heartbeat_interval, nullptr, host_allocator_, &service_));
  interface_ = loom_serve_finite_service_interface(service_);
  EXPECT_EQ(interface_.capacity_domain,
            loom_serve_device_residency_cache(device_));
  const loom_serve_wake_callback_t observer = {
      [](void* self) {
        auto* test = static_cast<SourceEmbeddingServiceTest*>(self);
        test->wakes_.fetch_add(1);
        iree_notification_post(&test->notification_, IREE_ALL_WAITERS);
      },
      this};
  interface_.set_observer(interface_.self, observer);
  if (auto* pool = loom_serve_device_memory_pool(device_)) {
    loom_serve_memory_pool_set_observer(pool, observer.wake,
                                        observer.user_data);
  }
}

void SourceEmbeddingServiceTest::TearDown() {
  if (http_.joinable()) {
    loom_serve_http_server_request_stop(server_);
    IREE_EXPECT_OK(http_.Join());
  }
  if (service_) {
    interface_.set_observer(interface_.self, {});
    loom_serve_finite_service_free(service_);
  }
  IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
  loom_serve_http_server_free(server_);
  IREE_EXPECT_OK(loom_serve_finite_model_free(model_));
  IREE_EXPECT_OK(loom_serve_finite_model_free(retained_));
  if (auto* pool = loom_serve_device_memory_pool(device_)) {
    loom_serve_memory_pool_set_observer(pool, nullptr, nullptr);
  }
  iree_notification_deinitialize(&notification_);
}

int SourceEmbeddingServiceTest::Send(const std::string& body,
                                     const char* target, const char* method,
                                     const std::string& headers) {
  if (http_.failed() || HasFatalFailure()) {
    return -1;
  }
  const std::string request = std::string(method) + " " + target +
                              " HTTP/1.1\r\nHost: local\r\nContent-Type: "
                              "application/json\r\nContent-Length: " +
                              std::to_string(body.size()) + "\r\n" + headers +
                              "\r\n" + body;
  int client = -1;
  [&] { IREE_ASSERT_OK(http_.Send(request, &client)); }();
  return client;
}

std::string SourceEmbeddingServiceTest::Read(int client, bool allow_empty) {
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

void SourceEmbeddingServiceTest::Accept(
    int client, loom_serve_http_connection_t** out_connection) {
  ASSERT_FALSE(HasFatalFailure());
  const loom_serve_http_request_t* request = nullptr;
  loom_serve_http_connection_t* connection = nullptr;
  IREE_ASSERT_OK(http_.TakeRequest(client, &connection, &request));
  IREE_ASSERT_OK(interface_.accept(interface_.self, connection, request, 0));
  if (out_connection) {
    *out_connection = connection;
  }
}

bool SourceEmbeddingServiceTest::Step(loom_serve_service_flags_t flags) {
  const uint64_t previous = wakes_.load();
  bool progress = false;
  IREE_EXPECT_OK(interface_.advance(interface_.self, flags, &progress));
  auto* pool = loom_serve_device_memory_pool(device_);
  if (interface_.is_active(interface_.self) ||
      (pool && loom_serve_memory_pool_is_pending(pool))) {
    struct loom_serve_test_embedding_wait_t {
      // Shared release counter.
      std::atomic<uint64_t>* wakes;
      // Last event already consumed by this turn.
      uint64_t previous;
    } wait{&wakes_, previous};
    iree_notification_await(
        &notification_,
        [](void* data) {
          const auto* wait =
              static_cast<loom_serve_test_embedding_wait_t*>(data);
          return wait->wakes->load() > wait->previous;
        },
        &wait, iree_infinite_timeout());
    return true;
  }
  // A native completion can arrive during advance after capacity was denied.
  // Consume that wake with another owner turn instead of ending the drain.
  return progress || wakes_.load() > previous;
}

void SourceEmbeddingServiceTest::Drain() {
  while (Step()) {
  }
}

void SourceEmbeddingServiceTest::CheckAbandonedCohort(
    loom_serve_service_flags_t flags, const char* expected) {
  const char* bodies[] = {R"({"model":"fixture","input":"a"})",
                          R"({"model":"fixture","input":"bbb"})"};
  int clients[IREE_ARRAYSIZE(bodies)] = {};
  loom_serve_http_connection_t* connections[IREE_ARRAYSIZE(bodies)] = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(bodies); ++i) {
    clients[i] = Send(bodies[i]);
    ASSERT_NO_FATAL_FAILURE(Accept(clients[i], &connections[i]));
  }
  ASSERT_TRUE(Step());  // CPU request preparation.
  ASSERT_TRUE(Step());  // Source batch preparation.
  ASSERT_TRUE(Step());  // Accepted device work; readiness may still be pending.
  ASSERT_STREQ(interface_.snapshot(interface_.self).phase, "running");

  if (!iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    for (size_t i = 0; i < IREE_ARRAYSIZE(clients); ++i) {
      const linger reset = {1, 0};
      ASSERT_EQ(
          setsockopt(clients[i], SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
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
  while (interface_.is_pending(interface_.self)) {
    ASSERT_TRUE(Step(flags));
  }
  EXPECT_STREQ(interface_.snapshot(interface_.self).phase, "idle");
  EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 0u);
  EXPECT_FALSE(interface_.is_pending(interface_.self));
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 2u);
  EXPECT_EQ(interface_.telemetry.counters->input_tokens, 4u);
  EXPECT_EQ(interface_.telemetry.counters->batches, 1u);
  Drain();
  if (iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    for (int client : clients) {
      EXPECT_TRUE(Read(client, /*allow_empty=*/true).empty());
    }
    interface_.set_observer(interface_.self, {});
    loom_serve_finite_service_free(std::exchange(service_, nullptr));
    ASSERT_NO_FATAL_FAILURE(CreateService(0));
  }

  const int next = Send(R"({"model":"fixture","input":[0]})");
  ASSERT_NO_FATAL_FAILURE(Accept(next));
  Drain();
  EXPECT_NE(Read(next).find(expected), std::string::npos);
  EXPECT_EQ(interface_.telemetry.counters->completed, 1u);
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
}

void SourceEmbeddingServiceTest::StartRouter() {
  interface_.set_observer(interface_.self, {});
  http_.Start([this]() -> iree_status_t {
    loom_serve_registry_t* registry = nullptr;
    IREE_RETURN_IF_ERROR(loom_serve_registry_allocate(
        1, &interface_, 0, {}, host_allocator_, &registry));
    iree_status_t status = loom_serve_router_run(
        loom_serve_host_executor(host_), server_, registry,
        {
            .request_quota = 16,
            .holding_capacity = 8,
            .resources = {.device_group = group_},
        },
        host_allocator_);
    return iree_status_join(status, loom_serve_registry_free(registry));
  });
}

void SourceEmbeddingServiceTest::StartManagedRouter(iree_host_size_t capacity) {
  if (service_) {
    interface_.set_observer(interface_.self, {});
  }
  http_.Start([this, capacity]() -> iree_status_t {
    const loom_serve_model_registration_options_t options = {
        .device_group = group_,
        .heartbeat_interval = 1000000,
    };
    loom_serve_registry_t* registry = nullptr;
    IREE_RETURN_IF_ERROR(loom_serve_registry_allocate(
        service_ ? 1 : 0, &interface_, capacity,
        loom_serve_model_registration_factory(&options), host_allocator_,
        &registry));
    iree_status_t status = loom_serve_router_run(
        loom_serve_host_executor(host_), server_, registry,
        {
            .request_quota = 16,
            .holding_capacity = 8,
            .resources = {.device_group = group_},
        },
        host_allocator_);
    EXPECT_TRUE(loom_serve_registry_counters(registry, &retired_counters_));
    return iree_status_join(status, loom_serve_registry_free(registry));
  });
}

std::string SourceEmbeddingServiceTest::Declaration(const std::string& name,
                                                    const std::string& source,
                                                    const char* admission,
                                                    const std::string& base) {
  return "{\"name\":\"" + name + "\",\"base_directory\":\"" +
         std::filesystem::absolute(base.empty() ? source : base).string() +
         "\",\"admission\":\"" + admission +
         "\",\"definition\":" + loom_serve::testing::ReadDefinition(source) +
         "}";
}

std::string SourceEmbeddingServiceTest::ModelStatus(const std::string& name) {
  const std::string listing = Read(Send({}, "/v1/status", "GET"));
  const size_t begin = listing.find("\"id\":\"" + name + "\"");
  if (begin == std::string::npos) {
    return {};
  }
  // Exact serialized object boundaries, not a readiness read of native state.
  const size_t end = listing.find("{\"object\":\"model\"", begin);
  return listing.substr(begin, end == std::string::npos ? end : end - begin);
}

std::string SourceEmbeddingServiceTest::WaitModel(const std::string& name,
                                                  const std::string& marker) {
  while (!http_.failed() && !HasFatalFailure()) {
    std::string state = ModelStatus(name);
    if (marker.empty() ? state.empty()
                       : state.find(marker) != std::string::npos) {
      return state;
    }
  }
  return {};
}

void SourceEmbeddingServiceTest::CheckInvalidSubmissionRetirement(
    const std::string& package) {
  StartManagedRouter();
  ASSERT_EQ(Read(Send(Declaration("broken", package, "drained"),
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  ASSERT_EQ(Read(Send(Declaration("cold", directory_, "drained"),
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  HttpClient cold(Send(R"({"model":"cold","input":"a"})"));
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_NO_FATAL_FAILURE(WaitModel("cold", R"("held_requests":1)"));
  HttpClient broken(Send(R"({"model":"broken","input":"a"})"));
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_NO_FATAL_FAILURE(WaitModel("broken", R"("held_requests":1)"));
  ASSERT_EQ(Read(Send(R"({"model":"broken"})", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);

  // Only successfully admitted work can reach the invalid second stage and
  // terminate the router. Assertion exits instead stop it in TearDown.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, http_.Join());
  EXPECT_TRUE(Read(broken.release(), /*allow_empty=*/true).empty());
  EXPECT_TRUE(Read(cold.release(), /*allow_empty=*/true).empty());
  // The cold request never left the router's holding set and is observed
  // through its aborted claim, not a model's execution counters.
  EXPECT_EQ(retired_counters_.requests, 1u);
  EXPECT_EQ(retired_counters_.failed, 1u);
  EXPECT_EQ(retired_counters_.cancelled, 0u);
  // Queue allocation records may still own backing after model retirement.
  // Trim is nonblocking; group destruction joins the remaining records.
  IREE_ASSERT_OK(loom_serve_device_trim(device_, 0));
}

uint64_t SourceEmbeddingServiceTest::CompletedRequests() {
  const std::string metrics = Read(Send({}, "/metrics", "GET"));
  const std::string prefix = "\nloom_serve_server_completed_total ";
  const size_t position = metrics.find(prefix);
  EXPECT_NE(position, std::string::npos);
  return position == std::string::npos
             ? 0
             : std::stoull(metrics.substr(position + prefix.size()));
}

uint64_t SourceEmbeddingServiceTest::Number(const std::string& response,
                                            const char* field,
                                            const char* object) {
  uint64_t result = 0;
  [&] {
    const auto start = response.find("\r\n\r\n");
    ASSERT_NE(start, std::string::npos) << response;
    auto value = iree_make_string_view(response.data() + start + 4,
                                       response.size() - start - 4);
    if (object) {
      IREE_ASSERT_OK(iree_json_lookup_object_value(
          value, iree_make_cstring_view(object), &value));
    }
    IREE_ASSERT_OK(iree_json_lookup_object_value(
        value, iree_make_cstring_view(field), &value));
    IREE_ASSERT_OK(iree_json_parse_uint64(value, &result));
  }();
  return result;
}

std::string SourceEmbeddingServiceTest::UploadVariant(
    uint64_t base, const std::string& directory, const char* provider,
    const char* replacement, bool include_parameter_source) {
  loom_serve::testing::SourcePackage package;
  std::ifstream input(directory + "/" + provider);
  EXPECT_TRUE(input.good());
  std::string source((std::istreambuf_iterator<char>(input)), {});
  const std::string original = "scalar.constant 1 : i32";
  const size_t position = source.find(original);
  EXPECT_NE(position, std::string::npos);
  if (position == std::string::npos) {
    return {};
  }
  source.replace(position, original.size(), replacement);
  package.Add(std::string("model/") + provider, source);
  [&] {
    IREE_ASSERT_OK(
        package.AddFile("model/config.loom", directory_ + "/config.loom"));
    if (include_parameter_source) {
      IREE_ASSERT_OK(
          package.AddFile("model/weights.loom", directory + "/weights.loom"));
    }
  }();
  const std::string json =
      "{\"model\":\"fixture\",\"base\":" + std::to_string(base) +
      ",\"sources\":[\"config.loom\",\"" + provider + "\"" +
      (include_parameter_source ? ",\"weights.loom\"" : "") +
      "],\"wait\":true}";
  int client = -1;
  [&] {
    IREE_ASSERT_OK(
        http_.Send(package.Request("/v1/models/variants", json), &client));
  }();
  return Read(client);
}

std::string SourceEmbeddingServiceTest::SelectVariant(uint64_t variant,
                                                      uint64_t expected) {
  return Read(
      Send("{\"model\":\"fixture\",\"variant\":" + std::to_string(variant) +
               ",\"expected\":" + std::to_string(expected) + "}",
           "/v1/models/variants/select"));
}

void SourceEmbeddingServiceTest::WriteCheckpointPackage(
    const std::string& target) {
  ASSERT_TRUE(std::filesystem::create_directory(target));
  for (const char* file :
       {"config.loom", "control.loom", "embed.loom", "tokenizer.json"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           target + "/" + file));
  }
  const char* checkpoint_source =
      std::getenv("LOOM_SERVE_TEST_CHECKPOINT_STAGE_SOURCE");
  for (const char* file : {"weights.loom", "policy.loom"}) {
    const std::string source =
        checkpoint_source && std::string(file) == "weights.loom"
            ? checkpoint_source
            : directory_ + "/variants/" + file;
    ASSERT_TRUE(std::filesystem::copy_file(source, target + "/" + file));
  }
  std::ifstream input(directory_ + "/prepare.loom");
  ASSERT_TRUE(input.good());
  std::string prepare((std::istreambuf_iterator<char>(input)), {});
  const std::string root = "bytes(\"656d626564\")";
  const auto root_position = prepare.find(root);
  ASSERT_NE(root_position, std::string::npos);
  prepare.replace(root_position, root.size(),
                  "bytes(\"656d6265645f776974685f706172616d6574657273\")");
  const auto result_position = prepare.find("  func.return %units,");
  ASSERT_NE(result_position, std::string::npos);
  prepare.insert(result_position, R"(
  %parameter_index = scalar.constant 0 : i32
  %parameter_input = scalar.constant 0 : i32
  %parameter_path = global.load @checkpoint : buffer
  %parameter_policy = global.load @policy : buffer
  %parameter_source = func.call @source(%directory, %parameter_path) : (buffer, buffer) -> (i32)
  func.call @parameter(%stage, %parameter_index, %parameter_input, %parameter_source, %parameter_policy) : (i32, i32, i32, i32, buffer) -> ()
)");
  const std::string declarations = R"(
func.decl import("prepare") @source(%directory: buffer, %path: buffer) -> (i32)
func.decl import("prepare") @parameter(%stage: i32, %binding: i32, %input: i32, %source: i32, %policy: buffer)
global.rodata.def @checkpoint = bytes("776569676874732e7361666574656e736f7273")
global.rodata.def @policy = bytes("706f6c6963792e6c6f6f6d")
)";
  std::ofstream bootstrap(target + "/prepare.loom");
  bootstrap << declarations << prepare;
  bootstrap.close();
  ASSERT_TRUE(bootstrap.good());
  std::ofstream manifest(target + "/model.json");
  manifest
      << R"({"name":"checkpoint","definition":{"kind":"embedding","prepare":"prepare.loom","control":"control.loom","sources":["config.loom","embed.loom","weights.loom"],"assets":"."}})";
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

void SourceEmbeddingServiceTest::UseDirectPackage() {
  const std::string target =
      iree::testing::MakeTempFilePath("direct_embedding");
  ASSERT_TRUE(std::filesystem::create_directory(target));
  for (const char* file : {"config.loom", "control.loom", "tokenizer.json"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           target + "/" + file));
  }
  const std::string stage = SourceServiceDirectStageSource()[0]
                                ? SourceServiceDirectStageSource()
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
  while ((position = prepare.find("@stage(", position)) != std::string::npos) {
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
    loom_serve_finite_service_free(std::exchange(service_, nullptr));
  }
  IREE_ASSERT_OK(loom_serve_finite_model_free(std::exchange(model_, nullptr)));
  ASSERT_NO_FATAL_FAILURE(CreateModel(target));
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
}

}  // namespace loom_serve::testing::embedding
