// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/runtime/storage/memory.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/router.h"
#include "loom/serve/service/speech/model.h"
#include "loom/serve/service/speech/service.h"
#include "loom/serve/service/speech/service_test.h"
#include "loom/serve/service/speech/stream.h"
#include "loom/serve/telemetry/latency.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/native_allocator.h"
#include "loom/serve/testing/package.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/testing/stalled_http_client.h"
#include "loom/serve/testing/worker_gate.h"

IREE_FLAG(string, model_source, "", "Speech fixture bootstrap source.");

namespace {

using loom_serve::testing::AcquireApplication;
using loom_serve::testing::kSpeechSlabSize;
using loom_serve::testing::ObservationCount;
using loom_serve::testing::OpenSpeech;
using loom_serve::testing::SpeechElasticBacking;
using loom_serve::testing::SpeechStream;
using loom_serve::testing::SpeechWorkerCount;

class SourceSpeechServiceTest : public ::testing::TestWithParam<bool> {
 protected:
  virtual bool NeedsStandaloneModel() const { return true; }

  static void WriteCheckpoint(const std::string& path,
                              const std::string& header,
                              const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    uint8_t length[8];
    iree_unaligned_store_le_u64(length, header.size());
    file.write(reinterpret_cast<const char*>(length), sizeof(length));
    file.write(header.data(), header.size());
    for (float value : values) {
      uint8_t bytes[4];
      iree_unaligned_store_le_f32(bytes, value);
      file.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }
    file.close();
    ASSERT_TRUE(file.good());
  }

  void SetUp() override {
    IREE_ASSERT_OK(AcquireApplication(allocator_, &host_, &group_));
    directory_ =
        std::filesystem::path(FLAG_model_source).parent_path().string();
    const bool weighted = GetParam();
    if (weighted) {
      assets_ = iree::testing::MakeTempFilePath("speech_assets");
      ASSERT_TRUE(std::filesystem::create_directory(assets_));
      ASSERT_TRUE(
          std::filesystem::copy_file(directory_ + "/variants/tokenizer.json",
                                     assets_ + "/tokenizer.json"));
      // Phase tensors span a second file in the same source-declared domain.
      // The voice archive repeats a key but retains independent ownership.
      const std::string factor =
          R"({"factor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
      ASSERT_NO_FATAL_FAILURE(
          WriteCheckpoint(assets_ + "/weights.safetensors", factor, {0.5f}));
      ASSERT_NO_FATAL_FAILURE(
          WriteCheckpoint(assets_ + "/voices.safetensors", factor, {0.25f}));
      ASSERT_NO_FATAL_FAILURE(WriteCheckpoint(
          assets_ + "/phases.safetensors",
          R"({"encoder":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
          R"("decoder":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",
          {0.125f, 4.0f}));
    }
    if (NeedsStandaloneModel()) {
      const iree_string_view_t providers[] = {
          IREE_SVL("generate.loom"), IREE_SVL("variants/weights.loom")};
      const loom_serve_speech_model_options_t options = {
          .sources =
              {
                  .base_directory = iree_make_cstring_view(directory_.c_str()),
                  .prepare = weighted ? IREE_SV("variants/prepare.loom")
                                      : IREE_SV("prepare.loom"),
                  .control = IREE_SV("control.loom"),
                  .count = weighted ? 2u : 1u,
                  .values = providers,
              },
          .asset_directory = iree_make_cstring_view(assets_.c_str()),
      };
      IREE_ASSERT_OK(loom_serve_speech_model_allocate(
          loom_serve_device_group_at(group_, 0), &options, allocator_,
          &model_));
      generator_ = loom_serve_speech_model_generator(model_);
      EXPECT_EQ(generator_.tokenizer != nullptr, weighted);
      EXPECT_EQ(generator_.format.sample_rate, 24000u);
      EXPECT_EQ(generator_.format.channels, 1u);
      EXPECT_EQ(generator_.block_capacity, 5u);
      EXPECT_EQ(generator_.request_capacity, 4u);
      EXPECT_EQ(generator_.batch_units, 4u);
    }
  }

  void TearDown() override {
    if (http_.joinable()) {
      loom_serve_http_server_request_stop(server_);
      IREE_EXPECT_OK(http_.Join());
    }
    if (service_) {
      AdvanceUntil([&] { return !interface_.is_pending(interface_.self); },
                   LOOM_SERVE_SERVICE_FLAG_STOPPING);
      interface_.set_observer(interface_.self, {});
      loom_serve_speech_service_free(service_);
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    IREE_EXPECT_OK(loom_serve_speech_model_free(model_));
  }

  void CreateServer() {
    const auto options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        loom_serve_host_proactor_pool(host_), &options, allocator_, &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(server_, &listener_options,
                                                     allocator_, &listener_));
    IREE_ASSERT_OK(http_.Initialize(server_, listener_));
  }

  void StartManaged() {
    ASSERT_EQ(model_, nullptr);
    ASSERT_NO_FATAL_FAILURE(CreateServer());
    http_.Start([this]() -> iree_status_t {
      const loom_serve_model_registration_options_t options = {
          .device_group = group_,
          .heartbeat_interval = 0,
      };
      loom_serve_registry_t* registry = nullptr;
      IREE_RETURN_IF_ERROR(loom_serve_registry_allocate(
          0, nullptr, 2, loom_serve_model_registration_factory(&options),
          allocator_, &registry));
      iree_status_t status = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {
              .request_quota = 8,
              .holding_capacity = 8,
              .resources = {.device_group = group_},
          },
          allocator_);
      return iree_status_join(status, loom_serve_registry_free(registry));
    });
  }

  void StartManual() {
    ASSERT_NO_FATAL_FAILURE(CreateServer());
    const loom_serve_speech_service_options_t options = {
        .name = IREE_SV("speech"),
        .pending_capacity = 4,
        .heartbeat_interval = 0,
    };
    IREE_ASSERT_OK(loom_serve_speech_service_allocate(
        host_, generator_, &options, nullptr, allocator_, &service_));
    interface_ = loom_serve_speech_service_interface(service_);
    interface_.set_observer(interface_.self,
                            {[](void* self) {
                               iree_notification_post(
                                   static_cast<iree_notification_t*>(self),
                                   IREE_ALL_WAITERS);
                             },
                             loom_serve_http_server_notification(server_)});
  }

  template <typename Predicate>
  void AdvanceUntil(Predicate ready, loom_serve_service_flags_t flags = 0) {
    auto* notification = loom_serve_http_server_notification(server_);
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      auto status = interface_.advance(interface_.self, flags, &progress);
      if (ready() || progress || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
      loom_serve_telemetry_record_t record;
      while (loom_serve_telemetry_stream_read(interface_.telemetry.stream,
                                              &telemetry_cursor_, &record) ==
             LOOM_SERVE_TELEMETRY_READ_RECORD) {
        if (record.event == &LOOM_SERVE_TELEMETRY_BATCH_SUBMIT) {
          largest_batch_ = std::max(largest_batch_, record.data.batch.requests);
        }
      }
    }
  }

  void Accept(int client, loom_serve_http_connection_t** connection) {
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(http_.TakeRequest(client, connection, &request));
    IREE_ASSERT_OK(interface_.accept(interface_.self, *connection, request, 0));
  }

  int Send(const std::string& method, const std::string& target,
           const std::string& body = "") {
    const std::string request = method + " " + target +
                                " HTTP/1.1\r\nHost: local\r\nContent-Type: "
                                "application/json\r\nContent-Length: " +
                                std::to_string(body.size()) + "\r\n\r\n" + body;
    int client = -1;
    [&] { IREE_ASSERT_OK(http_.Send(request, &client)); }();
    return client;
  }

  std::string Read(int client) {
    std::string response;
    [&] { IREE_ASSERT_OK(http_.Read(client, false, &response)); }();
    return response;
  }

  static std::string Request(const char* voice = "positive",
                             const std::string& input = "hi",
                             uint64_t seed = 3) {
    return std::string(R"({"model":"speech","input":")") + input +
           R"(","voice":")" + voice + R"(","seed":")" + std::to_string(seed) +
           R"(","response_format":"pcm"})";
  }

  static void CheckAudio(const std::string& response, size_t frames,
                         int sign = 1, uint64_t seed = 3) {
    ASSERT_EQ(response.find("HTTP/1.1 200 OK\r\n"), 0u) << response;
    const size_t header_end = response.find("\r\n\r\n");
    ASSERT_NE(header_end, std::string::npos);
    EXPECT_NE(response.substr(0, header_end)
                  .find("\r\nContent-Type: audio/pcm;rate=24000;channels=1"),
              std::string::npos);
    const auto body = response.substr(header_end + 4);
    ASSERT_EQ(body.size(), frames * 2);
    for (size_t i = 0; i < frames; ++i) {
      const int16_t expected =
          sign * static_cast<int16_t>((i + seed) % 8) * 4096;
      EXPECT_EQ(iree_unaligned_load_le_u16(body.data() + i * 2),
                static_cast<uint16_t>(expected));
    }
  }

  iree_status_t Prepare(const char* voice, uint64_t seed, void** invocation,
                        size_t row = 0, const char* text = "hi") {
    loom_serve_speech_request_t request;
    const std::string json = std::string(R"({"model":"speech","input":")") +
                             text + R"(","voice":")" + voice +
                             R"(","response_format":"pcm","seed":")" +
                             std::to_string(seed) + "\"}";
    IREE_RETURN_IF_ERROR(loom_serve_speech_request_initialize(
        iree_make_string_view(json.data(), json.size()), IREE_SV("speech"),
        allocator_, &request));
    iree_status_t status =
        generator_.prepare(generator_.self, row, &request, invocation);
    loom_serve_speech_request_deinitialize(&request);
    return status;
  }

  iree_status_t Admit(void* invocation) {
    loom_serve_residency_admission_t admission;
    loom_serve_execution_frontier_t frontier;
    IREE_RETURN_IF_ERROR(generator_.try_admit(generator_.self, invocation,
                                              &admission, &frontier));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    IREE_RETURN_IF_ERROR(Wait(frontier));
    generator_.complete_admit(generator_.self, invocation);
    return iree_ok_status();
  }

  iree_status_t Wait(const loom_serve_execution_frontier_t& frontier) {
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
      status = loom_serve_workspace_wait(
          loom_serve_device_workspace(loom_serve_device_group_at(group_, 0)),
          frontier.semaphores[i], frontier.values[i]);
    }
    return status;
  }

  iree_status_t Retire(void* invocation) {
    bool complete = false;
    while (!complete) {
      loom_serve_execution_frontier_t frontier;
      IREE_RETURN_IF_ERROR(generator_.release(generator_.self, invocation,
                                              &complete, &frontier));
      if (!complete) {
        IREE_RETURN_IF_ERROR(Wait(frontier));
      }
    }
    return iree_ok_status();
  }

  iree_status_t Submit(const std::vector<void*>& invocations,
                       loom_serve_execution_frontier_t* frontier) {
    uint64_t units = 0;
    uint64_t tag = 0;
    for (size_t i = 0; i < invocations.size(); ++i) {
      loom_serve_speech_work_t work;
      IREE_RETURN_IF_ERROR(
          generator_.prepare_work(generator_.self, invocations[i], &work));
      if (i) {
        EXPECT_EQ(work.tag, tag);
      }
      tag = work.tag;
      units += work.units;
    }
    EXPECT_LE(units, generator_.batch_units);
    IREE_RETURN_IF_ERROR(generator_.prepare_batch(
        generator_.self, invocations.size(), invocations.data()));
    return generator_.submit(generator_.self, frontier);
  }

  iree_status_t Generate(void* invocation, size_t stop_frames,
                         std::vector<float>* samples,
                         std::vector<size_t>* blocks) {
    bool complete = false;
    while (!complete && samples->size() < stop_frames) {
      loom_serve_execution_frontier_t frontier;
      IREE_RETURN_IF_ERROR(Submit({invocation}, &frontier));
      IREE_RETURN_IF_ERROR(Wait(frontier));
      loom_serve_speech_block_t block;
      IREE_RETURN_IF_ERROR(generator_.collect(generator_.self, 0, &block));
      EXPECT_EQ(block.position, samples->size());
      blocks->push_back(block.frame_count);
      for (size_t i = 0; i < block.frame_count; ++i) {
        samples->push_back(
            iree_unaligned_load_le_f32(block.samples.data + i * 4));
      }
      complete = block.complete;
      generator_.release_batch(generator_.self);
    }
    return iree_ok_status();
  }

  // Process application owns the host/device through global test teardown.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Shared worker owner, retained by the device group.
  loom_serve_host_t* host_ = nullptr;
  // Exact device/queue group for the current provider registration.
  loom_serve_device_group_t* group_ = nullptr;
  // Prepared source model, owning failed or unfinished invocations as well.
  loom_serve_speech_model_t* model_ = nullptr;
  // Borrowed callbacks into model_.
  loom_serve_speech_generator_t generator_ = {};
  // Source asset base retained through asynchronous registration.
  std::string directory_;
  // Checkpoint files remain alive through model and registration retirement.
  std::string assets_ = ".";
  // Loopback HTTP owner, created only by endpoint tests.
  loom_serve_http_server_t* server_ = nullptr;
  // Listener outlives every connected client and service turn.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Readiness-driven client and optional outer router thread.
  loom_serve::testing::HttpHarness http_;
  // Optional manually driven production service, joined before model teardown.
  loom_serve_speech_service_t* service_ = nullptr;
  // Borrowed service operations, used only by the serialized test owner.
  loom_serve_service_t interface_ = {};
  // Absolute cursor observing every numerical cohort at bounded owner turns.
  uint64_t telemetry_cursor_ = 0;
  // Largest actual source invocation observed through production telemetry.
  uint64_t largest_batch_ = 0;
};

class RegisteredSourceSpeechServiceTest : public SourceSpeechServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }
};

TEST_P(SourceSpeechServiceTest, SourceStagesPublishVariableAndEmptyBlocks) {
  for (int sign : {1, -1}) {
    void* invocation = nullptr;
    IREE_ASSERT_OK(
        Prepare(sign == 1 ? "positive" : "negative", 3, &invocation));
    IREE_ASSERT_OK(Admit(invocation));
    std::vector<float> samples;
    std::vector<size_t> blocks;
    IREE_ASSERT_OK(Generate(invocation, SIZE_MAX, &samples, &blocks));
    EXPECT_EQ(blocks, (std::vector<size_t>{0, 3, 0, 3, 5, 3, 3}));
    ASSERT_EQ(samples.size(), 17u);
    for (size_t i = 0; i < samples.size(); ++i) {
      EXPECT_EQ(samples[i], sign * static_cast<float>((i + 3) % 8) * 0.125f);
    }
    IREE_ASSERT_OK(Retire(invocation));
  }
}

TEST_P(SourceSpeechServiceTest, RowCapacityDenialLeavesParametersCold) {
  auto* device = loom_serve_device_group_at(group_, 0);
  auto* pool = loom_serve_device_memory_pool(device);
  auto* residency = loom_serve_speech_model_residency(model_);
  loom_serve_memory_statistics_t peer_statistics = {};
  const auto release_peer = [](loom_serve_virtual_buffer_t* buffer) {
    IREE_EXPECT_OK(loom_serve_virtual_buffer_free(buffer));
  };
  std::unique_ptr<loom_serve_virtual_buffer_t, decltype(release_peer)> peer(
      nullptr, release_peer);
  if (pool) {
    // Leave room for parameters and two state pages, but the real source row
    // requires input, conditioning, and stream pages as one admission.
    const uint64_t peer_bytes =
        loom_serve_memory_pool_limit(pool) -
        loom_serve_residency_statistics(residency).reserved_bytes -
        2 * kSpeechSlabSize;
    loom_serve_virtual_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, loom_serve_device_state_geometry(device), peer_bytes, 16,
        &peer_statistics, &buffer));
    peer.reset(buffer);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(buffer, 0, peer_bytes));
  }
  void* invocation = nullptr;
  IREE_ASSERT_OK(Prepare("positive", 3, &invocation));
  if (pool) {
    const uint64_t live = loom_serve_memory_pool_live_bytes(pool);
    loom_serve_residency_admission_t admission;
    loom_serve_execution_frontier_t frontier;
    IREE_ASSERT_OK(generator_.try_admit(generator_.self, invocation, &admission,
                                        &frontier));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(loom_serve_memory_pool_live_bytes(pool), live);
    EXPECT_EQ(loom_serve_residency_activity(residency).activations, 0u);
    peer.reset();
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  }
  IREE_ASSERT_OK(Admit(invocation));
  std::vector<float> samples;
  std::vector<size_t> blocks;
  IREE_ASSERT_OK(Generate(invocation, SIZE_MAX, &samples, &blocks));
  std::vector<float> expected(17);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<float>((i + 3) % 8) * 0.125f;
  }
  EXPECT_EQ(samples, expected);
  IREE_ASSERT_OK(Retire(invocation));
}

TEST_P(SourceSpeechServiceTest,
       ColdStateTransitionsPreserveWarmWaveformProgress) {
  void* warm = nullptr;
  IREE_ASSERT_OK(Prepare("positive", 3, &warm));
  IREE_ASSERT_OK(Admit(warm));
  std::vector<float> warm_samples;
  std::vector<size_t> warm_blocks;
  IREE_ASSERT_OK(Generate(warm, 3, &warm_samples, &warm_blocks));

  auto* device = loom_serve_device_group_at(group_, 0);
  const iree_string_view_t providers[] = {IREE_SVL("generate.loom")};
  const loom_serve_speech_model_options_t options = {
      .sources =
          {
              .base_directory = iree_make_cstring_view(directory_.c_str()),
              .prepare = IREE_SV("prepare.loom"),
              .control = IREE_SV("control.loom"),
              .count = IREE_ARRAYSIZE(providers),
              .values = providers,
          },
  };
  loom_serve_speech_model_t* model = nullptr;
  IREE_ASSERT_OK(
      loom_serve_speech_model_allocate(device, &options, allocator_, &model));
  const auto release_model = [](loom_serve_speech_model_t* value) {
    IREE_EXPECT_OK(loom_serve_speech_model_free(value));
  };
  const std::unique_ptr<loom_serve_speech_model_t, decltype(release_model)>
      owner(model, release_model);
  const auto cold = loom_serve_speech_model_generator(model);
  loom_serve_speech_request_t request;
  IREE_ASSERT_OK(loom_serve_speech_request_initialize(
      IREE_SV(R"({"model":"speech","input":"hi","voice":"negative",)"
              R"("seed":5,"response_format":"pcm"})"),
      IREE_SV("speech"), allocator_, &request));
  void* invocation = nullptr;
  iree_status_t prepared = cold.prepare(cold.self, 0, &request, &invocation);
  loom_serve_speech_request_deinitialize(&request);
  IREE_ASSERT_OK(prepared);
  const auto hold_native = [&]() {
    return SpeechElasticBacking()
               ? std::make_unique<loom_serve::testing::ScopedNativeHold>(device)
               : nullptr;
  };
  loom_serve_execution_frontier_t frontier;
  {
    const auto gate = hold_native();
    loom_serve_residency_admission_t admission;
    IREE_ASSERT_OK(
        cold.try_admit(cold.self, invocation, &admission, &frontier));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    if (SpeechElasticBacking()) {
      EXPECT_EQ(
          loom_serve_device_memory_statistics(device).retained.pending_bytes,
          3 * kSpeechSlabSize);
    }
    IREE_ASSERT_OK(Generate(warm, 6, &warm_samples, &warm_blocks));
    ASSERT_EQ(warm_samples.size(), 6u);
  }
  IREE_ASSERT_OK(Wait(frontier));
  cold.complete_admit(cold.self, invocation);
  std::vector<float> cold_samples;
  bool complete = false;
  while (!complete) {
    loom_serve_speech_work_t work;
    IREE_ASSERT_OK(cold.prepare_work(cold.self, invocation, &work));
    IREE_ASSERT_OK(cold.prepare_batch(cold.self, 1, &invocation));
    IREE_ASSERT_OK(cold.submit(cold.self, &frontier));
    IREE_ASSERT_OK(Wait(frontier));
    loom_serve_speech_block_t block;
    IREE_ASSERT_OK(cold.collect(cold.self, 0, &block));
    EXPECT_EQ(block.position, cold_samples.size());
    for (size_t i = 0; i < block.frame_count; ++i) {
      cold_samples.push_back(
          iree_unaligned_load_le_f32(block.samples.data + i * sizeof(float)));
    }
    complete = block.complete;
    cold.release_batch(cold.self);
  }
  ASSERT_EQ(cold_samples.size(), 17u);
  for (size_t i = 0; i < cold_samples.size(); ++i) {
    EXPECT_EQ(cold_samples[i], -static_cast<float>((i + 5) % 8) * 0.125f);
  }
  {
    const auto gate = hold_native();
    IREE_ASSERT_OK(cold.release(cold.self, invocation, &complete, &frontier));
    EXPECT_EQ(complete, !SpeechElasticBacking());
    if (SpeechElasticBacking()) {
      EXPECT_EQ(
          loom_serve_device_memory_statistics(device).retained.releasing_bytes,
          2 * kSpeechSlabSize);
    }
    IREE_ASSERT_OK(Generate(warm, SIZE_MAX, &warm_samples, &warm_blocks));
  }
  if (!complete) {
    IREE_ASSERT_OK(Wait(frontier));
    IREE_ASSERT_OK(cold.release(cold.self, invocation, &complete, &frontier));
    ASSERT_TRUE(complete);
  }
  ASSERT_EQ(warm_samples.size(), 17u);
  for (size_t i = 0; i < warm_samples.size(); ++i) {
    EXPECT_EQ(warm_samples[i], static_cast<float>((i + 3) % 8) * 0.125f);
  }
  IREE_ASSERT_OK(Retire(warm));
}

TEST_P(SourceSpeechServiceTest, RejectionAndCancellationPreserveReusableModel) {
  void* invocation = nullptr;
  iree_status_t rejected = Prepare("unknown!", 1, &invocation);
  EXPECT_EQ(iree_status_code(rejected), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(rejected);
  EXPECT_EQ(invocation, nullptr);
  IREE_ASSERT_OK(Prepare("positive", 1, &invocation));
  IREE_ASSERT_OK(Admit(invocation));
  std::vector<float> samples;
  std::vector<size_t> blocks;
  IREE_ASSERT_OK(Generate(invocation, 3, &samples, &blocks));
  ASSERT_EQ(samples.size(), 3u);
  IREE_ASSERT_OK(Retire(invocation));
  IREE_ASSERT_OK(Prepare("positive", 1, &invocation));
  IREE_ASSERT_OK(Admit(invocation));
  samples.clear();
  blocks.clear();
  IREE_ASSERT_OK(Generate(invocation, SIZE_MAX, &samples, &blocks));
  ASSERT_EQ(samples.size(), 17u);
  for (size_t i = 0; i < samples.size(); ++i) {
    EXPECT_EQ(samples[i], static_cast<float>((i + 1) % 8) * 0.125f);
  }
  IREE_ASSERT_OK(Retire(invocation));
}

TEST_P(SourceSpeechServiceTest, CohortsPreserveIndependentResidentProgress) {
  struct loom_serve_test_speech_request_t {
    // Source handle owns one stable row until completion or cancellation.
    void* invocation = nullptr;
    // Copied waveform survives cohort feedback reuse.
    std::vector<float> samples;
    // Request-specific voice and random sequence identity.
    int sign = 1;
    // Per-request seed, independent of neighbors and packed ordinal.
    uint64_t seed = 0;
    // Expected complete waveform length from immutable input.
    size_t frames = 0;
  } requests[5];
  const char* texts[] = {"a", "abc", "abcdef", "abcdefgh", "new"};
  auto start = [&](size_t index, size_t row) {
    auto& request = requests[index];
    request.sign = index % 2 ? -1 : 1;
    request.seed = 17 + index * 13;
    request.frames = std::string(texts[index]).size() * 8 + 1;
    IREE_ASSERT_OK(Prepare(request.sign > 0 ? "positive" : "negative",
                           request.seed, &request.invocation, row,
                           texts[index]));
    IREE_ASSERT_OK(Admit(request.invocation));
    // Initialization is a distinct compatible phase, collected after
    // retirement.
    loom_serve_execution_frontier_t frontier;
    IREE_ASSERT_OK(Submit({request.invocation}, &frontier));
    IREE_ASSERT_OK(Wait(frontier));
    loom_serve_speech_block_t block;
    IREE_ASSERT_OK(generator_.collect(generator_.self, 0, &block));
    EXPECT_EQ(block.frame_count, 0u);
    EXPECT_FALSE(block.complete);
    generator_.release_batch(generator_.self);
  };
  auto run = [&](const std::vector<size_t>& selected, size_t cancelled) {
    std::vector<void*> invocations;
    for (size_t index : selected) {
      invocations.push_back(requests[index].invocation);
    }
    loom_serve_execution_frontier_t frontier;
    IREE_ASSERT_OK(Submit(invocations, &frontier));
    // Cancellation after acceptance cannot remove an original batch ordinal.
    // Its storage remains live until the complete cohort has retired.
    IREE_ASSERT_OK(Wait(frontier));
    std::vector<size_t> finished;
    for (size_t ordinal = 0; ordinal < selected.size(); ++ordinal) {
      const size_t index = selected[ordinal];
      auto& request = requests[index];
      loom_serve_speech_block_t block;
      IREE_ASSERT_OK(generator_.collect(generator_.self, ordinal, &block));
      EXPECT_EQ(block.position, request.samples.size());
      for (size_t i = 0; i < block.frame_count; ++i) {
        request.samples.push_back(
            iree_unaligned_load_le_f32(block.samples.data + i * 4));
      }
      if (block.complete || index == cancelled) {
        finished.push_back(index);
      }
    }
    generator_.release_batch(generator_.self);
    for (size_t index : finished) {
      IREE_ASSERT_OK(Retire(requests[index].invocation));
      requests[index].invocation = nullptr;
    }
  };
  size_t tick = 0;
  size_t largest_cohort = 0;
  size_t paused_samples = 0;
  uint64_t full_live_bytes = 0;
  auto* device = loom_serve_device_group_at(group_, 0);
  const auto initial = loom_serve_device_memory_statistics(device).retained;
  const auto workspace =
      loom_serve_workspace_memory_usage(loom_serve_device_workspace(device));
  EXPECT_EQ(initial.committed_bytes,
            initial.cached_bytes + workspace.committed_bytes);
  bool live = true;
  while (live) {
    if (tick < 4) {
      ASSERT_NO_FATAL_FAILURE(start(tick, tick));
    } else if (tick == 4) {
      ASSERT_EQ(requests[0].invocation, nullptr);
      ASSERT_NO_FATAL_FAILURE(start(4, 0));
      paused_samples = requests[2].samples.size();
    }
    std::vector<size_t> selected;
    for (size_t i = 0; i < 5; ++i) {
      const size_t index = tick % 2 ? 4 - i : i;
      // A stalled consumer only makes its own request ineligible.
      if (index == 2 && tick >= 4 && tick < 9) {
        continue;
      }
      if (requests[index].invocation) {
        selected.push_back(index);
      }
    }
    if (selected.size() > largest_cohort) {
      largest_cohort = selected.size();
    }
    if (!selected.empty()) {
      ASSERT_NO_FATAL_FAILURE(run(selected, tick == 4 ? 1 : SIZE_MAX));
    }
    if (tick == 3) {
      const auto memory = loom_serve_device_memory_statistics(device).retained;
      full_live_bytes = memory.committed_bytes - memory.cached_bytes;
    }
    if (tick >= 4 && tick < 9) {
      EXPECT_EQ(requests[2].samples.size(), paused_samples);
    }
    live = tick < 4;
    for (const auto& request : requests) {
      live |= request.invocation != nullptr;
    }
    ++tick;
  }
  EXPECT_EQ(largest_cohort, 4u);
  if (SpeechElasticBacking()) {
    IREE_ASSERT_OK(
        loom_serve_memory_pool_await(loom_serve_device_memory_pool(device)));
    const auto idle = loom_serve_device_memory_statistics(device).retained;
    // Releasing an individual row cannot trim slabs shared by live neighbors.
    // The last release reclaims input/history; reusable conditioning survives.
    EXPECT_EQ(idle.reserved_bytes, initial.reserved_bytes);
    EXPECT_EQ(full_live_bytes - (idle.committed_bytes - idle.cached_bytes),
              2 * kSpeechSlabSize);
  }
  for (size_t index = 0; index < 5; ++index) {
    const auto& request = requests[index];
    if (index == 1) {
      EXPECT_LT(request.samples.size(), request.frames);
    } else {
      EXPECT_EQ(request.samples.size(), request.frames);
    }
    for (size_t i = 0; i < request.samples.size(); ++i) {
      EXPECT_EQ(
          request.samples[i],
          request.sign * static_cast<float>((i + request.seed) % 8) * 0.125f)
          << "request " << index << " sample " << i;
    }
  }
}

TEST_P(RegisteredSourceSpeechServiceTest,
       DynamicRegistrationStandardEndpointAndRemoval) {
  ASSERT_NO_FATAL_FAILURE(StartManaged());
  const std::string definition =
      GetParam()
          ? R"({"kind":"speech","prepare":"variants/prepare.loom",)"
            R"("control":"control.loom","sources":["generate.loom",)"
            R"("variants/weights.loom"],"pending_requests":4,"assets":")" +
                assets_ + "\"}"
          : loom_serve::testing::ReadDefinition(directory_);
  const std::string declaration =
      R"({"name":"speech","base_directory":")" +
      std::filesystem::absolute(directory_).string() + R"(","definition":)" +
      definition + "}";
  const auto registered =
      Read(Send("POST", "/v1/models/register", declaration));
  ASSERT_EQ(registered.find("HTTP/1.1 202"), 0u) << registered;
  CheckAudio(Read(Send("POST", "/v1/audio/speech", Request())), 17);
  if (GetParam()) {
    const auto tokens = Read(
        Send("POST", "/tokenize", R"({"model":"speech","content":"aba"})"));
    EXPECT_EQ(tokens.find("HTTP/1.1 200"), 0u) << tokens;
    EXPECT_NE(tokens.find(R"({"tokens":[0,1,0]})"), std::string::npos)
        << tokens;
  }
  EXPECT_EQ(Read(Send("POST", "/v1/audio/speech", Request("unknown!")))
                .find("HTTP/1.1 400"),
            0u);
  CheckAudio(Read(Send("POST", "/v1/audio/speech", Request("negative"))), 17,
             -1);
  const auto models = Read(Send("GET", "/v1/models"));
  EXPECT_NE(models.find(R"("output_modalities":["audio"])"), std::string::npos);
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
  const auto removing = Read(Send("POST", "/v1/models/unregister",
                                  R"({"model":"speech","mode":"drain"})"));
  ASSERT_EQ(removing.find("HTTP/1.1 202"), 0u) << removing;
  // Discovery hides removing models while their names remain reserved.
  // Status retains each incarnation until all of its resources retire.
  for (;;) {
    const auto status = Read(Send("GET", "/v1/status"));
    ASSERT_EQ(status.find("HTTP/1.1 200"), 0u) << status;
    if (status.find(R"("data":[])") != std::string::npos) {
      break;
    }
  }
  const auto retired = Read(Send("GET", "/metrics"));
  EXPECT_NE(retired.find("loom_serve_server_request_duration_seconds_count{"
                         "outcome=\"completed\"} 2\n"),
            std::string::npos);
  EXPECT_NE(retired.find("loom_serve_server_request_duration_seconds_count{"
                         "outcome=\"rejected\"} 1\n"),
            std::string::npos);
  EXPECT_NE(retired.find("loom_serve_server_first_output_seconds_count{} 2\n"),
            std::string::npos);
  const auto replacement =
      Read(Send("POST", "/v1/models/register", declaration));
  ASSERT_EQ(replacement.find("HTTP/1.1 202"), 0u) << replacement;
  CheckAudio(Read(Send("POST", "/v1/audio/speech", Request())), 17);
}

TEST_P(RegisteredSourceSpeechServiceTest,
       UploadedRegistrationOwnsDeferredSpeechSources) {
  ASSERT_NO_FATAL_FAILURE(StartManaged());
  {
    loom_serve::testing::SourcePackage package;
    for (const char* file : {"prepare.loom", "control.loom", "generate.loom"}) {
      IREE_ASSERT_OK(package.AddFile(std::string("model/") + file,
                                     directory_ + "/" + file));
    }
    const std::string declaration =
        R"({"name":"speech","admission":"drained","definition":)" +
        loom_serve::testing::ReadDefinition(directory_) + "}";
    int client = -1;
    IREE_ASSERT_OK(http_.Send(
        package.Request("/v1/models/register", declaration), &client));
    const auto registered = Read(client);
    ASSERT_EQ(registered.find("HTTP/1.1 202"), 0u) << registered;
    EXPECT_NE(registered.find(R"("registration":"registered")"),
              std::string::npos);
  }
  ASSERT_EQ(Read(Send("POST", "/v1/models/undrain", R"({"model":"speech"})"))
                .find("HTTP/1.1 200"),
            0u);
  CheckAudio(Read(Send("POST", "/v1/audio/speech", Request())), 17);
  CheckAudio(Read(Send("POST", "/v1/audio/speech", Request("negative"))), 17,
             -1);
}

TEST_P(SourceSpeechServiceTest,
       FirstAudioPrecedesEofAndResetRetiresBeforeReuse) {
  ASSERT_NO_FATAL_FAILURE(StartManual());
  const auto& interface = interface_;
  auto* notification = loom_serve_http_server_notification(server_);
  int client = Send("POST", "/v1/audio/speech", Request());
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client, &connection));
  ASSERT_NE(interface.telemetry.latency, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return interface.telemetry.counters->batches == 1 &&
           std::string(interface.snapshot(interface.self).phase) == "ready";
  }));
  // The first source block is empty and publishes neither headers nor audio.
  EXPECT_EQ(ObservationCount(interface.telemetry.latency->first_output), 0u);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return interface.telemetry.counters->batches == 2 &&
           std::string(interface.snapshot(interface.self).phase) == "ready";
  }));
  // The real service has published one three-frame block; no later generation
  // has been submitted. Observe actual socket bytes before advancing it again.
  std::string prefix;
  while (true) {
    IREE_ASSERT_OK(http_.AwaitResponse(client));
    char bytes[256];
    const auto received = recv(client, bytes, sizeof(bytes), 0);
    ASSERT_GT(received, 0);
    prefix.append(bytes, received);
    const auto header_end = prefix.find("\r\n\r\n");
    if (header_end != std::string::npos &&
        prefix.size() >= header_end + 4 + 6) {
      break;
    }
  }
  CheckAudio(prefix, 3);
  EXPECT_EQ(interface.telemetry.counters->completed, 0u);
  EXPECT_EQ(ObservationCount(interface.telemetry.latency->first_output), 1u);
  EXPECT_EQ(ObservationCount(interface.telemetry.latency
                                 ->outcomes[LOOM_SERVE_TELEMETRY_COMPLETED]),
            0u);
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
      connection, iree_infinite_timeout());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface.telemetry.counters->cancelled == 1; }));
  EXPECT_EQ(interface.telemetry.counters->batches, 2u);
  EXPECT_FALSE(interface.is_pending(interface.self));
  EXPECT_EQ(ObservationCount(interface.telemetry.latency
                                 ->outcomes[LOOM_SERVE_TELEMETRY_CANCELLED]),
            1u);
  EXPECT_EQ(ObservationCount(interface.telemetry.latency->first_output), 1u);
  client = Send("POST", "/v1/audio/speech", Request());
  ASSERT_NO_FATAL_FAILURE(Accept(client, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface.telemetry.counters->completed == 1; }));
  CheckAudio(Read(client), 17);
  EXPECT_EQ(ObservationCount(interface.telemetry.latency
                                 ->outcomes[LOOM_SERVE_TELEMETRY_COMPLETED]),
            1u);
  EXPECT_EQ(ObservationCount(interface.telemetry.latency->first_output), 2u);
}

TEST_P(SourceSpeechServiceTest,
       StaggeredHttpCohortsPreserveVoicesAndIndependentCompletion) {
  ASSERT_NO_FATAL_FAILURE(StartManual());
  const std::array<size_t, 5> lengths = {6, 12, 16, 20, 3};
  const std::array<uint64_t, 5> seeds = {1, 2, 5, 6, UINT64_MAX};
  std::array<loom_serve::testing::HttpClient, 5> clients;
  std::array<loom_serve_http_connection_t*, 5> connections = {};
  auto send = [&](size_t i) {
    *clients[i].put() = Send("POST", "/v1/audio/speech",
                             Request(i % 2 ? "negative" : "positive",
                                     std::string(lengths[i], 'a'), seeds[i]));
    ASSERT_NO_FATAL_FAILURE(Accept(clients[i].get(), &connections[i]));
  };
  // Each later arrival follows real progress by the previous request. Their
  // different positions cannot be represented by one shared source cursor.
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_NO_FATAL_FAILURE(send(i));
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
      if (std::string(interface_.snapshot(interface_.self).phase) != "ready") {
        return false;
      }
      loom_serve_http_request_snapshot_t request;
      for (size_t index = 0;
           interface_.inspection.request_at(interface_.self, index, &request);
           ++index) {
        if (request.id == i + 1) {
          return request.counters.batches >= 2;
        }
      }
      return false;
    }));
  }
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 1; }));
  CheckAudio(Read(clients[0].release()), lengths[0] * 8 + 1, 1, seeds[0]);
  EXPECT_EQ(largest_batch_, 4u);
  EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 3u);

  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(clients[1].get(), SOL_SOCKET, SO_LINGER, &reset,
                       sizeof(reset)),
            0);
  ASSERT_EQ(close(clients[1].release()), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connections[1], iree_infinite_timeout());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->cancelled == 1; }));
  ASSERT_NO_FATAL_FAILURE(send(4));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 4; }));
  for (size_t i = 2; i < clients.size(); ++i) {
    CheckAudio(Read(clients[i].release()), lengths[i] * 8 + 1, i % 2 ? -1 : 1,
               seeds[i]);
  }
  EXPECT_EQ(interface_.telemetry.counters->requests, 5u);
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
  EXPECT_FALSE(interface_.is_pending(interface_.self));
}

TEST_P(SourceSpeechServiceTest,
       CancellationDuringPackingDiscardsUnsubmittedCohort) {
  ASSERT_NO_FATAL_FAILURE(StartManual());
  std::array<loom_serve::testing::HttpClient, 2> clients;
  std::array<loom_serve_http_connection_t*, 2> connections = {};
  for (size_t i = 0; i < clients.size(); ++i) {
    *clients[i].put() = Send("POST", "/v1/audio/speech",
                             Request(i ? "negative" : "positive", "abcdefgh"));
    ASSERT_NO_FATAL_FAILURE(Accept(clients[i].get(), &connections[i]));
  }
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return largest_batch_ == 2 &&
           std::string(interface_.snapshot(interface_.self).phase) == "ready";
  }));
  const auto submitted = interface_.telemetry.counters->batches;
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return std::string(interface_.snapshot(interface_.self).phase) ==
           "preparing_batch";
  }));
  // Packing can finish on its CPU worker, but no owner has submitted the plan.
  // Observe the reset before resuming owner turns, preserving that boundary.
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(clients[0].get(), SOL_SOCKET, SO_LINGER, &reset,
                       sizeof(reset)),
            0);
  ASSERT_EQ(close(clients[0].release()), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connections[0], iree_infinite_timeout());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->cancelled == 1; }));
  EXPECT_EQ(interface_.telemetry.counters->batches, submitted);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 1; }));
  CheckAudio(Read(clients[1].release()), 65, -1);
}

TEST_P(SourceSpeechServiceTest,
       NativeCreditAndFinalOutputHaveIndependentOwners) {
  ASSERT_NO_FATAL_FAILURE(StartManual());
  SpeechStream stream(nullptr, loom_serve_speech_stream_close);
  IREE_ASSERT_OK(
      OpenSpeech(service_, IREE_SV("abcdef"), IREE_SV("negative"), 7, &stream));
  ASSERT_NE(stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return loom_serve_speech_stream_peek(stream.get()) != nullptr; }));
  const auto* first = loom_serve_speech_stream_peek(stream.get());
  const std::vector<uint8_t> first_samples(
      first->samples.data, first->samples.data + first->samples.data_length);
  std::array<loom_serve::testing::HttpClient, 2> clients;
  for (size_t i = 0; i < clients.size(); ++i) {
    *clients[i].put() = Send(
        "POST", "/v1/audio/speech",
        Request(i ? "negative" : "positive", std::string(i + 1, 'a'), 11 + i));
    loom_serve_http_connection_t* connection = nullptr;
    ASSERT_NO_FATAL_FAILURE(Accept(clients[i].get(), &connection));
  }
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 2; }));
  for (size_t i = 0; i < clients.size(); ++i) {
    CheckAudio(Read(clients[i].release()), (i + 1) * 8 + 1, i ? -1 : 1, 11 + i);
  }
  EXPECT_EQ(loom_serve_speech_stream_peek(stream.get()), first);
  EXPECT_EQ(first->position, 0u);
  EXPECT_EQ(
      std::vector<uint8_t>(first->samples.data,
                           first->samples.data + first->samples.data_length),
      first_samples);
  size_t frames = 0;
  while (true) {
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
      return loom_serve_speech_stream_peek(stream.get()) != nullptr;
    }));
    const auto* block = loom_serve_speech_stream_peek(stream.get());
    ASSERT_EQ(block->position, frames);
    ASSERT_EQ(block->samples.data_length, block->frame_count * sizeof(float));
    for (size_t i = 0; i < block->frame_count; ++i) {
      EXPECT_EQ(iree_unaligned_load_le_f32(block->samples.data + i * 4),
                -static_cast<float>((frames + i + 7) % 8) * 0.125f);
    }
    frames += block->frame_count;
    if (block->complete) {
      break;
    }
    loom_serve_speech_stream_consume(stream.get());
  }
  EXPECT_EQ(frames, 49u);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 3; }));
  EXPECT_FALSE(interface_.is_pending(interface_.self));
  const auto* final = loom_serve_speech_stream_peek(stream.get());
  const std::vector<uint8_t> final_samples(
      final->samples.data, final->samples.data + final->samples.data_length);
  interface_.set_observer(interface_.self, {});
  loom_serve_speech_service_free(service_);
  service_ = nullptr;
  IREE_ASSERT_OK(loom_serve_speech_model_free(model_));
  model_ = nullptr;
  EXPECT_EQ(loom_serve_speech_stream_format(stream.get()).sample_rate, 24000u);
  EXPECT_EQ(loom_serve_speech_stream_format(stream.get()).channels, 1u);
  EXPECT_EQ(loom_serve_speech_stream_peek(stream.get()), final);
  EXPECT_EQ(
      std::vector<uint8_t>(final->samples.data,
                           final->samples.data + final->samples.data_length),
      final_samples);
  EXPECT_FALSE(loom_serve_speech_stream_is_complete(stream.get()));
  loom_serve_speech_stream_consume(stream.get());
  EXPECT_TRUE(loom_serve_speech_stream_is_complete(stream.get()));
  IREE_EXPECT_OK(loom_serve_speech_stream_status(stream.get()));
}

TEST_P(SourceSpeechServiceTest, NativeSourceRejectionPreservesService) {
  ASSERT_NO_FATAL_FAILURE(StartManual());
  SpeechStream stream(nullptr, loom_serve_speech_stream_close);
  IREE_ASSERT_OK(OpenSpeech(service_, IREE_SV("hello"),
                            IREE_SV("unknown-voice"), 0, &stream));
  ASSERT_NE(stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return loom_serve_speech_stream_is_complete(stream.get()); }));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_speech_stream_status(stream.get()));
  EXPECT_EQ(interface_.telemetry.counters->rejected, 1u);
  const int client = Send("POST", "/v1/audio/speech", Request());
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface_.telemetry.counters->completed == 1; }));
  CheckAudio(Read(client), 17);
}

INSTANTIATE_TEST_SUITE_P(Parameters, SourceSpeechServiceTest,
                         ::testing::Bool());
INSTANTIATE_TEST_SUITE_P(Parameters, RegisteredSourceSpeechServiceTest,
                         ::testing::Bool());

}  // namespace
