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
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/router.h"
#include "loom/serve/service/speech/model.h"
#include "loom/serve/service/speech/service.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/package.h"
#include "loom/serve/testing/service_application.h"

IREE_FLAG(string, model_source, "", "Speech fixture bootstrap source.");
IREE_FLAG(int32_t, test_workers, 1, "Workers in the speech test application.");
IREE_FLAG(bool, test_elastic, false, "Use elastic backing for source speech.");

namespace {

// The target leaf injects the compiler/driver pair. Each test process retains
// one application; separate worker profiles preserve concurrent qualification.
iree_status_t AcquireApplication(iree_allocator_t allocator,
                                 loom_serve_host_t** out_host,
                                 loom_serve_device_group_t** out_group) {
  auto options = loom_serve_test_application_options_default();
  options.worker_count = FLAG_test_workers;
  options.backing = FLAG_test_elastic ? LOOM_SERVE_DEVICE_BACKING_ELASTIC
                                      : LOOM_SERVE_DEVICE_BACKING_FIXED;
  options.slab_size = FLAG_test_elastic ? 2 * 1024 * 1024 : 0;
  loom_serve_test_application_t* application = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_serve_test_application_acquire(&options, allocator, &application));
  *out_host = loom_serve_test_application_host(application);
  *out_group = loom_serve_test_application_device_group(application);
  return iree_ok_status();
}

class SourceSpeechServiceTest : public ::testing::TestWithParam<bool> {
 protected:
  static constexpr uint64_t kSlabSize = 2 * 1024 * 1024;

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
    const iree_string_view_t providers[] = {IREE_SVL("generate.loom"),
                                            IREE_SVL("variants/weights.loom")};
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
        loom_serve_device_group_at(group_, 0), &options, allocator_, &model_));
    generator_ = loom_serve_speech_model_generator(model_);
    EXPECT_EQ(generator_.format.sample_rate, 24000u);
    EXPECT_EQ(generator_.format.channels, 1u);
    EXPECT_EQ(generator_.block_capacity, 5u);
    EXPECT_EQ(generator_.request_capacity, 4u);
    EXPECT_EQ(generator_.batch_units, 4u);
  }

  void TearDown() override {
    if (http_.joinable()) {
      loom_serve_http_server_request_stop(server_);
      IREE_EXPECT_OK(http_.Join());
    }
    if (service_) {
      AdvanceUntil([&] { return !interface_.is_pending(interface_.self); },
                   LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING);
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
          {.request_quota = 8, .holding_capacity = 8}, allocator_);
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
  void AdvanceUntil(Predicate ready,
                    loom_serve_http_service_flags_t flags = 0) {
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
    EXPECT_TRUE(generator_.complete_admit(generator_.self, invocation));
    return iree_ok_status();
  }

  static iree_status_t Wait(const loom_serve_execution_frontier_t& frontier) {
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
      status = iree_hal_semaphore_wait(
          frontier.semaphores[i], frontier.values[i], iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE);
    }
    return status;
  }

  iree_status_t Retire(void* invocation) {
    return generator_.release(generator_.self, invocation);
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
  loom_serve_http_service_t interface_ = {};
  // Absolute cursor observing every numerical cohort at bounded owner turns.
  uint64_t telemetry_cursor_ = 0;
  // Largest actual source invocation observed through production telemetry.
  uint64_t largest_batch_ = 0;
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
  struct Request {
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
  uint64_t full_commitment = 0;
  auto* device = loom_serve_device_group_at(group_, 0);
  const auto initial = loom_serve_device_memory_statistics(device).retained;
  EXPECT_EQ(initial.committed_bytes, 0u);
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
      full_commitment =
          loom_serve_device_memory_statistics(device).retained.committed_bytes;
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
  if (FLAG_test_elastic) {
    const auto idle = loom_serve_device_memory_statistics(device).retained;
    // Releasing an individual row cannot trim slabs shared by live neighbors.
    // The last release reclaims input/history; reusable conditioning survives.
    EXPECT_EQ(idle.reserved_bytes, initial.reserved_bytes);
    EXPECT_EQ(full_commitment - idle.committed_bytes, 2 * kSlabSize);
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

TEST_P(SourceSpeechServiceTest, DynamicRegistrationStandardEndpointAndRemoval) {
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
  EXPECT_EQ(Read(Send("POST", "/v1/models/register", declaration))
                .find("HTTP/1.1 202"),
            0u);
  CheckAudio(Read(Send("POST", "/v1/audio/speech", Request())), 17);
  EXPECT_EQ(Read(Send("POST", "/v1/audio/speech", Request("unknown!")))
                .find("HTTP/1.1 400"),
            0u);
  CheckAudio(Read(Send("POST", "/v1/audio/speech", Request("negative"))), 17,
             -1);
  const auto models = Read(Send("GET", "/v1/models"));
  EXPECT_NE(models.find(R"("output_modalities":["audio"])"), std::string::npos);
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"speech","mode":"drain"})"))
                .find("HTTP/1.1 202"),
            0u);
  while (!http_.failed() && !HasFatalFailure()) {
    if (Read(Send("GET", "/v1/models")).find(R"("data":[])") !=
        std::string::npos) {
      break;
    }
  }
  EXPECT_EQ(Read(Send("POST", "/v1/models/register", declaration))
                .find("HTTP/1.1 202"),
            0u);
  CheckAudio(Read(Send("POST", "/v1/audio/speech", Request())), 17);
}

TEST_P(SourceSpeechServiceTest, UploadedRegistrationOwnsDeferredSpeechSources) {
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
  client = Send("POST", "/v1/audio/speech", Request());
  ASSERT_NO_FATAL_FAILURE(Accept(client, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return interface.telemetry.counters->completed == 1; }));
  CheckAudio(Read(client), 17);
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

// The numerical dependency is replaced; accepted HAL transfers, proactor
// waits, CPU tasks, TCP credit and service retirement remain production paths.
class SpeechReadinessTest : public ::testing::Test {
 protected:
  struct Invocation {
    // Stable row retained independently of its ordinal in the current cohort.
    size_t row = 0;
    // Source cursor advances only after retired feedback collection.
    uint64_t block = 0;
    // Text-derived output length, independent of neighboring requests.
    uint64_t blocks = 0;
    // Immutable voice waveform borrowed by real upload operations.
    std::vector<uint8_t> input;
    // Per-request feedback borrowed through accepted queue retirement.
    std::vector<uint8_t> output;
    // Actual queue allocation, retained through failure retirement.
    iree_hal_buffer_t* buffer = nullptr;
  };

  void Initialize(size_t frames = 3, uint64_t blocks = 4,
                  uint64_t held_block = 1, size_t capacity = 1,
                  size_t held_members = 1, bool start = true) {
    frames_ = frames;
    blocks_ = blocks;
    held_block_ = held_block;
    held_members_ = held_members;
    batch_.reserve(capacity);
    IREE_ASSERT_OK(AcquireApplication(allocator_, &host_, &group_));
    auto* device = loom_serve_device_group_at(group_, 0);
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution_));
    IREE_ASSERT_OK(loom_serve_execution_reserve_workspace(
        execution_, frames * 4 * capacity, 16));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
    const auto transport_options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        loom_serve_host_proactor_pool(host_), &transport_options, allocator_,
        &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(server_, &listener_options,
                                                     allocator_, &listener_));
    IREE_ASSERT_OK(http_.Initialize(server_, listener_));
    const loom_serve_speech_generator_t generator = {
        .self = this,
        .format = {24000, 1},
        .block_capacity = frames_,
        .request_capacity = capacity,
        .batch_units = capacity,
        .prepare =
            [](void* self, iree_host_size_t row,
               const loom_serve_speech_request_t* request, void** out) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              auto invocation = std::make_unique<Invocation>();
              invocation->row = row;
              invocation->blocks = iree_string_builder_size(&request->input);
              invocation->input.resize(test->frames_ * 4);
              invocation->output.resize(invocation->input.size());
              const float sample =
                  iree_string_view_equal(
                      iree_string_builder_view(&request->voice),
                      IREE_SV("negative"))
                      ? -0.5f
                      : 0.5f;
              for (size_t i = 0; i < test->frames_; ++i) {
                iree_unaligned_store_le_f32(invocation->input.data() + i * 4,
                                            sample);
              }
              *out = invocation.get();
              test->invocations_[row] = std::move(invocation);
              test->row_submitted_[row].store(0);
              return iree_ok_status();
            },
        .discard =
            [](void* self, void* invocation) {
              static_cast<SpeechReadinessTest*>(self)
                  ->invocations_[static_cast<Invocation*>(invocation)->row]
                  .reset();
            },
        .try_admit =
            [](void* self, void*, loom_serve_residency_admission_t* out,
               loom_serve_execution_frontier_t* frontier) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              *out = LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
              *frontier = loom_serve_execution_frontier(test->execution_);
              return iree_ok_status();
            },
        .complete_admit = [](void*, void*) { return true; },
        .prepare_work =
            [](void*, void*, loom_serve_speech_work_t* out) {
              *out = {.units = 1, .tag = 0};
              return iree_ok_status();
            },
        .prepare_batch =
            [](void* self, iree_host_size_t count, void* const* invocations) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              test->batch_.clear();
              for (size_t i = 0; i < count; ++i) {
                test->batch_.push_back(
                    static_cast<Invocation*>(invocations[i]));
              }
              return iree_ok_status();
            },
        .submit =
            [](void* self, loom_serve_execution_frontier_t* out) {
              return static_cast<SpeechReadinessTest*>(self)->Submit(out);
            },
        .collect =
            [](void* self, iree_host_size_t ordinal,
               loom_serve_speech_block_t* out) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              auto& invocation = *test->batch_[ordinal];
              *out = {
                  .position = invocation.block * test->frames_,
                  .frame_count = test->frames_,
                  .samples = iree_make_const_byte_span(
                      invocation.output.data(), invocation.output.size()),
                  .complete = invocation.block + 1 == invocation.blocks,
              };
              ++invocation.block;
              return iree_ok_status();
            },
        .release_batch =
            [](void* self) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              for (auto* invocation : test->batch_) {
                iree_hal_buffer_release(invocation->buffer);
                invocation->buffer = nullptr;
              }
              test->batch_.clear();
            },
        .release =
            [](void* self, void* invocation) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              test->invocations_[static_cast<Invocation*>(invocation)->row]
                  .reset();
              ++test->released_;
              test->http_.Notify();
              return iree_ok_status();
            },
    };
    const loom_serve_speech_service_options_t service_options = {
        .name = IREE_SV("speech"),
        .pending_capacity = 1,
        .heartbeat_interval = 0,
    };
    IREE_ASSERT_OK(loom_serve_speech_service_allocate(
        host_, generator, &service_options, nullptr, allocator_, &service_));
    actual_ = loom_serve_speech_service_interface(service_);
    // Observe completed owner turns without replacing any service behavior.
    wrapped_ = actual_;
    auto& wrapped = wrapped_;
    wrapped.self = this;
    wrapped.set_observer = [](void* self, loom_serve_wake_callback_t wake) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      test->actual_.set_observer(test->actual_.self, wake);
    };
    wrapped.is_pending = [](void* self) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      return test->actual_.is_pending(test->actual_.self);
    };
    wrapped.is_active = [](void* self) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      return test->actual_.is_active(test->actual_.self);
    };
    wrapped.accept = [](void* self, loom_serve_http_connection_t* connection,
                        const loom_serve_http_request_t* request,
                        loom_serve_http_service_flags_t flags) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      return test->actual_.accept(test->actual_.self, connection, request,
                                  flags);
    };
    wrapped.advance = [](void* self, loom_serve_http_service_flags_t flags,
                         bool* progress) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      iree_status_t status =
          test->actual_.advance(test->actual_.self, flags, progress);
      if (iree_status_is_ok(status) &&
          std::string(test->actual_.snapshot(test->actual_.self).phase) ==
              "delivering") {
        test->credit_wait_.store(true, std::memory_order_release);
        test->http_.Notify();
      }
      return status;
    };
    wrapped.snapshot = [](void* self) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      return test->actual_.snapshot(test->actual_.self);
    };
    wrapped.inspection.request_at =
        [](void* self, iree_host_size_t index,
           loom_serve_http_request_snapshot_t* out) {
          auto* test = static_cast<SpeechReadinessTest*>(self);
          return test->actual_.inspection.request_at(test->actual_.self, index,
                                                     out);
        };
    if (start) {
      StartRouter();
    } else {
      actual_.set_observer(actual_.self,
                           {[](void* self) {
                              iree_notification_post(
                                  static_cast<iree_notification_t*>(self),
                                  IREE_ALL_WAITERS);
                            },
                            loom_serve_http_server_notification(server_)});
    }
  }

  template <typename Predicate>
  void AdvanceUntil(Predicate ready,
                    loom_serve_http_service_flags_t flags = 0) {
    auto* notification = loom_serve_http_server_notification(server_);
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      auto status = actual_.advance(actual_.self, flags, &progress);
      if (ready() || progress || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
    }
  }

  void StartRouter() {
    http_.Start([this]() -> iree_status_t {
      loom_serve_registry_t* registry = nullptr;
      IREE_RETURN_IF_ERROR(loom_serve_registry_allocate(1, &wrapped_, 0, {},
                                                        allocator_, &registry));
      iree_status_t status = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {.request_quota = 4, .holding_capacity = 4}, allocator_);
      return iree_status_join(status, loom_serve_registry_free(registry));
    });
  }

  iree_status_t Submit(loom_serve_execution_frontier_t* out) {
    iree_status_t status = iree_ok_status();
    bool held = false;
    std::array<iree_hal_transfer_operation_t, 4> uploads = {};
    std::array<iree_hal_transfer_operation_t, 4> downloads = {};
    for (size_t i = 0; i < batch_.size() && iree_status_is_ok(status); ++i) {
      auto& invocation = *batch_[i];
      const auto reservation =
          iree_hal_make_pool_reservation_request(invocation.input.size(), 16);
      status = loom_serve_execution_alloca(execution_, &reservation,
                                           &invocation.buffer);
      held |= invocation.block == held_block_;
      uploads[i] = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload =
              {
                  .source = invocation.input.data(),
                  .target_buffer = invocation.buffer,
                  .length = invocation.input.size(),
              },
      };
      downloads[i] = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = invocation.buffer,
                  .target = invocation.output.data(),
                  .length = invocation.output.size(),
              },
      };
    }
    held &= batch_.size() >= held_members_;
    if (iree_status_is_ok(status) && held) {
      uint64_t value = 1;
      status = loom_serve_execution_barrier(execution_, {1, &gate_, &value});
    }
    uint64_t value = 0;
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_transfer(execution_, batch_.size(),
                                             uploads.data(), &value);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_feedback(execution_, batch_.size(),
                                             downloads.data(), &value);
    }
    if (iree_status_is_ok(status)) {
      auto frontier = loom_serve_execution_frontier(execution_);
      status = loom_serve_execution_barrier(
          execution_, {1, &frontier.semaphores[1], &frontier.values[1]});
    }
    for (size_t i = 0; i < batch_.size() && iree_status_is_ok(status); ++i) {
      status =
          loom_serve_execution_dealloca(execution_, batch_[i]->buffer, &value);
    }
    if (iree_status_is_ok(status)) {
      *out = loom_serve_execution_frontier(execution_);
      for (auto* invocation : batch_) {
        row_submitted_[invocation->row].fetch_add(1);
      }
      largest_batch_.store(std::max(largest_batch_.load(), batch_.size()));
      ++submitted_;
      if (held) {
        held_.store(true, std::memory_order_release);
      }
      http_.Notify();
    }
    return status;
  }

  void ReleaseDependency(bool fail = false) {
    if (!gate_ || gate_released_) {
      return;
    }
    gate_released_ = true;
    failed_ = fail;
    if (fail) {
      iree_hal_semaphore_fail(gate_, iree_make_status(IREE_STATUS_ABORTED,
                                                      "speech device witness"));
    } else {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
    }
  }

  void CheckStatus(iree_status_t status) {
    if (failed_) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
    } else {
      IREE_EXPECT_OK(status);
    }
  }

  void TearDown() override {
    if (server_) {
      loom_serve_http_server_request_stop(server_);
    }
    ReleaseDependency();
    if (http_.joinable()) {
      CheckStatus(http_.Join());
    } else if (service_ && !failed_) {
      AdvanceUntil([&] { return !actual_.is_pending(actual_.self); },
                   LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING);
    }
    if (service_) {
      actual_.set_observer(actual_.self, {});
    }
    loom_serve_speech_service_free(service_);
    // Failed readiness transfers request ownership to the model. Release its
    // views, join actual queue retirement, then release borrowed host bytes.
    for (auto& invocation : invocations_) {
      if (invocation) {
        iree_hal_buffer_release(invocation->buffer);
        invocation->buffer = nullptr;
      }
    }
    if (execution_) {
      CheckStatus(loom_serve_execution_drain(execution_));
      loom_serve_execution_release(execution_);
    }
    for (auto& invocation : invocations_) {
      invocation.reset();
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    iree_hal_semaphore_release(gate_);
  }

  iree_status_t Send(loom_serve::testing::HttpClient* client,
                     uint64_t blocks = 0, const char* voice = "positive") {
    const std::string body = std::string(R"({"model":"speech","input":")") +
                             std::string(blocks ? blocks : blocks_, 'a') +
                             R"(","voice":")" + voice +
                             R"(","response_format":"pcm"})";
    return http_.Send(
        "POST /v1/audio/speech HTTP/1.1\r\nHost: local\r\n"
        "Content-Type: application/json\r\nContent-Length: " +
            std::to_string(body.size()) + "\r\n\r\n" + body,
        client->put());
  }

  void CheckHealth() {
    loom_serve::testing::HttpClient health;
    IREE_ASSERT_OK(http_.Send("GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n",
                              health.put()));
    std::string response;
    IREE_ASSERT_OK(http_.Read(health.release(), false, &response));
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
  }

  static void CheckPcm(const std::string& response, size_t frames,
                       int sign = 1) {
    ASSERT_EQ(response.find("HTTP/1.1 200"), 0u);
    const auto begin = response.find("\r\n\r\n");
    ASSERT_NE(begin, std::string::npos);
    ASSERT_EQ(response.size() - begin - 4, frames * 2);
    for (size_t i = begin + 4; i < response.size(); i += 2) {
      ASSERT_EQ(iree_unaligned_load_le_u16(response.data() + i),
                static_cast<uint16_t>(sign * 16384));
    }
  }

  // Model and service use the same application resource owners.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Borrowed process task/proactor owner; one worker detects blocking waits.
  loom_serve_host_t* host_ = nullptr;
  // Borrowed process device group enclosing all execution allocations.
  loom_serve_device_group_t* group_ = nullptr;
  // Actual queue dependencies and tracked request views.
  loom_serve_execution_t* execution_ = nullptr;
  // Explicit device dependency held by the test's outer client thread.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Outer-thread gate signal/failure has been published once.
  bool gate_released_ = false;
  // Expected terminal device failure, read only by the outer test owner.
  bool failed_ = false;
  // Frames in every deterministic feedback block.
  size_t frames_ = 0;
  // Terminal block count for each request.
  uint64_t blocks_ = 0;
  // Per-request block index that imports gate_ before accepting transfers.
  uint64_t held_block_ = 0;
  // Minimum cohort size importing the held dependency.
  size_t held_members_ = 1;
  // Model-owned rows; terminal failures leave them here for actual teardown.
  std::array<std::unique_ptr<Invocation>, 4> invocations_;
  // Original selected ordinals survive cancellation and feedback collection.
  std::vector<Invocation*> batch_;
  // Accepted work per resident row, published without exposing source cursors.
  std::array<std::atomic<uint64_t>, 4> row_submitted_ = {};
  // Largest accepted cohort, independent of request completion order.
  std::atomic<size_t> largest_batch_{0};
  // A complete numerical cohort has accepted the held dependency.
  std::atomic<bool> held_{false};
  // Accepted generation count, published after both queues accept work.
  std::atomic<uint64_t> submitted_{0};
  // Successful release count, published by the serialized service owner.
  std::atomic<uint64_t> released_{0};
  // An owner turn observed exhausted carrier send credit.
  std::atomic<bool> credit_wait_{false};
  // HTTP owner outliving service, listener and client operations.
  loom_serve_http_server_t* server_ = nullptr;
  // Loopback listener borrowing the shared host network resources.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Production service under test, retained until router retirement.
  loom_serve_speech_service_t* service_ = nullptr;
  // Unmodified callbacks, forwarded by the observation wrapper.
  loom_serve_http_service_t actual_ = {};
  // Production operations with observational forwarding on owner turns.
  loom_serve_http_service_t wrapped_ = {};
  // Client readiness and the outer router thread.
  loom_serve::testing::HttpHarness http_;
};

TEST_F(SpeechReadinessTest,
       HeldDeviceKeepsHealthLiveAndResetRetiresBeforeReuse) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(Send(&client));
  IREE_ASSERT_OK(http_.Await(-1, [&] { return submitted_.load() == 2; }));
  IREE_ASSERT_OK(http_.AwaitResponse(client.get()));
  ASSERT_NO_FATAL_FAILURE(CheckHealth());
  EXPECT_EQ(released_.load(), 0u);
  const linger reset = {1, 0};
  ASSERT_EQ(
      setsockopt(client.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
      0);
  ASSERT_EQ(close(client.release()), 0);
  ReleaseDependency();
  IREE_ASSERT_OK(http_.Await(-1, [&] { return released_.load() == 1; }));
  IREE_ASSERT_OK(Send(&client));
  std::string response;
  IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
  CheckPcm(response, 12);
  EXPECT_EQ(released_.load(), 2u);
}

TEST_F(SpeechReadinessTest, ShutdownJoinsAcceptedWorkWithoutFurtherSynthesis) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(Send(&client));
  IREE_ASSERT_OK(http_.Await(-1, [&] { return submitted_.load() == 2; }));
  loom_serve_http_server_request_stop(server_);
  EXPECT_EQ(released_.load(), 0u);
  ReleaseDependency();
  IREE_ASSERT_OK(http_.Join());
  EXPECT_EQ(submitted_.load(), 2u);
  EXPECT_EQ(released_.load(), 1u);
}

TEST_F(SpeechReadinessTest, FailedReadinessKeepsRequestForModelRetirement) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(Send(&client));
  IREE_ASSERT_OK(http_.Await(-1, [&] { return submitted_.load() == 2; }));
  ASSERT_NO_FATAL_FAILURE(CheckHealth());
  ReleaseDependency(true);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, http_.Join());
  EXPECT_EQ(submitted_.load(), 2u);
  EXPECT_EQ(released_.load(), 0u);
  EXPECT_NE(invocations_[0], nullptr);
}

TEST_F(SpeechReadinessTest, SlowReaderResumesFromBoundedSendCredit) {
  ASSERT_NO_FATAL_FAILURE(Initialize(65536, 128, UINT64_MAX));
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(Send(&client));
  IREE_ASSERT_OK(http_.Await(
      -1, [&] { return credit_wait_.load(std::memory_order_acquire); }));
  ASSERT_NO_FATAL_FAILURE(CheckHealth());
  EXPECT_LT(submitted_.load(), 128u);
  std::string response;
  IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
  CheckPcm(response, 65536 * 128);
  EXPECT_EQ(submitted_.load(), 128u);
  EXPECT_EQ(released_.load(), 1u);
}

TEST_F(SpeechReadinessTest, CancelledBatchMemberRetiresWithOriginalCohort) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 16, 1, 4, 4, false));
  std::array<loom_serve::testing::HttpClient, 4> clients;
  std::array<loom_serve_http_connection_t*, 4> connections = {};
  for (size_t i = 0; i < clients.size(); ++i) {
    IREE_ASSERT_OK(Send(&clients[i], 16 + i, i % 2 ? "negative" : "positive"));
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(
        http_.TakeRequest(clients[i].get(), &connections[i], &request));
    IREE_ASSERT_OK(actual_.accept(actual_.self, connections[i], request, 0));
  }
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return held_.load(std::memory_order_acquire); }));
  EXPECT_EQ(largest_batch_.load(), 4u);
  const auto submitted = submitted_.load();
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
  bool progress = false;
  IREE_ASSERT_OK(actual_.advance(actual_.self, 0, &progress));
  EXPECT_TRUE(progress);
  EXPECT_EQ(released_.load(), 0u);
  EXPECT_EQ(actual_.snapshot(actual_.self).active_requests, 4u);
  EXPECT_EQ(submitted_.load(), submitted);
  ReleaseDependency();
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return actual_.telemetry.counters->cancelled == 1; }));
  EXPECT_EQ(released_.load(), 1u);
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return actual_.telemetry.counters->completed == 3; }));
  for (size_t i : {0u, 2u, 3u}) {
    std::string response;
    IREE_ASSERT_OK(http_.Read(clients[i].release(), false, &response));
    CheckPcm(response, 3 * (16 + i), i % 2 ? -1 : 1);
  }
  EXPECT_EQ(released_.load(), 4u);
  EXPECT_FALSE(actual_.is_pending(actual_.self));
}

TEST_F(SpeechReadinessTest, StalledReaderDoesNotBlockOtherRequests) {
  ASSERT_NO_FATAL_FAILURE(Initialize(65536, 128, UINT64_MAX, 4));
  loom_serve::testing::HttpClient stalled;
  IREE_ASSERT_OK(Send(&stalled));
  IREE_ASSERT_OK(http_.Await(
      -1, [&] { return credit_wait_.load(std::memory_order_acquire); }));
  std::array<loom_serve::testing::HttpClient, 3> peers;
  for (size_t i = 0; i < peers.size(); ++i) {
    IREE_ASSERT_OK(Send(&peers[i], i + 1, i % 2 ? "positive" : "negative"));
  }
  for (size_t i = 0; i < peers.size(); ++i) {
    std::string response;
    IREE_ASSERT_OK(http_.Read(peers[i].release(), false, &response));
    CheckPcm(response, 65536 * (i + 1), i % 2 ? 1 : -1);
  }
  EXPECT_EQ(released_.load(), 3u);
  EXPECT_LT(row_submitted_[0].load(), 128u);
  ASSERT_NO_FATAL_FAILURE(CheckHealth());
  std::string response;
  IREE_ASSERT_OK(http_.Read(stalled.release(), false, &response));
  CheckPcm(response, 65536 * 128);
  EXPECT_EQ(row_submitted_[0].load(), 128u);
  EXPECT_EQ(released_.load(), 4u);
}

INSTANTIATE_TEST_SUITE_P(Parameters, SourceSpeechServiceTest,
                         ::testing::Bool());

}  // namespace
