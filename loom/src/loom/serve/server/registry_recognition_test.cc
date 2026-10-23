// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/registry.h"
#include "loom/serve/server/resources.h"
#include "loom/serve/service/recognition/stream.h"
#include "loom/serve/service/recognition/transcription.h"
#include "loom/serve/telemetry/record.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/serve/transport/http/server.h"

IREE_FLAG(string, model, "", "Recognition model document for qualification.");
IREE_FLAG(string, input, "", "Optional 16-kHz mono PCM16LE recording.");
IREE_FLAG(string, http_input, "",
          "Optional same-duration PCM16LE recording for the HTTP peer.");
IREE_FLAG(int32_t, http_input_rate, 24000,
          "Sample rate of the optional HTTP peer recording.");
IREE_FLAG(string, expected_text, "E", "Expected final recognition text.");
IREE_FLAG(int32_t, test_workers, 1, "Shared application worker count.");

namespace {

struct loom_serve_recognition_test_input_t {
  // Caller ownership, independent of the managed model's lifetime.
  std::unique_ptr<loom_serve_recognition_stream_t,
                  decltype(&loom_serve_recognition_stream_close)>
      stream{nullptr, loom_serve_recognition_stream_close};
  // Input frames already accepted by this independent stream.
  size_t position = 0;
  // Explicit EOF has reached the stream, independently of source completion.
  bool finished = false;
};

class RegistryRecognitionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (FLAG_input[0]) {
      iree_io_file_contents_t* contents = nullptr;
      IREE_ASSERT_OK(iree_io_file_contents_read(
          iree_make_cstring_view(FLAG_input), allocator_, &contents));
      pcm_.assign(reinterpret_cast<const char*>(contents->buffer.data),
                  contents->buffer.data_length);
      iree_io_file_contents_free(contents);
    } else {
      pcm_.resize(18 * sizeof(int16_t));
      for (size_t i = 0; i < 18; ++i) {
        iree_unaligned_store_le_u16(pcm_.data() + i * sizeof(int16_t), 8192);
      }
    }
    ASSERT_FALSE(pcm_.empty());
    ASSERT_EQ(pcm_.size() % sizeof(int16_t), 0u);
    if (FLAG_http_input[0]) {
      ASSERT_GT(FLAG_http_input_rate, 0);
      iree_io_file_contents_t* contents = nullptr;
      IREE_ASSERT_OK(iree_io_file_contents_read(
          iree_make_cstring_view(FLAG_http_input), allocator_, &contents));
      http_pcm_.assign(reinterpret_cast<const char*>(contents->buffer.data),
                       contents->buffer.data_length);
      iree_io_file_contents_free(contents);
      http_input_rate_ = FLAG_http_input_rate;
    } else if (FLAG_input[0]) {
      http_pcm_ = pcm_;
      http_input_rate_ = 16000;
    } else {
      // Independently authored 24-kHz samples cover the same duration as the
      // native fixture. HTTP must convert them before the source sees frames.
      http_pcm_.resize(27 * sizeof(int16_t));
      for (size_t i = 0; i < 27; ++i) {
        iree_unaligned_store_le_u16(http_pcm_.data() + i * sizeof(int16_t),
                                    8192);
      }
      http_input_rate_ = 24000;
    }
    ASSERT_FALSE(http_pcm_.empty());
    ASSERT_EQ(http_pcm_.size() % sizeof(int16_t), 0u);
    ASSERT_EQ(
        (http_pcm_.size() / sizeof(int16_t) * 16000ull + http_input_rate_ - 1) /
            http_input_rate_,
        pcm_.size() / sizeof(int16_t));
    samples_.resize(pcm_.size() / sizeof(int16_t) * sizeof(float));
    for (size_t i = 0; i < pcm_.size() / sizeof(int16_t); ++i) {
      const int16_t sample =
          static_cast<int16_t>(iree_unaligned_load_le_u16(pcm_.data() + i * 2));
      iree_unaligned_store_le_f32(samples_.data() + i * sizeof(float),
                                  static_cast<float>(sample) / 32768.0f);
    }
    auto options = loom_serve_test_application_options_default();
    options.worker_count = FLAG_test_workers;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(&options, allocator_,
                                                       &application_));
    registration_options_.device_group =
        loom_serve_test_application_device_group(application_);
    const auto transport = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        loom_serve_host_proactor_pool(
            loom_serve_test_application_host(application_)),
        &transport, allocator_, &server_));
    auto listener = loom_serve_http_listener_options_default();
    listener.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(server_, &listener,
                                                     allocator_, &listener_));
    IREE_ASSERT_OK(http_.Initialize(server_, listener_));
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        0, nullptr, 1,
        loom_serve_model_registration_factory(&registration_options_),
        allocator_, &registry_));
    ASSERT_NO_FATAL_FAILURE(Register());
  }

  void TearDown() override {
    if (registry_ && loom_serve_registry_count(registry_)) {
      IREE_EXPECT_OK(
          loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_ABORT));
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; });
    }
    IREE_EXPECT_OK(loom_serve_registry_free(registry_));
    if (server_) {
      loom_serve_http_server_request_stop(server_);
    }
    if (reader_.joinable()) {
      reader_.join();
    }
    iree_status_free(read_status_);
    const loom_serve_resources_t resources = {
        .device_group = registration_options_.device_group,
    };
    loom_serve_resources_set_observer(&resources, {});
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
  }

  void Register() {
    loom_serve_model_configuration_t configuration;
    IREE_ASSERT_OK(loom_serve_model_configuration_load(
        iree_make_cstring_view(FLAG_model), iree_string_view_empty(),
        allocator_, &configuration));
    Configure(&configuration);
    name_.assign(configuration.name.data, configuration.name.size);
    configuration.start_drained = true;
    auto status =
        loom_serve_registry_register(registry_, &configuration, &index_);
    loom_serve_model_configuration_deinitialize(&configuration);
    IREE_ASSERT_OK(status);
  }

  virtual void Configure(loom_serve_model_configuration_t* configuration) {}

  loom_serve_registered_model_t* Model() {
    return loom_serve_registry_at(registry_, index_);
  }

  template <typename Predicate>
  void AdvanceUntil(Predicate ready) {
    auto* notification = loom_serve_http_server_notification(server_);
    const loom_serve_wake_callback_t wake = {
        [](void* self) {
          iree_notification_post(static_cast<iree_notification_t*>(self),
                                 IREE_ALL_WAITERS);
        },
        notification};
    const loom_serve_resources_t resources = {
        .device_group = registration_options_.device_group,
    };
    loom_serve_resources_set_observer(&resources, wake);
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      iree_status_t status = loom_serve_resources_advance(&resources);
      if (Model()->phase == LOOM_SERVE_REGISTRATION_READY) {
        auto& service = Model()->service;
        status = iree_status_join(
            status, service.advance(service.self,
                                    loom_serve_registry_service_flags(Model()),
                                    &progress));
      }
      bool registry_progress = false;
      if (iree_status_is_ok(status)) {
        status = loom_serve_registry_advance(registry_, index_, wake,
                                             &registry_progress);
      }
      status = iree_status_join(status, iree_status_clone(Model()->status));
      if (ready() || progress || registry_progress ||
          !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
    }
  }

  void Prepare() {
    Model()->closed = false;
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        [&] { return Model()->phase == LOOM_SERVE_REGISTRATION_READY; }));
  }

  iree_status_t Open(loom_serve_recognition_test_input_t* input,
                     loom_serve_recognition_output_flags_t outputs =
                         LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT) {
    const loom_serve_recognition_request_t request = {
        .outputs = outputs,
    };
    loom_serve_recognition_stream_t* stream = nullptr;
    iree_status_t status = loom_serve_registry_try_open_recognition(
        registry_, iree_make_cstring_view(name_.c_str()), &request, &stream);
    input->stream.reset(stream);
    return status;
  }

  void Feed(loom_serve_recognition_test_input_t* input) {
    auto* stream = input->stream.get();
    const auto format = loom_serve_recognition_stream_format(stream);
    ASSERT_EQ(format.sample_rate, 16000u);
    ASSERT_EQ(format.channels, 1u);
    const size_t count =
        std::min(samples_.size() / sizeof(float) - input->position,
                 loom_serve_recognition_stream_credit(stream));
    if (count) {
      iree_host_size_t accepted = 0;
      IREE_ASSERT_OK(loom_serve_recognition_stream_write(
          stream,
          iree_make_const_byte_span(
              samples_.data() + input->position * sizeof(float),
              count * sizeof(float)),
          &accepted));
      EXPECT_EQ(accepted, count);
      input->position += accepted;
    }
    if (input->position == samples_.size() / sizeof(float) &&
        !input->finished) {
      loom_serve_recognition_stream_finish(stream);
      input->finished = true;
    }
  }

  void AwaitRevision(loom_serve_recognition_test_input_t* input) {
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
      Feed(input);
      return HasFatalFailure() ||
             loom_serve_recognition_stream_peek(input->stream.get()) ||
             loom_serve_recognition_stream_is_complete(input->stream.get());
    }));
    IREE_ASSERT_OK(loom_serve_recognition_stream_status(input->stream.get()));
    ASSERT_NE(loom_serve_recognition_stream_peek(input->stream.get()), nullptr);
  }

  void ReadFinal(loom_serve_recognition_test_input_t* input,
                 std::string* text) {
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
      auto* stream = input->stream.get();
      while (loom_serve_recognition_stream_peek(stream)) {
        loom_serve_recognition_stream_consume(stream);
      }
      Feed(input);
      return HasFatalFailure() ||
             loom_serve_recognition_stream_is_complete(stream);
    }));
    IREE_ASSERT_OK(loom_serve_recognition_stream_status(input->stream.get()));
    ASSERT_TRUE(input->finished);
    const auto* document =
        loom_serve_recognition_stream_document(input->stream.get());
    ASSERT_NE(document, nullptr);
    ASSERT_GT(document->count, 0u);
    iree_string_builder_t output;
    iree_string_builder_initialize(allocator_, &output);
    auto status = loom_serve_transcription_format(document, true, &output);
    if (iree_status_is_ok(status)) {
      text->assign(iree_string_builder_buffer(&output),
                   iree_string_builder_size(&output));
    }
    iree_string_builder_deinitialize(&output);
    IREE_ASSERT_OK(status);
    EXPECT_EQ(*text, FLAG_expected_text);
  }

  void StartHttp() {
    const std::string body =
        "--voice\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n" +
        name_ +
        "\r\n--voice\r\nContent-Disposition: form-data; "
        "name=\"response_format\"\r\n\r\ntext\r\n--voice\r\n"
        "Content-Disposition: form-data; name=\"file\"; "
        "filename=\"voice.pcm\"\r\n"
        "Content-Type: audio/pcm;rate=" +
        std::to_string(http_input_rate_) + ";channels=1\r\n\r\n" + http_pcm_ +
        "\r\n--voice--\r\n";
    loom_serve::testing::HttpClient client;
    IREE_ASSERT_OK(
        http_.Send("POST /v1/audio/transcriptions HTTP/1.1\r\nHost: local\r\n"
                   "Content-Type: multipart/form-data; "
                   "boundary=voice\r\nContent-Length: " +
                       std::to_string(body.size()) + "\r\n\r\n" + body,
                   client.put()));
    loom_serve_http_connection_t* connection = nullptr;
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(http_.TakeRequest(client.get(), &connection, &request));
    IREE_ASSERT_OK(
        Model()->service.accept(Model()->service.self, connection, request, 0));
    reader_ = std::thread([this, descriptor = client.release()] {
      read_status_ = http_.Read(descriptor, false, &response_);
    });
  }

  // Process-owned shared application, borrowed through registry retirement.
  loom_serve_test_application_t* application_ = nullptr;
  // Allocator for registry metadata and caller-owned input/output.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Physical model factory borrowed through registry destruction.
  loom_serve_model_registration_options_t registration_options_ = {};
  // Managed model slot with production preparation, admission, and retirement.
  loom_serve_registry_t* registry_ = nullptr;
  // Slot index independent of the registration incarnation.
  iree_host_size_t index_ = 0;
  // Copied model name, retained across removal and reuse.
  std::string name_;
  // Original 16-kHz PCM16LE recording for the native reference.
  std::string pcm_;
  // Exact F32LE conversion of that same recording for native input.
  std::vector<uint8_t> samples_;
  // Independent PCM16LE recording sent through the finite upload converter.
  std::string http_pcm_;
  // Declared sample clock of the HTTP recording, independent of model geometry.
  uint32_t http_input_rate_ = 0;
  // Shared network proactor owner, outliving the registry and HTTP peer.
  loom_serve_http_server_t* server_ = nullptr;
  // Ephemeral loopback endpoint for the real transport peer.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Client transport with no duplicate service owner or execution scheduler.
  loom_serve::testing::HttpHarness http_;
  // Outer reader drains the socket while this owner advances model work.
  std::thread reader_;
  // Reader-published status, accessed only after joining.
  iree_status_t read_status_ = iree_ok_status();
  // Reader-published HTTP response, accessed only after joining.
  std::string response_;
};

TEST_F(RegistryRecognitionTest, NativeDrainAndHttpPeerPreserveFinalDocument) {
  loom_serve_recognition_test_input_t input;
  IREE_ASSERT_OK(Open(&input));
  EXPECT_EQ(input.stream, nullptr);
  EXPECT_EQ(Model()->phase, LOOM_SERVE_REGISTRATION_DECLARED);
  ASSERT_NO_FATAL_FAILURE(Prepare());
  IREE_ASSERT_OK(Open(&input));
  ASSERT_NE(input.stream, nullptr);
  EXPECT_EQ(loom_serve_registry_snapshot(Model()).active_requests, 1u);
  ASSERT_NO_FATAL_FAILURE(AwaitRevision(&input));
  const auto* first = loom_serve_recognition_stream_peek(input.stream.get());
  ASSERT_FALSE(first->transcript.flags &
               LOOM_SERVE_RECOGNITION_TRANSCRIPT_FINAL);
  const std::string held(first->transcript.text.data,
                         first->transcript.text.size);
  ASSERT_NO_FATAL_FAILURE(StartHttp());
  EXPECT_EQ(loom_serve_registry_snapshot(Model()).queued_requests, 1u);
  EXPECT_EQ(loom_serve_registry_snapshot(Model()).active_requests, 1u);
  loom_serve_recognition_test_input_t waiting;
  IREE_ASSERT_OK(Open(&waiting));
  EXPECT_EQ(waiting.stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    const auto* counters = Model()->service.telemetry.counters;
    return counters->completed || counters->rejected || counters->failed ||
           counters->cancelled;
  }));
  ASSERT_EQ(Model()->service.telemetry.counters->completed, 1u);
  EXPECT_EQ(Model()->service.telemetry.counters->requests, 2u);
  reader_.join();
  IREE_ASSERT_OK(iree_status_clone(read_status_));
  ASSERT_EQ(response_.find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(loom_serve_recognition_stream_peek(input.stream.get()), first);
  EXPECT_EQ(
      std::string(first->transcript.text.data, first->transcript.text.size),
      held);
  const uint64_t incarnation = Model()->incarnation;
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_DRAIN));
  EXPECT_EQ(loom_serve_registry_snapshot(Model()).active_requests, 1u);
  EXPECT_EQ(loom_serve_registry_service_flags(Model()), 0u);
  loom_serve_recognition_test_input_t rejected;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Open(&rejected));
  std::string text;
  ASSERT_NO_FATAL_FAILURE(ReadFinal(&input, &text));
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; }));
  loom_serve_telemetry_counters_t counters;
  ASSERT_TRUE(loom_serve_registry_counters(registry_, &counters));
  EXPECT_EQ(counters.requests, 2u);
  EXPECT_EQ(counters.completed, 2u);
  const auto body = response_.find("\r\n\r\n");
  ASSERT_NE(body, std::string::npos);
  EXPECT_EQ(response_.substr(body + 4), text);
  const auto* final =
      loom_serve_recognition_stream_document(input.stream.get());
  ASSERT_NE(final, nullptr);
  const auto* segment = loom_serve_recognition_document_at(final, 0);
  const std::string segment_text(segment->text.data, segment->text.size);

  ASSERT_NO_FATAL_FAILURE(Register());
  EXPECT_GT(Model()->incarnation, incarnation);
  ASSERT_NO_FATAL_FAILURE(Prepare());
  loom_serve_recognition_test_input_t replacement;
  IREE_ASSERT_OK(Open(&replacement));
  ASSERT_NE(replacement.stream, nullptr);
  std::string replacement_text;
  ASSERT_NO_FATAL_FAILURE(ReadFinal(&replacement, &replacement_text));
  EXPECT_EQ(replacement_text, text);
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_DRAIN));
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; }));
  IREE_ASSERT_OK(loom_serve_registry_free(registry_));
  registry_ = nullptr;
  EXPECT_EQ(loom_serve_recognition_stream_document(input.stream.get()), final);
  EXPECT_EQ(std::string(segment->text.data, segment->text.size), segment_text);
  IREE_EXPECT_OK(loom_serve_recognition_stream_status(input.stream.get()));
}

TEST_F(RegistryRecognitionTest, AbortRetiresNativeLedgerAndRetainsStatus) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  loom_serve_recognition_test_input_t first;
  loom_serve_recognition_test_input_t second;
  IREE_ASSERT_OK(Open(&first));
  ASSERT_NE(first.stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(AwaitRevision(&first));
  IREE_ASSERT_OK(Open(&second));
  ASSERT_NE(second.stream, nullptr);
  EXPECT_EQ(loom_serve_registry_snapshot(Model()).active_requests, 2u);
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_ABORT));
  second.stream.reset();
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; }));
  loom_serve_telemetry_counters_t counters;
  ASSERT_TRUE(loom_serve_registry_counters(registry_, &counters));
  EXPECT_EQ(counters.requests, 2u);
  EXPECT_EQ(counters.cancelled, 2u);
  IREE_ASSERT_OK(loom_serve_registry_free(registry_));
  registry_ = nullptr;
  EXPECT_TRUE(loom_serve_recognition_stream_is_complete(first.stream.get()));
  EXPECT_EQ(loom_serve_recognition_stream_peek(first.stream.get()), nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      loom_serve_recognition_stream_status(first.stream.get()));
}

TEST_F(RegistryRecognitionTest, InvalidOptionsLeaveAdmissionAvailable) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  const loom_serve_recognition_request_t request = {
      .temperature = -1.0f,
      .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
  };
  loom_serve_recognition_stream_t* stream = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_registry_try_open_recognition(
          registry_, iree_make_cstring_view(name_.c_str()), &request, &stream));
  EXPECT_EQ(stream, nullptr);
  EXPECT_EQ(loom_serve_registry_snapshot(Model()).queued_requests, 0u);
  EXPECT_EQ(loom_serve_registry_snapshot(Model()).active_requests, 0u);
  EXPECT_EQ(Model()->service.telemetry.counters->requests, 0u);
  loom_serve_recognition_test_input_t input;
  IREE_ASSERT_OK(Open(&input));
  ASSERT_NE(input.stream, nullptr);
  std::string text;
  ASSERT_NO_FATAL_FAILURE(ReadFinal(&input, &text));
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_DRAIN));
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; }));
  loom_serve_telemetry_counters_t counters;
  ASSERT_TRUE(loom_serve_registry_counters(registry_, &counters));
  EXPECT_EQ(counters.requests, 1u);
  EXPECT_EQ(counters.completed, 1u);
  EXPECT_EQ(counters.rejected, 0u);
}

// The compound source has both output families. Its deployment role controls
// HTTP exposure independently of each native request's output selection.
class RegistryActivityTest : public RegistryRecognitionTest {
 protected:
  void Configure(loom_serve_model_configuration_t* configuration) override {
    configuration->kind = LOOM_SERVE_MODEL_KIND_ACTIVITY;
  }
};

TEST_F(RegistryActivityTest, CompoundOutputSurvivesDrainAndReplacement) {
  EXPECT_EQ(Model()->service.kind, LOOM_SERVE_MODEL_KIND_ACTIVITY);
  EXPECT_EQ(Model()->service.endpoint, nullptr);
  EXPECT_EQ(Model()->service.accept, nullptr);
  EXPECT_EQ(Model()->service.capabilities.output_modalities.count, 0u);
  loom_serve_recognition_test_input_t input;
  IREE_ASSERT_OK(Open(&input, LOOM_SERVE_RECOGNITION_OUTPUT_ALL));
  EXPECT_EQ(input.stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(Prepare());
  EXPECT_EQ(Model()->service.kind, LOOM_SERVE_MODEL_KIND_ACTIVITY);
  EXPECT_EQ(Model()->service.endpoint, nullptr);
  EXPECT_EQ(Model()->service.accept, nullptr);
  const auto outputs = Model()->service.capabilities.output_modalities;
  ASSERT_EQ(outputs.count, 2u);
  EXPECT_EQ(std::string(outputs.values[0].data, outputs.values[0].size),
            "text");
  EXPECT_EQ(std::string(outputs.values[1].data, outputs.values[1].size),
            "activity");
  IREE_ASSERT_OK(Open(&input, LOOM_SERVE_RECOGNITION_OUTPUT_ALL));
  ASSERT_NE(input.stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(AwaitRevision(&input));
  const auto* held = loom_serve_recognition_stream_peek(input.stream.get());
  ASSERT_EQ(held->kind, LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY);
  const auto activity = held->activity;
  EXPECT_LT(activity.begin, activity.end);
  EXPECT_FLOAT_EQ(activity.speech_probability, 0.75f);

  loom_serve_recognition_test_input_t peer;
  IREE_ASSERT_OK(Open(&peer));
  ASSERT_NE(peer.stream, nullptr);
  std::string peer_text;
  ASSERT_NO_FATAL_FAILURE(ReadFinal(&peer, &peer_text));
  EXPECT_EQ(loom_serve_recognition_stream_peek(input.stream.get()), held);
  EXPECT_EQ(held->activity.begin, activity.begin);
  EXPECT_EQ(held->activity.end, activity.end);
  EXPECT_FLOAT_EQ(held->activity.speech_probability,
                  activity.speech_probability);

  const uint64_t incarnation = Model()->incarnation;
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_DRAIN));
  loom_serve_recognition_test_input_t rejected;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      Open(&rejected, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    auto* stream = input.stream.get();
    while (const auto* event = loom_serve_recognition_stream_peek(stream)) {
      if (event->kind == LOOM_SERVE_RECOGNITION_EVENT_TRANSCRIPT &&
          (event->transcript.flags & LOOM_SERVE_RECOGNITION_TRANSCRIPT_FINAL)) {
        return true;
      }
      loom_serve_recognition_stream_consume(stream);
    }
    Feed(&input);
    return HasFatalFailure() ||
           loom_serve_recognition_stream_is_complete(stream);
  }));
  const auto* final = loom_serve_recognition_stream_peek(input.stream.get());
  ASSERT_NE(final, nullptr);
  ASSERT_EQ(final->kind, LOOM_SERVE_RECOGNITION_EVENT_TRANSCRIPT);
  ASSERT_TRUE(final->transcript.flags &
              LOOM_SERVE_RECOGNITION_TRANSCRIPT_FINAL);
  EXPECT_EQ(
      std::string(final->transcript.text.data, final->transcript.text.size),
      peer_text);
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; }));
  ASSERT_NO_FATAL_FAILURE(Register());
  EXPECT_GT(Model()->incarnation, incarnation);
  ASSERT_NO_FATAL_FAILURE(Prepare());
  loom_serve_recognition_test_input_t replacement;
  IREE_ASSERT_OK(Open(&replacement));
  ASSERT_NE(replacement.stream, nullptr);
  std::string replacement_text;
  ASSERT_NO_FATAL_FAILURE(ReadFinal(&replacement, &replacement_text));
  EXPECT_EQ(replacement_text, peer_text);
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_DRAIN));
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; }));
  EXPECT_EQ(loom_serve_recognition_stream_peek(input.stream.get()), final);
  EXPECT_EQ(
      std::string(final->transcript.text.data, final->transcript.text.size),
      peer_text);
  loom_serve_recognition_stream_consume(input.stream.get());
  EXPECT_TRUE(loom_serve_recognition_stream_is_complete(input.stream.get()));
  IREE_EXPECT_OK(loom_serve_recognition_stream_status(input.stream.get()));
}

TEST_F(RegistryActivityTest, AbortRetiresActivityPeersAndRetainsCallerStatus) {
  ASSERT_NO_FATAL_FAILURE(Prepare());
  loom_serve_recognition_test_input_t first;
  loom_serve_recognition_test_input_t second;
  IREE_ASSERT_OK(Open(&first, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY));
  ASSERT_NE(first.stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(AwaitRevision(&first));
  ASSERT_EQ(loom_serve_recognition_stream_peek(first.stream.get())->kind,
            LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY);
  EXPECT_EQ(loom_serve_recognition_stream_document(first.stream.get()),
            nullptr);
  IREE_ASSERT_OK(Open(&second, LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY));
  ASSERT_NE(second.stream, nullptr);
  Feed(&second);
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_ABORT));
  second.stream.reset();
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; }));
  loom_serve_telemetry_counters_t counters;
  ASSERT_TRUE(loom_serve_registry_counters(registry_, &counters));
  EXPECT_EQ(counters.requests, 2u);
  EXPECT_EQ(counters.cancelled, 2u);
  EXPECT_TRUE(loom_serve_recognition_stream_is_complete(first.stream.get()));
  EXPECT_EQ(loom_serve_recognition_stream_peek(first.stream.get()), nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      loom_serve_recognition_stream_status(first.stream.get()));
}

}  // namespace
