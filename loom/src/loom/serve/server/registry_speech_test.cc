// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/registry.h"
#include "loom/serve/service/service.h"
#include "loom/serve/service/speech/output.h"
#include "loom/serve/service/speech/request.h"
#include "loom/serve/service/speech/stream.h"
#include "loom/serve/telemetry/record.h"
#include "loom/serve/telemetry/stream.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/serve/transport/http/server.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

IREE_FLAG(string, model, "",
          "Speech model document for registry qualification.");
IREE_FLAG(string, voice, "positive", "Primary model voice selector.");
IREE_FLAG(string, alternate_voice, "negative", "Second model voice selector.");
IREE_FLAG(string, third_voice, "positive", "Third model voice selector.");
IREE_FLAG(string, fourth_voice, "negative", "Fourth model voice selector.");
IREE_FLAG(string, text, "Hello.", "Text shared by native and HTTP requests.");
IREE_FLAG(int32_t, test_workers, 1, "Shared application worker count.");

namespace {

using SpeechStream = std::unique_ptr<loom_serve_speech_stream_t,
                                     decltype(&loom_serve_speech_stream_close)>;

class RegistrySpeechTest : public ::testing::Test {
 protected:
  void SetUp() override {
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
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
  }

  void Register() {
    loom_serve_model_configuration_t configuration;
    IREE_ASSERT_OK(loom_serve_model_configuration_load(
        iree_make_cstring_view(FLAG_model), iree_string_view_empty(),
        allocator_, &configuration));
    name_.assign(configuration.name.data, configuration.name.size);
    configuration.start_drained = true;
    auto status =
        loom_serve_registry_register(registry_, &configuration, &index_);
    loom_serve_model_configuration_deinitialize(&configuration);
    IREE_ASSERT_OK(status);
    telemetry_cursor_ = 0;
    largest_batch_ = 0;
  }

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
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      iree_status_t status = iree_ok_status();
      if (Model()->phase == LOOM_SERVE_REGISTRATION_READY) {
        auto& service = Model()->service;
        status = service.advance(service.self,
                                 loom_serve_registry_service_flags(Model()),
                                 &progress);
        loom_serve_telemetry_record_t record;
        loom_serve_telemetry_read_t result;
        while ((result = loom_serve_telemetry_stream_read(
                    service.telemetry.stream, &telemetry_cursor_, &record)) ==
               LOOM_SERVE_TELEMETRY_READ_RECORD) {
          if (record.event == &LOOM_SERVE_TELEMETRY_BATCH_SUBMIT) {
            largest_batch_ =
                std::max(largest_batch_, record.data.batch.requests);
          }
        }
        EXPECT_EQ(result, LOOM_SERVE_TELEMETRY_READ_EMPTY);
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

  iree_status_t Open(const char* voice, SpeechStream* out_stream,
                     const char* text = FLAG_text, uint64_t seed = 73) {
    loom_serve_speech_request_t request;
    IREE_RETURN_IF_ERROR(loom_serve_speech_request_initialize_text(
        iree_make_cstring_view(text), iree_make_cstring_view(voice), {}, seed,
        1.0f, allocator_, &request));
    loom_serve_speech_stream_t* stream = nullptr;
    iree_status_t status = loom_serve_registry_try_open_speech(
        registry_, iree_make_cstring_view(name_.c_str()), &request, &stream);
    loom_serve_speech_request_deinitialize(&request);
    out_stream->reset(stream);
    return status;
  }

  void AppendPcm(const loom_serve_speech_block_t* block, std::string* output) {
    EXPECT_EQ(block->position, output->size() / sizeof(int16_t));
    std::vector<char> samples(block->frame_count * sizeof(int16_t) + 1);
    iree_string_builder_t pcm;
    iree_string_builder_initialize_with_storage(samples.data(), samples.size(),
                                                &pcm);
    IREE_ASSERT_OK(loom_serve_speech_encode_pcm16(block->samples, &pcm));
    output->append(iree_string_builder_buffer(&pcm),
                   iree_string_builder_size(&pcm));
  }

  void AwaitOutput(loom_serve_speech_stream_t* stream) {
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
      return loom_serve_speech_stream_peek(stream) ||
             loom_serve_speech_stream_is_complete(stream);
    }));
    IREE_ASSERT_OK(loom_serve_speech_stream_status(stream));
    ASSERT_NE(loom_serve_speech_stream_peek(stream), nullptr);
  }

  // Leaves the final block published so its lifetime can cross model removal.
  void ReadPcm(loom_serve_speech_stream_t* stream, std::string* output) {
    const auto format = loom_serve_speech_stream_format(stream);
    ASSERT_EQ(format.sample_rate, 24000u);
    ASSERT_EQ(format.channels, 1u);
    while (true) {
      ASSERT_NO_FATAL_FAILURE(AwaitOutput(stream));
      const auto* block = loom_serve_speech_stream_peek(stream);
      ASSERT_NO_FATAL_FAILURE(AppendPcm(block, output));
      if (block->complete) {
        break;
      }
      loom_serve_speech_stream_consume(stream);
    }
    EXPECT_FALSE(output->empty());
  }

  void StartHttp() {
    iree_string_builder_t json;
    iree_string_builder_initialize(allocator_, &json);
    loom_output_stream_t output;
    loom_output_stream_for_builder(&json, &output);
    IREE_ASSERT_OK(iree_string_builder_append_cstring(&json, "{\"model\":"));
    IREE_ASSERT_OK(loom_json_write_escaped_string(
        &output, iree_make_cstring_view(name_.c_str())));
    IREE_ASSERT_OK(iree_string_builder_append_cstring(&json, ",\"input\":"));
    IREE_ASSERT_OK(loom_json_write_escaped_string(
        &output, iree_make_cstring_view(FLAG_text)));
    IREE_ASSERT_OK(iree_string_builder_append_cstring(&json, ",\"voice\":"));
    IREE_ASSERT_OK(loom_json_write_escaped_string(
        &output, iree_make_cstring_view(FLAG_voice)));
    IREE_ASSERT_OK(iree_string_builder_append_cstring(
        &json, ",\"seed\":73,\"response_format\":\"pcm\"}"));
    const std::string body(iree_string_builder_buffer(&json),
                           iree_string_builder_size(&json));
    iree_string_builder_deinitialize(&json);
    loom_serve::testing::HttpClient client;
    IREE_ASSERT_OK(
        http_.Send("POST /v1/audio/speech HTTP/1.1\r\nHost: local\r\n"
                   "Content-Type: application/json\r\nContent-Length: " +
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

  // Process-owned host/device resources; all requests share this application.
  loom_serve_test_application_t* application_ = nullptr;
  // Allocator for registry metadata and test-owned request/response buffers.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Factory capabilities borrowed until registry destruction.
  loom_serve_model_registration_options_t registration_options_ = {};
  // One managed model slot, prepared and retired through the real adapter.
  loom_serve_registry_t* registry_ = nullptr;
  // Stable slot index, independent of the incarnation stored in that slot.
  iree_host_size_t index_ = 0;
  // Model document's name, copied before its declaration moves into the
  // registry.
  std::string name_;
  // Absolute read position in the current registration's telemetry stream.
  uint64_t telemetry_cursor_ = 0;
  // Largest accepted cohort observed since registration or explicit reset.
  uint64_t largest_batch_ = 0;
  // Shared asynchronous network owner, outliving its listener and requests.
  loom_serve_http_server_t* server_ = nullptr;
  // Actual loopback endpoint for the HTTP peer.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Client transport without an additional service/router owner.
  loom_serve::testing::HttpHarness http_;
  // Outer client reader drains HTTP while the test thread advances native work.
  std::thread reader_;
  // Reader-owned result, accessed only after joining its thread.
  iree_status_t read_status_ = iree_ok_status();
  // Reader-owned bytes, accessed only after joining its thread.
  std::string response_;
};

TEST_F(RegistrySpeechTest, SharedOutputAndFinalHandleSurviveNameReuse) {
  SpeechStream stream(nullptr, loom_serve_speech_stream_close);
  IREE_ASSERT_OK(Open(FLAG_voice, &stream));
  EXPECT_EQ(stream, nullptr);
  EXPECT_EQ(Model()->phase, LOOM_SERVE_REGISTRATION_DECLARED);
  Model()->closed = false;
  IREE_ASSERT_OK(Open(FLAG_voice, &stream));
  EXPECT_EQ(stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return Model()->phase == LOOM_SERVE_REGISTRATION_READY; }));
  IREE_ASSERT_OK(Open(FLAG_voice, &stream));
  ASSERT_NE(stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(AwaitOutput(stream.get()));
  const auto* first = loom_serve_speech_stream_peek(stream.get());
  ASSERT_FALSE(first->complete);
  const std::vector<uint8_t> held(
      first->samples.data, first->samples.data + first->samples.data_length);
  ASSERT_NO_FATAL_FAILURE(StartHttp());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    const auto* counters = Model()->service.telemetry.counters;
    return counters->completed || counters->rejected || counters->failed ||
           counters->cancelled;
  }));
  ASSERT_EQ(Model()->service.telemetry.counters->completed, 1u);
  reader_.join();
  IREE_ASSERT_OK(iree_status_clone(read_status_));
  ASSERT_EQ(response_.find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(
      std::vector<uint8_t>(first->samples.data,
                           first->samples.data + first->samples.data_length),
      held);
  std::string pcm;
  ASSERT_NO_FATAL_FAILURE(ReadPcm(stream.get(), &pcm));
  const auto body = response_.find("\r\n\r\n");
  ASSERT_NE(body, std::string::npos);
  EXPECT_EQ(response_.substr(body + 4), pcm);
  const auto* final = loom_serve_speech_stream_peek(stream.get());
  const std::vector<uint8_t> samples(
      final->samples.data, final->samples.data + final->samples.data_length);
  const uint64_t incarnation = Model()->incarnation;
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_DRAIN));
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; }));
  ASSERT_NO_FATAL_FAILURE(Register());
  EXPECT_GT(Model()->incarnation, incarnation);
  Model()->closed = false;
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return Model()->phase == LOOM_SERVE_REGISTRATION_READY; }));
  SpeechStream replacement(nullptr, loom_serve_speech_stream_close);
  IREE_ASSERT_OK(Open(FLAG_voice, &replacement));
  ASSERT_NE(replacement, nullptr);
  std::string replacement_pcm;
  ASSERT_NO_FATAL_FAILURE(ReadPcm(replacement.get(), &replacement_pcm));
  EXPECT_EQ(replacement_pcm, pcm);
  EXPECT_EQ(loom_serve_speech_stream_peek(stream.get()), final);
  EXPECT_EQ(
      std::vector<uint8_t>(final->samples.data,
                           final->samples.data + final->samples.data_length),
      samples);
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_ABORT));
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; }));
  IREE_ASSERT_OK(loom_serve_registry_free(registry_));
  registry_ = nullptr;
  EXPECT_EQ(loom_serve_speech_stream_peek(stream.get()), final);
  loom_serve_speech_stream_consume(stream.get());
  EXPECT_TRUE(loom_serve_speech_stream_is_complete(stream.get()));
  IREE_EXPECT_OK(loom_serve_speech_stream_status(stream.get()));
}

TEST_F(RegistrySpeechTest, AbortDetachesLiveStreamsBeforeModelRetirement) {
  Model()->closed = false;
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return Model()->phase == LOOM_SERVE_REGISTRATION_READY; }));
  SpeechStream first(nullptr, loom_serve_speech_stream_close);
  SpeechStream second(nullptr, loom_serve_speech_stream_close);
  IREE_ASSERT_OK(Open(FLAG_voice, &first));
  ASSERT_NE(first, nullptr);
  ASSERT_NO_FATAL_FAILURE(AwaitOutput(first.get()));
  IREE_ASSERT_OK(Open(FLAG_alternate_voice, &second));
  ASSERT_NE(second, nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return std::string(
               Model()->service.snapshot(Model()->service.self).phase) ==
               "generating" ||
           loom_serve_speech_stream_is_complete(second.get());
  }));
  IREE_ASSERT_OK(loom_serve_speech_stream_status(second.get()));
  ASSERT_STREQ(Model()->service.snapshot(Model()->service.self).phase,
               "generating");
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_ABORT));
  SpeechStream rejected(nullptr, loom_serve_speech_stream_close);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Open(FLAG_voice, &rejected));
  second.reset();
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return loom_serve_registry_count(registry_) == 0; }));
  EXPECT_TRUE(loom_serve_speech_stream_is_complete(first.get()));
  EXPECT_EQ(loom_serve_speech_stream_peek(first.get()), nullptr);
  IREE_ASSERT_OK(loom_serve_registry_free(registry_));
  registry_ = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_serve_speech_stream_status(first.get()));
}

TEST_F(RegistrySpeechTest,
       StaggeredStreamsPreserveOutputAcrossCancelledSlotReuse) {
  Model()->closed = false;
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return Model()->phase == LOOM_SERVE_REGISTRATION_READY; }));
  const std::array<const char*, 4> voices = {
      FLAG_voice, FLAG_alternate_voice, FLAG_third_voice, FLAG_fourth_voice};
  const std::string text = FLAG_text;
  const std::array<std::string, 4> texts = {
      text + " We can keep speaking while another voice finishes.", text,
      text + " Each voice keeps its own place.",
      text + " A new turn can join while the conversation is already running."};
  std::array<std::string, 4> expected;
  for (size_t i = 0; i < voices.size(); ++i) {
    SCOPED_TRACE(i);
    SpeechStream stream(nullptr, loom_serve_speech_stream_close);
    IREE_ASSERT_OK(Open(voices[i], &stream, texts[i].c_str(), 73 + i));
    ASSERT_NE(stream, nullptr);
    ASSERT_NO_FATAL_FAILURE(ReadPcm(stream.get(), &expected[i]));
    loom_serve_speech_stream_consume(stream.get());
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
        [&] { return loom_serve_speech_stream_is_complete(stream.get()); }));
    IREE_ASSERT_OK(loom_serve_speech_stream_status(stream.get()));
  }

  std::array<SpeechStream, 4> streams = {
      SpeechStream(nullptr, loom_serve_speech_stream_close),
      SpeechStream(nullptr, loom_serve_speech_stream_close),
      SpeechStream(nullptr, loom_serve_speech_stream_close),
      SpeechStream(nullptr, loom_serve_speech_stream_close)};
  IREE_ASSERT_OK(Open(voices[0], &streams[0], texts[0].c_str(), 73));
  ASSERT_NE(streams[0], nullptr);
  ASSERT_NO_FATAL_FAILURE(AwaitOutput(streams[0].get()));
  const auto* held_block = loom_serve_speech_stream_peek(streams[0].get());
  ASSERT_FALSE(held_block->complete);
  const std::vector<uint8_t> held_samples(
      held_block->samples.data,
      held_block->samples.data + held_block->samples.data_length);

  IREE_ASSERT_OK(Open(voices[1], &streams[1], texts[1].c_str(), 74));
  ASSERT_NE(streams[1], nullptr);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return std::string(
               Model()->service.snapshot(Model()->service.self).phase) ==
               "generating" ||
           loom_serve_speech_stream_is_complete(streams[1].get());
  }));
  IREE_ASSERT_OK(loom_serve_speech_stream_status(streams[1].get()));
  ASSERT_STREQ(Model()->service.snapshot(Model()->service.self).phase,
               "generating");
  for (size_t i = 2; i < streams.size(); ++i) {
    IREE_ASSERT_OK(Open(voices[i], &streams[i], texts[i].c_str(), 73 + i));
    ASSERT_NE(streams[i], nullptr);
  }
  for (auto& stream : streams) {
    ASSERT_NO_FATAL_FAILURE(AwaitOutput(stream.get()));
    ASSERT_FALSE(loom_serve_speech_stream_peek(stream.get())->complete);
  }
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return std::string(
               Model()->service.snapshot(Model()->service.self).phase) ==
           "delivering";
  }));
  ASSERT_EQ(Model()->service.snapshot(Model()->service.self).active_requests,
            4u);

  // With every peer holding output, only this request can submit new work.
  // Closing before another owner turn leaves its accepted borrower retained.
  loom_serve_speech_stream_consume(streams[2].get());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return std::string(
               Model()->service.snapshot(Model()->service.self).phase) ==
               "generating" ||
           loom_serve_speech_stream_is_complete(streams[2].get());
  }));
  IREE_ASSERT_OK(loom_serve_speech_stream_status(streams[2].get()));
  ASSERT_STREQ(Model()->service.snapshot(Model()->service.self).phase,
               "generating");
  streams[2].reset();
  EXPECT_EQ(Model()->service.snapshot(Model()->service.self).active_requests,
            4u);
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return Model()->service.telemetry.counters->cancelled == 1; }));
  ASSERT_EQ(Model()->service.snapshot(Model()->service.self).active_requests,
            3u);
  IREE_ASSERT_OK(Open(voices[2], &streams[2], texts[2].c_str(), 75));
  ASSERT_NE(streams[2], nullptr);
  ASSERT_NO_FATAL_FAILURE(AwaitOutput(streams[2].get()));
  ASSERT_FALSE(loom_serve_speech_stream_peek(streams[2].get())->complete);
  EXPECT_EQ(Model()->service.snapshot(Model()->service.self).active_requests,
            4u);

  // The short peer can finish while all other consumers withhold credit.
  std::array<std::string, 4> actual;
  ASSERT_NO_FATAL_FAILURE(ReadPcm(streams[1].get(), &actual[1]));
  loom_serve_speech_stream_consume(streams[1].get());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return loom_serve_speech_stream_is_complete(streams[1].get()); }));
  IREE_ASSERT_OK(loom_serve_speech_stream_status(streams[1].get()));
  EXPECT_EQ(actual[1], expected[1]);
  streams[1].reset();

  // Both ready voices now share execution while retaining independent output
  // positions and completion. No reader waits for a peer's next audio block.
  largest_batch_ = 0;
  while (streams[2] || streams[3]) {
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
      for (size_t i = 2; i < streams.size(); ++i) {
        if (streams[i] &&
            (loom_serve_speech_stream_peek(streams[i].get()) ||
             loom_serve_speech_stream_is_complete(streams[i].get()))) {
          return true;
        }
      }
      return false;
    }));
    for (size_t i = 2; i < streams.size(); ++i) {
      if (!streams[i]) {
        continue;
      }
      if (const auto* block = loom_serve_speech_stream_peek(streams[i].get())) {
        ASSERT_NO_FATAL_FAILURE(AppendPcm(block, &actual[i]));
        loom_serve_speech_stream_consume(streams[i].get());
      }
      if (loom_serve_speech_stream_is_complete(streams[i].get())) {
        IREE_ASSERT_OK(loom_serve_speech_stream_status(streams[i].get()));
        EXPECT_EQ(actual[i], expected[i]);
        streams[i].reset();
      }
    }
  }
  EXPECT_GE(largest_batch_, 2u);
  ASSERT_EQ(loom_serve_speech_stream_peek(streams[0].get()), held_block);
  EXPECT_EQ(std::vector<uint8_t>(
                held_block->samples.data,
                held_block->samples.data + held_block->samples.data_length),
            held_samples);
  ASSERT_NO_FATAL_FAILURE(ReadPcm(streams[0].get(), &actual[0]));
  loom_serve_speech_stream_consume(streams[0].get());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      [&] { return loom_serve_speech_stream_is_complete(streams[0].get()); }));
  IREE_ASSERT_OK(loom_serve_speech_stream_status(streams[0].get()));
  EXPECT_EQ(actual[0], expected[0]);
  EXPECT_EQ(Model()->service.telemetry.counters->completed, 8u);
  EXPECT_EQ(Model()->service.telemetry.counters->cancelled, 1u);
  EXPECT_EQ(Model()->service.telemetry.counters->failed, 0u);
  EXPECT_EQ(Model()->service.telemetry.counters->rejected, 0u);
}

}  // namespace
