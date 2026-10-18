// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/generation.h"

#include <array>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/base64.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/testing/worker_gate.h"

IREE_FLAG(string, text_model, "", "Text model document.");
IREE_FLAG(string, speech_model, "", "Speech model document.");
IREE_FLAG(int32_t, test_workers, 1, "Shared task worker count.");

namespace {

class RealtimeGenerationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    auto options = loom_serve_test_application_options_default();
    options.worker_count = FLAG_test_workers;
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &options, iree_allocator_system(), &application));
    host_ = loom_serve_test_application_host(application);
    registration_.device_group =
        loom_serve_test_application_device_group(application);
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        0, nullptr, 2, loom_serve_model_registration_factory(&registration_),
        iree_allocator_system(), &registry_));
    const std::string weights =
        iree::testing::MakeTempFilePath("realtime_text", ".safetensors");
    std::string header =
        R"({"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream file(weights, std::ios::binary);
    uint8_t bytes[8];
    iree_unaligned_store_le_u64(bytes, header.size());
    file.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    file.write(header.data(), header.size());
    iree_unaligned_store_le_u32(bytes, 2);
    file.write(reinterpret_cast<const char*>(bytes), 4);
    file.close();
    ASSERT_TRUE(file.good());
    const std::array<const char*, 2> paths = {FLAG_text_model,
                                              FLAG_speech_model};
    for (size_t i = 0; i < paths.size(); ++i) {
      loom_serve_model_configuration_t configuration;
      IREE_ASSERT_OK(loom_serve_model_configuration_load(
          iree_make_cstring_view(paths[i]), iree_allocator_system(),
          &configuration));
      iree_host_size_t index = 0;
      auto status = iree_ok_status();
      if (configuration.kind == LOOM_SERVE_MODEL_KIND_TEXT) {
        void* owned_weights = nullptr;
        status = iree_allocator_clone(
            configuration.host_allocator,
            iree_make_const_byte_span(weights.c_str(), weights.size() + 1),
            &owned_weights);
        if (iree_status_is_ok(status)) {
          iree_allocator_free(
              configuration.host_allocator,
              const_cast<char*>(configuration.text.weights.data));
          configuration.text.weights = iree_make_string_view(
              static_cast<const char*>(owned_weights), weights.size());
        }
      }
      if (iree_status_is_ok(status)) {
        status =
            loom_serve_registry_register(registry_, &configuration, &index);
      }
      loom_serve_model_configuration_deinitialize(&configuration);
      IREE_ASSERT_OK(status);
      ASSERT_EQ(index, i);
    }
    IREE_ASSERT_OK(Until([&] {
      return loom_serve_registry_at(registry_, 0)->phase ==
                 LOOM_SERVE_REGISTRATION_READY &&
             loom_serve_registry_at(registry_, 1)->phase ==
                 LOOM_SERVE_REGISTRATION_READY;
    }));
  }

  iree_status_t Start(iree_string_view_t content, iree_string_view_t voice,
                      size_t index = 0) {
    loom_serve_realtime_binding_t text;
    loom_serve_realtime_binding_t speech;
    IREE_RETURN_IF_ERROR(loom_serve_realtime_binding_resolve(
        registry_, loom_serve_registry_at(registry_, 0)->service.name,
        1u << LOOM_SERVE_MODEL_KIND_TEXT, &text));
    IREE_RETURN_IF_ERROR(loom_serve_realtime_binding_resolve(
        registry_, loom_serve_registry_at(registry_, 1)->service.name,
        1u << LOOM_SERVE_MODEL_KIND_SPEECH, &speech));
    loom_serve_text_request_t request = {};
    iree_string_builder_initialize(iree_allocator_system(), &request.body);
    auto status = iree_string_builder_append_format(
        &request.body,
        "{\"model\":\"fixture-text\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"%.*s\"}],"
        "\"max_tokens\":4,\"stream\":true}",
        (int)content.size, content.data);
    if (iree_status_is_ok(status)) {
      status = loom_serve_realtime_generation_allocate(
          loom_serve_host_executor(host_), registry_, text, speech, &request,
          voice, 42, &filter_, Wake(), iree_allocator_system(),
          &generations_[index]);
    }
    iree_string_builder_deinitialize(&request.body);
    return status;
  }

  void TearDown() override {
    for (size_t i = 0; i < generations_.size(); ++i) {
      Close(i);
    }
    loom_serve_realtime_generation_filter_deinitialize(&filter_);
    if (registry_) {
      for (iree_host_size_t i = 0; i < 2; ++i) {
        auto* model = loom_serve_registry_at(registry_, i);
        if (model->service.name.size) {
          IREE_EXPECT_OK(
              loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT));
        }
      }
      IREE_EXPECT_OK(
          Until([&] { return loom_serve_registry_count(registry_) == 0; }));
      IREE_EXPECT_OK(loom_serve_registry_free(registry_));
    }
    iree_notification_deinitialize(&notification_);
  }

  void Close(size_t index) {
    if (!generations_[index]) {
      return;
    }
    loom_serve_realtime_generation_cancel(generations_[index]);
    IREE_EXPECT_OK(Until([&] {
      return !loom_serve_realtime_generation_pending(generations_[index]);
    }));
    loom_serve_realtime_generation_free(generations_[index]);
    generations_[index] = nullptr;
  }

  iree_status_t Receive(size_t index, bool* progress) {
    auto* generation = generations_[index];
    const auto text = loom_serve_realtime_generation_text(generation);
    if (text.size) {
      text_[index].append(text.data, text.size);
      loom_serve_realtime_generation_consume_text(generation);
      *progress = true;
    }
    const auto audio = loom_serve_realtime_generation_audio(generation);
    if (audio.size) {
      std::string decoded(iree_base64_decoded_size(audio), '\0');
      iree_host_size_t length = 0;
      IREE_RETURN_IF_ERROR(iree_base64_decode(
          audio, iree_make_byte_span(decoded.data(), decoded.size()), &length));
      pcm_[index].append(decoded.data(), length);
      loom_serve_realtime_generation_consume_audio(generation);
      *progress = true;
    }
    return iree_ok_status();
  }

  loom_serve_wake_callback_t Wake() {
    return {[](void* self) {
              iree_notification_post(static_cast<iree_notification_t*>(self),
                                     IREE_ALL_WAITERS);
            },
            &notification_};
  }

  template <typename Done>
  iree_status_t Until(Done done) {
    auto status = iree_ok_status();
    while (iree_status_is_ok(status) && !done()) {
      auto ticket = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      for (size_t i = 0; i < generations_.size() && iree_status_is_ok(status);
           ++i) {
        if (generations_[i]) {
          status = loom_serve_realtime_generation_advance(
              generations_[i], can_publish_[i], &progress);
          if (iree_status_is_ok(status) && receive_[i]) {
            status = Receive(i, &progress);
          }
        }
      }
      for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
        auto* model = loom_serve_registry_at(registry_, i);
        if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
          bool advanced = false;
          status = model->service.advance(
              model->service.self, loom_serve_registry_service_flags(model),
              &advanced);
          progress |= advanced;
        }
        if (iree_status_is_ok(status) && model->service.name.size) {
          bool advanced = false;
          status = loom_serve_registry_advance(registry_, i, Wake(), &advanced);
          progress |= advanced;
          status = iree_status_join(status, iree_status_clone(model->status));
        }
      }
      if (!iree_status_is_ok(status) || progress || done()) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
    return status;
  }

  // Shared application retained until its accepted model and CPU jobs retire.
  loom_serve_host_t* host_ = nullptr;
  // Factory borrows the application's immutable deployment group.
  loom_serve_model_registration_options_t registration_ = {};
  // Two real compiled model services advanced by this serialized owner.
  loom_serve_registry_t* registry_ = nullptr;
  // Model and conversion completion notification, independent of wall time.
  iree_notification_t notification_;
  // Production native text-to-speech owners, each with independent publication.
  std::array<loom_serve_realtime_generation_t*, 4> generations_ = {};
  // One immutable coefficient owner shared through all response retirement.
  loom_serve_realtime_generation_filter_t filter_ = {};
  // Publication credit withheld until the native speech block is ready.
  std::array<bool, 4> can_publish_ = {true, true, true, true};
  // Consumer interest, independently withheld while native output remains live.
  std::array<bool, 4> receive_ = {};
  // Received semantic text for exact isolated/concurrent comparison.
  std::array<std::string, 4> text_;
  // Decoded wire PCM, independent of packet partitioning.
  std::array<std::string, 4> pcm_;
};

TEST_F(RealtimeGenerationTest, CancellationRetainsWaveformUntilWorkerRelease) {
  IREE_ASSERT_OK(Start(IREE_SV("abxa"), IREE_SV("positive")));
  IREE_ASSERT_OK(Until([&] {
    return loom_serve_realtime_generation_text(generations_[0]).size != 0;
  }));
  can_publish_[0] = false;
  IREE_ASSERT_OK(Until([&] {
    const auto& service = loom_serve_registry_at(registry_, 1)->service;
    return std::strcmp(service.snapshot(service.self).phase, "delivering") == 0;
  }));
  ASSERT_FALSE(loom_serve_realtime_generation_pending(generations_[0]));
  std::vector<std::unique_ptr<loom_serve::testing::WorkerGate>> gates;
  for (int32_t i = 0; i < FLAG_test_workers; ++i) {
    gates.push_back(std::make_unique<loom_serve::testing::WorkerGate>(
        loom_serve_host_executor(host_)));
  }
  bool progress = false;
  IREE_ASSERT_OK(
      loom_serve_realtime_generation_advance(generations_[0], true, &progress));
  ASSERT_TRUE(progress);
  ASSERT_TRUE(loom_serve_realtime_generation_pending(generations_[0]));
  loom_serve_realtime_generation_cancel(generations_[0]);
  IREE_ASSERT_OK(loom_serve_realtime_generation_advance(generations_[0], false,
                                                        &progress));
  EXPECT_TRUE(loom_serve_realtime_generation_pending(generations_[0]));
  EXPECT_EQ(loom_serve_realtime_generation_text(generations_[0]).size, 0);
  EXPECT_EQ(loom_serve_realtime_generation_audio(generations_[0]).size, 0);
  EXPECT_EQ(filter_.phase, LOOM_SERVE_REALTIME_FILTER_PREPARING);
  receive_[1] = true;
  IREE_ASSERT_OK(Start(IREE_SV("ab"), IREE_SV("negative"), 1));
  IREE_ASSERT_OK(
      loom_serve_realtime_generation_advance(generations_[1], true, &progress));
  gates.clear();
  IREE_ASSERT_OK(Until([&] {
    return !loom_serve_realtime_generation_pending(generations_[0]) &&
           loom_serve_realtime_generation_complete(generations_[1]);
  }));
  EXPECT_EQ(filter_.phase, LOOM_SERVE_REALTIME_FILTER_READY);
  EXPECT_FALSE(pcm_[1].empty());
  EXPECT_EQ(loom_serve_realtime_generation_audio(generations_[0]).size, 0);
}

TEST_F(RealtimeGenerationTest, EmptyContentDoesNotValidateAnUnusedVoice) {
  IREE_ASSERT_OK(Start(IREE_SV("aaaaaaaaaaa"), IREE_SV("unknown-voice")));
  IREE_ASSERT_OK(Until([&] {
    return loom_serve_realtime_generation_complete(generations_[0]);
  }));
  const auto* result = loom_serve_realtime_generation_result(generations_[0]);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->content.size, 0);
  EXPECT_EQ(result->calls->count, 0);
  EXPECT_TRUE(iree_string_view_equal(result->finish_reason, IREE_SV("stop")));
  EXPECT_EQ(result->response.usage.output_tokens, 1);
  EXPECT_EQ(loom_serve_realtime_generation_text(generations_[0]).size, 0);
  EXPECT_EQ(loom_serve_realtime_generation_audio(generations_[0]).size, 0);
  EXPECT_EQ(loom_serve_realtime_generation_frames(generations_[0]), 0);
}

TEST_F(RealtimeGenerationTest,
       EmptyContentDoesNotWaitForClosedSpeechAdmission) {
  loom_serve_registry_at(registry_, 1)->closed = true;
  IREE_ASSERT_OK(Start(IREE_SV("aaaaaaaaaaa"), IREE_SV("positive")));
  IREE_ASSERT_OK(Until([&] {
    return loom_serve_realtime_generation_complete(generations_[0]);
  }));
  const auto* result = loom_serve_realtime_generation_result(generations_[0]);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->content.size, 0);
  EXPECT_EQ(loom_serve_realtime_generation_audio(generations_[0]).size, 0);
  EXPECT_EQ(loom_serve_realtime_generation_frames(generations_[0]), 0);
}

TEST_F(RealtimeGenerationTest,
       StaggeredResponsesKeepIndependentOutputAndCredit) {
  const std::array<iree_string_view_t, 4> prompts = {
      IREE_SV("abxa"), IREE_SV("ab"), IREE_SV("aba"), IREE_SV("abxxa")};
  const std::array<iree_string_view_t, 4> voices = {
      IREE_SV("positive"), IREE_SV("negative"), IREE_SV("positive"),
      IREE_SV("negative")};
  std::array<std::string, 4> expected_text, expected_pcm;
  receive_[0] = true;
  for (size_t i = 0; i < prompts.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(Close(0));
    text_[0].clear();
    pcm_[0].clear();
    IREE_ASSERT_OK(Start(prompts[i], voices[i]));
    IREE_ASSERT_OK(Until([&] {
      return loom_serve_realtime_generation_complete(generations_[0]);
    }));
    ASSERT_FALSE(pcm_[0].empty()) << "request " << i;
    EXPECT_EQ(loom_serve_realtime_generation_frames(generations_[0]),
              pcm_[0].size() / sizeof(int16_t));
    expected_text[i] = text_[0];
    expected_pcm[i] = pcm_[0];
  }
  ASSERT_NO_FATAL_FAILURE(Close(0));
  loom_serve_realtime_generation_filter_deinitialize(&filter_);
  text_ = {};
  pcm_ = {};
  receive_ = {true, false, true, true};
  IREE_ASSERT_OK(Start(prompts[0], voices[0]));
  IREE_ASSERT_OK(Start(prompts[1], voices[1], 1));
  IREE_ASSERT_OK(Until([&] {
    return loom_serve_realtime_generation_audio(generations_[1]).size != 0;
  }));
  IREE_ASSERT_OK(Start(prompts[2], voices[2], 2));
  IREE_ASSERT_OK(Start(prompts[3], voices[3], 3));
  IREE_ASSERT_OK(Until([&] { return !pcm_[2].empty(); }));
  loom_serve_realtime_generation_cancel(generations_[2]);
  IREE_ASSERT_OK(Until([&] {
    return loom_serve_realtime_generation_complete(generations_[0]) &&
           loom_serve_realtime_generation_complete(generations_[3]) &&
           !loom_serve_realtime_generation_pending(generations_[2]);
  }));
  EXPECT_FALSE(loom_serve_realtime_generation_complete(generations_[1]));
  EXPECT_TRUE(pcm_[1].empty());
  EXPECT_EQ(expected_pcm[2].substr(0, pcm_[2].size()), pcm_[2]);
  ASSERT_NO_FATAL_FAILURE(Close(2));
  receive_[1] = true;
  IREE_ASSERT_OK(Until([&] {
    return loom_serve_realtime_generation_complete(generations_[1]);
  }));
  for (size_t i : {0u, 1u, 3u}) {
    SCOPED_TRACE(i);
    EXPECT_EQ(text_[i], expected_text[i]);
    EXPECT_EQ(pcm_[i], expected_pcm[i]);
    EXPECT_EQ(loom_serve_realtime_generation_frames(generations_[i]),
              pcm_[i].size() / sizeof(int16_t));
  }
}

}  // namespace
