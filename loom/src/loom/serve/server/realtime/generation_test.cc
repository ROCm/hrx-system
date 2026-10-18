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

  iree_status_t Start(iree_string_view_t content, iree_string_view_t voice) {
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
          voice, 42, &audio_, Wake(), iree_allocator_system(), &generation_);
    }
    iree_string_builder_deinitialize(&request.body);
    return status;
  }

  void TearDown() override {
    if (generation_) {
      loom_serve_realtime_generation_cancel(generation_);
      IREE_EXPECT_OK(Until([&] {
        return !loom_serve_realtime_generation_pending(generation_);
      }));
      loom_serve_realtime_generation_free(generation_);
      generation_ = nullptr;
    }
    loom_serve_realtime_audio_deinitialize(&audio_);
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
      if (generation_) {
        status = loom_serve_realtime_generation_advance(
            generation_, can_publish_, &progress);
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
  // Production native text-to-speech response under test.
  loom_serve_realtime_generation_t* generation_ = nullptr;
  // Session storage retained through the response's accepted conversion job.
  loom_serve_realtime_audio_t audio_ = {};
  // Publication credit withheld until the native speech block is ready.
  bool can_publish_ = true;
};

TEST_F(RealtimeGenerationTest, CancellationRetainsWaveformUntilWorkerRelease) {
  IREE_ASSERT_OK(Start(IREE_SV("abxa"), IREE_SV("positive")));
  IREE_ASSERT_OK(Until([&] {
    return loom_serve_realtime_generation_text(generation_).size != 0;
  }));
  can_publish_ = false;
  IREE_ASSERT_OK(Until([&] {
    const auto& service = loom_serve_registry_at(registry_, 1)->service;
    return std::strcmp(service.snapshot(service.self).phase, "delivering") == 0;
  }));
  ASSERT_FALSE(loom_serve_realtime_generation_pending(generation_));
  std::vector<std::unique_ptr<loom_serve::testing::WorkerGate>> gates;
  for (int32_t i = 0; i < FLAG_test_workers; ++i) {
    gates.push_back(std::make_unique<loom_serve::testing::WorkerGate>(
        loom_serve_host_executor(host_)));
  }
  bool progress = false;
  IREE_ASSERT_OK(
      loom_serve_realtime_generation_advance(generation_, true, &progress));
  ASSERT_TRUE(progress);
  ASSERT_TRUE(loom_serve_realtime_generation_pending(generation_));
  loom_serve_realtime_generation_cancel(generation_);
  IREE_ASSERT_OK(
      loom_serve_realtime_generation_advance(generation_, false, &progress));
  EXPECT_TRUE(loom_serve_realtime_generation_pending(generation_));
  EXPECT_EQ(loom_serve_realtime_generation_text(generation_).size, 0);
  EXPECT_EQ(loom_serve_realtime_generation_audio(generation_).size, 0);
  gates.clear();
  IREE_ASSERT_OK(Until(
      [&] { return !loom_serve_realtime_generation_pending(generation_); }));
  EXPECT_EQ(loom_serve_realtime_generation_audio(generation_).size, 0);
}

TEST_F(RealtimeGenerationTest, EmptyContentDoesNotValidateAnUnusedVoice) {
  IREE_ASSERT_OK(Start(IREE_SV("aaaaaaaaaaa"), IREE_SV("unknown-voice")));
  IREE_ASSERT_OK(Until(
      [&] { return loom_serve_realtime_generation_complete(generation_); }));
  const auto* result = loom_serve_realtime_generation_result(generation_);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->content.size, 0);
  EXPECT_EQ(result->calls->count, 0);
  EXPECT_TRUE(iree_string_view_equal(result->finish_reason, IREE_SV("stop")));
  EXPECT_EQ(result->response.usage.output_tokens, 1);
  EXPECT_EQ(loom_serve_realtime_generation_text(generation_).size, 0);
  EXPECT_EQ(loom_serve_realtime_generation_audio(generation_).size, 0);
  EXPECT_EQ(loom_serve_realtime_generation_frames(generation_), 0);
}

TEST_F(RealtimeGenerationTest,
       EmptyContentDoesNotWaitForClosedSpeechAdmission) {
  loom_serve_registry_at(registry_, 1)->closed = true;
  IREE_ASSERT_OK(Start(IREE_SV("aaaaaaaaaaa"), IREE_SV("positive")));
  IREE_ASSERT_OK(Until(
      [&] { return loom_serve_realtime_generation_complete(generation_); }));
  const auto* result = loom_serve_realtime_generation_result(generation_);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->content.size, 0);
  EXPECT_EQ(loom_serve_realtime_generation_audio(generation_).size, 0);
  EXPECT_EQ(loom_serve_realtime_generation_frames(generation_), 0);
}

}  // namespace
