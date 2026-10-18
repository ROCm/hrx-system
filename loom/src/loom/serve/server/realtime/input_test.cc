// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/input.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/testing/service_application.h"
#include "loom/serve/testing/worker_gate.h"

IREE_FLAG(string, model, "", "Recognition model document.");
IREE_FLAG(int32_t, test_workers, 1, "Shared task worker count.");

namespace {

class RealtimeInputTest : public ::testing::Test {
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
        0, nullptr, 1, loom_serve_model_registration_factory(&registration_),
        iree_allocator_system(), &registry_));
    loom_serve_model_configuration_t configuration;
    IREE_ASSERT_OK(loom_serve_model_configuration_load(
        iree_make_cstring_view(FLAG_model), iree_allocator_system(),
        &configuration));
    name_.assign(configuration.name.data, configuration.name.size);
    iree_host_size_t index = 0;
    auto status =
        loom_serve_registry_register(registry_, &configuration, &index);
    loom_serve_model_configuration_deinitialize(&configuration);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(Until([&] {
      return loom_serve_registry_at(registry_, 0)->phase ==
             LOOM_SERVE_REGISTRATION_READY;
    }));
    loom_serve_realtime_binding_t binding;
    IREE_ASSERT_OK(loom_serve_realtime_binding_resolve(
        registry_, iree_make_string_view(name_.data(), name_.size()),
        1u << LOOM_SERVE_MODEL_KIND_RECOGNITION, &binding));
    IREE_ASSERT_OK(loom_serve_realtime_input_allocate(
        loom_serve_host_executor(host_), registry_, binding, Wake(),
        iree_allocator_system(), &input_));
  }

  void TearDown() override {
    stopping_ = true;
    if (input_) {
      IREE_EXPECT_OK(
          Until([&] { return !loom_serve_realtime_input_pending(input_); }));
      loom_serve_realtime_input_free(input_);
      input_ = nullptr;
    }
    if (registry_) {
      IREE_EXPECT_OK(loom_serve_registry_remove(
          loom_serve_registry_at(registry_, 0), LOOM_SERVE_REMOVAL_ABORT));
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
      if (input_) {
        loom_serve_realtime_input_advance(input_, stopping_, &progress);
      }
      auto* model = loom_serve_registry_at(registry_, 0);
      if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
        bool advanced = false;
        status = model->service.advance(
            model->service.self, loom_serve_registry_service_flags(model),
            &advanced);
        progress |= advanced;
      }
      if (iree_status_is_ok(status) && model->service.name.size) {
        bool advanced = false;
        status = loom_serve_registry_advance(registry_, 0, Wake(), &advanced);
        progress |= advanced;
        status = iree_status_join(status, iree_status_clone(model->status));
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

  // Shared application owner retained until all accepted model work retires.
  loom_serve_host_t* host_ = nullptr;
  // Registration factory borrows this immutable deployment group.
  loom_serve_model_registration_options_t registration_ = {};
  // Real source-backed recognition service, advanced on the test owner thread.
  loom_serve_registry_t* registry_ = nullptr;
  // Copied public deployment identity, stable for binding resolution.
  std::string name_;
  // Explicit transport/model/CPU completion notification; no timing polls.
  iree_notification_t notification_;
  // Production input owner under test, without a mock conversion path.
  loom_serve_realtime_input_t* input_ = nullptr;
  // Teardown closes items before joining their accepted borrowers.
  bool stopping_ = false;
};

TEST_F(RealtimeInputTest, CancelRetainsPcmUntilAcceptedConversionReleases) {
  const loom_serve_recognition_request_t request = {
      .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
  };
  loom_serve_realtime_input_item_t* item = nullptr;
  IREE_ASSERT_OK(loom_serve_realtime_input_begin(input_, &request, &item));
  const uint8_t samples[] = {0, 0, 0, 0};
  unsigned releases = 0;
  IREE_ASSERT_OK(loom_serve_realtime_input_append(
      input_, item, iree_make_const_byte_span(samples, sizeof(samples)),
      [](void* data) { ++*static_cast<unsigned*>(data); }, &releases));
  std::vector<std::unique_ptr<loom_serve::testing::WorkerGate>> gates;
  for (int32_t i = 0; i < FLAG_test_workers; ++i) {
    gates.push_back(std::make_unique<loom_serve::testing::WorkerGate>(
        loom_serve_host_executor(host_)));
  }
  bool progress = false;
  loom_serve_realtime_input_advance(input_, false, &progress);
  IREE_ASSERT_OK(loom_serve_realtime_input_status(item));
  ASSERT_TRUE(progress);
  loom_serve_realtime_input_close(item);
  loom_serve_realtime_input_advance(input_, false, &progress);
  ASSERT_TRUE(loom_serve_realtime_input_pending(input_));
  EXPECT_EQ(releases, 0u);
  gates.clear();
  IREE_ASSERT_OK(
      Until([&] { return !loom_serve_realtime_input_pending(input_); }));
  EXPECT_EQ(releases, 1u);
}

TEST_F(RealtimeInputTest, IndependentPcmItemsFlushFiniteConversion) {
  const loom_serve_recognition_request_t request = {
      .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
  };
  loom_serve_realtime_input_item_t* items[2] = {};
  unsigned releases = 0;
  const std::vector<uint8_t> samples(480, 0);
  for (auto& item : items) {
    IREE_ASSERT_OK(loom_serve_realtime_input_begin(input_, &request, &item));
    size_t offset = 0;
    for (size_t frames : {1u, 7u, 93u, 139u}) {
      IREE_ASSERT_OK(loom_serve_realtime_input_append(
          input_, item,
          iree_make_const_byte_span(samples.data() + offset, frames * 2),
          [](void* data) { ++*static_cast<unsigned*>(data); }, &releases));
      offset += frames * 2;
    }
    loom_serve_realtime_input_finish(item);
  }
  IREE_ASSERT_OK(Until([&] {
    return loom_serve_realtime_input_transcript(items[0]) &&
           loom_serve_realtime_input_transcript(items[1]);
  }));
  EXPECT_EQ(releases, 8u);
  for (auto* item : items) {
    IREE_ASSERT_OK(loom_serve_realtime_input_status(item));
    const auto* text = loom_serve_realtime_input_transcript(item);
    EXPECT_EQ(std::string(text->buffer, text->size), "A");
    EXPECT_EQ(loom_serve_realtime_input_frames(item), 240u);
    loom_serve_realtime_input_close(item);
  }
  IREE_ASSERT_OK(
      Until([&] { return !loom_serve_realtime_input_pending(input_); }));
}

TEST_F(RealtimeInputTest, PrimedActivityPreservesCoverageAndIndependentOutput) {
  for (auto outputs : {LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY,
                       LOOM_SERVE_RECOGNITION_OUTPUT_ALL}) {
    SCOPED_TRACE(outputs);
    const loom_serve_recognition_request_t activity_request = {
        .outputs = outputs,
    };
    loom_serve_realtime_input_item_t* activity = nullptr;
    IREE_ASSERT_OK(
        loom_serve_realtime_input_begin(input_, &activity_request, &activity));
    IREE_ASSERT_OK(Until([&] {
      return loom_serve_realtime_input_geometry(activity) != nullptr;
    }));
    const auto geometry = *loom_serve_realtime_input_geometry(activity);
    EXPECT_EQ(geometry.sample_rate, 16000u);
    EXPECT_GT(geometry.retention_frames, 4096u);
    EXPECT_FALSE(loom_serve_realtime_input_complete(activity));

    const loom_serve_recognition_request_t transcript_request = {
        .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT,
    };
    loom_serve_realtime_input_item_t* transcript = nullptr;
    IREE_ASSERT_OK(loom_serve_realtime_input_begin(input_, &transcript_request,
                                                   &transcript));
    const uint8_t samples[38] = {};
    unsigned releases = 0;
    for (auto* item : {activity, transcript}) {
      size_t offset = 0;
      for (size_t frames : {1u, 7u, 11u}) {
        IREE_ASSERT_OK(loom_serve_realtime_input_append(
            input_, item,
            iree_make_const_byte_span(samples + offset, frames * 2),
            [](void* data) { ++*static_cast<unsigned*>(data); }, &releases));
        offset += frames * 2;
      }
      loom_serve_realtime_input_finish(item);
    }
    // Keeping one source observation borrowed cannot stall another item.
    IREE_ASSERT_OK(Until([&] {
      return loom_serve_realtime_input_activity(activity) &&
             loom_serve_realtime_input_transcript(transcript);
    }));
    const auto* text = loom_serve_realtime_input_transcript(transcript);
    EXPECT_EQ(std::string(text->buffer, text->size), "A");
    uint64_t covered = 0;
    while (!loom_serve_realtime_input_complete(activity)) {
      IREE_ASSERT_OK(Until([&] {
        return loom_serve_realtime_input_activity(activity) ||
               loom_serve_realtime_input_complete(activity);
      }));
      IREE_ASSERT_OK(loom_serve_realtime_input_status(activity));
      if (const auto* observation =
              loom_serve_realtime_input_activity(activity)) {
        EXPECT_EQ(observation->begin, covered);
        EXPECT_GT(observation->end, observation->begin);
        EXPECT_EQ(observation->speech_probability, 0.0f);
        covered = observation->end;
        loom_serve_realtime_input_consume_activity(activity);
      }
    }
    EXPECT_EQ(covered, 13u);  // ceil(19 original frames * 16000 / 24000).
    EXPECT_EQ(releases, 6u);
    EXPECT_EQ(loom_serve_realtime_input_geometry(activity)->retention_frames,
              geometry.retention_frames);
    if (outputs & LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT) {
      text = loom_serve_realtime_input_transcript(activity);
      EXPECT_EQ(std::string(text->buffer, text->size), "A");
    }
    loom_serve_realtime_input_close(activity);
    loom_serve_realtime_input_close(transcript);
    IREE_ASSERT_OK(
        Until([&] { return !loom_serve_realtime_input_pending(input_); }));
  }
}

}  // namespace
