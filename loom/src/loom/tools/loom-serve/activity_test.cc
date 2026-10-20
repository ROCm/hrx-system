// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-serve/activity_test.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/registry.h"
#include "loom/serve/service/recognition/stream.h"
#include "loom/serve/service/recognition/transcription.h"
#include "loom/serve/testing/worker_gate.h"
#include "loom/tools/loom-serve/options.h"

IREE_FLAG(string, model, "", "Activity model document.");
IREE_FLAG(string, input, "", "Mono F32LE audio for learned qualification.");
IREE_FLAG(string, reference, "", "Independent per-hop F32LE probabilities.");
IREE_FLAG(string, transcription_model, "",
          "Optional learned STT model document.");
IREE_FLAG(string, expected_text, "", "Expected learned final transcript.");
IREE_FLAG(int32_t, sample_rate, 16000, "Input audio sample rate.");
IREE_FLAG(int32_t, stream_count, 4, "Independent membership histories.");
IREE_FLAG(bool, fixture, false,
          "Use deterministic fixture inputs and model roles.");
IREE_FLAG(
    int32_t, fixture_transcription_device, 0,
    "Device index for the fixture recognizer; activity uses device zero.");
IREE_FLAG(int32_t, pre_roll_frames, 12,
          "Test policy: retained onset pre-roll.");
IREE_FLAG(int32_t, hangover_frames, 18,
          "Test policy: silence before input EOF.");
IREE_FLAG(int32_t, retained_frames, 32,
          "Test input retention capacity in frames.");
IREE_FLAG(int64_t, expected_begin, 24,
          "Expected selected utterance start frame.");
IREE_FLAG(int64_t, expected_end, 111, "Expected selected utterance end frame.");

namespace {
using Stream = std::unique_ptr<loom_serve_recognition_stream_t,
                               decltype(&loom_serve_recognition_stream_close)>;

std::string Read(const char* path) {
  std::ifstream input(path, std::ios::binary);
  EXPECT_TRUE(input.good()) << path;
  std::string result{std::istreambuf_iterator<char>(input), {}};
  EXPECT_FALSE(input.bad()) << path;
  return result;
}

struct loom_serve_activity_test_stream_t {
  // Caller-owned handle, independent of registration/model lifetime.
  Stream stream{nullptr, loom_serve_recognition_stream_close};
  // Observer input for this independent recorded microphone history.
  std::vector<float> samples;
  // Frames already accepted by the bounded native ring.
  size_t written = 0;
  // Explicit input EOF has reached the native owner.
  bool finished = false;
  // Last contiguous source-covered frame.
  uint64_t covered = 0;
  // Observer-only probability receipt, never source execution state.
  std::vector<float> probabilities;
};

struct loom_serve_activity_test_utterance_t {
  // First replayed microphone frame, including bounded pre-roll.
  uint64_t begin;
  // Exclusive microphone frame passed to STT before explicit EOF.
  uint64_t end;
  // Final learned transcript copied before the caller releases its stream.
  std::string text;
};

// The executable owns one application for its entire lifetime, just like the
// server. Tests retire their registries before this shared device group closes.
class ActivityEnvironment : public ::testing::Environment {
 public:
  ActivityEnvironment(const loom_serve_compiler_options_t* compiler_options,
                      iree_hal_driver_registry_t* driver_registry)
      : compiler_options_(compiler_options),
        driver_registry_(driver_registry) {}

  void SetUp() override {
    IREE_ASSERT_OK(loom_serve_host_create_from_flags(
        compiler_options_, iree_allocator_system(), &host_));
    IREE_ASSERT_OK(loom_serve_device_group_allocate_from_flags(
        host_, driver_registry_, iree_allocator_system(), &group_));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
    loom_serve_host_release(host_);
  }

  loom_serve_host_t* host() const { return host_; }
  loom_serve_device_group_t* group() const { return group_; }

 private:
  // Exact compiler composition borrowed from the executable entry point.
  const loom_serve_compiler_options_t* compiler_options_;
  // Available non-task drivers, borrowed through application teardown.
  iree_hal_driver_registry_t* driver_registry_;
  // Shared task, compiler, and I/O services outlive every test registry.
  loom_serve_host_t* host_ = nullptr;
  // Ordered explicit device records, retained through all model retirement.
  loom_serve_device_group_t* group_ = nullptr;
};

// Installed by the physical executable before gtest starts the application.
ActivityEnvironment* activity_environment = nullptr;

class NativeActivityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    ASSERT_GE(FLAG_stream_count, 3);
    ASSERT_LE(FLAG_stream_count, 8);
    ASSERT_GT(FLAG_sample_rate, 0);
    ASSERT_GE(FLAG_pre_roll_frames, 0);
    ASSERT_GT(FLAG_hangover_frames, 0);
    ASSERT_GT(FLAG_retained_frames, FLAG_pre_roll_frames);
    ASSERT_TRUE(FLAG_fixture || FLAG_input[0]);
    ASSERT_TRUE(FLAG_fixture || FLAG_reference[0]);
    if (!FLAG_fixture) {
      const std::string bytes = Read(FLAG_input);
      ASSERT_EQ(bytes.size() % sizeof(float), 0u);
      samples_.resize(bytes.size() / sizeof(float));
      for (size_t i = 0; i < samples_.size(); ++i) {
        samples_[i] =
            iree_unaligned_load_le_f32(bytes.data() + i * sizeof(float));
      }
    } else {
      samples_.assign(181, 0.25f);
    }
    ASSERT_FALSE(samples_.empty());
    registration_.device_group = activity_environment->group();
    model_count_ = FLAG_fixture || FLAG_transcription_model[0] ? 2 : 1;
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        0, nullptr, model_count_,
        loom_serve_model_registration_factory(&registration_), allocator_,
        &registry_));
    ASSERT_NO_FATAL_FAILURE(Register(FLAG_model, 0));
    if (model_count_ == 2) {
      ASSERT_NO_FATAL_FAILURE(Register(
          FLAG_transcription_model[0] ? FLAG_transcription_model : FLAG_model,
          1));
    }
    IREE_ASSERT_OK(Until(
        [&] {
          for (size_t i = 0; i < model_count_; ++i) {
            if (Model(i)->phase != LOOM_SERVE_REGISTRATION_READY) {
              return false;
            }
          }
          return true;
        },
        [](bool*) { return iree_ok_status(); }));
  }

  void TearDown() override {
    if (registry_) {
      for (size_t i = 0; i < model_count_; ++i) {
        if (Model(i)->service.name.size) {
          IREE_EXPECT_OK(
              loom_serve_registry_remove(Model(i), LOOM_SERVE_REMOVAL_ABORT));
        }
      }
      IREE_EXPECT_OK(
          Until([&] { return loom_serve_registry_count(registry_) == 0; },
                [](bool*) { return iree_ok_status(); }));
      IREE_EXPECT_OK(loom_serve_registry_free(registry_));
    }
    iree_notification_deinitialize(&notification_);
  }

  void Register(const char* path, size_t role) {
    loom_serve_model_configuration_t configuration;
    IREE_ASSERT_OK(loom_serve_model_configuration_load(
        iree_make_cstring_view(path), allocator_, &configuration));
    auto status = iree_ok_status();
    if (FLAG_fixture) {
      configuration.device_index = role ? FLAG_fixture_transcription_device : 0;
      if (role == 0) {
        configuration.kind = LOOM_SERVE_MODEL_KIND_ACTIVITY;
        const char name[] = "activity";
        void* owned_name = nullptr;
        status = iree_allocator_clone(
            configuration.host_allocator,
            iree_make_const_byte_span(name, sizeof(name)), &owned_name);
        if (iree_status_is_ok(status)) {
          iree_allocator_free(configuration.host_allocator,
                              const_cast<char*>(configuration.name.data));
          configuration.name =
              iree_make_cstring_view(static_cast<const char*>(owned_name));
        }
      }
    }
    names_[role].assign(configuration.name.data, configuration.name.size);
    iree_host_size_t index = 0;
    if (iree_status_is_ok(status)) {
      status = loom_serve_registry_register(registry_, &configuration, &index);
    }
    loom_serve_model_configuration_deinitialize(&configuration);
    IREE_ASSERT_OK(status);
    EXPECT_EQ(index, role);
  }

  loom_serve_registered_model_t* Model(size_t index = 0) {
    return loom_serve_registry_at(registry_, index);
  }

  template <typename Done, typename Pump>
  iree_status_t Until(Done done, Pump pump) {
    const loom_serve_wake_callback_t wake = {
        [](void* self) {
          iree_notification_post(static_cast<iree_notification_t*>(self),
                                 IREE_ALL_WAITERS);
        },
        &notification_};
    auto status = iree_ok_status();
    while (iree_status_is_ok(status) && !done()) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      status = pump(&progress);
      for (size_t i = 0; i < model_count_ && iree_status_is_ok(status); ++i) {
        const size_t index = (cursor_ + i) % model_count_;
        auto* model = Model(index);
        if (!model->service.name.size) {
          continue;
        }
        bool advanced = false;
        if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
          auto& service = model->service;
          status = service.advance(service.self,
                                   loom_serve_registry_service_flags(model),
                                   &advanced);
          progress |= advanced;
        }
        if (iree_status_is_ok(status)) {
          status =
              loom_serve_registry_advance(registry_, index, wake, &advanced);
          progress |= advanced;
        }
        // Preparation failures remain recorded until removal finishes. Report
        // them during setup, but let teardown finish retiring accepted work.
        // A release failure prevents retirement and must still stop the wait.
        if (!model->removal ||
            model->phase == LOOM_SERVE_REGISTRATION_RELEASE_FAILED) {
          status = iree_status_join(status, iree_status_clone(model->status));
        }
      }
      cursor_ = (cursor_ + 1) % model_count_;
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

  iree_status_t Open(loom_serve_activity_test_stream_t* result,
                     size_t role = 0) {
    const loom_serve_recognition_output_flags_t outputs =
        role ? LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT
             : LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY;
    const loom_serve_recognition_request_t request = {
        .outputs = outputs,
    };
    loom_serve_recognition_stream_t* stream = nullptr;
    auto status = loom_serve_registry_try_open_recognition(
        registry_, iree_make_cstring_view(names_[role].c_str()), &request,
        &stream);
    result->stream.reset(stream);
    if (iree_status_is_ok(status) && stream) {
      const auto format = loom_serve_recognition_stream_format(stream);
      if (format.sample_rate != static_cast<uint32_t>(FLAG_sample_rate) ||
          format.channels != 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "input geometry differs");
      }
    }
    return status;
  }

  iree_status_t Feed(loom_serve_activity_test_stream_t* input, bool* progress,
                     bool finish = true) {
    auto* stream = input->stream.get();
    // Uneven transport extents must not become the model's numerical hop.
    const std::array<size_t, 5> sizes = {17, 251, 29, 613, 1021};
    const size_t count = std::min({input->samples.size() - input->written,
                                   loom_serve_recognition_stream_credit(stream),
                                   sizes[input->written % sizes.size()]});
    if (count) {
      iree_host_size_t accepted = 0;
      IREE_RETURN_IF_ERROR(loom_serve_recognition_stream_write(
          stream,
          iree_make_const_byte_span(input->samples.data() + input->written,
                                    count * sizeof(float)),
          &accepted));
      input->written += accepted;
      *progress |= accepted != 0;
    }
    if (finish && input->written == input->samples.size() && !input->finished) {
      loom_serve_recognition_stream_finish(stream);
      input->finished = true;
      *progress = true;
    }
    return loom_serve_recognition_stream_status(stream);
  }

  iree_status_t Drain(loom_serve_activity_test_stream_t* input,
                      bool* progress) {
    while (const auto* event =
               loom_serve_recognition_stream_peek(input->stream.get())) {
      if (event->kind != LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY ||
          event->activity.begin != input->covered ||
          event->activity.end > input->samples.size()) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "activity coverage differs");
      }
      input->covered = event->activity.end;
      input->probabilities.push_back(event->activity.speech_probability);
      loom_serve_recognition_stream_consume(input->stream.get());
      *progress = true;
    }
    return loom_serve_recognition_stream_status(input->stream.get());
  }

  bool Done(const loom_serve_activity_test_stream_t& input) {
    return input.stream &&
           loom_serve_recognition_stream_is_complete(input.stream.get());
  }

  void Run(loom_serve_activity_test_stream_t* input) {
    IREE_ASSERT_OK(Open(input));
    ASSERT_NE(input->stream, nullptr);
    IREE_ASSERT_OK(Until([&] { return Done(*input); },
                         [&](bool* progress) {
                           IREE_RETURN_IF_ERROR(Feed(input, progress));
                           return Drain(input, progress);
                         }));
    IREE_ASSERT_OK(loom_serve_recognition_stream_status(input->stream.get()));
    EXPECT_EQ(input->covered, input->samples.size());
  }

  iree_status_t FinalText(loom_serve_recognition_stream_t* stream,
                          std::string* text) {
    IREE_RETURN_IF_ERROR(loom_serve_recognition_stream_status(stream));
    const auto* document = loom_serve_recognition_stream_document(stream);
    if (!document) {
      return iree_make_status(IREE_STATUS_DATA_LOSS, "missing final document");
    }
    iree_string_builder_t output;
    iree_string_builder_initialize(allocator_, &output);
    auto status = loom_serve_transcription_format(document, true, &output);
    if (iree_status_is_ok(status)) {
      text->assign(iree_string_builder_buffer(&output),
                   iree_string_builder_size(&output));
    }
    iree_string_builder_deinitialize(&output);
    return status;
  }

  void Transcribe(const std::vector<float>& samples, std::string* text) {
    loom_serve_activity_test_stream_t input;
    input.samples = samples;
    IREE_ASSERT_OK(Open(&input, 1));
    ASSERT_NE(input.stream, nullptr);
    IREE_ASSERT_OK(
        Until([&] { return Done(input); },
              [&](bool* progress) {
                IREE_RETURN_IF_ERROR(Feed(&input, progress));
                while (loom_serve_recognition_stream_peek(input.stream.get())) {
                  loom_serve_recognition_stream_consume(input.stream.get());
                  *progress = true;
                }
                return loom_serve_recognition_stream_status(input.stream.get());
              }));
    IREE_ASSERT_OK(FinalText(input.stream.get(), text));
  }

  iree_status_t RecognizeActivity(
      const std::vector<float>& samples,
      std::vector<loom_serve_activity_test_utterance_t>* utterances,
      size_t* maximum_retained) {
    // Retain caller pre-roll plus accepted input and one pending activity
    // packet. Each packet covers at most one source input window.
    const uint64_t pre_roll = FLAG_pre_roll_frames;
    const uint64_t hangover = FLAG_hangover_frames;
    std::vector<float> ring(FLAG_retained_frames);
    uint64_t ring_begin = 0;
    uint64_t stt_position = 0;
    uint64_t utterance_begin = 0;
    uint64_t utterance_end = 0;
    uint64_t last_voice = 0;
    loom_serve_activity_test_stream_t detector;
    loom_serve_activity_test_stream_t stt;
    IREE_RETURN_IF_ERROR(Open(&detector));
    if (!detector.stream) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    const iree_host_size_t input_capacity =
        loom_serve_recognition_stream_input_capacity(detector.stream.get());
    if (pre_roll > ring.size() ||
        input_capacity > (ring.size() - pre_roll) / 2) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "activity retention requires pre-roll and two source input windows");
    }
    return Until(
        [&] { return detector.finished && Done(detector) && !stt.stream; },
        [&](bool* progress) {
          auto* stream = detector.stream.get();
          const size_t retained = detector.written - ring_begin;
          const size_t count =
              std::min({samples.size() - detector.written,
                        loom_serve_recognition_stream_credit(stream),
                        ring.size() - retained, size_t{317}});
          if (count) {
            iree_host_size_t accepted = 0;
            IREE_RETURN_IF_ERROR(loom_serve_recognition_stream_write(
                stream,
                iree_make_const_byte_span(samples.data() + detector.written,
                                          count * sizeof(float)),
                &accepted));
            for (size_t i = 0; i < accepted; ++i) {
              ring[(detector.written + i) % ring.size()] =
                  samples[detector.written + i];
            }
            detector.written += accepted;
            *maximum_retained = std::max(*maximum_retained,
                                         size_t(detector.written - ring_begin));
            *progress |= accepted != 0;
          }
          if (detector.written == samples.size() && !detector.finished) {
            loom_serve_recognition_stream_finish(stream);
            detector.finished = true;
            *progress = true;
          }
          while (const auto* event =
                     loom_serve_recognition_stream_peek(stream)) {
            if (event->kind != LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY ||
                event->activity.begin != detector.covered) {
              return iree_make_status(IREE_STATUS_DATA_LOSS,
                                      "activity clock changed");
            }
            const auto& activity = event->activity;
            const bool speaking = activity.speech_probability >= 0.5f;
            if (speaking && stt.stream && utterance_end) {
              break;
            }
            if (speaking && !stt.stream) {
              IREE_RETURN_IF_ERROR(Open(&stt, 1));
              if (!stt.stream) {
                break;
              }
              utterance_begin =
                  activity.begin > pre_roll ? activity.begin - pre_roll : 0;
              if (utterance_begin < ring_begin) {
                return iree_make_status(IREE_STATUS_DATA_LOSS,
                                        "pre-roll was overwritten");
              }
              stt_position = utterance_begin;
              utterance_end = 0;
              stt.finished = false;
            }
            if (speaking) {
              last_voice = activity.end;
            }
            if (stt.stream && !utterance_end &&
                activity.end - last_voice >= hangover) {
              utterance_end = activity.end;
            }
            detector.covered = activity.end;
            loom_serve_recognition_stream_consume(stream);
            *progress = true;
          }
          IREE_RETURN_IF_ERROR(loom_serve_recognition_stream_status(stream));
          if (Done(detector) && stt.stream && !utterance_end) {
            utterance_end = detector.covered;
          }
          if (stt.stream) {
            const uint64_t target =
                utterance_end ? utterance_end : detector.covered;
            const size_t offset = stt_position % ring.size();
            const size_t count = std::min(
                {size_t(target - stt_position),
                 loom_serve_recognition_stream_credit(stt.stream.get()),
                 ring.size() - offset});
            if (count) {
              iree_host_size_t accepted = 0;
              IREE_RETURN_IF_ERROR(loom_serve_recognition_stream_write(
                  stt.stream.get(),
                  iree_make_const_byte_span(ring.data() + offset,
                                            count * sizeof(float)),
                  &accepted));
              stt_position += accepted;
              *progress |= accepted != 0;
            }
            if (utterance_end && stt_position == utterance_end &&
                !stt.finished) {
              loom_serve_recognition_stream_finish(stt.stream.get());
              stt.finished = true;
              *progress = true;
            }
            while (loom_serve_recognition_stream_peek(stt.stream.get())) {
              loom_serve_recognition_stream_consume(stt.stream.get());
              *progress = true;
            }
            IREE_RETURN_IF_ERROR(
                loom_serve_recognition_stream_status(stt.stream.get()));
            if (Done(stt)) {
              loom_serve_activity_test_utterance_t utterance = {
                  .begin = utterance_begin,
                  .end = utterance_end,
                  .text = {},
              };
              IREE_RETURN_IF_ERROR(
                  FinalText(stt.stream.get(), &utterance.text));
              utterances->push_back(std::move(utterance));
              stt.stream.reset();
              *progress = true;
            }
          }
          uint64_t retain =
              detector.covered > pre_roll ? detector.covered - pre_roll : 0;
          if (stt.stream) {
            retain = std::min(retain, stt_position);
          }
          ring_begin = std::max(ring_begin, retain);
          return iree_ok_status();
        });
  }

  // System allocator shared by the real registry and its model owners.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Notification joins owner readiness without timing sleeps or polling
  // retries.
  iree_notification_t notification_;
  // Source registration factory using the linked device/compiler leaf.
  loom_serve_model_registration_options_t registration_ = {};
  // Registry of the detector and optional recognizer.
  loom_serve_registry_t* registry_ = nullptr;
  // Names remain stable after registration removal.
  std::array<std::string, 2> names_;
  // Actual number of configured model roles.
  size_t model_count_ = 1;
  // Rotating first service prevents owner iteration bias.
  size_t cursor_ = 0;
  // Recorded input owned by the test's microphone and independent oracle.
  std::vector<float> samples_;
};

TEST_F(NativeActivityTest, ProbabilityAndEmptyEof) {
  loom_serve_activity_test_stream_t input;
  input.samples = samples_;
  ASSERT_NO_FATAL_FAILURE(Run(&input));
  if (FLAG_reference[0]) {
    const auto expected = Read(FLAG_reference);
    ASSERT_EQ(expected.size(), input.probabilities.size() * sizeof(float));
    for (size_t i = 0; i < input.probabilities.size(); ++i) {
      EXPECT_NEAR(
          input.probabilities[i],
          iree_unaligned_load_le_f32(expected.data() + i * sizeof(float)),
          1.0e-5f)
          << i;
    }
  } else if (FLAG_fixture) {
    for (float probability : input.probabilities) {
      EXPECT_EQ(probability, 0.75f);
    }
  }
  loom_serve_activity_test_stream_t empty;
  ASSERT_NO_FATAL_FAILURE(Run(&empty));
  EXPECT_TRUE(empty.probabilities.empty());
}

TEST_F(NativeActivityTest, ChangingMembershipPreservesHistories) {
  std::vector<loom_serve_activity_test_stream_t> baseline(FLAG_stream_count);
  for (size_t i = 0; i < baseline.size(); ++i) {
    auto& input = baseline[i];
    input.samples = samples_;
    input.samples.resize(samples_.size() -
                         i * (samples_.size() / (baseline.size() + 1)));
    for (float& sample : input.samples) {
      sample *= (1.0f - i * 0.05f);
    }
    ASSERT_NO_FATAL_FAILURE(Run(&input));
    input.stream.reset();
  }
  std::vector<loom_serve_activity_test_stream_t> live(FLAG_stream_count);
  for (size_t i = 0; i < live.size(); ++i) {
    live[i].samples = baseline[i].samples;
  }
  IREE_ASSERT_OK(Open(&live[0]));
  ASSERT_NE(live[0].stream, nullptr);
  bool held = false;
  bool released = false;
  uint64_t held_end = 0;
  float held_probability = 0;
  IREE_ASSERT_OK(Until(
      [&] {
        return std::all_of(live.begin(), live.end(),
                           [&](const auto& input) { return Done(input); });
      },
      [&](bool* progress) {
        for (size_t i = 1; i < live.size(); ++i) {
          if (!live[i].stream && live[0].probabilities.size() >= i) {
            IREE_RETURN_IF_ERROR(Open(&live[i]));
            *progress |= bool(live[i].stream);
          }
        }
        for (size_t i = 0; i < live.size(); ++i) {
          auto& input = live[i];
          if (!input.stream) {
            continue;
          }
          IREE_RETURN_IF_ERROR(Feed(&input, progress));
          if (i == 1 && !released) {
            const auto* event =
                loom_serve_recognition_stream_peek(input.stream.get());
            if (event && !held) {
              held_end = event->activity.end;
              held_probability = event->activity.speech_probability;
              held = true;
              *progress = true;
            }
            if (held && Done(live.back())) {
              EXPECT_NE(event, nullptr);
              if (event) {
                EXPECT_EQ(event->activity.end, held_end);
                EXPECT_EQ(event->activity.speech_probability, held_probability);
              }
              released = true;
              *progress = true;
            }
            if (!released) {
              continue;
            }
          }
          IREE_RETURN_IF_ERROR(Drain(&input, progress));
        }
        return iree_ok_status();
      }));
  ASSERT_TRUE(held);
  ASSERT_TRUE(released);
  for (size_t i = 0; i < live.size(); ++i) {
    EXPECT_EQ(live[i].covered, baseline[i].covered);
    ASSERT_EQ(live[i].probabilities.size(), baseline[i].probabilities.size());
    EXPECT_EQ(std::memcmp(live[i].probabilities.data(),
                          baseline[i].probabilities.data(),
                          live[i].probabilities.size() * sizeof(float)),
              0)
        << i;
  }
}

TEST_F(NativeActivityTest, AcceptedCancellationAndRetainedFinalEvent) {
  loom_serve_activity_test_stream_t interrupted;
  interrupted.samples = samples_;
  IREE_ASSERT_OK(Open(&interrupted));
  ASSERT_NE(interrupted.stream, nullptr);
  IREE_ASSERT_OK(Until(
      [&] {
        return loom_serve_recognition_stream_peek(interrupted.stream.get()) !=
               nullptr;
      },
      [&](bool* progress) { return Feed(&interrupted, progress, false); }));
  bool consumed = false;
  IREE_ASSERT_OK(Drain(&interrupted, &consumed));
  {
    std::vector<std::unique_ptr<loom_serve::testing::WorkerGate>> gates;
    auto* executor = loom_serve_host_executor(activity_environment->host());
    for (size_t i = 0; i < iree_task_executor_worker_count(executor); ++i) {
      gates.push_back(
          std::make_unique<loom_serve::testing::WorkerGate>(executor));
    }
    bool progress = false;
    IREE_ASSERT_OK(Feed(&interrupted, &progress, false));
    auto& service = Model()->service;
    IREE_ASSERT_OK(service.advance(
        service.self, loom_serve_registry_service_flags(Model()), &progress));
    EXPECT_TRUE(progress);
    interrupted.stream.reset();
    IREE_ASSERT_OK(service.advance(
        service.self, loom_serve_registry_service_flags(Model()), &progress));
    EXPECT_EQ(service.telemetry.counters->cancelled, 0u);
  }
  loom_serve_activity_test_stream_t peer;
  peer.samples = samples_;
  ASSERT_NO_FATAL_FAILURE(Run(&peer));
  EXPECT_EQ(Model()->service.telemetry.counters->cancelled, 1u);
  loom_serve_activity_test_stream_t held;
  held.samples = samples_;
  IREE_ASSERT_OK(Open(&held));
  ASSERT_NE(held.stream, nullptr);
  IREE_ASSERT_OK(Until(
      [&] {
        const auto* event =
            loom_serve_recognition_stream_peek(held.stream.get());
        return event && event->activity.end == held.samples.size();
      },
      [&](bool* progress) {
        IREE_RETURN_IF_ERROR(Feed(&held, progress));
        const auto* event =
            loom_serve_recognition_stream_peek(held.stream.get());
        if (event && event->activity.end != held.samples.size()) {
          return Drain(&held, progress);
        }
        return iree_ok_status();
      }));
  IREE_ASSERT_OK(loom_serve_recognition_stream_status(held.stream.get()));
  const auto* final = loom_serve_recognition_stream_peek(held.stream.get());
  ASSERT_NE(final, nullptr);
  ASSERT_EQ(final->kind, LOOM_SERVE_RECOGNITION_EVENT_ACTIVITY);
  EXPECT_FALSE(loom_serve_recognition_stream_is_complete(held.stream.get()));
  const auto saved = final->activity;
  IREE_ASSERT_OK(loom_serve_registry_remove(Model(), LOOM_SERVE_REMOVAL_DRAIN));
  IREE_ASSERT_OK(Until([&] { return !Model()->service.name.size; },
                       [](bool*) { return iree_ok_status(); }));
  EXPECT_EQ(final, loom_serve_recognition_stream_peek(held.stream.get()));
  EXPECT_EQ(final->activity.begin, saved.begin);
  EXPECT_EQ(final->activity.end, saved.end);
  EXPECT_EQ(final->activity.speech_probability, saved.speech_probability);
  ASSERT_NO_FATAL_FAILURE(Register(FLAG_model, 0));
  IREE_ASSERT_OK(
      Until([&] { return Model()->phase == LOOM_SERVE_REGISTRATION_READY; },
            [](bool*) { return iree_ok_status(); }));
  loom_serve_activity_test_stream_t replacement;
  replacement.samples = samples_;
  ASSERT_NO_FATAL_FAILURE(Run(&replacement));
  EXPECT_EQ(peer.probabilities, replacement.probabilities);
  EXPECT_EQ(final, loom_serve_recognition_stream_peek(held.stream.get()));
  EXPECT_EQ(final->activity.speech_probability, saved.speech_probability);
  loom_serve_recognition_stream_consume(held.stream.get());
  EXPECT_TRUE(loom_serve_recognition_stream_is_complete(held.stream.get()));
}

TEST_F(NativeActivityTest, ActivityChoosesTranscriptionInput) {
  if (model_count_ < 2) {
    GTEST_SKIP() << "Requires a recognizer deployment";
  }
  std::vector<float> samples = samples_;
  if (FLAG_fixture) {
    samples.assign(181, 0.0f);
    std::fill(samples.begin() + 36, samples.begin() + 93, 0.25f);
  }
  std::string baseline;
  ASSERT_NO_FATAL_FAILURE(Transcribe(samples, &baseline));
  EXPECT_EQ(baseline, FLAG_fixture ? "O" : FLAG_expected_text);
  std::vector<loom_serve_activity_test_utterance_t> utterances;
  size_t retained = 0;
  IREE_ASSERT_OK(RecognizeActivity(samples, &utterances, &retained));
  ASSERT_EQ(utterances.size(), 1u);
  EXPECT_EQ(utterances[0].text, baseline);
  EXPECT_EQ(utterances[0].begin, static_cast<uint64_t>(FLAG_expected_begin));
  EXPECT_EQ(utterances[0].end, static_cast<uint64_t>(FLAG_expected_end));
  EXPECT_LE(retained, static_cast<size_t>(FLAG_retained_frames));
  const std::vector<float> silence(samples.size(), 0.0f);
  std::vector<loom_serve_activity_test_utterance_t> silent_utterances;
  size_t silent_retained = 0;
  IREE_ASSERT_OK(
      RecognizeActivity(silence, &silent_utterances, &silent_retained));
  EXPECT_TRUE(silent_utterances.empty());
  EXPECT_LE(silent_retained, static_cast<size_t>(FLAG_retained_frames));
}

TEST_F(NativeActivityTest, FollowingUtteranceRetainsIndependentInput) {
  if (!FLAG_fixture) {
    GTEST_SKIP() << "Requires the two-utterance fixture";
  }
  std::vector<float> samples(117, 0.0f);
  std::fill(samples.begin() + 18, samples.begin() + 36, 0.5f);
  std::fill(samples.begin() + 96, samples.end(), 0.25f);
  std::vector<loom_serve_activity_test_utterance_t> utterances;
  size_t retained = 0;
  IREE_ASSERT_OK(RecognizeActivity(samples, &utterances, &retained));
  ASSERT_EQ(utterances.size(), 2u);
  const uint64_t begins[] = {6, 84};
  const uint64_t ends[] = {54, 117};
  const char* expected[] = {"J", "F"};
  for (size_t i = 0; i < utterances.size(); ++i) {
    EXPECT_EQ(utterances[i].begin, begins[i]);
    EXPECT_EQ(utterances[i].end, ends[i]);
    std::string baseline;
    ASSERT_NO_FATAL_FAILURE(
        Transcribe(std::vector<float>(samples.begin() + begins[i],
                                      samples.begin() + ends[i]),
                   &baseline));
    EXPECT_EQ(utterances[i].text, baseline);
    EXPECT_EQ(utterances[i].text, expected[i]);
  }
  EXPECT_LE(retained, static_cast<size_t>(FLAG_retained_frames));
}

}  // namespace

int loom_serve_activity_test_main(
    int argc, char** argv,
    const loom_serve_compiler_options_t* compiler_options,
    iree_hal_driver_registry_t* driver_registry) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  activity_environment =
      static_cast<ActivityEnvironment*>(::testing::AddGlobalTestEnvironment(
          new ActivityEnvironment(compiler_options, driver_registry)));
  return RUN_ALL_TESTS();
}
