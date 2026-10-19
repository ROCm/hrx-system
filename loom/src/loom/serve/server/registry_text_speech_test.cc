// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

#include "iree/base/threading/notification.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/server/configuration.h"
#include "loom/serve/server/model_registration.h"
#include "loom/serve/server/registry.h"
#include "loom/serve/server/resources.h"
#include "loom/serve/service/service.h"
#include "loom/serve/service/speech/output.h"
#include "loom/serve/service/speech/request.h"
#include "loom/serve/service/speech/stream.h"
#include "loom/serve/service/text/stream.h"
#include "loom/serve/testing/service_application.h"

IREE_FLAG(string, text_model, "", "Text model document.");
IREE_FLAG(string, speech_model, "", "Speech model document.");
IREE_FLAG(string, text_request, "", "Streaming chat request file.");
IREE_FLAG(string, voice, "positive", "Speech voice.");
IREE_FLAG(int32_t, test_workers, 1, "Shared application worker count.");
IREE_FLAG(int32_t, seed, 42, "Speech seed.");
IREE_FLAG(bool, text_fixture, false, "Generate fixture weights/request.");

namespace {
using Text = std::unique_ptr<loom_serve_text_stream_t,
                             decltype(&loom_serve_text_stream_close)>;
using Speech = std::unique_ptr<loom_serve_speech_stream_t,
                               decltype(&loom_serve_speech_stream_close)>;

std::string String(iree_string_view_t view) { return {view.data, view.size}; }

std::string Read(const char* path) {
  std::ifstream input(path, std::ios::binary);
  EXPECT_TRUE(input.good()) << path;
  return {std::istreambuf_iterator<char>(input), {}};
}

struct loom_serve_text_speech_result_t {
  // Complete source-classified text, accumulated only for the oracle.
  std::string content;
  // Complete source-classified reasoning, never forwarded to speech.
  std::string reasoning;
  // Copied PCM16LE oracle output.
  std::string pcm;
};

class RegistryTextSpeechTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    auto options = loom_serve_test_application_options_default();
    options.worker_count = FLAG_test_workers;
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(&options, allocator_,
                                                       &application));
    registration_.device_group =
        loom_serve_test_application_device_group(application);
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        0, nullptr, 2, loom_serve_model_registration_factory(&registration_),
        allocator_, &registry_));
    ASSERT_NO_FATAL_FAILURE(Register(FLAG_text_model, true));
    ASSERT_NO_FATAL_FAILURE(Register(FLAG_speech_model, false));
    if (FLAG_text_fixture) {
      request_ =
          R"({"model":"fixture-text","stream":true,"max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"a"}]})";
    } else {
      request_ = Read(FLAG_text_request);
    }
    IREE_ASSERT_OK(
        Until([&] { return Ready(); }, [](bool*) { return iree_ok_status(); }));
  }

  void TearDown() override {
    if (registry_) {
      for (size_t i = 0; i < 2; ++i) {
        auto* model = loom_serve_registry_at(registry_, i);
        if (model->service.name.size) {
          IREE_EXPECT_OK(
              loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT));
        }
      }
      IREE_EXPECT_OK(
          Until([&] { return loom_serve_registry_count(registry_) == 0; },
                [](bool*) { return iree_ok_status(); }));
      IREE_EXPECT_OK(loom_serve_registry_free(registry_));
    }
    iree_notification_deinitialize(&notification_);
  }

  void Register(const char* path, bool text) {
    std::string weights;
    if (text && FLAG_text_fixture) {
      weights = iree::testing::MakeTempFilePath("registry-text-speech",
                                                ".safetensors");
      std::string header =
          R"({"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})";
      header.append((8 - header.size() % 8) % 8, ' ');
      std::ofstream output(weights, std::ios::binary);
      uint8_t bytes[8];
      iree_unaligned_store_le_u64(bytes, header.size());
      output.write(reinterpret_cast<const char*>(bytes), 8);
      output.write(header.data(), header.size());
      iree_unaligned_store_le_u32(bytes, 2);
      output.write(reinterpret_cast<const char*>(bytes), 4);
      output.close();
      ASSERT_TRUE(output.good());
    }
    loom_serve_model_configuration_t configuration;
    IREE_ASSERT_OK(loom_serve_model_configuration_load(
        iree_make_cstring_view(path), allocator_, &configuration));
    auto status = iree_ok_status();
    if (!weights.empty()) {
      void* owned_path = nullptr;
      status = iree_allocator_clone(
          allocator_, iree_make_const_byte_span(weights.data(), weights.size()),
          &owned_path);
      if (iree_status_is_ok(status)) {
        iree_allocator_free(allocator_, (void*)configuration.text.weights.data);
        configuration.text.weights = iree_make_string_view(
            static_cast<const char*>(owned_path), weights.size());
      }
    }
    names_[text ? 0 : 1] = String(configuration.name);
    iree_host_size_t index = 0;
    if (iree_status_is_ok(status)) {
      status = loom_serve_registry_register(registry_, &configuration, &index);
    }
    loom_serve_model_configuration_deinitialize(&configuration);
    IREE_ASSERT_OK(status);
  }

  bool Ready() const {
    return loom_serve_registry_at(registry_, 0)->phase ==
               LOOM_SERVE_REGISTRATION_READY &&
           loom_serve_registry_at(registry_, 1)->phase ==
               LOOM_SERVE_REGISTRATION_READY;
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
      for (size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
        const size_t index = (cursor_ + i) % 2;
        auto* model = loom_serve_registry_at(registry_, index);
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
        status = iree_status_join(status, iree_status_clone(model->status));
      }
      cursor_ ^= 1;
      if (iree_status_is_ok(status)) {
        const loom_serve_resources_t resources = {
            .device_group = registration_.device_group,
        };
        progress |= loom_serve_resources_schedule(&resources);
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

  iree_status_t OpenText(Text* result) {
    loom_serve_text_request_t request = {};
    iree_string_builder_initialize(allocator_, &request.body);
    auto status = iree_string_builder_append_string(
        &request.body, iree_make_cstring_view(request_.c_str()));
    if (iree_status_is_ok(status)) {
      status = Until([&] { return bool(*result); },
                     [&](bool* progress) {
                       loom_serve_text_stream_t* stream = nullptr;
                       auto admitted = loom_serve_registry_try_open_text(
                           registry_, iree_make_cstring_view(names_[0].c_str()),
                           &request, &stream);
                       result->reset(stream);
                       *progress |= stream != nullptr;
                       return admitted;
                     });
    }
    iree_string_builder_deinitialize(&request.body);
    return status;
  }

  iree_status_t OpenSpeech(Speech* result) {
    loom_serve_speech_request_t request;
    auto status = loom_serve_speech_request_initialize_stream(
        iree_make_cstring_view(FLAG_voice), {}, FLAG_seed, 1.f, allocator_,
        &request);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    status = Until([&] { return bool(*result); },
                   [&](bool* progress) {
                     loom_serve_speech_stream_t* stream = nullptr;
                     auto admitted = loom_serve_registry_try_open_speech(
                         registry_, iree_make_cstring_view(names_[1].c_str()),
                         &request, &stream);
                     result->reset(stream);
                     *progress |= stream != nullptr;
                     return admitted;
                   });
    loom_serve_speech_request_deinitialize(&request);
    return status;
  }

  iree_status_t Run(bool continuing, loom_serve_text_speech_result_t* result) {
    Text text(nullptr, loom_serve_text_stream_close);
    Speech speech(nullptr, loom_serve_speech_stream_close);
    IREE_RETURN_IF_ERROR(OpenText(&text));
    if (continuing) {
      IREE_RETURN_IF_ERROR(OpenSpeech(&speech));
    }
    size_t span_index = 0, byte_offset = 0;
    bool recorded_span = false, finished = false;
    loom_serve_speech_request_t finite_request = {};
    auto status = Until(
        [&] {
          return speech && loom_serve_speech_stream_is_complete(speech.get());
        },
        [&](bool* progress) -> iree_status_t {
          IREE_RETURN_IF_ERROR(loom_serve_text_stream_status(text.get()));
          if (speech) {
            IREE_RETURN_IF_ERROR(loom_serve_speech_stream_status(speech.get()));
          }
          if (const auto* block = loom_serve_text_stream_peek(text.get())) {
            while (span_index < loom_serve_text_stream_block_count(block)) {
              const auto span =
                  loom_serve_text_stream_block_at(block, span_index);
              const bool content =
                  span.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT;
              if (!recorded_span) {
                (content ? result->content : result->reasoning)
                    .append(span.text.data, span.text.size);
                recorded_span = true;
              }
              if (continuing && content && byte_offset < span.text.size) {
                iree_host_size_t count = 0;
                IREE_RETURN_IF_ERROR(loom_serve_speech_stream_write(
                    speech.get(),
                    iree_make_const_byte_span(span.text.data + byte_offset,
                                              span.text.size - byte_offset),
                    &count));
                byte_offset += count;
                *progress |= count != 0;
                if (byte_offset < span.text.size) {
                  break;
                }
              }
              ++span_index;
              byte_offset = 0;
              recorded_span = false;
            }
            if (span_index == loom_serve_text_stream_block_count(block)) {
              loom_serve_text_stream_consume(text.get());
              span_index = 0;
              *progress = true;
            }
          }
          if (!finished && loom_serve_text_stream_is_complete(text.get())) {
            const auto* final = loom_serve_text_stream_result(text.get());
            if (!final) {
              return iree_make_status(IREE_STATUS_DATA_LOSS,
                                      "missing successful text result");
            }
            EXPECT_EQ(String(final->content), result->content);
            EXPECT_EQ(String(final->reasoning), result->reasoning);
            EXPECT_EQ(final->response.usage.cached_tokens, 0u);
            if (continuing) {
              IREE_RETURN_IF_ERROR(
                  loom_serve_speech_stream_finish(speech.get()));
            } else {
              IREE_RETURN_IF_ERROR(loom_serve_speech_request_initialize_text(
                  iree_make_string_view(result->content.data(),
                                        result->content.size()),
                  iree_make_cstring_view(FLAG_voice), {}, FLAG_seed, 1.f,
                  allocator_, &finite_request));
            }
            finished = true;
            *progress = true;
          }
          if (finished && !speech) {
            loom_serve_speech_stream_t* admitted = nullptr;
            IREE_RETURN_IF_ERROR(loom_serve_registry_try_open_speech(
                registry_, iree_make_cstring_view(names_[1].c_str()),
                &finite_request, &admitted));
            speech.reset(admitted);
            *progress |= admitted != nullptr;
          }
          if (speech) {
            if (const auto* block =
                    loom_serve_speech_stream_peek(speech.get())) {
              const auto format = loom_serve_speech_stream_format(speech.get());
              EXPECT_EQ(
                  block->position,
                  result->pcm.size() / (sizeof(int16_t) * format.channels));
              iree_string_builder_t pcm;
              iree_string_builder_initialize(allocator_, &pcm);
              auto encoded =
                  loom_serve_speech_encode_pcm16(block->samples, &pcm);
              if (iree_status_is_ok(encoded)) {
                result->pcm += String(iree_string_builder_view(&pcm));
              }
              iree_string_builder_deinitialize(&pcm);
              IREE_RETURN_IF_ERROR(encoded);
              loom_serve_speech_stream_consume(speech.get());
              *progress = true;
            }
          }
          return iree_ok_status();
        });
    loom_serve_speech_request_deinitialize(&finite_request);
    status =
        iree_status_join(status, loom_serve_text_stream_status(text.get()));
    if (speech) {
      status = iree_status_join(status,
                                loom_serve_speech_stream_status(speech.get()));
    }
    return status;
  }

  // Single host allocator for the production owners.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Owner wake shared by every model's real asynchronous retirement.
  iree_notification_t notification_;
  // Factory capabilities live until every model is removed.
  loom_serve_model_registration_options_t registration_ = {};
  // Two stable native model slots.
  loom_serve_registry_t* registry_ = nullptr;
  // Text and speech deployment selectors.
  std::array<std::string, 2> names_;
  // Next first model for rotating bounded service turns.
  size_t cursor_ = 0;
  // Complete chat input, copied only by native text admission.
  std::string request_;
};

TEST_F(RegistryTextSpeechTest,
       CompleteAndContinuingInputProduceIdenticalAudio) {
  loom_serve_text_speech_result_t finite, continuing;
  IREE_ASSERT_OK(Run(false, &finite));
  ASSERT_FALSE(finite.content.empty());
  ASSERT_FALSE(finite.pcm.empty());
  IREE_ASSERT_OK(Run(true, &continuing));
  EXPECT_EQ(continuing.content, finite.content);
  EXPECT_EQ(continuing.reasoning, finite.reasoning);
  EXPECT_EQ(continuing.pcm, finite.pcm);
}

TEST_F(RegistryTextSpeechTest, HeldCreditRetiresWithoutBlockingPeer) {
  Text text(nullptr, loom_serve_text_stream_close);
  Speech speech(nullptr, loom_serve_speech_stream_close);
  IREE_ASSERT_OK(OpenText(&text));
  IREE_ASSERT_OK(OpenSpeech(&speech));
  size_t span_index = 0, byte_offset = 0, accepted = 0;
  auto transfer = [&](size_t limit, bool* progress) -> iree_status_t {
    IREE_RETURN_IF_ERROR(loom_serve_text_stream_status(text.get()));
    IREE_RETURN_IF_ERROR(loom_serve_speech_stream_status(speech.get()));
    if (const auto* block = loom_serve_text_stream_peek(text.get())) {
      while (span_index < loom_serve_text_stream_block_count(block)) {
        const auto span = loom_serve_text_stream_block_at(block, span_index);
        if (span.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT &&
            byte_offset < span.text.size) {
          const size_t length =
              std::min(span.text.size - byte_offset, limit - accepted);
          iree_host_size_t count = 0;
          IREE_RETURN_IF_ERROR(loom_serve_speech_stream_write(
              speech.get(),
              iree_make_const_byte_span(span.text.data + byte_offset, length),
              &count));
          accepted += count;
          byte_offset += count;
          *progress |= count != 0;
          if (byte_offset < span.text.size) {
            break;
          }
        }
        ++span_index;
        byte_offset = 0;
      }
      if (span_index == loom_serve_text_stream_block_count(block)) {
        loom_serve_text_stream_consume(text.get());
        span_index = 0;
        *progress = true;
      }
    }
    return iree_ok_status();
  };
  // The tiny fixture emits after its first scalar. Hold that exact cut so the
  // remaining answer exceeds its seven-byte credit independently of timing.
  // The learned source needs a real packet prefix and owns that policy itself.
  IREE_ASSERT_OK(Until(
      [&] {
        return loom_serve_speech_stream_peek(speech.get()) ||
               loom_serve_text_stream_is_complete(text.get());
      },
      [&](bool* progress) {
        return transfer(FLAG_text_fixture ? 1 : SIZE_MAX, progress);
      }));
  const auto* held_audio = loom_serve_speech_stream_peek(speech.get());
  ASSERT_NE(held_audio, nullptr);
  ASSERT_GT(held_audio->frame_count, 0u);
  ASSERT_FALSE(loom_serve_text_stream_is_complete(text.get()));
  IREE_ASSERT_OK(Until(
      [&] {
        return (loom_serve_speech_stream_write_credit(speech.get()) == 0 &&
                loom_serve_text_stream_peek(text.get())) ||
               loom_serve_text_stream_is_complete(text.get());
      },
      [&](bool* progress) { return transfer(SIZE_MAX, progress); }));
  ASSERT_EQ(loom_serve_speech_stream_write_credit(speech.get()), 0u);
  const auto* held_text = loom_serve_text_stream_peek(text.get());
  ASSERT_NE(held_text, nullptr);
  ASSERT_LT(span_index, loom_serve_text_stream_block_count(held_text));
  const auto span = loom_serve_text_stream_block_at(held_text, span_index);
  ASSERT_LT(byte_offset, span.text.size);
  const auto held_bytes = String(span.text);
  const auto held_samples =
      std::string(reinterpret_cast<const char*>(held_audio->samples.data),
                  held_audio->samples.data_length);

  Text peer(nullptr, loom_serve_text_stream_close);
  IREE_ASSERT_OK(OpenText(&peer));
  std::string peer_content;
  IREE_ASSERT_OK(
      Until([&] { return loom_serve_text_stream_is_complete(peer.get()); },
            [&](bool* progress) -> iree_status_t {
              IREE_RETURN_IF_ERROR(loom_serve_text_stream_status(peer.get()));
              if (const auto* block = loom_serve_text_stream_peek(peer.get())) {
                for (size_t i = 0;
                     i < loom_serve_text_stream_block_count(block); ++i) {
                  const auto value = loom_serve_text_stream_block_at(block, i);
                  if (value.kind == LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT) {
                    peer_content += String(value.text);
                  }
                }
                loom_serve_text_stream_consume(peer.get());
                *progress = true;
              }
              return iree_ok_status();
            }));
  ASSERT_FALSE(peer_content.empty());
  IREE_ASSERT_OK(loom_serve_text_stream_status(peer.get()));
  ASSERT_NE(loom_serve_text_stream_result(peer.get()), nullptr);
  EXPECT_EQ(String(loom_serve_text_stream_result(peer.get())->content),
            peer_content);
  EXPECT_EQ(loom_serve_text_stream_peek(text.get()), held_text);
  EXPECT_EQ(String(loom_serve_text_stream_block_at(held_text, span_index).text),
            held_bytes);
  EXPECT_EQ(loom_serve_speech_stream_peek(speech.get()), held_audio);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(held_audio->samples.data),
                        held_audio->samples.data_length),
            held_samples);
  EXPECT_EQ(loom_serve_speech_stream_write_credit(speech.get()), 0u);
  text.reset();
  speech.reset();
  IREE_ASSERT_OK(Until(
      [&] {
        for (size_t i = 0; i < 2; ++i) {
          const auto& service = loom_serve_registry_at(registry_, i)->service;
          if (service.is_pending(service.self) ||
              service.is_active(service.self) ||
              service.snapshot(service.self).queued_requests) {
            return false;
          }
        }
        return true;
      },
      [](bool*) { return iree_ok_status(); }));
  for (size_t i = 0; i < 2; ++i) {
    const auto* counters =
        loom_serve_registry_at(registry_, i)->service.telemetry.counters;
    EXPECT_EQ(counters->cancelled, 1u);
    EXPECT_EQ(counters->failed, 0u);
  }
}
}  // namespace
