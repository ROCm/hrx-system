// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/media_projector.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <string>

#include "iree/async/util/proactor_pool.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::ImageComponentTextServiceTest;
using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

static std::string Generate(Oracle& oracle, iree_host_size_t limit) {
  std::string result;
  for (iree_host_size_t i = 0; i < limit; ++i) {
    const int32_t token = oracle.Predict();
    if (token == 3) {
      break;
    }
    result.push_back("abx"[token]);
    oracle.Consume(token);
  }
  return result;
}

class MediaProjectorTest : public ImageComponentTextServiceTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(ImageComponentTextServiceTest::SetUp());
    iree_async_proactor_t* proactor = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_pool_get(
        loom_serve_host_proactor_pool(host_), 0, &proactor));
    IREE_ASSERT_OK(loom_serve_text_media_projector_allocate(
        loom_serve_host_executor(host_), proactor, image_component_,
        host_allocator_, &projector_));
    const loom_serve_wake_callback_t observer = {
        [](void* user_data) {
          iree_notification_post(static_cast<iree_notification_t*>(user_data),
                                 IREE_ALL_WAITERS);
        },
        loom_serve_http_server_notification(server_)};
    loom_serve_text_media_projector_set_observer(projector_, observer);
  }

  void TearDown() override {
    loom_serve_text_projected_media_deinitialize(&media_);
    loom_serve_text_input_deinitialize(&input_, host_allocator_);
    if (chat_initialized_) {
      loom_serve_text_chat_deinitialize(&chat_);
    }
    if (projector_) {
      loom_serve_text_media_projector_set_observer(projector_, {});
      loom_serve_text_media_projector_free(projector_);
    }
    ImageComponentTextServiceTest::TearDown();
  }

  void PrepareChat() {
    body_ = Request(TwoImageMessage(), 4);
    IREE_ASSERT_OK(loom_serve_text_chat_initialize(
        loom_serve_text_model_chat_policy(model_), IREE_SV("fixture-text"),
        iree_make_string_view(body_.data(), body_.size()), 1, 16,
        host_allocator_, &chat_));
    chat_initialized_ = true;
    std::array<int32_t, 96> scratch;
    IREE_ASSERT_OK(loom_serve_text_input_prepare(&chat_, {}, false,
                                                 scratch.size(), scratch.data(),
                                                 &input_, host_allocator_));
    ASSERT_EQ(input_.media_span_count, 2u);
  }

  void WaitForProgress() {
    auto* notification = loom_serve_http_server_notification(server_);
    for (;;) {
      const uint32_t ticket = iree_notification_prepare_wait(notification);
      bool progress =
          loom_serve_finite_component_advance(image_component_, false);
      progress |= loom_serve_text_media_projector_advance(
          projector_, /*shared_activity=*/false);
      if (progress || loom_serve_text_media_projector_is_complete(projector_)) {
        iree_notification_cancel_wait(notification);
        return;
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }

  void AdvanceProjector() {
    while (!loom_serve_text_media_projector_is_complete(projector_)) {
      WaitForProgress();
    }
  }

  void Take(bool expected_cancelled) {
    bool cancelled = false;
    IREE_ASSERT_OK(
        loom_serve_text_media_projector_take(projector_, &cancelled, &media_));
    EXPECT_EQ(cancelled, expected_cancelled);
  }

  void ExpectRows(std::initializer_list<uint32_t> image_indices,
                  std::initializer_list<std::array<uint32_t, 4>> rows) {
    ASSERT_EQ(media_.result_count, image_indices.size());
    ASSERT_EQ(media_.result_count, rows.size());
    const auto* input = loom_serve_text_projected_media_input(&media_);
    ASSERT_NE(input, nullptr);
    ASSERT_EQ(input->row_count, rows.size());
    iree_host_size_t ordinal = 0;
    auto expected_row = rows.begin();
    for (uint32_t image_index : image_indices) {
      SCOPED_TRACE(ordinal);
      EXPECT_EQ(media_.image_indices[ordinal], image_index);
      ASSERT_EQ(input->rows[ordinal].data_length, 16u);
      for (iree_host_size_t element = 0; element < 4; ++element) {
        EXPECT_EQ(iree_unaligned_load_le_u32(input->rows[ordinal].data +
                                             element * sizeof(uint32_t)),
                  (*expected_row)[element]);
      }
      ++ordinal;
      ++expected_row;
    }
  }

  static std::string TwoImageMessage() {
    static constexpr char kSecondPng[] =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEUDf+cz9ZFh"
        "AAAACklEQVQI12NgAAAAAgAB4iG8MwAAAABJRU5ErkJggg==";
    return std::string(
               R"({"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,)") +
           kPng +
           R"(","detail":"low"}},{"type":"text","text":"b"},{"type":"image_url","image_url":{"url":"data:image/png;base64,)" +
           kSecondPng + R"(","detail":"low"}},{"type":"text","text":"a"}]})";
  }

  // Owned serial request projector borrowing image_component_ and chat_.
  loom_serve_text_media_projector_t* projector_ = nullptr;
  // Request body backing image URL views through chat deinitialization.
  std::string body_;
  // Owned source-validated chat request.
  loom_serve_text_chat_t chat_ = {};
  // Whether chat_ owns initialized request state.
  bool chat_initialized_ = false;
  // Owned prepared text and media-span input.
  loom_serve_text_input_t input_ = {};
  // Owned detached projected results taken from projector_.
  loom_serve_text_projected_media_t media_ = {};
};

TEST_F(MediaProjectorTest,
       PrefixAndCancellationAvoidDemandBeforeSerialPixelProjection) {
  ASSERT_NO_FATAL_FAILURE(PrepareChat());

  IREE_ASSERT_OK(loom_serve_text_media_projector_begin(projector_, &chat_,
                                                       &input_, input_.count));
  EXPECT_FALSE(loom_serve_text_media_projector_is_active(projector_));
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_),
            LOOM_SERVE_FINITE_COMPONENT_DECLARED);

  IREE_ASSERT_OK(
      loom_serve_text_media_projector_begin(projector_, &chat_, &input_, 0));
  loom_serve_text_media_projector_cancel(projector_);
  EXPECT_TRUE(loom_serve_text_media_projector_advance(
      projector_, /*shared_activity=*/false));
  ASSERT_TRUE(loom_serve_text_media_projector_is_complete(projector_));
  ASSERT_NO_FATAL_FAILURE(Take(/*expected_cancelled=*/true));
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_),
            LOOM_SERVE_FINITE_COMPONENT_DECLARED);

  const auto& first = input_.media_spans[0];
  const iree_host_size_t first_end = first.input_offset + first.input_count;
  IREE_ASSERT_OK(loom_serve_text_media_projector_begin(projector_, &chat_,
                                                       &input_, first_end));
  ASSERT_NO_FATAL_FAILURE(AdvanceProjector());
  ASSERT_NO_FATAL_FAILURE(Take(/*expected_cancelled=*/false));
  ASSERT_NO_FATAL_FAILURE(ExpectRows({1}, {{{3, 127, 231, 0}}}));
  loom_serve_text_projected_media_deinitialize(&media_);

  // Cancellation after CPU preparation and device submission retains every
  // borrow until queue completion, then discards the detached result.
  IREE_ASSERT_OK(
      loom_serve_text_media_projector_begin(projector_, &chat_, &input_, 0));
  EXPECT_TRUE(loom_serve_text_media_projector_advance(
      projector_, /*shared_activity=*/false));
  ASSERT_TRUE(loom_serve_text_media_projector_is_pending(projector_));
  ASSERT_NO_FATAL_FAILURE(WaitForProgress());
  loom_serve_text_media_projector_cancel(projector_);
  ASSERT_NO_FATAL_FAILURE(AdvanceProjector());
  ASSERT_NO_FATAL_FAILURE(Take(/*expected_cancelled=*/true));
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_),
            LOOM_SERVE_FINITE_COMPONENT_READY);

  IREE_ASSERT_OK(
      loom_serve_text_media_projector_begin(projector_, &chat_, &input_, 0));
  ASSERT_NO_FATAL_FAILURE(AdvanceProjector());
  ASSERT_NO_FATAL_FAILURE(Take(/*expected_cancelled=*/false));
  ASSERT_NO_FATAL_FAILURE(
      ExpectRows({0, 1}, {{{255, 0, 0, 0}}, {{3, 127, 231, 0}}}));
}

class ImageTextServiceTest : public ImageComponentTextServiceTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(ImageComponentTextServiceTest::SetUp());
    auto options = ServiceOptions();
    options.prefix_cache_capacity = 2;
    options.chunk_size = 4;
    IREE_ASSERT_OK(loom_serve_text_service_allocate(
        model_, &options, nullptr, host_allocator_, &service_));
    interface_ = loom_serve_text_service_interface(service_);
    Observe(interface_);
  }

  void TearDown() override {
    if (service_) {
      ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
          interface_,
          [&] {
            return !interface_.is_pending(interface_.self) &&
                   !interface_.is_active(interface_.self) &&
                   !interface_.snapshot(interface_.self).queued_requests;
          },
          LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING));
      interface_.set_observer(interface_.self, {});
      loom_serve_text_service_free(service_, IREE_STATUS_OK);
      service_ = nullptr;
    }
    ImageComponentTextServiceTest::TearDown();
  }

  std::string CompleteRequest(const std::string& body,
                              const std::string& headers = "") {
    const int client = Send(body, headers);
    loom_serve_http_connection_t* connection = nullptr;
    Accept(client, interface_, &connection);
    if (HasFatalFailure()) {
      return {};
    }
    AdvanceUntil(interface_, [&] {
      return !interface_.is_pending(interface_.self) &&
             !interface_.is_active(interface_.self) &&
             !interface_.snapshot(interface_.self).queued_requests;
    });
    if (HasFatalFailure()) {
      return {};
    }
    return Read(client);
  }

  // Owned integrated text service borrowing image_component_.
  loom_serve_text_service_t* service_ = nullptr;
  // Borrowed callbacks invoked by this serialized test owner.
  loom_serve_http_service_t interface_ = {};
};

TEST_F(ImageTextServiceTest,
       ProjectsPixelsOnceAndKeepsRetainedHistoryIndependent) {
  ASSERT_EQ(interface_.capabilities.input_modalities.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(
      interface_.capabilities.input_modalities.values[0], IREE_SV("text")));
  EXPECT_TRUE(iree_string_view_equal(
      interface_.capabilities.input_modalities.values[1], IREE_SV("image")));
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_),
            LOOM_SERVE_FINITE_COMPONENT_DECLARED);

  const std::string image = ImageMessage("a", "image/png", kPng, "low", "b");
  Oracle oracle;
  oracle.Consume(Oracle::Id('a'));
  oracle.Consume(255);
  oracle.Consume(Oracle::Id('b'));
  const std::string first_expected = Generate(oracle, 4);
  const std::string first = Content(CompleteRequest(
      Request(image, 4), "X-Loom-Session: projected-history\r\n"));
  EXPECT_EQ(first, first_expected);
  EXPECT_NE(first, Oracle::Complete("axb", 4));
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_),
            LOOM_SERVE_FINITE_COMPONENT_READY);

  loom_serve_finite_model_t* image_model = nullptr;
  IREE_ASSERT_OK(
      loom_serve_finite_component_prepare(image_component_, &image_model));
  ASSERT_NE(image_model, nullptr);
  const bool elastic = loom_serve_residency_snapshot(
                           loom_serve_finite_model_residency(image_model))
                           .elastic;
  const uint64_t compilation_commands =
      loom_serve_finite_component_report(image_component_)->commands;
  ASSERT_NO_FATAL_FAILURE(TrimImageComponent());
  const auto trimmed_phase =
      loom_serve_finite_component_phase(image_component_);
  EXPECT_EQ(trimmed_phase, elastic ? LOOM_SERVE_FINITE_COMPONENT_READY
                                   : LOOM_SERVE_FINITE_COMPONENT_DECLARED);
  if (elastic) {
    EXPECT_FALSE(loom_serve_residency_snapshot(
                     loom_serve_finite_model_residency(image_model))
                     .active);
  }

  oracle.Consume("#b");
  const std::string second_expected = Generate(oracle, 3);
  const std::string continuation =
      image + "," + Message("assistant", first) + "," + Message("user", "b");
  const std::string second_response = CompleteRequest(
      Request(continuation, 3), "X-Loom-Session: projected-history\r\n");
  EXPECT_EQ(Content(second_response), second_expected);
  EXPECT_NE(second_response.find(R"("cached_tokens":6})"), std::string::npos)
      << second_response;
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_), trimmed_phase);
  EXPECT_EQ(loom_serve_finite_component_report(image_component_)->commands,
            compilation_commands);
  if (elastic) {
    EXPECT_FALSE(loom_serve_residency_snapshot(
                     loom_serve_finite_model_residency(image_model))
                     .active);
  }

  EXPECT_EQ(Content(CompleteRequest(Request(Message("user", "abxa"), 2))),
            Oracle::Complete("abxa", 2));
  EXPECT_EQ(loom_serve_finite_component_phase(image_component_), trimmed_phase);
}

TEST_F(ImageTextServiceTest, InputFailureAndDisconnectAreRequestLocal) {
  const std::string invalid_image =
      ImageMessage("a", "image/png", "bm90LWltYWdl", "low", "b");
  const std::string response = CompleteRequest(Request(invalid_image, 2));
  EXPECT_EQ(response.find("HTTP/1.1 400"), 0u) << response;
  EXPECT_EQ(interface_.telemetry.counters->rejected, 1u);
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);

  const std::string image = ImageMessage("a", "image/png", kPng, "low", "b");
  const int client = Send(Request(image, 2));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(client, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return loom_serve_finite_component_phase(image_component_) !=
           LOOM_SERVE_FINITE_COMPONENT_DECLARED;
  }));
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(client, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(client), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connection, iree_infinite_timeout());
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return !interface_.is_pending(interface_.self) &&
           !interface_.is_active(interface_.self) &&
           !interface_.snapshot(interface_.self).queued_requests;
  }));
  EXPECT_EQ(interface_.telemetry.counters->rejected, 1u);
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 1u);
  EXPECT_EQ(interface_.telemetry.counters->failed, 0u);
  EXPECT_FALSE(loom_serve_finite_component_is_pending(image_component_));
  EXPECT_EQ(Content(CompleteRequest(Request(Message("user", "abxa"), 2))),
            Oracle::Complete("abxa", 2));
}

class ManagedImageTextServiceTest : public SourceTextServiceTest {};

TEST_F(ManagedImageTextServiceTest, LazilyProjectsAnImageThroughTheRouter) {
  StartManaged();
  const char* source_path = std::getenv("LOOM_SERVE_TEST_PROJECTOR_SOURCE");
  ASSERT_NE(source_path, nullptr);
  const std::filesystem::path projector_directory =
      std::filesystem::absolute(
          source_path)
          .parent_path();
  const std::string image_fields =
      ",\"image\":{\"prepare\":\"" +
      (projector_directory / "prepare.loom").string() + "\",\"control\":\"" +
      (projector_directory / "control.loom").string() + "\",\"assets\":\"" +
      projector_directory.string() + "\"}";
  std::string declaration =
      Declaration("fixture-text", directory_, "open", image_fields);
  const std::string source_list = "\"sources\":[\"advance.loom\"]";
  const size_t source_position = declaration.find(source_list);
  ASSERT_NE(source_position, std::string::npos);
  declaration.replace(source_position, source_list.size(),
                      "\"sources\":[\"advance.loom\",\"" +
                          (projector_directory / "project.loom").string() +
                          "\"]");
  const std::string accepted =
      Read(Send(declaration, "", "/v1/models/register"));
  ASSERT_EQ(accepted.find("HTTP/1.1 202"), 0u) << accepted;
  const std::string ready =
      WaitModel("fixture-text", R"("registration":"ready")");
  EXPECT_NE(ready.find(R"("input_modalities":["text","image"])"),
            std::string::npos)
      << ready;

  Oracle oracle;
  oracle.Consume(Oracle::Id('a'));
  oracle.Consume(255);
  oracle.Consume(Oracle::Id('b'));
  const std::string message = ImageMessage("a", "image/png", kPng, "low", "b");
  EXPECT_EQ(Content(Read(Send(Request(message, 3)))), Generate(oracle, 3));

  const std::string removed =
      Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/unregister"));
  ASSERT_EQ(removed.find("HTTP/1.1 202"), 0u) << removed;
  WaitModel("fixture-text", {});
}

}  // namespace
