// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/media_projector.h"

#include <array>
#include <initializer_list>
#include <string>

#include "iree/async/util/proactor_pool.h"
#include "loom/serve/service/text/image_service_test_fixture.h"

namespace {

using loom_serve::testing::text::ImageComponentTextServiceTest;

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

}  // namespace
