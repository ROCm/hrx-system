// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdlib>
#include <cstring>

#include "loom/serve/media/image_request.h"
#include "loom/serve/service/finite/model.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class PixelProjectorExecutionTest : public SourceTextServiceTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    const char* source = std::getenv("LOOM_SERVE_TEST_PROJECTOR_SOURCE");
    ASSERT_NE(source, nullptr);
    IREE_ASSERT_OK(projector_document_.Load(source));
    const loom_serve_finite_model_options_t options = {
        .sources = projector_document_.value().sources,
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{}"),
        .specialization_cache = {2, 1024 * 1024},
    };
    IREE_ASSERT_OK(loom_serve_finite_model_allocate(
        device_, &options, host_allocator_, &projector_));
    loom_serve_finite_model_set_observer(
        projector_, {[](void* data) {
                       iree_notification_post(
                           static_cast<iree_notification_t*>(data),
                           IREE_ALL_WAITERS);
                     },
                     loom_serve_http_server_notification(server_)});
  }

  void TearDown() override {
    SourceTextServiceTest::TearDown();
    IREE_EXPECT_OK(loom_serve_finite_model_free(projector_));
    loom_serve_finite_result_release(result_);
    loom_serve_finite_snapshot_release(snapshot_);
    for (size_t i = 0; i < inputs_.size(); ++i) {
      iree_vm_buffer_release(prepared_[i].bytes);
      iree_vm_buffer_release(packets_[i]);
      loom_serve_media_image_input_deinitialize(&inputs_[i]);
    }
  }

  void Wait(loom_serve_execution_frontier_t ready) {
    for (size_t i = 0; i < IREE_ARRAYSIZE(ready.semaphores); ++i) {
      IREE_ASSERT_OK(
          loom_serve_workspace_wait(loom_serve_device_workspace(device_),
                                    ready.semaphores[i], ready.values[i]));
    }
  }

  static int32_t Fingerprint(float value, uint32_t image_row) {
    int32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits + image_row;
  }

  void ResolveCode() {
    auto* notification = loom_serve_http_server_notification(server_);
    bool complete = false;
    while (!complete) {
      const auto ticket = iree_notification_prepare_wait(notification);
      auto status =
          loom_serve_finite_model_advance_batch(projector_, false, &complete);
      if (complete || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(notification);
        IREE_ASSERT_OK(status);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }

  // Source ownership outlives asynchronous finite model work.
  loom_serve::testing::ModelDocument projector_document_;
  // Independent image model sharing the actual text device and workspace.
  loom_serve_finite_model_t* projector_ = nullptr;
  // Immutable compressed images retained only through packet realization.
  std::array<loom_serve_media_image_input_t, 2> inputs_ = {};
  // Real source input packets, not test-authored decoded pixel arrays.
  std::array<iree_vm_buffer_t*, 2> packets_ = {};
  // Source-produced batch inputs retaining original image ordinals.
  std::array<loom_serve_prepared_request_t, 2> prepared_ = {};
  // Device-only output supplied directly to the language consumer.
  loom_serve_finite_result_t* result_ = nullptr;
  // Final test observation, allocated only after the language result exists.
  loom_serve_finite_snapshot_t* snapshot_ = nullptr;
};

TEST_F(PixelProjectorExecutionTest, RgbAndRgbaCohortConditionsRetainedText) {
  const std::string urls[] = {
      std::string("data:image/x-portable-arbitrarymap;base64,") + kPam,
      std::string("data:image/x-portable-arbitrarymap;base64,") + kAlphaPam,
  };
  const loom_serve_media_image_limits_t limits = {
      .compressed_bytes = 2048,
      .width = 64,
      .height = 64,
      .pixels = 4096,
  };
  for (size_t i = 0; i < inputs_.size(); ++i) {
    auto status = loom_serve_media_image_input_initialize(
        iree_make_string_view(urls[i].data(), urls[i].size()), &limits,
        host_allocator_, &inputs_[i]);
    if (iree_status_is_unimplemented(status)) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
      return;
    }
    IREE_ASSERT_OK(status);
  }
  const uint8_t metadata[] = {9, 8, 7};
  const loom_serve_media_image_request_t requests[] = {
      {&inputs_[0], {}, 3},
      {&inputs_[1], iree_make_const_byte_span(metadata, sizeof(metadata)), 2},
  };
  iree_status_t statuses[2];
  IREE_ASSERT_OK(loom_serve_media_image_request_create_batch(
      2, requests, host_allocator_, packets_.data(), statuses));
  for (size_t i = 0; i < inputs_.size(); ++i) {
    IREE_ASSERT_OK(statuses[i]);
    loom_serve_media_image_input_deinitialize(&inputs_[i]);
    IREE_ASSERT_OK(loom_serve_finite_model_prepare_request(
        projector_, packets_[i], &prepared_[i]));
    EXPECT_EQ(prepared_[i].units, requests[i].output_rows);
  }
  const iree_host_size_t expected_rows[] = {3, 2};
  IREE_ASSERT_OK(loom_serve_finite_model_prepare_batch(
      projector_, prepared_.size(), prepared_.data(), expected_rows));
  ASSERT_NO_FATAL_FAILURE(ResolveCode());
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(
      loom_serve_finite_model_residency(projector_), &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_execution_frontier_t ready;
  IREE_ASSERT_OK(loom_serve_finite_model_submit(projector_, &result_, &ready));
  EXPECT_EQ(ready.values[1], 0u);
  const loom_serve_text_media_input_span_t images[] = {
      {
          .input_offset = 0,
          .input_count = 3,
          .request_index = 0,
          .result = result_,
      },
      {
          .input_offset = 1,
          .input_count = 2,
          .request_index = 1,
          .result = result_,
      },
  };
  const loom_serve_text_media_input_t media[] = {{images, 1}, {images + 1, 1}};
  const int32_t tokens[][4] = {{0, 0, 0, 1}, {2, 0, 0, 1}};
  const loom_serve_text_span_t spans[] = {
      {
          .row_index = 0,
          .token_count = 4,
          .token_ids = tokens[0],
          .flags = LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
          .media = &media[0],
      },
      {
          .row_index = 1,
          .token_count = 4,
          .token_ids = tokens[1],
          .flags = LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
          .media = &media[1],
      },
  };
  // No producer host wait or feature snapshot precedes text submission.
  IREE_ASSERT_OK(loom_serve_text_model_submit_epoch(
      model_, 1, 2, spans, nullptr, nullptr, nullptr, &ready));
  ASSERT_NO_FATAL_FAILURE(Wait(ready));
  loom_serve_text_result_t results[2];
  IREE_ASSERT_OK(loom_serve_text_model_complete_epoch(model_, results));
  const int32_t expected[][4] = {
      {Fingerprint(512, 0), Fingerprint(513, 1), Fingerprint(514, 2), 1},
      {2, Fingerprint(12, 0), Fingerprint(13, 1), 1},
  };
  for (size_t i = 0; i < 2; ++i) {
    Oracle oracle;
    for (int32_t value : expected[i]) {
      oracle.Consume(value);
    }
    EXPECT_EQ(results[i].known_count, 4u);
    auto* row = loom_serve_text_model_row(model_, i);
    EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());
    oracle.Consume(oracle.Predict());
    IREE_ASSERT_OK(loom_serve_text_row_decode(row));
    EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());
  }
  // Observe the projector only after text has consumed its device result.
  IREE_ASSERT_OK(
      loom_serve_finite_result_download(result_, &snapshot_, &ready));
  ASSERT_NO_FATAL_FAILURE(Wait(ready));
  loom_serve_finite_model_complete(projector_);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t row = 0; row < expected_rows[i]; ++row) {
      const auto bytes = loom_serve_finite_snapshot_row(snapshot_, i, row);
      ASSERT_EQ(bytes.data_length, 4 * sizeof(float));
      float values[4];
      std::memcpy(values, bytes.data, sizeof(values));
      EXPECT_EQ(values[0], (i ? 12.f : 512.f) + row);
      EXPECT_EQ(values[1], i ? 20.f : 0.f);
      EXPECT_EQ(values[2], i ? 30.f : 0.f);
      EXPECT_EQ(values[3], 2.f);
    }
  }
}

}  // namespace
