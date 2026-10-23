// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/media_projector.h"

#include <array>
#include <cstdlib>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/service_application.h"

namespace {

TEST(MediaRequestTest, BindsOptionalMetadataAtItsImageOffset) {
  const auto host_allocator = iree_allocator_system();
  for (const auto prefix : {IREE_SV(""), IREE_SV("earlier-image")}) {
    for (const auto metadata : {IREE_SV(""), IREE_SV("image-metadata")}) {
      loom_serve_text_chat_t chat = {};
      loom_serve_text_chat_image_t image = {};
      iree_string_builder_initialize(host_allocator, &chat.images.metadata);
      if (prefix.size) {
        IREE_ASSERT_OK(
            iree_string_builder_append_string(&chat.images.metadata, prefix));
      }
      if (metadata.size) {
        IREE_ASSERT_OK(
            iree_string_builder_append_string(&chat.images.metadata, metadata));
      }
      image.metadata_offset = prefix.size;
      image.metadata_length = metadata.size;
      chat.images.count = 1;
      chat.images.values = &image;
      loom_serve_text_media_span_t span = {.input_count = 1};
      const loom_serve_text_input_t input = {
          .media_spans = &span,
          .media_span_count = 1,
      };
      loom_serve_text_media_request_t* request = nullptr;
      IREE_ASSERT_OK(loom_serve_text_media_request_allocate(
          &chat, host_allocator, &request));
      EXPECT_FALSE(loom_serve_text_media_request_bind(request, &input, 0));
      const auto* bound = loom_serve_text_media_request_input(request);
      ASSERT_EQ(bound->span_count, 1u);
      const auto bytes = bound->spans[0].metadata;
      EXPECT_EQ(bytes.data_length, metadata.size);
      if (metadata.size) {
        EXPECT_EQ(std::memcmp(bytes.data, metadata.data, metadata.size), 0);
      } else {
        EXPECT_EQ(bytes.data, nullptr);
      }
      loom_serve_text_media_request_free(request);
      iree_string_builder_deinitialize(&chat.images.metadata);
    }
  }
}

class MediaProjectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    iree_string_builder_initialize(host_allocator_, &chat_.images.metadata);
    auto options = loom_serve_test_application_options_default();
    options.backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC;
    options.slab_size = 4096;
    options.memory_limit = 16 * 1024;
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &options, host_allocator_, &application));
    auto* host = loom_serve_test_application_host(application);
    device_ = loom_serve_test_application_device_at(application, 0);
    const char* source = std::getenv("LOOM_SERVE_TEST_PROJECTOR_SOURCE");
    ASSERT_NE(source, nullptr);
    IREE_ASSERT_OK(document_.Load(source));
    const loom_serve_finite_model_options_t model_options = {
        .sources = document_.value().sources,
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{}"),
    };
    IREE_ASSERT_OK(loom_serve_finite_model_allocate(device_, &model_options,
                                                    host_allocator_, &model_));
    iree_async_proactor_t* proactor = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_pool_get(
        loom_serve_host_proactor_pool(host), 0, &proactor));
    const loom_serve_wake_callback_t observer = {
        [](void* data) {
          iree_notification_post(static_cast<iree_notification_t*>(data),
                                 IREE_ALL_WAITERS);
        },
        &notification_,
    };
    loom_serve_workspace_set_observer(loom_serve_device_workspace(device_),
                                      observer);
    IREE_ASSERT_OK(loom_serve_text_media_projector_allocate(
        model_, requests_.size(), loom_serve_host_executor(host), proactor,
        observer, host_allocator_, &projector_));
    image_.data_url = IREE_SV(
        "data:image/x-portable-arbitrarymap;base64,"
        "UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCAzCk1BWFZBTCAyNTUKVFVQTFRZUEUg"
        "UkdCCkVOREhEUgr/AAAA/wAAAP////8=");
    image_.feature_count = 1;
    policy_.images.limits = {
        .compressed_bytes = 1024,
        .width = 2,
        .height = 2,
        .pixels = 4,
    };
    IREE_ASSERT_OK(
        loom_serve_media_image_inspect(image_.data_url, &policy_.images.limits,
                                       host_allocator_, &image_.info));
    chat_.policy = &policy_;
    chat_.images.count = 1;
    chat_.images.values = &image_;
    input_.media_spans = &span_;
    input_.media_span_count = 1;
  }

  void TearDown() override {
    if (projector_) {
      AdvanceUntil([&] {
        return !loom_serve_text_media_projector_is_pending(projector_);
      });
      for (auto* request : requests_) {
        EXPECT_TRUE(
            loom_serve_text_media_projector_cancel(projector_, request));
        iree_status_ignore(loom_serve_text_media_request_take_status(request));
        loom_serve_text_media_request_free(request);
      }
      loom_serve_text_media_projector_free(projector_);
    }
    IREE_EXPECT_OK(loom_serve_finite_model_free(model_));
    loom_serve_workspace_set_observer(loom_serve_device_workspace(device_), {});
    iree_notification_deinitialize(&notification_);
    iree_string_builder_deinitialize(&chat_.images.metadata);
  }

  template <typename Predicate>
  void AdvanceUntil(Predicate ready) {
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      auto status =
          loom_serve_workspace_advance(loom_serve_device_workspace(device_));
      status = iree_status_join(status, loom_serve_text_media_projector_advance(
                                            projector_, 0, &progress));
      if (ready() || progress || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(&notification_);
        IREE_ASSERT_OK(status);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }

  void Offer(size_t index, uint64_t consumer_bytes) {
    IREE_ASSERT_OK(loom_serve_text_media_request_allocate(
        &chat_, host_allocator_, &requests_[index]));
    EXPECT_FALSE(
        loom_serve_text_media_request_bind(requests_[index], &input_, 0));
    EXPECT_TRUE(loom_serve_text_media_projector_offer(
        projector_, requests_[index], consumer_bytes));
  }

  // Allocator for per-case model and request ownership.
  const iree_allocator_t host_allocator_ = iree_allocator_system();
  // Source ownership encloses every asynchronous compile and invocation.
  loom_serve::testing::ModelDocument document_;
  // Borrowed device whose workspace is driven by this serialized owner.
  loom_serve_device_t* device_ = nullptr;
  // Actual source-defined pixel producer, not a mocked projection service.
  loom_serve_finite_model_t* model_ = nullptr;
  // Serialized preparation/cohort/publication owner under test.
  loom_serve_text_media_projector_t* projector_ = nullptr;
  // Ordinary owner wake, joined before its storage is destroyed.
  iree_notification_t notification_ = {};
  // Only codec bounds are used by projection; no chat VM is invoked here.
  loom_serve_text_chat_policy_t policy_ = {};
  // Immutable inspected input borrowed by each independent request.
  loom_serve_text_chat_image_t image_ = {};
  // Real compressed image descriptor and source metadata, with no pixel plane.
  loom_serve_text_chat_t chat_ = {};
  // Trusted logical feature span produced before projection admission.
  loom_serve_text_media_span_t span_ = {.input_count = 1};
  // Borrowed token-coordinate plan used by the same bind path as text serving.
  loom_serve_text_input_t input_ = {};
  // Independent output/cancellation owners sharing a single native cohort.
  std::array<loom_serve_text_media_request_t*, 3> requests_ = {};
};

TEST_F(MediaProjectorTest, PublishesAccordingToPhaseFootprintAndKeepsOrdinals) {
  ASSERT_NO_FATAL_FAILURE(Offer(0, 0));
  ASSERT_NO_FATAL_FAILURE(Offer(1, 12 * 1024));
  ASSERT_NO_FATAL_FAILURE(Offer(2, 12 * 1024));
  ASSERT_TRUE(loom_serve_text_media_projector_start(projector_));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    const char* phase = loom_serve_text_media_projector_phase(projector_);
    return phase && std::strcmp(phase, "image_execute") == 0;
  }));
  // This owner has submitted but has not consumed completion. Native device
  // speed cannot change this boundary; no sleeps or repeated races are needed.
  EXPECT_TRUE(loom_serve_text_media_request_bind(requests_[0], &input_, 0));
  EXPECT_FALSE(loom_serve_text_media_request_bind(requests_[1], &input_, 0));
  EXPECT_FALSE(loom_serve_text_media_request_bind(requests_[2], &input_, 0));
  EXPECT_EQ(loom_serve_finite_model_parameters(model_).pin_count, 1u);
  ASSERT_TRUE(loom_serve_text_media_projector_cancel(projector_, requests_[1]));
  loom_serve_text_media_request_free(requests_[1]);
  requests_[1] = nullptr;
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return loom_serve_text_media_projector_is_collecting(projector_);
  }));
  EXPECT_EQ(loom_serve_finite_model_parameters(model_).pin_count, 0u);
  ASSERT_TRUE(loom_serve_text_media_request_bind(requests_[2], &input_, 0));
  const auto* early = loom_serve_text_media_request_input(requests_[0]);
  const auto* deferred = loom_serve_text_media_request_input(requests_[2]);
  EXPECT_EQ(early->spans[0].result, deferred->spans[0].result);
  EXPECT_EQ(early->spans[0].request_index, 0u);
  EXPECT_EQ(deferred->spans[0].request_index, 2u);

  loom_serve_finite_snapshot_t* snapshot = nullptr;
  loom_serve_execution_frontier_t ready;
  IREE_ASSERT_OK(loom_serve_finite_result_download(deferred->spans[0].result,
                                                   &snapshot, &ready));
  for (size_t i = 0; i < IREE_ARRAYSIZE(ready.semaphores); ++i) {
    IREE_EXPECT_OK(
        loom_serve_workspace_wait(loom_serve_device_workspace(device_),
                                  ready.semaphores[i], ready.values[i]));
  }
  const std::array<float, 4> expected = {513, 0, 0, 3};
  for (size_t index : {0u, 2u}) {
    const auto row = loom_serve_finite_snapshot_row(snapshot, index, 0);
    EXPECT_EQ(row.data_length, sizeof(expected));
    if (row.data_length == sizeof(expected)) {
      EXPECT_EQ(std::memcmp(row.data, expected.data(), sizeof(expected)), 0);
    }
    loom_serve_text_media_request_begin(requests_[index]);
    loom_serve_text_media_request_consume(requests_[index], 1);
  }
  loom_serve_finite_snapshot_release(snapshot);
}

TEST_F(MediaProjectorTest, RejectsImpossibleContinuationBeforeSubmission) {
  ASSERT_NO_FATAL_FAILURE(Offer(0, 16 * 1024));
  ASSERT_TRUE(loom_serve_text_media_projector_start(projector_));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return loom_serve_text_media_projector_is_collecting(projector_);
  }));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_text_media_request_take_status(requests_[0]));
  EXPECT_FALSE(loom_serve_text_media_request_bind(requests_[0], &input_, 0));
  EXPECT_EQ(loom_serve_finite_model_parameters(model_).pin_count, 0u);
}

}  // namespace
