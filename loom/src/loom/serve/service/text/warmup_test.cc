// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class TextWarmupTest : public SourceTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }

  void StartFailedWarmup() {
    const auto options = ModelOptions();
    IREE_ASSERT_OK(loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &model_));
    // Preparation indexes the real checkpoint without reading its payload.
    // Truncation makes the subsequent demanded read fail through normal I/O.
    std::filesystem::resize_file(weights_, 0);
    IREE_ASSERT_OK(loom_serve_text_warmup_allocate(
        device_, storage_owner_, model_->parameter_group, nullptr, nullptr,
        host_allocator_, &model_->warmup));
    loom_serve_text_warmup_poll(model_->warmup);
  }

  void CheckReload() {
    ASSERT_EQ(model_, nullptr);
    ASSERT_NO_FATAL_FAILURE(WriteWeights(2));
    const auto options = ModelOptions();
    IREE_ASSERT_OK(loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &model_));
    auto* row = loom_serve_text_model_row(model_, 0);
    const int32_t token = Oracle::Id('a');
    IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
    Oracle reference;
    reference.Consume("a");
    EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
  }
};

TEST_F(TextWarmupTest, FailedWarmupStillReleasesTheRetiredModel) {
  ASSERT_NO_FATAL_FAILURE(StartFailedWarmup());
  loom_serve_text_warmup_await(model_->warmup);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_serve_text_warmup_status(model_->warmup));

  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_model_retire(
      model_,
      [](void* payload) {
        iree_notification_post(static_cast<iree_notification_t*>(payload),
                               IREE_ALL_WAITERS);
      },
      &notification);
  IREE_EXPECT_OK(loom_runtime_workspace_await(
      loom_runtime_pool_workspace(storage_owner_), &notification,
      [](void* payload) {
        return loom_serve_text_model_is_retired(
            static_cast<loom_serve_text_model_t*>(payload));
      },
      model_));
  EXPECT_TRUE(loom_serve_text_model_is_retired(model_));
  IREE_EXPECT_OK(
      loom_serve_text_model_free_retired(std::exchange(model_, nullptr)));
  iree_notification_deinitialize(&notification);
  CheckReload();
}

TEST_F(TextWarmupTest, ColdShutdownReportsTheWarmupFailure) {
  ASSERT_NO_FATAL_FAILURE(StartFailedWarmup());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_serve_text_model_free(std::exchange(model_, nullptr)));
  CheckReload();
}

}  // namespace
