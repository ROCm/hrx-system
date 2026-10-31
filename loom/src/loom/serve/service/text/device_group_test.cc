// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "loom/serve/service/text/service_test_fixture.h"

namespace loom_serve::testing::text {

class GroupTextServiceTest : public SourceTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }
  iree_host_size_t DeviceCount() const override { return 2; }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_text_model_free(original_));
    IREE_EXPECT_OK(loom_serve_text_model_free(cancelled_));
    SourceTextServiceTest::TearDown();
  }

  // Drives the actual owner state machine, including draining every accepted
  // callback after cancellation. No scheduler worker blocks on this join.
  iree_status_t Prepare(size_t index, bool cancel,
                        loom_serve_compilation_report_t* report,
                        loom_serve_text_model_t** out_model) {
    const auto options = ModelOptions();
    iree_notification_t notification;
    iree_notification_initialize(&notification);
    loom_serve_text_load_t* load = nullptr;
    auto status = loom_serve_text_load_allocate(
        loom_serve_device_group_at(group_, index),
        loom_serve_device_group_pool_at(group_, index), &options,
        [](void* payload) {
          iree_notification_post(static_cast<iree_notification_t*>(payload),
                                 IREE_ALL_WAITERS);
        },
        &notification, host_allocator_, &load);
    if (iree_status_is_ok(status)) {
      bool complete = false;
      loom_serve_text_load_flags_t flags = 0;
      while (!complete) {
        const auto ticket = iree_notification_prepare_wait(&notification);
        status = loom_serve_text_load_advance(load, flags, &complete);
        if (complete) {
          iree_notification_cancel_wait(&notification);
        } else {
          IREE_CHECK_OK(status);
          iree_notification_commit_wait(&notification, ticket,
                                        IREE_DURATION_ZERO,
                                        IREE_TIME_INFINITE_FUTURE);
        }
        if (cancel) {
          flags = LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL;
        }
      }
      *report = *loom_serve_text_load_report(load);
      *out_model = loom_serve_text_load_take_model(load);
      loom_serve_text_load_free(load);
    }
    iree_notification_deinitialize(&notification);
    return status;
  }

  // Original compiling model, distinct from the HTTP service's device owner.
  loom_serve_text_model_t* original_ = nullptr;
  // Partial ownership returned by the cancelled asynchronous preparation.
  loom_serve_text_model_t* cancelled_ = nullptr;
};

TEST_F(GroupTextServiceTest,
       SharedNativeArtifactsSurviveOriginalAndCancelledPeerModels) {
  ASSERT_EQ(loom_serve_device_group_count(group_), 2u);
  EXPECT_NE(loom_serve_device_handle(loom_serve_device_group_at(group_, 0)),
            loom_serve_device_handle(loom_serve_device_group_at(group_, 1)));
  loom_serve_compilation_report_t first = {}, second = {}, cancelled = {};
  IREE_ASSERT_OK(Prepare(0, false, &first, &original_));
  ASSERT_GT(first.kernels, 0u);
  IREE_ASSERT_OK(Prepare(1, false, &second, &model_));
  EXPECT_EQ(second.kernels, 0u);
  EXPECT_GT(second.reused_kernels, 0u);
  EXPECT_EQ(second.entries, first.entries);
  IREE_ASSERT_OK(loom_serve_text_model_free(std::exchange(original_, nullptr)));

  // The second device's independently loaded code serves the exact source
  // oracle after the original compiler and parameter owners have retired.
  Start();
  const std::string prompt = "abxa";
  const int client = Send(Request(Message("user", prompt), 4));
  const auto before = loom_serve_pool_statistics(storage_owner_);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_CANCELLED,
                        Prepare(0, true, &cancelled, &cancelled_));
  ASSERT_NE(cancelled_, nullptr);
  EXPECT_TRUE(loom_serve_text_model_is_retired(cancelled_));
  IREE_ASSERT_OK(
      loom_serve_text_model_free_retired(std::exchange(cancelled_, nullptr)));
  EXPECT_EQ(Content(Read(client)), Oracle::Complete(prompt, 4));
  const auto after = loom_serve_pool_statistics(storage_owner_);
  EXPECT_EQ(after.retained.reserved_bytes, before.retained.reserved_bytes);
  EXPECT_EQ(after.retained.committed_bytes, before.retained.committed_bytes);
  EXPECT_EQ(Content(Read(Send(Request(Message("user", "bax"), 4)))),
            Oracle::Complete("bax", 4));
}

}  // namespace loom_serve::testing::text
