// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/reservation.h"

#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/testing/device.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class TextReservationTest : public SourceTextServiceTest,
                            public ::testing::WithParamInterface<bool> {
 protected:
  void TearDown() override {
    loom_serve_text_reservation_deinitialize(&reservation_);
    loom_serve_text_checkpoint_release(checkpoint_);
    SourceTextServiceTest::TearDown();
  }

  // Owned logical plan, including after an assertion aborts the test body.
  loom_serve_text_reservation_t reservation_ = {};
  // Owned prefix endpoint until explicitly transferred to the plan alone.
  loom_serve_text_checkpoint_t* checkpoint_ = nullptr;
};

TEST_P(TextReservationTest,
       DeferredForkSurvivesCancellationPeerWorkAndBackingTrim) {
  auto* source = loom_serve_text_model_row(model_, 0);
  auto* branch = loom_serve_text_model_row(model_, 1);
  auto* peer = loom_serve_text_model_row(model_, 2);
  const int32_t tokens[] = {0, 1, 2};
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(source, IREE_ARRAYSIZE(tokens), tokens));
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(source, &checkpoint_));
  ASSERT_NE(checkpoint_, nullptr);
  IREE_ASSERT_OK(loom_serve_text_row_reset(source));
  if (GetParam()) {
    IREE_ASSERT_OK(loom_serve_text_checkpoint_suspend(checkpoint_));
    ASSERT_GT(loom_serve_text_checkpoint_suspended_bytes(checkpoint_), 0u);
  }
  const auto available = loom_serve_text_model_pool_usage(model_).available;
  IREE_ASSERT_OK(loom_serve_text_reservation_prepare(
      branch, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, checkpoint_, 16,
      &reservation_));
  ASSERT_NE(reservation_.state, nullptr);
  EXPECT_EQ(loom_serve_text_row_position(branch), 0u);
  EXPECT_LT(loom_serve_text_model_pool_usage(model_).available, available);
  loom_serve_execution_frontier_t frontier;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_text_checkpoint_submit_suspend(checkpoint_, &frontier));

  // Cancellation returns COW and cold-restore IDs without consuming the source.
  loom_serve_text_reservation_deinitialize(&reservation_);
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, available);
  EXPECT_EQ(loom_serve_text_checkpoint_suspended_bytes(checkpoint_) != 0,
            GetParam());
  IREE_ASSERT_OK(loom_serve_text_reservation_prepare(
      branch, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, checkpoint_, 16,
      &reservation_));
  ASSERT_NE(reservation_.state, nullptr);
  if (GetParam()) {
    loom_serve_text_reservation_t concurrent = {};
    IREE_ASSERT_OK(loom_serve_text_reservation_prepare(
        peer, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, checkpoint_, 16,
        &concurrent));
    EXPECT_EQ(concurrent.state, nullptr);
    loom_serve_text_reservation_deinitialize(&concurrent);
  }

  // Dropping a caller handle cannot destroy the endpoint borrowed by the plan.
  loom_serve_text_checkpoint_release(checkpoint_);
  checkpoint_ = nullptr;
  const int32_t peer_tokens[] = {2, 1};
  IREE_ASSERT_OK(loom_serve_text_row_prefill(peer, IREE_ARRAYSIZE(peer_tokens),
                                             peer_tokens));
  Oracle peer_oracle;
  peer_oracle.Consume("xb");
  EXPECT_EQ(loom_serve_text_row_token(peer), peer_oracle.Predict());
  loom_serve_text_trim_result_t trim;
  IREE_ASSERT_OK(loom_serve_text_model_trim(model_, &trim));
  EXPECT_EQ(trim.moved_blocks, 0u);

  loom_serve_text_reserve_result_t admitted;
  IREE_ASSERT_OK(
      loom_serve_text_reservation_submit(&reservation_, &admitted, &frontier));
  ASSERT_EQ(admitted, LOOM_SERVE_TEXT_RESERVE_ACQUIRED);
  for (size_t i = 0; i < IREE_ARRAYSIZE(frontier.semaphores); ++i) {
    IREE_ASSERT_OK(
        loom_serve_workspace_wait(loom_serve_device_workspace(device_),
                                  frontier.semaphores[i], frontier.values[i]));
  }
  loom_serve_text_row_complete_reserve(branch);
  loom_serve_text_reservation_deinitialize(&reservation_);
  EXPECT_EQ(loom_serve_text_row_position(branch), 3u);
  const int32_t suffix = 2;
  IREE_ASSERT_OK(loom_serve_text_row_prefill(branch, 1, &suffix));
  Oracle oracle;
  oracle.Consume("abxx");
  EXPECT_EQ(loom_serve_text_row_token(branch), oracle.Predict());
  IREE_ASSERT_OK(loom_serve_text_row_reset(branch));
  IREE_ASSERT_OK(loom_serve_text_row_reset(peer));
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 256u);
}

INSTANTIATE_TEST_SUITE_P(
    ResidentAndSuspended, TextReservationTest,
    ::testing::ValuesIn(loom_serve_test_device()->service_backing ==
                                LOOM_SERVE_DEVICE_BACKING_ELASTIC
                            ? std::vector<bool>{false, true}
                            : std::vector<bool>{false}));

}  // namespace
