// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/weights_test_base.h"

namespace loom_serve::testing {
namespace {

class ResidencyPressureTest : public WeightsTest {
 protected:
  void TearDown() override {
    if (deferred_backing_) {
      deferred_backing_->statistics_observer = {};
      loom_serve_memory_pool_set_observer(pool_, nullptr, nullptr);
      deferred_backing_->ResumeAcquisitions();
    }
    if (residencies_[0]) {
      loom_serve_residency_set_reclaimer(residencies_[0], {});
    }
    WeightsTest::TearDown();
  }

  void RegisterHistory() {
    IREE_ASSERT_OK(Register(0, 0, 0));
    loom_serve_residency_set_reclaimer(
        residencies_[0],
        {this,
         [](void* self) {
           auto* owner = static_cast<ResidencyPressureTest*>(self);
           if (owner->history_pending_) {
             return false;
           }
           ++owner->history_requests_;
           owner->history_pending_ = true;
           return true;
         },
         [](void* self) {
           return static_cast<ResidencyPressureTest*>(self)->history_pending_;
         }});
  }

  // Counts accepted logical history releases, not physical trim attempts.
  uint32_t history_requests_ = 0;
  // History maintenance retains its backing until this owner completes it.
  bool history_pending_ = false;
  // Joins publication before classification without collecting owner counters.
  loom_serve_test_weight_activation_wake_t publication_;
  // Confirms the held dependency completed at the native observation boundary.
  bool observed_ = false;
};

TEST_P(ResidencyPressureTest, LiveWorkspaceDefersFutileHistoryEviction) {
  ASSERT_NO_FATAL_FAILURE(RegisterHistory());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, loom_serve_pool_state_geometry(storage_owner_), 3 * kSlabBytes,
      256, &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state_, 0, kSlabBytes));
  IREE_ASSERT_OK(
      loom_serve_execution_reserve_workspace(execution_, kSlabBytes, 256));
  const auto request = iree_hal_make_pool_reservation_request(kSlabBytes, 256);
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(execution_, 1, &request, &workspace_));
  const uint32_t value = 0x13579BDF;
  const iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill =
          {
              .target_buffer = workspace_,
              .length = sizeof(output_),
              .pattern = &value,
              .pattern_length = sizeof(value),
          },
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &fill, &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));

  // Two new slabs cannot fit beside live scratch in a two-slab pool, even if
  // every cached history is discarded. Pressure must leave that history alone.
  const loom_serve_memory_range_t growth = {state_, kSlabBytes, 2 * kSlabBytes};
  bool admitted = true;
  IREE_ASSERT_OK(
      loom_serve_memory_pool_try_commit(pool_, 1, &growth, &admitted));
  EXPECT_FALSE(admitted);
  EXPECT_EQ(history_requests_, 0u);
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = workspace_,
              .target = output_.data(),
              .length = sizeof(output_),
          },
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  for (const auto word : output_) {
    EXPECT_EQ(static_cast<uint32_t>(word), value);
  }

  IREE_ASSERT_OK(
      loom_serve_execution_dealloca(execution_, workspace_, &completion));
  iree_hal_buffer_release(workspace_);
  workspace_ = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  IREE_ASSERT_OK(loom_serve_test_workspace_await_idle(
      loom_serve_pool_workspace(storage_owner_)));
  // Workspace retirement makes history eviction useful. The regular policy
  // requests one logical release; only its owner's trim grants physical space.
  IREE_ASSERT_OK(loom_serve_pool_trim(storage_owner_, kSlabBytes));
  DrainNativeReturns();
  IREE_ASSERT_OK(
      loom_serve_memory_pool_try_commit(pool_, 1, &growth, &admitted));
  EXPECT_FALSE(admitted);
  EXPECT_EQ(history_requests_, 1u);
  loom_serve_virtual_buffer_begin_trim(state_);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(state_));
  history_pending_ = false;
  DrainNativeReturns();
  IREE_ASSERT_OK(
      loom_serve_memory_pool_try_commit(pool_, 1, &growth, &admitted));
  ASSERT_TRUE(admitted);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  EXPECT_EQ(state_statistics_.committed_bytes, 2 * kSlabBytes);
  EXPECT_EQ(history_requests_, 1u);
}

TEST_P(ResidencyPressureTest, FailedPublicationDoesNotInventHistoryPressure) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(2 * kSlabBytes));
  ASSERT_NO_FATAL_FAILURE(RegisterHistory());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, loom_serve_pool_state_geometry(storage_owner_), kSlabBytes, 256,
      &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state_, 0, kSlabBytes));
  IREE_ASSERT_OK(Compile(0, "direct_incomplete"));
  const auto root = Root(0, 0, kSlabBytes);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(Register(1, 0, 1));
  deferred_backing_->defer = true;
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
  deferred_backing_->WaitForAcquisition();
  EXPECT_EQ(loom_serve_weights_memory_usage(plans_[0]).pending_bytes,
            kSlabBytes);

  // Publish an actual source-allocation failure during native observation.
  // Owner counters stay unchanged until the next collection. Combining the
  // old pool total with freshly collected group counters invents pressure.
  loom_serve_memory_pool_set_observer(
      pool_, loom_serve_test_weight_activation_wake_t::Notify, &publication_);
  deferred_backing_->statistics_observer = [this] {
    deferred_backing_->statistics_observer = {};
    observed_ = true;
    deferred_backing_->FailOne();
    publication_.Await();
  };
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
      pool_, {.packed_bytes = kSlabBytes}, &admission));
  EXPECT_TRUE(observed_);
  EXPECT_EQ(history_requests_, 0u);
  loom_serve_memory_pool_set_observer(pool_, nullptr, nullptr);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_weights_await(plans_[0]));
  IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
      pool_, {.packed_bytes = kSlabBytes}, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_EQ(history_requests_, 0u);
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
}

INSTANTIATE_TEST_SUITE_P(Pressure, ResidencyPressureTest,
                         ::testing::Values(true));

}  // namespace
}  // namespace loom_serve::testing
