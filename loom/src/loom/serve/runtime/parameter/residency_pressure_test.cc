// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

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
         },
         [](void* self, const loom_serve_residency_forecast_t*,
            loom_serve_residency_projection_t* projection) {
           const auto* owner = static_cast<const ResidencyPressureTest*>(self);
           *projection = {
               .resident_bytes = owner->state_statistics_.committed_bytes +
                                 owner->state_statistics_.pending_bytes,
           };
           if (owner->history_retained_) {
             projection->minimum_bytes = projection->resident_bytes;
           }
           return iree_ok_status();
         }});
  }

  void Fill(iree_hal_buffer_t* buffer) {
    const iree_hal_transfer_operation_t fill = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill =
            {
                .target_buffer = buffer,
                .length = sizeof(output_),
                .pattern = &kWord,
                .pattern_length = sizeof(kWord),
            },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution_, 1, &fill, &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));
  }

  void Check(iree_hal_buffer_t* buffer) {
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffer,
                .target = output_.data(),
                .length = sizeof(output_),
            },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(execution_, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
    for (const auto word : output_) {
      EXPECT_EQ(static_cast<uint32_t>(word), kWord);
    }
  }

  static constexpr uint32_t kWord = 0x13579BDF;

  // Counts accepted logical history releases, not physical trim attempts.
  uint32_t history_requests_ = 0;
  // History maintenance retains its backing until this owner completes it.
  bool history_pending_ = false;
  // Explicit retention protects this fixture owner's complete backed state.
  bool history_retained_ = false;
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
  ASSERT_NO_FATAL_FAILURE(Fill(workspace_));

  // Two new slabs cannot fit beside live scratch in a two-slab pool, even if
  // every cached history is discarded. Pressure must leave that history alone.
  const loom_serve_memory_range_t growth = {state_, kSlabBytes, 2 * kSlabBytes};
  bool admitted = true;
  IREE_ASSERT_OK(
      loom_serve_memory_pool_try_commit(pool_, 1, &growth, &admitted));
  EXPECT_FALSE(admitted);
  EXPECT_EQ(history_requests_, 0u);
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  ASSERT_NO_FATAL_FAILURE(Check(workspace_));

  uint64_t completion = 0;
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

TEST_P(ResidencyPressureTest,
       RetainedStateProtectsBothPhysicalAndLogicalAdmission) {
  ASSERT_NO_FATAL_FAILURE(RegisterHistory());
  IREE_ASSERT_OK(Register(1, 0, 0));
  const auto geometry = loom_serve_pool_state_geometry(storage_owner_);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, geometry, kSlabBytes, 256, &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state_, 0, kSlabBytes));
  ASSERT_NO_FATAL_FAILURE(Fill(loom_serve_virtual_buffer_handle(state_)));
  history_retained_ = true;
  loom_runtime_storage_statistics_t incoming_statistics = {};
  loom_serve_virtual_buffer_t* incoming = nullptr;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, geometry, 2 * kSlabBytes, 256, &incoming_statistics, &incoming));
  const auto free_buffer = [](loom_serve_virtual_buffer_t* buffer) {
    IREE_EXPECT_OK(loom_serve_virtual_buffer_free(buffer));
  };
  const std::unique_ptr<loom_serve_virtual_buffer_t, decltype(free_buffer)>
      incoming_owner(incoming, free_buffer);
  const loom_serve_memory_range_t range = {incoming, 0, 2 * kSlabBytes};
  loom_serve_memory_operation_t* operation = nullptr;
  loom_serve_residency_admission_t logical;
  IREE_ASSERT_OK(loom_serve_residency_try_admit_state(1, &residencies_[1], 0, 1,
                                                      &range, nullptr, nullptr,
                                                      0, &logical, &operation));
  EXPECT_EQ(logical, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  EXPECT_EQ(operation, nullptr);
  loom_serve_memory_admission_t physical;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(pool_, 1, &range, nullptr,
                                                     0, &physical, &operation));
  EXPECT_EQ(physical, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  EXPECT_EQ(operation, nullptr);
  EXPECT_EQ(history_requests_, 0u);
  ASSERT_NO_FATAL_FAILURE(Check(loom_serve_virtual_buffer_handle(state_)));

  // Dropping retention changes eligibility, not actual backing. Both callers
  // use the same owner fact and wait for its ordinary asynchronous release.
  history_retained_ = false;
  IREE_ASSERT_OK(loom_serve_residency_try_admit_state(1, &residencies_[1], 0, 1,
                                                      &range, nullptr, nullptr,
                                                      0, &logical, &operation));
  EXPECT_EQ(logical, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  EXPECT_EQ(history_requests_, 1u);
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  loom_serve_virtual_buffer_begin_trim(state_);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(state_));
  history_pending_ = false;
  DrainNativeReturns();
  IREE_ASSERT_OK(loom_serve_residency_try_admit_state(1, &residencies_[1], 0, 1,
                                                      &range, nullptr, nullptr,
                                                      0, &logical, &operation));
  const auto free_operation = [](loom_serve_memory_operation_t* operation) {
    loom_serve_memory_operation_await(operation);
    loom_serve_memory_operation_free(operation);
  };
  const std::unique_ptr<loom_serve_memory_operation_t, decltype(free_operation)>
      operation_owner(operation, free_operation);
  ASSERT_EQ(logical, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_serve_memory_operation_query_status(operation));
  loom_serve_residency_release(residencies_[1]);
  loom_serve_residency_cancel_admission(residencies_[1]);
  ASSERT_NO_FATAL_FAILURE(Fill(loom_serve_virtual_buffer_handle(incoming)));
  ASSERT_NO_FATAL_FAILURE(Check(loom_serve_virtual_buffer_handle(incoming)));
}

TEST_P(ResidencyPressureTest, FailedPublicationDoesNotInventHistoryPressure) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(2 * kSlabBytes));
  ASSERT_NO_FATAL_FAILURE(RegisterHistory());
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, loom_serve_pool_state_geometry(storage_owner_), 2 * kSlabBytes,
      256, &state_statistics_, &state_));
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
      pool_, loom_serve_virtual_buffer_missing_demand(state_), &admission));
  EXPECT_TRUE(observed_);
  EXPECT_EQ(history_requests_, 0u);
  loom_serve_memory_pool_set_observer(pool_, nullptr, nullptr);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_weights_await(plans_[0]));
  IREE_ASSERT_OK(loom_serve_memory_pool_prepare(
      pool_, loom_serve_virtual_buffer_missing_demand(state_), &admission));
  EXPECT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  EXPECT_EQ(history_requests_, 0u);
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
}

TEST_P(ResidencyPressureTest, LogicalAdmissionCollectsFailureBeforeEviction) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(2 * kSlabBytes));
  ASSERT_NO_FATAL_FAILURE(RegisterHistory());
  IREE_ASSERT_OK(Register(1, 0, 0));
  const auto geometry = loom_serve_pool_state_geometry(storage_owner_);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, geometry, 2 * kSlabBytes, 256, &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state_, 0, kSlabBytes));
  ASSERT_NO_FATAL_FAILURE(Fill(loom_serve_virtual_buffer_handle(state_)));

  loom_runtime_storage_statistics_t incoming_statistics = {};
  loom_serve_virtual_buffer_t* incoming = nullptr;
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, geometry, kSlabBytes, 256, &incoming_statistics, &incoming));
  const auto free_buffer = [](loom_serve_virtual_buffer_t* buffer) {
    IREE_EXPECT_OK(loom_serve_virtual_buffer_free(buffer));
  };
  const std::unique_ptr<loom_serve_virtual_buffer_t, decltype(free_buffer)>
      incoming_owner(incoming, free_buffer);
  const auto free_operation = [](loom_serve_memory_operation_t* operation) {
    loom_serve_memory_operation_await(operation);
    loom_serve_memory_operation_free(operation);
  };

  deferred_backing_->defer = true;
  const loom_serve_memory_range_t failed_range = {state_, kSlabBytes,
                                                  kSlabBytes};
  loom_serve_memory_admission_t memory_admission;
  loom_serve_memory_operation_t* failed_operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool_, 1, &failed_range, nullptr, 0, &memory_admission,
      &failed_operation));
  const std::unique_ptr<loom_serve_memory_operation_t, decltype(free_operation)>
      failed_owner(failed_operation, free_operation);
  deferred_backing_->WaitForAcquisition();
  const loom_serve_memory_range_t incoming_range = {incoming, 0, kSlabBytes};
  loom_serve_residency_admission_t admission;
  loom_serve_memory_operation_t* incoming_operation = nullptr;
  IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
      1, &residencies_[1], 0, 1, &incoming_range, nullptr, nullptr, 0,
      &admission, &incoming_operation));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  EXPECT_EQ(incoming_operation, nullptr);
  EXPECT_EQ(history_requests_, 0u);
  loom_serve_memory_pool_set_observer(
      pool_, loom_serve_test_weight_activation_wake_t::Notify, &publication_);
  deferred_backing_->FailOne();
  publication_.Await();
  deferred_backing_->ResumeAcquisitions();
  // The failed allocation is complete but its owner charge is not published.
  // Logical admission must collect that result before requesting history loss.
  EXPECT_EQ(state_statistics_.pending_bytes, kSlabBytes);
  IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
      1, &residencies_[1], 0, 1, &incoming_range, nullptr, nullptr, 0,
      &admission, &incoming_operation));
  const std::unique_ptr<loom_serve_memory_operation_t, decltype(free_operation)>
      incoming_operation_owner(incoming_operation, free_operation);
  EXPECT_EQ(history_requests_, 0u);
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  if (admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    loom_serve_residency_release(residencies_[1]);
    loom_serve_memory_operation_await(incoming_operation);
    IREE_ASSERT_OK(
        loom_serve_memory_operation_query_status(incoming_operation));
    EXPECT_EQ(incoming_statistics.committed_bytes, kSlabBytes);
  }
  loom_serve_residency_cancel_admission(residencies_[1]);
  loom_serve_memory_operation_await(failed_operation);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_memory_operation_query_status(failed_operation));
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  ASSERT_NO_FATAL_FAILURE(Check(loom_serve_virtual_buffer_handle(state_)));
}

TEST_P(ResidencyPressureTest, WarmAdmissionDoesNotQueryNativeCapacity) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(2 * kSlabBytes));
  IREE_ASSERT_OK(Compile(0, "direct_incomplete"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[0]));
  loom_serve_residency_release(residencies_[0]);
  uint32_t queries = 0;
  deferred_backing_->statistics_observer = [&queries] { ++queries; };
  uint32_t forecasts = 0;
  loom_serve_residency_set_reclaimer(
      residencies_[0],
      {
          .self = &forecasts,
          .forecast =
              [](void* self, const loom_serve_residency_forecast_t*,
                 loom_serve_residency_projection_t* projection) {
                ++*static_cast<uint32_t*>(self);
                *projection = {};
                return iree_ok_status();
              },
      });
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(
      loom_serve_residency_try_admit(1, &residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  // A fitting private-workspace increase also reads only owner facts.
  IREE_EXPECT_OK(
      loom_serve_residency_check_private_peak(residencies_[0], kSlabBytes));
  EXPECT_EQ(queries, 0u);
  EXPECT_EQ(forecasts, 0u);
  deferred_backing_->statistics_observer = {};
  if (admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    ASSERT_NO_FATAL_FAILURE(ReadOutputOn(execution_, 0));
    for (int32_t i = 0; i < 8; ++i) {
      EXPECT_EQ(output_[i], 30 + i);
    }
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 1u);
    loom_serve_residency_release(residencies_[0]);
  }
}

TEST_P(ResidencyPressureTest, WorkspaceGrowthCollectsFailureBeforeDenial) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking(2 * kSlabBytes));
  ASSERT_NO_FATAL_FAILURE(RegisterHistory());
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  const auto geometry = loom_serve_pool_state_geometry(storage_owner_);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, geometry, 2 * kSlabBytes, 256, &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state_, 0, kSlabBytes));
  ASSERT_NO_FATAL_FAILURE(Fill(loom_serve_virtual_buffer_handle(state_)));

  deferred_backing_->defer = true;
  const loom_serve_memory_range_t failed_range = {state_, kSlabBytes,
                                                  kSlabBytes};
  loom_serve_memory_admission_t admission;
  loom_serve_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
      pool_, 1, &failed_range, nullptr, 0, &admission, &operation));
  deferred_backing_->WaitForAcquisition();
  loom_serve_memory_pool_set_observer(
      pool_, loom_serve_test_weight_activation_wake_t::Notify, &publication_);
  deferred_backing_->FailOne();
  publication_.Await();
  deferred_backing_->ResumeAcquisitions();
  EXPECT_EQ(state_statistics_.pending_bytes, kSlabBytes);

  // The command can grow beside the surviving history. Its failed neighbor
  // must no longer consume capacity, and checking capacity cannot evict it.
  IREE_EXPECT_OK(
      loom_serve_residency_check_private_peak(residencies_[0], kSlabBytes));
  EXPECT_EQ(history_requests_, 0u);
  EXPECT_EQ(loom_serve_residency_snapshot(residencies_[0]).pin_count, 1u);
  loom_serve_memory_operation_await(operation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_memory_operation_query_status(operation));
  loom_serve_memory_operation_free(operation);
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  ASSERT_NO_FATAL_FAILURE(Check(loom_serve_virtual_buffer_handle(state_)));
  loom_serve_residency_release(residencies_[0]);
}

INSTANTIATE_TEST_SUITE_P(Pressure, ResidencyPressureTest,
                         ::testing::Values(true));

}  // namespace
}  // namespace loom_serve::testing
