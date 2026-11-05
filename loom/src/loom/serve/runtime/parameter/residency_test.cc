// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/weights_test_base.h"

namespace loom_serve::testing {
namespace {

class ResidencyTest : public WeightsTest {
 protected:
  virtual void PrepareBacking() {}

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(WeightsTest::SetUp());
    ASSERT_NO_FATAL_FAILURE(PrepareBacking());
    WriteCheckpoint(adapter_path_.path(), 0);
    owner_executions_[0] = execution_;
    for (size_t i = 0; i < owner_executions_.size(); ++i) {
      if (i) {
        IREE_ASSERT_OK(AllocateWeightPool(device_owner_, GetParam(), kSlabBytes,
                                          2 * kSlabBytes, &owner_pools_[i]));
        IREE_ASSERT_OK(loom_serve_device_create_execution(
            device_owner_, owner_pools_[i], &owner_executions_[i]));
      } else {
        owner_pools_[i] = storage_owner_;
      }
      IREE_ASSERT_OK(Compile(i, "copy_raw"));
      const auto root = Root(i, 0);
      IREE_ASSERT_OK(loom_serve_weights_allocate(
          device_, transfer_, dispatch_,
          loom_serve_pool_storage(owner_pools_[i]), Pool(i),
          loom_serve_pool_weight_geometry(owner_pools_[i]), nullptr,
          loom_serve_pool_workspace(owner_pools_[i]), 0, 1, &root,
          host_allocator_, &plans_[i]));
      // Raw preparation records file ranges only. Each plan's queues and pool
      // establish its physical owner independently of this shared reflection.
      IREE_ASSERT_OK(
          Prepare(plans_[i], {i == 2 ? adapter_path_.path() : path_.path()}));
      IREE_ASSERT_OK(loom_serve_residency_group_allocate(
          Cache(i), 1, &plans_[i], host_allocator_, &parameter_groups_[i]));
      IREE_ASSERT_OK(loom_serve_residency_allocate(
          Cache(i), owner_executions_[i], 1, &parameter_groups_[i],
          host_allocator_, &residencies_[i]));
    }
  }

  void TearDown() override {
    if (deferred_backing_) {
      deferred_backing_->CompleteReturns();
    }
    FinishState();
    if (auxiliary_execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(auxiliary_execution_));
    }
    loom_serve_residency_free(auxiliary_residency_);
    loom_serve_execution_release(auxiliary_execution_);
    for (size_t i = 1; i < owner_executions_.size(); ++i) {
      if (owner_executions_[i]) {
        IREE_EXPECT_STATUS_IS(i == 2 ? failed_owner_status_ : IREE_STATUS_OK,
                              loom_serve_execution_drain(owner_executions_[i]));
      }
    }
    for (auto* residency : extra_residencies_) {
      loom_serve_residency_free(residency);
    }
    for (auto* buffer : state_buffers_) {
      IREE_EXPECT_OK(loom_serve_virtual_buffer_free(buffer));
    }
    iree_hal_semaphore_release(state_ready_);
    WeightsTest::TearDown();
    for (size_t i = 1; i < owner_executions_.size(); ++i) {
      loom_serve_execution_release(owner_executions_[i]);
      if (owner_pools_[i]) {
        IREE_EXPECT_OK(loom_serve_test_pool_trim_and_drain(owner_pools_[i], 0));
        EXPECT_FALSE(loom_serve_pool_is_in_use(owner_pools_[i]));
        IREE_EXPECT_OK(loom_serve_pool_free(owner_pools_[i]));
      }
    }
  }

  loom_serve_memory_pool_t* Pool(size_t owner) {
    return owner ? loom_serve_pool_memory(owner_pools_[owner]) : pool_;
  }

  loom_serve_residency_cache_t* Cache(size_t owner) {
    return owner ? loom_serve_pool_residency_cache(owner_pools_[owner])
                 : ResidencyCache();
  }

  void AllocateState(size_t slot, size_t owner, uint64_t bytes) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        Pool(owner), loom_serve_pool_state_geometry(owner_pools_[owner]), bytes,
        256, &state_usage_[slot], &state_buffers_[slot]));
  }

  void FinishState() {
    if (state_operation_) {
      loom_serve_memory_operation_await(state_operation_);
      IREE_EXPECT_OK(
          loom_serve_memory_operation_query_status(state_operation_));
      loom_serve_memory_operation_free(state_operation_);
      state_operation_ = nullptr;
    }
  }

  void ExpectUnactivated() {
    for (size_t i = 0; i < residencies_.size(); ++i) {
      EXPECT_EQ(loom_serve_residency_snapshot(residencies_[i]).pin_count, 0u);
      EXPECT_EQ(loom_serve_weights_activity(plans_[i]).activations, 0u);
      EXPECT_EQ(loom_serve_weights_statistics(plans_[i]).committed_bytes, 0u);
      EXPECT_EQ(loom_serve_weights_statistics(plans_[i]).pending_bytes, 0u);
    }
  }

  void ReadOwner(size_t owner) {
    IREE_ASSERT_OK(loom_serve_residency_await(residencies_[owner]));
    ASSERT_NO_FATAL_FAILURE(ReadOutputOn(owner_executions_[owner], owner));
    for (int32_t i = 0; i < 8; ++i) {
      EXPECT_EQ(output_[i], 30 + i);
    }
  }

  void AuxiliaryProgress() {
    auto*& execution = auxiliary_execution_;
    IREE_ASSERT_OK(loom_serve_device_create_execution(
        device_owner_, storage_owner_, &execution));
    auto*& residency = auxiliary_residency_;
    IREE_ASSERT_OK(loom_serve_residency_allocate(
        Cache(0), execution, 0, nullptr, host_allocator_, &residency));
    loom_serve_residency_admission_t admission;
    IREE_ASSERT_OK(loom_serve_residency_try_admit(1, &residency, &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    const uint32_t pattern = 0x13579bdf;
    const iree_hal_transfer_operation_t fill = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill =
            {
                .target_buffer = output_buffer_,
                .length = sizeof(output_),
                .pattern = &pattern,
                .pattern_length = sizeof(pattern),
            },
    };
    uint64_t value = 0;
    IREE_ASSERT_OK(loom_serve_execution_transfer(execution, 1, &fill, &value));
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = output_buffer_,
                .target = output_.data(),
                .length = sizeof(output_),
            },
    };
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(execution, 1, &download, &value));
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    for (int32_t value : output_) {
      EXPECT_EQ(static_cast<uint32_t>(value), pattern);
    }
    loom_serve_residency_cancel_admission(residency);
    loom_serve_residency_release(residency);
    loom_serve_residency_free(residency);
    residency = nullptr;
    loom_serve_execution_release(execution);
    execution = nullptr;
  }

  // Distinct budgets borrowing the process-owned device. Slot zero is borrowed
  // from the base fixture; remaining pools are owned here.
  std::array<loom_serve_pool_t*, 3> owner_pools_ = {};
  // The first execution is borrowed from the base; the remaining two are owned.
  std::array<loom_serve_execution_t*, 3> owner_executions_ = {};
  // Independent auxiliary work, owned through failed-assertion cleanup too.
  loom_serve_execution_t* auxiliary_execution_ = nullptr;
  // Auxiliary admission released only after its independent execution retires.
  loom_serve_residency_t* auxiliary_residency_ = nullptr;
  // Borrowers of existing groups, destroyed before their parameter owners.
  std::vector<loom_serve_residency_t*> extra_residencies_;
  // Independent pressure and request state, retired before physical pools.
  std::array<loom_serve_virtual_buffer_t*, 2> state_buffers_ = {};
  // Stable accounting through each state reservation's native release.
  std::array<loom_serve_memory_statistics_t, 2> state_usage_ = {};
  // One accepted caller-owned state union operation.
  loom_serve_memory_operation_t* state_operation_ = nullptr;
  // State readiness timeline belonging to the explicitly selected owner.
  iree_hal_semaphore_t* state_ready_ = nullptr;
  // Expected terminal readiness error on the last owner after file truncation.
  iree_status_code_t failed_owner_status_ = IREE_STATUS_OK;
};

TEST_P(ResidencyTest, LateOwnerDenialStartsNoEarlierLoads) {
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(0, 2, 2 * kSlabBytes));
    IREE_ASSERT_OK(
        loom_serve_virtual_buffer_commit(state_buffers_[0], 0, 2 * kSlabBytes));
    loom_serve_residency_admission_t denied;
    IREE_ASSERT_OK(loom_serve_residency_try_admit(
        residencies_.size(), residencies_.data(), &denied));
    ASSERT_EQ(denied, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    ExpectUnactivated();
    for (auto* residency : residencies_) {
      EXPECT_TRUE(loom_serve_residency_snapshot(residency).admission.waiting);
    }
    ASSERT_NO_FATAL_FAILURE(AuxiliaryProgress());
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(state_buffers_[0]));
    state_buffers_[0] = nullptr;
    IREE_ASSERT_OK(loom_serve_memory_pool_await(Pool(2)));
  }
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(
      residencies_.size(), residencies_.data(), &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  for (size_t i = 0; i < residencies_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    loom_serve_residency_cancel_admission(residencies_[i]);
    loom_serve_residency_release(residencies_[i]);
  }
}

class ElasticResidencyTest : public ResidencyTest {};

TEST_P(ElasticResidencyTest, ImpossiblePeerPreservesEarlierReclaimableHistory) {
  struct History {
    // Stable native state accounting owned by the fixture.
    const loom_serve_memory_statistics_t* memory;
    // Accepted logical release requests, not feasibility queries.
    uint32_t requests = 0;
    // Owner maintenance has not yet retired the accepted history release.
    bool pending = false;
  } history = {&state_usage_[0]};
  ASSERT_NO_FATAL_FAILURE(AllocateState(0, 0, 2 * kSlabBytes));
  ASSERT_NO_FATAL_FAILURE(AllocateState(1, 2, 2 * kSlabBytes));
  for (auto* state : state_buffers_) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state, 0, 2 * kSlabBytes));
  }
  loom_serve_residency_set_reclaimer(
      residencies_[0],
      {
          .self = &history,
          .request =
              [](void* self) {
                auto& history = *static_cast<History*>(self);
                if (history.pending) {
                  return false;
                }
                ++history.requests;
                history.pending = true;
                return true;
              },
          .is_pending =
              [](void* self) { return static_cast<History*>(self)->pending; },
          .forecast =
              [](void* self, const loom_serve_residency_forecast_t*,
                 loom_serve_residency_projection_t* projection) {
                const auto& history = *static_cast<History*>(self);
                *projection = {
                    .resident_bytes = history.memory->committed_bytes +
                                      history.memory->pending_bytes,
                };
                return iree_ok_status();
              },
      });
  // Owner zero could release history, but owner two has no eligible history.
  // The whole request is impossible regardless of membership traversal order.
  loom_serve_residency_t* reversed[] = {residencies_[2], residencies_[1],
                                        residencies_[0]};
  for (auto* members : {residencies_.data(), reversed}) {
    loom_serve_residency_admission_t admission;
    IREE_EXPECT_OK(loom_serve_residency_try_admit(3, members, &admission));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(history.requests, 0u);
    ExpectUnactivated();
  }
  // A later owner's failed cold forecast cannot authorize earlier eviction.
  loom_serve_residency_set_reclaimer(
      residencies_[2], {
                           .forecast =
                               [](void*, const loom_serve_residency_forecast_t*,
                                  loom_serve_residency_projection_t*) {
                                 return iree_make_status(
                                     IREE_STATUS_RESOURCE_EXHAUSTED,
                                     "forecast scratch allocation failed");
                               },
                       });
  loom_serve_residency_admission_t failed;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_residency_try_admit(
                            residencies_.size(), residencies_.data(), &failed));
  EXPECT_EQ(history.requests, 0u);
  ExpectUnactivated();
  loom_serve_residency_set_reclaimer(residencies_[2], {});
  IREE_EXPECT_OK(loom_serve_virtual_buffer_free(state_buffers_[1]));
  state_buffers_[1] = nullptr;
  IREE_EXPECT_OK(loom_serve_memory_pool_await(Pool(2)));
  loom_serve_residency_admission_t admission;
  IREE_EXPECT_OK(loom_serve_residency_try_admit(
      residencies_.size(), residencies_.data(), &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  EXPECT_EQ(history.requests, 1u);
  ExpectUnactivated();
  // Only actual owner maintenance can make this backing available.
  IREE_EXPECT_OK(loom_serve_virtual_buffer_free(state_buffers_[0]));
  state_buffers_[0] = nullptr;
  IREE_EXPECT_OK(loom_serve_memory_pool_await(Pool(0)));
  loom_serve_residency_set_reclaimer(residencies_[0], {});
  IREE_ASSERT_OK(loom_serve_residency_try_admit(
      residencies_.size(), residencies_.data(), &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  for (size_t i = 0; i < residencies_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    loom_serve_residency_cancel_admission(residencies_[i]);
    loom_serve_residency_release(residencies_[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(Owners, ElasticResidencyTest, ::testing::Values(true));

TEST_P(ResidencyTest, NonzeroStateOwnerCommitsOneUnionAfterCompleteAdmission) {
  std::array<loom_serve_memory_range_t, 2> ranges = {};
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(1, 2, 2 * kSlabBytes));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &state_ready_));
    ranges = {
        {{state_buffers_[1], 0, 32}, {state_buffers_[1], kSlabBytes, 32}}};
    loom_serve_residency_admission_t denied;
    IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
        residencies_.size(), residencies_.data(), 2, ranges.size(),
        ranges.data(), nullptr, state_ready_, 1, &denied, &state_operation_));
    ASSERT_EQ(denied, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(state_operation_, nullptr);
    EXPECT_EQ(state_usage_[1].committed_bytes, 0u);
    ExpectUnactivated();
    ranges[1].offset = 16;
  }
  for (uint64_t pin = 1; pin <= 2; ++pin) {
    loom_serve_residency_admission_t admission;
    IREE_ASSERT_OK(loom_serve_residency_try_admit_state(
        residencies_.size(), residencies_.data(), 2,
        GetParam() ? ranges.size() : 0, ranges.data(), nullptr, state_ready_,
        pin, &admission, &state_operation_));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    FinishState();
    if (GetParam()) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(state_ready_, pin,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
      EXPECT_EQ(state_usage_[1].committed_bytes, kSlabBytes);
    }
    for (size_t i = 0; i < residencies_.size(); ++i) {
      EXPECT_EQ(loom_serve_residency_snapshot(residencies_[i]).pin_count, pin);
      EXPECT_EQ(loom_serve_weights_activity(plans_[i]).activations, 1u);
      ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    }
  }
  for (auto* residency : residencies_) {
    loom_serve_residency_cancel_admission(residency);
    loom_serve_residency_release(residency);
    loom_serve_residency_release(residency);
  }
}

TEST_P(ResidencyTest, OverlappingSetsShareFifoOrderAndAllowAuxiliaryProgress) {
  loom_serve_residency_t* shared = nullptr;
  IREE_ASSERT_OK(loom_serve_residency_allocate(Cache(1), owner_executions_[1],
                                               1, &parameter_groups_[1],
                                               host_allocator_, &shared));
  extra_residencies_.push_back(shared);
  loom_serve_residency_t* first[] = {residencies_[0], residencies_[1]};
  loom_serve_residency_t* second[] = {residencies_[2], shared};
  loom_serve_residency_admission_t admission;
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(0, 1, 2 * kSlabBytes));
    IREE_ASSERT_OK(
        loom_serve_virtual_buffer_commit(state_buffers_[0], 0, 2 * kSlabBytes));
    IREE_ASSERT_OK(loom_serve_residency_try_admit(2, first, &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    IREE_ASSERT_OK(loom_serve_residency_try_admit(2, second, &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
    ExpectUnactivated();
    EXPECT_TRUE(loom_serve_residency_snapshot(shared).admission.waiting);
    ASSERT_NO_FATAL_FAILURE(AuxiliaryProgress());
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(state_buffers_[0]));
    state_buffers_[0] = nullptr;
    IREE_ASSERT_OK(loom_serve_memory_pool_await(Pool(1)));
    IREE_ASSERT_OK(loom_serve_residency_try_admit(2, second, &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  }
  IREE_ASSERT_OK(loom_serve_residency_try_admit(2, first, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  for (auto* residency : first) {
    loom_serve_residency_cancel_admission(residency);
  }
  IREE_ASSERT_OK(loom_serve_residency_try_admit(2, second, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  for (size_t i = 0; i < residencies_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
  }
  for (auto* residency : second) {
    loom_serve_residency_cancel_admission(residency);
    loom_serve_residency_release(residency);
  }
  for (auto* residency : first) {
    loom_serve_residency_release(residency);
  }
}

TEST_P(ResidencyTest, AcceptedOwnersSurviveLaterSourceFailure) {
  std::filesystem::resize_file(adapter_path_.path(), 0);
  loom_serve_residency_admission_t admission;
  auto status = loom_serve_residency_try_admit(residencies_.size(),
                                               residencies_.data(), &admission);
  // The I/O failure can arrive during activation or on its imported readiness.
  // Either result must keep all accepted owners protected until retirement.
  const bool imported = iree_status_is_ok(status);
  if (imported) {
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    status = loom_serve_residency_await(residencies_[2]);
  }
  EXPECT_NE(iree_status_code(status), IREE_STATUS_OK);
  failed_owner_status_ = imported ? iree_status_code(status) : IREE_STATUS_OK;
  iree_status_free(status);
  for (auto* residency : residencies_) {
    EXPECT_EQ(loom_serve_residency_snapshot(residency).pin_count, 1u);
  }
  ASSERT_NO_FATAL_FAILURE(ReadOwner(0));
  ASSERT_NO_FATAL_FAILURE(ReadOwner(1));
  ASSERT_NO_FATAL_FAILURE(AuxiliaryProgress());
  // The base owner releases all consumer roots, joins actual weight-view
  // retirement, and only then frees residencies and physical backing.
}

INSTANTIATE_TEST_SUITE_P(Owners, ResidencyTest, ::testing::Values(false, true));

class DeferredResidencyTest : public ResidencyTest {
 protected:
  void PrepareBacking() override { UseDeferredBacking(); }

  void DeferReturn() {
    ASSERT_NO_FATAL_FAILURE(AllocateState(0, 0, 2 * kSlabBytes));
    IREE_ASSERT_OK(
        loom_serve_virtual_buffer_commit(state_buffers_[0], 0, 2 * kSlabBytes));
    loom_serve_virtual_buffer_begin_trim(state_buffers_[0]);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(state_buffers_[0]));
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
    deferred_backing_->DeferReturns();
    loom_serve_memory_pool_trim(pool_, 0);
    deferred_backing_->WaitForReturn();
  }
};

TEST_P(DeferredResidencyTest, LateNativeReturnUndoesAllTemporaryProtection) {
  ASSERT_NO_FATAL_FAILURE(DeferReturn());
  IREE_ASSERT_OK(
      loom_serve_residency_prepare_working_set(residencies_[1], 0, 16));
  loom_serve_residency_t* owners[] = {residencies_[1], residencies_[2],
                                      residencies_[0]};
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(3, owners, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  ExpectUnactivated();
  // A parameter-free peer can use the first owner's entire workspace budget.
  // Any output credit retained by the denied attempt would reject it.
  loom_serve_residency_t* independent = nullptr;
  IREE_ASSERT_OK(loom_serve_residency_allocate(Cache(1), owner_executions_[1],
                                               0, nullptr, host_allocator_,
                                               &independent));
  extra_residencies_.push_back(independent);
  IREE_ASSERT_OK(
      loom_serve_residency_prepare_working_set(independent, 2 * kSlabBytes, 0));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(1, &independent, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_residency_release(independent);
  deferred_backing_->CompleteReturns();
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(3, owners, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  for (size_t i = 0; i < residencies_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    loom_serve_residency_cancel_admission(residencies_[i]);
    loom_serve_residency_release(residencies_[i]);
  }
}

TEST_P(DeferredResidencyTest, DeniedSetPreservesEarlierAcceptedPin) {
  IREE_ASSERT_OK(
      loom_serve_residency_prepare_working_set(residencies_[1], 0, 16));
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[1]));
  ASSERT_NO_FATAL_FAILURE(ReadOwner(1));
  ASSERT_NO_FATAL_FAILURE(DeferReturn());
  loom_serve_residency_t* owners[] = {residencies_[1], residencies_[2],
                                      residencies_[0]};
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(3, owners, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  for (size_t i = 0; i < residencies_.size(); ++i) {
    EXPECT_EQ(loom_serve_residency_snapshot(residencies_[i]).pin_count,
              i == 1 ? 1u : 0u);
    EXPECT_EQ(loom_serve_weights_activity(plans_[i]).activations,
              i == 1 ? 1u : 0u);
  }
  ASSERT_NO_FATAL_FAILURE(ReadOwner(1));
  deferred_backing_->CompleteReturns();
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(3, owners, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  for (size_t i = 0; i < residencies_.size(); ++i) {
    EXPECT_EQ(loom_serve_residency_snapshot(residencies_[i]).pin_count,
              i == 1 ? 2u : 1u);
    EXPECT_EQ(loom_serve_weights_activity(plans_[i]).activations, 1u);
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    loom_serve_residency_cancel_admission(residencies_[i]);
    loom_serve_residency_release(residencies_[i]);
  }
  loom_serve_residency_release(residencies_[1]);
}

INSTANTIATE_TEST_SUITE_P(Owners, DeferredResidencyTest,
                         ::testing::Values(true));

}  // namespace
}  // namespace loom_serve::testing
