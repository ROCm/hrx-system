// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_frontier.h"

#include <cstdlib>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_async_proactor_t* test_proactor() {
  static iree_async_proactor_t* proactor = nullptr;
  if (!proactor) {
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
    atexit([] {
      iree_async_proactor_release(proactor);
      proactor = nullptr;
    });
  }
  return proactor;
}

static iree_async_axis_t test_queue_axis(uint8_t queue_index) {
  return iree_async_axis_make_queue(/*session_epoch=*/1, /*machine_index=*/0,
                                    /*device_index=*/0, queue_index,
                                    /*queue_incarnation=*/0);
}

class QueueFrontierTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (iree_hal_semaphore_t* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
  }

  iree_hal_semaphore_t* CreateSemaphore() {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_EXPECT_OK(iree_hal_amd_xdna_semaphore_create(
        device_, test_proactor(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        iree_allocator_system(), &semaphore));
    semaphores_.push_back(semaphore);
    return semaphore;
  }

  void Publish(iree_hal_semaphore_t* semaphore, iree_async_axis_t axis,
               uint64_t epoch, uint64_t value) {
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, axis, epoch);
    ASSERT_TRUE(iree_hal_amd_xdna_semaphore_publish_signal(
        semaphore, axis,
        iree_async_single_frontier_as_const_frontier(&frontier),
        /*producer_frontier_exact=*/true, epoch, value));
  }

  void Complete(iree_hal_semaphore_t* semaphore, iree_async_axis_t axis,
                uint64_t epoch, uint64_t value) {
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, axis, epoch);
    IREE_ASSERT_OK(iree_async_semaphore_signal_untainted(
        reinterpret_cast<iree_async_semaphore_t*>(semaphore), value,
        iree_async_single_frontier_as_const_frontier(&frontier)));
  }

  iree_hal_amd_xdna_wait_resolution_t Resolve(
      iree_hal_semaphore_list_t waits,
      const iree_hal_amd_xdna_frontier_state_t* accepted_state,
      iree_hal_amd_xdna_wait_resolution_flags_t flags,
      iree_hal_amd_xdna_frontier_state_t* out_state) {
    iree_hal_amd_xdna_wait_resolution_t resolution =
        IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER;
    IREE_EXPECT_OK(iree_hal_amd_xdna_frontier_resolve_waits(
        device_, waits, accepted_state, /*initial_state=*/nullptr, flags,
        out_state, &resolution,
        /*out_deferred_wait_index=*/nullptr));
    return resolution;
  }

  uintptr_t device_storage_ = 0;
  iree_hal_device_t* device_ =
      reinterpret_cast<iree_hal_device_t*>(&device_storage_);
  std::vector<iree_hal_semaphore_t*> semaphores_;
};

TEST_F(QueueFrontierTest, AcceptedExactProducerUsesNativeFifo) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();
  const iree_async_axis_t producer_axis = test_queue_axis(2);
  Publish(semaphore, producer_axis, /*epoch=*/7, /*value=*/1);

  iree_hal_amd_xdna_frontier_state_t accepted_state;
  iree_hal_amd_xdna_frontier_state_initialize(&accepted_state);
  iree_hal_amd_xdna_frontier_state_advance(&accepted_state, producer_axis, 7);
  iree_hal_amd_xdna_frontier_state_t state;
  uint64_t value = 1;
  EXPECT_EQ(Resolve({1, &semaphore, &value}, &accepted_state,
                    IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_ALLOW_ACCEPTED_FIFO,
                    &state),
            IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY);
  EXPECT_TRUE(state.exact);
  EXPECT_TRUE(
      iree_hal_amd_xdna_frontier_state_dominates(&state, producer_axis, 7));
}

TEST_F(QueueFrontierTest, UndominatedLaterProducerDefers) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();
  const iree_async_axis_t producer_axis = test_queue_axis(3);
  Publish(semaphore, producer_axis, /*epoch=*/9, /*value=*/2);

  iree_hal_amd_xdna_frontier_state_t accepted_state;
  iree_hal_amd_xdna_frontier_state_initialize(&accepted_state);
  iree_hal_amd_xdna_frontier_state_t state;
  uint64_t value = 1;
  EXPECT_EQ(Resolve({1, &semaphore, &value}, &accepted_state,
                    IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_ALLOW_ACCEPTED_FIFO,
                    &state),
            IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER);

  iree_hal_amd_xdna_frontier_state_advance(&accepted_state, producer_axis, 9);
  EXPECT_EQ(Resolve({1, &semaphore, &value}, &accepted_state,
                    IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_ALLOW_ACCEPTED_FIFO,
                    &state),
            IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY);
}

TEST_F(QueueFrontierTest, DeferredIndexNamesFirstUnresolvedWait) {
  iree_hal_semaphore_t* reached = CreateSemaphore();
  iree_hal_semaphore_t* unresolved = CreateSemaphore();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(reached, 1, nullptr));
  iree_hal_semaphore_t* semaphores[] = {reached, unresolved};
  uint64_t values[] = {1, 1};
  iree_hal_amd_xdna_frontier_state_t state;
  iree_hal_amd_xdna_wait_resolution_t resolution =
      IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY;
  iree_host_size_t deferred_wait_index = IREE_HOST_SIZE_MAX;
  IREE_ASSERT_OK(iree_hal_amd_xdna_frontier_resolve_waits(
      device_, {IREE_ARRAYSIZE(semaphores), semaphores, values}, nullptr,
      /*initial_state=*/nullptr, IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE,
      &state, &resolution, &deferred_wait_index));
  EXPECT_EQ(resolution, IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER);
  EXPECT_EQ(deferred_wait_index, 1u);
}

TEST_F(QueueFrontierTest, ReachedHeterogeneousFanInRemainsExact) {
  iree_hal_semaphore_t* first = CreateSemaphore();
  iree_hal_semaphore_t* second = CreateSemaphore();
  const iree_async_axis_t first_axis = test_queue_axis(4);
  const iree_async_axis_t second_axis = test_queue_axis(7);
  Publish(first, first_axis, /*epoch=*/3, /*value=*/1);
  Complete(first, first_axis, /*epoch=*/3, /*value=*/1);
  Publish(second, second_axis, /*epoch=*/11, /*value=*/1);
  Complete(second, second_axis, /*epoch=*/11, /*value=*/1);

  iree_hal_semaphore_t* semaphores[] = {first, second};
  uint64_t values[] = {1, 1};
  iree_hal_amd_xdna_frontier_state_t state;
  EXPECT_EQ(Resolve({IREE_ARRAYSIZE(semaphores), semaphores, values}, nullptr,
                    IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE, &state),
            IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY);
  EXPECT_TRUE(state.exact);
  EXPECT_TRUE(
      iree_hal_amd_xdna_frontier_state_dominates(&state, first_axis, 3));
  EXPECT_TRUE(
      iree_hal_amd_xdna_frontier_state_dominates(&state, second_axis, 11));
}

TEST_F(QueueFrontierTest, SuffixResolutionAccumulatesReachedPrefix) {
  iree_hal_semaphore_t* prefix = CreateSemaphore();
  iree_hal_semaphore_t* suffix = CreateSemaphore();
  const iree_async_axis_t prefix_axis = test_queue_axis(4);
  const iree_async_axis_t suffix_axis = test_queue_axis(7);
  const iree_async_axis_t accepted_axis = test_queue_axis(9);
  Publish(prefix, prefix_axis, /*epoch=*/3, /*value=*/1);
  Complete(prefix, prefix_axis, /*epoch=*/3, /*value=*/1);
  Publish(suffix, suffix_axis, /*epoch=*/11, /*value=*/1);
  Complete(suffix, suffix_axis, /*epoch=*/11, /*value=*/1);

  uint64_t value = 1;
  iree_hal_amd_xdna_frontier_state_t state;
  EXPECT_EQ(Resolve({1, &prefix, &value}, nullptr,
                    IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE, &state),
            IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY);
  iree_hal_amd_xdna_frontier_state_t accepted_state;
  iree_hal_amd_xdna_frontier_state_initialize(&accepted_state);
  iree_hal_amd_xdna_frontier_state_advance(&accepted_state, accepted_axis, 5);
  iree_hal_amd_xdna_wait_resolution_t resolution =
      IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_DEFER;
  IREE_ASSERT_OK(iree_hal_amd_xdna_frontier_resolve_waits(
      device_, {1, &suffix, &value}, &accepted_state, &state,
      IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE, &state, &resolution,
      /*out_deferred_wait_index=*/nullptr));

  EXPECT_EQ(resolution, IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY);
  EXPECT_TRUE(state.exact);
  EXPECT_TRUE(
      iree_hal_amd_xdna_frontier_state_dominates(&state, prefix_axis, 3));
  EXPECT_TRUE(
      iree_hal_amd_xdna_frontier_state_dominates(&state, suffix_axis, 11));
  EXPECT_TRUE(
      iree_hal_amd_xdna_frontier_state_dominates(&state, accepted_axis, 5));
}

TEST_F(QueueFrontierTest, ReachedIndependentProducerDoesNotImportFutureAxis) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();
  const iree_async_axis_t future_axis = test_queue_axis(4);
  const iree_async_axis_t completed_axis = test_queue_axis(7);
  Publish(semaphore, future_axis, /*epoch=*/3, /*value=*/1);
  Publish(semaphore, completed_axis, /*epoch=*/11, /*value=*/2);
  Complete(semaphore, completed_axis, /*epoch=*/11, /*value=*/2);

  iree_hal_amd_xdna_frontier_state_t state;
  uint64_t value = 2;
  EXPECT_EQ(Resolve({1, &semaphore, &value}, nullptr,
                    IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE, &state),
            IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY);
  EXPECT_FALSE(state.exact);
  EXPECT_FALSE(
      iree_hal_amd_xdna_frontier_state_dominates(&state, future_axis, 3));
  EXPECT_TRUE(
      iree_hal_amd_xdna_frontier_state_dominates(&state, completed_axis, 11));
}

TEST_F(QueueFrontierTest, TaintedReachedValueMakesSnapshotInexact) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();
  const iree_async_axis_t producer_axis = test_queue_axis(5);
  Publish(semaphore, producer_axis, /*epoch=*/4, /*value=*/1);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 1, nullptr));

  iree_hal_amd_xdna_frontier_state_t state;
  uint64_t value = 1;
  EXPECT_EQ(Resolve({1, &semaphore, &value}, nullptr,
                    IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE, &state),
            IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY);
  EXPECT_FALSE(state.exact);
  EXPECT_EQ(state.frontier.entry_count, 0u);
}

TEST_F(QueueFrontierTest, CapacityOverflowPreservesLowerBound) {
  std::vector<iree_hal_semaphore_t*> waits;
  std::vector<uint64_t> values;
  waits.reserve(IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY + 1);
  values.resize(IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY + 1, 1);
  for (uint8_t i = 0; i < IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY + 1; ++i) {
    iree_hal_semaphore_t* semaphore = CreateSemaphore();
    const iree_async_axis_t axis = test_queue_axis(i);
    Publish(semaphore, axis, /*epoch=*/1, /*value=*/1);
    Complete(semaphore, axis, /*epoch=*/1, /*value=*/1);
    waits.push_back(semaphore);
  }

  iree_hal_amd_xdna_frontier_state_t state;
  EXPECT_EQ(Resolve({waits.size(), waits.data(), values.data()}, nullptr,
                    IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE, &state),
            IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY);
  EXPECT_FALSE(state.exact);
  EXPECT_EQ(state.frontier.entry_count, IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY);
}

TEST_F(QueueFrontierTest, SemaphoreFailurePropagates) {
  iree_hal_semaphore_t* semaphore = CreateSemaphore();
  iree_hal_semaphore_fail(semaphore,
                          iree_status_from_code(IREE_STATUS_ABORTED));
  uint64_t value = 1;
  iree_hal_amd_xdna_frontier_state_t state;
  iree_hal_amd_xdna_wait_resolution_t resolution =
      IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_READY;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_amd_xdna_frontier_resolve_waits(
          device_, {1, &semaphore, &value}, nullptr,
          /*initial_state=*/nullptr,
          IREE_HAL_AMD_XDNA_WAIT_RESOLUTION_FLAG_NONE, &state, &resolution,
          /*out_deferred_wait_index=*/nullptr));
}

}  // namespace
