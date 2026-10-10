// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/semaphore.h"

#include <cstdlib>
#include <initializer_list>
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

class FrontierBuilder {
 public:
  iree_async_frontier_t* Build(
      std::initializer_list<iree_async_frontier_entry_t> entries) {
    storage_.resize(sizeof(iree_async_frontier_t) +
                    entries.size() * sizeof(iree_async_frontier_entry_t));
    auto* frontier = reinterpret_cast<iree_async_frontier_t*>(storage_.data());
    iree_async_frontier_initialize(frontier,
                                   static_cast<uint8_t>(entries.size()));
    iree_host_size_t index = 0;
    for (const auto& entry : entries) {
      frontier->entries[index++] = entry;
    }
    return frontier;
  }

  iree_async_frontier_t* BuildDistinctQueues(uint8_t entry_count) {
    storage_.resize(sizeof(iree_async_frontier_t) +
                    entry_count * sizeof(iree_async_frontier_entry_t));
    auto* frontier = reinterpret_cast<iree_async_frontier_t*>(storage_.data());
    iree_async_frontier_initialize(frontier, entry_count);
    for (uint8_t i = 0; i < entry_count; ++i) {
      frontier->entries[i] = {test_queue_axis(i), static_cast<uint64_t>(i) + 1};
    }
    return frontier;
  }

 private:
  std::vector<uint8_t> storage_;
};

class SemaphoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static uintptr_t fake_device_storage = 0;
    device_ = reinterpret_cast<iree_hal_device_t*>(&fake_device_storage);
    IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                   IREE_HAL_SEMAPHORE_FLAG_NONE,
                                   /*initial_value=*/0, &semaphore_));
  }

  void TearDown() override { iree_hal_semaphore_release(semaphore_); }

  iree_status_t CreateSemaphore(
      iree_hal_queue_family_affinity_t queue_family_affinity,
      iree_hal_semaphore_flags_t flags, uint64_t initial_value,
      iree_hal_semaphore_t** out_semaphore) {
    return iree_hal_amd_xdna_semaphore_create(
        device_, test_proactor(), /*epoch_wait=*/{}, queue_family_affinity,
        initial_value, flags, iree_allocator_system(), out_semaphore);
  }

  iree_hal_device_t* device_ = nullptr;
  iree_hal_semaphore_t* semaphore_ = nullptr;
};

TEST_F(SemaphoreTest, PreservesCreatingDeviceFlagsAndAffinity) {
  EXPECT_TRUE(iree_hal_amd_xdna_semaphore_isa(semaphore_));
  EXPECT_TRUE(iree_hal_amd_xdna_semaphore_is_local(semaphore_, device_));

  static uintptr_t foreign_device_storage = 0;
  auto* foreign_device =
      reinterpret_cast<iree_hal_device_t*>(&foreign_device_storage);
  EXPECT_FALSE(
      iree_hal_amd_xdna_semaphore_is_local(semaphore_, foreign_device));

  const iree_hal_queue_family_affinity_t affinity =
      iree_hal_make_queue_family_affinity(3);
  const iree_hal_semaphore_flags_t flags =
      IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
      IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER;
  iree_hal_semaphore_t* configured = nullptr;
  IREE_ASSERT_OK(
      CreateSemaphore(affinity, flags, /*initial_value=*/0, &configured));
  EXPECT_EQ(iree_hal_amd_xdna_semaphore_flags(configured), flags);
  EXPECT_EQ(iree_hal_amd_xdna_semaphore_queue_family_affinity(configured),
            affinity);
  iree_hal_semaphore_release(configured);
}

TEST_F(SemaphoreTest, RetainsSixtyFourEntryFrontier) {
  FrontierBuilder builder;
  const iree_async_frontier_t* frontier =
      builder.BuildDistinctQueues(IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY);
  EXPECT_TRUE(iree_hal_amd_xdna_semaphore_publish_signal(
      semaphore_, test_queue_axis(63), frontier,
      /*producer_frontier_exact=*/true, /*producer_epoch=*/64,
      /*producer_value=*/1));

  const uint8_t actual_count = iree_async_semaphore_query_frontier(
      reinterpret_cast<iree_async_semaphore_t*>(semaphore_), nullptr, 0);
  EXPECT_EQ(actual_count, IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY);
}

TEST_F(SemaphoreTest, OverflowPreservesLowerBoundAndClearsMetadata) {
  FrontierBuilder builder;
  const iree_async_frontier_t* initial =
      builder.Build({{test_queue_axis(0), 1}});
  ASSERT_TRUE(iree_hal_amd_xdna_semaphore_publish_signal(
      semaphore_, test_queue_axis(0), initial,
      /*producer_frontier_exact=*/true, /*producer_epoch=*/1,
      /*producer_value=*/1));

  const iree_async_frontier_t* overflow =
      builder.BuildDistinctQueues(IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY + 1);
  EXPECT_FALSE(iree_hal_amd_xdna_semaphore_publish_signal(
      semaphore_, test_queue_axis(0), overflow,
      /*producer_frontier_exact=*/true, /*producer_epoch=*/2,
      /*producer_value=*/2));
  EXPECT_EQ(
      iree_async_semaphore_query_frontier(
          reinterpret_cast<iree_async_semaphore_t*>(semaphore_), nullptr, 0),
      1u);

  iree_hal_submitted_signal_flags_t flags = IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE;
  iree_async_axis_t producer_axis = 0;
  uint64_t producer_epoch = 0;
  uint64_t producer_value = 0;
  EXPECT_FALSE(iree_hal_submitted_signal_load(
      iree_hal_amd_xdna_semaphore_submitted_signal(semaphore_), &flags,
      &producer_axis, &producer_epoch, &producer_value));
}

TEST_F(SemaphoreTest, ExactnessRequiresCompleteDominatingProducerFrontier) {
  const iree_async_axis_t producer_axis = test_queue_axis(2);
  FrontierBuilder builder;
  const iree_async_frontier_t* exact_frontier =
      builder.Build({{producer_axis, 7}});
  ASSERT_TRUE(iree_hal_amd_xdna_semaphore_publish_signal(
      semaphore_, producer_axis, exact_frontier,
      /*producer_frontier_exact=*/true, /*producer_epoch=*/7,
      /*producer_value=*/1));

  iree_hal_submitted_signal_flags_t flags = IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE;
  iree_async_axis_t cached_axis = 0;
  uint64_t cached_epoch = 0;
  uint64_t cached_value = 0;
  ASSERT_TRUE(iree_hal_submitted_signal_load(
      iree_hal_amd_xdna_semaphore_submitted_signal(semaphore_), &flags,
      &cached_axis, &cached_epoch, &cached_value));
  EXPECT_EQ(flags, IREE_HAL_SUBMITTED_SIGNAL_FLAG_VALID |
                       IREE_HAL_SUBMITTED_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT);
  EXPECT_EQ(cached_axis, producer_axis);
  EXPECT_EQ(cached_epoch, 7u);
  EXPECT_EQ(cached_value, 1u);

  const iree_async_frontier_t* inexact_frontier =
      builder.Build({{producer_axis, 8}});
  ASSERT_TRUE(iree_hal_amd_xdna_semaphore_publish_signal(
      semaphore_, producer_axis, inexact_frontier,
      /*producer_frontier_exact=*/false, /*producer_epoch=*/8,
      /*producer_value=*/2));
  ASSERT_TRUE(iree_hal_submitted_signal_load(
      iree_hal_amd_xdna_semaphore_submitted_signal(semaphore_), &flags,
      &cached_axis, &cached_epoch, &cached_value));
  EXPECT_EQ(flags, IREE_HAL_SUBMITTED_SIGNAL_FLAG_VALID);
}

TEST_F(SemaphoreTest, IndependentFanInIsNotProducerExact) {
  FrontierBuilder builder;
  const iree_async_axis_t first_axis = test_queue_axis(1);
  const iree_async_axis_t second_axis = test_queue_axis(2);
  ASSERT_TRUE(iree_hal_amd_xdna_semaphore_publish_signal(
      semaphore_, first_axis, builder.Build({{first_axis, 4}}),
      /*producer_frontier_exact=*/true, /*producer_epoch=*/4,
      /*producer_value=*/1));
  ASSERT_TRUE(iree_hal_amd_xdna_semaphore_publish_signal(
      semaphore_, second_axis, builder.Build({{second_axis, 9}}),
      /*producer_frontier_exact=*/true, /*producer_epoch=*/9,
      /*producer_value=*/2));

  iree_hal_submitted_signal_flags_t flags = IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE;
  iree_async_axis_t cached_axis = 0;
  uint64_t cached_epoch = 0;
  uint64_t cached_value = 0;
  ASSERT_TRUE(iree_hal_submitted_signal_load(
      iree_hal_amd_xdna_semaphore_submitted_signal(semaphore_), &flags,
      &cached_axis, &cached_epoch, &cached_value));
  EXPECT_EQ(flags, IREE_HAL_SUBMITTED_SIGNAL_FLAG_VALID);
  EXPECT_EQ(cached_axis, second_axis);
  EXPECT_EQ(cached_epoch, 9u);
  EXPECT_EQ(cached_value, 2u);
}

TEST_F(SemaphoreTest, HostTimelineOperationsRemainAvailable) {
  iree_hal_semaphore_t* initialized = nullptr;
  IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                 IREE_HAL_SEMAPHORE_FLAG_NONE,
                                 /*initial_value=*/2, &initialized));
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(initialized, &value));
  EXPECT_EQ(value, 2u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(initialized, 5, nullptr));
  IREE_EXPECT_OK(iree_hal_semaphore_wait(
      initialized, 5, iree_immediate_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_query(initialized, &value));
  EXPECT_EQ(value, 5u);
  iree_hal_semaphore_release(initialized);
}

}  // namespace
