// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::amd::gpu::cts::GpuNativeTest;

class GpuNativeTimestampTest : public GpuNativeTest,
                               public ::testing::WithParamInterface<
                                   std::pair<amdf_queue_command_type_t, bool>> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return GetParam().first;
  }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    options_.queue.ring_byte_length = 4096;
    options_.queue.retirement_capacity = 4;
    options_.queue.argument_capacity = 16;
    Create();
    if (HasFatalFailure()) {
      return;
    }
    const auto* family = iree_hal_queue_family(queues_[0]);
    ASSERT_GT(family->spec->timestamp_frequency_hz, 0u);
    ASSERT_EQ(family->spec->timestamp_valid_bits, 64u);
    const iree_hal_pool_family_access_t access = {
        .family = family,
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &access,
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(group_, scope, &options,
                                             iree_allocator_system(), &pool_));
    for (auto& semaphore : semaphores_) {
      CreateSemaphore(&semaphore);
    }
  }

  void TearDown() override {
    iree_hal_buffer_release(buffer_);
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_hal_pool_release(pool_);
    GpuNativeTest::TearDown();
  }

  void Allocate(uint64_t byte_length) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(byte_length, 8),
        iree_infinite_timeout(), &buffer_));
  }

  void Wait(size_t index, uint64_t value) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphores_[index], value,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
  }

  const iree_hal_queue_barriers_t* Barriers() {
    return GetParam().second ? &barriers_ : nullptr;
  }

  // Pool outlives accepted operations and their borrowed buffer ownership.
  iree_hal_pool_t* pool_ = nullptr;
  // Caller reference, deliberately releasable before submitted work completes.
  iree_hal_buffer_t* buffer_ = nullptr;
  // Independent host gate, allocation, operation chain, and download timelines.
  std::array<iree_hal_semaphore_t*, 4> semaphores_ = {};
  // No implicit payload work; callers can supply a separate terminal release.
  iree_hal_barrier_list_t empty_ = {};
  // Tests default boundaries and explicitly empty payload boundaries.
  iree_hal_queue_barriers_t barriers_ = {&empty_, &empty_};
};

TEST_P(GpuNativeTimestampTest, PendingTargetSurvivesConsumerBeforeProducer) {
  constexpr uint64_t kGuard = UINT64_C(0xA57C99E1A57C99E1);
  uint64_t one = 1, two = 2, three = 3;
  const auto request = iree_hal_make_pool_reservation_request(32, 8);
  IREE_ASSERT_OK(iree_hal_queue_alloca(queues_[0], {1, &semaphores_[0], &one},
                                       {1, &semaphores_[1], &one}, pool_, 1,
                                       &request, &buffer_));
  // Native address resolution must wait for allocation and the producer fill.
  IREE_ASSERT_OK(iree_hal_queue_timestamp(
      queues_[1], {1, &semaphores_[2], &one}, {1, &semaphores_[2], &two},
      buffer_, 8, Barriers(), IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {1, &semaphores_[1], &one},
                                     {1, &semaphores_[2], &one}, buffer_, 0, 32,
                                     &kGuard, 4, nullptr, 0));
  // The empty-boundary form needs the explicit publication before host use.
  IREE_ASSERT_OK(iree_hal_queue_barrier(queues_[1], {1, &semaphores_[2], &two},
                                        {1, &semaphores_[2], &three}, nullptr,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  std::array<uint64_t, 4> output = {};
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {1, &semaphores_[2], &three}, {1, &semaphores_[3], &one},
      buffer_, 0, output.data(), sizeof(output), nullptr));
  iree_hal_buffer_release(buffer_);
  buffer_ = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores_[0], one, nullptr));
  Wait(3, one);
  EXPECT_EQ(output[0], kGuard);
  EXPECT_NE(output[1], kGuard);
  EXPECT_GT(output[1], 0u);
  EXPECT_EQ(output[2], kGuard);
  EXPECT_EQ(output[3], kGuard);
}

TEST_P(GpuNativeTimestampTest, SameFamilyTicksRemainOrderedAcrossRingReuse) {
  constexpr size_t kCount = 129;
  Allocate((kCount + 2) * sizeof(uint64_t));
  if (HasFatalFailure()) {
    return;
  }
  uint64_t value = 1;
  const uint32_t guard = 0xA57C99E1;
  IREE_ASSERT_OK(iree_hal_queue_fill(
      queues_[0], {}, {1, &semaphores_[2], &value}, buffer_, 0,
      (kCount + 2) * sizeof(uint64_t), &guard, sizeof(guard), nullptr, 0));
  for (size_t i = 0; i < kCount; ++i) {
    uint64_t previous = value++;
    IREE_ASSERT_OK(iree_hal_queue_timestamp(
        queues_[i % 2], {1, &semaphores_[2], &previous},
        {1, &semaphores_[2], &value}, buffer_, (i + 1) * sizeof(uint64_t),
        Barriers(), IREE_HAL_TIMESTAMP_FLAG_NONE));
  }
  // Both queues publish their writes before the host download. The public
  // ordering chain already establishes comparability of the device ticks.
  std::array<uint64_t, 2> released = {1, 1};
  for (size_t i = 0; i < queues_.size(); ++i) {
    IREE_ASSERT_OK(
        iree_hal_queue_barrier(queues_[i], {1, &semaphores_[2], &value},
                               {1, &semaphores_[i], &released[i]}, nullptr, 0));
  }
  std::vector<uint64_t> output(kCount + 2);
  uint64_t done = 1;
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[0], {2, semaphores_.data(), released.data()},
      {1, &semaphores_[3], &done}, buffer_, 0, output.data(),
      output.size() * sizeof(uint64_t), nullptr));
  Wait(3, done);
  EXPECT_EQ(output.front(), UINT64_C(0xA57C99E1A57C99E1));
  EXPECT_EQ(output.back(), UINT64_C(0xA57C99E1A57C99E1));
  for (size_t i = 1; i <= kCount; ++i) {
    EXPECT_GT(output[i], 0u);
    EXPECT_NE(output[i], UINT64_C(0xA57C99E1A57C99E1));
    if (i > 1) {
      EXPECT_GE(output[i], output[i - 1]);
    }
  }
}

TEST_P(GpuNativeTimestampTest, RejectsInvalidTargetBeforeQueueAcceptance) {
  Allocate(32);
  if (HasFatalFailure()) {
    return;
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_timestamp(queues_[0], {}, {}, buffer_, 32, nullptr, 0));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_timestamp(queues_[0], {}, {}, buffer_, 4, nullptr, 0));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_timestamp(queues_[0], {}, {}, buffer_, 0, nullptr, 1));
  uint64_t done = 1;
  IREE_ASSERT_OK(iree_hal_queue_timestamp(
      queues_[0], {}, {1, &semaphores_[2], &done}, buffer_, 24, nullptr, 0));
  Wait(2, done);
}

TEST_P(GpuNativeTimestampTest, FailedDependencyPreservesTarget) {
  Allocate(32);
  if (HasFatalFailure()) {
    return;
  }
  uint64_t one = 1;
  const uint32_t guard = 0xA57C99E1;
  IREE_ASSERT_OK(iree_hal_queue_fill(queues_[0], {}, {1, &semaphores_[2], &one},
                                     buffer_, 0, 32, &guard, sizeof(guard),
                                     nullptr, 0));
  Wait(2, one);
  IREE_ASSERT_OK(iree_hal_queue_timestamp(
      queues_[0], {1, &semaphores_[1], &one}, {1, &semaphores_[3], &one},
      buffer_, 8, nullptr, 0));
  iree_hal_semaphore_fail(semaphores_[1],
                          iree_status_from_code(IREE_STATUS_ABORTED));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_semaphore_wait(semaphores_[3], one, iree_infinite_timeout(), 0));
  std::array<uint64_t, 4> output = {};
  uint64_t two = 2;
  IREE_ASSERT_OK(iree_hal_queue_download(
      queues_[1], {}, {1, &semaphores_[2], &two}, buffer_, 0, output.data(),
      sizeof(output), nullptr));
  Wait(2, two);
  for (uint64_t value : output) {
    EXPECT_EQ(value, UINT64_C(0xA57C99E1A57C99E1));
  }
}

INSTANTIATE_TEST_SUITE_P(
    PayloadBoundaries, GpuNativeTimestampTest,
    ::testing::Values(std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, false),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, true),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, false),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, true)),
    [](const auto& info) {
      return std::string(info.param.first == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                             ? "Aql"
                             : "Pm4") +
             (info.param.second ? "Empty" : "Default");
    });

}  // namespace
