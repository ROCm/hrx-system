// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"

#include <array>
#include <atomic>
#include <barrier>
#include <cstring>
#include <semaphore>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct Allocator {
  // Cumulative successful system allocations, excluding reuse.
  std::atomic<size_t> allocations = 0;
  // Allocations not yet returned to the system.
  std::atomic<size_t> live = 0;
  // Inject failure after this many successful allocations.
  size_t allocation_limit = SIZE_MAX;

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* self_pointer,
                               iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* self = static_cast<Allocator*>(self_pointer);
    const bool had_pointer = *inout_pointer != nullptr;
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        self->allocations.load() >= self->allocation_limit) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected host allocation failure");
    }
    auto system = iree_allocator_system();
    iree_status_t status =
        system.ctl(system.self, command, params, inout_pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE && had_pointer) {
        self->live.fetch_sub(1);
      } else if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_pointer) {
        self->live.fetch_add(1);
        self->allocations.fetch_add(1);
      }
    }
    return status;
  }
};

class CapturePoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_amd_gpu_capture_pool_initialize(1024, allocator_.allocator(),
                                             &pool_);
  }
  void TearDown() override {
    iree_hal_amd_gpu_capture_pool_deinitialize(&pool_);
    EXPECT_EQ(allocator_.live.load(), 0u);
  }
  // Dependency allocator observes actual allocation/reclamation.
  Allocator allocator_;
  // Production embedded owner; tests return every acquired allocation.
  iree_hal_amd_gpu_capture_pool_t pool_;
};

TEST_F(CapturePoolTest, ReusesCompleteBlocksWithoutSystemAllocation) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_preallocate(&pool_, 8));
  for (size_t round = 0; round < 4; ++round) {
    std::array<iree_byte_span_t, 8> blocks;
    for (auto& block : blocks) {
      IREE_ASSERT_OK(
          iree_hal_amd_gpu_capture_pool_acquire(&pool_, 208, &block));
      EXPECT_EQ(block.data_length, 1024u);
      EXPECT_EQ(reinterpret_cast<uintptr_t>(block.data) % iree_max_align_t, 0u);
      std::memset(block.data, 0xA5, block.data_length);
    }
    for (auto block : blocks) {
      iree_hal_amd_gpu_capture_pool_release(&pool_, block);
    }
  }
  EXPECT_EQ(allocator_.allocations.load(), 8u);
  EXPECT_EQ(allocator_.live.load(), 8u);
}

TEST_F(CapturePoolTest, TrimPreservesActiveBlocksAndAcceptsLaterReturns) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_preallocate(&pool_, 4));
  iree_byte_span_t first, second;
  IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_acquire(&pool_, 1024, &first));
  IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_acquire(&pool_, 1024, &second));
  std::memset(first.data, 0x35, first.data_length);
  iree_hal_amd_gpu_capture_pool_release(&pool_, second);
  // Both the caller-available list and the returned list contain free blocks.
  iree_hal_amd_gpu_capture_pool_trim(&pool_);
  EXPECT_EQ(allocator_.live.load(), 1u);
  std::array<uint8_t, 1024> expected;
  expected.fill(0x35);
  EXPECT_EQ(std::memcmp(first.data, expected.data(), expected.size()), 0);
  iree_hal_amd_gpu_capture_pool_release(&pool_, first);
  iree_byte_span_t reused;
  IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_acquire(&pool_, 208, &reused));
  EXPECT_EQ(reused.data, first.data);
  EXPECT_EQ(allocator_.allocations.load(), 4u);
  iree_hal_amd_gpu_capture_pool_release(&pool_, reused);
  iree_hal_amd_gpu_capture_pool_trim(&pool_);
  EXPECT_EQ(allocator_.live.load(), 0u);
}

TEST_F(CapturePoolTest, OversizedStorageReturnsDirectlyToAllocator) {
  iree_byte_span_t block;
  IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_acquire(&pool_, 1025, &block));
  EXPECT_EQ(block.data_length, 1025u);
  std::memset(block.data, 0xA5, block.data_length);
  iree_hal_amd_gpu_capture_pool_trim(&pool_);
  EXPECT_EQ(allocator_.live.load(), 1u);
  iree_hal_amd_gpu_capture_pool_release(&pool_, block);
  EXPECT_EQ(allocator_.live.load(), 0u);
}

TEST_F(CapturePoolTest, PartialPreallocationFailurePreservesExistingStorage) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_preallocate(&pool_, 1));
  allocator_.allocation_limit = 3;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amd_gpu_capture_pool_preallocate(&pool_, 3));
  EXPECT_EQ(allocator_.live.load(), 1u);
  iree_byte_span_t block;
  IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_acquire(&pool_, 208, &block));
  iree_hal_amd_gpu_capture_pool_release(&pool_, block);
  IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_preallocate(&pool_, 0));
}

TEST_F(CapturePoolTest, AllocationFailureLeavesNoStorage) {
  allocator_.allocation_limit = 0;
  for (iree_host_size_t size : {208u, 1025u}) {
    iree_byte_span_t block = iree_make_byte_span(&size, sizeof(size));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_hal_amd_gpu_capture_pool_acquire(&pool_, size, &block));
    EXPECT_EQ(block.data, nullptr);
    EXPECT_EQ(block.data_length, 0u);
  }
  EXPECT_EQ(allocator_.live.load(), 0u);
}

TEST_F(CapturePoolTest, ConcurrentCallersRetirementAndTrim) {
  constexpr size_t kCallerCount = 4;
  constexpr size_t kBatchCount = 8;
  constexpr size_t kRoundCount = 128;
  struct Caller {
    // Captured allocations transfer to the retirement thread after publication.
    std::array<iree_byte_span_t, kBatchCount> blocks = {};
    // Exact publication of this caller's batch; notifications are never polled.
    std::binary_semaphore published{0};
  };
  std::array<Caller, kCallerCount> callers;
  std::barrier<> epoch(kCallerCount + 2);
  IREE_ASSERT_OK(iree_hal_amd_gpu_capture_pool_preallocate(
      &pool_, kCallerCount * kBatchCount));
  std::array<std::thread, kCallerCount> threads;
  for (size_t i = 0; i < kCallerCount; ++i) {
    threads[i] = std::thread([&, i] {
      for (size_t round = 0; round < kRoundCount; ++round) {
        epoch.arrive_and_wait();
        for (auto& block : callers[i].blocks) {
          IREE_EXPECT_OK(
              iree_hal_amd_gpu_capture_pool_acquire(&pool_, 208, &block));
          if (block.data) {
            std::memset(block.data, 1 + i + round, block.data_length);
          }
        }
        callers[i].published.release();
        epoch.arrive_and_wait();
      }
    });
  }
  std::thread retirement([&] {
    for (size_t round = 0; round < kRoundCount; ++round) {
      epoch.arrive_and_wait();
      for (size_t i = 0; i < kCallerCount; ++i) {
        callers[i].published.acquire();
        std::array<uint8_t, 1024> expected;
        expected.fill(1 + i + round);
        for (auto block : callers[i].blocks) {
          if (block.data) {
            EXPECT_EQ(std::memcmp(block.data, expected.data(), expected.size()),
                      0);
            iree_hal_amd_gpu_capture_pool_release(&pool_, block);
          }
        }
      }
      epoch.arrive_and_wait();
    }
  });
  for (size_t round = 0; round < kRoundCount; ++round) {
    epoch.arrive_and_wait();
    iree_hal_amd_gpu_capture_pool_trim(&pool_);
    epoch.arrive_and_wait();
  }
  for (auto& thread : threads) {
    thread.join();
  }
  retirement.join();
  iree_hal_amd_gpu_capture_pool_trim(&pool_);
  EXPECT_EQ(allocator_.live.load(), 0u);
}

}  // namespace
