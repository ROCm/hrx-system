// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/transient_buffer_pool.h"

#include <atomic>
#include <barrier>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TransientBufferPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_amd_fabric_transient_buffer_pool_initialize({this, Control},
                                                         &pool_);
  }
  void TearDown() override {
    iree_hal_amd_fabric_transient_buffer_pool_deinitialize(&pool_);
    EXPECT_EQ(live_.load(), 0u);
  }

  static iree_status_t Control(void* user_data,
                               iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* self = static_cast<TransientBufferPoolTest*>(user_data);
    const bool had_pointer = *inout_pointer != nullptr;
    if (command != IREE_ALLOCATOR_COMMAND_FREE && self->fail_growth_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "chunk growth");
    }
    const auto system = iree_allocator_system();
    iree_status_t status =
        system.ctl(system.self, command, params, inout_pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE && had_pointer) {
        self->live_.fetch_sub(1);
      } else if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_pointer) {
        self->allocations_.fetch_add(1);
        self->live_.fetch_add(1);
      }
    }
    return status;
  }

  static void Release(
      const iree_hal_amd_fabric_transient_buffer_slots_t& slots) {
    for (size_t i = 0; i < slots.count; ++i) {
      iree_hal_amd_fabric_transient_buffer_pool_release(
          slots.chunk, slots.data + i * slots.stride);
    }
  }

  // Pool under test, shared by acquisition and retirement callers.
  iree_hal_amd_fabric_transient_buffer_pool_t pool_;
  // Successful whole-chunk allocations.
  std::atomic<size_t> allocations_{0};
  // Chunks not yet retired by trim or pool destruction.
  std::atomic<size_t> live_{0};
  // Failure injection, changed only while no thread uses the allocator.
  bool fail_growth_ = false;
};

TEST_F(TransientBufferPoolTest, CapacityAlignmentAndLargeTables) {
  for (size_t size :
       {1u, 64u, 65u, 512u, 513u, (2u << 20) - 1, 2u << 20, (2u << 20) + 1}) {
    SCOPED_TRACE(size);
    iree_hal_amd_fabric_transient_buffer_slots_t slots;
    IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(
        &pool_, size, 3, &slots));
    ASSERT_GT(slots.count, 0u);
    EXPECT_LE(slots.count, 3u);
    ASSERT_GE(slots.stride, size);
    EXPECT_EQ(slots.stride & (slots.stride - 1), 0u);
    EXPECT_EQ(iree_hal_amd_fabric_transient_buffer_slot_capacity(slots.chunk),
              slots.stride);
    EXPECT_EQ(iree_hal_amd_fabric_transient_buffer_slot_pool(slots.chunk),
              &pool_);
    for (size_t i = 0; i < slots.count; ++i) {
      auto* data = slots.data + i * slots.stride;
      EXPECT_EQ(reinterpret_cast<uintptr_t>(data) % 64, 0u);
      data[0] = 0x12;
      data[size - 1] = 0x34;
    }
    Release(slots);
    iree_hal_amd_fabric_transient_buffer_pool_trim(&pool_);
    EXPECT_EQ(live_.load(), 0u);
  }
}

TEST_F(TransientBufferPoolTest, WholeBitmapWordAndFragmentedReuse) {
  iree_hal_amd_fabric_transient_buffer_slots_t slots;
  IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(&pool_, 512,
                                                                   64, &slots));
  ASSERT_EQ(slots.count, 64u);
  for (size_t i = 0; i < slots.count; ++i) {
    memset(slots.data + i * slots.stride, static_cast<int>(i), slots.stride);
  }
  for (size_t i = 1; i < slots.count; i += 2) {
    iree_hal_amd_fabric_transient_buffer_pool_release(
        slots.chunk, slots.data + i * slots.stride);
  }
  const auto allocations = allocations_.load();
  iree_hal_amd_fabric_transient_buffer_pool_trim(&pool_);
  std::vector<iree_hal_amd_fabric_transient_buffer_slots_t> replacements;
  for (size_t i = 1; i < slots.count; i += 2) {
    iree_hal_amd_fabric_transient_buffer_slots_t replacement;
    IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(
        &pool_, 512, 1, &replacement));
    memset(replacement.data, static_cast<int>(i), replacement.stride);
    replacements.push_back(replacement);
  }
  EXPECT_EQ(allocations_.load(), allocations);
  for (size_t i = 0; i < slots.count; i += 2) {
    const auto* data = slots.data + i * slots.stride;
    EXPECT_EQ(data[0], i);
    EXPECT_EQ(data[slots.stride - 1], i);
    iree_hal_amd_fabric_transient_buffer_pool_release(
        slots.chunk, slots.data + i * slots.stride);
  }
  for (size_t i = 0; i < replacements.size(); ++i) {
    const auto& replacement = replacements[i];
    EXPECT_EQ(replacement.data[0], i * 2 + 1);
    EXPECT_EQ(replacement.data[replacement.stride - 1], i * 2 + 1);
    Release(replacement);
  }
  iree_hal_amd_fabric_transient_buffer_pool_trim(&pool_);
  EXPECT_EQ(live_.load(), 0u);
}

TEST_F(TransientBufferPoolTest, FullChunkReturnsAndRetiresIndependently) {
  // A large native table exercises a complete chunk with fewer than 64 slots.
  iree_hal_amd_fabric_transient_buffer_slots_t first, second;
  IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(
      &pool_, 32768, 64, &first));
  IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(
      &pool_, 32768, 1, &second));
  ASSERT_NE(first.chunk, second.chunk);
  EXPECT_EQ(live_.load(), 2u);
  iree_hal_amd_fabric_transient_buffer_pool_release(first.chunk, first.data);
  iree_hal_amd_fabric_transient_buffer_pool_trim(&pool_);
  EXPECT_EQ(live_.load(), 2u);
  iree_hal_amd_fabric_transient_buffer_slots_t replacement;
  IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(
      &pool_, 32768, 1, &replacement));
  EXPECT_EQ(replacement.data, first.data);
  EXPECT_EQ(allocations_.load(), 2u);
  Release(first);
  iree_hal_amd_fabric_transient_buffer_pool_trim(&pool_);
  EXPECT_EQ(live_.load(), 1u);
  memset(second.data, 0xA5, second.stride);
  Release(second);
}

TEST_F(TransientBufferPoolTest, FailedGrowthPreservesExistingSlots) {
  iree_hal_amd_fabric_transient_buffer_slots_t existing, failed;
  IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(
      &pool_, 512, 1, &existing));
  memset(existing.data, 0xA5, existing.stride);
  fail_growth_ = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amd_fabric_transient_buffer_pool_acquire(
                            &pool_, 1024, 1, &failed));
  EXPECT_EQ(failed.chunk, nullptr);
  EXPECT_EQ(failed.count, 0u);
  // Warm reuse still works when the backing allocator cannot grow.
  iree_hal_amd_fabric_transient_buffer_slots_t warm;
  IREE_ASSERT_OK(
      iree_hal_amd_fabric_transient_buffer_pool_acquire(&pool_, 512, 1, &warm));
  EXPECT_NE(warm.data, existing.data);
  EXPECT_EQ(existing.data[0], 0xA5);
  EXPECT_EQ(existing.data[existing.stride - 1], 0xA5);
  Release(warm);
  Release(existing);
  fail_growth_ = false;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amd_fabric_transient_buffer_pool_acquire(
                            &pool_, SIZE_MAX, 1, &failed));
  EXPECT_EQ(allocations_.load(), 1u);
}

TEST_F(TransientBufferPoolTest, ConcurrentReuseReleaseAndTrim) {
  // Cover partial/full bitmap words and a one-slot chunk. The latter forces
  // every release through the full-to-available publication boundary.
  for (size_t size : {512u, 32768u, 1u << 20}) {
    SCOPED_TRACE(size);
    iree_hal_amd_fabric_transient_buffer_slots_t initial;
    IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(
        &pool_, size, 64, &initial));
    std::barrier<> start(4);
    std::vector<std::thread> threads;
    for (size_t worker = 0; worker < 3; ++worker) {
      threads.emplace_back([&, worker] {
        start.arrive_and_wait();
        for (size_t i = worker; i < initial.count; i += 3) {
          iree_hal_amd_fabric_transient_buffer_pool_release(
              initial.chunk, initial.data + i * initial.stride);
        }
        for (size_t i = 0; i < 16; ++i) {
          iree_hal_amd_fabric_transient_buffer_slots_t slots;
          IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_pool_acquire(
              &pool_, size, 3, &slots));
          for (size_t j = 0; j < slots.count; ++j) {
            auto* data = slots.data + j * slots.stride;
            data[0] = static_cast<uint8_t>(worker);
            data[slots.stride - 1] = static_cast<uint8_t>(worker);
            EXPECT_EQ(data[0], worker);
          }
          Release(slots);
        }
      });
    }
    start.arrive_and_wait();
    for (size_t i = 0; i < 16; ++i) {
      iree_hal_amd_fabric_transient_buffer_pool_trim(&pool_);
    }
    for (auto& thread : threads) {
      thread.join();
    }
    iree_hal_amd_fabric_transient_buffer_pool_trim(&pool_);
    EXPECT_EQ(live_.load(), 0u);
  }
}

}  // namespace
