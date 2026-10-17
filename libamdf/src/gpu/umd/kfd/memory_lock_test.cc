// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/memory_lock.h"

#include <array>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace {

class MemoryLockTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(amdf_gpu_kfd_memory_lock_initialize(&lock_), 0);
  }
  void TearDown() override {
    EXPECT_FALSE(lock_.occupied);
    EXPECT_EQ(lock_.pending_mask, 0u);
    amdf_gpu_kfd_memory_lock_deinitialize(&lock_);
  }

  std::thread Enqueue(amdf_gpu_kfd_memory_priority_t priority, uint32_t value) {
    EXPECT_EQ(pthread_mutex_lock(&lock_.mutex), 0);
    const auto* previous = lock_.buckets[priority].tail;
    EXPECT_EQ(pthread_mutex_unlock(&lock_.mutex), 0);
    std::thread caller([this, priority, value] {
      amdf_gpu_kfd_memory_lock_acquire(&lock_, priority);
      order_.push_back(value);
      amdf_gpu_kfd_memory_lock_release(&lock_);
    });
    // The test owns the active native operation. Observe actual enrollment
    // before admitting the next caller; thread startup is not readiness.
    while (true) {
      EXPECT_EQ(pthread_mutex_lock(&lock_.mutex), 0);
      const bool enrolled = lock_.buckets[priority].tail != previous;
      EXPECT_EQ(pthread_mutex_unlock(&lock_.mutex), 0);
      if (enrolled) {
        break;
      }
      std::this_thread::yield();
    }
    return caller;
  }

  // Production native admission state shared by all modeled NUMA callers.
  amdf_gpu_kfd_memory_lock_t lock_;
  // Native critical-section order, protected by the production lock.
  std::vector<uint32_t> order_;
};

TEST_F(MemoryLockTest, UncontendedClassesLeaveNoWaiterStorage) {
  for (auto priority :
       {AMDF_GPU_KFD_MEMORY_PRIORITY_MAP, AMDF_GPU_KFD_MEMORY_PRIORITY_RELEASE,
        AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE}) {
    amdf_gpu_kfd_memory_lock_acquire(&lock_, priority);
    EXPECT_TRUE(lock_.occupied);
    EXPECT_EQ(lock_.pending_mask, 0u);
    amdf_gpu_kfd_memory_lock_release(&lock_);
  }
}

TEST_F(MemoryLockTest, MappingThenReleaseThenAcquisitionPreservesEachFifo) {
  amdf_gpu_kfd_memory_lock_acquire(&lock_,
                                   AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE);
  std::array<std::thread, 6> callers = {
      Enqueue(AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE, 4),
      Enqueue(AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE, 5),
      Enqueue(AMDF_GPU_KFD_MEMORY_PRIORITY_RELEASE, 2),
      Enqueue(AMDF_GPU_KFD_MEMORY_PRIORITY_RELEASE, 3),
      Enqueue(AMDF_GPU_KFD_MEMORY_PRIORITY_MAP, 0),
      Enqueue(AMDF_GPU_KFD_MEMORY_PRIORITY_MAP, 1),
  };
  amdf_gpu_kfd_memory_lock_release(&lock_);
  for (auto& caller : callers) {
    caller.join();
  }
  EXPECT_EQ(order_, (std::vector<uint32_t>{0, 1, 2, 3, 4, 5}));
}

TEST_F(MemoryLockTest, ArrivalCannotStealAnAlreadyGrantedTurn) {
  for (uint32_t i = 0; i < 128; ++i) {
    amdf_gpu_kfd_memory_lock_acquire(&lock_, AMDF_GPU_KFD_MEMORY_PRIORITY_MAP);
    auto caller = Enqueue(AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE, i * 2);
    amdf_gpu_kfd_memory_lock_release(&lock_);
    amdf_gpu_kfd_memory_lock_acquire(&lock_, AMDF_GPU_KFD_MEMORY_PRIORITY_MAP);
    order_.push_back(i * 2 + 1);
    amdf_gpu_kfd_memory_lock_release(&lock_);
    caller.join();
  }
  ASSERT_EQ(order_.size(), 256u);
  for (size_t i = 0; i < order_.size(); ++i) {
    EXPECT_EQ(order_[i], i);
  }
}

TEST_F(MemoryLockTest, ConcurrentCallersKeepNativeMutationsExclusive) {
  uint32_t value = 0;
  std::array<std::thread, 12> callers;
  for (size_t i = 0; i < callers.size(); ++i) {
    callers[i] = std::thread([&, i] {
      const auto priority = static_cast<amdf_gpu_kfd_memory_priority_t>(i % 3);
      for (uint32_t j = 0; j < 1000; ++j) {
        amdf_gpu_kfd_memory_lock_acquire(&lock_, priority);
        ++value;
        amdf_gpu_kfd_memory_lock_release(&lock_);
      }
    });
  }
  for (auto& caller : callers) {
    caller.join();
  }
  EXPECT_EQ(value, 12000u);
}

}  // namespace
