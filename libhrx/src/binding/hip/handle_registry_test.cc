// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/handle_registry.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class HandleRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override { iree_hip_handle_registry_initialize(&registry_); }

  void TearDown() override {
    iree_hip_handle_registry_deinitialize(&registry_);
  }

  iree_hip_handle_registry_t registry_;
};

uintptr_t retained_handle = 0;

void RetainHandle(uintptr_t handle) { retained_handle = handle; }

std::atomic<uint64_t> concurrent_retain_count{0};

void CountRetain(uintptr_t handle) {
  if (handle != 0) {
    concurrent_retain_count.fetch_add(1, std::memory_order_relaxed);
  }
}

TEST_F(HandleRegistryTest, LookupRetainsOnlyLiveHandles) {
  constexpr uintptr_t handle = 0x1234;
  IREE_ASSERT_OK(iree_hip_handle_registry_insert(&registry_, handle));

  retained_handle = 0;
  EXPECT_TRUE(
      iree_hip_handle_registry_lookup_retain(&registry_, handle, RetainHandle));
  EXPECT_EQ(retained_handle, handle);

  EXPECT_TRUE(iree_hip_handle_registry_remove(&registry_, handle));
  retained_handle = 0;
  EXPECT_FALSE(
      iree_hip_handle_registry_lookup_retain(&registry_, handle, RetainHandle));
  EXPECT_EQ(retained_handle, 0u);
}

TEST_F(HandleRegistryTest, RejectsDuplicateAndRepeatedRemoval) {
  constexpr uintptr_t handle = 0x5678;
  IREE_ASSERT_OK(iree_hip_handle_registry_insert(&registry_, handle));
  iree_status_t duplicate_status =
      iree_hip_handle_registry_insert(&registry_, handle);
  EXPECT_EQ(iree_status_code(duplicate_status), IREE_STATUS_ALREADY_EXISTS);
  iree_status_free(duplicate_status);
  EXPECT_TRUE(iree_hip_handle_registry_remove(&registry_, handle));
  EXPECT_FALSE(iree_hip_handle_registry_remove(&registry_, handle));
}

TEST_F(HandleRegistryTest, RetainsEntriesAcrossGrowthAndTombstoneReuse) {
  constexpr uintptr_t handle_count = 128;
  for (uintptr_t handle = 1; handle <= handle_count; ++handle) {
    IREE_ASSERT_OK(iree_hip_handle_registry_insert(&registry_, handle));
  }
  for (uintptr_t handle = 2; handle <= handle_count; handle += 2) {
    EXPECT_TRUE(iree_hip_handle_registry_remove(&registry_, handle));
  }
  for (uintptr_t handle = handle_count + 1; handle <= handle_count * 2;
       ++handle) {
    IREE_ASSERT_OK(iree_hip_handle_registry_insert(&registry_, handle));
  }

  for (uintptr_t handle = 1; handle <= handle_count * 2; ++handle) {
    const bool expected_live = handle > handle_count || (handle & 1) != 0;
    retained_handle = 0;
    EXPECT_EQ(iree_hip_handle_registry_lookup_retain(&registry_, handle,
                                                     RetainHandle),
              expected_live);
    EXPECT_EQ(retained_handle, expected_live ? handle : 0u);
    if (expected_live) {
      EXPECT_TRUE(iree_hip_handle_registry_remove(&registry_, handle));
    }
  }
}

TEST_F(HandleRegistryTest, ConcurrentLookupsAndRemovalRejectStaleHandles) {
  constexpr uintptr_t handle_count = 256;
  constexpr int thread_count = 8;
  for (uintptr_t handle = 1; handle <= handle_count; ++handle) {
    IREE_ASSERT_OK(iree_hip_handle_registry_insert(&registry_, handle));
  }

  concurrent_retain_count.store(0, std::memory_order_relaxed);
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (int thread = 0; thread < thread_count; ++thread) {
    threads.emplace_back([&, thread] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (uintptr_t handle = thread + 1; handle <= handle_count;
           handle += thread_count) {
        iree_hip_handle_registry_lookup_retain(&registry_, handle, CountRetain);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (uintptr_t handle = 1; handle <= handle_count; ++handle) {
    EXPECT_TRUE(iree_hip_handle_registry_remove(&registry_, handle));
  }
  for (std::thread& thread : threads) thread.join();

  for (uintptr_t handle = 1; handle <= handle_count; ++handle) {
    EXPECT_FALSE(iree_hip_handle_registry_lookup_retain(&registry_, handle,
                                                        CountRetain));
  }
  EXPECT_LE(concurrent_retain_count.load(std::memory_order_relaxed),
            handle_count);
}

}  // namespace
