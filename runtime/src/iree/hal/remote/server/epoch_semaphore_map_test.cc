// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/epoch_semaphore_map.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct TestSemaphore {
  iree_hal_resource_t resource;
  int* destroy_count;
};

static void DestroyTestSemaphore(iree_async_semaphore_t* base_semaphore) {
  auto* semaphore = reinterpret_cast<TestSemaphore*>(base_semaphore);
  ++*semaphore->destroy_count;
}

static const iree_hal_semaphore_vtable_t kTestSemaphoreVTable = {
    /*.async=*/
    {
        /*.destroy=*/DestroyTestSemaphore,
        /*.query=*/nullptr,
        /*.signal=*/nullptr,
        /*.on_fail=*/nullptr,
    },
    /*.wait=*/nullptr,
    /*.import_timepoint=*/nullptr,
    /*.export_timepoint=*/nullptr,
};

static iree_hal_semaphore_t* InitializeTestSemaphore(TestSemaphore* semaphore,
                                                     int* destroy_count) {
  iree_hal_resource_initialize(&kTestSemaphoreVTable, &semaphore->resource);
  semaphore->destroy_count = destroy_count;
  return reinterpret_cast<iree_hal_semaphore_t*>(semaphore);
}

struct FailingAllocator {
  bool fail_allocations = false;
  iree_host_size_t allocation_count = 0;
  iree_host_size_t free_count = 0;

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailingAllocator*>(self);
    switch (command) {
      case IREE_ALLOCATOR_COMMAND_MALLOC:
      case IREE_ALLOCATOR_COMMAND_CALLOC:
        if (allocator->fail_allocations) {
          return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
        }
        ++allocator->allocation_count;
        break;
      case IREE_ALLOCATOR_COMMAND_FREE:
        ++allocator->free_count;
        break;
      default:
        break;
    }
    iree_allocator_t system_allocator = iree_allocator_system();
    return system_allocator.ctl(system_allocator.self, command, params,
                                inout_ptr);
  }
};

class EpochSemaphoreMapTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_hal_remote_server_epoch_semaphore_map_deinitialize(
        &map_, iree_allocator_system());
  }

  iree_hal_remote_server_epoch_semaphore_map_t map_ = {};
};

TEST_F(EpochSemaphoreMapTest, InsertRetainsAndLookupBorrows) {
  int destroy_count = 0;
  TestSemaphore test_semaphore;
  iree_hal_semaphore_t* semaphore =
      InitializeTestSemaphore(&test_semaphore, &destroy_count);

  IREE_ASSERT_OK(iree_hal_remote_server_epoch_semaphore_map_insert(
      &map_, /*axis=*/1, /*epoch=*/2, semaphore, iree_allocator_system()));
  iree_hal_semaphore_release(semaphore);
  EXPECT_EQ(destroy_count, 0);
  EXPECT_EQ(iree_hal_remote_server_epoch_semaphore_map_lookup(&map_, /*axis=*/1,
                                                              /*epoch=*/2),
            semaphore);
  EXPECT_EQ(iree_hal_remote_server_epoch_semaphore_map_lookup(&map_, /*axis=*/2,
                                                              /*epoch=*/2),
            nullptr);
  EXPECT_EQ(iree_hal_remote_server_epoch_semaphore_map_lookup(&map_, /*axis=*/1,
                                                              /*epoch=*/3),
            nullptr);

  iree_hal_remote_server_epoch_semaphore_map_deinitialize(
      &map_, iree_allocator_system());
  EXPECT_EQ(destroy_count, 1);
}

TEST_F(EpochSemaphoreMapTest, DuplicateDoesNotReplaceOrRetain) {
  int first_destroy_count = 0;
  TestSemaphore first_test_semaphore;
  iree_hal_semaphore_t* first_semaphore =
      InitializeTestSemaphore(&first_test_semaphore, &first_destroy_count);
  IREE_ASSERT_OK(iree_hal_remote_server_epoch_semaphore_map_insert(
      &map_, /*axis=*/1, /*epoch=*/2, first_semaphore,
      iree_allocator_system()));
  iree_hal_semaphore_release(first_semaphore);

  int duplicate_destroy_count = 0;
  TestSemaphore duplicate_test_semaphore;
  iree_hal_semaphore_t* duplicate_semaphore = InitializeTestSemaphore(
      &duplicate_test_semaphore, &duplicate_destroy_count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        iree_hal_remote_server_epoch_semaphore_map_insert(
                            &map_, /*axis=*/1, /*epoch=*/2, duplicate_semaphore,
                            iree_allocator_system()));
  EXPECT_EQ(iree_hal_remote_server_epoch_semaphore_map_lookup(&map_, /*axis=*/1,
                                                              /*epoch=*/2),
            first_semaphore);
  iree_hal_semaphore_release(duplicate_semaphore);
  EXPECT_EQ(duplicate_destroy_count, 1);

  iree_hal_remote_server_epoch_semaphore_map_deinitialize(
      &map_, iree_allocator_system());
  EXPECT_EQ(first_destroy_count, 1);
}

TEST_F(EpochSemaphoreMapTest, RemoveTransfersRetainedReference) {
  int destroy_count = 0;
  TestSemaphore test_semaphore;
  iree_hal_semaphore_t* semaphore =
      InitializeTestSemaphore(&test_semaphore, &destroy_count);
  IREE_ASSERT_OK(iree_hal_remote_server_epoch_semaphore_map_insert(
      &map_, /*axis=*/1, /*epoch=*/2, semaphore, iree_allocator_system()));
  iree_hal_semaphore_release(semaphore);

  iree_hal_semaphore_t* removed =
      iree_hal_remote_server_epoch_semaphore_map_remove(&map_, /*axis=*/1,
                                                        /*epoch=*/2);
  EXPECT_EQ(removed, semaphore);
  EXPECT_EQ(iree_hal_remote_server_epoch_semaphore_map_lookup(&map_, /*axis=*/1,
                                                              /*epoch=*/2),
            nullptr);
  EXPECT_EQ(iree_hal_remote_server_epoch_semaphore_map_remove(&map_, /*axis=*/1,
                                                              /*epoch=*/2),
            nullptr);
  EXPECT_EQ(destroy_count, 0);
  iree_hal_semaphore_release(removed);
  EXPECT_EQ(destroy_count, 1);
}

TEST_F(EpochSemaphoreMapTest, TombstonesAndGrowthPreserveMappings) {
  constexpr iree_host_size_t kSemaphoreCount = 160;
  std::array<TestSemaphore, kSemaphoreCount> test_semaphores;
  std::array<int, kSemaphoreCount> destroy_counts = {};

  for (iree_host_size_t i = 0; i < kSemaphoreCount; ++i) {
    iree_hal_semaphore_t* semaphore =
        InitializeTestSemaphore(&test_semaphores[i], &destroy_counts[i]);
    IREE_ASSERT_OK(iree_hal_remote_server_epoch_semaphore_map_insert(
        &map_, /*axis=*/i % 7, /*epoch=*/i + 1, semaphore,
        iree_allocator_system()));
    iree_hal_semaphore_release(semaphore);
  }
  for (iree_host_size_t i = 0; i < kSemaphoreCount; i += 2) {
    iree_hal_semaphore_t* removed =
        iree_hal_remote_server_epoch_semaphore_map_remove(&map_, /*axis=*/i % 7,
                                                          /*epoch=*/i + 1);
    ASSERT_NE(removed, nullptr);
    iree_hal_semaphore_release(removed);
  }
  for (iree_host_size_t i = 0; i < kSemaphoreCount; ++i) {
    EXPECT_EQ(iree_hal_remote_server_epoch_semaphore_map_lookup(
                  &map_, /*axis=*/i % 7, /*epoch=*/i + 1),
              i % 2 == 0 ? nullptr
                         : reinterpret_cast<iree_hal_semaphore_t*>(
                               &test_semaphores[i]));
  }

  iree_hal_remote_server_epoch_semaphore_map_deinitialize(
      &map_, iree_allocator_system());
  for (int destroy_count : destroy_counts) {
    EXPECT_EQ(destroy_count, 1);
  }
}

TEST_F(EpochSemaphoreMapTest, MoveTransfersAllOwnership) {
  int destroy_count = 0;
  TestSemaphore test_semaphore;
  iree_hal_semaphore_t* semaphore =
      InitializeTestSemaphore(&test_semaphore, &destroy_count);
  IREE_ASSERT_OK(iree_hal_remote_server_epoch_semaphore_map_insert(
      &map_, /*axis=*/1, /*epoch=*/2, semaphore, iree_allocator_system()));
  iree_hal_semaphore_release(semaphore);

  iree_hal_remote_server_epoch_semaphore_map_t moved_map = {};
  iree_hal_remote_server_epoch_semaphore_map_move(&map_, &moved_map);
  EXPECT_EQ(map_.slots, nullptr);
  EXPECT_EQ(map_.count, 0u);
  EXPECT_EQ(map_.used_count, 0u);
  EXPECT_EQ(map_.capacity, 0u);
  EXPECT_EQ(iree_hal_remote_server_epoch_semaphore_map_lookup(
                &moved_map, /*axis=*/1, /*epoch=*/2),
            semaphore);

  iree_hal_remote_server_epoch_semaphore_map_deinitialize(
      &moved_map, iree_allocator_system());
  EXPECT_EQ(destroy_count, 1);
}

TEST(EpochSemaphoreMapAllocationTest, FailedGrowthPreservesExistingMappings) {
  constexpr iree_host_size_t kExistingCount = 47;
  std::array<TestSemaphore, kExistingCount + 1> test_semaphores;
  std::array<int, kExistingCount + 1> destroy_counts = {};
  FailingAllocator allocator;
  iree_hal_remote_server_epoch_semaphore_map_t map = {};

  for (iree_host_size_t i = 0; i < kExistingCount; ++i) {
    iree_hal_semaphore_t* semaphore =
        InitializeTestSemaphore(&test_semaphores[i], &destroy_counts[i]);
    IREE_ASSERT_OK(iree_hal_remote_server_epoch_semaphore_map_insert(
        &map, /*axis=*/i % 5, /*epoch=*/i + 1, semaphore,
        allocator.allocator()));
    iree_hal_semaphore_release(semaphore);
  }
  ASSERT_EQ(allocator.allocation_count, 1u);

  iree_hal_semaphore_t* rejected_semaphore = InitializeTestSemaphore(
      &test_semaphores[kExistingCount], &destroy_counts[kExistingCount]);
  allocator.fail_allocations = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_remote_server_epoch_semaphore_map_insert(
                            &map, /*axis=*/9, /*epoch=*/999, rejected_semaphore,
                            allocator.allocator()));
  EXPECT_EQ(map.count, kExistingCount);
  EXPECT_EQ(iree_hal_remote_server_epoch_semaphore_map_lookup(&map, /*axis=*/9,
                                                              /*epoch=*/999),
            nullptr);
  for (iree_host_size_t i = 0; i < kExistingCount; ++i) {
    EXPECT_EQ(iree_hal_remote_server_epoch_semaphore_map_lookup(
                  &map, /*axis=*/i % 5, /*epoch=*/i + 1),
              reinterpret_cast<iree_hal_semaphore_t*>(&test_semaphores[i]));
  }
  iree_hal_semaphore_release(rejected_semaphore);
  EXPECT_EQ(destroy_counts[kExistingCount], 1);

  allocator.fail_allocations = false;
  iree_hal_remote_server_epoch_semaphore_map_deinitialize(
      &map, allocator.allocator());
  EXPECT_EQ(allocator.free_count, 1u);
  for (int destroy_count : destroy_counts) {
    EXPECT_EQ(destroy_count, 1);
  }
}

}  // namespace
