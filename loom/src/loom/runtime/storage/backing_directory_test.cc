// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/backing_directory.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(BackingDirectoryTest, ReusesTombstonesAtTheLoadLimit) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  for (iree_host_size_t index = 0; index < 6; ++index) {
    loom_runtime_backing_group_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_runtime_backing_directory_insert(&directory, index, &entry));
  }
  ASSERT_EQ(directory.capacity, 8u);
  for (iree_host_size_t index = 0; index < 5; ++index) {
    loom_runtime_backing_directory_remove(
        &directory, loom_runtime_backing_directory_find(&directory, index));
  }
  ASSERT_EQ(directory.tombstone_count, 5u);

  loom_runtime_backing_group_t* replacement = nullptr;
  IREE_ASSERT_OK(
      loom_runtime_backing_directory_insert(&directory, 100, &replacement));
  EXPECT_EQ(directory.capacity, 8u);
  EXPECT_EQ(directory.tombstone_count, 4u);
  loom_runtime_backing_directory_deinitialize(&directory);
}

TEST(BackingDirectoryTest, TombstoneOnAnotherChainPreservesLookupTermination) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  loom_runtime_backing_group_t* entry = nullptr;
  IREE_ASSERT_OK(loom_runtime_backing_directory_insert(&directory, 0, &entry));
  loom_runtime_backing_directory_remove(&directory, entry);
  IREE_ASSERT_OK(loom_runtime_backing_directory_insert(&directory, 1, &entry));
  // An absent probe must still encounter a genuinely vacant bucket.
  ASSERT_LT(directory.count + directory.tombstone_count, directory.capacity);
  EXPECT_EQ(loom_runtime_backing_directory_find(&directory, 2), nullptr);
  loom_runtime_backing_directory_deinitialize(&directory);
}

TEST(BackingDirectoryTest, RehashesAndReusesDeletedProbeSlots) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);

  // These keys share their initial bucket at the first table capacity.
  for (const iree_host_size_t index : {0u, 8u, 16u}) {
    loom_runtime_backing_group_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_runtime_backing_directory_insert(&directory, index, &entry));
  }
  EXPECT_EQ(directory.count, 3u);
  loom_runtime_backing_directory_remove(
      &directory, loom_runtime_backing_directory_find(&directory, 8));
  EXPECT_EQ(directory.count, 2u);
  EXPECT_NE(loom_runtime_backing_directory_find(&directory, 16), nullptr);

  loom_runtime_backing_group_t* replacement = nullptr;
  IREE_ASSERT_OK(
      loom_runtime_backing_directory_insert(&directory, 24, &replacement));
  EXPECT_EQ(directory.count, 3u);
  EXPECT_EQ(directory.tombstone_count, 0u);

  for (iree_host_size_t i = 1; i <= 128; ++i) {
    loom_runtime_backing_group_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_runtime_backing_directory_insert(&directory, i * 7, &entry));
  }
  loom_runtime_backing_group_t* high = nullptr;
  IREE_ASSERT_OK(loom_runtime_backing_directory_insert(
      &directory, IREE_HOST_SIZE_MAX - 123, &high));
  high->in_flight = true;
  for (iree_host_size_t i = 1; i <= 128; ++i) {
    EXPECT_NE(loom_runtime_backing_directory_find_const(&directory, i * 7),
              nullptr);
  }
  EXPECT_EQ(loom_runtime_backing_directory_find_const(&directory,
                                                      IREE_HOST_SIZE_MAX - 123),
            high);

  loom_runtime_backing_directory_discard_empty(&directory);
  EXPECT_EQ(directory.count, 1u);
  EXPECT_GT(directory.capacity, 8u);
  IREE_ASSERT_OK(loom_runtime_backing_directory_shrink(&directory));
  EXPECT_EQ(directory.capacity, 2u);
  const auto* shrunk_high = loom_runtime_backing_directory_find_const(
      &directory, IREE_HOST_SIZE_MAX - 123);
  ASSERT_NE(shrunk_high, nullptr);
  EXPECT_TRUE(shrunk_high->in_flight);
  loom_runtime_backing_directory_deinitialize(&directory);
}

TEST(BackingDirectoryTest, LinearLeaseSurvivesFragmentationAndReturnsToInline) {
  // Only parent identity participates in directory indexing.
  int identities[2] = {};
  auto* first =
      reinterpret_cast<loom_runtime_backing_parent_t*>(&identities[0]);
  auto* second =
      reinterpret_cast<loom_runtime_backing_parent_t*>(&identities[1]);
  const auto allocator = iree_allocator_system();
  constexpr uint64_t stride = 4096;
  loom_runtime_backing_group_t group = {};
  const uint64_t last = UINT64_C(1) << 63;
  // The first physical page falls near the end of the logical word. Both
  // disjoint selections use the same source displacement without allocating.
  IREE_ASSERT_OK(loom_runtime_backing_group_assign(
      &group, iree_allocator_null(), UINT64_C(3) << 59, stride, {first, 0}));
  IREE_ASSERT_OK(loom_runtime_backing_group_assign(
      &group, iree_allocator_null(), last, stride, {first, 4 * stride}));
  EXPECT_EQ(loom_runtime_backing_group_slot(&group, 60, stride).offset, stride);

  IREE_EXPECT_NOT_OK(loom_runtime_backing_group_assign(
      &group, iree_allocator_null(), 3, stride, {second, 7 * stride}));
  ASSERT_EQ(group.assigned, (UINT64_C(3) << 59) | last);
  IREE_ASSERT_OK(loom_runtime_backing_group_assign(&group, allocator, 3, stride,
                                                   {second, 7 * stride}));
  IREE_ASSERT_OK(loom_runtime_backing_group_assign(
      &group, allocator, UINT64_C(3) << 61, stride, {second, 9 * stride}));

  uint64_t selected = group.assigned;
  auto check_run = [&](int expected_position, iree_host_size_t expected_count,
                       loom_runtime_backing_parent_t* expected_parent,
                       uint64_t expected_offset) {
    int position = 0;
    loom_runtime_backing_slot_t slot;
    const auto count = loom_runtime_backing_group_take_lease(
        &group, &selected, stride, &position, &slot);
    EXPECT_EQ(position, expected_position);
    EXPECT_EQ(count, expected_count);
    EXPECT_EQ(slot.parent, expected_parent);
    EXPECT_EQ(slot.offset, expected_offset);
  };
  check_run(0, 2, second, 7 * stride);
  check_run(59, 2, first, 0);
  check_run(61, 2, second, 9 * stride);
  check_run(63, 1, first, 4 * stride);
  EXPECT_EQ(selected, 0u);
  loom_runtime_backing_group_clear(&group, allocator, ~last, stride);
  EXPECT_EQ(group.fragment_capacity, 0u);
  EXPECT_EQ(loom_runtime_backing_group_slot(&group, 63, stride).offset,
            4 * stride);
  loom_runtime_backing_group_clear(&group, allocator, last, stride);
  EXPECT_EQ(group.assigned, 0u);
}

}  // namespace
