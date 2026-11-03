// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/backing_directory.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(BackingDirectoryTest, ReusesTombstonesAtTheLoadLimit) {
  loom_serve_backing_directory_t directory;
  loom_serve_backing_directory_initialize(iree_allocator_system(), &directory);
  for (iree_host_size_t index = 0; index < 6; ++index) {
    loom_serve_backing_entry_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_serve_backing_directory_insert(&directory, index, &entry));
  }
  ASSERT_EQ(directory.capacity, 8u);
  for (iree_host_size_t index = 0; index < 5; ++index) {
    loom_serve_backing_directory_remove(
        &directory, loom_serve_backing_directory_find(&directory, index));
  }
  ASSERT_EQ(directory.tombstone_count, 5u);

  loom_serve_backing_entry_t* replacement = nullptr;
  IREE_ASSERT_OK(
      loom_serve_backing_directory_insert(&directory, 100, &replacement));
  EXPECT_EQ(directory.capacity, 8u);
  EXPECT_EQ(directory.tombstone_count, 4u);
  loom_serve_backing_directory_deinitialize(&directory);
}

TEST(BackingDirectoryTest, RehashesAndReusesDeletedProbeSlots) {
  loom_serve_backing_directory_t directory;
  loom_serve_backing_directory_initialize(iree_allocator_system(), &directory);

  // These keys share their initial bucket at the first table capacity.
  for (const iree_host_size_t index : {0u, 8u, 16u}) {
    loom_serve_backing_entry_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_serve_backing_directory_insert(&directory, index, &entry));
    entry->keep = true;
  }
  EXPECT_EQ(directory.count, 3u);
  loom_serve_backing_directory_remove(
      &directory, loom_serve_backing_directory_find(&directory, 8));
  EXPECT_EQ(directory.count, 2u);
  EXPECT_NE(loom_serve_backing_directory_find(&directory, 16), nullptr);

  loom_serve_backing_entry_t* replacement = nullptr;
  IREE_ASSERT_OK(
      loom_serve_backing_directory_insert(&directory, 24, &replacement));
  EXPECT_EQ(directory.count, 3u);
  EXPECT_EQ(directory.tombstone_count, 0u);

  for (iree_host_size_t i = 1; i <= 128; ++i) {
    loom_serve_backing_entry_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_serve_backing_directory_insert(&directory, i * 7, &entry));
  }
  loom_serve_backing_entry_t* high = nullptr;
  IREE_ASSERT_OK(loom_serve_backing_directory_insert(
      &directory, IREE_HOST_SIZE_MAX - 123, &high));
  high->in_flight = true;
  for (iree_host_size_t i = 1; i <= 128; ++i) {
    EXPECT_NE(loom_serve_backing_directory_find_const(&directory, i * 7),
              nullptr);
  }
  EXPECT_EQ(loom_serve_backing_directory_find_const(&directory,
                                                    IREE_HOST_SIZE_MAX - 123),
            high);

  loom_serve_backing_directory_discard_empty(&directory);
  EXPECT_EQ(directory.count, 1u);
  EXPECT_GT(directory.capacity, 8u);
  IREE_ASSERT_OK(loom_serve_backing_directory_shrink(&directory));
  EXPECT_EQ(directory.capacity, 8u);
  const auto* shrunk_high = loom_serve_backing_directory_find_const(
      &directory, IREE_HOST_SIZE_MAX - 123);
  ASSERT_NE(shrunk_high, nullptr);
  EXPECT_TRUE(shrunk_high->in_flight);
  loom_serve_backing_directory_deinitialize(&directory);
}

}  // namespace
