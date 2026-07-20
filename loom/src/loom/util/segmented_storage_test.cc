// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/segmented_storage.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class SegmentedStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(SegmentedStorageTest, InlineAndPrimaryPagePointersStayStable) {
  loom_segmented_storage_t storage;
  loom_segmented_storage_initialize(sizeof(uint64_t), alignof(uint64_t),
                                    &storage);

  constexpr uint32_t kSegmentCount =
      LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT + 1;
  uint64_t* pointers[kSegmentCount];
  for (uint32_t i = 0; i < kSegmentCount; ++i) {
    void* segment = nullptr;
    IREE_ASSERT_OK(loom_segmented_storage_append(&storage, &arena_, &segment));
    pointers[i] = static_cast<uint64_t*>(segment);
    *pointers[i] = 0xCAFE0000u + i;
  }

  EXPECT_EQ(storage.segment_count, kSegmentCount);
  for (uint32_t i = 0; i < kSegmentCount; ++i) {
    EXPECT_EQ(loom_segmented_storage_segment(&storage, i), pointers[i]);
    EXPECT_EQ(*static_cast<const uint64_t*>(
                  loom_segmented_storage_const_segment(&storage, i)),
              0xCAFE0000u + i);
  }
}

TEST_F(SegmentedStorageTest, SecondaryPointerPage) {
  loom_segmented_storage_t storage;
  loom_segmented_storage_initialize(sizeof(uint32_t), alignof(uint32_t),
                                    &storage);

  constexpr uint32_t kSegmentCount =
      LOOM_SEGMENTED_STORAGE_SEGMENTS_PER_PAGE + 1;
  for (uint32_t i = 0; i < kSegmentCount; ++i) {
    void* segment = nullptr;
    IREE_ASSERT_OK(loom_segmented_storage_append(&storage, &arena_, &segment));
    *static_cast<uint32_t*>(segment) = i;
  }

  EXPECT_EQ(storage.segment_count, kSegmentCount);
  for (uint32_t i = 0; i < kSegmentCount; ++i) {
    EXPECT_EQ(*static_cast<const uint32_t*>(
                  loom_segmented_storage_const_segment(&storage, i)),
              i);
  }
}

TEST_F(SegmentedStorageTest, MoveInlineDirectory) {
  loom_segmented_storage_t source;
  loom_segmented_storage_initialize(sizeof(uint32_t), alignof(uint32_t),
                                    &source);
  for (uint32_t i = 0; i < 8; ++i) {
    void* segment = nullptr;
    IREE_ASSERT_OK(loom_segmented_storage_append(&source, &arena_, &segment));
    *static_cast<uint32_t*>(segment) = i;
  }

  loom_segmented_storage_t storage;
  loom_segmented_storage_move(&source, &storage);

  EXPECT_EQ(source.segment_count, 0u);
  EXPECT_EQ(source.primary_page, nullptr);
  EXPECT_EQ(storage.primary_page, nullptr);
  for (uint32_t i = 0; i < storage.segment_count; ++i) {
    EXPECT_EQ(*static_cast<const uint32_t*>(
                  loom_segmented_storage_const_segment(&storage, i)),
              i);
  }
}

TEST_F(SegmentedStorageTest, InlineDirectoryCopyIsSelfContained) {
  loom_segmented_storage_t source;
  loom_segmented_storage_initialize(sizeof(uint32_t), alignof(uint32_t),
                                    &source);
  for (uint32_t i = 0; i < 8; ++i) {
    void* segment = nullptr;
    IREE_ASSERT_OK(loom_segmented_storage_append(&source, &arena_, &segment));
    *static_cast<uint32_t*>(segment) = i;
  }

  loom_segmented_storage_t copy = source;
  source.inline_segments[0] = nullptr;

  EXPECT_NE(loom_segmented_storage_const_segment(&copy, 0), nullptr);
  for (uint32_t i = 0; i < copy.segment_count; ++i) {
    EXPECT_EQ(*static_cast<const uint32_t*>(
                  loom_segmented_storage_const_segment(&copy, i)),
              i);
  }
}

TEST_F(SegmentedStorageTest, MoveExpandedDirectory) {
  loom_segmented_storage_t source;
  loom_segmented_storage_initialize(sizeof(uint32_t), alignof(uint32_t),
                                    &source);
  constexpr uint32_t kSegmentCount =
      LOOM_SEGMENTED_STORAGE_INLINE_SEGMENT_COUNT + 1;
  for (uint32_t i = 0; i < kSegmentCount; ++i) {
    void* segment = nullptr;
    IREE_ASSERT_OK(loom_segmented_storage_append(&source, &arena_, &segment));
    *static_cast<uint32_t*>(segment) = i;
  }
  void** source_primary_page = source.primary_page;

  loom_segmented_storage_t storage;
  loom_segmented_storage_move(&source, &storage);

  EXPECT_EQ(source.segment_count, 0u);
  EXPECT_EQ(storage.primary_page, source_primary_page);
  for (uint32_t i = 0; i < storage.segment_count; ++i) {
    EXPECT_EQ(*static_cast<const uint32_t*>(
                  loom_segmented_storage_const_segment(&storage, i)),
              i);
  }
}

TEST_F(SegmentedStorageTest, OveralignedPayloads) {
  loom_segmented_storage_t storage;
  loom_segmented_storage_initialize(/*segment_size=*/192,
                                    /*segment_alignment=*/256, &storage);

  for (uint32_t i = 0; i < 64; ++i) {
    void* segment = nullptr;
    IREE_ASSERT_OK(loom_segmented_storage_append(&storage, &arena_, &segment));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(segment) % 256, 0u);
  }
}

TEST_F(SegmentedStorageTest, ArenaResetReusesPoolBlocks) {
  loom_segmented_storage_t storage;
  loom_segmented_storage_initialize(/*segment_size=*/4096,
                                    /*segment_alignment=*/64, &storage);
  for (uint32_t i = 0; i < 32; ++i) {
    void* segment = nullptr;
    IREE_ASSERT_OK(loom_segmented_storage_append(&storage, &arena_, &segment));
  }

  iree_arena_block_pool_statistics_t warm_statistics;
  iree_arena_block_pool_query_statistics(&block_pool_, &warm_statistics);
  iree_arena_reset(&arena_);
  loom_segmented_storage_initialize(/*segment_size=*/4096,
                                    /*segment_alignment=*/64, &storage);
  for (uint32_t i = 0; i < 32; ++i) {
    void* segment = nullptr;
    IREE_ASSERT_OK(loom_segmented_storage_append(&storage, &arena_, &segment));
  }

  iree_arena_block_pool_statistics_t reused_statistics;
  iree_arena_block_pool_query_statistics(&block_pool_, &reused_statistics);
  EXPECT_EQ(reused_statistics.block_system_allocation_count,
            warm_statistics.block_system_allocation_count);
  EXPECT_EQ(reused_statistics.oversized_allocation_count,
            warm_statistics.oversized_allocation_count);
}

}  // namespace
}  // namespace loom
