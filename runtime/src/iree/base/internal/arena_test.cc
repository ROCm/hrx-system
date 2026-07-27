// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"

#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Block size used by all tests. 4KB is a reasonable default that fits several
// allocations per block and exercises the block-chaining logic.
static constexpr iree_host_size_t kBlockSize = 4096;

//===----------------------------------------------------------------------===//
// iree_arena_block_pool_t
//===----------------------------------------------------------------------===//

TEST(ArenaBlockPool, Lifetime) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  EXPECT_EQ(pool.total_block_size, kBlockSize);
  EXPECT_EQ(pool.usable_block_size, kBlockSize - sizeof(iree_arena_block_t));
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(ArenaBlockPool, Preallocate) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  IREE_ASSERT_OK(iree_arena_block_pool_preallocate(&pool, 4));
  // Preallocated blocks should be available for acquire.
  for (int i = 0; i < 4; ++i) {
    iree_arena_block_t* block = NULL;
    void* ptr = NULL;
    IREE_ASSERT_OK(iree_arena_block_pool_acquire(&pool, &block, &ptr));
    ASSERT_NE(block, nullptr);
    ASSERT_NE(ptr, nullptr);
    iree_arena_block_pool_release(&pool, block, block);
  }
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(ArenaBlockPool, AcquireRelease) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_block_t* block = NULL;
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_block_pool_acquire(&pool, &block, &ptr));
  ASSERT_NE(block, nullptr);
  ASSERT_NE(ptr, nullptr);
  // The returned pointer should be usable for the full usable_block_size.
  memset(ptr, 0xAB, pool.usable_block_size);
  iree_arena_block_pool_release(&pool, block, block);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(ArenaBlockPool, Trim) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  IREE_ASSERT_OK(iree_arena_block_pool_preallocate(&pool, 8));
  // Trim releases all free blocks back to the system allocator.
  iree_arena_block_pool_trim(&pool);
  // The pool is still usable after trimming — new acquires allocate fresh.
  iree_arena_block_t* block = NULL;
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_block_pool_acquire(&pool, &block, &ptr));
  ASSERT_NE(block, nullptr);
  iree_arena_block_pool_release(&pool, block, block);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(ArenaBlockPool, StatisticsTrackOnlySystemAllocations) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);

  iree_arena_block_pool_statistics_t statistics;
  iree_arena_block_pool_query_statistics(&pool, &statistics);
  EXPECT_EQ(statistics.block_system_allocation_count, 0u);
  EXPECT_EQ(statistics.block_system_allocation_bytes, 0u);
  EXPECT_EQ(statistics.oversized_allocation_count, 0u);
  EXPECT_EQ(statistics.oversized_allocation_bytes, 0u);

  IREE_ASSERT_OK(iree_arena_block_pool_preallocate(&pool, 2));
  iree_arena_block_pool_query_statistics(&pool, &statistics);
#if IREE_STATISTICS_ENABLE
  EXPECT_EQ(statistics.block_system_allocation_count, 2u);
  EXPECT_EQ(statistics.block_system_allocation_bytes, kBlockSize * 2);
#else
  EXPECT_EQ(statistics.block_system_allocation_count, 0u);
  EXPECT_EQ(statistics.block_system_allocation_bytes, 0u);
#endif  // IREE_STATISTICS_ENABLE

  iree_arena_block_t* first_block = NULL;
  void* first_ptr = NULL;
  IREE_ASSERT_OK(
      iree_arena_block_pool_acquire(&pool, &first_block, &first_ptr));
  iree_arena_block_t* second_block = NULL;
  void* second_ptr = NULL;
  IREE_ASSERT_OK(
      iree_arena_block_pool_acquire(&pool, &second_block, &second_ptr));
  iree_arena_block_pool_query_statistics(&pool, &statistics);
#if IREE_STATISTICS_ENABLE
  EXPECT_EQ(statistics.block_system_allocation_count, 2u);
#else
  EXPECT_EQ(statistics.block_system_allocation_count, 0u);
#endif  // IREE_STATISTICS_ENABLE

  iree_arena_block_pool_release(&pool, first_block, first_block);
  iree_arena_block_pool_release(&pool, second_block, second_block);
  iree_arena_block_pool_trim(&pool);

  iree_arena_block_t* cold_block = NULL;
  void* cold_ptr = NULL;
  IREE_ASSERT_OK(iree_arena_block_pool_acquire(&pool, &cold_block, &cold_ptr));
  iree_arena_block_pool_query_statistics(&pool, &statistics);
#if IREE_STATISTICS_ENABLE
  EXPECT_EQ(statistics.block_system_allocation_count, 3u);
  EXPECT_EQ(statistics.block_system_allocation_bytes, kBlockSize * 3);
#else
  EXPECT_EQ(statistics.block_system_allocation_count, 0u);
  EXPECT_EQ(statistics.block_system_allocation_bytes, 0u);
#endif  // IREE_STATISTICS_ENABLE

  iree_arena_block_pool_release(&pool, cold_block, cold_block);
  iree_arena_block_pool_deinitialize(&pool);
}

//===----------------------------------------------------------------------===//
// iree_arena_allocator_t
//===----------------------------------------------------------------------===//

TEST(Arena, Lifetime) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, BasicAllocation) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, 128, &ptr));
  ASSERT_NE(ptr, nullptr);
  // Returned memory should be writable.
  memset(ptr, 0xCD, 128);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, MultipleAllocations) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Allocate several chunks that fit within a single block.
  void* pointers[16] = {};
  for (int i = 0; i < 16; ++i) {
    IREE_ASSERT_OK(iree_arena_allocate(&arena, 64, &pointers[i]));
    ASSERT_NE(pointers[i], nullptr);
  }
  // All pointers should be distinct.
  for (int i = 0; i < 16; ++i) {
    for (int j = i + 1; j < 16; ++j) {
      EXPECT_NE(pointers[i], pointers[j])
          << "allocations " << i << " and " << j << " overlap";
    }
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, NaturalAlignment) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Every allocation from the arena should be naturally aligned to
  // iree_max_align_t regardless of the requested size.
  for (iree_host_size_t size = 1; size <= 37; ++size) {
    void* ptr = NULL;
    IREE_ASSERT_OK(iree_arena_allocate(&arena, size, &ptr));
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignof(max_align_t), 0u)
        << "allocation of " << size << " bytes not naturally aligned";
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, OversizedAllocation) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Allocate something larger than the block size.
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, kBlockSize * 4, &ptr));
  ASSERT_NE(ptr, nullptr);
  memset(ptr, 0xEF, kBlockSize * 4);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, AllocationThatOnlyFitsBeforeAlignmentIsOversized) {
  if (iree_alignof(iree_arena_block_t) == iree_max_align_t) {
    GTEST_SKIP() << "every representable block has naturally aligned usable "
                    "space on this ABI";
  }

  static constexpr iree_host_size_t kOddBlockSize =
      sizeof(iree_arena_block_t) + iree_max_align_t * 8 +
      iree_alignof(iree_arena_block_t);
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kOddBlockSize, iree_allocator_system(),
                                   &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  ASSERT_FALSE(
      iree_host_size_has_alignment(pool.usable_block_size, iree_max_align_t));

  void* block_limit = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(
      &arena, iree_arena_block_pool_max_allocation_size(&pool), &block_limit));
  ASSERT_NE(block_limit, nullptr);
  EXPECT_NE(arena.block_head, nullptr);
  EXPECT_EQ(arena.allocation_head, nullptr);
  iree_arena_block_t* const full_block = arena.block_head;

  void* near_block_limit = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate(&arena, pool.usable_block_size, &near_block_limit));
  ASSERT_NE(near_block_limit, nullptr);
  memset(near_block_limit, 0xAB, pool.usable_block_size);
  EXPECT_EQ(arena.block_head, full_block);
  EXPECT_NE(arena.allocation_head, nullptr);

  void* block_allocation = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, 32, &block_allocation));
  ASSERT_NE(block_allocation, nullptr);
  memset(block_allocation, 0xCD, 32);
  EXPECT_NE(arena.block_head, full_block);
  EXPECT_LE(arena.block_bytes_remaining, pool.usable_block_size);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, Reset) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Allocate some memory and then reset. The arena should be reusable.
  for (int round = 0; round < 3; ++round) {
    void* ptr = NULL;
    IREE_ASSERT_OK(iree_arena_allocate(&arena, 256, &ptr));
    ASSERT_NE(ptr, nullptr);
    // Also allocate an oversized chunk to exercise that cleanup path.
    IREE_ASSERT_OK(iree_arena_allocate(&arena, kBlockSize * 2, &ptr));
    ASSERT_NE(ptr, nullptr);
    iree_arena_reset(&arena);
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, StatisticsTrackSystemAllocationAcrossRestoreAndReuse) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, 128, &ptr));
  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(&arena);

  const iree_host_size_t half_plus_one = pool.usable_block_size / 2 + 1;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, half_plus_one, &ptr));
  IREE_ASSERT_OK(iree_arena_allocate(&arena, half_plus_one, &ptr));
  const iree_host_size_t oversized_payload_bytes = kBlockSize * 2;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, oversized_payload_bytes, &ptr));
#if IREE_STATISTICS_ENABLE
  const iree_host_size_t oversized_allocation_bytes =
      sizeof(iree_arena_oversized_allocation_t) + oversized_payload_bytes;
#endif  // IREE_STATISTICS_ENABLE

  iree_arena_block_pool_statistics_t statistics;
  iree_arena_block_pool_query_statistics(&pool, &statistics);
#if IREE_STATISTICS_ENABLE
  EXPECT_EQ(statistics.block_system_allocation_count, 2u);
  EXPECT_EQ(statistics.block_system_allocation_bytes, kBlockSize * 2);
  EXPECT_EQ(statistics.oversized_allocation_count, 1u);
  EXPECT_EQ(statistics.oversized_allocation_bytes, oversized_allocation_bytes);
#else
  EXPECT_EQ(statistics.block_system_allocation_count, 0u);
  EXPECT_EQ(statistics.block_system_allocation_bytes, 0u);
  EXPECT_EQ(statistics.oversized_allocation_count, 0u);
  EXPECT_EQ(statistics.oversized_allocation_bytes, 0u);
#endif  // IREE_STATISTICS_ENABLE

  iree_arena_checkpoint_restore(&checkpoint);
  iree_arena_block_pool_query_statistics(&pool, &statistics);
#if IREE_STATISTICS_ENABLE
  EXPECT_EQ(statistics.block_system_allocation_count, 2u);
  EXPECT_EQ(statistics.oversized_allocation_count, 1u);
  EXPECT_EQ(statistics.oversized_allocation_bytes, oversized_allocation_bytes);
#else
  EXPECT_EQ(statistics.block_system_allocation_count, 0u);
  EXPECT_EQ(statistics.oversized_allocation_count, 0u);
  EXPECT_EQ(statistics.oversized_allocation_bytes, 0u);
#endif  // IREE_STATISTICS_ENABLE

  iree_arena_reset(&arena);
  IREE_ASSERT_OK(iree_arena_allocate(&arena, 128, &ptr));
  iree_arena_block_pool_query_statistics(&pool, &statistics);
#if IREE_STATISTICS_ENABLE
  EXPECT_EQ(statistics.block_system_allocation_count, 2u);
  EXPECT_EQ(statistics.oversized_allocation_count, 1u);
#else
  EXPECT_EQ(statistics.block_system_allocation_count, 0u);
  EXPECT_EQ(statistics.oversized_allocation_count, 0u);
#endif  // IREE_STATISTICS_ENABLE

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, CheckpointRestoreWithinBlock) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  uint8_t* retained_ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, 128, (void**)&retained_ptr));
  memset(retained_ptr, 0xAB, 128);
  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(&arena);

  void* discarded_ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, 64, &discarded_ptr));
  memset(discarded_ptr, 0xCD, 64);
  EXPECT_GT(arena.used_allocation_size, checkpoint.used_allocation_size);

  iree_arena_checkpoint_restore(&checkpoint);
  EXPECT_EQ(arena.total_allocation_size, checkpoint.total_allocation_size);
  EXPECT_EQ(arena.used_allocation_size, checkpoint.used_allocation_size);
  EXPECT_EQ(arena.block_head, checkpoint.block_head);
  EXPECT_EQ(arena.block_tail, checkpoint.block_tail);
  EXPECT_EQ(arena.block_bytes_remaining, checkpoint.block_bytes_remaining);
  for (iree_host_size_t i = 0; i < 128; ++i) {
    EXPECT_EQ(retained_ptr[i], 0xAB);
  }

  void* reused_ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, 64, &reused_ptr));
  EXPECT_EQ(reused_ptr, discarded_ptr);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, CheckpointRestoreReleasesBlocksAndOversizedAllocations) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  void* retained_ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, 128, &retained_ptr));
  ASSERT_NE(retained_ptr, nullptr);
  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(&arena);

  void* block_ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, kBlockSize / 2, &block_ptr));
  IREE_ASSERT_OK(iree_arena_allocate(&arena, kBlockSize / 2, &block_ptr));
  void* oversized_ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, kBlockSize * 2, &oversized_ptr));
  EXPECT_NE(arena.block_head, checkpoint.block_head);
  EXPECT_NE(arena.allocation_head, checkpoint.allocation_head);

  iree_arena_checkpoint_restore(&checkpoint);
  EXPECT_EQ(arena.allocation_head, checkpoint.allocation_head);
  EXPECT_EQ(arena.block_head, checkpoint.block_head);
  EXPECT_EQ(arena.block_tail, checkpoint.block_tail);
  EXPECT_EQ(arena.total_allocation_size, checkpoint.total_allocation_size);
  EXPECT_EQ(arena.used_allocation_size, checkpoint.used_allocation_size);
  EXPECT_EQ(arena.block_bytes_remaining, checkpoint.block_bytes_remaining);

  void* reused_ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, kBlockSize / 2, &reused_ptr));
  EXPECT_NE(reused_ptr, nullptr);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, OversizedAllocationAlignment) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Allocate something larger than a single block to force the oversized path.
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, kBlockSize * 2, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ((uintptr_t)ptr % iree_max_align_t, 0u)
      << "oversized allocation must be aligned to iree_max_align_t";

  // A second oversized allocation should also be aligned.
  void* ptr2 = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, kBlockSize + 1, &ptr2));
  ASSERT_NE(ptr2, nullptr);
  EXPECT_EQ((uintptr_t)ptr2 % iree_max_align_t, 0u)
      << "oversized allocation must be aligned to iree_max_align_t";

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, ArrayAllocationCheckedMul) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Normal array allocation should succeed.
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate_array(&arena, 10, 8, &ptr));
  ASSERT_NE(ptr, nullptr);

  // Overflow should return RESOURCE_EXHAUSTED.
  void* overflow_ptr = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &arena, IREE_HOST_SIZE_MAX, IREE_HOST_SIZE_MAX, &overflow_ptr);
  EXPECT_TRUE(iree_status_is_resource_exhausted(status));
  iree_status_ignore(status);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, BlockChaining) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Allocate enough to force multiple block acquisitions. Each allocation
  // consumes more than half the usable block size, so two cannot fit in one
  // block.
  iree_host_size_t half_plus_one = pool.usable_block_size / 2 + 1;
  void* ptr1 = NULL;
  void* ptr2 = NULL;
  IREE_ASSERT_OK(iree_arena_allocate(&arena, half_plus_one, &ptr1));
  IREE_ASSERT_OK(iree_arena_allocate(&arena, half_plus_one, &ptr2));
  ASSERT_NE(ptr1, nullptr);
  ASSERT_NE(ptr2, nullptr);
  EXPECT_NE(ptr1, ptr2);
  // Two allocations each exceeding half the block forced a second block.
  EXPECT_GE(arena.total_allocation_size, kBlockSize * 2);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

//===----------------------------------------------------------------------===//
// iree_arena_allocate_aligned
//===----------------------------------------------------------------------===//

TEST(Arena, AlignedAtOrBelowNatural) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Requesting alignment at or below iree_max_align_t should work the same as
  // a normal allocation — no extra padding.
  void* ptr = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate_aligned(&arena, 64, alignof(max_align_t), &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignof(max_align_t), 0u);

  // Sub-natural alignment.
  IREE_ASSERT_OK(iree_arena_allocate_aligned(&arena, 64, 4, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 4, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, AlignedAboveNatural) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Request 64-byte alignment (cache-line), which exceeds iree_max_align_t
  // on most platforms (typically 8 or 16).
  static constexpr iree_host_size_t kAlignment = 64;
  for (int i = 0; i < 8; ++i) {
    void* ptr = NULL;
    IREE_ASSERT_OK(iree_arena_allocate_aligned(&arena, 100, kAlignment, &ptr));
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % kAlignment, 0u)
        << "allocation " << i << " not aligned to " << kAlignment;
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, AlignedLargePowerOfTwo) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // 256-byte alignment with a small allocation — exercises the over-allocation
  // and pointer-forward logic with significant padding waste.
  static constexpr iree_host_size_t kAlignment = 256;
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate_aligned(&arena, 32, kAlignment, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % kAlignment, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, AlignedZeroLength) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Zero-length aligned allocation should succeed (returns a valid pointer
  // that happens to be aligned but has no usable bytes).
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_allocate_aligned(&arena, 0, 64, &ptr));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

//===----------------------------------------------------------------------===//
// iree_arena_grow_array
//===----------------------------------------------------------------------===//

TEST(Arena, GrowArrayFromEmpty) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  iree_host_size_t capacity = 0;
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_arena_grow_array(&arena, /*existing_count=*/0,
                                       /*minimum_capacity=*/4, sizeof(uint32_t),
                                       &capacity, &ptr));
  EXPECT_GE(capacity, 4u);
  EXPECT_NE(ptr, nullptr);

  // Write to verify the allocation is usable.
  uint32_t* array = (uint32_t*)ptr;
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    array[i] = (uint32_t)i;
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, GrowArrayCopiesExisting) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  // Start with an initial allocation.
  iree_host_size_t capacity = 4;
  void* ptr = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate_array(&arena, capacity, sizeof(uint32_t), &ptr));
  uint32_t* array = (uint32_t*)ptr;
  array[0] = 100;
  array[1] = 200;
  array[2] = 300;
  array[3] = 400;

  // Grow. Existing 4 elements should be preserved.
  IREE_ASSERT_OK(iree_arena_grow_array(&arena, /*existing_count=*/4,
                                       /*minimum_capacity=*/0, sizeof(uint32_t),
                                       &capacity, &ptr));
  EXPECT_GE(capacity, 8u);  // Doubled from 4.
  array = (uint32_t*)ptr;
  EXPECT_EQ(array[0], 100u);
  EXPECT_EQ(array[1], 200u);
  EXPECT_EQ(array[2], 300u);
  EXPECT_EQ(array[3], 400u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(Arena, GrowArrayRespectsMinimum) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  iree_host_size_t capacity = 2;
  void* ptr = NULL;
  IREE_ASSERT_OK(
      iree_arena_allocate_array(&arena, capacity, sizeof(uint32_t), &ptr));

  // minimum_capacity (100) > doubled (4), so should use 100.
  IREE_ASSERT_OK(iree_arena_grow_array(&arena, /*existing_count=*/2,
                                       /*minimum_capacity=*/100,
                                       sizeof(uint32_t), &capacity, &ptr));
  EXPECT_GE(capacity, 100u);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

//===----------------------------------------------------------------------===//
// iree_arena_allocator (iree_allocator_t interface)
//===----------------------------------------------------------------------===//

TEST(Arena, AllocatorInterface) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);

  iree_allocator_t allocator = iree_arena_allocator(&arena);

  // malloc.
  void* ptr = NULL;
  IREE_ASSERT_OK(iree_allocator_malloc(allocator, 64, &ptr));
  ASSERT_NE(ptr, nullptr);

  // calloc should zero-fill.
  void* zeroed = NULL;
  IREE_ASSERT_OK(iree_allocator_malloc(allocator, 0, &zeroed));
  // Free is a no-op (arenas don't free individual allocations).
  iree_allocator_free(allocator, ptr);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

}  // namespace
