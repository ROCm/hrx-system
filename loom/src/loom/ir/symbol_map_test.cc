// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/symbol_map.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class SymbolMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    memset(&map_, 0, sizeof(map_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_symbol_map_t map_;
};

TEST_F(SymbolMapTest, FindEmpty) {
  EXPECT_EQ(loom_symbol_map_find(&map_, 0), LOOM_SYMBOL_ID_INVALID);
  EXPECT_EQ(loom_symbol_map_find(&map_, 42), LOOM_SYMBOL_ID_INVALID);
}

TEST_F(SymbolMapTest, ReservedNameIdsAreRejected) {
  EXPECT_EQ(loom_symbol_map_find(&map_, LOOM_STRING_ID_INVALID),
            LOOM_SYMBOL_ID_INVALID);
  EXPECT_EQ(loom_symbol_map_find(&map_, LOOM_SYMBOL_MAP_TOMBSTONE),
            LOOM_SYMBOL_ID_INVALID);
  EXPECT_FALSE(loom_symbol_map_erase(&map_, LOOM_STRING_ID_INVALID));
  EXPECT_FALSE(loom_symbol_map_erase(&map_, LOOM_SYMBOL_MAP_TOMBSTONE));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_symbol_map_insert(&map_, &arena_, LOOM_STRING_ID_INVALID, 1));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_symbol_map_insert(&map_, &arena_, LOOM_SYMBOL_MAP_TOMBSTONE, 1));

  uint16_t symbol_id = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_symbol_map_find_or_insert(&map_, &arena_, LOOM_STRING_ID_INVALID, 1,
                                     &symbol_id));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_symbol_map_find_or_insert(&map_, &arena_, LOOM_SYMBOL_MAP_TOMBSTONE,
                                     1, &symbol_id));
}

TEST_F(SymbolMapTest, InsertAndFind) {
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 10, 5));
  EXPECT_EQ(loom_symbol_map_find(&map_, 10), 5);
}

TEST_F(SymbolMapTest, InsertMultiple) {
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 1, 100));
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 2, 200));
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 3, 300));
  EXPECT_EQ(loom_symbol_map_find(&map_, 1), 100);
  EXPECT_EQ(loom_symbol_map_find(&map_, 2), 200);
  EXPECT_EQ(loom_symbol_map_find(&map_, 3), 300);
}

TEST_F(SymbolMapTest, FindMissing) {
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 5, 50));
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 10, 100));
  EXPECT_EQ(loom_symbol_map_find(&map_, 7), LOOM_SYMBOL_ID_INVALID);
  EXPECT_EQ(loom_symbol_map_find(&map_, 0), LOOM_SYMBOL_ID_INVALID);
}

TEST_F(SymbolMapTest, FindOrInsert_NewEntry) {
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_symbol_map_find_or_insert(&map_, &arena_, 10, 42, &symbol_id));
  EXPECT_EQ(symbol_id, 42);
  EXPECT_EQ(loom_symbol_map_find(&map_, 10), 42);
}

TEST_F(SymbolMapTest, FindOrInsert_ExistingEntry) {
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 10, 42));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_symbol_map_find_or_insert(&map_, &arena_, 10, 99, &symbol_id));
  // Returns existing value, not the new one.
  EXPECT_EQ(symbol_id, 42);
  EXPECT_EQ(loom_symbol_map_find(&map_, 10), 42);
}

TEST_F(SymbolMapTest, FindOrInsert_Mixed) {
  // Simulate parser pattern: define @main, reference @main, define @helper.
  uint16_t id = LOOM_SYMBOL_ID_INVALID;

  IREE_ASSERT_OK(loom_symbol_map_find_or_insert(&map_, &arena_, 1, 0, &id));
  EXPECT_EQ(id, 0);  // New: @main gets symbol 0.

  IREE_ASSERT_OK(loom_symbol_map_find_or_insert(&map_, &arena_, 1, 99, &id));
  EXPECT_EQ(id, 0);  // Existing: @main still symbol 0.

  IREE_ASSERT_OK(loom_symbol_map_find_or_insert(&map_, &arena_, 2, 1, &id));
  EXPECT_EQ(id, 1);  // New: @helper gets symbol 1.
}

TEST_F(SymbolMapTest, GrowthRehash) {
  // Insert enough entries to trigger multiple growth events.
  // Initial capacity is 16, grows at ~70% occupancy.
  for (uint32_t i = 0; i < 200; ++i) {
    IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, (loom_string_id_t)i,
                                          (uint16_t)i));
  }
  // Verify all entries survived rehashing.
  for (uint32_t i = 0; i < 200; ++i) {
    EXPECT_EQ(loom_symbol_map_find(&map_, (loom_string_id_t)i), (uint16_t)i)
        << "name_id=" << i;
  }
  EXPECT_EQ(map_.count, 200u);
  EXPECT_GE(map_.capacity, 200u);
}

TEST_F(SymbolMapTest, SequentialKeys) {
  // Realistic pattern: interned string IDs are assigned 0, 1, 2, ...
  for (uint32_t i = 0; i < 1000; ++i) {
    IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, (loom_string_id_t)i,
                                          (uint16_t)i));
  }
  for (uint32_t i = 0; i < 1000; ++i) {
    EXPECT_EQ(loom_symbol_map_find(&map_, (loom_string_id_t)i), (uint16_t)i);
  }
}

TEST_F(SymbolMapTest, SparseKeys) {
  // Non-sequential name_ids (not all strings are symbol names).
  loom_string_id_t keys[] = {3, 17, 42, 100, 255, 1000, 5000};
  for (int i = 0; i < 7; ++i) {
    IREE_ASSERT_OK(
        loom_symbol_map_insert(&map_, &arena_, keys[i], (uint16_t)i));
  }
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(loom_symbol_map_find(&map_, keys[i]), (uint16_t)i);
  }
  // Gaps should return INVALID.
  EXPECT_EQ(loom_symbol_map_find(&map_, 0), LOOM_SYMBOL_ID_INVALID);
  EXPECT_EQ(loom_symbol_map_find(&map_, 4), LOOM_SYMBOL_ID_INVALID);
  EXPECT_EQ(loom_symbol_map_find(&map_, 50), LOOM_SYMBOL_ID_INVALID);
}

TEST_F(SymbolMapTest, Erase_Found) {
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 10, 5));
  EXPECT_EQ(loom_symbol_map_find(&map_, 10), 5);
  EXPECT_TRUE(loom_symbol_map_erase(&map_, 10));
  EXPECT_EQ(loom_symbol_map_find(&map_, 10), LOOM_SYMBOL_ID_INVALID);
  EXPECT_EQ(map_.count, 0u);
  EXPECT_EQ(map_.tombstone_count, 1u);
}

TEST_F(SymbolMapTest, Erase_NotFound) {
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 10, 5));
  EXPECT_FALSE(loom_symbol_map_erase(&map_, 99));
  EXPECT_EQ(map_.count, 1u);
  EXPECT_EQ(map_.tombstone_count, 0u);
}

TEST_F(SymbolMapTest, Erase_EmptyMap) {
  EXPECT_FALSE(loom_symbol_map_erase(&map_, 10));
}

TEST_F(SymbolMapTest, Erase_ThenReinsertSameKey) {
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 10, 5));
  EXPECT_TRUE(loom_symbol_map_erase(&map_, 10));
  // Reinsert with a different symbol_id.
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 10, 77));
  EXPECT_EQ(loom_symbol_map_find(&map_, 10), 77);
  EXPECT_EQ(map_.count, 1u);
  // Tombstone was reclaimed by the insert.
  EXPECT_EQ(map_.tombstone_count, 0u);
}

TEST_F(SymbolMapTest, Erase_ThenInsertDifferentKey) {
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 10, 5));
  EXPECT_TRUE(loom_symbol_map_erase(&map_, 10));
  // Insert a different key. May or may not reuse the tombstone slot
  // depending on hash distribution, but the map must be correct.
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 20, 77));
  EXPECT_EQ(loom_symbol_map_find(&map_, 10), LOOM_SYMBOL_ID_INVALID);
  EXPECT_EQ(loom_symbol_map_find(&map_, 20), 77);
}

TEST_F(SymbolMapTest, Erase_ManyThenGrow) {
  // Fill, erase most, then insert more to trigger growth.
  // Tombstones should be cleaned up during rehash.
  for (uint32_t i = 0; i < 10; ++i) {
    IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, (loom_string_id_t)i,
                                          (uint16_t)i));
  }
  for (uint32_t i = 0; i < 8; ++i) {
    EXPECT_TRUE(loom_symbol_map_erase(&map_, (loom_string_id_t)i));
  }
  EXPECT_EQ(map_.count, 2u);
  EXPECT_EQ(map_.tombstone_count, 8u);

  // Insert enough to trigger growth past the tombstone-inflated load.
  for (uint32_t i = 100; i < 120; ++i) {
    IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, (loom_string_id_t)i,
                                          (uint16_t)i));
  }
  // After growth, tombstones are dropped.
  EXPECT_EQ(map_.tombstone_count, 0u);
  EXPECT_EQ(map_.count, 22u);

  // Erased entries stay gone.
  for (uint32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(loom_symbol_map_find(&map_, (loom_string_id_t)i),
              LOOM_SYMBOL_ID_INVALID);
  }
  // Surviving and new entries are present.
  EXPECT_EQ(loom_symbol_map_find(&map_, 8), 8);
  EXPECT_EQ(loom_symbol_map_find(&map_, 9), 9);
  for (uint32_t i = 100; i < 120; ++i) {
    EXPECT_EQ(loom_symbol_map_find(&map_, (loom_string_id_t)i), (uint16_t)i);
  }
}

TEST_F(SymbolMapTest, Erase_ProbeChainIntegrity) {
  // Force a probe chain by inserting keys that hash to the same slot,
  // then erase the middle one. Keys after the tombstone must still be
  // found. We insert enough sequential IDs that some will collide
  // (capacity 16, mask 15, so IDs with the same low bits of their
  // Fibonacci hash will share a slot).
  for (uint32_t i = 0; i < 14; ++i) {
    IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, (loom_string_id_t)i,
                                          (uint16_t)i));
  }
  // Erase entries in the middle of the ID range.
  EXPECT_TRUE(loom_symbol_map_erase(&map_, 5));
  EXPECT_TRUE(loom_symbol_map_erase(&map_, 6));
  EXPECT_TRUE(loom_symbol_map_erase(&map_, 7));

  // All non-erased entries must still be found.
  for (uint32_t i = 0; i < 14; ++i) {
    if (i >= 5 && i <= 7) {
      EXPECT_EQ(loom_symbol_map_find(&map_, (loom_string_id_t)i),
                LOOM_SYMBOL_ID_INVALID)
          << "erased name_id=" << i << " should not be found";
    } else {
      EXPECT_EQ(loom_symbol_map_find(&map_, (loom_string_id_t)i), (uint16_t)i)
          << "name_id=" << i << " should be found";
    }
  }
}

TEST_F(SymbolMapTest, FindOrInsert_AfterErase) {
  IREE_ASSERT_OK(loom_symbol_map_insert(&map_, &arena_, 10, 5));
  EXPECT_TRUE(loom_symbol_map_erase(&map_, 10));

  // find_or_insert on an erased key should insert fresh.
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_symbol_map_find_or_insert(&map_, &arena_, 10, 77, &symbol_id));
  EXPECT_EQ(symbol_id, 77);
  EXPECT_EQ(loom_symbol_map_find(&map_, 10), 77);
}

}  // namespace
}  // namespace loom
