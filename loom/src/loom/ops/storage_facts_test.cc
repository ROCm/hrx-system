// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/storage_facts.h"

#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class StorageFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table_, &arena_, 0));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Backing blocks for interned reference payloads.
  iree_arena_block_pool_t block_pool_;
  // Storage retained for the lifetime of the fact table.
  iree_arena_allocator_t arena_;
  // Real interner used by storage fact construction and queries.
  loom_value_fact_table_t table_;
};

TEST_F(StorageFactsTest, EqualReferencesShareInternedIdentity) {
  for (uint32_t space = 0; space < LOOM_STORAGE_SPACE_COUNT_; ++space) {
    loom_storage_reference_facts_t references[2];
    std::memset(&references[0], 0xA5, sizeof(references[0]));
    std::memset(&references[1], 0x5A, sizeof(references[1]));
    loom_value_facts_t facts[2];
    for (int i = 0; i < 2; ++i) {
      auto& reference = references[i];
      // Populate semantic fields independently of their previous storage.
      // Equal references must intern identically after ordinary member stores.
      reference.byte_offset = loom_value_facts_exact_i64(16);
      reference.target_byte_offset = loom_value_facts_unknown();
      reference.valid_byte_length = loom_value_facts_exact_i64(64);
      reference.minimum_alignment = 16;
      reference.backing_value_id = 1;
      reference.storage_space = space;
      IREE_ASSERT_OK(loom_storage_facts_make_reference(&table_.context,
                                                       &reference, &facts[i]));
    }
    EXPECT_NE(facts[0].extension_id, LOOM_VALUE_FACT_EXTENSION_ID_NONE);
    EXPECT_EQ(facts[0].extension_id, facts[1].extension_id);

    references[1].backing_value_id = 2;
    IREE_ASSERT_OK(loom_storage_facts_make_reference(
        &table_.context, &references[1], &facts[1]));
    EXPECT_NE(facts[0].extension_id, facts[1].extension_id);
  }
}

TEST_F(StorageFactsTest, ReservationAndViewPreserveBackingIdentity) {
  loom_value_facts_t reservation;
  IREE_ASSERT_OK(loom_storage_facts_make_reserve(
      &table_.context, 7, LOOM_STORAGE_SPACE_WORKGROUP, 128, 64, &reservation));
  loom_value_facts_t view;
  IREE_ASSERT_OK(loom_storage_facts_make_static_view(
      &table_.context, reservation, 16, 64, &view));

  loom_storage_reference_facts_t reference;
  ASSERT_TRUE(
      loom_storage_facts_query_reference(&table_.context, view, &reference));
  EXPECT_EQ(reference.backing_value_id, 7u);
  EXPECT_EQ(reference.storage_space, LOOM_STORAGE_SPACE_WORKGROUP);
  EXPECT_EQ(reference.minimum_alignment, 16u);
  EXPECT_EQ(reference.byte_offset.range_lo, 16);
  EXPECT_EQ(reference.byte_offset.range_hi, 16);
  EXPECT_EQ(reference.valid_byte_length.range_lo, 64);
  EXPECT_EQ(reference.valid_byte_length.range_hi, 64);

  loom_value_facts_t repeated_view;
  IREE_ASSERT_OK(loom_storage_facts_make_static_view(
      &table_.context, reservation, 16, 64, &repeated_view));
  EXPECT_EQ(view.extension_id, repeated_view.extension_id);
}

}  // namespace
}  // namespace loom
