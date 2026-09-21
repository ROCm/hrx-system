// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_layout.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class FactLayoutTest : public ::testing::Test {
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

  // Backing storage shared by independently scoped fact tables.
  iree_arena_block_pool_t block_pool_;
  // Storage owning the primary fact table and its bindings.
  iree_arena_allocator_t arena_;
  // Primary scope under test.
  loom_value_fact_table_t table_ = {};
};

TEST_F(FactLayoutTest, AbsentBindingsDoNotAllocate) {
  const auto bytes = arena_.used_allocation_size;
  EXPECT_EQ(loom_value_fact_table_query_layout_strides(&table_, 100).count, 0);
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_strides(&table_, 100, {}));
  IREE_ASSERT_OK(loom_value_fact_table_forward_layout_strides(&table_, 1, 100));
  EXPECT_EQ(table_.layout_origins, nullptr);
  EXPECT_EQ(arena_.used_allocation_size, bytes);
}

TEST_F(FactLayoutTest, BindingsOwnCopiedIdsAndReuseUnchangedStorage) {
  loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID, 2};
  IREE_ASSERT_OK(
      loom_value_fact_table_define_layout_strides(&table_, 3, {strides, 3}));
  const auto first = loom_value_fact_table_query_layout_strides(&table_, 3);
  ASSERT_EQ(first.count, 3);
  EXPECT_NE(first.values, strides);
  const auto bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(
      loom_value_fact_table_define_layout_strides(&table_, 3, {strides, 3}));
  EXPECT_EQ(loom_value_fact_table_query_layout_strides(&table_, 3).values,
            first.values);
  EXPECT_EQ(arena_.used_allocation_size, bytes);

  strides[0] = 4;
  IREE_ASSERT_OK(
      loom_value_fact_table_define_layout_strides(&table_, 3, {strides, 2}));
  const auto second = loom_value_fact_table_query_layout_strides(&table_, 3);
  ASSERT_EQ(second.count, 2);
  EXPECT_EQ(second.values[0], 4);
  EXPECT_EQ(second.values[1], LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(first.values[0], 1);
}

TEST_F(FactLayoutTest, ForwardingSharesAndReplacesScopedBindings) {
  const loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID};
  IREE_ASSERT_OK(
      loom_value_fact_table_define_layout_strides(&table_, 2, {strides, 2}));
  IREE_ASSERT_OK(loom_value_fact_table_forward_layout_strides(&table_, 2, 100));
  EXPECT_EQ(loom_value_fact_table_query_layout_strides(&table_, 100).values,
            loom_value_fact_table_query_layout_strides(&table_, 2).values);
  IREE_ASSERT_OK(loom_value_fact_table_forward_layout_strides(&table_, 3, 100));
  EXPECT_EQ(loom_value_fact_table_query_layout_strides(&table_, 100).count, 0);
  EXPECT_EQ(loom_value_fact_table_query_layout_strides(&table_, 2).count, 2);
}

TEST_F(FactLayoutTest, UndefinitionAndScopeResetWithdrawBindings) {
  const loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID};
  IREE_ASSERT_OK(
      loom_value_fact_table_define_layout_strides(&table_, 2, {strides, 2}));
  IREE_ASSERT_OK(loom_value_fact_table_forward_layout_strides(&table_, 2, 3));
  loom_value_fact_table_undefine(&table_, 2);
  EXPECT_EQ(loom_value_fact_table_query_layout_strides(&table_, 2).count, 0);
  EXPECT_EQ(loom_value_fact_table_query_layout_strides(&table_, 3).count, 2);
  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(table_.layout_origins, nullptr);
  EXPECT_EQ(loom_value_fact_table_query_layout_strides(&table_, 3).count, 0);
}

TEST_F(FactLayoutTest, NumericTransferDoesNotTransferSsaBindings) {
  const loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID};
  IREE_ASSERT_OK(
      loom_value_fact_table_define_layout_strides(&table_, 2, {strides, 2}));
  const loom_value_facts_t numeric_strides[] = {loom_value_facts_make(8, 12, 1),
                                                loom_value_facts_exact_i64(1)};
  loom_value_fact_encoding_summary_t summary = {};
  summary.address_layout = {LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED, 2,
                            numeric_strides};
  loom_value_facts_t facts;
  IREE_ASSERT_OK(
      loom_value_facts_make_encoding_summary(&table_.context, summary, &facts));
  loom_value_fact_table_t target = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &arena_, 0));
  loom_value_facts_t copied;
  IREE_ASSERT_OK(
      loom_value_fact_table_clone_fact(&target, &table_, facts, &copied));
  IREE_ASSERT_OK(loom_value_fact_table_define(&target, 3, copied));
  loom_value_fact_encoding_summary_t received;
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(&target.context, copied,
                                                      &received));
  EXPECT_EQ(received.address_layout.strides[0].range_lo, 8);
  EXPECT_EQ(received.address_layout.strides[0].range_hi, 12);
  EXPECT_EQ(loom_value_fact_table_query_layout_strides(&target, 3).count, 0);
  EXPECT_EQ(target.layout_origins, nullptr);
}

TEST_F(FactLayoutTest, SameIdSnapshotOwnsBindingsAfterSourceArenaReset) {
  iree_arena_allocator_t source_arena;
  iree_arena_initialize(&block_pool_, &source_arena);
  loom_value_fact_table_t source = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &source_arena, 0));
  const loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID};
  IREE_ASSERT_OK(
      loom_value_fact_table_define_layout_strides(&source, 2, {strides, 2}));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 2, loom_value_facts_unknown()));
  const loom_value_id_t values[] = {2};
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &table_, {&source, values, 1}, nullptr));
  EXPECT_NE(loom_value_fact_table_query_layout_strides(&table_, 2).values,
            loom_value_fact_table_query_layout_strides(&source, 2).values);
  iree_arena_deinitialize(&source_arena);
  const auto copied = loom_value_fact_table_query_layout_strides(&table_, 2);
  ASSERT_EQ(copied.count, 2);
  EXPECT_EQ(copied.values[0], 1);
  EXPECT_EQ(copied.values[1], LOOM_VALUE_ID_INVALID);
}

TEST_F(FactLayoutTest, SnapshotWithoutBindingWithdrawsPreviousOrigin) {
  const loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID};
  IREE_ASSERT_OK(
      loom_value_fact_table_define_layout_strides(&table_, 2, {strides, 2}));
  loom_value_fact_table_t source = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 0));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 2, loom_value_facts_unknown()));
  const loom_value_id_t values[] = {2};
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &table_, {&source, values, 1}, nullptr));
  EXPECT_EQ(loom_value_fact_table_query_layout_strides(&table_, 2).count, 0);
}

}  // namespace
}  // namespace loom
