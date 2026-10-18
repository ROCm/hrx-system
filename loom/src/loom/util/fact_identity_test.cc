// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_identity.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class FactIdentityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &transient_arena_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize_with_arenas(
        &table_, &arena_, &transient_arena_, 0));
  }
  void TearDown() override {
    iree_arena_deinitialize(&transient_arena_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  bool Publish(loom_value_id_t result, loom_value_id_t source) {
    IREE_CHECK_OK(loom_value_fact_table_define(&table_, result,
                                               loom_value_facts_unknown()));
    bool changed = false;
    IREE_CHECK_OK(loom_value_fact_table_define_identity(&table_, result, source,
                                                        &changed));
    return changed;
  }
  loom_value_id_t Identity(loom_value_id_t value) const {
    return loom_value_fact_table_query_identity(&table_, value);
  }
  // Shared backing for persistent entries and transient publications.
  iree_arena_block_pool_t pool_ = {};
  // Direct entry and touched-value storage, retained across scopes.
  iree_arena_allocator_t arena_ = {};
  // Publication storage, reset only after clearing the fact scope.
  iree_arena_allocator_t transient_arena_ = {};
  // Owner of inferred facts and shared identity publications.
  loom_value_fact_table_t table_ = {};
};

TEST_F(FactIdentityTest, ReplacementKeepsOldValueAvailableForNewUsers) {
  EXPECT_TRUE(Publish(10, 1));
  EXPECT_TRUE(Publish(11, 10));
  IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(&table_, 1, 2));
  EXPECT_EQ(Identity(10), 2u);
  EXPECT_EQ(Identity(11), 2u);
  EXPECT_EQ(Identity(1), 1u);
  EXPECT_TRUE(Publish(12, 1));
  EXPECT_EQ(Identity(12), 1u);
  IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(&table_, 1, 3));
  EXPECT_EQ(Identity(12), 3u);
  EXPECT_EQ(Identity(10), 2u);
  EXPECT_EQ(Identity(11), 2u);
}

TEST_F(FactIdentityTest, ReplacementDoesNotConsumeDeferredChangeReports) {
  EXPECT_TRUE(Publish(10, 1));
  EXPECT_TRUE(Publish(11, 10));
  IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(&table_, 1, 2));
  EXPECT_EQ(Identity(11), 2u);
  EXPECT_TRUE(Publish(10, 2));
  EXPECT_TRUE(Publish(11, 10));
  EXPECT_FALSE(Publish(10, 2));
  EXPECT_FALSE(Publish(11, 10));
}

TEST_F(FactIdentityTest, ReplacingAnAliasDoesNotRedirectItsIndependentSource) {
  EXPECT_TRUE(Publish(10, 1));
  EXPECT_TRUE(Publish(11, 1));
  IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(&table_, 10, 2));
  EXPECT_EQ(Identity(1), 1u);
  EXPECT_EQ(Identity(11), 1u);
  loom_value_fact_table_erase_identity_value(&table_, 10);
  EXPECT_EQ(Identity(11), 1u);
}

TEST_F(FactIdentityTest, MergesExistingClassesWithoutLosingTheirUsers) {
  EXPECT_TRUE(Publish(10, 1));
  EXPECT_TRUE(Publish(11, 2));
  EXPECT_TRUE(Publish(12, 3));
  IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(&table_, 1, 2));
  IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(&table_, 2, 3));
  for (loom_value_id_t value : {10u, 11u, 12u}) {
    EXPECT_EQ(Identity(value), 3u);
  }
  EXPECT_TRUE(Publish(13, 2));
  EXPECT_EQ(Identity(13), 2u);
  EXPECT_TRUE(Publish(10, 3));
  EXPECT_TRUE(Publish(11, 3));
  EXPECT_FALSE(Publish(12, 3));
}

TEST_F(FactIdentityTest, InferenceRetargetsAnOwnRepresentative) {
  EXPECT_TRUE(Publish(10, 1));
  EXPECT_TRUE(Publish(1, 2));
  EXPECT_EQ(Identity(10), 2u);
  loom_value_fact_table_erase_identity_value(&table_, 1);
  EXPECT_EQ(Identity(10), 2u);
  IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(&table_, 2, 3));
  EXPECT_EQ(Identity(10), 3u);
}

TEST_F(FactIdentityTest, CyclicUndefinitionAndWithdrawalKeepDistinctLifetimes) {
  EXPECT_TRUE(Publish(10, 1));
  loom_value_fact_table_undefine(&table_, 1);
  EXPECT_EQ(Identity(10), 1u);
  loom_value_fact_table_undefine(&table_, 10);
  EXPECT_EQ(Identity(10), 10u);
  EXPECT_TRUE(Publish(10, 1));
  loom_value_fact_table_erase_identity_value(&table_, 1);
  EXPECT_EQ(Identity(10), 10u);
  EXPECT_TRUE(Publish(11, 10));
  EXPECT_EQ(Identity(11), 10u);
  EXPECT_TRUE(Publish(10, 2));
  EXPECT_EQ(Identity(11), 2u);
}

TEST_F(FactIdentityTest, ClearsUnseededRepresentativesAndReusesEntryStorage) {
  EXPECT_TRUE(Publish(10, 1));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&table_, 1));
  IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(&table_, 1, 2));
  EXPECT_TRUE(Publish(11, 1));
  const auto* entries = table_.identities.entries;
  loom_value_fact_table_clear_scope(&table_);
  iree_arena_reset(&transient_arena_);
  EXPECT_EQ(table_.identities.entries, entries);
  EXPECT_EQ(table_.identities.nodes, nullptr);
  for (loom_value_id_t value : {1u, 2u, 10u, 11u}) {
    EXPECT_EQ(Identity(value), value);
    EXPECT_EQ(table_.identities.entries[value], nullptr);
  }
  EXPECT_TRUE(Publish(10, 2));
  EXPECT_EQ(Identity(10), 2u);
}

TEST_F(FactIdentityTest, ClonesOwnIndependentPublicationStorage) {
  EXPECT_TRUE(Publish(10, 1));
  EXPECT_TRUE(Publish(11, 10));
  loom_value_fact_table_t clone = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &arena_, 0));
  const loom_value_id_t selected[] = {10, 11};
  const loom_value_fact_table_view_t view = {&table_, selected,
                                             IREE_ARRAYSIZE(selected)};
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(&clone, view, nullptr));
  EXPECT_NE(clone.identities.entries[10], table_.identities.entries[10]);
  IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(&table_, 1, 2));
  EXPECT_EQ(Identity(10), 2u);
  EXPECT_EQ(loom_value_fact_table_query_identity(&clone, 10), 1u);
  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(loom_value_fact_table_query_identity(&clone, 11), 1u);
  IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(&clone, 1, 3));
  EXPECT_EQ(loom_value_fact_table_query_identity(&clone, 10), 3u);
}

TEST_F(FactIdentityTest, CloningSelfIdentityRemovesPreviousAlias) {
  EXPECT_TRUE(Publish(10, 1));
  loom_value_fact_table_t source = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 0));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 10, loom_value_facts_unknown()));
  const loom_value_id_t selected[] = {10};
  const loom_value_fact_table_view_t view = {&source, selected, 1};
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(&table_, view, nullptr));
  EXPECT_EQ(Identity(10), 10u);
  EXPECT_TRUE(Publish(11, 10));
  EXPECT_EQ(Identity(11), 10u);
}

TEST_F(FactIdentityTest, SerialReplacementsKeepLinearStorageAndBoundedDepth) {
  constexpr uint32_t count = 1024;
  for (uint32_t i = 0; i < count; ++i) {
    EXPECT_TRUE(Publish(count + i, 0));
  }
  for (uint32_t i = 0; i + 1 < count; ++i) {
    IREE_ASSERT_OK(
        loom_value_fact_table_replace_identity_value(&table_, i, i + 1));
  }
  uint32_t nodes = 0;
  for (auto* node = table_.identities.nodes; node; node = node->next) {
    ++nodes;
    uint32_t depth = 0;
    for (auto* parent = node->parent; parent; parent = parent->parent) {
      ++depth;
    }
    EXPECT_LE(depth, 1u);
  }
  EXPECT_EQ(nodes, count);
  for (uint32_t i = 0; i < count; ++i) {
    EXPECT_EQ(Identity(count + i), count - 1);
  }
}

TEST_F(FactIdentityTest, BalancedClassMergesHaveLogarithmicDepth) {
  constexpr uint32_t count = 1024;
  for (uint32_t i = 0; i < count; ++i) {
    EXPECT_TRUE(Publish(count + i, i));
  }
  for (uint32_t width = 2; width <= count; width *= 2) {
    for (uint32_t begin = 0; begin < count; begin += width) {
      IREE_ASSERT_OK(loom_value_fact_table_replace_identity_value(
          &table_, begin + width / 2 - 1, begin + width - 1));
    }
  }
  for (uint32_t i = 0; i < count; ++i) {
    EXPECT_EQ(Identity(count + i), count - 1);
    uint32_t depth = 0;
    for (auto* node = table_.identities.entries[count + i]; node->parent;
         node = node->parent) {
      ++depth;
    }
    EXPECT_LE(depth, 10u);
  }
}

}  // namespace
}  // namespace loom
