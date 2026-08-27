// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_projection.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SelectedProjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_bytecode_selected_projection_initialize(iree_allocator_system(),
                                                 &projection_);
  }

  void TearDown() override {
    loom_bytecode_selected_projection_deinitialize(&projection_);
  }

  loom_bytecode_selected_projection_t projection_;
};

TEST_F(SelectedProjectionTest, EmptyProjectionMisses) {
  uint32_t target_id = 0;
  EXPECT_FALSE(loom_bytecode_selected_projection_lookup(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE,
      /*source_ordinal=*/0, &target_id));
}

TEST_F(SelectedProjectionTest, DomainsAreIndependent) {
  IREE_ASSERT_OK(loom_bytecode_selected_projection_insert(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_ENCODING, 7, 11));
  IREE_ASSERT_OK(loom_bytecode_selected_projection_insert(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SOURCE, 7, 13));
  IREE_ASSERT_OK(loom_bytecode_selected_projection_insert(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE, 7, 17));
  IREE_ASSERT_OK(loom_bytecode_selected_projection_insert(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION, 7, 19));
  IREE_ASSERT_OK(loom_bytecode_selected_projection_insert(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SYMBOL_NAME, 7,
      23));

  uint32_t target_id = 0;
  EXPECT_TRUE(loom_bytecode_selected_projection_lookup(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_ENCODING, 7,
      &target_id));
  EXPECT_EQ(target_id, 11u);
  EXPECT_TRUE(loom_bytecode_selected_projection_lookup(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SOURCE, 7,
      &target_id));
  EXPECT_EQ(target_id, 13u);
  EXPECT_TRUE(loom_bytecode_selected_projection_lookup(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE, 7,
      &target_id));
  EXPECT_EQ(target_id, 17u);
  EXPECT_TRUE(loom_bytecode_selected_projection_lookup(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION, 7,
      &target_id));
  EXPECT_EQ(target_id, 19u);
  EXPECT_TRUE(loom_bytecode_selected_projection_lookup(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SYMBOL_NAME, 7,
      &target_id));
  EXPECT_EQ(target_id, 23u);
}

TEST_F(SelectedProjectionTest, RehashPreserves4096ReachedFacts) {
  constexpr uint32_t kCount = 4096;
  for (uint32_t i = 0; i < kCount; ++i) {
    const auto domain =
        static_cast<loom_bytecode_selected_projection_domain_t>(i % 5u);
    IREE_ASSERT_OK(loom_bytecode_selected_projection_insert(
        &projection_, domain, i, kCount - i));
  }

  EXPECT_EQ(projection_.slots.count, kCount);
  EXPECT_GE(projection_.slots.capacity, kCount * 2u);
  for (uint32_t i = 0; i < kCount; ++i) {
    const auto domain =
        static_cast<loom_bytecode_selected_projection_domain_t>(i % 5u);
    uint32_t target_id = 0;
    ASSERT_TRUE(loom_bytecode_selected_projection_lookup(&projection_, domain,
                                                         i, &target_id));
    EXPECT_EQ(target_id, kCount - i);
  }
}

TEST_F(SelectedProjectionTest, SupportsMaximumPackedIdentities) {
  constexpr uint32_t kMaximum = (UINT32_C(1) << 24) - 1;
  IREE_ASSERT_OK(loom_bytecode_selected_projection_insert(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION, kMaximum,
      kMaximum));
  uint32_t target_id = 0;
  EXPECT_TRUE(loom_bytecode_selected_projection_lookup(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION, kMaximum,
      &target_id));
  EXPECT_EQ(target_id, kMaximum);
}

TEST_F(SelectedProjectionTest, IdenticalInsertionIsStable) {
  IREE_ASSERT_OK(loom_bytecode_selected_projection_insert(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE, 23, 42));
  const iree_host_size_t capacity = projection_.slots.capacity;
  IREE_ASSERT_OK(loom_bytecode_selected_projection_insert(
      &projection_, LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE, 23, 42));
  EXPECT_EQ(projection_.slots.count, 1u);
  EXPECT_EQ(projection_.slots.capacity, capacity);
}

}  // namespace
