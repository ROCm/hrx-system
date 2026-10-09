// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/clobber_index.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptors.h"

namespace loom {
namespace {

class LowAllocationClobberIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    iree_arena_initialize(&block_pool_, &result_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&result_arena_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t scratch_arena_;
  iree_arena_allocator_t result_arena_;
};

TEST_F(LowAllocationClobberIndexTest,
       GroupsProgramOrderedEventsBySharedStorage) {
  loom_low_reg_class_t reg_classes[5] = {};
  reg_classes[1].alias_set_id = 1;
  reg_classes[2].alias_set_id = 1;
  reg_classes[3].flags = LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
  reg_classes[4].flags = LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
  loom_low_descriptor_set_t descriptor_set = {
      .reg_classes = reg_classes,
      .reg_class_count = IREE_ARRAYSIZE(reg_classes)};

  loom_low_allocation_clobber_builder_t builder;
  loom_low_allocation_clobber_builder_initialize(&scratch_arena_, &builder);
  IREE_ASSERT_OK(loom_low_allocation_clobber_builder_record(
      &builder, loom_low_reg_class_storage_key(&descriptor_set, 0), 3, 2,
      /*permits_definition=*/false));
  IREE_ASSERT_OK(loom_low_allocation_clobber_builder_record(
      &builder, loom_low_reg_class_storage_key(&descriptor_set, 1), 1, 3,
      /*permits_definition=*/false));
  IREE_ASSERT_OK(loom_low_allocation_clobber_builder_record(
      &builder, loom_low_reg_class_storage_key(&descriptor_set, 0), 3, 5,
      /*permits_definition=*/true));
  IREE_ASSERT_OK(loom_low_allocation_clobber_builder_record(
      &builder, loom_low_reg_class_storage_key(&descriptor_set, 3), 7, 6,
      /*permits_definition=*/false));
  IREE_ASSERT_OK(loom_low_allocation_clobber_builder_record(
      &builder, loom_low_reg_class_storage_key(&descriptor_set, 2), 1, 8,
      /*permits_definition=*/true));
  IREE_ASSERT_OK(loom_low_allocation_clobber_builder_record(
      &builder, loom_low_reg_class_storage_key(&descriptor_set, 4), 7, 9,
      /*permits_definition=*/false));

  loom_low_allocation_clobber_index_t index;
  IREE_ASSERT_OK(loom_low_allocation_clobber_builder_build(
      &builder, &descriptor_set, &result_arena_, &index));
  iree_arena_reset(&scratch_arena_);

  EXPECT_EQ(index.permitted_definitions.bit_count, 6u);
  ASSERT_NE(index.permitted_definitions.words, nullptr);

  const auto unique_range =
      loom_low_allocation_clobber_index_range(&index, 0, 3);
  ASSERT_EQ(unique_range.end - unique_range.start, 2u);
  EXPECT_EQ(index.points[unique_range.start], 2u);
  EXPECT_EQ(index.points[unique_range.start + 1u], 5u);
  EXPECT_FALSE(
      iree_bitmap_test(index.permitted_definitions, unique_range.start));
  EXPECT_TRUE(
      iree_bitmap_test(index.permitted_definitions, unique_range.start + 1u));
  const auto empty_range =
      loom_low_allocation_clobber_index_range(&index, 0, 2);
  EXPECT_EQ(empty_range.start, empty_range.end);

  const auto alias_range =
      loom_low_allocation_clobber_index_range(&index, 1, 1);
  const auto other_alias_range =
      loom_low_allocation_clobber_index_range(&index, 2, 1);
  EXPECT_EQ(alias_range.start, other_alias_range.start);
  EXPECT_EQ(alias_range.end, other_alias_range.end);
  ASSERT_EQ(alias_range.end - alias_range.start, 2u);
  EXPECT_EQ(index.points[alias_range.start], 3u);
  EXPECT_EQ(index.points[alias_range.start + 1u], 8u);

  const auto explicit_range =
      loom_low_allocation_clobber_index_range(&index, 3, 7);
  const auto other_explicit_range =
      loom_low_allocation_clobber_index_range(&index, 4, 7);
  EXPECT_EQ(explicit_range.start, other_explicit_range.start);
  EXPECT_EQ(explicit_range.end, other_explicit_range.end);
  ASSERT_EQ(explicit_range.end - explicit_range.start, 2u);
  EXPECT_EQ(index.points[explicit_range.start], 6u);
  EXPECT_EQ(index.points[explicit_range.start + 1u], 9u);
}

TEST_F(LowAllocationClobberIndexTest, OmitsUnusedDefinitionBitmap) {
  loom_low_reg_class_t reg_class = {};
  loom_low_descriptor_set_t descriptor_set = {.reg_classes = &reg_class,
                                              .reg_class_count = 1};

  loom_low_allocation_clobber_builder_t builder;
  loom_low_allocation_clobber_builder_initialize(&scratch_arena_, &builder);
  IREE_ASSERT_OK(loom_low_allocation_clobber_builder_record(
      &builder, loom_low_reg_class_storage_key(&descriptor_set, 0), 0, 3,
      /*permits_definition=*/false));
  loom_low_allocation_clobber_index_t index;
  IREE_ASSERT_OK(loom_low_allocation_clobber_builder_build(
      &builder, &descriptor_set, &result_arena_, &index));

  EXPECT_EQ(index.permitted_definitions.bit_count, 1u);
  EXPECT_EQ(index.permitted_definitions.words, nullptr);
  const auto range = loom_low_allocation_clobber_index_range(&index, 0, 0);
  ASSERT_EQ(range.end - range.start, 1u);
  EXPECT_EQ(index.points[range.start], 3u);
}

}  // namespace
}  // namespace loom
