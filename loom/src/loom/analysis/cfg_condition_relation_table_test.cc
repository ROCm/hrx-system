// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/cfg_condition_relation_table.h"

#include <array>
#include <cstdint>
#include <initializer_list>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

using MatrixRow = loom_condition_relation_matrix_row_t;
using MatrixView = loom_condition_relation_matrix_view_t;
using SetId = loom_condition_relation_set_id_t;

class CfgConditionRelationTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &set_arena_);
    iree_arena_initialize(&block_pool_, &publication_arena_);
    iree_arena_initialize(&block_pool_, &retained_arena_);
    IREE_ASSERT_OK(loom_condition_relation_set_builder_allocate(
        128, &set_arena_, &set_builder_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&retained_arena_);
    iree_arena_deinitialize(&publication_arena_);
    iree_arena_deinitialize(&set_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  SetId Intern(std::initializer_list<uint32_t> values) {
    SetId set = LOOM_CONDITION_RELATION_SET_EMPTY;
    IREE_CHECK_OK(loom_condition_relation_set_builder_intern(
        set_builder_, values.begin(), values.size(), &set));
    return set;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t set_arena_;
  iree_arena_allocator_t publication_arena_;
  iree_arena_allocator_t retained_arena_;
  loom_condition_relation_set_builder_t* set_builder_ = nullptr;
};

TEST_F(CfgConditionRelationTableTest, InternsEqualPagesAcrossViews) {
  const SetId set_a = Intern({3, 5});
  const SetId set_shared = Intern({7});
  const SetId set_b = Intern({67});
  const SetId set_c = Intern({68});
  std::array<MatrixRow, 4> first_rows = {{
      {1, {set_a, 0, 0}},
      {3, {0, 0, 0}},
      {5, {0, set_shared, 0}},
      {65, {0, set_b, 0}},
  }};
  std::array<MatrixRow, 4> second_rows = {{
      {1, {set_a, 0, 0}},
      {4, {0, 0, 0}},
      {5, {0, set_shared, 0}},
      {66, {0, set_c, 0}},
  }};
  std::array<loom_cfg_condition_relation_table_builder_view_t, 2> views = {};
  views[0].integer_relations = {first_rows.data(),
                                static_cast<uint32_t>(first_rows.size())};
  views[1].integer_relations = {second_rows.data(),
                                static_cast<uint32_t>(second_rows.size())};
  const loom_cfg_condition_operand_domain_t operand_domain = {};
  loom_cfg_condition_relation_table_builder_t builder = {
      .operand_domain = &operand_domain,
      .set_builder = set_builder_,
      .views = views.data(),
      .view_count = static_cast<uint32_t>(views.size()),
      .block_count = static_cast<uint32_t>(views.size()),
  };
  loom_cfg_condition_relation_table_t table = {};
  IREE_ASSERT_OK(loom_cfg_condition_relation_table_publish(
      &builder, &table, &publication_arena_, &retained_arena_));

  ASSERT_EQ(table.view_count, 2u);
  const MatrixView& first = table.views[0].integer_relations;
  const MatrixView& second = table.views[1].integer_relations;
  ASSERT_EQ(first.encoding, LOOM_CONDITION_RELATION_MATRIX_VIEW_PAGES);
  ASSERT_EQ(second.encoding, LOOM_CONDITION_RELATION_MATRIX_VIEW_PAGES);
  ASSERT_EQ(first.entry_count, 2u);
  ASSERT_EQ(second.entry_count, 2u);
  EXPECT_EQ(first.entries.pages[0], second.entries.pages[0]);
  EXPECT_NE(first.entries.pages[1], second.entries.pages[1]);
  EXPECT_NE(loom_condition_relation_matrix_view_find(&first, 1), nullptr);
  EXPECT_EQ(loom_condition_relation_matrix_view_find(&first, 3), nullptr);
  EXPECT_NE(loom_condition_relation_matrix_view_find(&first, 5), nullptr);
  EXPECT_NE(loom_condition_relation_matrix_view_find(&first, 65), nullptr);
  EXPECT_EQ(loom_condition_relation_matrix_view_find(&first, 66), nullptr);
  EXPECT_EQ(loom_condition_relation_matrix_view_find(&second, 65), nullptr);
  EXPECT_NE(loom_condition_relation_matrix_view_find(&second, 66), nullptr);
}

}  // namespace
}  // namespace loom
