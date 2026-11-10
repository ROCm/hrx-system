// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
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
  // Storage owning the primary fact table and its recipes.
  iree_arena_allocator_t arena_;
  // Primary scope under test.
  loom_value_fact_table_t table_ = {};
};

TEST_F(FactLayoutTest, AbsentRecipesDoNotAllocate) {
  const auto bytes = arena_.used_allocation_size;
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, 100).kind,
            LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE);
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_explicit_strides(
      &table_, 100, nullptr, 0));
  IREE_ASSERT_OK(loom_value_fact_table_forward_layout_origin(&table_, 1, 100));
  EXPECT_EQ(table_.layout_origins, nullptr);
  EXPECT_EQ(arena_.used_allocation_size, bytes);
}

TEST_F(FactLayoutTest, ExplicitRecipesOwnIdsAndReuseUnchangedStorage) {
  loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID, 2};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_explicit_strides(
      &table_, 3, strides, 3));
  const auto first = loom_value_fact_table_query_layout_origin(&table_, 3);
  ASSERT_EQ(first.kind, LOOM_VALUE_FACT_LAYOUT_ORIGIN_EXPLICIT_STRIDES);
  ASSERT_EQ(first.count, 3);
  EXPECT_NE(first.values.explicit_strides, strides);
  const auto bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_explicit_strides(
      &table_, 3, strides, 3));
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, 3)
                .values.explicit_strides,
            first.values.explicit_strides);
  EXPECT_EQ(arena_.used_allocation_size, bytes);

  strides[0] = 4;
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_explicit_strides(
      &table_, 3, strides, 2));
  const auto second = loom_value_fact_table_query_layout_origin(&table_, 3);
  ASSERT_EQ(second.count, 2);
  EXPECT_EQ(second.values.explicit_strides[0], 4);
  EXPECT_EQ(second.values.explicit_strides[1], LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(first.values.explicit_strides[0], 1);
}

TEST_F(FactLayoutTest, DenseRecipesOwnPackedDimensions) {
  uint64_t dimensions[] = {loom_dim_pack_dynamic(1), loom_dim_pack_static(4),
                           loom_dim_pack_dynamic(2)};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_dense_dimensions(
      &table_, 3, dimensions, 3));
  const auto first = loom_value_fact_table_query_layout_origin(&table_, 3);
  ASSERT_EQ(first.kind, LOOM_VALUE_FACT_LAYOUT_ORIGIN_DENSE_DIMENSIONS);
  ASSERT_EQ(first.count, 3);
  EXPECT_NE(first.values.dense_dimensions, dimensions);
  EXPECT_EQ(first.values.dense_dimensions[0], dimensions[0]);
  EXPECT_EQ(first.values.dense_dimensions[1], dimensions[1]);
  EXPECT_EQ(first.values.dense_dimensions[2], dimensions[2]);

  const auto bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_dense_dimensions(
      &table_, 3, dimensions, 3));
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, 3)
                .values.dense_dimensions,
            first.values.dense_dimensions);
  EXPECT_EQ(arena_.used_allocation_size, bytes);
}

TEST_F(FactLayoutTest, PhysicalEncodingIdentitySurvivesRecipeChanges) {
  constexpr loom_value_id_t encoding = 3;
  constexpr loom_value_id_t physical_encoding = 7;
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_physical_encoding_value(
      &table_, encoding, physical_encoding));
  auto origin = loom_value_fact_table_query_layout_origin(&table_, encoding);
  EXPECT_EQ(origin.kind, LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE);
  EXPECT_EQ(origin.count, 0);
  EXPECT_TRUE(origin.has_physical_encoding_value);
  EXPECT_EQ(origin.physical_encoding_value_id, physical_encoding);

  const loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_explicit_strides(
      &table_, encoding, strides, IREE_ARRAYSIZE(strides)));
  origin = loom_value_fact_table_query_layout_origin(&table_, encoding);
  EXPECT_EQ(origin.kind, LOOM_VALUE_FACT_LAYOUT_ORIGIN_EXPLICIT_STRIDES);
  EXPECT_TRUE(origin.has_physical_encoding_value);
  EXPECT_EQ(origin.physical_encoding_value_id, physical_encoding);

  IREE_ASSERT_OK(loom_value_fact_table_define_layout_explicit_strides(
      &table_, encoding, nullptr, 0));
  origin = loom_value_fact_table_query_layout_origin(&table_, encoding);
  EXPECT_EQ(origin.kind, LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE);
  EXPECT_TRUE(origin.has_physical_encoding_value);
  EXPECT_EQ(origin.physical_encoding_value_id, physical_encoding);

  IREE_ASSERT_OK(loom_value_fact_table_define_layout_physical_encoding_value(
      &table_, encoding, LOOM_VALUE_ID_INVALID));
  origin = loom_value_fact_table_query_layout_origin(&table_, encoding);
  EXPECT_FALSE(origin.has_physical_encoding_value);
}

TEST_F(FactLayoutTest, PhysicalEncodingIdentityForwardsAndCopiesAcrossScopes) {
  constexpr loom_value_id_t encoding = 3;
  constexpr loom_value_id_t forwarded = 100;
  constexpr loom_value_id_t physical_encoding = 0;
  const uint64_t dimensions[] = {loom_dim_pack_dynamic(1),
                                 loom_dim_pack_static(4)};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_dense_dimensions(
      &table_, encoding, dimensions, IREE_ARRAYSIZE(dimensions)));
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_physical_encoding_value(
      &table_, encoding, physical_encoding));
  IREE_ASSERT_OK(loom_value_fact_table_forward_layout_origin(&table_, encoding,
                                                             forwarded));

  const auto source =
      loom_value_fact_table_query_layout_origin(&table_, encoding);
  const auto shared =
      loom_value_fact_table_query_layout_origin(&table_, forwarded);
  EXPECT_TRUE(loom_value_fact_layout_origin_equal(source, shared));
  EXPECT_EQ(source.values.data, shared.values.data);
  EXPECT_TRUE(shared.has_physical_encoding_value);
  EXPECT_EQ(shared.physical_encoding_value_id, physical_encoding);

  loom_value_fact_table_t copied_table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&copied_table, &arena_, 0));
  IREE_ASSERT_OK(
      loom_value_fact_table_clone_origins(&copied_table, &table_, encoding));
  const auto copied =
      loom_value_fact_table_query_layout_origin(&copied_table, encoding);
  EXPECT_TRUE(loom_value_fact_layout_origin_equal(source, copied));
  EXPECT_NE(source.values.data, copied.values.data);

  auto different_identity = copied;
  different_identity.physical_encoding_value_id = 9;
  EXPECT_FALSE(loom_value_fact_layout_origin_equal(source, different_identity));
}

TEST_F(FactLayoutTest, ForwardingSharesAndReplacesRecipes) {
  const loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_explicit_strides(
      &table_, 2, strides, 2));
  IREE_ASSERT_OK(loom_value_fact_table_forward_layout_origin(&table_, 2, 100));
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, 100).values.data,
            loom_value_fact_table_query_layout_origin(&table_, 2).values.data);
  IREE_ASSERT_OK(loom_value_fact_table_forward_layout_origin(&table_, 3, 100));
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, 100).kind,
            LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE);
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, 2).count, 2);
}

TEST_F(FactLayoutTest, RelativeRangeGrowthPreservesBorrowedRecipes) {
  constexpr loom_value_id_t first = 1u << 16;
  const loom_value_id_t strides[] = {first + 1, LOOM_VALUE_ID_INVALID};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_explicit_strides(
      &table_, first + 32, strides, 2));
  const auto borrowed =
      loom_value_fact_table_query_layout_origin(&table_, first + 32);
  IREE_ASSERT_OK(
      loom_value_fact_table_forward_layout_origin(&table_, first + 32, first));
  IREE_ASSERT_OK(
      loom_value_fact_table_forward_layout_origin(&table_, first, first + 120));
  EXPECT_LT(arena_.used_allocation_size, 8192u);
  for (uint32_t offset : {0u, 32u, 120u}) {
    const auto current =
        loom_value_fact_table_query_layout_origin(&table_, first + offset);
    EXPECT_EQ(current.values.data, borrowed.values.data);
    ASSERT_EQ(current.count, 2);
    EXPECT_EQ(current.values.explicit_strides[0], strides[0]);
    EXPECT_EQ(current.values.explicit_strides[1], strides[1]);
  }
  loom_value_fact_table_clear_layout_origin(&table_, first);
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, first).kind,
            LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE);
  EXPECT_EQ(borrowed.values.explicit_strides[0], strides[0]);
  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(
      loom_value_fact_table_query_layout_origin(&table_, first + 120).kind,
      LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE);
}

TEST_F(FactLayoutTest, UndefinitionAndScopeResetWithdrawRecipes) {
  const loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_explicit_strides(
      &table_, 2, strides, 2));
  IREE_ASSERT_OK(loom_value_fact_table_forward_layout_origin(&table_, 2, 3));
  loom_value_fact_table_undefine(&table_, 2);
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, 2).kind,
            LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE);
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, 3).count, 2);
  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(table_.layout_origins, nullptr);
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, 3).kind,
            LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE);
}

TEST_F(FactLayoutTest, NumericTransferDoesNotTransferScopedRecipes) {
  const loom_value_id_t strides[] = {1, LOOM_VALUE_ID_INVALID};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_explicit_strides(
      &table_, 2, strides, 2));
  const loom_value_facts_t numeric_strides[] = {loom_value_facts_make(8, 12, 1),
                                                loom_value_facts_exact_i64(1)};
  loom_value_fact_encoding_summary_t summary = {
      .address_layout =
          {
              .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
              .rank = 2,
              .strides = numeric_strides,
          },
  };
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
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&target, 3).kind,
            LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE);
  EXPECT_EQ(target.layout_origins, nullptr);
}

TEST_F(FactLayoutTest, SameIdSnapshotOwnsDenseRecipeAfterSourceReset) {
  iree_arena_allocator_t source_arena;
  iree_arena_initialize(&block_pool_, &source_arena);
  loom_value_fact_table_t source = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &source_arena, 0));
  const uint64_t dimensions[] = {loom_dim_pack_dynamic(1),
                                 loom_dim_pack_static(4)};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_dense_dimensions(
      &source, 2, dimensions, 2));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 2, loom_value_facts_unknown()));
  const loom_value_id_t values[] = {2};
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &table_, {&source, values, 1}, nullptr));
  EXPECT_NE(loom_value_fact_table_query_layout_origin(&table_, 2).values.data,
            loom_value_fact_table_query_layout_origin(&source, 2).values.data);
  iree_arena_deinitialize(&source_arena);
  const auto copied = loom_value_fact_table_query_layout_origin(&table_, 2);
  ASSERT_EQ(copied.kind, LOOM_VALUE_FACT_LAYOUT_ORIGIN_DENSE_DIMENSIONS);
  ASSERT_EQ(copied.count, 2);
  EXPECT_EQ(copied.values.dense_dimensions[0], dimensions[0]);
  EXPECT_EQ(copied.values.dense_dimensions[1], dimensions[1]);
}

TEST_F(FactLayoutTest, DenseAxisRecipeUsesBoundedSuffixProduct) {
  constexpr loom_value_id_t dynamic_outer = 11;
  constexpr loom_value_id_t dynamic_inner = 12;
  const uint64_t dimensions[] = {
      loom_dim_pack_dynamic(dynamic_outer), loom_dim_pack_static(4),
      loom_dim_pack_dynamic(dynamic_inner), loom_dim_pack_static(8)};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_dense_dimensions(
      &table_, 100, dimensions, 4));
  const auto origin = loom_value_fact_table_query_layout_origin(&table_, 100);

  loom_value_fact_layout_axis_t axis;
  loom_value_fact_layout_query_axis(
      &table_, origin, 0, loom_value_facts_make(0, INT64_MAX, 1), &axis);
  EXPECT_EQ(axis.kind, LOOM_VALUE_FACT_LAYOUT_AXIS_DYNAMIC);
  EXPECT_EQ(axis.static_coefficient, 32);
  ASSERT_EQ(axis.dynamic_factor_count, 1);
  EXPECT_EQ(axis.dynamic_factors[0], dynamic_inner);

  loom_value_fact_layout_query_axis(
      &table_, origin, 2, loom_value_facts_make(0, INT64_MAX, 1), &axis);
  EXPECT_EQ(axis.kind, LOOM_VALUE_FACT_LAYOUT_AXIS_STATIC);
  EXPECT_EQ(axis.static_coefficient, 8);

  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, dynamic_inner,
                                              loom_value_facts_exact_i64(3)));
  loom_value_fact_layout_query_axis(
      &table_, origin, 0, loom_value_facts_make(0, INT64_MAX, 1), &axis);
  EXPECT_EQ(axis.kind, LOOM_VALUE_FACT_LAYOUT_AXIS_STATIC);
  EXPECT_EQ(axis.static_coefficient, 96);
}

TEST_F(FactLayoutTest, ExactFactsNeedNoRetainedRecipe) {
  loom_value_fact_layout_axis_t axis;
  loom_value_fact_layout_query_axis(&table_, {}, 7,
                                    loom_value_facts_exact_i64(24), &axis);
  EXPECT_EQ(axis.kind, LOOM_VALUE_FACT_LAYOUT_AXIS_STATIC);
  EXPECT_EQ(axis.static_coefficient, 24);
  EXPECT_EQ(axis.dynamic_factor_count, 0);
}

TEST_F(FactLayoutTest, ZeroDenseSuffixCancelsDynamicFactors) {
  const uint64_t dimensions[] = {loom_dim_pack_static(4),
                                 loom_dim_pack_dynamic(12),
                                 loom_dim_pack_static(0)};
  IREE_ASSERT_OK(loom_value_fact_table_define_layout_dense_dimensions(
      &table_, 100, dimensions, 3));
  loom_value_fact_layout_axis_t axis;
  loom_value_fact_layout_query_axis(
      &table_, loom_value_fact_table_query_layout_origin(&table_, 100), 0,
      loom_value_facts_make(0, INT64_MAX, 1), &axis);
  EXPECT_EQ(axis.kind, LOOM_VALUE_FACT_LAYOUT_AXIS_STATIC);
  EXPECT_EQ(axis.static_coefficient, 0);
  EXPECT_EQ(axis.dynamic_factor_count, 0);
}

}  // namespace
}  // namespace loom
