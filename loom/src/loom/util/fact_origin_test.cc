// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_origin.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

static iree_status_t CloneScopeValues(loom_value_fact_table_t* target,
                                      const loom_value_fact_table_t* source) {
  std::vector<loom_value_id_t> values;
  loom_value_fact_scope_iterator_t iterator;
  loom_value_fact_table_scope_begin(source, &iterator);
  for (loom_value_id_t value_id = loom_value_fact_scope_next(&iterator);
       value_id != LOOM_VALUE_ID_INVALID;
       value_id = loom_value_fact_scope_next(&iterator)) {
    values.push_back(value_id);
  }
  return loom_value_fact_table_clone_values(
      target, {source, values.data(), values.size()}, nullptr);
}

class FactOriginStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Backing pool shared by independently owned fact tables.
  iree_arena_block_pool_t block_pool_;
  // Persistent storage used for table entries and standalone clones.
  iree_arena_allocator_t arena_;
};

// Inspects the stored record independently of module/type validation in
// queries.
template <typename Origins>
static const auto& OriginEntry(const Origins& origins,
                               loom_value_id_t value_id) {
  return origins.entries[loom_value_fact_range_offset(origins.range, value_id)];
}

static loom_value_fact_uniform_scale_origin_t UniformScaleOrigin(
    loom_value_id_t source_value_id, loom_value_id_t scale_value_id) {
  loom_value_fact_uniform_scale_origin_t origin = {
      .source_value_id = source_value_id,
      .scale_value_id = scale_value_id,
  };
  return origin;
}

static loom_value_fact_exact_lane_origin_t ExactLaneOrigin(
    loom_value_id_t source_value_id, uint32_t source_lane_offset = 0,
    uint32_t source_lane_stride = 1) {
  return {
      .source_value_id = source_value_id,
      .source_lane_offset = source_lane_offset,
      .source_lane_stride = source_lane_stride,
  };
}

static loom_value_fact_contextual_query_origin_t ContextualQueryOrigin(
    loom_parameterized_attr_kind_t family_kind, loom_attribute_t key) {
  loom_value_fact_contextual_query_origin_t origin = {
      .family_kind = family_kind,
      .key = key,
  };
  return origin;
}

TEST_F(FactOriginStorageTest, UniformElementOriginsClearOnlyTouchedEntries) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  IREE_ASSERT_OK(
      loom_value_fact_table_define_uniform_element_origin(&table, 5, 2, 1));
  IREE_ASSERT_OK(
      loom_value_fact_table_define_uniform_element_origin(&table, 8, 3, 3));
  IREE_ASSERT_OK(
      loom_value_fact_table_define_uniform_element_origin(&table, 8, 4, 2));

  EXPECT_TRUE(
      loom_value_fact_range_contains(table.uniform_element_origins.range, 8));
  EXPECT_EQ(table.touched_count, 2u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).scalar_value_id, 2u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).exact_scalar_value_id,
            1u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 8).scalar_value_id, 4u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 8).exact_scalar_value_id,
            2u);

  loom_value_fact_uniform_element_origin_t* const entries =
      table.uniform_element_origins.entries;
  loom_value_fact_table_clear_scope(&table);

  EXPECT_EQ(table.uniform_element_origins.entries, entries);
  EXPECT_EQ(table.touched_count, 0u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).scalar_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).exact_scalar_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 8).scalar_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 8).exact_scalar_value_id,
            LOOM_VALUE_ID_INVALID);

  IREE_ASSERT_OK(
      loom_value_fact_table_define_uniform_element_origin(&table, 5, 7, 6));
  EXPECT_EQ(table.touched_count, 1u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).scalar_value_id, 7u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).exact_scalar_value_id,
            6u);
}

TEST_F(FactOriginStorageTest, UniformScaleOriginsClearOnlyTouchedEntries) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
      &table, 5,
      UniformScaleOrigin(/*source_value_id=*/2,
                         /*scale_value_id=*/3)));
  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
      &table, 8,
      UniformScaleOrigin(/*source_value_id=*/4,
                         /*scale_value_id=*/5)));
  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
      &table, 8,
      UniformScaleOrigin(/*source_value_id=*/6,
                         /*scale_value_id=*/7)));

  EXPECT_TRUE(
      loom_value_fact_range_contains(table.uniform_scale_origins.range, 8));
  EXPECT_EQ(table.touched_count, 2u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).source_value_id, 2u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).scale_value_id, 3u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 8).source_value_id, 6u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 8).scale_value_id, 7u);

  loom_value_fact_uniform_scale_origin_t* const entries =
      table.uniform_scale_origins.entries;
  loom_value_fact_table_clear_scope(&table);

  EXPECT_EQ(table.uniform_scale_origins.entries, entries);
  EXPECT_EQ(table.touched_count, 0u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).source_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).scale_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 8).source_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 8).scale_value_id,
            LOOM_VALUE_ID_INVALID);

  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
      &table, 5,
      UniformScaleOrigin(/*source_value_id=*/9,
                         /*scale_value_id=*/10)));
  EXPECT_EQ(table.touched_count, 1u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).source_value_id, 9u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).scale_value_id, 10u);
}

TEST_F(FactOriginStorageTest, ExactLaneOriginsAreSparseAndScopeLocal) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 1024));

  EXPECT_EQ(table.exact_lane_origins.entries, nullptr);
  EXPECT_EQ(table.exact_lane_origins.count, 0u);
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &table, 900,
      ExactLaneOrigin(/*source_value_id=*/12,
                      /*source_lane_offset=*/3,
                      /*source_lane_stride=*/2)));
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &table, 900, ExactLaneOrigin(/*source_value_id=*/20)));
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &table, 7, ExactLaneOrigin(/*source_value_id=*/3)));

  EXPECT_EQ(table.exact_lane_origins.count, 2u);
  EXPECT_LT(table.exact_lane_origins.capacity, (iree_host_size_t)900);

  loom_value_fact_exact_lane_origin_entry_t* const entries =
      table.exact_lane_origins.entries;
  const iree_host_size_t capacity = table.exact_lane_origins.capacity;
  loom_value_fact_table_clear_scope(&table);

  EXPECT_EQ(table.exact_lane_origins.entries, entries);
  EXPECT_EQ(table.exact_lane_origins.capacity, capacity);
  EXPECT_EQ(table.exact_lane_origins.count, 0u);

  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &table, 5, ExactLaneOrigin(/*source_value_id=*/2)));
  EXPECT_EQ(table.exact_lane_origins.entries, entries);
  EXPECT_EQ(table.exact_lane_origins.count, 1u);
}

TEST_F(FactOriginStorageTest, ExactScalarOriginsAreDirectAndScopeLocal) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 1024));

  EXPECT_EQ(table.exact_scalar_origins.entries, nullptr);
  EXPECT_EQ(table.exact_scalar_origins.range.capacity, 0u);
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_scalar_origin(
      &table, 900, {.source_value_id = 12}));
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_scalar_origin(
      &table, 900, {.source_value_id = 20}));
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_scalar_origin(
      &table, 7, {.source_value_id = 3}));

  EXPECT_TRUE(
      loom_value_fact_range_contains(table.exact_scalar_origins.range, 900));
  EXPECT_TRUE(
      loom_value_fact_range_contains(table.exact_scalar_origins.range, 7));
  EXPECT_EQ(table.touched_count, 2u);
  EXPECT_EQ(table.exact_scalar_origins.entries[loom_value_fact_range_offset(
                table.exact_scalar_origins.range, 900)],
            20u);
  EXPECT_EQ(table.exact_scalar_origins.entries[loom_value_fact_range_offset(
                table.exact_scalar_origins.range, 7)],
            3u);

  loom_value_id_t* const entries = table.exact_scalar_origins.entries;
  const loom_value_fact_range_t range = table.exact_scalar_origins.range;
  loom_value_fact_table_clear_scope(&table);

  EXPECT_EQ(table.exact_scalar_origins.entries, entries);
  EXPECT_EQ(table.exact_scalar_origins.range.first_value_id,
            range.first_value_id);
  EXPECT_EQ(table.exact_scalar_origins.range.capacity, range.capacity);
  EXPECT_EQ(table.touched_count, 0u);
  EXPECT_EQ(table.exact_scalar_origins.entries[loom_value_fact_range_offset(
                table.exact_scalar_origins.range, 900)],
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(table.exact_scalar_origins.entries[loom_value_fact_range_offset(
                table.exact_scalar_origins.range, 7)],
            LOOM_VALUE_ID_INVALID);

  IREE_ASSERT_OK(loom_value_fact_table_define_exact_scalar_origin(
      &table, 7, {.source_value_id = 2}));
  EXPECT_EQ(table.exact_scalar_origins.entries, entries);
  EXPECT_EQ(table.exact_scalar_origins.entries[loom_value_fact_range_offset(
                table.exact_scalar_origins.range, 7)],
            2u);
}

TEST_F(FactOriginStorageTest, ContextualQueryOriginsAreSparseAndScopeLocal) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 1024));

  EXPECT_EQ(table.contextual_query_origins.entries, nullptr);
  EXPECT_EQ(table.contextual_query_origins.origins, nullptr);
  const loom_parameterized_attr_kind_t family_kind =
      LOOM_PARAMETERIZED_ATTR_KIND(/*dialect_id=*/3, /*family_index=*/7);
  IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
      &table, 900, ContextualQueryOrigin(family_kind, loom_attr_enum(4))));

  ASSERT_TRUE(loom_value_fact_range_contains(
      table.contextual_query_origins.range, 900));
  EXPECT_EQ(table.contextual_query_origins.touched_count, 1u);
  EXPECT_EQ(table.contextual_query_origins.origin_count, 1u);
  const uint32_t origin_id = OriginEntry(table.contextual_query_origins, 900);
  ASSERT_NE(origin_id, 0u);
  const loom_value_fact_contextual_query_origin_t& origin =
      table.contextual_query_origins.origins[origin_id - 1];
  EXPECT_EQ(origin.family_kind, family_kind);
  EXPECT_EQ(loom_attr_as_enum(origin.key), 4u);

  uint32_t* const entries = table.contextual_query_origins.entries;
  loom_value_fact_contextual_query_origin_t* const origins =
      table.contextual_query_origins.origins;
  loom_value_fact_table_clear_scope(&table);

  EXPECT_EQ(table.contextual_query_origins.entries, entries);
  EXPECT_EQ(table.contextual_query_origins.origins, origins);
  EXPECT_EQ(OriginEntry(table.contextual_query_origins, 900), 0u);
  EXPECT_EQ(table.contextual_query_origins.touched_count, 0u);
  EXPECT_EQ(table.contextual_query_origins.origin_count, 0u);

  IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
      &table, 5, ContextualQueryOrigin(family_kind, loom_attr_enum(9))));
  EXPECT_EQ(table.contextual_query_origins.touched_count, 1u);
  EXPECT_EQ(table.contextual_query_origins.origin_count, 1u);
  const uint32_t reused_origin_id =
      OriginEntry(table.contextual_query_origins, 5);
  ASSERT_NE(reused_origin_id, 0u);
  EXPECT_EQ(
      loom_attr_as_enum(
          table.contextual_query_origins.origins[reused_origin_id - 1].key),
      9u);
}

TEST_F(FactOriginStorageTest,
       OriginRangesPreserveClonesAndReuseClearedStorage) {
  constexpr loom_value_id_t first = 1u << 16;
  const loom_value_id_t values[] = {first + 32, first + 7, first + 40, first};
  const auto family =
      LOOM_PARAMETERIZED_ATTR_KIND(/*dialect_id=*/3, /*family_index=*/7);
  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  for (loom_value_id_t value : values) {
    IREE_ASSERT_OK(loom_value_fact_table_define(&table, value,
                                                loom_value_facts_unknown()));
    IREE_ASSERT_OK(loom_value_fact_table_define_uniform_element_origin(
        &table, value, first + 1, first + 2));
    IREE_ASSERT_OK(loom_value_fact_table_define_static_lane_origin(
        &table, value, {first + 3, value - first, 2}));
    IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
        &table, value, UniformScaleOrigin(first + 4, first + 5)));
    IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
        &table, value,
        ContextualQueryOrigin(
            family, loom_attr_enum(static_cast<uint8_t>(value - first)))));
  }
  const auto* uniform_entries = table.uniform_element_origins.entries;
  const auto* lane_entries = table.static_lane_origins.entries;
  const auto* scale_entries = table.uniform_scale_origins.entries;
  const auto* query_entries = table.contextual_query_origins.entries;
  loom_value_fact_table_t clone = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &clone, {&table, values, IREE_ARRAYSIZE(values)}, nullptr));

  auto expect_origins = [&](const loom_value_fact_table_t& current) {
    for (auto range : {current.uniform_element_origins.range,
                       current.static_lane_origins.range,
                       current.uniform_scale_origins.range,
                       current.contextual_query_origins.range}) {
      EXPECT_LT(range.capacity, 128u);
    }
    for (loom_value_id_t value : values) {
      const auto uniform = OriginEntry(current.uniform_element_origins, value);
      EXPECT_EQ(uniform.scalar_value_id, first + 1);
      EXPECT_EQ(uniform.exact_scalar_value_id, first + 2);
      const auto lane = OriginEntry(current.static_lane_origins, value);
      EXPECT_EQ(lane.source_value_id, first + 3);
      EXPECT_EQ(lane.source_lane_offset, value - first);
      EXPECT_EQ(lane.source_lane_stride, 2u);
      const auto scale = OriginEntry(current.uniform_scale_origins, value);
      EXPECT_EQ(scale.source_value_id, first + 4);
      EXPECT_EQ(scale.scale_value_id, first + 5);
      const auto query = OriginEntry(current.contextual_query_origins, value);
      ASSERT_NE(query, 0u);
      const auto origin = current.contextual_query_origins.origins[query - 1];
      EXPECT_EQ(origin.family_kind, family);
      EXPECT_EQ(loom_attr_as_enum(origin.key), value - first);
    }
  };
  expect_origins(table);
  expect_origins(clone);
  loom_value_fact_table_clear_scope(&table);
  expect_origins(clone);
  for (loom_value_id_t value : values) {
    EXPECT_EQ(OriginEntry(table.uniform_element_origins, value).scalar_value_id,
              LOOM_VALUE_ID_INVALID);
    EXPECT_EQ(OriginEntry(table.static_lane_origins, value).source_value_id,
              LOOM_VALUE_ID_INVALID);
    EXPECT_EQ(OriginEntry(table.uniform_scale_origins, value).source_value_id,
              LOOM_VALUE_ID_INVALID);
    EXPECT_EQ(OriginEntry(table.contextual_query_origins, value), 0u);
  }
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &table, {&clone, values, IREE_ARRAYSIZE(values)}, nullptr));
  expect_origins(table);
  EXPECT_EQ(table.uniform_element_origins.entries, uniform_entries);
  EXPECT_EQ(table.static_lane_origins.entries, lane_entries);
  EXPECT_EQ(table.uniform_scale_origins.entries, scale_entries);
  EXPECT_EQ(table.contextual_query_origins.entries, query_entries);
}

TEST_F(FactOriginStorageTest, CloneValuesCopiesUniformElementOrigins) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  loom_value_facts_t source_facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &source.context, loom_value_facts_exact_i64(42), &source_facts));
  IREE_ASSERT_OK(loom_value_fact_table_define(&source, 7, source_facts));
  IREE_ASSERT_OK(
      loom_value_fact_table_define_uniform_element_origin(&source, 7, 2, 1));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 8));
  IREE_ASSERT_OK(CloneScopeValues(&target, &source));

  EXPECT_TRUE(loom_value_facts_query_uniform_element(
      &target.context, loom_value_fact_table_lookup(&target, 7), nullptr));
  ASSERT_TRUE(
      loom_value_fact_range_contains(target.uniform_element_origins.range, 7));
  EXPECT_EQ(OriginEntry(target.uniform_element_origins, 7).scalar_value_id, 2u);
  EXPECT_EQ(
      OriginEntry(target.uniform_element_origins, 7).exact_scalar_value_id, 1u);

  loom_value_fact_table_clear_origins(&source, 7);
  IREE_ASSERT_OK(CloneScopeValues(&target, &source));
  EXPECT_EQ(OriginEntry(target.uniform_element_origins, 7).scalar_value_id,
            LOOM_VALUE_ID_INVALID);

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactOriginStorageTest, CloneValuesCopiesUniformScaleOrigins) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 7, loom_value_facts_unknown()));
  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
      &source, 7,
      UniformScaleOrigin(/*source_value_id=*/2,
                         /*scale_value_id=*/3)));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 8));
  IREE_ASSERT_OK(CloneScopeValues(&target, &source));

  ASSERT_TRUE(
      loom_value_fact_range_contains(target.uniform_scale_origins.range, 7));
  EXPECT_EQ(OriginEntry(target.uniform_scale_origins, 7).source_value_id, 2u);
  EXPECT_EQ(OriginEntry(target.uniform_scale_origins, 7).scale_value_id, 3u);

  loom_value_fact_table_clear_origins(&source, 7);
  IREE_ASSERT_OK(CloneScopeValues(&target, &source));
  EXPECT_EQ(OriginEntry(target.uniform_scale_origins, 7).source_value_id,
            LOOM_VALUE_ID_INVALID);

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactOriginStorageTest, CloneValuesCopiesExactLaneOrigins) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 7, loom_value_facts_unknown()));
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &source, 7,
      ExactLaneOrigin(/*source_value_id=*/2,
                      /*source_lane_offset=*/1,
                      /*source_lane_stride=*/2)));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 8));
  IREE_ASSERT_OK(CloneScopeValues(&target, &source));

  EXPECT_EQ(target.exact_lane_origins.count, 1u);
  EXPECT_LT(target.exact_lane_origins.capacity, (iree_host_size_t)8);

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactOriginStorageTest, CloneValuesCopiesExactScalarOrigins) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 7, loom_value_facts_unknown()));
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_scalar_origin(
      &source, 7, {.source_value_id = 2}));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 8));
  IREE_ASSERT_OK(CloneScopeValues(&target, &source));

  ASSERT_TRUE(
      loom_value_fact_range_contains(target.exact_scalar_origins.range, 7));
  EXPECT_EQ(target.exact_scalar_origins.entries[loom_value_fact_range_offset(
                target.exact_scalar_origins.range, 7)],
            2u);

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactOriginStorageTest, CloneValuesCopiesContextualQueryOrigins) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 7, loom_value_facts_unknown()));
  const loom_parameterized_attr_kind_t family_kind =
      LOOM_PARAMETERIZED_ATTR_KIND(/*dialect_id=*/3, /*family_index=*/7);
  IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
      &source, 7, ContextualQueryOrigin(family_kind, loom_attr_i64(32))));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 8));
  IREE_ASSERT_OK(CloneScopeValues(&target, &source));

  ASSERT_TRUE(
      loom_value_fact_range_contains(target.contextual_query_origins.range, 7));
  const uint32_t origin_id = OriginEntry(target.contextual_query_origins, 7);
  ASSERT_NE(origin_id, 0u);
  const loom_value_fact_contextual_query_origin_t& origin =
      target.contextual_query_origins.origins[origin_id - 1];
  EXPECT_EQ(origin.family_kind, family_kind);
  EXPECT_EQ(loom_attr_as_i64(origin.key), 32);

  const auto bytes = target_arena.used_allocation_size;
  IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
      &source, 7, ContextualQueryOrigin(family_kind, loom_attr_i64(64))));
  IREE_ASSERT_OK(CloneScopeValues(&target, &source));
  EXPECT_EQ(loom_attr_as_i64(origin.key), 64);
  loom_value_fact_table_clear_origins(&source, 7);
  IREE_ASSERT_OK(CloneScopeValues(&target, &source));
  EXPECT_EQ(origin.family_kind, LOOM_PARAMETERIZED_ATTR_KIND_ANY);
  EXPECT_EQ(target.contextual_query_origins.touched_count, 1u);
  EXPECT_EQ(target_arena.used_allocation_size, bytes);

  iree_arena_deinitialize(&target_arena);
}

class FactOriginAllocationTest : public FactOriginStorageTest {
 protected:
  void SetUp() override {
    // Each array reaches the backing allocator so every publication allocation
    // can fail independently.
    iree_arena_block_pool_initialize(sizeof(iree_arena_block_t),
                                     {this, Allocate}, &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<FactOriginAllocationTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected origin allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  // Number of attempted backing allocations.
  iree_host_size_t allocation_count_ = 0;
  // Backing allocation to fail, or SIZE_MAX for unrestricted allocation.
  iree_host_size_t failure_index_ = SIZE_MAX;
};

TEST_F(FactOriginAllocationTest, FailedQueryPublicationCanClearAndReuseScope) {
  const auto origin = ContextualQueryOrigin(
      LOOM_PARAMETERIZED_ATTR_KIND(/*dialect_id=*/3, /*family_index=*/7),
      loom_attr_i64(32));
  for (iree_host_size_t failure = 0;; ++failure) {
    loom_value_fact_table_t table = {};
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
    allocation_count_ = 0;
    failure_index_ = failure;
    iree_status_t status =
        loom_value_fact_table_define_contextual_query_origin(&table, 7, origin);
    if (iree_status_is_ok(status)) {
      break;
    }
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(table.contextual_query_origins.origin_count, 0u);
    EXPECT_EQ(table.contextual_query_origins.touched_count, 0u);
    loom_value_fact_table_clear_scope(&table);
    failure_index_ = SIZE_MAX;
    IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
        &table, 3, origin));
    IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
        &table, 7, origin));
    EXPECT_EQ(table.contextual_query_origins.origin_count, 2u);
    EXPECT_EQ(table.contextual_query_origins.touched_count, 2u);
    EXPECT_NE(OriginEntry(table.contextual_query_origins, 3),
              OriginEntry(table.contextual_query_origins, 7));
    iree_arena_reset(&arena_);
  }
}

class FactOriginTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    const struct {
      // Registered dialect identifier.
      loom_dialect_id_t dialect;
      // Dialect operation inventory.
      const loom_op_vtable_t* const* (*vtables)(iree_host_size_t*);
    } dialects[] = {
        {LOOM_DIALECT_TEST, loom_test_dialect_vtables},
        {LOOM_DIALECT_CFG, loom_cfg_dialect_vtables},
        {LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables},
        {LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables},
        {LOOM_DIALECT_INDEX, loom_index_dialect_vtables},
        {LOOM_DIALECT_TARGET, loom_target_dialect_vtables},
    };
    for (const auto& dialect : dialects) {
      iree_host_size_t count = 0;
      const auto* vtables = dialect.vtables(&count);
      IREE_ASSERT_OK(loom_context_register_dialect(
          &context_, dialect.dialect, vtables, static_cast<uint16_t>(count)));
    }
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("origins"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_rewriter_initialize(&rewriter_, module_, &arena_);
    loom_string_id_t name;
    IREE_ASSERT_OK(loom_builder_intern_string(&rewriter_.builder,
                                              IREE_SV("origins"), &name));
    uint16_t symbol;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    const loom_type_t types[] = {scalar_type_,
                                 scalar_type_,
                                 vector_type_,
                                 vector_type_,
                                 index_type_,
                                 index_type_,
                                 loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                 loom_type_scalar(LOOM_SCALAR_TYPE_I32)};
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &rewriter_.builder, 0, 0, 0, {.module_id = 0, .symbol_id = symbol},
        types, IREE_ARRAYSIZE(types), nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &function));
    const auto body =
        loom_func_like_body(loom_func_like_cast(module_, function));
    loom_builder_enter_region(&rewriter_.builder, function, body);
    const auto block = loom_region_entry_block(body);
    for (uint16_t i = 0; i < IREE_ARRAYSIZE(inputs_); ++i) {
      inputs_[i] = loom_block_arg_id(block, i);
    }
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table_, &arena_, 0));
    loom_rewriter_attach_value_facts(&rewriter_, &table_);
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_op_t* Splat(loom_value_id_t scalar) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_vector_splat_build(
        &rewriter_.builder, scalar, vector_type_, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* Refine(loom_value_id_t source) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_vector_refine_build(
        &rewriter_.builder, source, loom_module_value_type(module_, source),
        LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  bool Compute(loom_op_t* op) {
    bool changed = false;
    IREE_CHECK_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                              op, &changed));
    return changed;
  }

  loom_value_id_t Uniform(loom_value_id_t value) {
    loom_value_id_t scalar = LOOM_VALUE_ID_INVALID;
    loom_value_fact_table_query_uniform_element_origin(&table_, module_, value,
                                                       &scalar);
    return scalar;
  }

  // Shared storage backing the module and analysis arenas.
  iree_arena_block_pool_t pool_;
  // Analysis entries and rewrite worklists.
  iree_arena_allocator_t arena_;
  // Registered inference and canonicalization implementations.
  loom_context_t context_;
  // Source module owned by the fixture.
  loom_module_t* module_ = nullptr;
  // Mutation boundary that publishes changed producer facts.
  loom_rewriter_t rewriter_ = {};
  // Materialization proof owner borrowed by the rewriter.
  loom_value_fact_table_t table_ = {};
  // Dynamic scalar, vector, and index pairs followed by a loop condition and
  // one fixed-width integer conversion source.
  loom_value_id_t inputs_[8];
  // Scalar element type used by the vector fixtures.
  const loom_type_t scalar_type_ = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  // Four-lane aggregate used by the vector fixtures.
  const loom_type_t vector_type_ = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  // Dynamic layout stride and target-query type.
  const loom_type_t index_type_ = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
};

TEST_F(FactOriginTest, RetargetedAliasMaterializesTheCurrentScalar) {
  auto* first = Splat(inputs_[0]);
  auto* second = Splat(inputs_[1]);
  auto* alias = Refine(loom_vector_splat_result(first));
  const int64_t offset = 0;
  const auto slice_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(2), 0);
  loom_op_t* slice = nullptr;
  IREE_ASSERT_OK(loom_vector_slice_build(
      &rewriter_.builder, loom_vector_refine_result(alias), nullptr, 0, &offset,
      1, slice_type, LOOM_LOCATION_UNKNOWN, &slice));
  const auto result = loom_vector_slice_result(slice);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&rewriter_.builder, &result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));
  IREE_ASSERT_OK(loom_rewriter_enable_worklist(&rewriter_));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, alias, 0,
                                           loom_vector_splat_result(second)));
  while (auto* op = loom_rewriter_pop(&rewriter_)) {
    bool folded = false;
    IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter_, op, &folded));
  }
  IREE_ASSERT_OK(loom_vector_slice_canonicalize(slice, &rewriter_));
  const auto* replacement = loom_value_def_op(
      loom_module_value(module_, loom_op_const_operands(use)[0]));
  ASSERT_TRUE(loom_vector_splat_isa(replacement));
  EXPECT_EQ(loom_vector_splat_scalar(replacement), inputs_[1]);
}

TEST_F(FactOriginTest, ExactScalarCastOriginsFlattenThroughAliases) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);

  loom_op_t* index_cast = nullptr;
  IREE_ASSERT_OK(loom_index_cast_build(&rewriter_.builder, inputs_[7], i32_type,
                                       index_type_, LOOM_LOCATION_UNKNOWN,
                                       &index_cast));
  EXPECT_FALSE(Compute(index_cast));
  const loom_value_id_t index_value = loom_index_cast_result(index_cast);
  loom_value_fact_exact_scalar_origin_t origin;
  ASSERT_TRUE(loom_value_fact_table_query_exact_scalar_origin(
      &table_, module_, index_value, &origin));
  EXPECT_EQ(origin.source_value_id, inputs_[7]);

  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(
      &rewriter_.builder, &index_value, 1, /*predicates=*/nullptr,
      /*predicates_count=*/0, &index_type_, 1, LOOM_LOCATION_UNKNOWN, &alias));
  EXPECT_FALSE(Compute(alias));
  const loom_value_id_t alias_value =
      loom_index_assume_results(alias).values[0];
  ASSERT_TRUE(loom_value_fact_table_query_exact_scalar_origin(
      &table_, module_, alias_value, &origin));
  EXPECT_EQ(origin.source_value_id, inputs_[7]);

  loom_op_t* i64_cast = nullptr;
  IREE_ASSERT_OK(loom_index_cast_build(&rewriter_.builder, alias_value,
                                       index_type_, i64_type,
                                       LOOM_LOCATION_UNKNOWN, &i64_cast));
  EXPECT_FALSE(Compute(i64_cast));
  ASSERT_TRUE(loom_value_fact_table_query_exact_scalar_origin(
      &table_, module_, loom_index_cast_result(i64_cast), &origin));
  EXPECT_EQ(origin.source_value_id, inputs_[7]);

  loom_op_t* narrowing_cast = nullptr;
  IREE_ASSERT_OK(loom_index_cast_build(&rewriter_.builder, inputs_[4],
                                       index_type_, i32_type,
                                       LOOM_LOCATION_UNKNOWN, &narrowing_cast));
  EXPECT_FALSE(Compute(narrowing_cast));
  EXPECT_FALSE(loom_value_fact_table_query_exact_scalar_origin(
      &table_, module_, loom_index_cast_result(narrowing_cast), &origin));

  loom_op_t* unsigned_cast = nullptr;
  IREE_ASSERT_OK(loom_index_cast_build(&rewriter_.builder, inputs_[7], i32_type,
                                       offset_type, LOOM_LOCATION_UNKNOWN,
                                       &unsigned_cast));
  EXPECT_FALSE(Compute(unsigned_cast));
  EXPECT_FALSE(loom_value_fact_table_query_exact_scalar_origin(
      &table_, module_, loom_index_cast_result(unsigned_cast), &origin));
}

TEST_F(FactOriginTest, IntegerCastPublicationRetargetsAndWithdraws) {
  const auto byte_type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  const auto offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_op_t* narrow = nullptr;
  IREE_ASSERT_OK(loom_index_cast_build(&rewriter_.builder, inputs_[4],
                                       index_type_, byte_type,
                                       LOOM_LOCATION_UNKNOWN, &narrow));
  const auto byte = loom_index_cast_result(narrow);
  loom_op_t* widen = nullptr;
  IREE_ASSERT_OK(loom_index_cast_build(&rewriter_.builder, byte, byte_type,
                                       offset_type, LOOM_LOCATION_UNKNOWN,
                                       &widen));
  const auto wide = loom_index_cast_result(widen);
  auto cast = loom_value_fact_table_query_integer_cast(&table_, wide);
  EXPECT_EQ(cast.immediate.source_value_id, byte);
  EXPECT_EQ(cast.composed.source_value_id, inputs_[4]);
  EXPECT_EQ(cast.composed.kind, LOOM_INTEGER_CAST_UNSIGNED);
  EXPECT_EQ(cast.composed.bit_count, 8);

  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, narrow, 0, inputs_[5]));
  EXPECT_TRUE(Compute(widen));
  cast = loom_value_fact_table_query_integer_cast(&table_, wide);
  EXPECT_EQ(cast.composed.source_value_id, inputs_[5]);

  loom_op_t* constant = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(&rewriter_.builder, loom_attr_i64(5),
                                           index_type_, LOOM_LOCATION_UNKNOWN,
                                           &constant));
  IREE_ASSERT_OK(loom_rewriter_set_operand(
      &rewriter_, narrow, 0, loom_index_constant_result(constant)));
  EXPECT_EQ(loom_value_fact_table_query_integer_cast(&table_, byte)
                .immediate.bit_count,
            0);
  EXPECT_TRUE(Compute(widen));
  EXPECT_EQ(loom_value_fact_table_query_integer_cast(&table_, wide)
                .immediate.bit_count,
            0);

  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, narrow, 0, inputs_[4]));
  EXPECT_TRUE(Compute(widen));
  cast = loom_value_fact_table_query_integer_cast(&table_, wide);
  EXPECT_EQ(cast.composed.source_value_id, inputs_[4]);
  const auto bytes = arena_.used_allocation_size;
  EXPECT_FALSE(Compute(narrow));
  EXPECT_FALSE(Compute(widen));
  EXPECT_EQ(arena_.used_allocation_size, bytes);
}

TEST_F(FactOriginTest, NumericQueryDoesNotPublishIntegerCasts) {
  const auto byte_type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  loom_op_t* narrow = nullptr;
  IREE_ASSERT_OK(loom_index_cast_build(&rewriter_.builder, inputs_[4],
                                       index_type_, byte_type,
                                       LOOM_LOCATION_UNKNOWN, &narrow));
  const auto result = loom_index_cast_result(narrow);
  loom_value_fact_table_clear_origins(&table_, result);
  const auto input = loom_value_facts_unknown();
  loom_value_facts_t output;
  loom_fact_context_t query = table_.context;
  ASSERT_EQ(query.publication, nullptr);
  IREE_ASSERT_OK(loom_op_vtable(module_, narrow)
                     ->infer_facts(&query, module_, narrow, &input, &output));
  EXPECT_EQ(output.range_lo, INT8_MIN);
  EXPECT_EQ(output.range_hi, INT8_MAX);
  EXPECT_EQ(loom_value_fact_table_query_integer_cast(&table_, result)
                .immediate.bit_count,
            0);
  EXPECT_TRUE(Compute(narrow));
  EXPECT_EQ(loom_value_fact_table_query_integer_cast(&table_, result)
                .immediate.bit_count,
            8);
}

TEST_F(FactOriginTest, CyclicPublicationRefreshesOutsideMaterialization) {
  const auto function =
      loom_func_like_cast(module_, rewriter_.builder.ip.parent_op);
  auto* region = loom_func_like_body(function);
  auto* splat = Splat(inputs_[0]);
  loom_block_t* header = nullptr;
  loom_block_t* latch = nullptr;
  loom_block_t* exit = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, region, &header));
  IREE_ASSERT_OK(loom_region_append_block(module_, region, &latch));
  IREE_ASSERT_OK(loom_region_append_block(module_, region, &exit));
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&rewriter_.builder, header, nullptr, 0,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&rewriter_.builder, header);
  auto* alias = Refine(loom_vector_splat_result(splat));
  IREE_ASSERT_OK(loom_cfg_cond_br_build(&rewriter_.builder, inputs_[6], latch,
                                        exit, LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&rewriter_.builder, latch);
  IREE_ASSERT_OK(loom_cfg_br_build(&rewriter_.builder, header, nullptr, 0,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&rewriter_.builder, exit);
  const auto slice_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(2), 0);
  const int64_t offset = 0;
  loom_op_t* slice = nullptr;
  IREE_ASSERT_OK(loom_vector_slice_build(
      &rewriter_.builder, loom_vector_refine_result(alias), nullptr, 0, &offset,
      1, slice_type, LOOM_LOCATION_UNKNOWN, &slice));
  const auto result = loom_vector_slice_result(slice);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&rewriter_.builder, &result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));
  IREE_ASSERT_OK(loom_test_yield_build(&rewriter_.builder, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &branch));
  IREE_ASSERT_OK(loom_rewriter_enable_analysis(&rewriter_, function, &table_));
  // Rebuilding the same auxiliary proofs must not notify unchanged users.
  const auto* structure =
      loom_value_fact_table_lookup_cfg_region(&table_, region);
  const auto* component =
      &structure->control_flow.components
           .values[structure->graph.blocks[header->region_index].component];
  iree_arena_allocator_t scratch;
  iree_arena_initialize(&pool_, &scratch);
  bool changed = false;
  iree_status_t status = loom_value_fact_table_recompute_cfg_component(
      &table_, module_, structure, component, &scratch,
      [](void* data, loom_value_id_t) {
        *static_cast<bool*>(data) = true;
        return iree_ok_status();
      },
      &changed);
  iree_arena_deinitialize(&scratch);
  IREE_ASSERT_OK(status);
  EXPECT_FALSE(changed);
  IREE_ASSERT_OK(loom_rewriter_enable_worklist(&rewriter_));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, splat, 0, inputs_[1]));
  while (auto* op = loom_rewriter_pop(&rewriter_)) {
    bool folded = false;
    IREE_ASSERT_OK(loom_rewriter_try_fold(&rewriter_, op, &folded));
  }
  EXPECT_EQ(Uniform(loom_vector_refine_result(alias)), inputs_[1]);
  EXPECT_EQ(Uniform(result), inputs_[1]);
  IREE_ASSERT_OK(loom_vector_slice_canonicalize(slice, &rewriter_));
  const auto* replacement = loom_value_def_op(
      loom_module_value(module_, loom_op_const_operands(use)[0]));
  ASSERT_TRUE(loom_vector_splat_isa(replacement));
  EXPECT_EQ(loom_vector_splat_scalar(replacement), inputs_[1]);
}

TEST_F(FactOriginTest, ConditionalUniformProofIsWithdrawnAndStorageReused) {
  const loom_value_id_t elements[] = {inputs_[0], inputs_[0], inputs_[0],
                                      inputs_[0]};
  loom_op_t* vector = nullptr;
  IREE_ASSERT_OK(loom_vector_from_elements_build(
      &rewriter_.builder, elements, IREE_ARRAYSIZE(elements), vector_type_,
      LOOM_LOCATION_UNKNOWN, &vector));
  const auto result = loom_vector_from_elements_result(vector);
  EXPECT_EQ(Uniform(result), inputs_[0]);
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, vector, 1, inputs_[1]));
  EXPECT_EQ(Uniform(result), LOOM_VALUE_ID_INVALID);
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, vector, 1, inputs_[0]));
  const auto touched = table_.touched_count;
  const auto bytes = arena_.used_allocation_size;
  EXPECT_FALSE(Compute(vector));
  EXPECT_EQ(arena_.used_allocation_size, bytes);
  EXPECT_EQ(table_.touched_count, touched);
  EXPECT_EQ(Uniform(result), inputs_[0]);
}

TEST_F(FactOriginTest, LaneProofChangesReachBothResultsAndSlices) {
  const auto half = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                                        loom_dim_pack_static(2), 0);
  loom_op_t* split = nullptr;
  IREE_ASSERT_OK(loom_vector_deinterleave_build(&rewriter_.builder, 0,
                                                inputs_[2], half, half,
                                                LOOM_LOCATION_UNKNOWN, &split));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, split, 0, inputs_[3]));
  for (uint16_t i = 0; i < 2; ++i) {
    loom_value_fact_static_lane_origin_t origin;
    ASSERT_TRUE(loom_value_fact_table_query_static_lane_origin(
        &table_, module_, loom_op_const_results(split)[i], &origin));
    EXPECT_EQ(origin.source_value_id, inputs_[3]);
    EXPECT_EQ(origin.source_lane_offset, i);
    EXPECT_EQ(origin.source_lane_stride, 2u);
  }
  const int64_t offset = 1;
  loom_op_t* slice = nullptr;
  IREE_ASSERT_OK(loom_vector_slice_build(&rewriter_.builder, inputs_[2],
                                         nullptr, 0, &offset, 1, half,
                                         LOOM_LOCATION_UNKNOWN, &slice));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, slice, 0, inputs_[3]));
  const auto result = loom_vector_slice_result(slice);
  loom_value_fact_exact_lane_origin_t exact;
  ASSERT_TRUE(loom_value_fact_table_query_exact_lane_origin(&table_, module_,
                                                            result, &exact));
  EXPECT_EQ(exact.source_value_id, inputs_[3]);
  EXPECT_EQ(exact.source_lane_offset, 1u);
  const auto count = table_.exact_lane_origins.count;
  const auto bytes = arena_.used_allocation_size;
  for (unsigned i = 0; i < 8; ++i) {
    loom_value_fact_table_undefine(&table_, result);
    EXPECT_FALSE(loom_value_fact_table_query_exact_lane_origin(&table_, module_,
                                                               result, &exact));
    EXPECT_TRUE(Compute(slice));
  }
  EXPECT_EQ(table_.exact_lane_origins.count, count);
  EXPECT_EQ(arena_.used_allocation_size, bytes);
}

TEST_F(FactOriginTest, ScaleProofTracksOperandsAndAbsence) {
  auto* scale = Splat(inputs_[0]);
  loom_op_t* product = nullptr;
  IREE_ASSERT_OK(loom_vector_mulf_build(
      &rewriter_.builder, 0, inputs_[2], loom_vector_splat_result(scale),
      vector_type_, LOOM_LOCATION_UNKNOWN, &product));
  const auto result = loom_vector_mulf_result(product);
  loom_value_fact_uniform_scale_origin_t origin;
  ASSERT_TRUE(loom_value_fact_table_query_uniform_scale_origin(
      &table_, module_, result, &origin));
  EXPECT_EQ(origin.source_value_id, inputs_[2]);
  EXPECT_EQ(origin.scale_value_id, inputs_[0]);
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, product, 1, inputs_[3]));
  EXPECT_FALSE(loom_value_fact_table_query_uniform_scale_origin(
      &table_, module_, result, &origin));
  EXPECT_FALSE(Compute(product));
}

TEST_F(FactOriginTest, TwoUniformFactorsPublishOneStableScaleProof) {
  auto* left = Splat(inputs_[0]);
  auto* right = Splat(inputs_[1]);
  loom_op_t* product = nullptr;
  IREE_ASSERT_OK(loom_vector_mulf_build(
      &rewriter_.builder, 0, loom_vector_splat_result(left),
      loom_vector_splat_result(right), vector_type_, LOOM_LOCATION_UNKNOWN,
      &product));
  const auto bytes = arena_.used_allocation_size;
  EXPECT_FALSE(Compute(product));
  EXPECT_EQ(arena_.used_allocation_size, bytes);
}

TEST_F(FactOriginTest, ContextualAliasWithdrawsWithoutChangingOtherValues) {
  loom_op_t* query = nullptr;
  IREE_ASSERT_OK(loom_target_subgroup_size_build(
      &rewriter_.builder, index_type_, LOOM_LOCATION_UNKNOWN, &query));
  const auto source = loom_target_subgroup_size_result(query);
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&rewriter_.builder, &source, 1,
                                         nullptr, 0, &index_type_, 1,
                                         LOOM_LOCATION_UNKNOWN, &alias));
  const auto result = loom_op_const_results(alias)[0];
  const auto count = table_.contextual_query_origins.origin_count;
  for (unsigned i = 0; i < 8; ++i) {
    IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, alias, 0, inputs_[4]));
    loom_value_fact_contextual_query_origin_t origin;
    EXPECT_FALSE(loom_value_fact_table_query_contextual_query_origin(
        &table_, module_, result, &origin));
    EXPECT_TRUE(loom_value_fact_table_query_contextual_query_origin(
        &table_, module_, source, &origin));
    IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter_, alias, 0, source));
    EXPECT_TRUE(loom_value_fact_table_query_contextual_query_origin(
        &table_, module_, result, &origin));
  }
  EXPECT_EQ(table_.contextual_query_origins.origin_count, count);
  EXPECT_EQ(table_.contextual_query_origins.touched_count, count);
  EXPECT_FALSE(Compute(alias));
}

TEST_F(FactOriginTest, LayoutAliasPublishesOnlyChangedBindings) {
  const auto type =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  const int64_t dynamic = INT64_MIN;
  loom_op_t* first = nullptr;
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_encoding_layout_strided_build(
      &rewriter_.builder, &inputs_[4], 1, &dynamic, 1, type,
      LOOM_LOCATION_UNKNOWN, &first));
  IREE_ASSERT_OK(loom_encoding_layout_strided_build(
      &rewriter_.builder, &inputs_[5], 1, &dynamic, 1, type,
      LOOM_LOCATION_UNKNOWN, &second));
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(loom_encoding_layout_assume_strided_build(
      &rewriter_.builder, loom_encoding_layout_strided_result(first), 1, type,
      LOOM_LOCATION_UNKNOWN, &alias));
  const auto result = loom_encoding_layout_assume_strided_result(alias);
  const auto before =
      loom_value_fact_table_query_layout_origin(&table_, result);
  ASSERT_EQ(before.kind, LOOM_VALUE_FACT_LAYOUT_ORIGIN_EXPLICIT_STRIDES);
  ASSERT_EQ(before.count, 1u);
  EXPECT_EQ(before.values.explicit_strides[0], inputs_[4]);
  IREE_ASSERT_OK(loom_rewriter_set_operand(
      &rewriter_, alias, 0, loom_encoding_layout_strided_result(second)));
  const auto after = loom_value_fact_table_query_layout_origin(&table_, result);
  ASSERT_EQ(after.count, 1u);
  EXPECT_EQ(after.values.explicit_strides[0], inputs_[5]);
  EXPECT_EQ(before.values.explicit_strides[0], inputs_[4]);
  const auto bytes = arena_.used_allocation_size;
  EXPECT_FALSE(Compute(alias));
  EXPECT_EQ(
      loom_value_fact_table_query_layout_origin(&table_, result).values.data,
      after.values.data);
  EXPECT_EQ(arena_.used_allocation_size, bytes);
}

TEST_F(FactOriginTest, NumericQueryDoesNotPublishUniformOrigins) {
  auto* splat = Splat(inputs_[0]);
  const auto result = loom_vector_splat_result(splat);
  loom_value_fact_table_clear_origins(&table_, result);
  ASSERT_EQ(Uniform(result), LOOM_VALUE_ID_INVALID);
  const auto input = loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 2.0);
  loom_value_facts_t output;
  loom_fact_context_t query = table_.context;
  ASSERT_EQ(query.publication, nullptr);
  IREE_ASSERT_OK(loom_op_vtable(module_, splat)
                     ->infer_facts(&query, module_, splat, &input, &output));
  loom_value_fact_uniform_element_t element;
  ASSERT_TRUE(loom_value_facts_query_uniform_element(&query, output, &element));
  double value;
  ASSERT_TRUE(loom_value_facts_as_exact_float(LOOM_SCALAR_TYPE_F32,
                                              element.element, &value));
  EXPECT_EQ(value, 2.0);
  EXPECT_EQ(Uniform(result), LOOM_VALUE_ID_INVALID);
  EXPECT_TRUE(Compute(splat));
  EXPECT_EQ(Uniform(result), inputs_[0]);
}

TEST_F(FactOriginTest, NumericQueryDoesNotPublishLayoutOrigins) {
  const auto type =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  const int64_t dynamic = INT64_MIN;
  loom_op_t* layout = nullptr;
  IREE_ASSERT_OK(loom_encoding_layout_strided_build(
      &rewriter_.builder, &inputs_[4], 1, &dynamic, 1, type,
      LOOM_LOCATION_UNKNOWN, &layout));
  const auto result = loom_encoding_layout_strided_result(layout);
  loom_value_fact_table_clear_origins(&table_, result);
  const auto input = loom_value_facts_exact_i64(4);
  loom_value_facts_t output;
  loom_fact_context_t query = table_.context;
  ASSERT_EQ(query.publication, nullptr);
  IREE_ASSERT_OK(loom_op_vtable(module_, layout)
                     ->infer_facts(&query, module_, layout, &input, &output));
  loom_value_fact_encoding_summary_t summary;
  ASSERT_TRUE(
      loom_value_facts_query_encoding_summary(&query, output, &summary));
  ASSERT_EQ(summary.address_layout.rank, 1);
  EXPECT_EQ(summary.address_layout.strides[0].range_lo, 4);
  EXPECT_EQ(summary.address_layout.strides[0].range_hi, 4);
  EXPECT_EQ(loom_value_fact_table_query_layout_origin(&table_, result).kind,
            LOOM_VALUE_FACT_LAYOUT_ORIGIN_NONE);
  EXPECT_TRUE(Compute(layout));
  const auto origin =
      loom_value_fact_table_query_layout_origin(&table_, result);
  ASSERT_EQ(origin.kind, LOOM_VALUE_FACT_LAYOUT_ORIGIN_EXPLICIT_STRIDES);
  ASSERT_EQ(origin.count, 1);
  EXPECT_EQ(origin.values.explicit_strides[0], inputs_[4]);
}

}  // namespace
}  // namespace loom
