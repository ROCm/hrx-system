// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_cast.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class FactCastTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table_, &arena_, 0));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static loom_value_fact_integer_cast_t Cast(loom_value_id_t source,
                                             uint8_t width,
                                             loom_integer_cast_kind_t kind) {
    const loom_integer_cast_t cast = {
        .source_value_id = source,
        .bit_count = width,
        .kind = static_cast<uint8_t>(kind),
        .unsigned_bit_count = 0,
    };
    return {.immediate = cast, .composed = cast};
  }

  // Shared backing blocks for independently scoped tables.
  iree_arena_block_pool_t pool_;
  // Storage for the table's fact segments and transient publications.
  iree_arena_allocator_t arena_;
  // Primary populated scope under test.
  loom_value_fact_table_t table_ = {};
};

TEST_F(FactCastTest, AbsentProofDoesNotAllocate) {
  const auto bytes = arena_.used_allocation_size;
  EXPECT_EQ(loom_value_fact_table_query_integer_cast(&table_, 100)
                .immediate.bit_count,
            0);
  IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(&table_, 100, {}));
  loom_value_fact_table_clear_integer_cast(&table_, 100);
  EXPECT_EQ(table_.integer_casts, nullptr);
  EXPECT_EQ(arena_.used_allocation_size, bytes);
}

TEST_F(FactCastTest, PublicationReusesSlotsAndWithdrawsBothEndpoints) {
  auto cast = Cast(3, 8, LOOM_INTEGER_CAST_SIGNED);
  cast.composed.source_value_id = 1;
  IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(&table_, 7, cast));
  const auto bytes = arena_.used_allocation_size;
  for (int i = 0; i < 32; ++i) {
    IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(&table_, 7, cast));
    EXPECT_TRUE(loom_value_fact_integer_cast_equal(
        loom_value_fact_table_query_integer_cast(&table_, 7), cast));
    loom_value_fact_table_clear_origins(&table_, 7);
    const auto absent = loom_value_fact_table_query_integer_cast(&table_, 7);
    EXPECT_EQ(absent.immediate.bit_count, 0);
    EXPECT_EQ(absent.composed.bit_count, 0);
  }
  EXPECT_EQ(arena_.used_allocation_size, bytes);
}

TEST_F(FactCastTest, SnapshotsObserveInterpretationAndEndpointChanges) {
  auto cast = Cast(2, 8, LOOM_INTEGER_CAST_SIGNED);
  IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(&table_, 5, cast));
  loom_value_fact_origin_snapshot_t original;
  loom_value_fact_table_snapshot_origins(&table_, 5, &original);

  cast.composed =
      loom_integer_cast_compose(cast.composed, LOOM_INTEGER_CAST_UNSIGNED, 16);
  cast.immediate = Cast(3, 16, LOOM_INTEGER_CAST_UNSIGNED).immediate;
  IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(&table_, 5, cast));
  loom_value_fact_origin_snapshot_t unsigned_result;
  loom_value_fact_table_snapshot_origins(&table_, 5, &unsigned_result);
  EXPECT_FALSE(
      loom_value_fact_origin_snapshot_equal(&original, &unsigned_result));

  cast.composed.source_value_id = 4;
  IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(&table_, 5, cast));
  loom_value_fact_origin_snapshot_t new_source;
  loom_value_fact_table_snapshot_origins(&table_, 5, &new_source);
  EXPECT_FALSE(
      loom_value_fact_origin_snapshot_equal(&unsigned_result, &new_source));
  EXPECT_TRUE(loom_value_fact_origin_snapshot_equal(&new_source, &new_source));
}

TEST_F(FactCastTest, ClonedScopeOwnsItsPublicationAndAbsence) {
  const auto cast = Cast(2, 16, LOOM_INTEGER_CAST_UNSIGNED);
  IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(&table_, 6, cast));
  iree_arena_allocator_t clone_arena;
  iree_arena_initialize(&pool_, &clone_arena);
  loom_value_fact_table_t clone = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &clone_arena, 0));
  IREE_ASSERT_OK(loom_value_fact_table_clone_origins(&clone, &table_, 6));
  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(table_.integer_casts, nullptr);
  EXPECT_TRUE(loom_value_fact_integer_cast_equal(
      loom_value_fact_table_query_integer_cast(&clone, 6), cast));
  IREE_ASSERT_OK(loom_value_fact_table_clone_origins(&clone, &table_, 6));
  EXPECT_EQ(
      loom_value_fact_table_query_integer_cast(&clone, 6).immediate.bit_count,
      0);
  iree_arena_deinitialize(&clone_arena);
}

TEST_F(FactCastTest, SparseStorageDependsOnCastCountNotValueSpan) {
  // Numeric slots are already owned by ordinary inference; account for only
  // the sparse cast storage added to that same scope.
  for (uint32_t i = 0; i < 64; ++i) {
    IREE_ASSERT_OK(loom_value_fact_table_touch(&table_, 10000 + 1024 * i));
  }
  const auto bytes = arena_.used_allocation_size;
  for (uint32_t i = 0; i < 64; ++i) {
    IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(
        &table_, 10000 + 1024 * i, Cast(i, 8, LOOM_INTEGER_CAST_UNSIGNED)));
  }
  EXPECT_LT(arena_.used_allocation_size - bytes, 8192u);
  for (uint32_t i = 0; i < 64; ++i) {
    const auto cast =
        loom_value_fact_table_query_integer_cast(&table_, 10000 + 1024 * i);
    EXPECT_EQ(cast.composed.source_value_id, i);
    EXPECT_EQ(cast.composed.bit_count, 8);
    EXPECT_EQ(
        loom_value_fact_table_query_integer_cast(&table_, 10001 + 1024 * i)
            .immediate.bit_count,
        0);
  }
}

TEST_F(FactCastTest, TraversalVisitsOnlyCurrentPublications) {
  const auto cast = Cast(1, 8, LOOM_INTEGER_CAST_UNSIGNED);
  IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(&table_, 2, cast));
  IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(&table_, 3, cast));
  loom_value_fact_table_clear_integer_cast(&table_, 2);
  std::vector<loom_value_id_t> visited;
  IREE_ASSERT_OK(loom_value_fact_table_for_each_integer_cast(
      &table_,
      [](void* user_data, loom_value_id_t value,
         loom_value_fact_integer_cast_t publication) -> iree_status_t {
        auto& visited = *static_cast<std::vector<loom_value_id_t>*>(user_data);
        visited.push_back(value);
        EXPECT_EQ(publication.composed.source_value_id, 1u);
        EXPECT_EQ(publication.composed.bit_count, 8);
        return iree_ok_status();
      },
      &visited));
  ASSERT_EQ(visited.size(), 1u);
  EXPECT_EQ(visited[0], 3u);

  loom_value_fact_table_clear_scope(&table_);
  visited.clear();
  IREE_ASSERT_OK(loom_value_fact_table_for_each_integer_cast(
      &table_,
      [](void* user_data, loom_value_id_t value,
         loom_value_fact_integer_cast_t) -> iree_status_t {
        static_cast<std::vector<loom_value_id_t>*>(user_data)->push_back(value);
        return iree_ok_status();
      },
      &visited));
  EXPECT_TRUE(visited.empty());
}

// Evaluate retained payloads independently of the relation query. Signed and
// mixed interpretations preserve their low bits even when their values differ.
static uint64_t Interpret(const loom_integer_cast_t& cast, uint64_t value) {
  const uint64_t mask =
      cast.bit_count == 64 ? UINT64_MAX : (UINT64_C(1) << cast.bit_count) - 1;
  value &= mask;
  if (cast.kind != LOOM_INTEGER_CAST_UNSIGNED &&
      (value & (UINT64_C(1) << (cast.bit_count - 1)))) {
    value |= ~mask;
  }
  if (cast.kind == LOOM_INTEGER_CAST_SIGNED_THEN_UNSIGNED &&
      cast.unsigned_bit_count != 64) {
    value &= (UINT64_C(1) << cast.unsigned_bit_count) - 1;
  }
  return value;
}

TEST_F(FactCastTest, SharedPayloadRelationsMatchConcreteBits) {
  std::vector<loom_value_fact_integer_cast_t> casts;
  for (uint8_t width : {1, 8, 16, 32, 64}) {
    for (auto kind : {LOOM_INTEGER_CAST_SIGNED, LOOM_INTEGER_CAST_UNSIGNED}) {
      casts.push_back(Cast(1, width, kind));
    }
    for (uint8_t extension : {8, 16, 32, 64}) {
      if (extension <= width) {
        continue;
      }
      auto mixed = Cast(1, width, LOOM_INTEGER_CAST_SIGNED);
      mixed.composed = loom_integer_cast_compose(
          mixed.composed, LOOM_INTEGER_CAST_UNSIGNED, extension);
      mixed.immediate =
          Cast(7 + width, extension, LOOM_INTEGER_CAST_UNSIGNED).immediate;
      casts.push_back(mixed);
    }
  }
  const uint64_t boundaries[] = {0,         1,
                                 127,       128,
                                 255,       256,
                                 32767,     32768,
                                 65535,     65536,
                                 INT32_MAX, UINT32_MAX,
                                 INT64_MAX, UINT64_C(1) << 63,
                                 UINT64_MAX};
  for (const auto& value_cast : casts) {
    IREE_ASSERT_OK(
        loom_value_fact_table_define_integer_cast(&table_, 2, value_cast));
    for (const auto& source_cast : casts) {
      IREE_ASSERT_OK(
          loom_value_fact_table_define_integer_cast(&table_, 3, source_cast));
      loom_integer_cast_t relative;
      const bool found = loom_value_fact_table_query_integer_cast_from(
          &table_, 2, 3, &relative);
      ASSERT_EQ(found, value_cast.composed.bit_count <=
                           source_cast.composed.bit_count);
      if (!found) {
        continue;
      }
      for (uint64_t bits : boundaries) {
        EXPECT_EQ(Interpret(relative, Interpret(source_cast.composed, bits)),
                  Interpret(value_cast.composed, bits));
      }
    }
  }
}

TEST_F(FactCastTest, IntermediatePayloadRebasesWithoutRecoveringHighBits) {
  IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(
      &table_, 2, Cast(1, 16, LOOM_INTEGER_CAST_SIGNED)));
  IREE_ASSERT_OK(loom_value_fact_table_define_integer_cast(
      &table_, 3, Cast(1, 8, LOOM_INTEGER_CAST_UNSIGNED)));
  loom_integer_cast_t relative;
  ASSERT_TRUE(
      loom_value_fact_table_query_integer_cast_from(&table_, 3, 2, &relative));
  EXPECT_EQ(relative.source_value_id, 2u);
  EXPECT_EQ(relative.bit_count, 8);
  EXPECT_TRUE(loom_integer_cast_is_identity(&relative,
                                            loom_value_facts_make(1, 16, 1)));
  EXPECT_FALSE(loom_integer_cast_is_identity(
      &relative, loom_value_facts_make(-16, -1, 1)));
  EXPECT_FALSE(
      loom_value_fact_table_query_integer_cast_from(&table_, 2, 3, &relative));
  EXPECT_FALSE(
      loom_value_fact_table_query_integer_cast_from(&table_, 3, 4, &relative));
}

}  // namespace
}  // namespace loom
