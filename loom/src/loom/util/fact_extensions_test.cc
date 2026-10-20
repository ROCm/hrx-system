// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_extensions.h"

#include <cstdint>
#include <limits>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/float_facts.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class FactExtensionsTest : public ::testing::Test {
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

  // Blocks backing fact extension storage for one test.
  iree_arena_block_pool_t block_pool_;
  // Owns all fact tables and extension payloads created by the test.
  iree_arena_allocator_t arena_;
};

TEST_F(FactExtensionsTest, TypedMeetPreservesStableExtension) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t extension = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(7), &extension));
  loom_value_facts_t lhs = loom_value_facts_exact_i64(4);
  lhs.extension_id = extension.extension_id;
  loom_value_facts_t rhs = loom_value_facts_exact_i64(12);
  rhs.extension_id = extension.extension_id;

  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I64, loom_dim_pack_static(4), 0);
  loom_value_facts_t joined = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &table, nullptr, vector_type, &table, lhs, &table, rhs, &joined));

  EXPECT_EQ(joined.range_lo, 4);
  EXPECT_EQ(joined.range_hi, 12);
  loom_value_fact_uniform_element_t uniform = {};
  EXPECT_TRUE(
      loom_value_facts_query_uniform_element(&table.context, joined, &uniform));
  EXPECT_EQ(uniform.element.range_lo, 7);
}

TEST_F(FactExtensionsTest, AggregateExactFloatFormats) {
  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  const loom_scalar_type_t types[] = {
      LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
      LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,    LOOM_SCALAR_TYPE_F64,
  };
  for (const auto left_type : types) {
    for (const auto right_type : types) {
      SCOPED_TRACE(::testing::Message()
                   << "left=" << left_type << " right=" << right_type);
      loom_value_facts_t lanes[2];
      const auto unknown = loom_value_facts_unknown();
      loom_value_facts_eval_float_conversion(left_type, LOOM_SCALAR_TYPE_F64,
                                             {}, &unknown, &lanes[0]);
      loom_value_facts_eval_float_conversion(right_type, LOOM_SCALAR_TYPE_F64,
                                             {}, &unknown, &lanes[1]);
      loom_value_facts_mark_workgroup_uniform(&lanes[0]);
      loom_value_facts_mark_workgroup_uniform(&lanes[1]);
      loom_value_facts_t uniform;
      IREE_ASSERT_OK(loom_value_facts_make_uniform_element(&table.context,
                                                           lanes[0], &uniform));
      EXPECT_EQ(loom_value_facts_exact_float_types(uniform),
                loom_value_facts_exact_float_types(lanes[0]));
      EXPECT_FALSE(loom_value_facts_is_subgroup_uniform(uniform));

      loom_value_facts_t aggregate;
      IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
          &table.context, {lanes, IREE_ARRAYSIZE(lanes)}, &aggregate));
      EXPECT_EQ(loom_value_facts_exact_float_types(aggregate),
                loom_value_facts_exact_float_types(lanes[0]) &
                    loom_value_facts_exact_float_types(lanes[1]));
      EXPECT_FALSE(loom_value_facts_is_subgroup_uniform(aggregate));
      loom_value_fact_small_static_lanes_t retained;
      ASSERT_TRUE(loom_value_facts_query_small_static_lanes(
          &table.context, aggregate, &retained));
      ASSERT_EQ(retained.count, 2);
      for (size_t i = 0; i < retained.count; ++i) {
        EXPECT_TRUE(loom_value_facts_equal(retained.lanes[i], lanes[i]));
      }
    }
  }
}

TEST_F(FactExtensionsTest, AggregateFormatsRequireEveryComponent) {
  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT + 1];
  for (auto& lane : lanes) {
    lane = loom_value_facts_known_nan();
  }
  for (const size_t count :
       {size_t{0}, size_t{1}, size_t{LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT},
        IREE_ARRAYSIZE(lanes)}) {
    loom_value_facts_t aggregate;
    IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
        &table.context, {lanes, count}, &aggregate));
    EXPECT_EQ(loom_value_facts_exact_float_types(aggregate),
              count > 0 && count <= LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT
                  ? LOOM_SCALAR_TYPE_SET_FLOAT
                  : LOOM_SCALAR_TYPE_SET_NONE);
  }
  lanes[0] = loom_value_facts_unknown();
  loom_value_facts_t aggregate;
  IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
      &table.context, {lanes, 2}, &aggregate));
  EXPECT_EQ(loom_value_facts_exact_float_types(aggregate),
            LOOM_SCALAR_TYPE_SET_NONE);

  // Integer topology domains share physical flag bits with float formats.
  auto coordinate = loom_value_facts_make(0, 31, 1);
  ASSERT_TRUE(loom_value_facts_mark_topology_domain(
      &coordinate, LOOM_VALUE_FACT_TOPOLOGY_VALUE_SUBGROUP_LANE_ID,
      LOOM_VALUE_FACT_TOPOLOGY_AXIS_LANE));
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(&table.context,
                                                       coordinate, &aggregate));
  EXPECT_EQ(loom_value_facts_exact_float_types(aggregate),
            LOOM_SCALAR_TYPE_SET_NONE);
  EXPECT_FALSE(loom_value_facts_is_float(aggregate));
  lanes[0] = coordinate;
  lanes[1] = coordinate;
  IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
      &table.context, {lanes, 2}, &aggregate));
  EXPECT_EQ(loom_value_facts_exact_float_types(aggregate),
            LOOM_SCALAR_TYPE_SET_NONE);
  EXPECT_FALSE(loom_value_facts_is_float(aggregate));
}

TEST_F(FactExtensionsTest, TypedMeetPreservesFloatingClassesAcrossTables) {
  loom_value_fact_table_t source = {0};
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &arena_, 0));
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (loom_scalar_type_t scalar_type :
       {LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16, LOOM_SCALAR_TYPE_BF16,
        LOOM_SCALAR_TYPE_F32, LOOM_SCALAR_TYPE_F64}) {
    const double subnormal = scalar_type == LOOM_SCALAR_TYPE_F8E5M2 ? 0x1p-16
                             : scalar_type == LOOM_SCALAR_TYPE_F16  ? 0x1p-24
                             : scalar_type == LOOM_SCALAR_TYPE_BF16 ? 0x1p-133
                             : scalar_type == LOOM_SCALAR_TYPE_F32  ? 0x1p-149
                                                                    : 0x1p-1074;
    const double values[] = {0.0,       -0.0,     1.0,       2.0,
                             subnormal, infinity, -infinity, nan};
    const loom_type_t type = loom_type_scalar(scalar_type);
    for (size_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
      for (size_t j = 0; j < IREE_ARRAYSIZE(values); ++j) {
        SCOPED_TRACE(::testing::Message()
                     << "type=" << scalar_type << " lhs=" << i << " rhs=" << j);
        const auto lhs = loom_value_facts_exact_float(scalar_type, values[i]);
        const auto rhs = loom_value_facts_exact_float(scalar_type, values[j]);
        loom_value_facts_t joined;
        IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
            &target, nullptr, type, &source, lhs, &target, rhs, &joined));
        EXPECT_TRUE(loom_value_facts_is_float(joined));
        EXPECT_EQ(loom_value_facts_is_finite(joined), i < 5 && j < 5);
        EXPECT_EQ(loom_value_facts_is_nan(joined), i == 7 && j == 7);
        EXPECT_EQ(loom_value_facts_is_inf(joined),
                  (i == 5 || i == 6) && (j == 5 || j == 6));
        EXPECT_EQ(loom_value_facts_is_not_nan(joined), i != 7 && j != 7);
        EXPECT_EQ(loom_value_facts_is_not_inf(joined),
                  i != 5 && i != 6 && j != 5 && j != 6);
        EXPECT_EQ(loom_value_facts_is_exact(joined), i == j);
        EXPECT_EQ(loom_value_facts_is_not_subnormal(joined), i != 4 && j != 4);
        EXPECT_TRUE(loom_value_facts_is_cluster_uniform(joined));
      }
    }
  }
}

TEST_F(FactExtensionsTest,
       TypedMeetPreservesFloatDistributionWithoutExactness) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  auto finite = loom_value_facts_unknown();
  finite.flags = LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_FINITE |
                 LOOM_VALUE_FACT_NOT_NAN | LOOM_VALUE_FACT_NOT_INF;
  loom_value_facts_mark_workgroup_uniform(&finite);
  const auto exact = loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.0);
  loom_value_facts_t joined;
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &table, nullptr, type, &table, finite, &table, exact, &joined));
  EXPECT_TRUE(loom_value_facts_is_finite(joined));
  EXPECT_FALSE(loom_value_facts_is_exact(joined));
  EXPECT_EQ(loom_value_facts_uniform_scope(joined),
            LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);

  loom_value_facts_mark_lane_varying(&finite);
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &table, nullptr, type, &table, finite, &table, exact, &joined));
  EXPECT_TRUE(loom_value_facts_is_finite(joined));
  EXPECT_TRUE(loom_value_facts_is_lane_varying(joined));
  EXPECT_FALSE(loom_value_facts_is_subgroup_uniform(joined));

  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &table, nullptr, type, &table, finite, &table, loom_value_facts_unknown(),
      &joined));
  EXPECT_FALSE(loom_value_facts_is_finite(joined));
  EXPECT_FALSE(loom_value_facts_is_not_nan(joined));
  EXPECT_FALSE(loom_value_facts_is_not_inf(joined));
  EXPECT_TRUE(loom_value_facts_is_lane_varying(joined));
}

TEST_F(FactExtensionsTest, TypedMeetJoinsFiniteFloatIntervals) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  auto lhs = loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, -4.0, 2.0);
  auto rhs = loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, -1.0, 8.0);
  loom_value_facts_mark_workgroup_uniform(&lhs);
  loom_value_facts_mark_lane_varying(&rhs);

  loom_value_facts_t joined;
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &table, nullptr, type, &table, lhs, &table, rhs, &joined));
  double lo = 0.0;
  double hi = 0.0;
  ASSERT_TRUE(
      loom_value_facts_as_float_range(LOOM_SCALAR_TYPE_F32, joined, &lo, &hi));
  EXPECT_DOUBLE_EQ(lo, -4.0);
  EXPECT_DOUBLE_EQ(hi, 8.0);
  EXPECT_TRUE(loom_value_facts_is_finite(joined));
  EXPECT_TRUE(loom_value_facts_is_lane_varying(joined));
}

TEST_F(FactExtensionsTest, TypedWidenPreservesFloatingClasses) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_F64);
  const auto previous = loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F64, 1.0);
  for (const uint32_t iteration : {0u, 2u, 7u}) {
    loom_value_facts_t widened;
    const auto subnormal = loom_value_facts_exact_float(
        LOOM_SCALAR_TYPE_F64, std::numeric_limits<double>::denorm_min());
    IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
        &table, nullptr, type, &table, previous, &table, subnormal, iteration,
        &widened));
    EXPECT_TRUE(loom_value_facts_is_finite(widened));
    EXPECT_FALSE(loom_value_facts_is_exact(widened));
    EXPECT_FALSE(loom_value_facts_is_not_subnormal(widened));
    EXPECT_TRUE(loom_value_facts_is_cluster_uniform(widened));
    double lo = 0.0;
    double hi = 0.0;
    EXPECT_EQ(loom_value_facts_as_float_range(LOOM_SCALAR_TYPE_F64, widened,
                                              &lo, &hi),
              iteration < 2);
    if (iteration < 2) {
      EXPECT_DOUBLE_EQ(lo, std::numeric_limits<double>::denorm_min());
      EXPECT_DOUBLE_EQ(hi, 1.0);
    }

    IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
        &table, nullptr, type, &table, previous, &table,
        loom_value_facts_known_nan(), iteration, &widened));
    EXPECT_FALSE(loom_value_facts_is_finite(widened));
    EXPECT_FALSE(loom_value_facts_is_not_nan(widened));
    EXPECT_TRUE(loom_value_facts_is_not_inf(widened));

    IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
        &table, nullptr, type, &table, previous, &table,
        loom_value_facts_unknown(), iteration, &widened));
    EXPECT_TRUE(loom_value_facts_is_unknown(widened));
  }
}

TEST_F(FactExtensionsTest, TypedWidenRetainsStableFloatInterval) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  auto previous =
      loom_value_facts_make_float_range(LOOM_SCALAR_TYPE_F32, -8.0, 8.0);
  auto next = previous;
  loom_value_facts_mark_workgroup_uniform(&previous);
  loom_value_facts_mark_lane_varying(&next);

  loom_value_facts_t widened;
  IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
      &table, nullptr, type, &table, previous, &table, next,
      /*iteration=*/7, &widened));
  double lo = 0.0;
  double hi = 0.0;
  ASSERT_TRUE(
      loom_value_facts_as_float_range(LOOM_SCALAR_TYPE_F32, widened, &lo, &hi));
  EXPECT_DOUBLE_EQ(lo, -8.0);
  EXPECT_DOUBLE_EQ(hi, 8.0);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(widened));
}

TEST_F(FactExtensionsTest, TypedWidenAdvancesEndpointsThroughCarrierDomains) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  const auto check = [&](loom_value_facts_t previous, loom_value_facts_t next,
                         int64_t expected_lo, int64_t expected_hi,
                         int64_t expected_divisor) {
    for (uint32_t iteration : {2u, 7u}) {
      loom_value_facts_t widened;
      IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
          &table, nullptr, loom_type_scalar(LOOM_SCALAR_TYPE_I64), &table,
          previous, &table, next, iteration, &widened));
      EXPECT_EQ(widened.range_lo, expected_lo);
      EXPECT_EQ(widened.range_hi, expected_hi);
      EXPECT_EQ(widened.known_divisor, expected_divisor);
    }
  };
  check(loom_value_facts_make(4, 36, 4), loom_value_facts_make(4, 68, 4), 4,
        UINT32_MAX, 4);
  check(loom_value_facts_make(-36, -4, 4), loom_value_facts_make(-68, -4, 4),
        INT32_MIN, -4, 4);
  check(loom_value_facts_make(-24, 24, 12), loom_value_facts_make(-42, 42, 6),
        INT32_MIN, INT32_MAX, 6);
  check(loom_value_facts_exact_i64(24), loom_value_facts_exact_i64(35), 24,
        UINT32_MAX, 1);
  check(loom_value_facts_exact_i64(0), loom_value_facts_exact_i64(16), 0,
        UINT32_MAX, 16);
  check(loom_value_facts_exact_i64(16), loom_value_facts_exact_i64(0), 0, 16,
        16);
  check(loom_value_facts_make(4, 36, 4), loom_value_facts_unknown(), INT64_MIN,
        INT64_MAX, 1);
  check(loom_value_facts_unknown(), loom_value_facts_make(4, 36, 4), INT64_MIN,
        INT64_MAX, 1);
  check(loom_value_facts_make(-24, 24, 12), loom_value_facts_make(-12, 12, 6),
        -24, 24, 6);
}

TEST_F(FactExtensionsTest, TypedWidenCarrierDomainFitsLoopSolveBudget) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const struct {
    loom_value_facts_t next;
    loom_value_facts_t expected;
  } steps[] = {
      {loom_value_facts_make(0, 1, 1), loom_value_facts_make(0, 1, 1)},
      {loom_value_facts_make(-1, 1, 1), loom_value_facts_make(INT32_MIN, 1, 1)},
      {loom_value_facts_make((int64_t)INT32_MIN - 1, 1, 1),
       loom_value_facts_make(INT64_MIN, 1, 1)},
      {loom_value_facts_make(INT64_MIN, 2, 1),
       loom_value_facts_make(INT64_MIN, INT32_MAX, 1)},
      {loom_value_facts_make(INT64_MIN, (int64_t)INT32_MAX + 1, 1),
       loom_value_facts_make(INT64_MIN, INT64_MAX, 1)},
  };
  loom_value_facts_t previous = loom_value_facts_exact_i64(1);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(steps); ++i) {
    loom_value_facts_t widened;
    IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
        &table, nullptr, type, &table, previous, &table, steps[i].next,
        /*iteration=*/static_cast<uint32_t>(2 + i), &widened));
    EXPECT_TRUE(loom_value_facts_equal(widened, steps[i].expected));
    previous = widened;
  }
  loom_value_facts_t converged;
  IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
      &table, nullptr, type, &table, previous, &table, previous,
      /*iteration=*/7, &converged));
  EXPECT_TRUE(loom_value_facts_equal(converged, previous));
}

TEST_F(FactExtensionsTest, TypedWidenRetainsIntegerTypeDomains) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  const struct {
    // Scalar domain whose expanding endpoints are checked.
    loom_scalar_type_t type;
    // Inclusive lower bound of the scalar domain.
    int64_t lower;
    // Inclusive upper bound of the scalar domain.
    int64_t upper;
  } domains[] = {
      {LOOM_SCALAR_TYPE_I1, 0, 1},
      {LOOM_SCALAR_TYPE_I8, INT8_MIN, INT8_MAX},
      {LOOM_SCALAR_TYPE_I16, INT16_MIN, INT16_MAX},
      {LOOM_SCALAR_TYPE_I32, INT32_MIN, INT32_MAX},
      {LOOM_SCALAR_TYPE_I64, INT64_MIN, INT64_MAX},
      {LOOM_SCALAR_TYPE_INDEX, INT64_MIN, INT64_MAX},
      {LOOM_SCALAR_TYPE_OFFSET, 0, INT64_MAX},
  };
  for (const auto& domain : domains) {
    SCOPED_TRACE(loom_scalar_type_name(domain.type));
    const auto type = loom_type_scalar(domain.type);
    for (uint32_t iteration : {0u, 1u, 2u, 7u}) {
      loom_value_facts_t widened;
      IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
          &table, nullptr, type, &table, loom_value_facts_exact_i64(0), &table,
          loom_value_facts_exact_i64(1), iteration, &widened));
      EXPECT_EQ(widened.range_lo, 0);
      EXPECT_EQ(widened.range_hi,
                iteration < 2
                    ? 1
                    : (domain.upper < UINT32_MAX ? domain.upper : UINT32_MAX));
      IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
          &table, nullptr, type, &table, loom_value_facts_exact_i64(1), &table,
          loom_value_facts_exact_i64(0), iteration, &widened));
      EXPECT_EQ(widened.range_lo, 0);
      EXPECT_EQ(widened.range_hi, 1);
    }
    loom_value_facts_t widened;
    IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
        &table, nullptr, type, &table, loom_value_facts_unknown(), &table,
        loom_value_facts_make(0, 1, 1), /*iteration=*/2, &widened));
    EXPECT_EQ(widened.range_lo, domain.lower);
    EXPECT_EQ(widened.range_hi, domain.upper);
  }
}

TEST_F(FactExtensionsTest, TypedWidenPreservesRangeAcrossMetadataChanges) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  auto previous = loom_value_facts_make(0, 24, 8);
  auto next = loom_value_facts_make(0, 24, 6);
  loom_value_facts_mark_workgroup_uniform(&previous);
  loom_value_facts_mark_lane_varying(&next);
  loom_value_facts_t widened;
  IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
      &table, nullptr, loom_type_scalar(LOOM_SCALAR_TYPE_I32), &table, previous,
      &table, next,
      /*iteration=*/2, &widened));
  EXPECT_EQ(widened.range_lo, 0);
  EXPECT_EQ(widened.range_hi, 24);
  EXPECT_EQ(widened.known_divisor, 2);
  EXPECT_TRUE(loom_value_facts_is_non_negative(widened));
  EXPECT_TRUE(loom_value_facts_is_lane_varying(widened));
}

TEST_F(FactExtensionsTest, TypedWidenPreservesStableExtension) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t extension = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(3), &extension));
  loom_value_facts_t previous = loom_value_facts_exact_i64(4);
  previous.extension_id = extension.extension_id;
  loom_value_facts_t next = loom_value_facts_exact_i64(8);
  next.extension_id = extension.extension_id;

  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I64, loom_dim_pack_static(4), 0);
  loom_value_facts_t widened = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_fact_table_widen_for_type(
      &table, nullptr, vector_type, &table, previous, &table, next,
      /*iteration=*/2, &widened));

  EXPECT_EQ(widened.range_lo, 4);
  EXPECT_EQ(widened.range_hi, UINT32_MAX);
  EXPECT_EQ(widened.known_divisor, 4);
  loom_value_fact_uniform_element_t uniform = {};
  EXPECT_TRUE(loom_value_facts_query_uniform_element(&table.context, widened,
                                                     &uniform));
  EXPECT_EQ(uniform.element.range_lo, 3);
}

}  // namespace
}  // namespace loom
