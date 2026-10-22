// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/carrier.h"

#include "iree/testing/gtest.h"
#include "loom/ops/index/compare.h"
#include "loom/ops/index/ops.h"
#include "loom/target/facts.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

TEST(IndexCarrierTest, ReadsTypedTargetWidths) {
  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.index_bitwidth = 32;
  target_facts.storage.snapshot.offset_bitwidth = 64;
  loom_fact_context_t context = {.target_facts = &target_facts};

  EXPECT_EQ(
      loom_index_target_carrier_bitwidth(&context, LOOM_SCALAR_TYPE_INDEX), 32);
  EXPECT_EQ(
      loom_index_target_carrier_bitwidth(&context, LOOM_SCALAR_TYPE_OFFSET),
      64);
}

TEST(IndexCarrierTest, DistinguishesTargetlessAndInvalidCarriers) {
  loom_fact_context_t context = {};
  EXPECT_EQ(
      loom_index_target_carrier_bitwidth(&context, LOOM_SCALAR_TYPE_INDEX), 0);
  EXPECT_EQ(loom_index_target_carrier_bitwidth(&context, LOOM_SCALAR_TYPE_I32),
            -1);

  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.index_bitwidth = 128;
  context.target_facts = &target_facts;
  EXPECT_EQ(
      loom_index_target_carrier_bitwidth(&context, LOOM_SCALAR_TYPE_INDEX), -1);
}

TEST(IndexCarrierTest, DescribesSelectedIntegerCarrierDomains) {
  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.index_bitwidth = 32;
  target_facts.storage.snapshot.offset_bitwidth = 32;
  loom_fact_context_t context = {.target_facts = &target_facts};

  loom_value_facts_t index_facts =
      loom_index_integer_carrier_domain_facts(&context, LOOM_SCALAR_TYPE_INDEX);
  EXPECT_EQ(index_facts.range_lo, INT32_MIN);
  EXPECT_EQ(index_facts.range_hi, INT32_MAX);

  loom_value_facts_t offset_facts = loom_index_integer_carrier_domain_facts(
      &context, LOOM_SCALAR_TYPE_OFFSET);
  EXPECT_EQ(offset_facts.range_lo, 0);
  EXPECT_EQ(offset_facts.range_hi, (int64_t)UINT32_MAX);

  loom_value_facts_t i16_facts =
      loom_index_integer_carrier_domain_facts(&context, LOOM_SCALAR_TYPE_I16);
  EXPECT_EQ(i16_facts.range_lo, INT16_MIN);
  EXPECT_EQ(i16_facts.range_hi, INT16_MAX);

  loom_value_facts_t targetless_index =
      loom_index_integer_carrier_domain_facts(nullptr, LOOM_SCALAR_TYPE_INDEX);
  EXPECT_TRUE(loom_value_facts_is_unknown(targetless_index));
}

TEST(IndexCarrierTest, WrapsOffsetFactsAtSelectedWidth) {
  loom_target_facts_t target_facts = {};
  loom_fact_context_t context = {.target_facts = &target_facts};
  const loom_value_facts_t negative_one = loom_value_facts_exact_i64(-1);

  target_facts.storage.snapshot.offset_bitwidth = 32;
  loom_value_facts_t narrow =
      loom_index_wrap_offset_facts(&context, negative_one);
  EXPECT_TRUE(loom_value_facts_is_exact(narrow));
  EXPECT_EQ(narrow.range_lo, (int64_t)UINT32_MAX);

  target_facts.storage.snapshot.offset_bitwidth = 64;
  loom_value_facts_t wide =
      loom_index_wrap_offset_facts(&context, negative_one);
  EXPECT_EQ(wide.range_lo, 0);
  EXPECT_EQ(wide.range_hi, INT64_MAX);
  EXPECT_FALSE(loom_value_facts_is_exact(wide));

  loom_value_facts_t targetless =
      loom_index_wrap_offset_facts(nullptr, negative_one);
  EXPECT_TRUE(loom_value_facts_equal(targetless, wide));
}

TEST(IndexCarrierTest, UnsignedComparisonsMatchCarrierBits) {
  constexpr uint8_t predicates[] = {
      LOOM_INDEX_CMP_PREDICATE_ULT, LOOM_INDEX_CMP_PREDICATE_ULE,
      LOOM_INDEX_CMP_PREDICATE_UGT, LOOM_INDEX_CMP_PREDICATE_UGE};
  constexpr int64_t values[] = {
      INT64_MIN,  INT32_MIN, -1, 0, 1, INT32_MAX, int64_t(INT32_MAX) + 1,
      UINT32_MAX, INT64_MAX};
  for (uint8_t width : {32, 64}) {
    loom_target_facts_t target_facts = {};
    target_facts.storage.snapshot.index_bitwidth = width;
    target_facts.storage.snapshot.offset_bitwidth = width;
    loom_fact_context_t context = {
        .target_facts = width ? &target_facts : nullptr,
    };
    const uint64_t mask = width == 32 ? UINT32_MAX : UINT64_MAX;
    for (auto type : {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET}) {
      for (int64_t left : values) {
        if (width == 32 && (left < INT32_MIN || left > int64_t(UINT32_MAX))) {
          continue;
        }
        const auto lhs = loom_value_facts_exact_i64(left);
        for (int64_t right : values) {
          if (width == 32 &&
              (right < INT32_MIN || right > int64_t(UINT32_MAX))) {
            continue;
          }
          const auto rhs = loom_value_facts_exact_i64(right);
          const uint64_t left_bits = uint64_t(left) & mask;
          const uint64_t right_bits = uint64_t(right) & mask;
          const bool expected[] = {
              (left_bits < right_bits), (left_bits <= right_bits),
              (left_bits > right_bits), (left_bits >= right_bits)};
          for (size_t i = 0; i < IREE_ARRAYSIZE(predicates); ++i) {
            SCOPED_TRACE(::testing::Message()
                         << "width " << unsigned(width) << ", type " << type
                         << ", " << left << ", " << right << ", predicate "
                         << unsigned(predicates[i]));
            bool result = false;
            ASSERT_TRUE(loom_index_cmp_result_from_facts(
                &context, type, predicates[i], &lhs, &rhs, &result));
            EXPECT_EQ(result, expected[i]);
          }
        }
      }
    }
  }
}

TEST(IndexCarrierTest, UnsignedEndpointProofsRetainUnknownCarriers) {
  const auto zero = loom_value_facts_exact_i64(0);
  const auto maximum = loom_value_facts_exact_i64(-1);
  const loom_value_facts_t ranges[] = {
      loom_value_facts_unknown(), loom_value_facts_make(-4, 4, 1),
      loom_value_facts_make(0, UINT32_MAX, 1),
      loom_value_facts_make(INT32_MIN, INT32_MAX, 1),
      loom_value_facts_make(int64_t(UINT32_MAX) + 1, INT64_MAX, 1)};
  for (uint8_t width : {0, 32, 64}) {
    loom_target_facts_t target_facts = {};
    target_facts.storage.snapshot.index_bitwidth = width;
    target_facts.storage.snapshot.offset_bitwidth = width;
    loom_fact_context_t context = {
        .target_facts = width ? &target_facts : nullptr,
    };
    for (auto type : {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET}) {
      for (const auto& range : ranges) {
        const struct {
          // Endpoint on the right side of the comparison.
          loom_value_facts_t endpoint;
          // Unsigned ordering to prove against the endpoint.
          uint8_t predicate;
          // Universal result, regardless of the incoming range.
          bool expected;
        } cases[] = {{zero, LOOM_INDEX_CMP_PREDICATE_ULT, false},
                     {zero, LOOM_INDEX_CMP_PREDICATE_UGE, true},
                     {maximum, LOOM_INDEX_CMP_PREDICATE_UGT, false},
                     {maximum, LOOM_INDEX_CMP_PREDICATE_ULE, true}};
        for (const auto& test : cases) {
          bool result = false;
          ASSERT_TRUE(loom_index_cmp_result_from_facts(
              &context, type, test.predicate, &range, &test.endpoint, &result));
          EXPECT_EQ(result, test.expected);
        }
      }
    }
  }
}

TEST(IndexCarrierTest, UnsignedIntervalsPreserveTheFullPositiveEnvelope) {
  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.index_bitwidth = 32;
  target_facts.storage.snapshot.offset_bitwidth = 32;
  loom_fact_context_t context = {.target_facts = &target_facts};
  const auto prefix = loom_value_facts_make(0, int64_t(INT32_MAX) + 1, 1);
  const auto high =
      loom_value_facts_make(int64_t(INT32_MAX) + 2, UINT32_MAX, 1);
  const auto signed_high = loom_value_facts_make(INT32_MIN + 2, -1, 1);
  for (auto type : {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET}) {
    for (const auto& suffix : {high, signed_high}) {
      bool result = false;
      ASSERT_TRUE(loom_index_cmp_result_from_facts(&context, type,
                                                   LOOM_INDEX_CMP_PREDICATE_ULT,
                                                   &prefix, &suffix, &result));
      EXPECT_TRUE(result);
      ASSERT_TRUE(loom_index_cmp_result_from_facts(&context, type,
                                                   LOOM_INDEX_CMP_PREDICATE_UGE,
                                                   &suffix, &prefix, &result));
      EXPECT_TRUE(result);
    }
  }
}

TEST(IndexCarrierTest, TargetlessUnsignedOrderWaitsForCarrierWidth) {
  const auto maximum = loom_value_facts_exact_i64(-1);
  const auto positive = loom_value_facts_exact_i64(UINT32_MAX);
  bool result = false;
  EXPECT_FALSE(loom_index_cmp_result_from_facts(nullptr, LOOM_SCALAR_TYPE_INDEX,
                                                LOOM_INDEX_CMP_PREDICATE_UGT,
                                                &maximum, &positive, &result));
  EXPECT_FALSE(loom_index_cmp_result_from_facts(nullptr, LOOM_SCALAR_TYPE_INDEX,
                                                LOOM_INDEX_CMP_PREDICATE_ULT,
                                                &positive, &maximum, &result));
  ASSERT_TRUE(loom_index_cmp_result_from_facts(nullptr, LOOM_SCALAR_TYPE_INDEX,
                                               LOOM_INDEX_CMP_PREDICATE_UGE,
                                               &maximum, &positive, &result));
  EXPECT_TRUE(result);
}

TEST(IndexCarrierTest, OutOfCarrierRangesDoNotProveRelativeOrder) {
  loom_target_facts_t target_facts = {};
  target_facts.storage.snapshot.index_bitwidth = 32;
  target_facts.storage.snapshot.offset_bitwidth = 32;
  loom_fact_context_t context = {.target_facts = &target_facts};
  const auto outside = loom_value_facts_exact_i64(int64_t(UINT32_MAX) + 1);
  const auto one = loom_value_facts_exact_i64(1);
  bool result = false;
  for (auto type : {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET}) {
    EXPECT_FALSE(loom_index_cmp_result_from_facts(
        &context, type, LOOM_INDEX_CMP_PREDICATE_ULT, &outside, &one, &result));
    EXPECT_FALSE(loom_index_cmp_result_from_facts(
        &context, type, LOOM_INDEX_CMP_PREDICATE_ULT, &one, &outside, &result));
  }
}

}  // namespace
}  // namespace loom
