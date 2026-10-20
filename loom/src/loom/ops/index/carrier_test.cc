// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/carrier.h"

#include "iree/testing/gtest.h"
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

}  // namespace
}  // namespace loom
