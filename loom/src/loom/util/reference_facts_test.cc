// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/reference_facts.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(ReferenceFactsTest, FreshnessRequiresTheSameInvocation) {
  const loom_value_fact_reference_origin_t origins[] = {
      {},
      {1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, 10},
      {1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, 11},
      {1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, LOOM_VALUE_ID_INVALID},
      {1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION,
       LOOM_VALUE_ID_INVALID},
      {1, 1, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, 12},
      {1, 1, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION,
       LOOM_VALUE_ID_INVALID},
      {2, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, 13},
      {2, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION,
       LOOM_VALUE_ID_INVALID},
  };
  // Each row lists the origins proved disjoint from that row. All other pairs
  // remain conservative, including two unknown or two allocated origins.
  const uint16_t disjoint_masks[] = {
      0,      1 << 4, 1 << 4, 1 << 4, (1 << 1) | (1 << 2) | (1 << 3),
      1 << 6, 1 << 5, 1 << 8, 1 << 7,
  };
  for (size_t i = 0; i < IREE_ARRAYSIZE(origins); ++i) {
    for (size_t j = 0; j < IREE_ARRAYSIZE(origins); ++j) {
      EXPECT_EQ(loom_value_fact_reference_origins_are_disjoint(origins[i],
                                                               origins[j]),
                (disjoint_masks[i] & (1 << j)) != 0)
          << "origin pair " << i << ", " << j;
    }
  }
}

TEST(ReferenceFactsTest, JoinsKeepCommonEntryGuarantees) {
  const loom_value_fact_reference_origin_t first = {
      1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, 10};
  const loom_value_fact_reference_origin_t second = {
      1, 0, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY, 11};
  auto joined = loom_value_fact_reference_origin_meet(first, second);
  EXPECT_EQ(joined.kind, LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY);
  EXPECT_EQ(joined.entry_value_id, LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(joined.function_symbol_id, 1);
  EXPECT_TRUE(loom_value_fact_reference_origin_equal(
      loom_value_fact_reference_origin_meet(joined, first), joined));
  EXPECT_TRUE(loom_value_fact_reference_origin_equal(
      loom_value_fact_reference_origin_meet(first, first), first));

  auto allocation = first;
  allocation.kind = LOOM_VALUE_FACT_REFERENCE_ORIGIN_ALLOCATION;
  allocation.entry_value_id = LOOM_VALUE_ID_INVALID;
  EXPECT_EQ(loom_value_fact_reference_origin_meet(first, allocation).kind,
            LOOM_VALUE_FACT_REFERENCE_ORIGIN_UNKNOWN);
  auto foreign = first;
  foreign.function_symbol_id = 2;
  EXPECT_EQ(loom_value_fact_reference_origin_meet(first, foreign).kind,
            LOOM_VALUE_FACT_REFERENCE_ORIGIN_UNKNOWN);
}

TEST(ReferenceFactsTest, RestrictIndependenceRequiresOneSharedDomain) {
  const loom_value_fact_restrict_access_t accesses[] = {
      {}, {1, 0}, {1, 2}, {1, 3}, {2, 2},
  };
  const uint8_t independent_masks[] = {
      0, (1u << 2) | (1u << 3), (1u << 1) | (1u << 3), (1u << 1) | (1u << 2), 0,
  };
  for (size_t i = 0; i < IREE_ARRAYSIZE(accesses); ++i) {
    for (size_t j = 0; j < IREE_ARRAYSIZE(accesses); ++j) {
      EXPECT_EQ(loom_value_fact_restrict_accesses_are_independent(accesses[i],
                                                                  accesses[j]),
                (independent_masks[i] & (1u << j)) != 0)
          << "restrict access pair " << i << ", " << j;
    }
  }
}

}  // namespace
}  // namespace loom
