// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/memory_access.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace loom {
namespace {

static loom_low_memory_access_summary_t MakeStridedSummary(
    uint32_t alias_root_id, uint64_t stride_bytes, uint64_t begin_bytes,
    uint64_t end_bytes) {
  return (loom_low_memory_access_summary_t){
      /*.memory_space=*/LOOM_LOW_MEMORY_SPACE_WORKGROUP,
      /*.alias_root_id=*/alias_root_id,
      /*.alias_group_id=*/LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      /*.precision_flags=*/LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE |
          LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
          LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL,
      /*.strided_interval=*/
      {
          /*.stride_bytes=*/stride_bytes,
          /*.begin_bytes=*/begin_bytes,
          /*.end_bytes=*/end_bytes,
      },
  };
}

static loom_low_memory_access_summary_t MakeIntervalSummary(
    loom_low_byte_interval_t* interval, uint32_t alias_root_id,
    int64_t begin_bytes, int64_t end_bytes) {
  *interval = (loom_low_byte_interval_t){
      /*.begin_facts=*/loom_value_facts_make(begin_bytes, begin_bytes, 1),
      /*.end_facts=*/loom_value_facts_make(end_bytes, end_bytes, 1),
      /*.begin_expr_id=*/LOOM_LOW_MEMORY_EXPR_ID_NONE,
      /*.end_expr_id=*/LOOM_LOW_MEMORY_EXPR_ID_NONE,
      /*.precision_flags=*/LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE |
          LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE,
  };
  return (loom_low_memory_access_summary_t){
      /*.memory_space=*/LOOM_LOW_MEMORY_SPACE_WORKGROUP,
      /*.alias_root_id=*/alias_root_id,
      /*.alias_group_id=*/LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      /*.precision_flags=*/LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE |
          LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT |
          LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL,
      /*.strided_interval=*/{},
      /*.byte_interval=*/interval,
  };
}

TEST(MemoryAccessTest, StridedSlotsWithinOneRootAreDisjoint) {
  const loom_low_memory_access_summary_t slot0 =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/0, /*end_bytes=*/16);
  const loom_low_memory_access_summary_t slot1 =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/16, /*end_bytes=*/32);
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(&slot0, &slot1));
}

TEST(MemoryAccessTest, OverlappingStridedSlotsMayAlias) {
  const loom_low_memory_access_summary_t slot0 =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/0, /*end_bytes=*/16);
  const loom_low_memory_access_summary_t overlap =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/8, /*end_bytes=*/24);
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(&slot0, &overlap));
}

TEST(MemoryAccessTest, EqualResiduesMayAliasAcrossStrideInstances) {
  const loom_low_memory_access_summary_t left =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/0, /*end_bytes=*/16);
  const loom_low_memory_access_summary_t right = left;
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(&left, &right));
}

TEST(MemoryAccessTest, StridedProofRequiresComparableRootAndStride) {
  loom_low_memory_access_summary_t no_root =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/0, /*end_bytes=*/16);
  no_root.alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE;
  no_root.precision_flags &= ~LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  const loom_low_memory_access_summary_t slot1 =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/64,
                         /*begin_bytes=*/16, /*end_bytes=*/32);
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(&no_root, &slot1));

  const loom_low_memory_access_summary_t other_stride =
      MakeStridedSummary(/*alias_root_id=*/7, /*stride_bytes=*/128,
                         /*begin_bytes=*/16, /*end_bytes=*/32);
  EXPECT_TRUE(
      loom_low_memory_access_summaries_may_alias(&slot1, &other_stride));
}

TEST(MemoryAccessTest, IntervalEnvelopeRequiresComparableRoot) {
  loom_low_byte_interval_t left_interval;
  loom_low_byte_interval_t right_interval;
  loom_low_memory_access_summary_t left =
      MakeIntervalSummary(&left_interval, /*alias_root_id=*/11,
                          /*begin_bytes=*/0,
                          /*end_bytes=*/16);
  loom_low_memory_access_summary_t right =
      MakeIntervalSummary(&right_interval, /*alias_root_id=*/11,
                          /*begin_bytes=*/32,
                          /*end_bytes=*/48);
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(&left, &right));

  left.alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE;
  left.precision_flags &= ~LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  right.alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE;
  right.precision_flags &= ~LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(&left, &right));
}

}  // namespace
}  // namespace loom
