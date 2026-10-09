// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/relocation_group.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(LowAllocationRelocationGroupTest, RetainsCoalescedTransportClosure) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  loom_low_reg_class_t reg_classes[2] = {};
  reg_classes[0].alias_set_id = 1;
  reg_classes[0].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL;
  reg_classes[1].alias_set_id = 2;
  reg_classes[1].flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL;
  loom_low_descriptor_set_t descriptors = {
      .reg_classes = reg_classes,
      .reg_class_count = IREE_ARRAYSIZE(reg_classes)};

  // Reversed assignment order exercises the value-to-assignment join. Values
  // 0..6 share coalesced subranges; 7 is an edge destination, 8 is a real copy,
  // and 9 names a separate storage class despite using the same location.
  const uint32_t bases[] = {8, 10, 11, 11, 11, 8, 8, 8, 0, 8};
  const uint32_t counts[] = {4, 2, 1, 1, 1, 4, 2, 4, 1, 4};
  loom_low_allocation_assignment_t assignments[10] = {};
  uint32_t indices[10];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(assignments); ++i) {
    indices[i] = IREE_ARRAYSIZE(assignments) - i - 1;
    auto& assignment = assignments[indices[i]];
    assignment.value_id = i;
    assignment.unit_count = counts[i];
    assignment.descriptor_reg_class_id = i == 9 ? 1 : 0;
    assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    assignment.location_base = bases[i];
    assignment.location_count = counts[i];
  }
  loom_liveness_analysis_t liveness = {.value_count =
                                           IREE_ARRAYSIZE(assignments)};
  loom_low_allocation_assignment_map_t map = {
      .liveness = &liveness,
      .assignments = assignments,
      .assignment_count = IREE_ARRAYSIZE(assignments),
      .assignment_indices_by_value_ordinal = indices};

  const auto relation = [](uint32_t result, uint32_t source,
                           uint32_t result_offset, uint32_t source_offset,
                           uint32_t count, loom_low_placement_cause_t cause) {
    loom_low_placement_relation_t row = {};
    row.result_ordinal = result;
    row.source_ordinal = source;
    row.result_unit_offset = result_offset;
    row.source_unit_offset = source_offset;
    row.unit_count = count;
    row.kind = cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE
                   ? LOOM_LOW_PLACEMENT_RELATION_SUBRANGE
               : cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT
                   ? LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART
                   : LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    row.cause = cause;
    row.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE |
                (cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT
                     ? LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD
                     : LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED);
    return row;
  };
  loom_low_placement_relation_t relations[] = {
      relation(1, 0, 0, 2, 2, LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE),
      relation(2, 1, 0, 1, 1, LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE),
      relation(3, 2, 0, 0, 1, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT),
      relation(4, 3, 0, 0, 1, LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY),
      relation(5, 1, 2, 0, 2, LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT),
      relation(5, 6, 0, 0, 2, LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT),
      relation(7, 5, 0, 0, 4, LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH),
      relation(8, 4, 0, 0, 1, LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY),
      relation(9, 0, 0, 0, 4, LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY),
  };
  loom_low_placement_table_t placement = {
      .value_count = IREE_ARRAYSIZE(assignments),
      .relations = relations,
      .relation_count = IREE_ARRAYSIZE(relations)};
  loom_low_allocation_relocation_groups_t groups = {};
  IREE_ASSERT_OK(loom_low_allocation_relocation_groups_initialize(
      &descriptors, &placement, &map, &arena, &groups));
  for (uint32_t i = 0; i < 7; ++i) {
    EXPECT_EQ(groups.representatives[indices[0]],
              groups.representatives[indices[i]]);
  }
  for (uint32_t i = 7; i < 10; ++i) {
    EXPECT_EQ(groups.representatives[indices[i]], indices[i]);
    EXPECT_EQ(groups.next_members[indices[i]], indices[i]);
  }
  bool visited[10] = {};
  uint32_t member = indices[2];
  for (uint32_t i = 0; i < 7; ++i) {
    EXPECT_FALSE(visited[member]);
    visited[member] = true;
    member = groups.next_members[member];
  }
  EXPECT_EQ(member, indices[2]);
  for (uint32_t i = 0; i < 7; ++i) {
    EXPECT_TRUE(visited[indices[i]]);
  }

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

}  // namespace
}  // namespace loom
