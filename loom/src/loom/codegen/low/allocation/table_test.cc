// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/table.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(LowAllocationTableTest, MapsAssignmentsByValueOrdinal) {
  const loom_value_id_t value_ids[] = {10, 11, 12};
  const uint32_t assignment_indices[] = {1, UINT32_MAX, 0};
  loom_low_allocation_assignment_t assignments[2] = {};
  assignments[0].value_id = 12;
  assignments[1].value_id = 10;

  loom_low_allocation_table_t table = {};
  table.liveness.value_ids = value_ids;
  table.liveness.value_count = IREE_ARRAYSIZE(value_ids);
  table.assignments = assignments;
  table.assignment_count = IREE_ARRAYSIZE(assignments);
  table.assignment_indices_by_value_ordinal = assignment_indices;

  uint32_t assignment_index = 99;
  EXPECT_EQ(&assignments[1],
            loom_low_allocation_assignment_for_value_ordinal(
                &table, /*value_ordinal=*/0, &assignment_index));
  EXPECT_EQ(assignment_index, 1u);

  assignment_index = 99;
  EXPECT_EQ(nullptr, loom_low_allocation_assignment_for_value_ordinal(
                         &table, /*value_ordinal=*/1, &assignment_index));
  EXPECT_EQ(assignment_index, UINT32_MAX);

  assignment_index = 99;
  EXPECT_EQ(nullptr, loom_low_allocation_assignment_for_value_ordinal(
                         &table, /*value_ordinal=*/5, &assignment_index));
  EXPECT_EQ(assignment_index, UINT32_MAX);
}

TEST(LowAllocationTableTest, FindsEdgeCopyGroupsBySourceOrdinal) {
  loom_low_allocation_edge_copy_group_t groups[3] = {};
  groups[0].source_ordinal = 2;
  groups[1].source_ordinal = 7;
  groups[2].source_ordinal = 11;

  loom_low_allocation_table_t table = {};
  table.edge_copy_groups = groups;
  table.edge_copy_group_count = IREE_ARRAYSIZE(groups);

  EXPECT_EQ(&groups[0],
            loom_low_allocation_find_edge_copy_group_by_source_ordinal(
                &table, /*source_ordinal=*/2));
  EXPECT_EQ(&groups[1],
            loom_low_allocation_find_edge_copy_group_by_source_ordinal(
                &table, /*source_ordinal=*/7));
  EXPECT_EQ(&groups[2],
            loom_low_allocation_find_edge_copy_group_by_source_ordinal(
                &table, /*source_ordinal=*/11));
  EXPECT_EQ(nullptr, loom_low_allocation_find_edge_copy_group_by_source_ordinal(
                         &table, /*source_ordinal=*/8));
}

TEST(LowAllocationTableTest, FindsPacketMoveTemporaryGroupsBySourceOrdinal) {
  loom_low_allocation_packet_move_temporary_group_t groups[3] = {};
  groups[0].source_ordinal = 3;
  groups[1].source_ordinal = 5;
  groups[2].source_ordinal = 13;

  loom_low_allocation_table_t table = {};
  table.packet_move_temporary_groups = groups;
  table.packet_move_temporary_group_count = IREE_ARRAYSIZE(groups);

  EXPECT_EQ(
      &groups[0],
      loom_low_allocation_find_packet_move_temporary_group_by_source_ordinal(
          &table, /*source_ordinal=*/3));
  EXPECT_EQ(
      &groups[1],
      loom_low_allocation_find_packet_move_temporary_group_by_source_ordinal(
          &table, /*source_ordinal=*/5));
  EXPECT_EQ(
      &groups[2],
      loom_low_allocation_find_packet_move_temporary_group_by_source_ordinal(
          &table, /*source_ordinal=*/13));
  EXPECT_EQ(
      nullptr,
      loom_low_allocation_find_packet_move_temporary_group_by_source_ordinal(
          &table, /*source_ordinal=*/12));
}

}  // namespace
}  // namespace loom
