// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/destructive_reuse.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/allocation/unit_liveness_builder.h"

namespace loom {
namespace {

class DestructiveReuseTest : public ::testing::Test {
 protected:
  static loom_low_placement_relation_t Relation(
      loom_value_ordinal_t result, loom_value_ordinal_t source,
      uint32_t result_offset, uint32_t count,
      loom_low_placement_cause_t cause) {
    loom_low_placement_relation_t row = {};
    row.result_ordinal = result;
    row.source_ordinal = source;
    row.result_unit_offset = result_offset;
    row.unit_count = count;
    row.kind = cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT
                   ? LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART
                   : LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    row.cause = cause;
    row.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE |
                (cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT
                     ? LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD
                     : LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED);
    return row;
  }

  static loom_liveness_interval_t Interval(loom_value_id_t value_id,
                                           uint32_t start_point,
                                           uint32_t end_point,
                                           uint32_t unit_count) {
    loom_liveness_interval_t interval = {};
    interval.value_id = value_id;
    interval.start_point = start_point;
    interval.end_point = end_point;
    interval.unit_count = unit_count;
    return interval;
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &decision_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&decision_arena_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Two source words form a tuple, a fact identity forwards the tuple, and a
  // destructive result consumes that identity at point 4. Ordinals deliberately
  // put the identity and result before their sources, as CFG layout can do.
  void Refine(uint32_t low_end, uint32_t high_end, uint32_t write_point,
              bool low_live_at_write = true) {
    loom_low_allocation_unit_liveness_value_t unit_values[] = {
        {0, 0}, {2, 0}, {4, 0}, {6, 0}, {7, 0}};
    uint32_t unit_start_points[] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t unit_ends[] = {write_point, write_point,     6,
                            6,           write_point - 1, write_point - 1,
                            low_end,     high_end};
    uint64_t incomplete_segment_words[] = {0};
    loom_low_allocation_unit_liveness_t units = {};
    units.values = unit_values;
    units.start_points = unit_start_points;
    units.end_points = unit_ends;
    units.point_count = IREE_ARRAYSIZE(unit_ends);
    units.values_with_incomplete_storage_segments = {
        /*.bit_count=*/IREE_ARRAYSIZE(unit_values),
        /*.words=*/incomplete_segment_words,
    };

    const loom_liveness_segment_t segments[] = {
        {/*.start_point=*/0,
         /*.end_point=*/low_live_at_write ? low_end : write_point},
        {/*.start_point=*/write_point + 1, /*.end_point=*/low_end},
        {/*.start_point=*/0, /*.end_point=*/high_end},
    };
    const loom_liveness_segment_range_t segment_ranges[] = {
        {},
        {},
        {},
        {/*.start=*/0, /*.count=*/low_live_at_write ? 1u : 2u},
        {/*.start=*/2, /*.count=*/1},
    };
    const loom_liveness_interval_t intervals[] = {
        Interval(/*value_id=*/0, /*start_point=*/0, /*end_point=*/write_point,
                 /*unit_count=*/2),
        Interval(/*value_id=*/1, /*start_point=*/0, /*end_point=*/6,
                 /*unit_count=*/2),
        Interval(/*value_id=*/2, /*start_point=*/0,
                 /*end_point=*/write_point - 1, /*unit_count=*/2),
        Interval(/*value_id=*/3, /*start_point=*/0, /*end_point=*/low_end,
                 /*unit_count=*/1),
        Interval(/*value_id=*/4, /*start_point=*/0, /*end_point=*/high_end,
                 /*unit_count=*/1),
    };
    const uint32_t interval_indices[] = {0, 1, 2, 3, 4};
    loom_liveness_analysis_t liveness = {};
    liveness.intervals = intervals;
    liveness.interval_count = IREE_ARRAYSIZE(intervals);
    liveness.value_interval_indices = interval_indices;
    liveness.segments = segments;
    liveness.segment_count = IREE_ARRAYSIZE(segments);
    liveness.value_segment_ranges = segment_ranges;
    liveness.value_count = IREE_ARRAYSIZE(segment_ranges);
    units.storage_segments.entries = segments;

    relations_[0] = Relation(0, 2, 0, 2, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT);
    relations_[1] = Relation(1, 0, 0, 2, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT);
    relations_[1].flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE;
    relations_[1].write_point = write_point;
    relations_[2] = Relation(2, 3, 0, 1, LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT);
    relations_[3] = Relation(2, 4, 1, 1, LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT);
    const loom_low_placement_relation_range_t ranges[] = {
        {0, 1}, {1, 1}, {2, 2}, {4, 0}, {4, 0}};
    const loom_value_ordinal_t storage_order[] = {1, 0, 2, 3, 4};
    const loom_value_ordinal_t tied_origins[] = {2, 2, 2, 3, 4};
    const uint32_t write_relations[] = {1};
    loom_low_placement_table_t placement = {};
    placement.relations = relations_;
    placement.relation_count = IREE_ARRAYSIZE(relations_);
    placement.value_count = IREE_ARRAYSIZE(ranges);
    placement.ranges_by_result_ordinal = ranges;
    placement.storage_value_order = storage_order;
    placement.storage_value_order_count = IREE_ARRAYSIZE(storage_order);
    placement.tied_storage_origins_by_value_ordinal = tied_origins;
    placement.storage = {write_relations, IREE_ARRAYSIZE(write_relations),
                         LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES |
                             LOOM_LOW_PLACEMENT_STORAGE_FLAG_IDENTITY_ALIASES};
    IREE_ASSERT_OK(loom_low_allocation_unit_liveness_retain_tied_storage(
        &units, &liveness, &placement, &arena_, &decision_arena_));
    IREE_ASSERT_OK(loom_low_allocation_refine_destructive_reuse(
        &units, &liveness, &placement, &arena_));
  }

  // Owns scratch blocks reused by each refinement.
  iree_arena_block_pool_t pool_;
  // Owns physical segment arrays and refinement scratch.
  iree_arena_allocator_t arena_;
  // Owns query metadata used only while making allocation decisions.
  iree_arena_allocator_t decision_arena_;
  // Retains permissions for the identity, write, and two borrowed words.
  loom_low_placement_relation_t relations_[4];
};

TEST_F(DestructiveReuseTest, PreservesOnlyTheLiveComponent) {
  Refine(/*low_end=*/7, /*high_end=*/2, /*write_point=*/4);
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations_[0]));
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations_[1]));
  EXPECT_FALSE(loom_low_placement_relation_can_alias(&relations_[2]));
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations_[3]));
}

TEST_F(DestructiveReuseTest, KeepsIdentityUsesBeforeTheWriteCoalescible) {
  Refine(/*low_end=*/4, /*high_end=*/4, /*write_point=*/4);
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations_[2]));
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations_[3]));
}

TEST_F(DestructiveReuseTest, IgnoresObservationsOnDisjointPaths) {
  Refine(/*low_end=*/7, /*high_end=*/2, /*write_point=*/4,
         /*low_live_at_write=*/false);
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations_[2]));
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations_[3]));
}

TEST_F(DestructiveReuseTest, PreservesRequiredTiedFamilyObservations) {
  loom_low_allocation_unit_liveness_value_t unit_values[] = {
      {0, 0}, {1, 0}, {2, 0}, {3, 6}};
  uint32_t unit_start_points[] = {0, 0, 0, 6};
  uint32_t unit_ends[] = {4, 5, 4, 7};
  uint64_t incomplete_segment_words[] = {0};
  loom_low_allocation_unit_liveness_t units = {};
  units.values = unit_values;
  units.start_points = unit_start_points;
  units.end_points = unit_ends;
  units.point_count = IREE_ARRAYSIZE(unit_ends);
  units.values_with_incomplete_storage_segments = {
      /*.bit_count=*/IREE_ARRAYSIZE(unit_values),
      /*.words=*/incomplete_segment_words,
  };

  const loom_liveness_segment_t segments[] = {
      {/*.start_point=*/0, /*.end_point=*/4},
      {/*.start_point=*/6, /*.end_point=*/7},
  };
  const loom_liveness_segment_range_t segment_ranges[] = {
      {}, {}, {/*.start=*/0, /*.count=*/1}, {/*.start=*/1, /*.count=*/1}};
  const loom_liveness_interval_t intervals[] = {
      Interval(/*value_id=*/0, /*start_point=*/0, /*end_point=*/4,
               /*unit_count=*/1),
      Interval(/*value_id=*/1, /*start_point=*/0, /*end_point=*/5,
               /*unit_count=*/1),
      Interval(/*value_id=*/2, /*start_point=*/0, /*end_point=*/4,
               /*unit_count=*/1),
      Interval(/*value_id=*/3, /*start_point=*/6, /*end_point=*/7,
               /*unit_count=*/1),
  };
  const uint32_t interval_indices[] = {0, 1, 2, 3};
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);
  liveness.value_interval_indices = interval_indices;
  liveness.segments = segments;
  liveness.segment_count = IREE_ARRAYSIZE(segments);
  liveness.value_segment_ranges = segment_ranges;
  liveness.value_count = IREE_ARRAYSIZE(segment_ranges);
  units.storage_segments.entries = segments;

  loom_low_placement_relation_t relations[] = {
      Relation(0, 2, 0, 1, LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY),
      Relation(1, 0, 0, 1, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT),
      Relation(3, 2, 0, 1, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT),
  };
  relations[1].flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE;
  relations[1].write_point = 4;
  const loom_low_placement_relation_range_t ranges[] = {
      {0, 1}, {1, 1}, {2, 0}, {2, 1}};
  const loom_value_ordinal_t storage_order[] = {1, 3, 0, 2};
  const loom_value_ordinal_t tied_origins[] = {0, 0, 2, 2};
  const uint32_t write_relations[] = {1};
  loom_low_placement_table_t placement = {};
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  placement.value_count = IREE_ARRAYSIZE(ranges);
  placement.ranges_by_result_ordinal = ranges;
  placement.storage_value_order = storage_order;
  placement.storage_value_order_count = IREE_ARRAYSIZE(storage_order);
  placement.tied_storage_origins_by_value_ordinal = tied_origins;
  placement.storage = {write_relations, IREE_ARRAYSIZE(write_relations),
                       LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES |
                           LOOM_LOW_PLACEMENT_STORAGE_FLAG_IDENTITY_ALIASES};

  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_retain_tied_storage(
      &units, &liveness, &placement, &arena_, &decision_arena_));
  IREE_ASSERT_OK(loom_low_allocation_refine_destructive_reuse(
      &units, &liveness, &placement, &arena_));
  EXPECT_FALSE(loom_low_placement_relation_can_alias(&relations[0]));
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations[1]));
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations[2]));
}

TEST_F(DestructiveReuseTest, PreservesMappedResultUnitAcrossSourceWrite) {
  loom_low_allocation_unit_liveness_value_t unit_values[] = {
      {0, 0}, {2, 1}, {3, 2}, {5, 4}};
  uint32_t unit_start_points[] = {0, 0, 1, 2, 2, 4};
  uint32_t unit_ends[] = {2, 2, 4, 3, 3, 5};
  uint64_t incomplete_segment_words[] = {0};
  loom_low_allocation_unit_liveness_t units = {};
  units.values = unit_values;
  units.start_points = unit_start_points;
  units.end_points = unit_ends;
  units.point_count = IREE_ARRAYSIZE(unit_ends);
  units.values_with_incomplete_storage_segments = {
      /*.bit_count=*/IREE_ARRAYSIZE(unit_values),
      /*.words=*/incomplete_segment_words,
  };

  const loom_liveness_interval_t intervals[] = {
      Interval(/*value_id=*/0, /*start_point=*/0, /*end_point=*/2,
               /*unit_count=*/2),
      Interval(/*value_id=*/1, /*start_point=*/1, /*end_point=*/4,
               /*unit_count=*/1),
      Interval(/*value_id=*/2, /*start_point=*/2, /*end_point=*/3,
               /*unit_count=*/2),
      Interval(/*value_id=*/3, /*start_point=*/4, /*end_point=*/5,
               /*unit_count=*/1),
  };
  const uint32_t interval_indices[] = {0, 1, 2, 3};
  const loom_liveness_segment_range_t segment_ranges[] = {{}, {}, {}, {}};
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);
  liveness.value_interval_indices = interval_indices;
  liveness.value_segment_ranges = segment_ranges;
  liveness.value_count = IREE_ARRAYSIZE(intervals);

  loom_low_placement_relation_t relations[] = {
      Relation(1, 0, 0, 1, LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE),
      Relation(2, 0, 0, 2, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT),
      Relation(3, 1, 0, 1, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT),
  };
  relations[0].source_unit_offset = 1;
  relations[1].flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE;
  relations[1].write_point = 2;
  const loom_low_placement_relation_range_t ranges[] = {
      {0, 0}, {0, 1}, {1, 1}, {2, 1}};
  const loom_value_ordinal_t storage_order[] = {2, 3, 1, 0};
  const loom_value_ordinal_t tied_origins[] = {0, 1, 0, 1};
  const uint32_t write_relations[] = {1};
  loom_low_placement_table_t placement = {};
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  placement.value_count = IREE_ARRAYSIZE(ranges);
  placement.ranges_by_result_ordinal = ranges;
  placement.storage_value_order = storage_order;
  placement.storage_value_order_count = IREE_ARRAYSIZE(storage_order);
  placement.tied_storage_origins_by_value_ordinal = tied_origins;
  placement.storage = {write_relations, IREE_ARRAYSIZE(write_relations),
                       LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES |
                           LOOM_LOW_PLACEMENT_STORAGE_FLAG_IDENTITY_ALIASES};

  IREE_ASSERT_OK(loom_low_allocation_unit_liveness_retain_tied_storage(
      &units, &liveness, &placement, &arena_, &decision_arena_));
  IREE_ASSERT_OK(loom_low_allocation_refine_destructive_reuse(
      &units, &liveness, &placement, &arena_));
  EXPECT_FALSE(loom_low_placement_relation_can_alias(&relations[0]));
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations[1]));
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relations[2]));
}

TEST_F(DestructiveReuseTest, UsesTheAcceptedScheduleWritePoint) {
  Refine(/*low_end=*/4, /*high_end=*/4, /*write_point=*/3);
  EXPECT_FALSE(loom_low_placement_relation_can_alias(&relations_[2]));
  EXPECT_FALSE(loom_low_placement_relation_can_alias(&relations_[3]));
}

}  // namespace
}  // namespace loom
