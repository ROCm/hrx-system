// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/testing/allocation_checker.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/placement.h"

namespace loom {
namespace {

loom_liveness_interval_t MakeInterval(loom_value_id_t value_id,
                                      uint32_t start_point, uint32_t end_point,
                                      loom_liveness_value_class_t value_class) {
  loom_liveness_interval_t interval = {};
  interval.value_id = value_id;
  interval.start_point = start_point;
  interval.end_point = end_point;
  interval.value_class = value_class;
  interval.unit_count = 1;
  return interval;
}

loom_low_allocation_assignment_t MakeAssignment(
    loom_value_id_t value_id, uint32_t start_point, uint32_t end_point,
    uint32_t location_base, uint32_t unit_point_start,
    loom_liveness_value_class_t value_class) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.descriptor_reg_class_id = value_class.register_class_id;
  assignment.start_point = start_point;
  assignment.end_point = end_point;
  assignment.unit_count = 1;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_base = location_base;
  assignment.location_count = 1;
  assignment.unit_point_start = unit_point_start;
  return assignment;
}

loom_low_placement_relation_t MakeAliasRelation(
    loom_value_ordinal_t result_ordinal, loom_value_ordinal_t source_ordinal,
    loom_low_placement_relation_flags_t flags) {
  loom_low_placement_relation_t relation = {};
  relation.result_ordinal = result_ordinal;
  relation.source_ordinal = source_ordinal;
  relation.unit_count = 1;
  relation.kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
  relation.flags = flags | LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  return relation;
}

class AllocationCheckerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);

    reg_class_.alloc_unit_bits = 32;
    reg_class_.allocatable_count = 8;
    descriptor_set_.stable_id = 1;
    descriptor_set_.reg_classes = &reg_class_;
    descriptor_set_.reg_class_count = 1;

    value_class_.type_kind = LOOM_TYPE_REGISTER;
    value_class_.register_descriptor_set_stable_id = descriptor_set_.stable_id;
    value_class_.register_class_id = 0;
    for (uint32_t i = 0; i < 3; ++i) {
      value_ids_[i] = i + 1;
      interval_indices_[i] = i;
      assignment_indices_[i] = i;
      intervals_[i] = MakeInterval(value_ids_[i], 0, 4, value_class_);
      assignments_[i] = MakeAssignment(value_ids_[i], 0, 4, i, i, value_class_);
      unit_start_points_[i] = 0;
      unit_end_points_[i] = 4;
    }

    frame_.target.descriptor_set = &descriptor_set_;
    frame_.schedule.target.descriptor_set = &descriptor_set_;
    frame_.schedule.value_ids = value_ids_;
    frame_.schedule.value_count = 2;
    frame_.allocation.target.descriptor_set = &descriptor_set_;
    frame_.allocation.liveness.intervals = intervals_;
    frame_.allocation.liveness.interval_count = 2;
    frame_.allocation.liveness.value_ids = value_ids_;
    frame_.allocation.liveness.value_count = 2;
    frame_.allocation.liveness.value_interval_indices = interval_indices_;
    frame_.allocation.placement.value_ids = value_ids_;
    frame_.allocation.placement.value_count = 2;
    frame_.allocation.assignments = assignments_;
    frame_.allocation.assignment_count = 2;
    frame_.allocation.assignment_indices_by_value_ordinal = assignment_indices_;
    frame_.allocation.unit_start_points = unit_start_points_;
    frame_.allocation.unit_end_points = unit_end_points_;
    frame_.allocation.unit_point_count = 2;
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_low_allocation_check_result_t Check() {
    loom_low_allocation_check_result_t result = {};
    IREE_EXPECT_OK(loom_low_allocation_check_frame(&frame_, &arena_, &result));
    return result;
  }

  void ConfigureTiedCopies() {
    // a is overwritten by b, then b is copied to c before c is overwritten
    // by d. The allocator reserves a's storage through b and c's through d.
    static constexpr uint32_t kStarts[] = {0, 1, 2, 4};
    static constexpr uint32_t kEnds[] = {1, 3, 4, 5};
    static constexpr uint32_t kStorageEnds[] = {3, 3, 5, 5};
    for (uint32_t i = 0; i < 4; ++i) {
      value_ids_[i] = i + 1;
      interval_indices_[i] = assignment_indices_[i] = i;
      intervals_[i] =
          MakeInterval(value_ids_[i], kStarts[i], kEnds[i], value_class_);
      assignments_[i] = MakeAssignment(value_ids_[i], kStarts[i],
                                       kStorageEnds[i], 0, i, value_class_);
      unit_start_points_[i] = kStarts[i];
      unit_end_points_[i] = kStorageEnds[i];
    }
    frame_.schedule.value_count = 4;
    frame_.allocation.liveness.interval_count = 4;
    frame_.allocation.liveness.value_count = 4;
    frame_.allocation.placement.value_count = 4;
    frame_.allocation.assignment_count = 4;
    frame_.allocation.unit_point_count = 4;
    relations_[0] =
        MakeAliasRelation(1, 0, LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD);
    relations_[0].cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
    relations_[1] =
        MakeAliasRelation(2, 1, LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED);
    relations_[1].cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY;
    relations_[2] =
        MakeAliasRelation(3, 2, LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD);
    relations_[2].cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
    frame_.allocation.placement.relations = relations_;
    frame_.allocation.placement.relation_count = 3;
  }

  void ConfigureThreeValues() {
    frame_.schedule.value_count = 3;
    frame_.allocation.liveness.interval_count = 3;
    frame_.allocation.liveness.value_count = 3;
    frame_.allocation.placement.value_count = 3;
    frame_.allocation.assignment_count = 3;
    frame_.allocation.unit_point_count = 3;
  }

  void ConfigureRefinedReservation(uint32_t temporary_end_point) {
    intervals_[0] = MakeInterval(value_ids_[0], /*start_point=*/2,
                                 /*end_point=*/4, value_class_);
    intervals_[0].unit_count = 2;
    assignments_[0] = MakeAssignment(
        value_ids_[0], /*start_point=*/0, /*end_point=*/4,
        /*location_base=*/0, /*unit_point_start=*/0, value_class_);
    assignments_[0].unit_count = 2;
    assignments_[0].location_count = 2;
    assignments_[0].flags =
        LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
    unit_start_points_[0] = 0;
    unit_start_points_[1] = 2;
    unit_end_points_[0] = 4;
    unit_end_points_[1] = 4;

    intervals_[1] = MakeInterval(value_ids_[1], /*start_point=*/0,
                                 temporary_end_point, value_class_);
    assignments_[1] = MakeAssignment(
        value_ids_[1], /*start_point=*/0, temporary_end_point,
        /*location_base=*/1, /*unit_point_start=*/2, value_class_);
    unit_start_points_[2] = 0;
    unit_end_points_[2] = temporary_end_point;
    frame_.allocation.unit_point_count = 3;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_low_reg_class_t reg_class_ = {};
  loom_low_descriptor_set_t descriptor_set_ = {};
  loom_liveness_value_class_t value_class_ = {};
  loom_value_id_t value_ids_[4] = {};
  uint32_t interval_indices_[4] = {};
  uint32_t assignment_indices_[4] = {};
  loom_liveness_interval_t intervals_[4] = {};
  loom_low_allocation_assignment_t assignments_[4] = {};
  uint32_t unit_start_points_[4] = {};
  uint32_t unit_end_points_[4] = {};
  loom_low_placement_relation_t relations_[3] = {};
  loom_low_emission_frame_t frame_ = {};
};

TEST_F(AllocationCheckerTest, AcceptsDisjointAssignments) {
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_EQ(result.violation_count, 0u);
}

TEST_F(AllocationCheckerTest, AcceptsAcquisitionBeforeTiedSourceDefinition) {
  // Region layout puts the tied descendant before its dominating source. The
  // source owns storage from the descendant's start, not just its own start.
  intervals_[0].start_point = 3;
  assignments_[1].location_base = assignments_[0].location_base;
  loom_low_placement_relation_t relation =
      MakeAliasRelation(1, 0, LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD);
  relation.cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
  frame_.allocation.placement.relations = &relation;
  frame_.allocation.placement.relation_count = 1;
  EXPECT_EQ(Check().violation_count, 0u);
}

TEST_F(AllocationCheckerTest, RejectsAcquisitionAfterSemanticStart) {
  assignments_[0].start_point = 1;
  const auto result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_SHAPE);
}

TEST_F(AllocationCheckerTest, EarlyAcquisitionConflictsBeforeDefinition) {
  intervals_[0].start_point = 2;
  intervals_[1].end_point = assignments_[1].end_point = unit_end_points_[1] = 2;
  assignments_[1].location_base = assignments_[0].location_base;
  const auto result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
}

TEST_F(AllocationCheckerTest, RejectsOverlappingLiveAssignments) {
  assignments_[1].location_base = assignments_[0].location_base;
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
  EXPECT_EQ(result.first_violation.value_id, value_ids_[0]);
  EXPECT_EQ(result.first_violation.related_value_id, value_ids_[1]);
}

TEST_F(AllocationCheckerTest, IntersectsUnsegmentedUnitsWithSparseLifetimes) {
  loom_liveness_segment_t segments[] = {{0, 2}, {5, 7}};
  frame_.allocation.liveness.segments = segments;
  frame_.allocation.storage_segments = segments;
  frame_.allocation.liveness.segment_count = IREE_ARRAYSIZE(segments);
  for (uint32_t sparse_ordinal = 0; sparse_ordinal < 2; ++sparse_ordinal) {
    const uint32_t contiguous_ordinal = 1 - sparse_ordinal;
    for (uint32_t begin = 0; begin < 7; ++begin) {
      for (uint32_t end = begin + 1; end <= 7; ++end) {
        SCOPED_TRACE(::testing::Message()
                     << "sparse_ordinal=" << sparse_ordinal
                     << " begin=" << begin << " end=" << end);
        intervals_[sparse_ordinal] =
            MakeInterval(value_ids_[sparse_ordinal], 0, 7, value_class_);
        assignments_[sparse_ordinal] = MakeAssignment(
            value_ids_[sparse_ordinal], 0, 7, 0, sparse_ordinal, value_class_);
        assignments_[sparse_ordinal].liveness_segments = {0, 2};
        unit_start_points_[sparse_ordinal] = 0;
        unit_end_points_[sparse_ordinal] = 7;
        intervals_[contiguous_ordinal] = MakeInterval(
            value_ids_[contiguous_ordinal], begin, end, value_class_);
        assignments_[contiguous_ordinal] =
            MakeAssignment(value_ids_[contiguous_ordinal], begin, end, 0,
                           contiguous_ordinal, value_class_);
        unit_start_points_[contiguous_ordinal] = begin;
        unit_end_points_[contiguous_ordinal] = end;
        const bool overlaps = begin < 2 || end > 5;
        const loom_low_allocation_check_result_t result = Check();
        EXPECT_EQ(result.violation_count, overlaps ? 1u : 0u);
        if (overlaps) {
          EXPECT_EQ(result.first_violation.kind,
                    LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
        }
      }
    }
  }
}

TEST_F(AllocationCheckerTest, ClipsSparseLifetimesAtRefinedUnitStarts) {
  loom_liveness_segment_t segments[] = {{0, 2}, {5, 7}, {1, 4}};
  frame_.allocation.liveness.segments = segments;
  frame_.allocation.storage_segments = segments;
  frame_.allocation.liveness.segment_count = IREE_ARRAYSIZE(segments);
  for (uint32_t sparse_ordinal = 0; sparse_ordinal < 2; ++sparse_ordinal) {
    const uint32_t other_ordinal = 1 - sparse_ordinal;
    intervals_[sparse_ordinal] =
        MakeInterval(value_ids_[sparse_ordinal], 5, 7, value_class_);
    intervals_[sparse_ordinal].unit_count = 2;
    assignments_[sparse_ordinal] =
        MakeAssignment(value_ids_[sparse_ordinal], 0, 7, 0, 0, value_class_);
    assignments_[sparse_ordinal].unit_count = 2;
    assignments_[sparse_ordinal].location_count = 2;
    assignments_[sparse_ordinal].flags =
        LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
    assignments_[sparse_ordinal].liveness_segments = {0, 2};
    unit_start_points_[0] = 0;
    unit_end_points_[0] = 7;
    unit_end_points_[1] = 7;
    intervals_[other_ordinal] =
        MakeInterval(value_ids_[other_ordinal], 1, 4, value_class_);
    assignments_[other_ordinal] =
        MakeAssignment(value_ids_[other_ordinal], 1, 4, 1, 2, value_class_);
    assignments_[other_ordinal].liveness_segments = {2, 1};
    unit_start_points_[2] = 1;
    unit_end_points_[2] = 4;
    frame_.allocation.unit_point_count = 3;
    for (uint32_t start = 0; start <= 5; ++start) {
      SCOPED_TRACE(::testing::Message() << "sparse_ordinal=" << sparse_ordinal
                                        << " refined_start=" << start);
      unit_start_points_[1] = start;
      const loom_low_allocation_check_result_t result = Check();
      EXPECT_EQ(result.violation_count, start < 2 ? 1u : 0u);
    }
  }
}

TEST_F(AllocationCheckerTest, AcceptsReuseEndingAtFutureUnitStart) {
  ConfigureRefinedReservation(/*temporary_end_point=*/2);
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_EQ(result.violation_count, 0u);
}

TEST_F(AllocationCheckerTest, RejectsReuseOverlappingFutureUnitStart) {
  ConfigureRefinedReservation(/*temporary_end_point=*/3);
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
}

TEST_F(AllocationCheckerTest, AcceptsExplicitStorageAlias) {
  assignments_[1].location_base = assignments_[0].location_base;
  loom_low_placement_relation_t relation =
      MakeAliasRelation(1, 0, LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD);
  frame_.allocation.placement.relations = &relation;
  frame_.allocation.placement.relation_count = 1;
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_EQ(result.violation_count, 0u);
}

TEST_F(AllocationCheckerTest, AcceptsCopyOfDestructiveSuccessor) {
  ConfigureTiedCopies();
  EXPECT_EQ(Check().violation_count, 0u);
}

TEST_F(AllocationCheckerTest, RejectsOverwriteOfLiveCopiedValue) {
  for (uint32_t live_end : {4u, 5u}) {
    SCOPED_TRACE(live_end);
    ConfigureTiedCopies();
    // b can die at d's write, but not remain live afterward. Mandatory
    // storage identities alone cannot justify overwriting b's old bits.
    intervals_[1].end_point = live_end;
    assignments_[0].end_point = unit_end_points_[0] = live_end;
    assignments_[1].end_point = unit_end_points_[1] = live_end;
    const auto result = Check();
    if (live_end == 4) {
      EXPECT_EQ(result.violation_count, 0u);
    } else {
      EXPECT_GT(result.violation_count, 0u);
      EXPECT_EQ(result.first_violation.kind,
                LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
    }
  }
}

TEST_F(AllocationCheckerTest, RejectsUnrelatedStorageReuseThroughTiedChain) {
  ConfigureTiedCopies();
  // Remove the copy linking the two mandatory storage identities.
  relations_[1] = relations_[2];
  frame_.allocation.placement.relation_count = 2;
  const auto result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
}

TEST_F(AllocationCheckerTest, RejectsOverwriteOfLiveCopySource) {
  ConfigureTiedCopies();
  // c copies a before b overwrites a, and still needs those old bits afterward.
  relations_[1].source_ordinal = 0;
  intervals_[0].end_point = 2;
  assignments_[0].end_point = unit_end_points_[0] = 4;
  intervals_[1].start_point = assignments_[1].start_point =
      unit_start_points_[1] = 2;
  intervals_[1].end_point = assignments_[1].end_point = unit_end_points_[1] = 4;
  intervals_[2].start_point = assignments_[2].start_point =
      unit_start_points_[2] = 1;
  const auto result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
}

TEST_F(AllocationCheckerTest, HardAliasesPreserveSubrangeOffsets) {
  for (uint32_t copy_location : {0u, 1u}) {
    SCOPED_TRACE(copy_location);
    ConfigureTiedCopies();
    frame_.schedule.value_count = 3;
    frame_.allocation.liveness.interval_count = 3;
    frame_.allocation.liveness.value_count = 3;
    frame_.allocation.placement.value_count = 3;
    frame_.allocation.assignment_count = 3;
    frame_.allocation.placement.relation_count = 2;
    intervals_[0].unit_count = 2;
    intervals_[0].end_point = 4;
    assignments_[0].unit_count = assignments_[0].location_count = 2;
    assignments_[0].end_point = 4;
    assignments_[1].location_base = 1;
    assignments_[1].unit_point_start = 2;
    assignments_[2].location_base = copy_location;
    assignments_[2].unit_point_start = 3;
    assignments_[2].end_point = 4;
    unit_start_points_[1] = 0;
    unit_start_points_[2] = 1;
    unit_start_points_[3] = 2;
    unit_end_points_[0] = unit_end_points_[3] = 4;
    unit_end_points_[1] = unit_end_points_[2] = 3;
    relations_[0].kind = LOOM_LOW_PLACEMENT_RELATION_SUBRANGE;
    relations_[0].source_unit_offset = 1;
    // Only a's second unit is overwritten by b and copied into c. The first
    // unit remains independently live and cannot share c's location.
    const auto result = Check();
    if (copy_location == 1) {
      EXPECT_EQ(result.violation_count, 0u);
    } else {
      EXPECT_GT(result.violation_count, 0u);
      EXPECT_EQ(result.first_violation.kind,
                LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
    }
  }
}

TEST_F(AllocationCheckerTest, AcceptsCopiesOfSharedContents) {
  ConfigureThreeValues();
  assignments_[1].location_base = assignments_[0].location_base;
  assignments_[2].location_base = assignments_[0].location_base;
  loom_low_placement_relation_t relations[] = {
      MakeAliasRelation(1, 0, LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED),
      MakeAliasRelation(2, 0, LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED),
  };
  for (auto& relation : relations) {
    relation.cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY;
  }
  frame_.allocation.placement.relations = relations;
  frame_.allocation.placement.relation_count = IREE_ARRAYSIZE(relations);
  EXPECT_EQ(Check().violation_count, 0u);

  // A chain carries the same contents even when relation rows are reversed.
  relations[1].source_ordinal = 1;
  const auto first = relations[0];
  relations[0] = relations[1];
  relations[1] = first;
  EXPECT_EQ(Check().violation_count, 0u);
}

TEST_F(AllocationCheckerTest, StorageTiesDoNotImplyTransitiveContentIdentity) {
  ConfigureThreeValues();
  // The copy is defined at one and overwritten at two while the original
  // remains live. Its storage reservation includes the successor's lifetime.
  for (uint32_t i = 1; i < 3; ++i) {
    intervals_[i].start_point = assignments_[i].start_point =
        unit_start_points_[i] = i;
  }
  intervals_[1].end_point = 2;
  assignments_[1].location_base = assignments_[0].location_base;
  assignments_[2].location_base = assignments_[0].location_base;
  loom_low_placement_relation_t relations[] = {
      MakeAliasRelation(1, 0, LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED),
      MakeAliasRelation(2, 1, LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD),
  };
  relations[0].cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY;
  relations[1].cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
  frame_.allocation.placement.relations = relations;
  frame_.allocation.placement.relation_count = IREE_ARRAYSIZE(relations);
  const loom_low_allocation_check_result_t result = Check();
  // Both the extended copy reservation and the successor conflict with the
  // original after the destructive write.
  EXPECT_EQ(result.violation_count, 2u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
  EXPECT_EQ(result.first_violation.value_id, value_ids_[0]);
  EXPECT_EQ(result.first_violation.related_value_id, value_ids_[1]);
}

TEST_F(AllocationCheckerTest, DistinguishesContentsAtDifferentSourceOffsets) {
  ConfigureThreeValues();
  intervals_[0].unit_count = 2;
  assignments_[0].unit_count = 2;
  assignments_[0].location_count = 2;
  assignments_[0].location_base = 2;
  assignments_[1].unit_point_start = 2;
  assignments_[1].location_base = 0;
  assignments_[2].unit_point_start = 3;
  assignments_[2].location_base = 0;
  unit_end_points_[3] = 4;
  frame_.allocation.unit_point_count = 4;
  loom_low_placement_relation_t relations[] = {
      MakeAliasRelation(1, 0, LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED),
      MakeAliasRelation(2, 0, LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED),
  };
  for (auto& relation : relations) {
    relation.kind = LOOM_LOW_PLACEMENT_RELATION_SUBRANGE;
    relation.cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE;
  }
  relations[1].source_unit_offset = 1;
  frame_.allocation.placement.relations = relations;
  frame_.allocation.placement.relation_count = IREE_ARRAYSIZE(relations);
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_EQ(result.violation_count, 1u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT);
  EXPECT_EQ(result.first_violation.value_id, value_ids_[1]);
  EXPECT_EQ(result.first_violation.related_value_id, value_ids_[2]);

  relations[1].source_unit_offset = 0;
  EXPECT_EQ(Check().violation_count, 0u);
}

TEST_F(AllocationCheckerTest, RejectsFixedLocationMismatch) {
  loom_low_allocation_resolved_fixed_value_t fixed = {};
  fixed.value_ordinal = 0;
  fixed.assignment.value_id = value_ids_[0];
  fixed.assignment.descriptor_reg_class_id = 0;
  fixed.assignment.location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  fixed.assignment.location_base = 7;
  fixed.assignment.location_count = 1;
  frame_.allocation.fixed_values = &fixed;
  frame_.allocation.fixed_value_count = 1;
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_FIXED_LOCATION);
}

TEST_F(AllocationCheckerTest, RejectsReservedLocationOverlap) {
  loom_low_allocation_resolved_reserved_range_t reserved = {};
  reserved.descriptor_reg_class_id = 0;
  reserved.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  reserved.location_base = assignments_[1].location_base;
  reserved.location_count = 1;
  frame_.allocation.reserved_ranges = &reserved;
  frame_.allocation.reserved_range_count = 1;
  const loom_low_allocation_check_result_t result = Check();
  EXPECT_GT(result.violation_count, 0u);
  EXPECT_EQ(result.first_violation.kind,
            LOOM_LOW_ALLOCATION_CHECK_VIOLATION_RESERVED_LOCATION);
}

}  // namespace
}  // namespace loom
