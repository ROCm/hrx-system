// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage_liveness_index.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class LowAllocationStorageLivenessIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static loom_low_allocation_assignment_t Assignment(
      uint16_t reg_class_id, uint32_t location, uint32_t unit_count,
      uint32_t start_point, uint32_t end_point, uint32_t unit_point_start,
      loom_low_allocation_location_kind_t location_kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    loom_low_allocation_assignment_t assignment = {
        .descriptor_reg_class_id = reg_class_id,
        .flags = LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS,
        .start_point = start_point,
        .end_point = end_point,
        .unit_count = unit_count,
        .location_kind = location_kind,
        .location_base = location,
        .location_count = unit_count,
        .unit_point_start = unit_point_start};
    return assignment;
  }

  static loom_low_move_location_t Location(
      uint16_t reg_class_id, uint32_t location,
      loom_low_allocation_location_kind_t location_kind =
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return loom_low_move_location_t{
        .location_kind = static_cast<uint16_t>(location_kind),
        .descriptor_reg_class_id = reg_class_id,
        .location = location,
    };
  }

  static loom_low_descriptor_set_t DescriptorSet(
      const loom_low_reg_class_t* reg_classes,
      iree_host_size_t reg_class_count) {
    loom_low_descriptor_set_t descriptor_set = {};
    descriptor_set.reg_classes = reg_classes;
    descriptor_set.reg_class_count = reg_class_count;
    return descriptor_set;
  }

  static loom_low_allocation_unit_liveness_t UnitLiveness(
      uint32_t* start_points, uint32_t* end_points,
      iree_host_size_t point_count) {
    loom_low_allocation_unit_liveness_t unit_liveness = {
        .start_points = start_points,
        .end_points = end_points,
        .point_count = point_count};
    return unit_liveness;
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(LowAllocationStorageLivenessIndexTest,
       IndexesRefinedUnitsAndHalfOpenBounds) {
  const loom_low_reg_class_t reg_classes[1] = {};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  const loom_low_allocation_assignment_t assignment =
      Assignment(/*reg_class_id=*/0, /*location=*/4, /*unit_count=*/2,
                 /*start_point=*/2, /*end_point=*/9,
                 /*unit_point_start=*/0);
  uint32_t unit_start_points[] = {2, 6};
  uint32_t unit_end_points[] = {5, 9};
  const loom_low_allocation_unit_liveness_t unit_liveness = UnitLiveness(
      unit_start_points, unit_end_points, IREE_ARRAYSIZE(unit_end_points));

  loom_low_allocation_storage_liveness_index_t index;
  IREE_ASSERT_OK(loom_low_allocation_storage_liveness_index_initialize(
      &descriptor_set, &assignment, /*assignment_count=*/1, &unit_liveness,
      &arena_, &index));

  const loom_low_move_location_t first =
      Location(/*reg_class_id=*/0, /*location=*/4);
  const loom_low_move_location_t second =
      Location(/*reg_class_id=*/0, /*location=*/5);
  const loom_low_move_location_t outside =
      Location(/*reg_class_id=*/0, /*location=*/6);
  EXPECT_FALSE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &first, /*point=*/1));
  EXPECT_TRUE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &first, /*point=*/2));
  EXPECT_TRUE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &first, /*point=*/4));
  EXPECT_FALSE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &first, /*point=*/5));
  EXPECT_FALSE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &second, /*point=*/5));
  EXPECT_TRUE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &second, /*point=*/6));
  EXPECT_FALSE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &second, /*point=*/9));
  EXPECT_FALSE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &outside, /*point=*/4));
}

TEST_F(LowAllocationStorageLivenessIndexTest,
       SortsReusedStorageIndependentlyOfAssignmentOrder) {
  loom_low_reg_class_t reg_classes[3] = {};
  reg_classes[0].alias_set_id = 7;
  reg_classes[1].alias_set_id = 7;
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  const loom_low_allocation_assignment_t assignments[] = {
      Assignment(/*reg_class_id=*/2, /*location=*/10, /*unit_count=*/1,
                 /*start_point=*/0, /*end_point=*/14,
                 /*unit_point_start=*/0),
      Assignment(/*reg_class_id=*/0, /*location=*/10, /*unit_count=*/1,
                 /*start_point=*/10, /*end_point=*/12,
                 /*unit_point_start=*/1),
      Assignment(/*reg_class_id=*/0, /*location=*/10, /*unit_count=*/1,
                 /*start_point=*/2, /*end_point=*/4,
                 /*unit_point_start=*/2),
      Assignment(/*reg_class_id=*/0, /*location=*/10, /*unit_count=*/1,
                 /*start_point=*/6, /*end_point=*/8,
                 /*unit_point_start=*/3),
      Assignment(/*reg_class_id=*/0, /*location=*/10, /*unit_count=*/1,
                 /*start_point=*/3, /*end_point=*/11,
                 /*unit_point_start=*/4,
                 /*location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID),
  };
  uint32_t unit_start_points[] = {0, 10, 2, 6, 3};
  uint32_t unit_end_points[] = {14, 12, 4, 8, 11};
  const loom_low_allocation_unit_liveness_t unit_liveness = UnitLiveness(
      unit_start_points, unit_end_points, IREE_ARRAYSIZE(unit_end_points));

  loom_low_allocation_storage_liveness_index_t index;
  IREE_ASSERT_OK(loom_low_allocation_storage_liveness_index_initialize(
      &descriptor_set, assignments, IREE_ARRAYSIZE(assignments), &unit_liveness,
      &arena_, &index));

  const loom_low_move_location_t alias =
      Location(/*reg_class_id=*/1, /*location=*/10);
  for (uint32_t point = 0; point < 14; ++point) {
    SCOPED_TRACE(point);
    const bool expected = (point >= 2 && point < 4) ||
                          (point >= 6 && point < 8) ||
                          (point >= 10 && point < 12);
    EXPECT_EQ(loom_low_allocation_storage_liveness_index_is_live_at_point(
                  &index, &alias, point),
              expected);
  }
  const loom_low_move_location_t target_id = Location(
      /*reg_class_id=*/1, /*location=*/10,
      LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID);
  EXPECT_FALSE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &target_id, /*point=*/2));
  EXPECT_TRUE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &target_id, /*point=*/3));
  EXPECT_TRUE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &target_id, /*point=*/10));
  EXPECT_FALSE(loom_low_allocation_storage_liveness_index_is_live_at_point(
      &index, &target_id, /*point=*/11));
}

TEST_F(LowAllocationStorageLivenessIndexTest,
       PreservesLongReservationsAcrossShorterAliasingSegments) {
  loom_low_reg_class_t reg_classes[2] = {};
  reg_classes[0].alias_set_id = 7;
  reg_classes[1].alias_set_id = 7;
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  const loom_liveness_segment_t segments[] = {{2, 4}, {8, 10}, {16, 18}};
  uint32_t unit_start_points[] = {1, 2};
  uint32_t unit_end_points[] = {14, 18};
  loom_low_allocation_unit_liveness_t unit_liveness = UnitLiveness(
      unit_start_points, unit_end_points, IREE_ARRAYSIZE(unit_end_points));
  unit_liveness.storage_segments.entries = segments;

  for (auto kind : {LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
                    LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID}) {
    SCOPED_TRACE(kind);
    loom_low_allocation_assignment_t assignments[] = {
        Assignment(/*reg_class_id=*/0, /*location=*/8, /*unit_count=*/1,
                   /*start_point=*/1, /*end_point=*/14,
                   /*unit_point_start=*/0, kind),
        Assignment(/*reg_class_id=*/1, /*location=*/8, /*unit_count=*/1,
                   /*start_point=*/2, /*end_point=*/18,
                   /*unit_point_start=*/1, kind),
    };
    assignments[1].liveness_segments = {0, IREE_ARRAYSIZE(segments)};
    loom_low_allocation_storage_liveness_index_t index;
    IREE_ASSERT_OK(loom_low_allocation_storage_liveness_index_initialize(
        &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
        &unit_liveness, &arena_, &index));
    const loom_low_move_location_t alias = Location(
        /*reg_class_id=*/1, /*location=*/8, kind);
    for (uint32_t point = 0; point <= 19; ++point) {
      SCOPED_TRACE(point);
      // The latest start at 8 ends at 10, but the earlier reservation remains
      // live through 13. Only the gap at 14..15 is actually free.
      const bool expected =
          (point >= 1 && point < 14) || (point >= 16 && point < 18);
      EXPECT_EQ(loom_low_allocation_storage_liveness_index_is_live_at_point(
                    &index, &alias, point),
                expected);
    }
  }
}

TEST_F(LowAllocationStorageLivenessIndexTest,
       IntersectsSparseReservationsWithRefinedUnits) {
  loom_low_reg_class_t reg_classes[2] = {};
  reg_classes[0].alias_set_id = 7;
  reg_classes[1].alias_set_id = 7;
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  const loom_liveness_segment_t segments[] = {
      {4, 6}, {2, 4}, {6, 8}, {11, 16}, {18, 20},
  };
  uint32_t unit_start_points[] = {2, 7, 10, 8, 8};
  uint32_t unit_end_points[] = {8, 20, 10, 10, 11};
  loom_low_allocation_unit_liveness_t unit_liveness = UnitLiveness(
      unit_start_points, unit_end_points, IREE_ARRAYSIZE(unit_end_points));
  unit_liveness.storage_segments.entries = segments;

  for (auto kind : {LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
                    LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID}) {
    SCOPED_TRACE(kind);
    loom_low_allocation_assignment_t assignments[] = {
        Assignment(/*reg_class_id=*/0, /*location=*/10, /*unit_count=*/4,
                   /*start_point=*/2, /*end_point=*/20,
                   /*unit_point_start=*/0, kind),
        Assignment(/*reg_class_id=*/1, /*location=*/11, /*unit_count=*/1,
                   /*start_point=*/8, /*end_point=*/11,
                   /*unit_point_start=*/4, kind),
    };
    assignments[0].liveness_segments = {1, 4};
    loom_low_allocation_storage_liveness_index_t index;
    IREE_ASSERT_OK(loom_low_allocation_storage_liveness_index_initialize(
        &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
        &unit_liveness, &arena_, &index));

    for (uint32_t unit = 0; unit < 4; ++unit) {
      SCOPED_TRACE(unit);
      const loom_low_move_location_t alias = Location(
          /*reg_class_id=*/1, /*location=*/10 + unit, kind);
      for (uint32_t point = 0; point <= 21; ++point) {
        SCOPED_TRACE(point);
        const bool in_segment =
            (point >= 2 && point < 4) || (point >= 6 && point < 8) ||
            (point >= 11 && point < 16) || (point >= 18 && point < 20);
        const bool refined_live =
            point >= unit_start_points[unit] && point < unit_end_points[unit];
        // The other assignment has no sparse range: its continuous lifetime
        // fills this part of the second unit's hole without filling any other.
        const bool continuous_live = unit == 1 && point >= 8 && point < 11;
        EXPECT_EQ(loom_low_allocation_storage_liveness_index_is_live_at_point(
                      &index, &alias, point),
                  (in_segment && refined_live) || continuous_live);
      }
    }
  }
}

TEST_F(LowAllocationStorageLivenessIndexTest,
       ResolvesExplicitRegisterAtomicAliases) {
  loom_low_reg_class_t reg_classes[2] = {};
  for (auto& reg_class : reg_classes) {
    reg_class.flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                      LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
  }
  reg_classes[0].allocatable_count = 2;
  reg_classes[1].allocatable_count = 1;
  reg_classes[1].physical_register_candidate_start = 2;
  reg_classes[0].candidate_lookup.register_count = 2;
  reg_classes[1].candidate_lookup.ordinal_start = 2;
  reg_classes[1].candidate_lookup.register_base = 2;
  reg_classes[1].candidate_lookup.register_count = 1;
  const uint16_t candidate_ordinals[] = {0, 1, 0};
  const uint16_t candidates[] = {0, 1, 2};
  const uint16_t allocation_ordinals[] = {0, 1, 0};
  const uint16_t atomic_units[] = {0, 1, 0, 1};
  const loom_low_physical_register_t registers[] = {
      {.name_string_ref = 0, .atomic_unit_start = 0, .atomic_unit_count = 1},
      {.name_string_ref = 0, .atomic_unit_start = 1, .atomic_unit_count = 1},
      {.name_string_ref = 0,
       .atomic_unit_start = 2,
       .atomic_unit_count = 2,
       .reserved = 0,
       .view_lookup = {.ordinal_start = 0, .class_base = 0, .class_count = 1}},
  };
  const uint32_t view_ordinals[] = {0};
  const uint16_t view_units[] = {0, 1};
  const loom_low_physical_register_view_t views[] = {
      {.physical_register_id = 2,
       .reg_class_id = 0,
       .unit_candidate_ordinal_start = 0,
       .unit_count = 2},
  };
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = IREE_ARRAYSIZE(reg_classes);
  descriptor_set.physical_registers = registers;
  descriptor_set.physical_register_count = IREE_ARRAYSIZE(registers);
  descriptor_set.physical_register_candidate_ordinals = candidate_ordinals;
  descriptor_set.physical_register_candidate_ordinal_count =
      IREE_ARRAYSIZE(candidate_ordinals);
  descriptor_set.physical_register_candidate_ids = candidates;
  descriptor_set.physical_register_candidate_count = IREE_ARRAYSIZE(candidates);
  descriptor_set.physical_register_allocation_ordinals = allocation_ordinals;
  descriptor_set.physical_register_atomic_units = atomic_units;
  descriptor_set.physical_register_atomic_unit_count =
      IREE_ARRAYSIZE(atomic_units);
  descriptor_set.physical_register_view_ordinals = view_ordinals;
  descriptor_set.physical_register_view_ordinal_count =
      IREE_ARRAYSIZE(view_ordinals);
  descriptor_set.physical_register_views = views;
  descriptor_set.physical_register_view_count = IREE_ARRAYSIZE(views);
  descriptor_set.physical_register_view_unit_candidate_ordinals = view_units;
  descriptor_set.physical_register_view_unit_candidate_ordinal_count =
      IREE_ARRAYSIZE(view_units);
  loom_low_allocation_assignment_t assignment =
      Assignment(/*reg_class_id=*/0, /*location=*/2, /*unit_count=*/2,
                 /*start_point=*/2, /*end_point=*/9,
                 /*unit_point_start=*/0);
  uint32_t unit_start_points[] = {2, 6};
  uint32_t unit_end_points[] = {5, 9};
  loom_low_allocation_unit_liveness_t unit_liveness = UnitLiveness(
      unit_start_points, unit_end_points, IREE_ARRAYSIZE(unit_end_points));
  const loom_liveness_segment_t segments[] = {{2, 3}, {4, 7}, {8, 9}};
  unit_liveness.storage_segments.entries = segments;
  const loom_low_move_location_t wide =
      Location(/*reg_class_id=*/1, /*location=*/2);
  for (uint32_t segment_count : {0u, 3u}) {
    SCOPED_TRACE(segment_count);
    assignment.liveness_segments = {0, segment_count};
    loom_low_allocation_storage_liveness_index_t index;
    IREE_ASSERT_OK(loom_low_allocation_storage_liveness_index_initialize(
        &descriptor_set, &assignment, /*assignment_count=*/1, &unit_liveness,
        &arena_, &index));

    for (uint32_t point = 0; point < 11; ++point) {
      SCOPED_TRACE(point);
      const bool in_segment = segment_count == 0 || (point >= 2 && point < 3) ||
                              (point >= 4 && point < 7) ||
                              (point >= 8 && point < 9);
      const bool refined_live =
          (point >= 2 && point < 5) || (point >= 6 && point < 9);
      EXPECT_EQ(loom_low_allocation_storage_liveness_index_is_live_at_point(
                    &index, &wide, point),
                in_segment && refined_live);
    }
  }
}

}  // namespace
}  // namespace loom
