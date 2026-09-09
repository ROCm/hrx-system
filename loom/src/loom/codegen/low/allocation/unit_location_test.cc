// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_location.h"

#include "iree/testing/gtest.h"
namespace loom {
namespace {

loom_liveness_value_class_t ValueClass(uint16_t reg_class_id) {
  loom_liveness_value_class_t value_class = {};
  value_class.type_kind = LOOM_TYPE_REGISTER;
  value_class.register_class_id = reg_class_id;
  return value_class;
}

loom_low_allocation_assignment_t Assignment(
    uint16_t reg_class_id, loom_low_allocation_location_kind_t location_kind =
                               LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_class = ValueClass(reg_class_id);
  assignment.descriptor_reg_class_id = reg_class_id;
  assignment.start_point = 2;
  assignment.end_point = 10;
  assignment.unit_count = 3;
  assignment.location_kind = location_kind;
  assignment.location_base = 7;
  assignment.location_count = 3;
  return assignment;
}

loom_low_move_location_t Location(
    uint16_t reg_class_id, uint32_t location,
    loom_low_allocation_location_kind_t location_kind =
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
  loom_low_move_location_t unit_location = {};
  unit_location.location_kind = location_kind;
  unit_location.descriptor_reg_class_id = reg_class_id;
  unit_location.location = location;
  return unit_location;
}

loom_low_reg_class_t RegClass(uint16_t alias_set_id) {
  loom_low_reg_class_t reg_class = {};
  reg_class.alias_set_id = alias_set_id;
  return reg_class;
}

loom_low_descriptor_set_t DescriptorSet(const loom_low_reg_class_t* reg_classes,
                                        iree_host_size_t reg_class_count) {
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = reg_class_count;
  return descriptor_set;
}

TEST(LowAllocationUnitLocationTest, MapsAssignmentUnitLocations) {
  const loom_low_allocation_assignment_t assignment =
      Assignment(/*reg_class_id=*/3);
  const loom_low_reg_class_t reg_classes[4] = {};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));

  const loom_low_move_location_t unit_location =
      loom_low_allocation_assignment_unit_location(&descriptor_set, &assignment,
                                                   /*unit_index=*/1);

  EXPECT_EQ(unit_location.location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(unit_location.descriptor_reg_class_id, 3);
  EXPECT_EQ(unit_location.location, 8u);
}

TEST(LowAllocationUnitLocationTest, ComparesLocationsAndStorageClasses) {
  const loom_low_move_location_t location =
      Location(/*reg_class_id=*/1, /*location=*/4);
  const loom_low_move_location_t same_location =
      Location(/*reg_class_id=*/1, /*location=*/4);
  const loom_low_move_location_t sibling_location =
      Location(/*reg_class_id=*/1, /*location=*/5);
  const loom_low_move_location_t different_class =
      Location(/*reg_class_id=*/2, /*location=*/4);

  EXPECT_TRUE(
      loom_low_allocation_unit_locations_equal(&location, &same_location));
  EXPECT_FALSE(
      loom_low_allocation_unit_locations_equal(&location, &sibling_location));
  EXPECT_TRUE(loom_low_allocation_unit_storage_classes_equal(
      &location, &sibling_location));
  EXPECT_FALSE(loom_low_allocation_unit_storage_classes_equal(
      &location, &different_class));
}

TEST(LowAllocationUnitLocationTest, ClassifiesRegisterMoves) {
  const loom_low_move_location_t source =
      Location(/*reg_class_id=*/1, /*location=*/4);
  const loom_low_move_location_t destination =
      Location(/*reg_class_id=*/1, /*location=*/5);
  const loom_low_move_location_t identical_destination =
      Location(/*reg_class_id=*/1, /*location=*/4);
  const loom_low_move_location_t spill_destination = Location(
      /*reg_class_id=*/1, /*location=*/0,
      LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT);

  EXPECT_TRUE(loom_low_allocation_unit_locations_form_register_move(
      &source, &destination));
  EXPECT_FALSE(loom_low_allocation_unit_locations_form_register_move(
      &source, &identical_destination));
  EXPECT_FALSE(loom_low_allocation_unit_locations_form_register_move(
      &source, &spill_destination));
}

TEST(LowAllocationUnitLocationTest, DetectsLiveUnitAtPoint) {
  const loom_low_reg_class_t reg_classes[] = {RegClass(/*alias_set_id=*/0)};
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));

  loom_low_allocation_assignment_t assignment = Assignment(/*reg_class_id=*/0);
  assignment.location_base = 4;
  assignment.location_count = 2;
  assignment.unit_count = 2;
  assignment.unit_point_start = 0;
  assignment.flags = LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
  uint32_t unit_start_points[] = {2, 6};
  uint32_t unit_end_points[] = {5, 9};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.start_points = unit_start_points;
  unit_liveness.end_points = unit_end_points;
  unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);

  const loom_low_move_location_t first_unit =
      Location(/*reg_class_id=*/0, /*location=*/4);
  const loom_low_move_location_t second_unit =
      Location(/*reg_class_id=*/0, /*location=*/5);
  const loom_low_move_location_t outside_unit =
      Location(/*reg_class_id=*/0, /*location=*/6);

  EXPECT_TRUE(loom_low_allocation_unit_location_is_live_at_point(
      &descriptor_set, &assignment, /*assignment_count=*/1, &unit_liveness,
      &first_unit, /*point=*/4));
  EXPECT_FALSE(loom_low_allocation_unit_location_is_live_at_point(
      &descriptor_set, &assignment, /*assignment_count=*/1, &unit_liveness,
      &first_unit, /*point=*/6));
  EXPECT_FALSE(loom_low_allocation_unit_location_is_live_at_point(
      &descriptor_set, &assignment, /*assignment_count=*/1, &unit_liveness,
      &second_unit, /*point=*/4));
  EXPECT_TRUE(loom_low_allocation_unit_location_is_live_at_point(
      &descriptor_set, &assignment, /*assignment_count=*/1, &unit_liveness,
      &second_unit, /*point=*/6));
  EXPECT_FALSE(loom_low_allocation_unit_location_is_live_at_point(
      &descriptor_set, &assignment, /*assignment_count=*/1, &unit_liveness,
      &outside_unit, /*point=*/4));
}

TEST(LowAllocationUnitLocationTest, WideScratchOverlapsEveryNarrowUnit) {
  // The pair is two independently live narrow units but one wide register.
  // Its first narrow unit can be dead while the second is still occupied.
  loom_low_reg_class_t reg_classes[2] = {};
  for (auto& reg_class : reg_classes) {
    reg_class.flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                      LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS;
  }
  reg_classes[0].allocatable_count = 2;
  reg_classes[1].allocatable_count = 1;
  reg_classes[1].physical_register_candidate_start = 2;
  const uint16_t candidates[] = {0, 1, 2};
  const uint16_t allocation_ordinals[] = {0, 1, 0};
  const uint16_t atomic_units[] = {0, 1, 0, 1};
  const loom_low_physical_register_t registers[] = {
      {/*.name_string_offset=*/0, /*.atomic_unit_start=*/0,
       /*.atomic_unit_count=*/1},
      {/*.name_string_offset=*/0, /*.atomic_unit_start=*/1,
       /*.atomic_unit_count=*/1},
      {/*.name_string_offset=*/0, /*.atomic_unit_start=*/2,
       /*.atomic_unit_count=*/2},
  };
  const uint16_t view_units[] = {0, 1};
  const loom_low_physical_register_view_t views[] = {
      {/*.physical_register_id=*/2, /*.reg_class_id=*/0,
       /*.unit_candidate_ordinal_start=*/0, /*.unit_count=*/2},
  };
  loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  descriptor_set.physical_registers = registers;
  descriptor_set.physical_register_count = IREE_ARRAYSIZE(registers);
  descriptor_set.physical_register_candidate_ids = candidates;
  descriptor_set.physical_register_candidate_count = IREE_ARRAYSIZE(candidates);
  descriptor_set.physical_register_allocation_ordinals = allocation_ordinals;
  descriptor_set.physical_register_atomic_units = atomic_units;
  descriptor_set.physical_register_atomic_unit_count =
      IREE_ARRAYSIZE(atomic_units);
  descriptor_set.physical_register_views = views;
  descriptor_set.physical_register_view_count = IREE_ARRAYSIZE(views);
  descriptor_set.physical_register_view_unit_candidate_ordinals = view_units;
  descriptor_set.physical_register_view_unit_candidate_ordinal_count =
      IREE_ARRAYSIZE(view_units);

  loom_low_allocation_assignment_t assignment = Assignment(/*reg_class_id=*/0);
  assignment.location_base = 2;
  assignment.location_count = 2;
  assignment.unit_count = 2;
  assignment.flags = LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS;
  uint32_t unit_start_points[] = {2, 6};
  uint32_t unit_end_points[] = {5, 9};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.start_points = unit_start_points;
  unit_liveness.end_points = unit_end_points;
  unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);
  const loom_low_move_location_t wide =
      Location(/*reg_class_id=*/1, /*location=*/2);

  for (uint32_t point = 0; point < 11; ++point) {
    SCOPED_TRACE(point);
    EXPECT_EQ(loom_low_allocation_unit_location_is_live_at_point(
                  &descriptor_set, &assignment, /*assignment_count=*/1,
                  &unit_liveness, &wide, point),
              (point >= 2 && point < 5) || (point >= 6 && point < 9));
  }
}

}  // namespace
}  // namespace loom
