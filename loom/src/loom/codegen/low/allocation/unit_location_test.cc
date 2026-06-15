// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_location.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

using ::iree::StatusCode;

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

loom_low_allocation_unit_location_t Location(
    uint16_t reg_class_id, uint32_t location,
    loom_low_allocation_location_kind_t location_kind =
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
  loom_low_allocation_unit_location_t unit_location = {};
  unit_location.location_kind = location_kind;
  unit_location.value_class = ValueClass(reg_class_id);
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

  loom_low_allocation_unit_location_t unit_location = {};
  IREE_ASSERT_OK(loom_low_allocation_assignment_unit_location(
      &assignment, /*unit_index=*/1, &unit_location));

  EXPECT_EQ(unit_location.location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_TRUE(loom_liveness_value_class_equal(unit_location.value_class,
                                              assignment.value_class));
  EXPECT_EQ(unit_location.descriptor_reg_class_id, 3);
  EXPECT_EQ(unit_location.location, 8u);
}

TEST(LowAllocationUnitLocationTest, RejectsOutOfRangeAssignmentUnit) {
  const loom_low_allocation_assignment_t assignment =
      Assignment(/*reg_class_id=*/3);

  loom_low_allocation_unit_location_t unit_location = {};
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange,
                        loom_low_allocation_assignment_unit_location(
                            &assignment, /*unit_index=*/3, &unit_location));
}

TEST(LowAllocationUnitLocationTest, ComparesLocationsAndStorageClasses) {
  const loom_low_allocation_unit_location_t location =
      Location(/*reg_class_id=*/1, /*location=*/4);
  const loom_low_allocation_unit_location_t same_location =
      Location(/*reg_class_id=*/1, /*location=*/4);
  const loom_low_allocation_unit_location_t sibling_location =
      Location(/*reg_class_id=*/1, /*location=*/5);
  const loom_low_allocation_unit_location_t different_class =
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
  const loom_low_allocation_unit_location_t source =
      Location(/*reg_class_id=*/1, /*location=*/4);
  const loom_low_allocation_unit_location_t destination =
      Location(/*reg_class_id=*/1, /*location=*/5);
  const loom_low_allocation_unit_location_t identical_destination =
      Location(/*reg_class_id=*/1, /*location=*/4);
  const loom_low_allocation_unit_location_t spill_destination = Location(
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
  assignment.unit_end_point_start = 0;
  const uint32_t unit_end_points[] = {5, 9};

  const loom_low_allocation_unit_location_t first_unit =
      Location(/*reg_class_id=*/0, /*location=*/4);
  const loom_low_allocation_unit_location_t second_unit =
      Location(/*reg_class_id=*/0, /*location=*/5);
  const loom_low_allocation_unit_location_t outside_unit =
      Location(/*reg_class_id=*/0, /*location=*/6);

  EXPECT_TRUE(loom_low_allocation_unit_location_is_live_at_point(
      &descriptor_set, &assignment, /*assignment_count=*/1, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &first_unit, /*point=*/4));
  EXPECT_FALSE(loom_low_allocation_unit_location_is_live_at_point(
      &descriptor_set, &assignment, /*assignment_count=*/1, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &first_unit, /*point=*/6));
  EXPECT_TRUE(loom_low_allocation_unit_location_is_live_at_point(
      &descriptor_set, &assignment, /*assignment_count=*/1, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &second_unit, /*point=*/6));
  EXPECT_FALSE(loom_low_allocation_unit_location_is_live_at_point(
      &descriptor_set, &assignment, /*assignment_count=*/1, unit_end_points,
      IREE_ARRAYSIZE(unit_end_points), &outside_unit, /*point=*/4));
}

}  // namespace
}  // namespace loom
