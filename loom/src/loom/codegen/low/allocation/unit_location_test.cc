// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/unit_location.h"

#include "iree/testing/gtest.h"
namespace loom {
namespace {

loom_low_allocation_assignment_t Assignment(
    uint16_t reg_class_id, loom_low_allocation_location_kind_t location_kind =
                               LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
  loom_low_allocation_assignment_t assignment = {};
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

}  // namespace
}  // namespace loom
