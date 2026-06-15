// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

loom_low_reg_class_t RegClass(uint16_t alias_set_id,
                              uint16_t allocatable_count = 0,
                              loom_low_reg_class_flags_t flags = 0) {
  loom_low_reg_class_t reg_class = {};
  reg_class.alias_set_id = alias_set_id;
  reg_class.allocatable_count = allocatable_count;
  reg_class.flags = flags;
  return reg_class;
}

loom_low_descriptor_set_t DescriptorSet(const loom_low_reg_class_t* reg_classes,
                                        iree_host_size_t reg_class_count) {
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.reg_classes = reg_classes;
  descriptor_set.reg_class_count = reg_class_count;
  return descriptor_set;
}

loom_low_allocation_assignment_t Assignment(
    uint16_t descriptor_reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.descriptor_reg_class_id = descriptor_reg_class_id;
  assignment.location_kind = location_kind;
  assignment.location_base = location_base;
  assignment.location_count = location_count;
  return assignment;
}

TEST(LowAllocationStorageTest, MatchesConcreteAndAliasRanges) {
  const loom_low_reg_class_t reg_classes[3] = {
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/0),
  };
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  const loom_low_allocation_assignment_t lhs = Assignment(
      /*descriptor_reg_class_id=*/0,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/2);
  const loom_low_allocation_assignment_t alias_same = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/2);
  const loom_low_allocation_assignment_t alias_overlap = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/5,
      /*location_count=*/2);
  const loom_low_allocation_assignment_t different_class = Assignment(
      /*descriptor_reg_class_id=*/2,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/2);
  const loom_low_allocation_assignment_t different_kind = Assignment(
      /*descriptor_reg_class_id=*/1, LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
      /*location_base=*/4, /*location_count=*/2);

  EXPECT_FALSE(
      loom_low_allocation_assignment_location_range_equal(&lhs, &alias_same));
  EXPECT_TRUE(loom_low_allocation_storage_assignment_ranges_equal(
      &descriptor_set, &lhs, &alias_same));
  EXPECT_TRUE(loom_low_allocation_storage_assignment_locations_share(
      &descriptor_set, &lhs, &alias_same));
  EXPECT_TRUE(loom_low_allocation_storage_assignment_ranges_overlap(
      &descriptor_set, &lhs, &alias_overlap));
  EXPECT_FALSE(loom_low_allocation_storage_assignment_locations_share(
      &descriptor_set, &lhs, &alias_overlap));
  EXPECT_FALSE(loom_low_allocation_storage_assignment_ranges_equal(
      &descriptor_set, &lhs, &different_class));
  EXPECT_FALSE(loom_low_allocation_storage_assignment_classes_share(
      &descriptor_set, &lhs, &different_kind));

  const loom_low_allocation_assignment_t empty_lhs = Assignment(
      /*descriptor_reg_class_id=*/0,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/0);
  const loom_low_allocation_assignment_t empty_alias_same = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/0);
  EXPECT_FALSE(loom_low_allocation_storage_assignment_ranges_equal(
      &descriptor_set, &empty_lhs, &empty_alias_same));
  EXPECT_TRUE(loom_low_allocation_storage_assignment_locations_share(
      &descriptor_set, &empty_lhs, &empty_alias_same));
}

TEST(LowAllocationStorageTest, MatchesAndOverlapsAliasSubranges) {
  const loom_low_reg_class_t reg_classes[2] = {
      RegClass(/*alias_set_id=*/7),
      RegClass(/*alias_set_id=*/7),
  };
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  const loom_low_allocation_assignment_t lhs = Assignment(
      /*descriptor_reg_class_id=*/0,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/10,
      /*location_count=*/8);
  const loom_low_allocation_assignment_t rhs = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/12,
      /*location_count=*/8);

  EXPECT_TRUE(loom_low_allocation_storage_assignment_subranges_equal(
      &descriptor_set, &lhs, 2, &rhs, 0, /*unit_count=*/2));
  EXPECT_TRUE(loom_low_allocation_storage_assignment_subranges_overlap(
      &descriptor_set, &lhs, 1, &rhs, 0, /*unit_count=*/2));
  EXPECT_FALSE(loom_low_allocation_storage_assignment_subranges_overlap(
      &descriptor_set, &lhs, 0, &rhs, 4, /*unit_count=*/2));
}

TEST(LowAllocationStorageTest, SharesRegisterClassAliasSets) {
  const loom_low_reg_class_t reg_classes[3] = {
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/1),
      RegClass(/*alias_set_id=*/0),
  };
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));

  EXPECT_TRUE(
      loom_low_allocation_storage_reg_classes_share(&descriptor_set, 0, 0));
  EXPECT_TRUE(
      loom_low_allocation_storage_reg_classes_share(&descriptor_set, 0, 1));
  EXPECT_FALSE(
      loom_low_allocation_storage_reg_classes_share(&descriptor_set, 0, 2));
}

TEST(LowAllocationStorageTest, MapsRegisterClassToLocationKind) {
  const loom_low_reg_class_t allocatable_reg_class =
      RegClass(/*alias_set_id=*/0, /*allocatable_count=*/16);
  const loom_low_reg_class_t physical_reg_class =
      RegClass(/*alias_set_id=*/0, /*allocatable_count=*/0,
               LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
  const loom_low_reg_class_t target_id_reg_class = RegClass(/*alias_set_id=*/0);

  EXPECT_EQ(LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            loom_low_allocation_storage_reg_class_location_kind(
                &allocatable_reg_class));
  EXPECT_EQ(
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      loom_low_allocation_storage_reg_class_location_kind(&physical_reg_class));
  EXPECT_EQ(LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
            loom_low_allocation_storage_reg_class_location_kind(
                &target_id_reg_class));
}

}  // namespace
}  // namespace loom
