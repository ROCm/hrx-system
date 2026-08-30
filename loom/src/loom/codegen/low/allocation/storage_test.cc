// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage.h"

#include "iree/testing/gtest.h"
#include "loom/target/test/descriptors.h"

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
  assignment.unit_count = location_count;
  return assignment;
}

bool FindExplicitPhysicalRegisterView(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id, uint32_t unit_count,
    uint32_t first_candidate_ordinal, uint32_t maximum_pressure_extent,
    uint32_t* out_physical_register_id) {
  *out_physical_register_id = UINT32_MAX;
  for (uint32_t physical_register_id = 0;
       physical_register_id < descriptor_set->physical_register_count;
       ++physical_register_id) {
    uint32_t actual_first_candidate_ordinal = 0;
    uint32_t pressure_extent = 0;
    if (loom_low_allocation_storage_explicit_physical_register_view(
            descriptor_set, descriptor_reg_class_id, physical_register_id,
            unit_count, &actual_first_candidate_ordinal, &pressure_extent) &&
        actual_first_candidate_ordinal == first_candidate_ordinal &&
        pressure_extent <= maximum_pressure_extent) {
      *out_physical_register_id = physical_register_id;
      return true;
    }
  }
  return false;
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

TEST(LowAllocationStorageTest, EvaluatesConcretePlacementRelations) {
  const loom_low_reg_class_t reg_class = RegClass(/*alias_set_id=*/0);
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(&reg_class, /*reg_class_count=*/1);
  const loom_low_allocation_assignment_t result = Assignment(
      /*descriptor_reg_class_id=*/0,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/2);
  const loom_low_allocation_assignment_t source = Assignment(
      /*descriptor_reg_class_id=*/0,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/5,
      /*location_count=*/2);
  const loom_low_allocation_assignment_t alias = Assignment(
      /*descriptor_reg_class_id=*/0,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/4,
      /*location_count=*/2);

  loom_low_placement_relation_t relation = {
      /*.op=*/nullptr,
      /*.result_ordinal=*/0,
      /*.source_ordinal=*/1,
      /*.result_unit_offset=*/0,
      /*.source_unit_offset=*/0,
      /*.unit_count=*/1,
      /*.location_mask=*/1,
      /*.kind=*/LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION,
  };
  EXPECT_TRUE(loom_low_allocation_storage_placement_relation_satisfied(
      &descriptor_set, &relation, &result, &source));
  EXPECT_FALSE(loom_low_allocation_storage_placement_relation_satisfied(
      &descriptor_set, &relation, &result, &alias));

  relation.kind = LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE;
  relation.location_mask = 0;
  EXPECT_TRUE(loom_low_allocation_storage_placement_relation_satisfied(
      &descriptor_set, &relation, &result, &source));
  EXPECT_FALSE(loom_low_allocation_storage_placement_relation_satisfied(
      &descriptor_set, &relation, &result, &alias));
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

TEST(LowAllocationStorageTest, MatchesExplicitRegisterAtomicStorage) {
  const loom_low_reg_class_t reg_classes[2] = {
      RegClass(/*alias_set_id=*/0, /*allocatable_count=*/2,
               LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                   LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS),
      RegClass(/*alias_set_id=*/0, /*allocatable_count=*/1,
               LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
                   LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS),
  };
  const uint16_t atomic_units[] = {0, 1, 0, 2, 3};
  const loom_low_physical_register_t physical_registers[] = {
      {
          /*.name_string_offset=*/0,
          /*.atomic_unit_start=*/0,
          /*.atomic_unit_count=*/2,
          /*.reserved=*/0,
      },
      {
          /*.name_string_offset=*/0,
          /*.atomic_unit_start=*/2,
          /*.atomic_unit_count=*/1,
          /*.reserved=*/0,
      },
      {
          /*.name_string_offset=*/0,
          /*.atomic_unit_start=*/3,
          /*.atomic_unit_count=*/2,
          /*.reserved=*/0,
      },
  };
  const uint16_t candidates[] = {0, 2, 1};
  loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(reg_classes, IREE_ARRAYSIZE(reg_classes));
  descriptor_set.physical_registers = physical_registers;
  descriptor_set.physical_register_count = IREE_ARRAYSIZE(physical_registers);
  descriptor_set.physical_register_candidate_ids = candidates;
  descriptor_set.physical_register_candidate_count = IREE_ARRAYSIZE(candidates);
  descriptor_set.physical_register_atomic_units = atomic_units;
  descriptor_set.physical_register_atomic_unit_count =
      IREE_ARRAYSIZE(atomic_units);
  loom_low_reg_class_t explicit_reg_classes[2] = {reg_classes[0],
                                                  reg_classes[1]};
  explicit_reg_classes[1].physical_register_candidate_start = 2;
  descriptor_set.reg_classes = explicit_reg_classes;

  const loom_low_allocation_assignment_t wide0 = Assignment(
      /*descriptor_reg_class_id=*/0,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/0,
      /*location_count=*/1);
  const loom_low_allocation_assignment_t low0 = Assignment(
      /*descriptor_reg_class_id=*/1,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/1,
      /*location_count=*/1);
  const loom_low_allocation_assignment_t wide1 = Assignment(
      /*descriptor_reg_class_id=*/0,
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, /*location_base=*/2,
      /*location_count=*/1);

  EXPECT_TRUE(
      loom_low_allocation_storage_reg_classes_share(&descriptor_set, 0, 1));
  EXPECT_FALSE(loom_low_allocation_storage_assignment_ranges_equal(
      &descriptor_set, &wide0, &low0));
  EXPECT_TRUE(loom_low_allocation_storage_assignment_ranges_overlap(
      &descriptor_set, &wide0, &low0));
  EXPECT_FALSE(loom_low_allocation_storage_assignment_ranges_overlap(
      &descriptor_set, &wide0, &wide1));
  EXPECT_EQ(loom_low_allocation_storage_assignment_atomic_unit_count(
                &descriptor_set, &wide0),
            2u);
  EXPECT_EQ(loom_low_allocation_storage_assignment_pressure_extent(
                &descriptor_set, &wide1),
            2u);
}

TEST(LowAllocationStorageTest, ResolvesExplicitAggregateRegisterViews) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  ASSERT_TRUE(loom_low_descriptor_set_lookup_register_class(
      descriptor_set, IREE_SV("test.explicit32"), &reg_class_id, nullptr));

  uint32_t pair_register_id = UINT32_MAX;
  ASSERT_TRUE(FindExplicitPhysicalRegisterView(
      descriptor_set, reg_class_id, /*unit_count=*/2,
      /*first_candidate_ordinal=*/0, /*maximum_pressure_extent=*/4,
      &pair_register_id));
  uint32_t first_candidate_ordinal = UINT32_MAX;
  uint32_t pressure_extent = 0;
  EXPECT_TRUE(loom_low_allocation_storage_explicit_physical_register_view(
      descriptor_set, reg_class_id, pair_register_id, /*unit_count=*/2,
      &first_candidate_ordinal, &pressure_extent));
  EXPECT_EQ(first_candidate_ordinal, 0u);
  EXPECT_EQ(pressure_extent, 2u);

  uint32_t unavailable_register_id = UINT32_MAX;
  EXPECT_FALSE(FindExplicitPhysicalRegisterView(
      descriptor_set, reg_class_id, /*unit_count=*/2,
      /*first_candidate_ordinal=*/0, /*maximum_pressure_extent=*/1,
      &unavailable_register_id));
  EXPECT_FALSE(FindExplicitPhysicalRegisterView(
      descriptor_set, reg_class_id, /*unit_count=*/2,
      /*first_candidate_ordinal=*/1, /*maximum_pressure_extent=*/4,
      &unavailable_register_id));
  EXPECT_FALSE(FindExplicitPhysicalRegisterView(
      descriptor_set, reg_class_id, /*unit_count=*/3,
      /*first_candidate_ordinal=*/0, /*maximum_pressure_extent=*/4,
      &unavailable_register_id));

  const uint32_t first_register_id =
      loom_low_descriptor_set_physical_register_candidate(descriptor_set,
                                                          reg_class_id, 0);
  const uint32_t second_register_id =
      loom_low_descriptor_set_physical_register_candidate(descriptor_set,
                                                          reg_class_id, 1);
  const loom_low_allocation_assignment_t pair =
      Assignment(reg_class_id, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
                 pair_register_id, /*location_count=*/2);
  const loom_low_allocation_assignment_t first =
      Assignment(reg_class_id, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
                 first_register_id, /*location_count=*/1);
  const loom_low_allocation_assignment_t second =
      Assignment(reg_class_id, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
                 second_register_id, /*location_count=*/1);
  uint32_t unit_register_id = UINT32_MAX;
  EXPECT_TRUE(loom_low_allocation_storage_assignment_unit_physical_register(
      descriptor_set, &pair, 0, &unit_register_id));
  EXPECT_EQ(unit_register_id, first_register_id);
  EXPECT_TRUE(loom_low_allocation_storage_assignment_unit_physical_register(
      descriptor_set, &pair, 1, &unit_register_id));
  EXPECT_EQ(unit_register_id, second_register_id);
  EXPECT_TRUE(loom_low_allocation_storage_assignment_subranges_equal(
      descriptor_set, &pair, 0, &first, 0, /*unit_count=*/1));
  EXPECT_TRUE(loom_low_allocation_storage_assignment_subranges_equal(
      descriptor_set, &pair, 1, &second, 0, /*unit_count=*/1));
  EXPECT_FALSE(loom_low_allocation_storage_assignment_subranges_overlap(
      descriptor_set, &pair, 0, &second, 0, /*unit_count=*/1));

  uint32_t quad_register_id = UINT32_MAX;
  ASSERT_TRUE(FindExplicitPhysicalRegisterView(
      descriptor_set, reg_class_id, /*unit_count=*/4,
      /*first_candidate_ordinal=*/0, /*maximum_pressure_extent=*/4,
      &quad_register_id));
  EXPECT_TRUE(loom_low_allocation_storage_explicit_physical_register_view(
      descriptor_set, reg_class_id, quad_register_id, /*unit_count=*/4,
      &first_candidate_ordinal, &pressure_extent));
  EXPECT_EQ(first_candidate_ordinal, 0u);
  EXPECT_EQ(pressure_extent, 4u);
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
