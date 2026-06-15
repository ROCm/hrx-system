// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/search.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

class LowAllocationSearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* AllocateModule() {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                       nullptr, iree_allocator_system(),
                                       &module));
    return module;
  }

  loom_value_id_t DefineValue(loom_module_t* module) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(
        module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &value_id));
    return value_id;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
};

loom_liveness_value_class_t RegisterValueClass(uint64_t descriptor_set_id) {
  loom_liveness_value_class_t value_class = {};
  value_class.type_kind = LOOM_TYPE_REGISTER;
  value_class.register_descriptor_set_stable_id = descriptor_set_id;
  value_class.register_class_id = 0;
  return value_class;
}

loom_liveness_interval_t Interval(loom_value_id_t value_id, uint32_t start,
                                  uint32_t end,
                                  loom_liveness_value_class_t value_class,
                                  uint32_t unit_count) {
  loom_liveness_interval_t interval = {};
  interval.value_id = value_id;
  interval.start_point = start;
  interval.end_point = end;
  interval.value_class = value_class;
  interval.unit_count = unit_count;
  return interval;
}

loom_low_allocation_assignment_t Assignment(
    loom_value_id_t value_id, uint32_t start, uint32_t end,
    loom_liveness_value_class_t value_class, uint32_t location_base,
    uint32_t location_count, uint32_t unit_end_point_start) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.value_class = value_class;
  assignment.descriptor_reg_class_id = 0;
  assignment.start_point = start;
  assignment.end_point = end;
  assignment.unit_count = location_count;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_base = location_base;
  assignment.location_count = location_count;
  assignment.unit_end_point_start = unit_end_point_start;
  return assignment;
}

loom_low_allocation_class_capacity_t Capacity(uint32_t max_units) {
  loom_low_allocation_class_capacity_t capacity = {};
  capacity.descriptor_reg_class_id = 0;
  capacity.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  capacity.max_units = max_units;
  capacity.alloc_unit_bits = 32;
  capacity.spill_slot_space = LOOM_LOW_SPILL_SLOT_SPACE_UNKNOWN;
  capacity.is_spillable = true;
  capacity.is_bounded = true;
  return capacity;
}

loom_low_descriptor_set_t DescriptorSet(const loom_low_reg_class_t* reg_class,
                                        uint64_t stable_id) {
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.stable_id = stable_id;
  descriptor_set.reg_classes = reg_class;
  descriptor_set.reg_class_count = 1;
  return descriptor_set;
}

loom_low_reg_class_t RegClass(uint32_t allocatable_count,
                              loom_low_reg_class_flags_t flags) {
  loom_low_reg_class_t reg_class = {};
  reg_class.flags = flags;
  reg_class.alloc_unit_bits = 32;
  reg_class.allocatable_count = (uint16_t)allocatable_count;
  reg_class.spill_class_id = LOOM_LOW_REG_CLASS_NONE;
  return reg_class;
}

loom_low_resolved_target_t ResolvedTarget(
    const loom_low_descriptor_set_t* descriptor_set) {
  loom_low_resolved_target_t target = {};
  target.descriptor_set = descriptor_set;
  target.descriptor_set_key = IREE_SV("test");
  return target;
}

loom_liveness_block_info_t Block(uint32_t start_point, uint32_t end_point) {
  loom_liveness_block_info_t block = {};
  block.start_point = start_point;
  block.end_point = end_point;
  return block;
}

TEST_F(LowAllocationSearchTest, FindsFreeLocationAfterActiveAndReservedRanges) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t candidate_value = DefineValue(module);
  const loom_value_id_t active_value = DefineValue(module);
  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, candidate_value,
                                        /*ordinal=*/0);
  loom_module_value_ordinal_scratch_set(module, active_value,
                                        /*ordinal=*/1);

  const uint64_t descriptor_set_id = 7;
  const loom_liveness_value_class_t value_class =
      RegisterValueClass(descriptor_set_id);
  const loom_liveness_interval_t intervals[] = {
      Interval(candidate_value, /*start=*/2, /*end=*/6, value_class,
               /*unit_count=*/2),
      Interval(active_value, /*start=*/0, /*end=*/10, value_class,
               /*unit_count=*/2),
  };
  const uint32_t interval_indices[] = {0, 1};
  const loom_value_id_t value_ids[] = {candidate_value, active_value};
  const loom_liveness_block_info_t blocks[] = {Block(/*start_point=*/0,
                                                     /*end_point=*/12)};
  loom_liveness_analysis_t liveness = {};
  liveness.blocks = blocks;
  liveness.block_count = IREE_ARRAYSIZE(blocks);
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);
  liveness.value_ids = value_ids;
  liveness.value_count = IREE_ARRAYSIZE(value_ids);
  liveness.value_interval_indices = interval_indices;

  uint32_t unit_end_point_starts[] = {0, 2};
  uint32_t unit_end_points[] = {6, 6, 10, 10};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.end_point_starts_by_value_ordinal = unit_end_point_starts;
  unit_liveness.end_points = unit_end_points;
  unit_liveness.end_point_count = IREE_ARRAYSIZE(unit_end_points);

  const loom_low_reg_class_t reg_class =
      RegClass(/*allocatable_count=*/8, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(&reg_class, descriptor_set_id);
  const loom_low_resolved_target_t target = ResolvedTarget(&descriptor_set);
  uint32_t max_assigned_location_end_by_reg_class[] = {0};
  loom_low_allocation_resolved_reserved_range_t reserved_ranges[] = {
      {
          /*.descriptor_reg_class_id=*/0,
          /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          /*.location_base=*/2,
          /*.location_count=*/2,
      },
  };
  loom_low_allocation_target_constraints_t target_constraints = {};
  target_constraints.target = &target;
  target_constraints.reserved_ranges = reserved_ranges;
  target_constraints.reserved_range_count = IREE_ARRAYSIZE(reserved_ranges);
  target_constraints.max_assigned_location_end_by_reg_class =
      max_assigned_location_end_by_reg_class;

  loom_low_allocation_assignment_t assignments[] = {
      Assignment(active_value, /*start=*/0, /*end=*/10, value_class,
                 /*location_base=*/0, /*location_count=*/2,
                 /*unit_end_point_start=*/2),
  };
  uint32_t assignment_indices_by_value_ordinal[] = {UINT32_MAX, 0};
  loom_low_allocation_assignment_map_t assignment_map = {};
  assignment_map.module = module;
  assignment_map.liveness = &liveness;
  assignment_map.assignments = assignments;
  assignment_map.assignment_count = IREE_ARRAYSIZE(assignments);
  assignment_map.assignment_indices_by_value_ordinal =
      assignment_indices_by_value_ordinal;

  loom_low_allocation_active_set_t active_set = {};
  IREE_ASSERT_OK(loom_low_allocation_active_set_initialize(
      /*assignment_capacity=*/1, /*unit_capacity=*/8, &arena_, &active_set));
  loom_low_allocation_active_set_insert(
      &active_set, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/0);

  loom_low_allocation_storage_lease_state_t storage_leases = {};
  loom_low_allocation_search_context_t context = {};
  context.module = module;
  context.descriptor_set = &descriptor_set;
  context.liveness = &liveness;
  context.unit_liveness = &unit_liveness;
  context.target_constraints = &target_constraints;
  context.assignment_map = &assignment_map;
  context.active_set = &active_set;
  context.storage_leases = &storage_leases;

  uint32_t location_base = UINT32_MAX;
  EXPECT_TRUE(loom_low_allocation_search_find_free_location(
      &context, &intervals[0], Capacity(/*max_units=*/8), &location_base));
  EXPECT_EQ(location_base, 4u);

  loom_module_value_ordinal_scratch_clear(module, candidate_value);
  loom_module_value_ordinal_scratch_clear(module, active_value);
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
}

TEST_F(LowAllocationSearchTest, SelectsActiveSpillVictimSet) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t candidate_value = DefineValue(module);
  const loom_value_id_t active_value = DefineValue(module);
  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, candidate_value,
                                        /*ordinal=*/0);
  loom_module_value_ordinal_scratch_set(module, active_value,
                                        /*ordinal=*/1);

  const uint64_t descriptor_set_id = 11;
  const loom_liveness_value_class_t value_class =
      RegisterValueClass(descriptor_set_id);
  const loom_liveness_interval_t intervals[] = {
      Interval(candidate_value, /*start=*/2, /*end=*/6, value_class,
               /*unit_count=*/2),
      Interval(active_value, /*start=*/0, /*end=*/12, value_class,
               /*unit_count=*/2),
  };
  const uint32_t interval_indices[] = {0, 1};
  const loom_value_id_t value_ids[] = {candidate_value, active_value};
  const loom_liveness_block_info_t blocks[] = {Block(/*start_point=*/0,
                                                     /*end_point=*/16)};
  loom_liveness_analysis_t liveness = {};
  liveness.blocks = blocks;
  liveness.block_count = IREE_ARRAYSIZE(blocks);
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);
  liveness.value_ids = value_ids;
  liveness.value_count = IREE_ARRAYSIZE(value_ids);
  liveness.value_interval_indices = interval_indices;

  uint32_t unit_end_point_starts[] = {0, 2};
  uint32_t unit_end_points[] = {6, 6, 12, 12};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.end_point_starts_by_value_ordinal = unit_end_point_starts;
  unit_liveness.end_points = unit_end_points;
  unit_liveness.end_point_count = IREE_ARRAYSIZE(unit_end_points);

  const loom_low_reg_class_t reg_class =
      RegClass(/*allocatable_count=*/2, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(&reg_class, descriptor_set_id);
  const loom_low_resolved_target_t target = ResolvedTarget(&descriptor_set);
  uint32_t max_assigned_location_end_by_reg_class[] = {2};
  loom_low_allocation_target_constraints_t target_constraints = {};
  target_constraints.target = &target;
  target_constraints.max_assigned_location_end_by_reg_class =
      max_assigned_location_end_by_reg_class;

  loom_low_allocation_assignment_t assignments[] = {
      Assignment(active_value, /*start=*/0, /*end=*/12, value_class,
                 /*location_base=*/0, /*location_count=*/2,
                 /*unit_end_point_start=*/2),
  };
  uint32_t assignment_indices_by_value_ordinal[] = {UINT32_MAX, 0};
  loom_low_allocation_assignment_map_t assignment_map = {};
  assignment_map.module = module;
  assignment_map.liveness = &liveness;
  assignment_map.assignments = assignments;
  assignment_map.assignment_count = IREE_ARRAYSIZE(assignments);
  assignment_map.assignment_indices_by_value_ordinal =
      assignment_indices_by_value_ordinal;

  loom_low_allocation_active_set_t active_set = {};
  IREE_ASSERT_OK(loom_low_allocation_active_set_initialize(
      /*assignment_capacity=*/1, /*unit_capacity=*/8, &arena_, &active_set));
  loom_low_allocation_active_set_insert(
      &active_set, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/0);

  loom_low_allocation_storage_lease_state_t storage_leases = {};
  loom_low_allocation_search_context_t context = {};
  context.module = module;
  context.descriptor_set = &descriptor_set;
  context.liveness = &liveness;
  context.unit_liveness = &unit_liveness;
  context.target_constraints = &target_constraints;
  context.assignment_map = &assignment_map;
  context.active_set = &active_set;
  context.storage_leases = &storage_leases;

  const loom_low_allocation_class_capacity_t capacity = Capacity(
      /*max_units=*/2);
  loom_low_allocation_search_spill_victim_set_t victim_set = {};
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/false, &arena_, &victim_set));
  ASSERT_TRUE(victim_set.found);
  EXPECT_EQ(victim_set.location_base, 0u);
  ASSERT_EQ(victim_set.assignment_count, 1u);
  EXPECT_EQ(victim_set.assignment_indices[0], 0u);

  bool can_spill = false;
  loom_low_allocation_class_capacity_t spill_capacity = {};
  IREE_ASSERT_OK(loom_low_allocation_search_assignment_spill_capacity(
      &context, &assignments[0], &can_spill, &spill_capacity));
  EXPECT_TRUE(can_spill);
  EXPECT_EQ(spill_capacity.max_units, 2u);

  loom_module_value_ordinal_scratch_clear(module, candidate_value);
  loom_module_value_ordinal_scratch_clear(module, active_value);
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
