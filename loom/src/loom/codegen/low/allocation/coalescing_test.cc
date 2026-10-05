// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/coalescing.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

class LowAllocationCoalescingTest : public ::testing::Test {
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

  // Backing blocks for module and allocation scratch.
  iree_arena_block_pool_t block_pool_;
  // Scratch storage for coalescing queries.
  iree_arena_allocator_t arena_;
  // Context shared by the fixture's modules.
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
                                  loom_liveness_value_class_t value_class) {
  loom_liveness_interval_t interval = {};
  interval.value_id = value_id;
  interval.start_point = start;
  interval.end_point = end;
  interval.value_class = value_class;
  interval.unit_count = 1;
  return interval;
}

loom_low_allocation_assignment_t Assignment(
    loom_value_id_t value_id, uint32_t start, uint32_t end,
    loom_liveness_value_class_t value_class, uint32_t location_base,
    uint32_t unit_point_start) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.descriptor_reg_class_id = value_class.register_class_id;
  assignment.start_point = start;
  assignment.end_point = end;
  assignment.unit_count = 1;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_base = location_base;
  assignment.location_count = 1;
  assignment.unit_point_start = unit_point_start;
  return assignment;
}

TEST_F(LowAllocationCoalescingTest, ConcatMayPrecedeItsSourceAssignments) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t value_ids[] = {DefineValue(module), DefineValue(module),
                                       DefineValue(module)};
  loom_module_value_ordinal_scratch_acquire(module);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(value_ids); ++i) {
    loom_module_value_ordinal_scratch_set(module, value_ids[i], i);
  }

  const loom_liveness_value_class_t value_class = RegisterValueClass(17);
  loom_liveness_interval_t intervals[] = {
      Interval(value_ids[0], /*start=*/10, /*end=*/12, value_class),
      Interval(value_ids[1], /*start=*/11, /*end=*/12, value_class),
      Interval(value_ids[2], /*start=*/2, /*end=*/14, value_class),
  };
  intervals[2].unit_count = 2;
  const uint32_t interval_indices[] = {0, 1, 2};
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);
  liveness.value_ids = value_ids;
  liveness.value_count = IREE_ARRAYSIZE(value_ids);
  liveness.value_interval_indices = interval_indices;

  loom_low_placement_relation_t relations[2] = {};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(relations); ++i) {
    relations[i].result_ordinal = 2;
    relations[i].source_ordinal = i;
    relations[i].result_unit_offset = i;
    relations[i].unit_count = 1;
    relations[i].kind = LOOM_LOW_PLACEMENT_RELATION_SUBRANGE;
    relations[i].cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
    relations[i].flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  }
  const loom_low_placement_relation_range_t result_ranges[] = {
      {0, 0}, {0, 0}, {0, 2}};
  const loom_low_placement_relation_range_t source_ranges[] = {
      {0, 1}, {1, 1}, {2, 0}};
  const uint32_t source_relations[] = {0, 1};
  loom_low_placement_table_t placement = {};
  placement.value_ids = value_ids;
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  placement.ranges_by_result_ordinal = result_ranges;
  placement.ranges_by_source_ordinal = source_ranges;
  placement.relation_indices_by_source_ordinal = source_relations;

  loom_low_allocation_assignment_t first_assignment =
      Assignment(value_ids[0], /*start=*/0, /*end=*/12, value_class,
                 /*location_base=*/0, /*unit_point_start=*/0);
  uint32_t assignment_indices[] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
  loom_low_allocation_assignment_map_t assignment_map = {};
  assignment_map.module = module;
  assignment_map.liveness = &liveness;
  assignment_map.assignments = &first_assignment;
  assignment_map.assignment_indices_by_value_ordinal = assignment_indices;

  loom_low_reg_class_t reg_class = {};
  reg_class.flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL;
  reg_class.alloc_unit_bits = 32;
  reg_class.allocatable_count = 8;
  reg_class.spill_class_id = LOOM_LOW_REG_CLASS_NONE;
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.stable_id = 17;
  descriptor_set.reg_classes = &reg_class;
  descriptor_set.reg_class_count = 1;
  loom_low_resolved_target_t target = {};
  target.descriptor_set = &descriptor_set;
  target.descriptor_set_key = IREE_SV("test");
  loom_low_allocation_target_constraints_t target_constraints = {};
  target_constraints.target = &target;
  loom_low_allocation_search_context_t search_context = {};
  search_context.descriptor_set = &descriptor_set;
  search_context.liveness = &liveness;
  search_context.placement = &placement;

  loom_low_allocation_coalescing_context_t context = {};
  context.arena = &arena_;
  context.liveness = &liveness;
  context.placement = &placement;
  context.assignment_map = &assignment_map;
  context.target_constraints = &target_constraints;
  context.search_context = &search_context;

  // Block layout can put a concat use before its defining block. Allocation
  // then reserves the result first, even when some operands were defined in
  // the entry block. The later sources can reuse that reservation.
  for (uint32_t assigned_count : {0u, 1u}) {
    SCOPED_TRACE(assigned_count);
    intervals[0].start_point = assigned_count ? 0 : 10;
    assignment_map.assignment_count = assigned_count;
    assignment_indices[0] = assigned_count ? 0 : UINT32_MAX;
    bool assigned = true;
    IREE_EXPECT_OK(loom_low_allocation_coalescing_assign_structural_interval(
        &context, &intervals[2], &assigned));
    EXPECT_FALSE(assigned);
    EXPECT_EQ(assignment_map.assignment_count, assigned_count);
    EXPECT_EQ(assignment_indices[2], UINT32_MAX);
  }

  for (loom_value_id_t value_id : value_ids) {
    loom_module_value_ordinal_scratch_clear(module, value_id);
  }
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
}

TEST_F(LowAllocationCoalescingTest,
       OptionalStructuralAliasMayPrecedeItsSourceAssignment) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t value_ids[] = {DefineValue(module),
                                       DefineValue(module)};
  loom_module_value_ordinal_scratch_acquire(module);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(value_ids); ++i) {
    loom_module_value_ordinal_scratch_set(module, value_ids[i], i);
  }

  const loom_liveness_value_class_t value_class = RegisterValueClass(17);
  loom_liveness_interval_t intervals[] = {
      Interval(value_ids[0], /*start=*/10, /*end=*/12, value_class),
      Interval(value_ids[1], /*start=*/2, /*end=*/14, value_class),
  };
  const uint32_t interval_indices[] = {0, 1};
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);
  liveness.value_ids = value_ids;
  liveness.value_count = IREE_ARRAYSIZE(value_ids);
  liveness.value_interval_indices = interval_indices;

  loom_low_placement_relation_t relation = {};
  relation.result_ordinal = 1;
  relation.source_ordinal = 0;
  relation.unit_count = 1;
  relation.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED |
                   LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  const loom_low_placement_relation_range_t result_ranges[] = {{0, 0}, {0, 1}};
  const loom_low_placement_relation_range_t source_ranges[] = {{0, 1}, {1, 0}};
  const uint32_t source_relations[] = {0};
  loom_low_placement_table_t placement = {};
  placement.value_ids = value_ids;
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = &relation;
  placement.relation_count = 1;
  placement.ranges_by_result_ordinal = result_ranges;
  placement.ranges_by_source_ordinal = source_ranges;
  placement.relation_indices_by_source_ordinal = source_relations;

  uint32_t assignment_indices[] = {UINT32_MAX, UINT32_MAX};
  loom_low_allocation_assignment_map_t assignment_map = {};
  assignment_map.module = module;
  assignment_map.liveness = &liveness;
  assignment_map.assignment_indices_by_value_ordinal = assignment_indices;

  loom_low_allocation_coalescing_context_t context = {};
  context.arena = &arena_;
  context.liveness = &liveness;
  context.placement = &placement;
  context.assignment_map = &assignment_map;

  for (loom_low_placement_cause_t cause :
       {LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY, LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE,
        LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE}) {
    SCOPED_TRACE(cause);
    const bool is_slice = cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE;
    intervals[0].unit_count = is_slice ? 2 : 1;
    relation.kind = is_slice ? LOOM_LOW_PLACEMENT_RELATION_SUBRANGE
                             : LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    relation.cause = cause;
    relation.source_unit_offset = is_slice ? 1 : 0;
    bool assigned = true;
    IREE_EXPECT_OK(loom_low_allocation_coalescing_assign_structural_interval(
        &context, &intervals[1], &assigned));
    EXPECT_FALSE(assigned);
    EXPECT_EQ(assignment_map.assignment_count, 0u);
    EXPECT_EQ(assignment_indices[0], UINT32_MAX);
    EXPECT_EQ(assignment_indices[1], UINT32_MAX);
  }

  for (loom_value_id_t value_id : value_ids) {
    loom_module_value_ordinal_scratch_clear(module, value_id);
  }
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
