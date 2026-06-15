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

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
};

struct AppendState {
  loom_low_allocation_assignment_t* assignments;
  iree_host_size_t assignment_capacity;
  iree_host_size_t assignment_count;
  uint32_t* assignment_indices_by_value_ordinal;
  loom_low_allocation_assignment_map_t* assignment_map;
};

static iree_status_t AppendAssignment(
    void* user_data, const loom_low_allocation_assignment_t* assignment,
    const loom_value_id_t* ignored_storage_lease_value_ids,
    uint16_t ignored_storage_lease_value_count,
    uint32_t* out_assignment_index) {
  (void)ignored_storage_lease_value_ids;
  (void)ignored_storage_lease_value_count;
  AppendState* state = (AppendState*)user_data;
  if (state->assignment_count >= state->assignment_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "test assignment capacity exhausted");
  }
  const uint32_t assignment_index = (uint32_t)state->assignment_count;
  state->assignments[state->assignment_count++] = *assignment;
  state->assignment_map->assignment_count = state->assignment_count;
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_assignment_map_value_ordinal_for_value(
          state->assignment_map, assignment->value_id, &value_ordinal)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test assignment value is outside the map");
  }
  state->assignment_indices_by_value_ordinal[value_ordinal] = assignment_index;
  if (out_assignment_index) {
    *out_assignment_index = assignment_index;
  }
  return iree_ok_status();
}

static iree_status_t UnexpectedConsumptionQuery(
    void* user_data, loom_consumption_region_query_t** out_query) {
  (void)user_data;
  *out_query = nullptr;
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "test did not expect a consumption query");
}

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
    uint32_t unit_end_point_start) {
  loom_low_allocation_assignment_t assignment = {};
  assignment.value_id = value_id;
  assignment.value_class = value_class;
  assignment.descriptor_reg_class_id = 0;
  assignment.start_point = start;
  assignment.end_point = end;
  assignment.unit_count = 1;
  assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignment.location_base = location_base;
  assignment.location_count = 1;
  assignment.unit_end_point_start = unit_end_point_start;
  return assignment;
}

TEST_F(LowAllocationCoalescingTest, AssignsTiedIntervalToSourceLocation) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t source_value = DefineValue(module);
  const loom_value_id_t result_value = DefineValue(module);
  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, source_value,
                                        /*ordinal=*/0);
  loom_module_value_ordinal_scratch_set(module, result_value,
                                        /*ordinal=*/1);

  const uint64_t descriptor_set_id = 17;
  const loom_liveness_value_class_t value_class =
      RegisterValueClass(descriptor_set_id);
  const loom_liveness_interval_t intervals[] = {
      Interval(source_value, /*start=*/0, /*end=*/8, value_class),
      Interval(result_value, /*start=*/2, /*end=*/6, value_class),
  };
  const uint32_t interval_indices[] = {0, 1};
  const loom_value_id_t value_ids[] = {source_value, result_value};
  const loom_liveness_block_info_t blocks[] = {
      {
          /*.block=*/nullptr,
          /*.start_point=*/0,
          /*.end_point=*/10,
          /*.live_in_values=*/nullptr,
          /*.live_in_count=*/0,
          /*.live_out_values=*/nullptr,
          /*.live_out_count=*/0,
      },
  };
  loom_liveness_analysis_t liveness = {};
  liveness.blocks = blocks;
  liveness.block_count = IREE_ARRAYSIZE(blocks);
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);
  liveness.value_ids = value_ids;
  liveness.value_count = IREE_ARRAYSIZE(value_ids);
  liveness.value_interval_indices = interval_indices;

  uint32_t unit_end_point_starts[] = {0, 1};
  uint32_t unit_end_points[] = {8, 6};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.end_point_starts_by_value_ordinal = unit_end_point_starts;
  unit_liveness.end_points = unit_end_points;
  unit_liveness.end_point_count = IREE_ARRAYSIZE(unit_end_points);

  loom_low_reg_class_t reg_class = {};
  reg_class.flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL;
  reg_class.alloc_unit_bits = 32;
  reg_class.allocatable_count = 8;
  reg_class.spill_class_id = LOOM_LOW_REG_CLASS_NONE;
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.stable_id = descriptor_set_id;
  descriptor_set.reg_classes = &reg_class;
  descriptor_set.reg_class_count = 1;
  loom_low_resolved_target_t target = {};
  target.descriptor_set = &descriptor_set;
  target.descriptor_set_key = IREE_SV("test");
  uint32_t max_assigned_location_end_by_reg_class[] = {4};
  loom_low_allocation_target_constraints_t target_constraints = {};
  target_constraints.target = &target;
  target_constraints.max_assigned_location_end_by_reg_class =
      max_assigned_location_end_by_reg_class;

  loom_low_placement_relation_t relation = {};
  relation.result_ordinal = 1;
  relation.source_ordinal = 0;
  relation.unit_count = 1;
  relation.kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
  relation.cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
  relation.flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD;
  loom_low_placement_relation_range_t ranges_by_result[] = {
      {
          /*.start=*/0,
          /*.count=*/0,
      },
      {
          /*.start=*/0,
          /*.count=*/1,
      },
  };
  loom_low_placement_relation_range_t ranges_by_source[] = {
      {
          /*.start=*/0,
          /*.count=*/0,
      },
      {
          /*.start=*/0,
          /*.count=*/0,
      },
  };
  loom_low_placement_table_t placement = {};
  placement.value_ids = value_ids;
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = &relation;
  placement.relation_count = 1;
  placement.ranges_by_result_ordinal = ranges_by_result;
  placement.ranges_by_source_ordinal = ranges_by_source;

  loom_low_allocation_assignment_t assignments[2] = {
      Assignment(source_value, /*start=*/0, /*end=*/8, value_class,
                 /*location_base=*/3, /*unit_end_point_start=*/0),
  };
  uint32_t assignment_indices_by_value_ordinal[] = {0, UINT32_MAX};
  loom_low_allocation_assignment_map_t assignment_map = {};
  assignment_map.module = module;
  assignment_map.liveness = &liveness;
  assignment_map.assignments = assignments;
  assignment_map.assignment_count = 1;
  assignment_map.assignment_indices_by_value_ordinal =
      assignment_indices_by_value_ordinal;
  AppendState append_state = {};
  append_state.assignments = assignments;
  append_state.assignment_capacity = IREE_ARRAYSIZE(assignments);
  append_state.assignment_count = 1;
  append_state.assignment_indices_by_value_ordinal =
      assignment_indices_by_value_ordinal;
  append_state.assignment_map = &assignment_map;

  loom_low_allocation_active_set_t active_set = {};
  IREE_ASSERT_OK(loom_low_allocation_active_set_initialize(
      /*assignment_capacity=*/2, /*unit_capacity=*/8, &arena_, &active_set));
  loom_low_allocation_active_set_insert(&active_set, &descriptor_set,
                                        assignments, /*assignment_count=*/1,
                                        /*assignment_index=*/0);
  loom_low_allocation_storage_lease_state_t storage_leases = {};
  loom_low_allocation_search_context_t search_context = {};
  search_context.module = module;
  search_context.descriptor_set = &descriptor_set;
  search_context.liveness = &liveness;
  search_context.unit_liveness = &unit_liveness;
  search_context.target_constraints = &target_constraints;
  search_context.assignment_map = &assignment_map;
  search_context.active_set = &active_set;
  search_context.storage_leases = &storage_leases;

  loom_low_allocation_coalescing_context_t context = {};
  context.arena = &arena_;
  context.liveness = &liveness;
  context.placement = &placement;
  context.target_constraints = &target_constraints;
  context.assignment_map = &assignment_map;
  context.search_context = &search_context;
  context.append_assignment = AppendAssignment;
  context.consumption_query = UnexpectedConsumptionQuery;
  context.user_data = &append_state;

  bool assigned = false;
  IREE_ASSERT_OK(loom_low_allocation_coalescing_assign_tied_interval(
      &context, &intervals[1], &assigned));
  ASSERT_TRUE(assigned);
  ASSERT_EQ(append_state.assignment_count, 2u);
  EXPECT_EQ(assignment_indices_by_value_ordinal[1], 1u);
  EXPECT_EQ(assignments[1].value_id, result_value);
  EXPECT_EQ(assignments[1].location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(assignments[1].location_base, 3u);
  EXPECT_EQ(assignments[1].location_count, 1u);

  loom_module_value_ordinal_scratch_clear(module, source_value);
  loom_module_value_ordinal_scratch_clear(module, result_value);
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
