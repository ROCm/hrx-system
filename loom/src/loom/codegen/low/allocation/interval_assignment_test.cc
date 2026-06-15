// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/interval_assignment.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

class LowAllocationIntervalAssignmentTest : public ::testing::Test {
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

TEST_F(LowAllocationIntervalAssignmentTest,
       AssignsRegisterIntervalToFirstFreeLocation) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t value = DefineValue(module);
  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, value, /*ordinal=*/0);

  const uint64_t descriptor_set_id = 17;
  const loom_liveness_value_class_t value_class =
      RegisterValueClass(descriptor_set_id);
  const loom_liveness_interval_t interval = {
      /*.value_id=*/value,
      /*.start_point=*/0,
      /*.end_point=*/8,
      /*.value_class=*/value_class,
      /*.unit_count=*/2,
  };
  const loom_value_id_t value_ids[] = {value};
  const uint32_t interval_indices[] = {0};
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = &interval;
  liveness.interval_count = 1;
  liveness.value_ids = value_ids;
  liveness.value_count = IREE_ARRAYSIZE(value_ids);
  liveness.value_interval_indices = interval_indices;

  uint32_t unit_end_point_start[] = {0};
  uint32_t unit_end_points[] = {8, 8};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.end_point_starts_by_value_ordinal = unit_end_point_start;
  unit_liveness.end_points = unit_end_points;
  unit_liveness.end_point_count = IREE_ARRAYSIZE(unit_end_points);

  loom_low_placement_relation_range_t placement_ranges[] = {
      {
          /*.start=*/0,
          /*.count=*/0,
      },
  };
  loom_low_placement_table_t placement = {};
  placement.value_ids = value_ids;
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.ranges_by_result_ordinal = placement_ranges;
  placement.ranges_by_source_ordinal = placement_ranges;

  loom_low_reg_class_t reg_class = {};
  reg_class.flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL;
  reg_class.alloc_unit_bits = 32;
  reg_class.allocatable_count = 4;
  reg_class.spill_class_id = LOOM_LOW_REG_CLASS_NONE;
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.stable_id = descriptor_set_id;
  descriptor_set.reg_classes = &reg_class;
  descriptor_set.reg_class_count = 1;
  loom_low_resolved_target_t target = {};
  target.descriptor_set = &descriptor_set;
  target.descriptor_set_key = IREE_SV("test");
  loom_op_t function_op = {};
  loom_low_allocation_target_constraints_t target_constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      module, &function_op, &target, /*budgets=*/nullptr,
      /*budget_count=*/0, /*reserved_ranges=*/nullptr,
      /*reserved_range_count=*/0, /*emitter=*/iree_diagnostic_emitter_t{},
      &arena_, &target_constraints));

  loom_low_allocation_storage_lease_state_t storage_leases = {};
  const loom_low_allocation_interval_assignment_context_t context = {
      /*.module=*/module,
      /*.body=*/nullptr,
      /*.function_op=*/&function_op,
      /*.target=*/&target,
      /*.liveness=*/&liveness,
      /*.placement=*/&placement,
      /*.target_constraints=*/&target_constraints,
      /*.unit_liveness=*/&unit_liveness,
      /*.storage_leases=*/&storage_leases,
      /*.arena=*/&arena_,
  };
  loom_low_allocation_interval_assignment_result_t result = {};
  IREE_ASSERT_OK(
      loom_low_allocation_interval_assignment_build(&context, &result));

  ASSERT_EQ(result.assignment_count, 1u);
  ASSERT_NE(result.assignments, nullptr);
  EXPECT_EQ(result.assignment_indices_by_value_ordinal[0], 0u);
  EXPECT_EQ(result.assignments[0].value_id, value);
  EXPECT_EQ(result.assignments[0].descriptor_reg_class_id, 0u);
  EXPECT_EQ(result.assignments[0].location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(result.assignments[0].location_base, 0u);
  EXPECT_EQ(result.assignments[0].location_count, 2u);
  EXPECT_EQ(result.assignments[0].unit_end_point_start, 0u);
  EXPECT_EQ(result.assignments[0].end_point, 8u);

  loom_module_value_ordinal_scratch_clear(module, value);
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
}

TEST_F(LowAllocationIntervalAssignmentTest,
       AssignsTiedFixedValuesToSharedLocation) {
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
      {
          /*.value_id=*/source_value,
          /*.start_point=*/0,
          /*.end_point=*/6,
          /*.value_class=*/value_class,
          /*.unit_count=*/1,
      },
      {
          /*.value_id=*/result_value,
          /*.start_point=*/2,
          /*.end_point=*/6,
          /*.value_class=*/value_class,
          /*.unit_count=*/1,
      },
  };
  loom_value_id_t value_ids[] = {source_value, result_value};
  const uint32_t interval_indices[] = {0, 1};
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);
  liveness.value_ids = value_ids;
  liveness.value_count = IREE_ARRAYSIZE(value_ids);
  liveness.value_interval_indices = interval_indices;

  uint32_t unit_end_point_starts[] = {0, 1};
  uint32_t unit_end_points[] = {6, 6};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.end_point_starts_by_value_ordinal = unit_end_point_starts;
  unit_liveness.end_points = unit_end_points;
  unit_liveness.end_point_count = IREE_ARRAYSIZE(unit_end_points);

  const uint32_t relation_indices_by_source[] = {0};
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
          /*.count=*/1,
      },
      {
          /*.start=*/1,
          /*.count=*/0,
      },
  };
  loom_low_placement_table_t placement = {};
  placement.value_ids = value_ids;
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = &relation;
  placement.relation_count = 1;
  placement.ranges_by_result_ordinal = ranges_by_result;
  placement.relation_indices_by_source_ordinal = relation_indices_by_source;
  placement.ranges_by_source_ordinal = ranges_by_source;

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
  loom_op_t function_op = {};
  loom_low_allocation_target_constraints_t target_constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      module, &function_op, &target, /*budgets=*/nullptr,
      /*budget_count=*/0, /*reserved_ranges=*/nullptr,
      /*reserved_range_count=*/0, /*emitter=*/iree_diagnostic_emitter_t{},
      &arena_, &target_constraints));
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {
          /*.value_id=*/source_value,
          /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          /*.location_base=*/3,
          /*.location_count=*/1,
      },
      {
          /*.value_id=*/result_value,
          /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          /*.location_base=*/3,
          /*.location_count=*/1,
      },
  };
  loom_local_value_domain_t value_domain = {};
  value_domain.module = module;
  value_domain.value_ids = value_ids;
  value_domain.value_count = IREE_ARRAYSIZE(value_ids);
  value_domain.flags = LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED;
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_resolve_fixed_values(
      &target_constraints, &liveness, &value_domain, fixed_values,
      IREE_ARRAYSIZE(fixed_values), &arena_));

  loom_low_allocation_storage_lease_state_t storage_leases = {};
  const loom_low_allocation_interval_assignment_context_t context = {
      /*.module=*/module,
      /*.body=*/nullptr,
      /*.function_op=*/&function_op,
      /*.target=*/&target,
      /*.liveness=*/&liveness,
      /*.placement=*/&placement,
      /*.target_constraints=*/&target_constraints,
      /*.unit_liveness=*/&unit_liveness,
      /*.storage_leases=*/&storage_leases,
      /*.arena=*/&arena_,
  };
  loom_low_allocation_interval_assignment_result_t result = {};
  IREE_ASSERT_OK(
      loom_low_allocation_interval_assignment_build(&context, &result));

  ASSERT_EQ(result.assignment_count, 2u);
  EXPECT_EQ(result.assignment_indices_by_value_ordinal[0], 0u);
  EXPECT_EQ(result.assignment_indices_by_value_ordinal[1], 1u);
  EXPECT_EQ(result.assignments[0].value_id, source_value);
  EXPECT_EQ(result.assignments[0].location_base, 3u);
  EXPECT_EQ(result.assignments[1].value_id, result_value);
  EXPECT_EQ(result.assignments[1].location_base, 3u);

  loom_module_value_ordinal_scratch_clear(module, source_value);
  loom_module_value_ordinal_scratch_clear(module, result_value);
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
