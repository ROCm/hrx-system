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
#include "loom/ops/test/ops.h"
#include "loom/target/registers.h"
#include "loom/util/cfg_graph.h"

namespace loom {
namespace {

class LowAllocationIntervalAssignmentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    iree_arena_initialize(&block_pool_, &decision_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&decision_arena_);
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

  // Shared pool for published results and allocation decision storage.
  iree_arena_block_pool_t block_pool_;
  // Owns prerequisite analyses and published assignment tables.
  iree_arena_allocator_t arena_;
  // Owns allocation decisions and checkpoint-scoped assignment scratch.
  iree_arena_allocator_t decision_arena_;
  // Context for the module-local value domain.
  loom_context_t context_;
};

loom_liveness_value_class_t RegisterValueClass(uint64_t descriptor_set_id) {
  loom_liveness_value_class_t value_class = {
      .type_kind = LOOM_TYPE_REGISTER,
      .register_class_id = 0,
      .register_descriptor_set_stable_id = descriptor_set_id,
  };
  return value_class;
}

TEST_F(LowAllocationIntervalAssignmentTest,
       AssignsOrTrafficBoundsSingleRegisterInterval) {
  loom_module_t* module = AllocateModule();
  const uint64_t descriptor_set_id = 17;
  loom_builder_t builder = {};
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* producer_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder, loom_attr_i64(0),
      loom_low_register_type(descriptor_set_id, /*register_class_id=*/0,
                             /*unit_count=*/2),
      LOOM_LOCATION_UNKNOWN, &producer_op));
  const loom_value_id_t value = loom_test_constant_result(producer_op);
  loom_op_t* use_op = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder, &value, /*values_count=*/1,
                                     LOOM_LOCATION_UNKNOWN, &use_op));
  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, value, /*ordinal=*/0);

  const loom_liveness_value_class_t value_class =
      RegisterValueClass(descriptor_set_id);
  const loom_liveness_interval_t interval = {
      .value_id = value,
      .start_point = 0,
      .end_point = 8,
      .value_class = value_class,
      .unit_count = 2,
  };
  loom_value_id_t value_ids[] = {value};
  loom_local_value_domain_t value_domain = {
      .module = module,
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .flags = LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED,
  };
  const uint32_t interval_indices[] = {0};
  loom_liveness_analysis_t liveness = {
      .region = module->body,
      .intervals = &interval,
      .interval_count = 1,
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .value_interval_indices = interval_indices,
  };

  loom_low_allocation_unit_liveness_value_t unit_values[] = {{0, 0}};
  uint32_t unit_start_points[] = {0, 0};
  uint32_t unit_end_points[] = {8, 8};
  uint64_t edge_handoff_words[] = {0};
  loom_low_allocation_unit_liveness_t unit_liveness = {
      .values = unit_values,
      .start_points = unit_start_points,
      .end_points = unit_end_points,
      .point_count = IREE_ARRAYSIZE(unit_end_points),
      .values_with_incomplete_storage_segments = {liveness.value_count,
                                                  edge_handoff_words},
  };

  loom_low_placement_relation_range_t placement_ranges[] = {
      {
          .start = 0,
          .count = 0,
      },
  };
  loom_low_placement_table_t placement = {
      .module = module,
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .ranges_by_result_ordinal = placement_ranges,
      .ranges_by_source_ordinal = placement_ranges,
  };

  loom_low_reg_class_t reg_class = {
      .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
      .alloc_unit_bits = 32,
      .allocatable_count = 4,
      .spill_class_id = LOOM_LOW_REG_CLASS_NONE,
  };
  loom_low_descriptor_set_t descriptor_set = {
      .stable_id = descriptor_set_id,
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  loom_low_resolved_target_t target = {
      .descriptor_set = &descriptor_set,
  };
  target.descriptor_set_key = IREE_SV("test");
  loom_op_t function_op = {};
  loom_low_allocation_target_constraints_t target_constraints = {};
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
      module, &function_op, &target, /*budgets=*/nullptr,
      /*budget_count=*/0, /*reserved_ranges=*/nullptr,
      /*reserved_range_count=*/0, /*emitter=*/iree_diagnostic_emitter_t{},
      &arena_, &target_constraints));
  loom_cfg_graph_t function_cfg_graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module, module->body, &arena_, &function_cfg_graph));

  loom_low_allocation_storage_lease_state_t storage_leases = {};
  const loom_low_allocation_interval_assignment_context_t context = {
      .module = module,
      .body = module->body,
      .function_op = &function_op,
      .target = &target,
      .liveness = &liveness,
      .value_domain = &value_domain,
      .schedule = nullptr,
      .placement = &placement,
      .preferences = nullptr,
      .target_constraints = &target_constraints,
      .entry_locations = nullptr,
      .entry_location_count = 0,
      .storage_transport = nullptr,
      .unit_liveness = &unit_liveness,
      .storage_leases = &storage_leases,
      .arena = &arena_,
      .function_cfg_graph = &function_cfg_graph,
  };
  loom_low_allocation_interval_assignment_result_t result = {};
  // The enclosing allocator may retain facts in the scratch arena's prefix.
  uint32_t* retained_fact = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&decision_arena_, sizeof(*retained_fact),
                                     (void**)&retained_fact));
  *retained_fact = 42;
  const iree_host_size_t decision_bytes = decision_arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_low_allocation_interval_assignment_build(
      &context, &decision_arena_, &result));
  EXPECT_EQ(decision_arena_.used_allocation_size, decision_bytes);
  EXPECT_EQ(*retained_fact, 42u);

  // The returned tables remain valid after assignment scratch is reclaimed.
  iree_arena_reset(&decision_arena_);
  iree_arena_block_pool_trim(&block_pool_);
  ASSERT_EQ(result.assignment_count, 1u);
  ASSERT_NE(result.assignments, nullptr);
  EXPECT_EQ(result.assignment_indices_by_value_ordinal[0], 0u);
  EXPECT_EQ(result.assignments[0].value_id, value);
  EXPECT_EQ(result.assignments[0].descriptor_reg_class_id, 0u);
  EXPECT_EQ(result.assignments[0].location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(result.assignments[0].location_base, 0u);
  EXPECT_EQ(result.assignments[0].location_count, 2u);
  EXPECT_EQ(result.assignments[0].unit_point_start, 0u);
  EXPECT_EQ(result.assignments[0].end_point, 8u);
  EXPECT_EQ(result.unit_end_points, unit_end_points);
  EXPECT_EQ(result.spill_count, 0u);
  EXPECT_EQ(result.spill_plan_count, 0u);
  EXPECT_EQ(result.remark_count, 0u);
  EXPECT_EQ(result.spill_plans, nullptr);
  EXPECT_EQ(result.remarks, nullptr);
  EXPECT_TRUE(result.has_packable_aggregates);

  // Replaying a retained location bypasses first-fit search while rebuilding
  // the assignment's current lifetime and indexes.
  loom_low_allocation_assignment_t retained_assignment = result.assignments[0];
  retained_assignment.location_base = 2;
  const loom_low_allocation_assignment_t* retained_assignments[] = {
      &retained_assignment,
  };
  loom_low_allocation_interval_assignment_context_t replay_context = context;
  replay_context.search_strategy =
      LOOM_LOW_ALLOCATION_SEARCH_STRATEGY_RETAINED_LOCATIONS;
  replay_context.retained_assignments_by_value_ordinal = retained_assignments;
  result = {};
  IREE_ASSERT_OK(loom_low_allocation_interval_assignment_build(
      &replay_context, &decision_arena_, &result));
  ASSERT_EQ(result.assignment_count, 1u);
  EXPECT_FALSE(result.retained_location_replay_failed);
  EXPECT_EQ(result.assignments[0].location_base, 2u);
  EXPECT_EQ(result.assignments[0].start_point, 0u);
  EXPECT_EQ(result.assignments[0].end_point, 8u);

  retained_assignments[0] = nullptr;
  result = {};
  IREE_ASSERT_OK(loom_low_allocation_interval_assignment_build(
      &replay_context, &decision_arena_, &result));
  EXPECT_TRUE(result.retained_location_replay_failed);
  EXPECT_EQ(target_constraints.error_count, 0u);

  // A comparison attempt stops as soon as its first irreversible spill cannot
  // improve the incumbent traffic. The same attempt completes when that spill
  // remains below the exclusive bound.
  reg_class.allocatable_count = 1;

  uint64_t spill_traffic_exclusive_limit = 16;
  loom_low_allocation_interval_assignment_context_t bounded_context = context;
  bounded_context.spill_traffic_exclusive_limit =
      &spill_traffic_exclusive_limit;
  result = {};
  IREE_ASSERT_OK(loom_low_allocation_interval_assignment_build(
      &bounded_context, &decision_arena_, &result));
  EXPECT_TRUE(result.reached_spill_traffic_limit);
  EXPECT_EQ(result.assignment_count, 1u);
  EXPECT_EQ(result.spill_count, 1u);
  EXPECT_EQ(result.spill_traffic_bytes, spill_traffic_exclusive_limit);
  EXPECT_EQ(result.spill_plan_count, 0u);
  EXPECT_EQ(result.remark_count, 0u);

  ++spill_traffic_exclusive_limit;
  result = {};
  IREE_ASSERT_OK(loom_low_allocation_interval_assignment_build(
      &bounded_context, &decision_arena_, &result));
  EXPECT_FALSE(result.reached_spill_traffic_limit);
  EXPECT_EQ(result.assignment_count, 1u);
  EXPECT_EQ(result.spill_count, 1u);
  EXPECT_EQ(result.spill_traffic_bytes, 16u);
  EXPECT_EQ(result.spill_plan_count, 1u);
  EXPECT_EQ(result.remark_count, 1u);

  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
}

TEST_F(LowAllocationIntervalAssignmentTest,
       AssignsExplicitRegistersAndOddOrdinalAggregateView) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t first_value = DefineValue(module);
  const loom_value_id_t second_value = DefineValue(module);
  const loom_value_id_t aggregate_value = DefineValue(module);
  loom_module_value_ordinal_scratch_acquire(module);
  loom_module_value_ordinal_scratch_set(module, first_value, /*ordinal=*/0);
  loom_module_value_ordinal_scratch_set(module, second_value, /*ordinal=*/1);
  loom_module_value_ordinal_scratch_set(module, aggregate_value,
                                        /*ordinal=*/2);

  const uint64_t descriptor_set_id = 23;
  const loom_liveness_value_class_t value_class =
      RegisterValueClass(descriptor_set_id);
  const loom_liveness_interval_t intervals[] = {
      {
          .value_id = first_value,
          .start_point = 0,
          .end_point = 4,
          .value_class = value_class,
          .unit_count = 1,
      },
      {
          .value_id = second_value,
          .start_point = 0,
          .end_point = 4,
          .value_class = value_class,
          .unit_count = 1,
      },
      {
          .value_id = aggregate_value,
          .start_point = 5,
          .end_point = 8,
          .value_class = value_class,
          .unit_count = 2,
      },
  };
  loom_value_id_t value_ids[] = {first_value, second_value, aggregate_value};
  loom_local_value_domain_t value_domain = {
      .module = module,
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .flags = LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED,
  };
  const uint32_t interval_indices[] = {0, 1, 2};
  loom_liveness_analysis_t liveness = {
      .intervals = intervals,
      .interval_count = IREE_ARRAYSIZE(intervals),
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .value_interval_indices = interval_indices,
  };

  loom_low_allocation_unit_liveness_value_t unit_values[] = {
      {0, 0}, {1, 0}, {2, 5}};
  uint32_t unit_start_points[] = {0, 0, 5, 5};
  uint32_t unit_end_points[] = {4, 4, 8, 8};
  uint64_t edge_handoff_words[] = {0};
  loom_low_allocation_unit_liveness_t unit_liveness = {
      .values = unit_values,
      .start_points = unit_start_points,
      .end_points = unit_end_points,
      .point_count = IREE_ARRAYSIZE(unit_end_points),
      .values_with_incomplete_storage_segments = {liveness.value_count,
                                                  edge_handoff_words},
  };

  loom_low_placement_relation_range_t placement_ranges[3] = {};
  loom_low_placement_table_t placement = {
      .module = module,
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .ranges_by_result_ordinal = placement_ranges,
      .ranges_by_source_ordinal = placement_ranges,
  };

  loom_low_reg_class_t reg_class = {
      .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
               LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS,
      .alloc_unit_bits = 512,
      .allocatable_count = 2,
      .spill_class_id = LOOM_LOW_REG_CLASS_NONE,
      .physical_atomic_unit_count = 2,
  };
  loom_low_physical_register_t physical_registers[] = {
      {
          .name_string_ref = 0,
          .atomic_unit_start = 0,
          .atomic_unit_count = 2,
          .reserved = 0,
      },
      {
          .name_string_ref = 0,
          .atomic_unit_start = 2,
          .atomic_unit_count = 2,
          .reserved = 0,
      },
      {
          .name_string_ref = 0,
          .atomic_unit_start = 4,
          .atomic_unit_count = 4,
          .reserved = 0,
          .view_lookup =
              {
                  .ordinal_start = 0,
                  .class_base = 0,
                  .class_count = 1,
              },
      },
      {
          .name_string_ref = 0,
          .atomic_unit_start = 8,
          .atomic_unit_count = 4,
          .reserved = 0,
          .view_lookup =
              {
                  .ordinal_start = 1,
                  .class_base = 0,
                  .class_count = 0,
              },
      },
  };
  reg_class.candidate_lookup.register_count = 2;
  const uint16_t candidate_ordinals[] = {1, 0};
  const uint16_t physical_register_candidates[] = {1, 0};
  const uint16_t physical_register_atomic_units[] = {0, 1, 2, 3, 0, 1,
                                                     2, 3, 0, 1, 2, 3};
  const uint32_t view_ordinals[] = {0, 1};
  const loom_low_physical_register_view_t physical_register_views[] = {
      {
          .physical_register_id = 2,
          .reg_class_id = 0,
          .unit_candidate_ordinal_start = 0,
          .unit_count = 2,
          .packing_rank = 0,
      },
      {
          .physical_register_id = 3,
          .reg_class_id = 0,
          .unit_candidate_ordinal_start = 2,
          .unit_count = 2,
          .packing_rank = 0,
      },
  };
  const uint16_t physical_register_view_unit_candidate_ordinals[] = {1, 0, 0,
                                                                     1};
  const uint16_t allocation_ordinals[] = {0, 1};
  loom_low_descriptor_set_t descriptor_set = {
      .stable_id = descriptor_set_id,
      .reg_classes = &reg_class,
      .reg_class_count = 1,
      .physical_registers = physical_registers,
      .physical_register_count = IREE_ARRAYSIZE(physical_registers),
      .physical_register_candidate_ids = physical_register_candidates,
      .physical_register_allocation_ordinals = allocation_ordinals,
      .physical_register_candidate_count =
          IREE_ARRAYSIZE(physical_register_candidates),
      .physical_register_candidate_ordinals = candidate_ordinals,
      .physical_register_candidate_ordinal_count =
          IREE_ARRAYSIZE(candidate_ordinals),
      .physical_register_atomic_units = physical_register_atomic_units,
      .physical_register_atomic_unit_count =
          IREE_ARRAYSIZE(physical_register_atomic_units),
      .physical_register_view_ordinals = view_ordinals,
      .physical_register_view_ordinal_count = 1,
      .physical_register_views = physical_register_views,
      .physical_register_view_count = 1,
      .physical_register_view_unit_candidate_ordinals =
          physical_register_view_unit_candidate_ordinals,
      .physical_register_view_unit_candidate_ordinal_count =
          IREE_ARRAYSIZE(physical_register_view_unit_candidate_ordinals),
  };
  loom_low_resolved_target_t target = {
      .descriptor_set = &descriptor_set,
  };
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
      .module = module,
      .body = nullptr,
      .function_op = &function_op,
      .target = &target,
      .liveness = &liveness,
      .value_domain = &value_domain,
      .schedule = nullptr,
      .placement = &placement,
      .preferences = nullptr,
      .target_constraints = &target_constraints,
      .entry_locations = nullptr,
      .entry_location_count = 0,
      .storage_transport = nullptr,
      .unit_liveness = &unit_liveness,
      .storage_leases = &storage_leases,
      .arena = &arena_,
  };
  loom_low_allocation_interval_assignment_result_t result = {};
  IREE_ASSERT_OK(loom_low_allocation_interval_assignment_build(
      &context, &decision_arena_, &result));

  ASSERT_EQ(result.assignment_count, 3u);
  EXPECT_FALSE(result.has_packable_aggregates);
  EXPECT_EQ(result.assignments[0].location_base, 1u);
  EXPECT_EQ(result.assignments[0].location_count, 1u);
  EXPECT_EQ(result.assignments[1].location_base, 0u);
  EXPECT_EQ(result.assignments[1].location_count, 1u);
  EXPECT_EQ(result.assignments[2].location_base, 2u);
  EXPECT_EQ(result.assignments[2].location_count, 2u);

  // A later physical ID can have the preferred first-unit candidate ordinal.
  // Aggregate search must inspect it even after finding a zero-penalty view.
  physical_registers[3].view_lookup.class_count = 1;
  descriptor_set.physical_register_view_ordinal_count =
      IREE_ARRAYSIZE(view_ordinals);
  descriptor_set.physical_register_view_count =
      IREE_ARRAYSIZE(physical_register_views);
  IREE_ASSERT_OK(loom_low_allocation_interval_assignment_build(
      &context, &decision_arena_, &result));
  ASSERT_EQ(result.assignment_count, 3u);
  EXPECT_EQ(result.assignments[0].location_base, 1u);
  EXPECT_EQ(result.assignments[1].location_base, 0u);
  EXPECT_EQ(result.assignments[2].location_base, 3u);
  EXPECT_EQ(result.assignments[2].location_count, 2u);

  // Current liveness rejects a retained location that aliases an active value
  // without turning the speculative replay into a user-facing target error.
  loom_low_allocation_assignment_t retained_assignments[] = {
      result.assignments[0],
      result.assignments[1],
      result.assignments[2],
  };
  retained_assignments[1].location_base = retained_assignments[0].location_base;
  const loom_low_allocation_assignment_t* retained_by_value_ordinal[] = {
      &retained_assignments[0],
      &retained_assignments[1],
      &retained_assignments[2],
  };
  loom_low_allocation_interval_assignment_context_t replay_context = context;
  replay_context.search_strategy =
      LOOM_LOW_ALLOCATION_SEARCH_STRATEGY_RETAINED_LOCATIONS;
  replay_context.retained_assignments_by_value_ordinal =
      retained_by_value_ordinal;
  result = {};
  IREE_ASSERT_OK(loom_low_allocation_interval_assignment_build(
      &replay_context, &decision_arena_, &result));
  EXPECT_TRUE(result.retained_location_replay_failed);
  EXPECT_EQ(target_constraints.error_count, 0u);

  loom_local_value_domain_release(&value_domain);
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
          .value_id = source_value,
          .start_point = 0,
          .end_point = 6,
          .value_class = value_class,
          .unit_count = 1,
      },
      {
          .value_id = result_value,
          .start_point = 2,
          .end_point = 6,
          .value_class = value_class,
          .unit_count = 1,
      },
  };
  loom_value_id_t value_ids[] = {source_value, result_value};
  const uint32_t interval_indices[] = {0, 1};
  loom_liveness_analysis_t liveness = {
      .intervals = intervals,
      .interval_count = IREE_ARRAYSIZE(intervals),
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .value_interval_indices = interval_indices,
  };

  loom_low_allocation_unit_liveness_value_t unit_values[] = {{0, 0}, {1, 2}};
  uint32_t unit_start_points[] = {0, 0};
  uint32_t unit_end_points[] = {6, 6};
  uint64_t edge_handoff_words[] = {0};
  loom_low_allocation_unit_liveness_t unit_liveness = {
      .values = unit_values,
      .start_points = unit_start_points,
      .end_points = unit_end_points,
      .point_count = IREE_ARRAYSIZE(unit_end_points),
      .values_with_incomplete_storage_segments = {liveness.value_count,
                                                  edge_handoff_words},
  };

  const uint32_t relation_indices_by_source[] = {0};
  loom_low_placement_relation_t relation = {
      .result_ordinal = 1,
      .source_ordinal = 0,
      .unit_count = 1,
      .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE,
      .cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT,
      .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
               LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
  };
  loom_low_placement_relation_range_t ranges_by_result[] = {
      {
          .start = 0,
          .count = 0,
      },
      {
          .start = 0,
          .count = 1,
      },
  };
  loom_low_placement_relation_range_t ranges_by_source[] = {
      {
          .start = 0,
          .count = 1,
      },
      {
          .start = 1,
          .count = 0,
      },
  };
  loom_low_placement_table_t placement = {
      .module = module,
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .relations = &relation,
      .relation_count = 1,
      .ranges_by_result_ordinal = ranges_by_result,
      .relation_indices_by_source_ordinal = relation_indices_by_source,
      .ranges_by_source_ordinal = ranges_by_source,
  };
  const loom_value_ordinal_t tied_origins[] = {0, 0};
  placement.tied_storage_origins_by_value_ordinal = tied_origins;
  const loom_value_ordinal_t storage_order[] = {1, 0};
  placement.storage_value_order = storage_order;
  placement.storage_value_order_count = IREE_ARRAYSIZE(storage_order);

  loom_low_reg_class_t reg_class = {
      .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
      .alloc_unit_bits = 32,
      .allocatable_count = 8,
      .spill_class_id = LOOM_LOW_REG_CLASS_NONE,
  };
  loom_low_descriptor_set_t descriptor_set = {
      .stable_id = descriptor_set_id,
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  loom_low_resolved_target_t target = {
      .descriptor_set = &descriptor_set,
  };
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
          .value_id = source_value,
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_base = 3,
          .location_count = 1,
      },
      {
          .value_id = result_value,
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_base = 3,
          .location_count = 1,
      },
  };
  loom_local_value_domain_t value_domain = {
      .module = module,
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .flags = LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED,
  };
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_resolve_fixed_locations(
      &target_constraints, &liveness, &value_domain,
      placement.operand_constraints_by_interval,
      placement.tied_storage_origins_by_value_ordinal, fixed_values,
      IREE_ARRAYSIZE(fixed_values), &arena_));
  IREE_ASSERT_OK(loom_low_allocation_target_constraints_finalize_fixed_values(
      &target_constraints, &liveness, &unit_liveness, &arena_));

  loom_low_allocation_storage_lease_state_t storage_leases = {};
  const loom_low_allocation_interval_assignment_context_t context = {
      .module = module,
      .body = nullptr,
      .function_op = &function_op,
      .target = &target,
      .liveness = &liveness,
      .value_domain = &value_domain,
      .schedule = nullptr,
      .placement = &placement,
      .preferences = nullptr,
      .target_constraints = &target_constraints,
      .entry_locations = nullptr,
      .entry_location_count = 0,
      .storage_transport = nullptr,
      .unit_liveness = &unit_liveness,
      .storage_leases = &storage_leases,
      .arena = &arena_,
  };
  loom_low_allocation_interval_assignment_result_t result = {};
  IREE_ASSERT_OK(loom_low_allocation_interval_assignment_build(
      &context, &decision_arena_, &result));

  ASSERT_EQ(result.assignment_count, 2u);
  EXPECT_EQ(result.assignment_indices_by_value_ordinal[0], 0u);
  EXPECT_EQ(result.assignment_indices_by_value_ordinal[1], 1u);
  EXPECT_EQ(result.assignments[0].value_id, source_value);
  EXPECT_EQ(result.assignments[0].location_base, 3u);
  EXPECT_EQ(result.assignments[1].value_id, result_value);
  EXPECT_EQ(result.assignments[1].location_base, 3u);
  EXPECT_FALSE(result.has_packable_aggregates);

  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
}

// Four required owners occupy the low window before a fifth required value
// arrives. The earlier fixed interval distinguishes current availability from
// availability over a moved owner's complete lifetime.
class LowRequiredRegisterRelocationTest
    : public LowAllocationIntervalAssignmentTest {
 protected:
  void Run() {
    loom_module_t* module = AllocateModule();
    constexpr uint64_t kDescriptorSetId = 29;
    constexpr uint32_t kValueCount = 6;
    loom_value_id_t values[kValueCount];
    uint32_t interval_indices[kValueCount];
    loom_liveness_interval_t intervals[kValueCount] = {};
    loom_low_allocation_unit_liveness_value_t unit_values[kValueCount] = {};
    uint32_t starts[kValueCount] = {0, 1, 2, 3, 4, 6};
    uint32_t ends[kValueCount] = {prior_end_, 12, 12, 12, 12, 12};
    uint64_t required_bits = 0;
    loom_module_value_ordinal_scratch_acquire(module);
    for (uint32_t i = 0; i < kValueCount; ++i) {
      const uint16_t class_id = alias_class_ && i == 5 ? 1 : 0;
      IREE_ASSERT_OK(loom_module_define_value(
          module, loom_low_register_type(kDescriptorSetId, class_id, 1),
          &values[i]));
      loom_module_value_ordinal_scratch_set(module, values[i], i);
      interval_indices[i] = i;
      intervals[i] = {
          .value_id = values[i],
          .start_point = starts[i],
          .end_point = ends[i],
          .value_class =
              {
                  .type_kind = LOOM_TYPE_REGISTER,
                  .register_class_id = class_id,
                  .register_descriptor_set_stable_id = kDescriptorSetId,
              },
          .unit_count = 1,
      };
      unit_values[i] = {i, starts[i]};
      required_bits |= UINT64_C(1) << values[i];
    }
    loom_local_value_domain_t value_domain = {
        .module = module,
        .value_ids = values,
        .value_count = kValueCount,
        .flags = LOOM_LOCAL_VALUE_DOMAIN_FLAG_ACQUIRED,
    };
    loom_liveness_analysis_t liveness = {
        .intervals = intervals,
        .interval_count = kValueCount,
        .value_ids = values,
        .value_count = kValueCount,
        .value_interval_indices = interval_indices,
    };
    uint64_t incomplete_storage = 0;
    loom_low_allocation_unit_liveness_t unit_liveness = {
        .values = unit_values,
        .start_points = starts,
        .end_points = ends,
        .point_count = kValueCount,
        .values_with_incomplete_storage_segments = {kValueCount,
                                                    &incomplete_storage},
    };
    loom_low_placement_relation_range_t ranges[kValueCount] = {};
    loom_low_placement_operand_constraints_t operands[kValueCount] = {};
    operands[5].addressable_unit_count = 4;
    if (constrain_blockers_) {
      for (uint32_t i = 1; i < 5; ++i) {
        operands[i].addressable_unit_count = 4;
      }
    }
    loom_low_placement_table_t placement = {
        .module = module,
        .value_ids = values,
        .value_count = kValueCount,
        .ranges_by_result_ordinal = ranges,
        .ranges_by_source_ordinal = ranges,
        .operand_constraints_by_interval = operands,
    };
    const loom_low_reg_class_t classes[] = {
        {
            .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
            .alloc_unit_bits = 64,
            .allocatable_count = capacity_,
            .alias_set_id = 1,
            .spill_class_id = LOOM_LOW_REG_CLASS_NONE,
        },
        {
            .flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
            .alloc_unit_bits = 32,
            .allocatable_count = 4,
            .alias_set_id = 1,
            .spill_class_id = LOOM_LOW_REG_CLASS_NONE,
        },
    };
    const loom_low_descriptor_set_t descriptors = {
        .stable_id = kDescriptorSetId,
        .reg_classes = classes,
        .reg_class_count = IREE_ARRAYSIZE(classes),
    };
    const loom_low_resolved_target_t target = {
        .descriptor_set = &descriptors,
    };
    loom_op_t function = {};
    loom_low_allocation_target_constraints_t constraints = {};
    IREE_ASSERT_OK(loom_low_allocation_target_constraints_initialize(
        module, &function, &target, nullptr, 0, nullptr, 0, {}, &arena_,
        &constraints));
    loom_low_allocation_fixed_value_t fixed[5] = {
        {
            .value_id = values[0],
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 4,
            .location_count = 1,
        },
    };
    iree_host_size_t fixed_count = 1;
    if (fix_blockers_) {
      for (uint32_t i = 1; i < 5; ++i) {
        fixed[fixed_count++] = {
            .value_id = values[i],
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = i - 1,
            .location_count = 1,
        };
      }
    }
    IREE_ASSERT_OK(
        loom_low_allocation_target_constraints_resolve_fixed_locations(
            &constraints, &liveness, &value_domain, operands, nullptr, fixed,
            fixed_count, &arena_));
    IREE_ASSERT_OK(loom_low_allocation_target_constraints_finalize_fixed_values(
        &constraints, &liveness, &unit_liveness, &arena_));
    loom_low_allocation_storage_lease_state_t leases = {};
    const loom_low_allocation_interval_assignment_context_t context = {
        .module = module,
        .function_op = &function,
        .target = &target,
        .liveness = &liveness,
        .value_domain = &value_domain,
        .placement = &placement,
        .target_constraints = &constraints,
        .unit_liveness = &unit_liveness,
        .storage_leases = &leases,
        .arena = &arena_,
        .required_register_values = {module->values.count, &required_bits},
    };
    loom_low_allocation_interval_assignment_result_t result = {};
    IREE_ASSERT_OK(loom_low_allocation_interval_assignment_build(
        &context, &decision_arena_, &result));
    error_count_ = constraints.error_count;
    if (!error_count_) {
      ASSERT_EQ(result.assignment_count, kValueCount);
      EXPECT_EQ(result.spill_count, 0u);
      for (uint32_t i = 0; i < kValueCount; ++i) {
        assignments_[i] =
            result.assignments[result.assignment_indices_by_value_ordinal[i]];
      }
    }
    loom_local_value_domain_release(&value_domain);
    loom_module_free(module);
  }

  void CheckMoved(uint32_t destination) {
    ASSERT_EQ(error_count_, 0u);
    ASSERT_LT(assignments_[5].location_base, 4u);
    uint64_t occupied = UINT64_C(1) << assignments_[5].location_base;
    uint32_t moved = 0;
    for (uint32_t i = 1; i < 5; ++i) {
      const auto& assignment = assignments_[i];
      EXPECT_EQ(assignment.location_kind,
                LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
      EXPECT_TRUE(iree_any_bit_set(
          assignment.flags,
          LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REQUIRES_REGISTER));
      EXPECT_EQ(occupied & (UINT64_C(1) << assignment.location_base), 0u);
      occupied |= UINT64_C(1) << assignment.location_base;
      if (assignment.location_base >= 4) {
        EXPECT_EQ(assignment.location_base, destination);
        ++moved;
      }
    }
    EXPECT_EQ(moved, 1u);
  }

  // Physical extent of the owners' register class.
  uint16_t capacity_ = 5;
  // End point of the earlier value fixed above the low window.
  uint32_t prior_end_ = 1;
  // Whether each blocking owner also has a low-window instruction constraint.
  bool constrain_blockers_ = false;
  // Whether immutable fixed-location claims protect all blocking owners.
  bool fix_blockers_ = false;
  // Whether the incoming value uses a distinct class sharing the same bank.
  bool alias_class_ = false;
  // Placement errors reported by the production assignment attempt.
  uint32_t error_count_ = 0;
  // Copies of completed assignments indexed by the fixture's value ordinal.
  loom_low_allocation_assignment_t assignments_[6] = {};
};

TEST_F(LowRequiredRegisterRelocationTest, MovesAboveInstructionWindow) {
  Run();
  CheckMoved(4);
}

TEST_F(LowRequiredRegisterRelocationTest, ChecksExpiredAssignments) {
  capacity_ = 6;
  prior_end_ = 5;
  Run();
  CheckMoved(5);
}

TEST_F(LowRequiredRegisterRelocationTest, RejectsEarlierInterference) {
  prior_end_ = 5;
  Run();
  EXPECT_NE(error_count_, 0u);
}

TEST_F(LowRequiredRegisterRelocationTest, PreservesOwnerInstructionLimits) {
  constrain_blockers_ = true;
  Run();
  EXPECT_NE(error_count_, 0u);
}

TEST_F(LowRequiredRegisterRelocationTest, PreservesFixedLocations) {
  fix_blockers_ = true;
  Run();
  EXPECT_NE(error_count_, 0u);
}

TEST_F(LowRequiredRegisterRelocationTest, MovesAliasClassOwners) {
  alias_class_ = true;
  Run();
  CheckMoved(4);
}

}  // namespace
}  // namespace loom
