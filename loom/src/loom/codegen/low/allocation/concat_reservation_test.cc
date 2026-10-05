// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/concat_reservation.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/target/registers.h"
#include "loom/target/residency.h"

namespace loom {
namespace {

TEST(LowAllocationConcatReservationTest, ChoosesOnlyLegalAssemblies) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  loom_context_t ir_context;
  loom_context_initialize(iree_allocator_system(), &ir_context);
  IREE_CHECK_OK(loom_context_finalize(&ir_context));
  loom_module_t* module = nullptr;
  IREE_CHECK_OK(loom_module_allocate(&ir_context, IREE_SV("test"), &pool,
                                     nullptr, iree_allocator_system(),
                                     &module));
  loom_value_id_t value_ids[4];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(value_ids); ++i) {
    IREE_CHECK_OK(loom_module_define_value(
        module, loom_low_register_type(17, 0, i == 2 ? 4 : 2), &value_ids[i]));
  }
  loom_module_value_ordinal_scratch_acquire(module);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(value_ids); ++i) {
    loom_module_value_ordinal_scratch_set(module, value_ids[i], i);
  }

  loom_liveness_value_class_t value_class = {};
  value_class.type_kind = LOOM_TYPE_REGISTER;
  value_class.register_descriptor_set_stable_id = 17;
  value_class.register_class_id = 0;
  loom_liveness_interval_t intervals[4] = {};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(intervals); ++i) {
    intervals[i].value_id = value_ids[i];
    intervals[i].value_class = value_class;
    intervals[i].unit_count = i == 2 ? 4 : 2;
  }
  intervals[0].start_point = 2;
  intervals[0].end_point = 4;
  intervals[1].start_point = 3;
  intervals[1].end_point = 4;
  intervals[2].start_point = 4;
  intervals[2].end_point = 10;
  const uint32_t interval_indices[] = {0, 1, 2, 3};
  loom_liveness_analysis_t liveness = {};
  liveness.intervals = intervals;
  liveness.interval_count = IREE_ARRAYSIZE(intervals);
  liveness.value_ids = value_ids;
  liveness.value_count = IREE_ARRAYSIZE(value_ids);
  liveness.value_interval_indices = interval_indices;

  loom_low_allocation_unit_liveness_value_t unit_values[] = {
      {0, 2}, {2, 3}, {4, 4}, {8, 5}};
  uint32_t unit_start_points[] = {2, 2, 3, 3, 2, 2, 3, 3, 5, 5};
  uint32_t unit_end_points[] = {4, 4, 4, 4, 10, 10, 10, 10, 14, 14};
  uint64_t incomplete_storage_words[] = {0};
  loom_low_allocation_unit_liveness_t unit_liveness = {};
  unit_liveness.values = unit_values;
  unit_liveness.start_points = unit_start_points;
  unit_liveness.end_points = unit_end_points;
  unit_liveness.point_count = IREE_ARRAYSIZE(unit_end_points);
  unit_liveness.values_with_incomplete_storage_segments = {
      liveness.value_count, incomplete_storage_words};

  loom_low_placement_relation_t relations[2] = {};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(relations); ++i) {
    relations[i].result_ordinal = 2;
    relations[i].source_ordinal = i;
    relations[i].result_unit_offset = i * 2;
    relations[i].unit_count = 2;
    relations[i].kind = LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART;
    relations[i].cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
    relations[i].flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  }
  const loom_low_placement_relation_range_t result_ranges[] = {
      {0, 0}, {0, 0}, {0, 2}, {2, 0}};
  const loom_low_placement_relation_range_t source_ranges[] = {
      {0, 1}, {1, 1}, {2, 0}, {2, 0}};
  const uint32_t source_relations[] = {0, 1};
  loom_low_placement_table_t placement = {};
  placement.value_ids = value_ids;
  placement.value_count = IREE_ARRAYSIZE(value_ids);
  placement.relations = relations;
  placement.relation_count = IREE_ARRAYSIZE(relations);
  placement.ranges_by_result_ordinal = result_ranges;
  placement.ranges_by_source_ordinal = source_ranges;
  placement.relation_indices_by_source_ordinal = source_relations;
  loom_low_placement_operand_constraints_t operands[4] = {};
  operands[0].addressable_unit_count = 16;
  placement.operand_constraints_by_interval = operands;

  loom_low_reg_class_t reg_class = {};
  reg_class.flags = LOOM_LOW_REG_CLASS_FLAG_PHYSICAL;
  reg_class.alloc_unit_bits = 32;
  reg_class.allocatable_count = 16;
  reg_class.spill_class_id = LOOM_LOW_REG_CLASS_NONE;
  loom_low_descriptor_set_t descriptors = {};
  descriptors.stable_id = 17;
  descriptors.reg_classes = &reg_class;
  descriptors.reg_class_count = 1;
  loom_low_resolved_target_t target = {};
  target.descriptor_set = &descriptors;
  loom_low_allocation_resolved_reserved_range_t prefix = {};
  prefix.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  prefix.location_count = 2;
  uint32_t high_water[] = {6};
  uint32_t constrained_high_water[] = {prefix.location_count};
  loom_low_allocation_target_constraints_t constraints = {};
  constraints.module = module;
  constraints.target = &target;
  constraints.reserved_ranges = &prefix;
  constraints.reserved_range_count = 1;
  constraints.max_assigned_location_end_by_reg_class = high_water;
  constraints.max_constrained_location_end_by_reg_class =
      constrained_high_water;

  loom_low_allocation_assignment_t future = {};
  future.value_id = value_ids[3];
  future.descriptor_reg_class_id = value_class.register_class_id;
  future.unit_count = 2;
  future.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  future.location_count = 2;
  future.unit_point_start = 8;
  future.end_point = 14;
  const uint32_t assignment_indices[] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, 0};
  loom_low_allocation_assignment_map_t assignments = {};
  assignments.module = module;
  assignments.liveness = &liveness;
  assignments.assignments = &future;
  assignments.assignment_count = 1;
  assignments.assignment_indices_by_value_ordinal = assignment_indices;
  loom_low_allocation_storage_lease_state_t leases = {};
  loom_low_allocation_search_context_t context = {};
  context.module = module;
  context.descriptor_set = &descriptors;
  context.liveness = &liveness;
  context.unit_liveness = &unit_liveness;
  context.target_constraints = &constraints;
  context.assignment_map = &assignments;
  context.placement = &placement;
  context.storage_leases = &leases;

  const loom_low_placement_value_ref_t values[] = {
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 0},
      {0, LOOM_LOW_PLACEMENT_VALUE_OPERAND, 1}};
  const loom_low_placement_predicate_t predicate = {
      0, 1, 0, 0, 1, LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION, 3};
  const loom_low_placement_clause_t clause = {0, 1, 1,
                                              LOOM_LOW_PLACEMENT_CLAUSE_ANY};
  const loom_low_placement_preference_t preference = {values, &predicate,
                                                      &clause, 2, 1};
  const loom_low_placement_preference_use_t use = {&preference, 0, 1, {2, 3}};
  const loom_low_placement_preference_binding_t bindings[] = {{2, 0}, {3, 1}};
  const uint32_t use_indices[] = {0, 0};
  const uint32_t offsets[] = {0, 0, 0, 1, 2};
  loom_low_placement_preference_index_t preferences = {};
  preferences.uses = &use;
  preferences.bindings = bindings;
  preferences.use_indices = use_indices;
  preferences.offsets_by_origin = offsets;
  preferences.use_count = 1;
  preferences.binding_count = 2;
  preferences.max_incident_use_count = 1;
  preferences.max_incident_binding_count = 2;
  preferences.max_memo_entry_count = 4;
  loom_low_allocation_preference_workspace_t workspace = {};
  IREE_ASSERT_OK(loom_low_allocation_preference_workspace_initialize(
      &preferences, &arena, &workspace));
  context.preference_workspace = &workspace;

  // Both two-unit sources fit at 2 and 4 before the future reservation starts.
  // Their four-unit result must also fit from point 4 through point 10. A
  // reservation starting at 10 is disjoint and must not force preferred
  // four-unit alignment on the otherwise valid base-2 assembly.
  for (uint32_t future_base : {2u, 4u}) {
    for (uint32_t future_start : {5u, 10u}) {
      SCOPED_TRACE(::testing::Message() << "future base=" << future_base
                                        << " start=" << future_start);
      future.location_base = future_base;
      future.start_point = future_start;
      intervals[3].start_point = future_start;
      intervals[3].end_point = 14;
      unit_values[3].acquisition_start_point = future_start;
      unit_start_points[8] = unit_start_points[9] = future_start;
      loom_low_allocation_active_set_t active_set = {};
      IREE_CHECK_OK(loom_low_allocation_active_set_initialize(1, 16, 16, &arena,
                                                              &active_set));
      loom_low_allocation_active_set_insert(&active_set, &descriptors, &future,
                                            1, 0);
      context.active_set = &active_set;
      for (bool prefer_aggregate : {false, true}) {
        context.preferences = prefer_aggregate ? &preferences : nullptr;
        loom_low_allocation_assignment_t reservation;
        IREE_EXPECT_OK(loom_low_allocation_concat_reservation_find(
            &context, &intervals[0], &relations[0], &intervals[2],
            &result_ranges[2], value_ids, 2, &reservation));
        if (future_start == 5) {
          EXPECT_EQ(reservation.value_id, value_ids[2]);
          EXPECT_EQ(reservation.location_count, 4u);
          EXPECT_GE(reservation.location_base, prefix.location_count);
          EXPECT_TRUE(reservation.location_base + reservation.location_count <=
                          future_base ||
                      reservation.location_base >=
                          future_base + future.location_count);
        } else if (prefer_aggregate && future_base == 2) {
          // The ordinary base-2 assembly is legal, but its aggregate violates
          // the preference. Base4 wins without changing either live range.
          EXPECT_EQ(reservation.value_id, value_ids[2]);
          EXPECT_EQ(reservation.location_base, 4u);
        } else {
          // Equal preference cost preserves the ordinary packing choice.
          EXPECT_EQ(reservation.value_id, LOOM_VALUE_ID_INVALID);
        }
        EXPECT_EQ(assignments.assignment_count, 1u);
      }
    }
  }

  // The zero-cost preferred base 4 crosses the register cliff. Base 3 has
  // the same score while preserving the attainable tier. A launch ceiling
  // at or below the lower tier makes the full domain uniform instead.
  future.location_base = 2;
  high_water[0] = 4;
  loom_low_allocation_active_set_t active_set = {};
  IREE_ASSERT_OK(loom_low_allocation_active_set_initialize(1, 16, 16, &arena,
                                                           &active_set));
  loom_low_allocation_active_set_insert(&active_set, &descriptors, &future, 1,
                                        0);
  context.active_set = &active_set;
  context.preferences = &preferences;
  const iree_string_view_t resource_names[] = {IREE_SVL("register")};
  const loom_target_residency_cliff_t cliffs[] = {{0, 8, 8, 4}};
  const loom_target_residency_cliff_range_t cliff_ranges[] = {{0, 1}};
  loom_target_residency_model_t residency_model = {};
  residency_model.best_tier = 8;
  residency_model.direct_resources = {resource_names, cliffs, 1, cliff_ranges,
                                      1};
  for (uint32_t tier_limit : {8u, 4u, 0u}) {
    context.residency =
        loom_target_residency_view(&residency_model, tier_limit);
    loom_low_allocation_assignment_t reservation;
    IREE_ASSERT_OK(loom_low_allocation_concat_reservation_find(
        &context, &intervals[0], &relations[0], &intervals[2],
        &result_ranges[2], value_ids, 2, &reservation));
    EXPECT_EQ(reservation.value_id, value_ids[2]);
    EXPECT_EQ(reservation.location_base, tier_limit == 8 ? 3u : 4u);
  }

  // A shared register class does not imply the same addressable window. The
  // source can only occupy base 2, while the longer-lived aggregate conflicts
  // there. Its wider result domain cannot justify moving that source above 3.
  operands[0].addressable_unit_count = 4;
  context.residency = {};
  future.location_base = 4;
  future.start_point = intervals[3].start_point = 5;
  unit_start_points[8] = unit_start_points[9] = 5;
  loom_low_allocation_active_set_t window_active_set = {};
  IREE_ASSERT_OK(loom_low_allocation_active_set_initialize(1, 16, 16, &arena,
                                                           &window_active_set));
  loom_low_allocation_active_set_insert(&window_active_set, &descriptors,
                                        &future, 1, 0);
  context.active_set = &window_active_set;
  for (bool prefer_aggregate : {false, true}) {
    SCOPED_TRACE(prefer_aggregate);
    context.preferences = prefer_aggregate ? &preferences : nullptr;
    loom_low_allocation_assignment_t reservation;
    IREE_ASSERT_OK(loom_low_allocation_concat_reservation_find(
        &context, &intervals[0], &relations[0], &intervals[2],
        &result_ranges[2], value_ids, 2, &reservation));
    EXPECT_EQ(reservation.value_id, LOOM_VALUE_ID_INVALID);
  }

  for (loom_value_id_t value_id : value_ids) {
    loom_module_value_ordinal_scratch_clear(module, value_id);
  }
  loom_module_value_ordinal_scratch_release(module);
  loom_module_free(module);
  iree_arena_deinitialize(&arena);
  loom_context_deinitialize(&ir_context);
  iree_arena_block_pool_deinitialize(&pool);
}

}  // namespace
}  // namespace loom
