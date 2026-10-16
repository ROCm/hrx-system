// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/allocation/search.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

class LowAllocationCoveringSpillSearchTest : public ::testing::Test {
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
  return loom_liveness_value_class_t{
      .type_kind = LOOM_TYPE_REGISTER,
      .register_class_id = 0,
      .register_descriptor_set_stable_id = descriptor_set_id,
  };
}

loom_liveness_interval_t Interval(loom_value_id_t value_id, uint32_t start,
                                  uint32_t end,
                                  loom_liveness_value_class_t value_class,
                                  uint32_t unit_count) {
  return loom_liveness_interval_t{
      .value_id = value_id,
      .start_point = start,
      .end_point = end,
      .value_class = value_class,
      .unit_count = unit_count,
  };
}

loom_low_allocation_assignment_t Assignment(
    loom_value_id_t value_id, uint32_t start, uint32_t end,
    loom_liveness_value_class_t value_class, uint32_t location_base,
    uint32_t location_count, uint32_t unit_point_start) {
  return loom_low_allocation_assignment_t{
      .value_id = value_id,
      .descriptor_reg_class_id = value_class.register_class_id,
      .start_point = start,
      .end_point = end,
      .unit_count = location_count,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = location_base,
      .location_count = location_count,
      .unit_point_start = unit_point_start,
  };
}

loom_low_allocation_class_capacity_t Capacity(uint32_t max_units) {
  return loom_low_allocation_class_capacity_t{
      .descriptor_reg_class_id = 0,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .max_units = max_units,
      .alloc_unit_bits = 32,
      .spill_slot_space = LOOM_LOW_SPILL_SLOT_SPACE_UNKNOWN,
      .is_spillable = true,
      .is_bounded = true,
  };
}

loom_low_descriptor_set_t DescriptorSet(const loom_low_reg_class_t* reg_class,
                                        uint64_t stable_id) {
  return loom_low_descriptor_set_t{
      .stable_id = stable_id,
      .reg_classes = reg_class,
      .reg_class_count = 1,
  };
}

loom_low_reg_class_t RegClass(uint32_t allocatable_count,
                              loom_low_reg_class_flags_t flags) {
  return loom_low_reg_class_t{
      .flags = flags,
      .alloc_unit_bits = 32,
      .allocatable_count = (uint16_t)allocatable_count,
      .spill_class_id = LOOM_LOW_REG_CLASS_NONE,
  };
}

loom_low_resolved_target_t ResolvedTarget(
    const loom_low_descriptor_set_t* descriptor_set) {
  loom_low_resolved_target_t target = {
      .descriptor_set = descriptor_set,
  };
  target.descriptor_set_key = IREE_SV("test");
  return target;
}

loom_liveness_block_info_t Block(uint32_t start_point, uint32_t end_point) {
  return loom_liveness_block_info_t{
      .start_point = start_point,
      .end_point = end_point,
  };
}

TEST_F(LowAllocationCoveringSpillSearchTest,
       SelectsCoveringSpillForForwardedRequiredSource) {
  loom_module_t* module = AllocateModule();
  const loom_value_id_t candidate_value = DefineValue(module);
  const loom_value_id_t required_source_value = DefineValue(module);
  const loom_value_id_t covering_result_value = DefineValue(module);
  const loom_value_id_t other_result_value = DefineValue(module);
  loom_local_value_domain_t value_domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
      module, module->body, &arena_, &value_domain));
  for (const loom_value_id_t value_id :
       {candidate_value, required_source_value, covering_result_value,
        other_result_value}) {
    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    IREE_ASSERT_OK(loom_local_value_domain_register_value(
        &value_domain, value_id, &value_ordinal));
  }

  const uint64_t descriptor_set_id = 13;
  const loom_liveness_value_class_t value_class =
      RegisterValueClass(descriptor_set_id);
  loom_liveness_interval_t intervals[] = {
      Interval(candidate_value, /*start=*/8, /*end=*/12, value_class,
               /*unit_count=*/4),
      Interval(required_source_value, /*start=*/0, /*end=*/2, value_class,
               /*unit_count=*/2),
      Interval(covering_result_value, /*start=*/2, /*end=*/20, value_class,
               /*unit_count=*/2),
      Interval(other_result_value, /*start=*/2, /*end=*/20, value_class,
               /*unit_count=*/2),
  };
  intervals[0].definition_point = 8;
  intervals[1].definition_point = 0;
  intervals[2].definition_point = 2;
  intervals[3].definition_point = 2;
  const uint32_t interval_indices[] = {0, 1, 2, 3};
  const loom_value_id_t value_ids[] = {
      candidate_value,
      required_source_value,
      covering_result_value,
      other_result_value,
  };
  const loom_liveness_block_info_t blocks[] = {
      Block(/*start_point=*/0, /*end_point=*/24),
  };
  loom_liveness_analysis_t liveness = {
      .blocks = blocks,
      .block_count = IREE_ARRAYSIZE(blocks),
      .intervals = intervals,
      .interval_count = IREE_ARRAYSIZE(intervals),
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .value_interval_indices = interval_indices,
  };

  loom_low_allocation_unit_liveness_value_t unit_values[] = {
      {0, 8},
      {4, 0},
      {6, 2},
      {8, 2},
  };
  uint32_t unit_end_points[] = {12, 12, 12, 12, 20, 20, 20, 20, 20, 20};
  uint64_t incomplete_storage_segment_words[] = {0};
  uint64_t independent_storage_extension_words[] = {0};
  uint32_t observation_value_links[] = {
      0,
      IREE_ARRAYSIZE(value_ids) + 1,
      0,
      0,
  };
  loom_low_allocation_decomposed_use_t decomposed_observations[] = {
      {.operation_index = 2},
  };
  loom_low_allocation_unit_liveness_t unit_liveness = {
      .values = unit_values,
      .end_points = unit_end_points,
      .point_count = IREE_ARRAYSIZE(unit_end_points),
      .values_with_incomplete_storage_segments =
          {
              liveness.value_count,
              incomplete_storage_segment_words,
          },
      .values_with_independent_storage_extensions =
          {
              liveness.value_count,
              independent_storage_extension_words,
          },
      .observations =
          {
              .value_links = observation_value_links,
              .entries = decomposed_observations,
              .count = IREE_ARRAYSIZE(decomposed_observations),
              .capacity = IREE_ARRAYSIZE(decomposed_observations),
          },
  };

  const loom_low_reg_class_t reg_class =
      RegClass(/*allocatable_count=*/6, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
  const loom_low_descriptor_set_t descriptor_set =
      DescriptorSet(&reg_class, descriptor_set_id);
  const loom_low_resolved_target_t target = ResolvedTarget(&descriptor_set);
  uint32_t max_assigned_location_end_by_reg_class[] = {6};
  loom_low_allocation_target_constraints_t target_constraints = {
      .target = &target,
      .max_assigned_location_end_by_reg_class =
          max_assigned_location_end_by_reg_class,
  };

  loom_low_allocation_assignment_t assignments[] = {
      Assignment(required_source_value, /*start=*/0, /*end=*/20, value_class,
                 /*location_base=*/0, /*location_count=*/2,
                 /*unit_point_start=*/4),
      Assignment(covering_result_value, /*start=*/2, /*end=*/20, value_class,
                 /*location_base=*/2, /*location_count=*/2,
                 /*unit_point_start=*/6),
      Assignment(other_result_value, /*start=*/2, /*end=*/20, value_class,
                 /*location_base=*/4, /*location_count=*/2,
                 /*unit_point_start=*/8),
  };
  uint32_t assignment_indices_by_value_ordinal[] = {UINT32_MAX, 0, 1, 2};
  loom_low_allocation_assignment_map_t assignment_map = {
      .module = module,
      .liveness = &liveness,
      .assignments = assignments,
      .assignment_count = IREE_ARRAYSIZE(assignments),
      .assignment_indices_by_value_ordinal =
          assignment_indices_by_value_ordinal,
  };

  loom_low_allocation_active_set_t active_set = {};
  IREE_ASSERT_OK(loom_low_allocation_active_set_initialize(
      &descriptor_set, /*assignment_capacity=*/3,
      /*program_point_count=*/25, /*unit_capacity=*/6, &arena_, &active_set));
  loom_low_allocation_active_set_insert(
      &active_set, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/0);
  loom_low_allocation_active_set_insert(
      &active_set, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/1);
  loom_low_allocation_active_set_insert(
      &active_set, &descriptor_set, assignments, IREE_ARRAYSIZE(assignments),
      /*assignment_index=*/2);

  loom_low_placement_relation_t relations[] = {
      {
          .result_ordinal = 2,
          .source_ordinal = 1,
          .unit_count = 2,
          .kind = LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART,
          .cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT,
          .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED |
                   LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
      },
      {
          .result_ordinal = 3,
          .source_ordinal = 1,
          .unit_count = 2,
          .kind = LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART,
          .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED |
                   LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
      },
  };
  const uint32_t relation_indices_by_source_ordinal[] = {0, 1};
  const loom_low_placement_relation_range_t ranges_by_source_ordinal[] = {
      {0, 0},
      {0, 2},
      {2, 0},
      {2, 0},
  };
  loom_low_placement_table_t placement = {
      .module = module,
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .relations = relations,
      .relation_count = IREE_ARRAYSIZE(relations),
      .relation_indices_by_source_ordinal = relation_indices_by_source_ordinal,
      .ranges_by_source_ordinal = ranges_by_source_ordinal,
  };
  loom_low_allocation_storage_lease_state_t empty_storage_leases = {};
  loom_low_allocation_spill_plan_traffic_t spill_traffic[] = {
      {.store_count = 1, .store_bytes = 16},
      {.store_count = 1, .store_bytes = 8},
      {.store_count = 1, .store_bytes = 8},
      {.store_count = 1, .store_bytes = 8},
  };
  loom_low_allocation_search_context_t context = {
      .module = module,
      .descriptor_set = &descriptor_set,
      .liveness = &liveness,
      .unit_liveness = &unit_liveness,
      .target_constraints = &target_constraints,
      .assignment_map = &assignment_map,
      .placement = &placement,
      .active_set = &active_set,
      .storage_leases = &empty_storage_leases,
      .spill_traffic_by_value_ordinal = spill_traffic,
  };
  assignments[0].flags |= LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REQUIRES_REGISTER;
  iree_bitmap_set(unit_liveness.values_with_incomplete_storage_segments,
                  /*bit_index=*/1);

  const loom_low_allocation_class_capacity_t capacity = Capacity(4);
  loom_low_allocation_search_workspace_t workspace = {};
  loom_low_allocation_search_spill_victim_set_t victim_set = {};
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  ASSERT_TRUE(victim_set.found);
  EXPECT_EQ(victim_set.location_base, 0u);
  ASSERT_EQ(victim_set.assignment_count, 1u);
  EXPECT_EQ(victim_set.assignment_indices[0], 1u);
  ASSERT_EQ(victim_set.released_value_count, 1u);
  EXPECT_EQ(victim_set.released_value_ids[0], required_source_value);

  // The source and covering victim both have outstanding asynchronous leases.
  // Spilling the victim retires its lease, while the source lease must be
  // independently releasable before the incoming assignment.
  constexpr uint32_t kScheduleNodeCount = 9;
  loom_low_schedule_node_t schedule_nodes[kScheduleNodeCount] = {};
  uint32_t scheduled_node_indices[kScheduleNodeCount] = {};
  for (uint32_t i = 0; i < kScheduleNodeCount; ++i) {
    schedule_nodes[i].block_index = 0;
    schedule_nodes[i].scheduled_ordinal = i;
    scheduled_node_indices[i] = i;
  }
  schedule_nodes[0].operand_count = 1;
  schedule_nodes[0].value_ordinals.inline_value_ordinals[0] = 1;
  schedule_nodes[1].operand_count = 1;
  schedule_nodes[1].value_ordinals.inline_value_ordinals[0] = 2;
  loom_low_schedule_block_t schedule_blocks[] = {{
      .node_count = kScheduleNodeCount,
      .scheduled_node_count = kScheduleNodeCount,
  }};
  uint32_t value_producer_nodes[] = {
      LOOM_LOW_SCHEDULE_NODE_NONE,
      LOOM_LOW_SCHEDULE_NODE_NONE,
      LOOM_LOW_SCHEDULE_NODE_NONE,
      LOOM_LOW_SCHEDULE_NODE_NONE,
  };
  const loom_op_t* function_op =
      reinterpret_cast<const loom_op_t*>(static_cast<uintptr_t>(1));
  loom_low_schedule_table_t schedule = {
      .module = module,
      .function_op = function_op,
      .target = target,
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
      .value_producer_nodes = value_producer_nodes,
      .blocks = schedule_blocks,
      .block_count = IREE_ARRAYSIZE(schedule_blocks),
      .nodes = schedule_nodes,
      .node_count = IREE_ARRAYSIZE(schedule_nodes),
      .scheduled_node_indices = scheduled_node_indices,
      .scheduled_node_count = IREE_ARRAYSIZE(scheduled_node_indices),
  };

  loom_low_storage_lease_record_t lease_records[2] = {};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(lease_records); ++i) {
    lease_records[i].packet_index = i;
    lease_records[i].node_index = i;
    lease_records[i].block_index = 0;
    lease_records[i].scheduled_ordinal = i;
    lease_records[i].kind = LOOM_LOW_STORAGE_LEASE_SOURCE_READ;
    lease_records[i].attachment = LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND;
    lease_records[i].unit_count = 2;
    lease_records[i].release_scope =
        LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS;
    lease_records[i].release_class_id = 1;
    lease_records[i].release_action_id = 1;
    lease_records[i].release_before_scheduled_ordinal_plus_one = 20;
    lease_records[i].flags = LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE;
  }
  lease_records[0].flags |= LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE;
  const loom_low_storage_lease_table_t lease_table = {
      .schedule = &schedule,
      .records = lease_records,
      .record_count = IREE_ARRAYSIZE(lease_records),
  };
  const loom_low_allocation_storage_identity_t storage_identity = {};
  loom_low_allocation_storage_lease_state_t storage_leases = {};
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_initialize(
      &lease_table, module, function_op, &value_domain, &liveness,
      &storage_identity, &unit_liveness, &arena_, &storage_leases));
  loom_low_allocation_storage_lease_state_record_assignment(
      &storage_leases, &descriptor_set, &liveness, assignments,
      /*assignment_index=*/0, /*value_ordinal=*/1);
  loom_low_allocation_storage_lease_state_record_assignment(
      &storage_leases, &descriptor_set, &liveness, assignments,
      /*assignment_index=*/1, /*value_ordinal=*/2);
  context.storage_leases = &storage_leases;

  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_TRUE(victim_set.found);

  lease_records[0].flags &= ~LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE;
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  lease_records[0].flags |= LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE;

  schedule_blocks[0].scheduled_node_count = 8;
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  schedule_blocks[0].scheduled_node_count = kScheduleNodeCount;

  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  ASSERT_TRUE(victim_set.found);
  loom_low_allocation_assignment_t spilled_covering = assignments[1];
  spilled_covering.location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT;
  spilled_covering.location_base = 0;
  loom_low_allocation_storage_lease_state_record_spill(
      &storage_leases, &descriptor_set, &spilled_covering,
      /*value_ordinal=*/2);
  const loom_low_allocation_assignment_t incoming =
      Assignment(candidate_value, /*start=*/8, /*end=*/12, value_class,
                 /*location_base=*/0, /*location_count=*/4,
                 /*unit_point_start=*/0);
  IREE_ASSERT_OK(loom_low_allocation_storage_lease_state_record_release_actions(
      &storage_leases, &descriptor_set, &liveness, &incoming,
      /*ignored_value_ids=*/nullptr, /*ignored_value_count=*/0));
  ASSERT_EQ(storage_leases.release_action_count, 1u);
  EXPECT_EQ(storage_leases.instances[0].end_point, incoming.start_point);
  EXPECT_EQ(storage_leases.instances[1].end_point, 0u);
  IREE_ASSERT_OK(
      loom_low_allocation_storage_lease_state_finalize(&storage_leases));
  context.storage_leases = &empty_storage_leases;

  // Ordinary semantic liveness needs no release. Only a producer-retained
  // physical observation can be shortened by materializing the result.
  iree_bitmap_reset(unit_liveness.values_with_incomplete_storage_segments,
                    /*bit_index=*/1);
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  iree_bitmap_set(unit_liveness.values_with_incomplete_storage_segments,
                  /*bit_index=*/1);

  // A concrete decomposed observation, rather than a generic incomplete
  // storage fact alone, establishes what materializing the result discharges.
  observation_value_links[1] = 0;
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  observation_value_links[1] = IREE_ARRAYSIZE(value_ids) + 1;

  // Boundary demand, native writes, tied storage, and ordinary physical uses
  // survive materialization and independently forbid shortening the source.
  iree_bitmap_set(unit_liveness.values_with_independent_storage_extensions,
                  /*bit_index=*/1);
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  iree_bitmap_reset(unit_liveness.values_with_independent_storage_extensions,
                    /*bit_index=*/1);

  // Partial, captured, and still-semantically-live sources do not transfer
  // their complete remaining storage obligation to the result.
  relations[0].unit_count = 1;
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  relations[0].unit_count = 2;
  relations[0].flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_CAPTURED_PART;
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  relations[0].flags &= ~LOOM_LOW_PLACEMENT_RELATION_FLAG_CAPTURED_PART;
  intervals[1].end_point = 3;
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  intervals[1].end_point = 2;

  // A second still-live forwarded result keeps the source observable after
  // spilling either result alone.
  relations[1].cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  relations[1].cause = LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN;

  // A mandatory tied-storage component can remain observable independently of
  // the forwarding result and must not be shortened by materializing it.
  const loom_value_ordinal_t tied_origins[] = {0, 1, 2, 1};
  placement.tied_storage_origins_by_value_ordinal = tied_origins;
  relations[1].cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  relations[1].cause = LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN;
  placement.tied_storage_origins_by_value_ordinal = nullptr;

  assignments[1].flags |= LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REQUIRES_REGISTER;
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);
  assignments[1].flags &=
      ~LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REQUIRES_REGISTER;

  // A covering result already retired from the active set cannot be selected
  // as the victim that releases an active source.
  loom_low_allocation_active_set_remove(&active_set, assignments,
                                        IREE_ARRAYSIZE(assignments),
                                        /*assignment_index=*/1);
  IREE_ASSERT_OK(loom_low_allocation_search_find_active_spill_victim_set(
      &context, &intervals[0], &capacity,
      /*interval_requires_register=*/true, &workspace, &arena_, &victim_set));
  EXPECT_FALSE(victim_set.found);

  loom_local_value_domain_release(&value_domain);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
