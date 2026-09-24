// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/block.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/schedule/target_pressure.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

TEST(ScheduleBlockTest, RetainsOwnedRowsWithShiftedNodeIndices) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  const auto* descriptors = loom_test_low_core_descriptor_set();
  loom_low_schedule_options_t options = {};
  options.strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;

  // Node indices shift because an earlier block gained an operation. The
  // unchanged block keeps its selected order, including two simultaneous
  // instructions. Operation pointers are identities owned outside both tables.
  loom_op_t operations[4] = {};
  loom_block_t block = {};
  block.region_index = 1;
  loom_low_schedule_block_t previous_blocks[2] = {};
  auto& previous_block = previous_blocks[1];
  previous_block.block = &block;
  previous_block.node_start = 1;
  previous_block.node_count = 4;
  previous_block.scheduled_node_start = 1;
  previous_block.scheduled_node_count = 4;
  previous_block.candidate_decisions.start = 1;
  previous_block.candidate_decisions.count = 1;
  loom_low_schedule_node_t previous_nodes[5] = {};
  const uint32_t descriptor_ordinals[] = {
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_LOAD_V4I32,
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_FAST_I32,
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_STORE_V4I32,
  };
  for (uint32_t i = 0; i < 4; ++i) {
    auto& node = previous_nodes[1 + i];
    node.op = &operations[i];
    node.block_index = 1;
    if (i < IREE_ARRAYSIZE(descriptor_ordinals)) {
      const uint32_t ordinal = descriptor_ordinals[i];
      node.descriptor = &descriptors->descriptors[ordinal];
      node.schedule_class_id =
          loom_low_descriptor_set_descriptor_view_at(descriptors, ordinal)
              ->schedule_class_id;
      node.schedule_class =
          &descriptors->schedule_classes[node.schedule_class_id];
    }
  }
  previous_nodes[3].issue_cycle = 8;
  previous_nodes[4].issue_cycle = 9;
  const uint32_t previous_order[] = {0, 2, 1, 3, 4};
  loom_low_schedule_pressure_step_t previous_steps[5] = {};
  for (uint32_t i = 1; i < IREE_ARRAYSIZE(previous_order); ++i) {
    previous_steps[i].node_index = previous_order[i];
    previous_steps[i].block_index = 1;
    previous_steps[i].scheduled_ordinal = i - 1;
    previous_steps[i].issue_cycle =
        previous_nodes[previous_order[i]].issue_cycle;
  }
  loom_low_schedule_candidate_decision_t previous_decisions[2] = {};
  previous_decisions[1].block_index = 1;
  previous_decisions[1].chosen_node = 2;
  previous_decisions[1].rejected_node = 1;
  previous_decisions[1].ready_candidate_count = 2;
  previous_decisions[1].scored_candidate_count = 2;
  loom_low_schedule_table_t previous = {};
  previous.blocks = previous_blocks;
  previous.nodes = previous_nodes;
  previous.scheduled_node_indices = previous_order;
  previous.pressure_steps = previous_steps;
  previous.candidate_decisions = previous_decisions;

  loom_low_schedule_block_t blocks[2] = {};
  blocks[1].block = &block;
  blocks[1].node_start = 2;
  blocks[1].node_count = 4;
  loom_low_schedule_node_t nodes[6] = {};
  for (uint32_t i = 0; i < 4; ++i) {
    nodes[2 + i].op = &operations[i];
    nodes[2 + i].block_index = 1;
  }
  uint32_t order[6] = {};
  const loom_op_t* ordered_ops[6] = {};
  loom_low_schedule_issue_group_t groups[6] = {};
  loom_liveness_block_order_t liveness_orders[2] = {};
  loom_low_schedule_pressure_step_t steps[6] = {};
  loom_low_schedule_candidate_decision_t decisions[6] = {};
  loom_low_schedule_effect_use_t effects[8] = {};
  loom_low_schedule_hazard_use_t hazards[8] = {};
  loom_low_schedule_hazard_state_t hazard_states[8] = {};
  std::vector<loom_low_schedule_model_summary_t> models(
      descriptors->schedule_class_count);
  std::vector<loom_low_schedule_resource_summary_t> resources(
      descriptors->resource_count);
  for (uint16_t i = 0; i < descriptors->resource_count; ++i) {
    resources[i].capacity_per_cycle =
        descriptors->resources[i].capacity_per_cycle;
  }
  loom_low_schedule_build_state_t state = {};
  state.arena = &arena;
  state.options = &options;
  state.target.descriptor_set = descriptors;
  state.blocks = blocks;
  state.nodes = nodes;
  state.scheduled_node_indices = order;
  state.scheduled_ops = ordered_ops;
  state.scheduled_node_count = 2;
  state.issue_groups = groups;
  state.liveness_block_orders = liveness_orders;
  state.pressure_steps = steps;
  state.pressure_step_count = 2;
  state.candidate_decisions = decisions;
  state.candidate_decision_count = 1;
  state.effect_uses = effects;
  state.effect_use_capacity = IREE_ARRAYSIZE(effects);
  state.hazard_uses = hazards;
  state.hazard_use_capacity = IREE_ARRAYSIZE(hazards);
  state.hazard_states = hazard_states;
  state.hazard_state_capacity = IREE_ARRAYSIZE(hazard_states);
  state.model_summaries = models.data();
  state.resource_summaries = resources.data();
  IREE_ASSERT_OK(loom_low_schedule_resource_calendar_initialize(
      descriptors, &arena, &state.resource_calendar));
  loom_low_schedule_pressure_state_t pressure = {};
  loom_low_schedule_block_begin(&state, 1);
  IREE_ASSERT_OK(loom_low_schedule_block_retain(&state, &pressure, &previous));

  EXPECT_EQ(order[2], 3u);
  EXPECT_EQ(order[3], 2u);
  EXPECT_EQ(order[4], 4u);
  EXPECT_EQ(order[5], 5u);
  EXPECT_EQ(blocks[1].issue_group_count, 3u);
  EXPECT_EQ(groups[0].scheduled_node_count, 2u);
  EXPECT_EQ(groups[1].issue_cycle, 8u);
  EXPECT_EQ(liveness_orders[1].ops, ordered_ops + 2);
  EXPECT_EQ(liveness_orders[1].op_count, 4u);
  EXPECT_EQ(blocks[1].candidate_decisions.start, 1u);
  EXPECT_EQ(blocks[1].candidate_decisions.count, 1u);
  EXPECT_EQ(decisions[1].chosen_node, 3u);
  EXPECT_EQ(decisions[1].rejected_node, 2u);
  EXPECT_EQ(steps[2].node_index, 3u);
  EXPECT_EQ(steps[3].node_index, 2u);
  EXPECT_EQ(steps[4].issue_cycle, 8u);
  EXPECT_GT(state.effect_use_count, 0u);
  EXPECT_GT(state.hazard_use_count, 0u);
  EXPECT_GT(state.resource_use_count, 0u);
  EXPECT_EQ(effects[0].node_index, 2u);
  EXPECT_EQ(effects[0].scheduled_ordinal, 1u);
  EXPECT_EQ(hazards[0].node_index, 2u);

  // Results retain target-owned descriptors and newly owned rows, never old
  // nodes or diagnostics. Retiring those tables cannot change the result.
  previous_nodes[2] = {};
  previous_steps[1] = {};
  previous_decisions[1] = {};
  EXPECT_EQ(nodes[3].descriptor,
            &descriptors->descriptors[descriptor_ordinals[1]]);
  EXPECT_EQ(steps[2].node_index, 3u);
  EXPECT_EQ(decisions[1].ready_candidate_count, 2u);
  EXPECT_EQ(ordered_ops[2], &operations[1]);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
}

TEST(ScheduleBlockTest, MergesRoundedClassPeaksWithoutDoubleCountingCliffs) {
  // Two classes contribute rounded high-water marks to one resource. Its
  // source baseline already crossed the first cliff, so only the second can
  // charge the schedule. Block peaks need not coincide or arrive in order.
  const loom_target_residency_derived_member_t members[] = {{0, 0, 4},
                                                            {0, 1, 4}};
  const uint16_t member_indices[] = {0, 1};
  const loom_target_residency_derived_member_range_t ranges[] = {{0, 1},
                                                                 {1, 1}};
  const loom_target_residency_cliff_t cliffs[] = {{0, 9, 4, 3}, {0, 13, 3, 2}};
  loom_target_residency_derived_resource_t resource = {};
  resource.member_count = 2;
  resource.cliff_count = 2;
  loom_target_residency_derived_resource_table_t table = {};
  table.resources = &resource;
  table.resource_count = 1;
  table.members = members;
  table.member_count = IREE_ARRAYSIZE(members);
  table.cliffs = cliffs;
  table.cliff_count = IREE_ARRAYSIZE(cliffs);
  table.member_indices_by_direct_resource = member_indices;
  table.member_ranges_by_direct_resource = ranges;
  loom_low_descriptor_set_t descriptors = {};
  descriptors.reg_class_count = 2;
  const uint64_t retained_peaks[] = {5, 2, 2, 5, 1, 1};
  uint64_t result_peaks[6] = {};
  loom_low_schedule_table_t previous = {};
  previous.block_pressure_peaks = retained_peaks;
  loom_low_schedule_build_state_t state = {};
  state.target.descriptor_set = &descriptors;
  state.pressure_resources = &table;
  state.block_pressure_peaks = result_peaks;
  uint64_t high_water[] = {0, 0};
  loom_low_schedule_resource_pressure_record_t record = {};
  record.next_cliff_index = 1;
  loom_low_schedule_pressure_state_t pressure = {};
  pressure.resources.peak_live_units_by_reg_class = high_water;
  pressure.resources.records = &record;

  loom_low_schedule_pressure_retain_block(&state, &pressure, &previous, 0);
  EXPECT_EQ(record.current_peak_units, 12u);
  EXPECT_EQ(pressure.resources.pressure_cliff_penalty, 0u);
  loom_low_schedule_pressure_retain_block(&state, &pressure, &previous, 1);
  EXPECT_EQ(record.current_peak_units, 16u);
  EXPECT_EQ(record.next_cliff_index, 2u);
  EXPECT_EQ(pressure.resources.pressure_cliff_penalty, 1u);
  loom_low_schedule_pressure_retain_block(&state, &pressure, &previous, 2);
  EXPECT_EQ(record.current_peak_units, 16u);
  EXPECT_EQ(pressure.resources.pressure_cliff_penalty, 1u);
  EXPECT_EQ(high_water[0], 5u);
  EXPECT_EQ(high_water[1], 5u);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(result_peaks); ++i) {
    EXPECT_EQ(result_peaks[i], retained_peaks[i]);
  }
}

}  // namespace
}  // namespace loom
