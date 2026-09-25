// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/move_plan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class LowAllocationMovePlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &scratch_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_);
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  ModulePtr Parse(const char* source) {
    loom_text_parse_options_t options = {};
    options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("move_plan_test.loom"), &context_,
                                  &pool_, &options, &module));
    return ModulePtr(module);
  }

  void CheckCursor(const loom_low_allocation_move_plan_t& plan,
                   const loom_region_t* region,
                   loom_low_allocation_move_cursor_t* cursor,
                   uint32_t* visited_count) {
    const auto& liveness = *plan.context.assignment_map.liveness;
    const loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      const loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        const auto* point =
            loom_low_allocation_move_plan_next_operation(&plan, op, cursor);
        // Independent identity lookup is deliberately confined to the test.
        const loom_liveness_operation_point_t* expected = nullptr;
        for (iree_host_size_t i = 0; i < liveness.operation_count; ++i) {
          if (liveness.operation_points[i].op == op) {
            expected = &liveness.operation_points[i];
            break;
          }
        }
        EXPECT_EQ(point, expected);
        ++*visited_count;
        if (!loom_liveness_analysis_includes_region_tree(&liveness)) {
          continue;
        }
        loom_region_t* const* regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          if (regions[i]) {
            CheckCursor(plan, regions[i], cursor, visited_count);
          }
        }
      }
    }
  }

  iree_arena_block_pool_t pool_ = {};
  iree_arena_allocator_t arena_ = {};
  iree_arena_allocator_t scratch_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(LowAllocationMovePlanTest,
       RetainsSourceIdentityAcrossScheduledSubtrees) {
  ModulePtr module = Parse(R"(
low.func.def target<test.low.core> @subtrees(%condition: reg<test.i32>, %lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  %sum = test.add.i32 %lhs, %rhs
  %result = low.scf.if %condition -> (reg<test.i32>) {
    %copy = copy %sum : reg<test.i32> -> reg<test.i32>
    low.scf.yield %copy : reg<test.i32>
  } else {
    %nested = low.scf.if %condition -> (reg<test.i32>) {
      %other = copy %rhs : reg<test.i32> -> reg<test.i32>
      low.scf.yield %other : reg<test.i32>
    } else {
      low.scf.yield %lhs : reg<test.i32>
    }
    low.scf.yield %nested : reg<test.i32>
  }
  low.br ^exit(%result: reg<test.i32>)
^exit(%returned: reg<test.i32>):
  return %returned
}
)");
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &frame_accepted));
  ASSERT_TRUE(frame_accepted);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  ASSERT_GT(frame.allocation.liveness.operation_count,
            frame.schedule.node_count);
  const loom_low_schedule_table_t* const schedules[] = {&frame.schedule,
                                                        nullptr};
  for (const auto* schedule : schedules) {
    loom_low_allocation_move_plan_context_t context = {};
    context.assignment_map.liveness = &frame.allocation.liveness;
    context.schedule = schedule;
    loom_low_allocation_move_plan_t plan = {};
    IREE_ASSERT_OK(loom_low_allocation_move_plan_initialize(
        &context, /*move_input_capacity=*/2, /*raw_group_capacity=*/1, &arena_,
        &scratch_, &plan));
    EXPECT_EQ(plan.operation_indices_by_source_node == nullptr,
              schedule == nullptr);
    const auto scratch_size = scratch_.used_allocation_size;
    for (int traversal = 0; traversal < 2; ++traversal) {
      loom_low_allocation_move_cursor_t cursor = {};
      uint32_t visited_count = 0;
      CheckCursor(plan, frame.allocation.liveness.region, &cursor,
                  &visited_count);
      EXPECT_EQ(visited_count, frame.allocation.liveness.operation_count);
    }
    EXPECT_EQ(scratch_.used_allocation_size, scratch_size);
  }
}

TEST_F(LowAllocationMovePlanTest, SourceCursorUsesAcceptedPermutation) {
  ModulePtr module = Parse(R"(
low.func.def target<test.low.core> @reordered(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  %producer = test.event.slow.i32 %lhs, %rhs
  %consumer = test.add.i32 %producer, %rhs
  %constant = test.const.issued.i32 1
  %result = test.add.i32 %consumer, %constant
  return %result
}
)");
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &frame_accepted));
  ASSERT_TRUE(frame_accepted);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  ASSERT_LT(frame.schedule.nodes[2].scheduled_ordinal,
            frame.schedule.nodes[1].scheduled_ordinal);
  loom_low_allocation_move_plan_context_t context = {};
  context.assignment_map.liveness = &frame.allocation.liveness;
  context.schedule = &frame.schedule;
  loom_low_allocation_move_plan_t plan = {};
  IREE_ASSERT_OK(loom_low_allocation_move_plan_initialize(
      &context, /*move_input_capacity=*/2, /*raw_group_capacity=*/1, &arena_,
      &scratch_, &plan));
  loom_low_allocation_move_cursor_t cursor = {};
  uint32_t visited_count = 0;
  CheckCursor(plan, frame.allocation.liveness.region, &cursor, &visited_count);
  EXPECT_EQ(visited_count, frame.allocation.liveness.operation_count);
}

TEST_F(LowAllocationMovePlanTest, CycleRowsSurviveScratchRelease) {
  ModulePtr module = Parse(R"(
low.func.def target<test.low.core> @swap(%lhs: reg<test.phys>, %rhs: reg<test.phys>) -> (reg<test.phys x2>) asm {
  %pair = concat(%lhs, %rhs) : (reg<test.phys>, reg<test.phys>) -> reg<test.phys x2>
  low.br ^exit(%pair: reg<test.phys x2>)
^exit(%returned: reg<test.phys x2>):
  return %returned
}
)");
  loom_op_t* function = loom_block_op(loom_module_block(module.get()), 0);
  const loom_region_t* body = loom_low_func_def_body(function);
  const loom_block_t* entry = loom_region_const_block(body, 0);
  const loom_block_t* exit = loom_region_const_block(body, 1);
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(entry, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, 1, 1},
      {loom_block_arg_id(entry, 1),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, 0, 1},
      {loom_block_arg_id(exit, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, 0, 2},
  };
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
  options.allocation_fixed_values = fixed_values;
  options.allocation_fixed_value_count = IREE_ARRAYSIZE(fixed_values);
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), function, &options, &arena_, &frame, &frame_accepted));
  ASSERT_TRUE(frame_accepted);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.allocation.placement.max_move_group_unit_count, 2u);
  EXPECT_EQ(frame.allocation.placement.packet_move_unit_count, 2u);
  EXPECT_EQ(frame.allocation.placement.branch_unit_count, 2u);
  ASSERT_EQ(frame.allocation.edge_copy_group_count, 1u);
  iree_arena_block_pool_trim(&pool_);
  const auto& group = frame.allocation.edge_copy_groups[0].move_group;
  ASSERT_EQ(group.moves.count, 3u);
  ASSERT_EQ(group.scratch_move_index_count, 1u);
  const auto* moves = frame.allocation.moves + group.moves.start;
  const auto scratch_index =
      frame.allocation.scratch_move_indices[group.scratch_move_index_start];
  EXPECT_EQ(scratch_index, group.moves.start);
  EXPECT_GE(moves[0].destination.location, 2u);
  uint32_t locations[32] = {};
  locations[0] = 23;
  locations[1] = 17;
  for (iree_host_size_t i = 0; i < group.moves.count; ++i) {
    ASSERT_LT(moves[i].destination.location, IREE_ARRAYSIZE(locations));
    ASSERT_LT(moves[i].source.location, IREE_ARRAYSIZE(locations));
    locations[moves[i].destination.location] =
        locations[moves[i].source.location];
  }
  EXPECT_EQ(locations[0], 17u);
  EXPECT_EQ(locations[1], 23u);
}

}  // namespace
}  // namespace loom
