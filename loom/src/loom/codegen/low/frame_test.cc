// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/frame_test_fixture.h"
#include "loom/codegen/low/guarded_motion.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/schedule/diagnostics.h"
#include "loom/codegen/low/schedule/physical_issue.h"
#include "loom/error/error_catalog.h"
#include "loom/target/facts_builder.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/target_records.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;
using LowEmissionFrameTest = ::loom::testing::LowEmissionFrameTest;

TEST_F(LowEmissionFrameTest, ResidencyQueryConsumesRetainedFunctionFacts) {
  ModulePtr module = ParseModule();
  static const loom_target_residency_model_t model = {/*.best_tier=*/4};
  loom_target_facts_t target_facts = {};
  loom_target_facts_builder_initialize(&loom_test_target_fact_type,
                                       loom_test_target_bundles.values[1],
                                       &target_facts);
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.function_target_facts = &target_facts;
  options.residency_query =
      [](const loom_low_resolved_target_t* target,
         const loom_low_storage_layout_space_sizes_t* storage_sizes) {
        EXPECT_NE(target->target_facts, nullptr);
        EXPECT_EQ(target->descriptor_set, loom_test_low_core_descriptor_set());
        EXPECT_EQ(storage_sizes->workgroup_bytes, 64u);
        return loom_target_residency_view(&model, 2);
      };
  loom_low_emission_frame_t frame = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &accepted));
  ASSERT_TRUE(accepted);
  // The function model has been released; the frame retains only the borrowed
  // immutable policy and its value ceiling, not analysis-owned storage.
  iree_arena_block_pool_trim(&block_pool_);
  EXPECT_EQ(frame.residency.model, &model);
  EXPECT_EQ(frame.residency.tier_limit, 2u);
}

TEST_F(LowEmissionFrameTest, ReusedRegisterWaitsForPreviousPhysicalRead) {
  const auto strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @physical_reuse(%seed: reg<test.phys>) -> (reg<test.phys>) asm {
  %old = test.event.write.fast.phys %seed
  test.event.read.late.phys %old
  %next = test.event.write.fast.phys %seed
  return %next
}
)");
  loom_op_t* function = loom_block_op(loom_module_block(module.get()), 0);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {body->arg_ids[0], LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, 1, 1},
      {loom_op_const_results(loom_block_op(body, 0))[0],
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, 0, 1},
      {loom_op_const_results(loom_block_op(body, 2))[0],
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, 0, 1},
  };
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = strategy;
  options.schedule_flags = LOOM_LOW_SCHEDULE_FLAG_RETAIN_DEPENDENCY_INDEX;
  options.allocation_fixed_values = fixed_values;
  options.allocation_fixed_value_count = IREE_ARRAYSIZE(fixed_values);
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), function, &options, &arena_, &frame, &frame_accepted));
  ASSERT_TRUE(frame_accepted);
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  // Physical timing consumes the retained index after scheduler scratch dies.
  iree_arena_block_pool_trim(&block_pool_);
  loom_low_physical_issue_t issue = {};
  IREE_ASSERT_OK(
      loom_low_physical_issue_initialize(&frame.schedule, &arena_, &issue));
  uint32_t cycles[3] = {};
  uint32_t next_cycle = 0;
  for (uint32_t i = 0; i < frame.schedule.scheduled_node_count; ++i) {
    const auto packet = loom_low_packet_at(&frame.schedule, i);
    if (packet.descriptor == nullptr) {
      continue;
    }
    uint16_t registers[2] = {};
    ASSERT_LE(packet.descriptor->operand_count, IREE_ARRAYSIZE(registers));
    for (uint16_t j = 0; j < packet.descriptor->operand_count; ++j) {
      registers[j] = (uint16_t)loom_low_packet_descriptor_operand_assignment(
                         &frame.allocation, &packet, j)
                         ->location_base;
    }
    const loom_low_physical_instruction_t instruction = {
        packet.descriptor_ordinal, registers};
    const uint32_t proposed_cycle =
        iree_max(iree_max(next_cycle, packet.node->issue_cycle),
                 loom_low_physical_issue_source_ready_cycle(&issue, i));
    const uint32_t cycle = loom_low_physical_issue_find_earliest_issue_cycle(
        &issue, &instruction, 1, proposed_cycle);
    IREE_ASSERT_OK(
        loom_low_physical_issue_commit(&issue, &instruction, 1, cycle));
    loom_low_physical_issue_commit_source(&issue, i, cycle);
    cycles[packet.node->source_ordinal] = cycle;
    next_cycle = cycle + 1;
  }
  // SSA lifetimes permit reuse, but the physical read has not finished.
  EXPECT_GE(cycles[2], cycles[1] + 3);
  const auto read_packet = loom_low_packet_at_node(&frame.schedule, 1);
  const auto write_packet = loom_low_packet_at_node(&frame.schedule, 2);
  const uint16_t read_registers[] = {0};
  uint16_t write_registers[] = {0, 1};
  const loom_low_physical_instruction_t pair[] = {
      {read_packet.descriptor_ordinal, read_registers},
      {write_packet.descriptor_ordinal, write_registers},
  };
  EXPECT_FALSE(
      loom_low_physical_issue_group_fits(frame.target.descriptor_set, pair, 2));
  write_registers[0] = 1;
  EXPECT_TRUE(
      loom_low_physical_issue_group_fits(frame.target.descriptor_set, pair, 2));

  // A negative hardware RAW permits early native issue, but semantic source
  // edges retain producer-before-consumer issue order in this admission model.
  loom_low_physical_issue_t semantic_issue = {};
  IREE_ASSERT_OK(loom_low_physical_issue_initialize(&frame.schedule, &arena_,
                                                    &semantic_issue));
  const auto producer_packet = loom_low_packet_at_node(&frame.schedule, 0);
  loom_low_physical_issue_commit_source(
      &semantic_issue, (uint32_t)producer_packet.packet_index, 10);
  EXPECT_EQ(loom_low_physical_issue_source_ready_cycle(
                &semantic_issue, (uint32_t)read_packet.packet_index),
            10u);
}

TEST_F(LowEmissionFrameTest, EveryStrategyEnforcesIssueResourceCapacity) {
  constexpr loom_low_schedule_strategy_t kStrategies[] = {
      LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
      LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING,
      LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
  };
  for (const loom_low_schedule_strategy_t strategy : kStrategies) {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @resource_capacity(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %first = test.resource.serial.i32 %lhs, %rhs
  %second = test.resource.serial.i32 %lhs, %rhs
  return %first, %second
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame, strategy));
    // Emission consumers retain the frame after scheduler scratch is freed.
    iree_arena_block_pool_trim(&block_pool_);
    ASSERT_GE(frame.schedule.node_count, 2u);
    EXPECT_EQ(frame.schedule.nodes[0].issue_cycle, 0u);
    EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 4u);
  }
}

TEST_F(LowEmissionFrameTest, DependencyTimingUsesMostSpecificModel) {
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @schedule_class_fallback(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  %producer = test.event.fast.i32 %lhs, %rhs
  %consumer = test.add.i32 %producer, %rhs
  return %consumer
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ASSERT_EQ(frame.schedule.dependencies.count, 2u);
    ExpectDependencyTiming(frame, 0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 7,
                           LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_SCHEDULE_CLASS,
                           LOOM_LOW_MODEL_QUALITY_EXACT);
    EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 7u);
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @exact_zero(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  %producer = test.event.fast.i32 %lhs, %rhs
  %consumer = test.event.consume.early.i32 %producer, %rhs
  return %consumer
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ASSERT_EQ(frame.schedule.dependencies.count, 2u);
    ExpectDependencyTiming(frame, 0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 0,
                           LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
                           LOOM_LOW_MODEL_QUALITY_EXACT);
    EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 0u);
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @calibrated_positive(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  %producer = test.event.slow.i32 %lhs, %rhs
  %consumer = test.event.consume.early.i32 %producer, %rhs
  return %consumer
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ASSERT_EQ(frame.schedule.dependencies.count, 2u);
    ExpectDependencyTiming(frame, 0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, 3,
                           LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
                           LOOM_LOW_MODEL_QUALITY_CALIBRATED);
    EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 3u);
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @signed_negative(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  %producer = test.event.fast.i32 %lhs, %rhs
  %consumer = test.event.consume.late.i32 %producer, %rhs
  return %consumer
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ASSERT_EQ(frame.schedule.dependencies.count, 2u);
    ExpectDependencyTiming(frame, 0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, -2,
                           LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
                           LOOM_LOW_MODEL_QUALITY_EXACT);
    EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 0u);
  }
}

TEST_F(LowEmissionFrameTest, SharedResourceCapacityShapesIssueCycles) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @shared_resource(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>, reg<test.i32>) asm {
  %first = test.event.fast.i32 %lhs, %rhs
  %second = test.event.fast.i32 %lhs, %rhs
  %third = test.event.fast.i32 %lhs, %rhs
  return %first, %second, %third
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  ASSERT_EQ(frame.schedule.node_count, 4u);
  EXPECT_EQ(frame.schedule.nodes[0].issue_cycle, 0u);
  EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 0u);
  EXPECT_EQ(frame.schedule.nodes[2].issue_cycle, 2u);
}

TEST_F(LowEmissionFrameTest, EffectTimingDistinguishesReadWriteDirection) {
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @write_after_read(%read_value: reg<test.i32>, %write_value: reg<test.i32>) asm {
  test.event.memory.read.i32 %read_value
  test.event.memory.write.i32 %write_value
  return
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ASSERT_EQ(frame.schedule.dependencies.count, 1u);
    ExpectDependencyTiming(frame, 0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, 0,
                           LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
                           LOOM_LOW_MODEL_QUALITY_EXACT);
    EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 0u);
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @read_after_write(%write_value: reg<test.i32>, %read_value: reg<test.i32>) asm {
  test.event.memory.write.i32 %write_value
  test.event.memory.read.i32 %read_value
  return
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ASSERT_EQ(frame.schedule.dependencies.count, 1u);
    ExpectDependencyTiming(frame, 0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, 2,
                           LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
                           LOOM_LOW_MODEL_QUALITY_EXACT);
    EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 2u);
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @write_after_write(%first_value: reg<test.i32>, %second_value: reg<test.i32>) asm {
  test.event.memory.write.i32 %first_value
  test.event.memory.write.i32 %second_value
  return
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ASSERT_EQ(frame.schedule.dependencies.count, 1u);
    ExpectDependencyTiming(frame, 0, 1, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, 1,
                           LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
                           LOOM_LOW_MODEL_QUALITY_EXACT);
    EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 1u);
  }
}

TEST_F(LowEmissionFrameTest, SourceOrderBoundariesPreserveSegments) {
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @move_boundary(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %before = test.mul.i32 %lhs, %rhs
  %moved = move %lhs : reg<test.i32> -> reg<test.i32>
  %after = test.add.i32 %moved, %rhs
  return %before, %after
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ASSERT_EQ(frame.schedule.node_count, 4u);
    EXPECT_TRUE(
        iree_any_bit_set(frame.schedule.nodes[1].flags,
                         LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY));
    EXPECT_EQ(frame.schedule.nodes[0].scheduled_ordinal, 0u);
    EXPECT_EQ(frame.schedule.nodes[1].scheduled_ordinal, 1u);
    EXPECT_EQ(frame.schedule.nodes[2].scheduled_ordinal, 2u);
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @schedule_fence(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %before = test.add.i32 %lhs, %rhs
  low.schedule.fence
  %after = test.mul.i32 %lhs, %rhs
  return %before, %after
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ASSERT_EQ(frame.schedule.node_count, 4u);
    EXPECT_TRUE(
        iree_any_bit_set(frame.schedule.nodes[1].flags,
                         LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY));
    EXPECT_EQ(frame.schedule.nodes[0].scheduled_ordinal, 0u);
    EXPECT_EQ(frame.schedule.nodes[1].scheduled_ordinal, 1u);
    EXPECT_EQ(frame.schedule.nodes[2].scheduled_ordinal, 2u);
    EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 1u);
    EXPECT_EQ(frame.schedule.nodes[2].issue_cycle, 1u);
  }
  {
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @barrier_segments(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>, reg<test.i32>) asm {
  %before = test.add.i32 %lhs, %rhs
  test.barrier
  %middle = test.mul.i32 %lhs, %rhs
  test.barrier
  %after = test.add.i32 %lhs, %rhs
  return %before, %middle, %after
}
)");
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
    ASSERT_EQ(frame.schedule.node_count, 6u);
    for (uint32_t i = 0; i < frame.schedule.node_count; ++i) {
      EXPECT_EQ(frame.schedule.nodes[i].scheduled_ordinal, i);
    }
    EXPECT_TRUE(
        iree_any_bit_set(frame.schedule.nodes[1].flags,
                         LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY));
    EXPECT_TRUE(
        iree_any_bit_set(frame.schedule.nodes[3].flags,
                         LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY));
    ExpectDependencyTiming(frame, 1, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, 2,
                           LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
                           LOOM_LOW_MODEL_QUALITY_EXACT);
    EXPECT_EQ(frame.schedule.nodes[4].issue_cycle, 4u);
  }
}

TEST_F(LowEmissionFrameTest, StructuralPreambleStaysBeforeInstructions) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @preamble(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  %resource = resource<native_pointer> {index = 0, source_type = buffer} : reg<test.ptr>
  %live0 = live_in<test.arg0> : reg<test.i32>
  %live1 = live_in<test.arg0> : reg<test.i32>
  %value = test.add.i32 %lhs, %rhs
  return %value
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  ASSERT_EQ(frame.schedule.node_count, 5u);
  EXPECT_TRUE(loom_low_resource_isa(frame.schedule.nodes[0].op));
  EXPECT_TRUE(loom_low_live_in_isa(frame.schedule.nodes[1].op));
  EXPECT_TRUE(loom_low_live_in_isa(frame.schedule.nodes[2].op));
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(frame.schedule.nodes[i].scheduled_ordinal, i);
    EXPECT_EQ(frame.schedule.nodes[i].issue_cycle, 0u);
  }
  EXPECT_EQ(frame.schedule.nodes[3].scheduled_ordinal, 3u);
}

TEST_F(LowEmissionFrameTest, EquivalentDescriptorsRetainSelectedModel) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @alternatives(%lhs0: reg<test.i32>, %rhs0: reg<test.i32>, %lhs1: reg<test.i32>, %rhs1: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %a = test.schedule.alternative.a.i32 %lhs0, %rhs0
  %b = test.schedule.alternative.a.i32 %lhs1, %rhs1
  return %a, %b
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  ASSERT_EQ(frame.schedule.node_count, 3u);
  EXPECT_EQ(frame.schedule.nodes[1].source_descriptor_ordinal,
            TEST_LOW_CORE_DESCRIPTOR_REF_TEST_SCHEDULE_ALTERNATIVE_A_I32);
  EXPECT_EQ(frame.schedule.nodes[1].descriptor,
            loom_low_descriptor_set_descriptor_at(
                descriptor_set,
                TEST_LOW_CORE_DESCRIPTOR_REF_TEST_SCHEDULE_ALTERNATIVE_B_I32));
  EXPECT_EQ(frame.schedule.nodes[0].issue_cycle, 0u);
  EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 0u);
}

TEST_F(LowEmissionFrameTest, IssuedConstantFillsDependencyLatency) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @constant_latency(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  %producer = test.event.slow.i32 %lhs, %rhs
  %constant = test.const.issued.i32 1
  %consumer = test.add.i32 %producer, %rhs
  %result = test.add.i32 %consumer, %constant
  return %result
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  ASSERT_EQ(frame.schedule.node_count, 5u);
  EXPECT_EQ(frame.schedule.nodes[0].issue_cycle, 0u);
  EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 0u);
  EXPECT_EQ(frame.schedule.nodes[2].issue_cycle, 1u);
  EXPECT_EQ(frame.schedule.nodes[3].issue_cycle, 2u);
}

TEST_F(LowEmissionFrameTest, MaterializationPreservesReadyDescriptorPair) {
  constexpr loom_low_schedule_strategy_t kScheduleStrategies[] = {
      LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
      LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING,
      LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
  };
  for (const loom_low_schedule_strategy_t strategy : kScheduleStrategies) {
    SCOPED_TRACE(static_cast<int>(strategy));
    ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @constant_pair(%input: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %first = test.const.issued.i32 1
  %second = test.const.issued.i32 2
  %first_result = test.add.i32 %input, %first
  %second_result = test.add.i32 %input, %second
  return %first_result, %second_result
}
)");
    const loom_low_descriptor_set_t* descriptor_set =
        loom_test_low_core_descriptor_set();
    const loom_low_descriptor_t* constant =
        loom_low_descriptor_set_descriptor_at(
            descriptor_set, TEST_LOW_CORE_DESCRIPTOR_REF_TEST_CONST_ISSUED_I32);
    loom_low_schedule_pair_affinity_t affinity = {};
    affinity.first_descriptor = constant;
    affinity.second_descriptor = constant;
    affinity.priority = 1;
    loom_low_emission_frame_options_t options = {};
    options.descriptor_registry = &registry_.registry;
    options.schedule_pair_affinities = {&affinity, 1};
    options.schedule_strategy = strategy;
    loom_low_emission_frame_t frame = {};
    bool frame_accepted = false;
    IREE_ASSERT_OK(loom_low_emission_frame_build(
        module.get(), loom_block_op(loom_module_block(module.get()), 0),
        &options, &arena_, &frame, &frame_accepted));
    ASSERT_TRUE(frame_accepted);
    ASSERT_EQ(frame.schedule.error_count, 0u);
    EXPECT_EQ(frame.schedule.nodes[1].scheduled_ordinal,
              frame.schedule.nodes[0].scheduled_ordinal + 1);
    EXPECT_LT(frame.schedule.nodes[1].scheduled_ordinal,
              frame.schedule.nodes[2].scheduled_ordinal);
  }
}

TEST_F(LowEmissionFrameTest, MaterializationPairCannotExceedRegisterBudget) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @bounded_pair(%address: reg<test.ptr>, %value: reg<test.i32 x4>) asm {
  %first = test.const.issued.i32 0
  %second = test.const.issued.i32 4
  test.store.index.v4i32 %address, %first, %value
  test.store.index.v4i32 %address, %second, %value
  return
}
)");
  const loom_low_descriptor_t* constant = loom_low_descriptor_set_descriptor_at(
      loom_test_low_core_descriptor_set(),
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_CONST_ISSUED_I32);
  loom_low_schedule_pair_affinity_t affinity = {};
  affinity.first_descriptor = constant;
  affinity.second_descriptor = constant;
  affinity.priority = 1;
  loom_low_allocation_budget_t budget = {};
  budget.register_class = IREE_SV("test.i32");
  budget.max_units = 5;
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_pair_affinities = {&affinity, 1};
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
  options.allocation_budgets = &budget;
  options.allocation_budget_count = 1;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &frame_accepted));
  ASSERT_TRUE(frame_accepted);
  ASSERT_EQ(frame.schedule.error_count, 0u);
  EXPECT_LT(frame.schedule.nodes[2].scheduled_ordinal,
            frame.schedule.nodes[1].scheduled_ordinal);
  EXPECT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.allocation.spill_count, 0u);
}

TEST_F(LowEmissionFrameTest, RetainsOnlyDecidedSpillsAfterAllocation) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @spills(%first: reg<test.i32>, %second: reg<test.i32>, %third: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>, reg<test.i32>) asm {
  return %first, %second, %third
}
)");
  loom_low_allocation_budget_t budget = {};
  budget.register_class = IREE_SV("test.i32");
  budget.max_units = 1;
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
  options.allocation_budgets = &budget;
  options.allocation_budget_count = 1;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &frame_accepted));
  ASSERT_TRUE(frame_accepted);
  iree_arena_block_pool_trim(&block_pool_);
  const loom_low_allocation_table_t& allocation = frame.allocation;
  ASSERT_EQ(allocation.error_count, 0u);
  ASSERT_EQ(allocation.assignment_count, 3u);
  ASSERT_EQ(allocation.spill_count, 2u);
  ASSERT_EQ(allocation.spill_plan_count, 2u);
  ASSERT_EQ(allocation.remark_count, 2u);
  for (uint32_t i = 0; i < 2; ++i) {
    const loom_low_allocation_spill_plan_t& plan = allocation.spill_plans[i];
    EXPECT_EQ(plan.assignment_index, i + 1);
    EXPECT_EQ(plan.value_id, allocation.assignments[i + 1].value_id);
    EXPECT_EQ(plan.slot_index, i);
    EXPECT_EQ(plan.byte_size, 4u);
    EXPECT_EQ(plan.store_count, 1u);
    EXPECT_EQ(plan.reload_count, 1u);
    EXPECT_EQ(allocation.remarks[i].assignment_index, plan.assignment_index);
    EXPECT_EQ(allocation.remarks[i].budget_units, 1u);
    EXPECT_EQ(allocation.remarks[i].required_units, 1u);
  }
}

TEST_F(LowEmissionFrameTest, FeedbackConsumesTheAcceptedFrame) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @feedback(%lhs: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  %first = test.resource.serial.i32 %lhs, %rhs
  %second = test.resource.serial.i32 %rhs, %lhs
  %copy = copy %first : reg<test.i32> -> reg<test.i32>
  %sum = test.add.i32 %copy, %second
  return %sum
}
)");
  struct CapturedDiagnostics {
    // Total feedback records delivered by the frame.
    uint32_t count = 0;
    // Pressure budget retained independently of the caller's option storage.
    uint32_t pressure_budget = 0;
    // Number of allocation copy decisions observed.
    uint32_t copy_count = 0;
  } captured;
  iree_diagnostic_emitter_t emitter = {};
  emitter.fn = [](void* user_data, const loom_diagnostic_emission_t* emission) {
    auto* captured = static_cast<CapturedDiagnostics*>(user_data);
    ++captured->count;
    if (emission->error == LOOM_ERR_BACKEND_003) {
      captured->pressure_budget = emission->params[5].u32;
    } else if (emission->error == LOOM_ERR_BACKEND_006) {
      ++captured->copy_count;
    }
    return iree_ok_status();
  };
  emitter.user_data = &captured;
  loom_low_allocation_budget_t budget = {IREE_SV("test.i32"), 8};
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
  options.schedule_diagnostic_flags =
      LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS |
      LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS |
      LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS |
      LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS |
      LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY;
  options.allocation_budgets = &budget;
  options.allocation_budget_count = 1;
  options.allocation_diagnostic_flags =
      LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS;
  options.emitter = emitter;
  options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  spill_free_options.materialization_options.has_supported_storage_spaces =
      true;
  spill_free_options.materialization_options.supported_storage_spaces =
      LOOM_LOW_STORAGE_SPACE_SET_NONE;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame, &frame_accepted));
  ASSERT_TRUE(frame_accepted);
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(statistics.frame_build_count, 1u);
  EXPECT_EQ(statistics.allocation_run_count, 1u);
  EXPECT_GT(captured.count, 1u);
  EXPECT_EQ(captured.pressure_budget, 8u);
  EXPECT_EQ(captured.copy_count, 1u);

  loom_low_emission_frame_options_t quiet_options = options;
  quiet_options.schedule_diagnostic_flags = 0;
  quiet_options.allocation_diagnostic_flags = 0;
  quiet_options.statistics = nullptr;
  loom_low_emission_frame_t quiet_frame = {};
  bool quiet_frame_accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0),
      &quiet_options, &spill_free_options, &arena_, &quiet_frame,
      &quiet_frame_accepted));
  ASSERT_TRUE(quiet_frame_accepted);
  // Release pooled scratch before comparing schedules and formatting retained
  // diagnostics. ASAN catches any accidental result borrowing from that state.
  iree_arena_block_pool_trim(&block_pool_);
  ASSERT_EQ(quiet_frame.schedule.node_count, frame.schedule.node_count);
  for (iree_host_size_t i = 0; i < frame.schedule.node_count; ++i) {
    EXPECT_EQ(quiet_frame.schedule.scheduled_node_indices[i],
              frame.schedule.scheduled_node_indices[i]);
    EXPECT_EQ(quiet_frame.schedule.nodes[i].issue_cycle,
              frame.schedule.nodes[i].issue_cycle);
  }
  ASSERT_EQ(quiet_frame.allocation.assignment_count,
            frame.allocation.assignment_count);
  for (iree_host_size_t i = 0; i < frame.allocation.assignment_count; ++i) {
    const auto& quiet = quiet_frame.allocation.assignments[i];
    const auto& verbose = frame.allocation.assignments[i];
    EXPECT_EQ(quiet.value_id, verbose.value_id);
    EXPECT_EQ(quiet.location_kind, verbose.location_kind);
    EXPECT_EQ(quiet.location_base, verbose.location_base);
    EXPECT_EQ(quiet.location_count, verbose.location_count);
  }

  // Formatting is a read-only consumer after the model and option scratch
  // have gone away. Neither another plan nor another arena allocation occurs.
  budget.max_units = 1;
  captured = {};
  const iree_host_size_t used_bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_low_schedule_diagnostics_emit(
      &frame.schedule, options.schedule_diagnostic_flags, emitter));
  EXPECT_GT(captured.count, 0u);
  EXPECT_EQ(captured.pressure_budget, 8u);
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
  EXPECT_EQ(statistics.frame_build_count, 1u);
}

TEST_F(LowEmissionFrameTest, FeedbackConsumesTheRejectedFrame) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @too_wide(%wide: reg<test.special x2>) -> (reg<test.special x2>) asm {
  return %wide
}
)");
  struct CapturedFailure {
    // Number of terminal allocation diagnostics.
    uint32_t count = 0;
    // Original operation anchoring the diagnostic.
    const loom_op_t* op = nullptr;
  } captured;
  loom_low_allocation_budget_t budget = {IREE_SV("test.special"), 1};
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
  options.allocation_budgets = &budget;
  options.allocation_budget_count = 1;
  options.emitter.fn = [](void* user_data,
                          const loom_diagnostic_emission_t* emission) {
    auto* captured = static_cast<CapturedFailure*>(user_data);
    EXPECT_EQ(emission->error, LOOM_ERR_BACKEND_005);
    EXPECT_EQ(emission->params[5].u32, 1u);
    EXPECT_EQ(emission->params[6].u32, 2u);
    ++captured->count;
    captured->op = emission->op;
    return iree_ok_status();
  };
  options.emitter.user_data = &captured;
  options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  spill_free_options.materialization_options.has_supported_storage_spaces =
      true;
  spill_free_options.materialization_options.supported_storage_spaces =
      LOOM_LOW_STORAGE_SPACE_SET_NONE;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = true;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame, &frame_accepted));
  EXPECT_FALSE(frame_accepted);
  ASSERT_EQ(frame.allocation.error_count, 1u);
  EXPECT_EQ(captured.count, 1u);
  EXPECT_EQ(captured.op, frame.allocation.failure.op);
  EXPECT_EQ(statistics.frame_build_count, 1u);
  EXPECT_EQ(statistics.allocation_run_count, 1u);

  const iree_host_size_t used_bytes = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_low_allocation_diagnostics_emit(
      &frame.allocation, /*flags=*/0, options.emitter));
  EXPECT_EQ(captured.count, 2u);
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
}

TEST_F(LowEmissionFrameTest, FinalValidatorRejectsWithoutDiagnosticSink) {
  ModulePtr module = ParseModule();
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  spill_free_options.materialization_options.has_supported_storage_spaces =
      true;
  spill_free_options.materialization_options.supported_storage_spaces =
      LOOM_LOW_STORAGE_SPACE_SET_NONE;
  bool validator_invoked = false;
  spill_free_options.validate_frame =
      [](void* user_data, const loom_low_emission_frame_t* frame,
         iree_arena_allocator_t* arena, bool* out_accepted) {
        bool* invoked = static_cast<bool*>(user_data);
        EXPECT_FALSE(*invoked);
        *invoked = true;
        EXPECT_EQ(frame->schedule.error_count, 0u);
        EXPECT_EQ(frame->allocation.error_count, 0u);
        (void)arena;
        *out_accepted = false;
        return iree_ok_status();
      };
  spill_free_options.validate_frame_user_data = &validator_invoked;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = true;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame, &frame_accepted));
  EXPECT_TRUE(validator_invoked);
  EXPECT_FALSE(frame_accepted);
}

TEST_F(LowEmissionFrameTest, InputErrorsAreNotDeferredOrReplayed) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @invalid_budget(%value: reg<test.i32>) -> (reg<test.i32>) asm {
  return %value
}
)");
  uint32_t diagnostic_count = 0;
  const loom_low_allocation_budget_t budgets[] = {
      {IREE_SV("test.i32"), 1},
      {IREE_SV("test.i32"), 2},
  };
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
  options.allocation_budgets = budgets;
  options.allocation_budget_count = IREE_ARRAYSIZE(budgets);
  options.emitter.fn = [](void* user_data,
                          const loom_diagnostic_emission_t* emission) {
    EXPECT_EQ(emission->error, LOOM_ERR_BACKEND_024);
    ++*static_cast<uint32_t*>(user_data);
    return iree_ok_status();
  };
  options.emitter.user_data = &diagnostic_count;
  options.statistics = &statistics;
  loom_low_emission_frame_spill_free_options_t spill_free_options = {};
  spill_free_options.materialization_options.has_supported_storage_spaces =
      true;
  spill_free_options.materialization_options.supported_storage_spaces =
      LOOM_LOW_STORAGE_SPACE_SET_NONE;
  loom_low_emission_frame_t frame = {};
  bool frame_accepted = true;
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame, &frame_accepted));
  EXPECT_FALSE(frame_accepted);
  ASSERT_EQ(frame.allocation.error_count, 1u);
  EXPECT_FALSE(
      loom_low_allocation_failure_is_present(&frame.allocation.failure));
  EXPECT_EQ(diagnostic_count, 1u);
  EXPECT_EQ(statistics.frame_build_count, 1u);
  EXPECT_EQ(statistics.allocation_run_count, 1u);
}

TEST_F(LowEmissionFrameTest, SingletonReservationsPreserveTiedStateUpdates) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @fixed_updates(%state: reg<test.fixed.r0>, %delta: reg<test.i32>, %independent: reg<test.explicit32>) -> (reg<test.fixed.r0>, reg<test.explicit32>) asm {
  %first = test.fixed.update.i32 %state, %delta
  %second = test.fixed.update.i32 %first, %delta
  return %second, %independent
}
)");
  ASSERT_NE(module, nullptr);
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.allocation.spill_count, 0u);
  EXPECT_EQ(frame.allocation.materialized_copy_count, 0u);
  uint32_t fixed_count = 0;
  for (iree_host_size_t i = 0; i < frame.allocation.assignment_count; ++i) {
    const auto& assignment = frame.allocation.assignments[i];
    if (assignment.descriptor_reg_class_id ==
        TEST_LOW_CORE_REG_CLASS_ID_TEST_EXPLICIT32) {
      EXPECT_NE(assignment.location_base, 0u);
    }
    if (assignment.descriptor_reg_class_id !=
        TEST_LOW_CORE_REG_CLASS_ID_TEST_FIXED_R0) {
      continue;
    }
    EXPECT_EQ(assignment.location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
    EXPECT_EQ(assignment.location_base, 0u);
    ++fixed_count;
  }
  EXPECT_EQ(fixed_count, 3u);
}

TEST_F(LowEmissionFrameTest, SingletonTiedStateCrossesDynamicLoopBackedge) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @fixed_loop(%initial: reg<test.fixed.r0>, %count: reg<test.i32>) -> (reg<test.fixed.r0>) asm {
  %decrement = test.const.i32 -1
  low.br ^loop(%initial: reg<test.fixed.r0>, %count: reg<test.i32>)
^loop(%state: reg<test.fixed.r0>, %remaining: reg<test.i32>):
  low.cond_br %remaining, ^body, ^exit : reg<test.i32>
^body:
  %next_state = test.fixed.update.i32 %state, %decrement
  %next_remaining = test.add.i32 %remaining, %decrement
  low.br ^loop(%next_state: reg<test.fixed.r0>, %next_remaining: reg<test.i32>)
^exit:
  return %state
}
)");
  ASSERT_NE(module, nullptr);
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.allocation.spill_count, 0u);
  EXPECT_EQ(frame.allocation.materialized_copy_count, 0u);
  for (iree_host_size_t i = 0; i < frame.allocation.assignment_count; ++i) {
    const auto& assignment = frame.allocation.assignments[i];
    if (assignment.descriptor_reg_class_id ==
        TEST_LOW_CORE_REG_CLASS_ID_TEST_FIXED_R0) {
      EXPECT_EQ(assignment.location_base, 0u);
    }
  }
}

TEST_F(LowEmissionFrameTest, StateOnlyScheduleLivenessIsTransient) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @state_live_out(%state: reg<test.schedule_state>, %rhs: reg<test.schedule_state>) asm {
  low.br ^use
^use:
  %read = test.explicit.state.add.schedule_state %rhs, %state
  return
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
  iree_arena_block_pool_trim(&block_pool_);

  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  EXPECT_EQ(frame.schedule.liveness.region, nullptr);
}

TEST_F(LowEmissionFrameTest, FailedScheduleDoesNotPublishSourceSuffixBounds) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @dependency_cycle(%lhs: reg<test.schedule_state>, %rhs: reg<test.schedule_state>) asm {
  low.br ^cycle
^cycle:
  %old = test.add.schedule_state %lhs, %rhs
  %writer = test.state.add.schedule_state %lhs, %rhs
  %cycle = test.explicit.state.add.schedule_state %writer, %old
  return
}
)");
  uint32_t diagnostic_count = 0;
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
  options.emitter.fn = [](void* user_data,
                          const loom_diagnostic_emission_t* emission) {
    EXPECT_EQ(emission->error, LOOM_ERR_BACKEND_044);
    ++*static_cast<uint32_t*>(user_data);
    return iree_ok_status();
  };
  options.emitter.user_data = &diagnostic_count;
  loom_low_emission_frame_t frame = {};
  bool accepted = true;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &accepted));

  EXPECT_FALSE(accepted);
  EXPECT_EQ(frame.schedule.error_count, 1u);
  EXPECT_EQ(frame.schedule.source_suffix_issue_cycle_lower_bounds, nullptr);
  EXPECT_EQ(diagnostic_count, 1u);
}

TEST_F(LowEmissionFrameTest, ProvenNoOpGuardedMotionSkipsTrialFrame) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @guarded_tail(%base: reg<test.ptr>, %origin: reg<test.i32>, %pixel: reg<test.i32>, %color: reg<test.i32 x4>) -> (reg<test.i32>) asm {
  %condition = test.event.fast.i32 %origin, %pixel
  low.cond_br %condition, ^store, ^done : reg<test.i32>
^store:
  %index = test.total.add.i32 %origin, %pixel
  test.store.index.v4i32 %base, %index, %color
  %tail = test.event.fast.i32 %origin, %pixel
  %result = test.add.i32 %tail, %pixel
  return %result
^done:
  return %condition
}
)");
  loom_low_planning_statistics_t statistics = {};
  loom_low_emission_frame_options_t options = {};
  options.descriptor_registry = &registry_.registry;
  options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
  options.statistics = &statistics;
  loom_low_emission_frame_t frame = {};
  bool accepted = false;
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame, &accepted));

  EXPECT_TRUE(accepted);
  EXPECT_NE(frame.schedule.source_suffix_issue_cycle_lower_bounds, nullptr);
  EXPECT_EQ(statistics.frame_build_count, 1u);
  EXPECT_EQ(statistics.allocation_run_count, 1u);

  loom_low_schedule_table_t unproven_schedule = frame.schedule;
  unproven_schedule.source_suffix_issue_cycle_lower_bounds = nullptr;
  loom_low_guarded_motion_plan_t unproven_plan = {};
  IREE_ASSERT_OK(loom_low_guarded_motion_plan(&unproven_schedule, &arena_,
                                              &unproven_plan));
  ASSERT_EQ(unproven_plan.region_count, 1u);
  const loom_low_guarded_motion_region_t& region = unproven_plan.regions[0];
  const uint32_t source_suffix = region.node_start + region.node_count;
  const uint32_t source_block =
      frame.schedule.nodes[region.node_start].block_index;
  const loom_low_schedule_block_t& block = frame.schedule.blocks[source_block];
  ASSERT_NE(block.issue_group_count, 0u);
  const uint32_t source_extent =
      frame.schedule
          .issue_groups[block.issue_group_start + block.issue_group_count - 1]
          .issue_cycle;
  EXPECT_EQ(
      frame.schedule.source_suffix_issue_cycle_lower_bounds[source_suffix],
      source_extent);
}

TEST_F(LowEmissionFrameTest, RetainsEveryUnobservedWrite) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @unobserved_writes(%address: reg<test.ptr>, %value: reg<test.i32 x4>) asm {
  test.store.v4i32 %address, %value
  test.store.v4i32 %address, %value
  test.store.v4i32 %address, %value
  %loaded = test.load.v4i32 %address
  return
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  // Write-after-write issue order does not establish completion, so the read
  // must wait for every outstanding write.
  ExpectEffectPredecessors(frame, /*consumer_node=*/3, {0, 1, 2});
}

TEST_F(LowEmissionFrameTest, RetiresCompletedMemoryFrontiers) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @batched_accesses(%address: reg<test.ptr>, %value: reg<test.i32 x4>) asm {
  %loaded0 = test.load.v4i32 %address
  %loaded1 = test.load.v4i32 %address
  %loaded2 = test.load.v4i32 %address
  test.store.v4i32 %address, %value
  test.store.v4i32 %address, %value
  %loaded5 = test.load.v4i32 %address
  %loaded6 = test.load.v4i32 %address
  test.store.v4i32 %address, %value
  test.store.v4i32 %address, %value
  %loaded9 = test.load.v4i32 %address
  return
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  // Each opposite-kind burst observes the complete outstanding frontier. Once
  // observed, that frontier retires and does not leak into the next burst.
  ExpectEffectPredecessors(frame, /*consumer_node=*/3, {0, 1, 2});
  ExpectEffectPredecessors(frame, /*consumer_node=*/4, {0, 1, 2, 3});
  ExpectEffectPredecessors(frame, /*consumer_node=*/5, {3, 4});
  ExpectEffectPredecessors(frame, /*consumer_node=*/6, {3, 4});
  ExpectEffectPredecessors(frame, /*consumer_node=*/7, {3, 4, 5, 6});
  ExpectEffectPredecessors(frame, /*consumer_node=*/8, {5, 6, 7});
  ExpectEffectPredecessors(frame, /*consumer_node=*/9, {7, 8});
}

TEST_F(LowEmissionFrameTest, OrderedEffectPreservesMemoryFrontier) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @ordered_effect(%address: reg<test.ptr>, %value: reg<test.i32 x4>) asm {
  test.store.v4i32 %address, %value
  test.store.v4i32 %address, %value
  %loaded2 = test.load.v4i32 %address
  test.barrier
  %loaded4 = test.load.v4i32 %address
  test.store.v4i32 %address, %value
  test.store.v4i32 %address, %value
  %loaded7 = test.load.v4i32 %address
  return
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  // The ordered effect joins the current frontier without replacing it. The
  // next opposite-kind burst retires the older memory accesses but retains the
  // ordered effect itself.
  ExpectEffectPredecessors(frame, /*consumer_node=*/4, {0, 1, 3});
  ExpectEffectPredecessors(frame, /*consumer_node=*/7, {3, 5, 6});
}

TEST_F(LowEmissionFrameTest, CopyExtendsLoopHeaderStorageLifetime) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @copied_header(%seed: reg<test.i32>, %rhs: reg<test.i32>) asm {
  low.br ^loop(%seed: reg<test.i32>)
^loop(%cursor: reg<test.i32>):
  %view = copy %cursor : reg<test.i32> -> reg<test.i32>
  %alias = copy %view : reg<test.i32> -> reg<test.i32>
  %read = test.add.i32 %alias, %rhs
  %next = test.mul.i32 %cursor, %rhs
  test.event.memory.write.i32 %read
  low.br ^loop(%next: reg<test.i32>)
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/4, {3});
}

TEST_F(LowEmissionFrameTest, DetachedAndPartialCopiesDoNotExtendHeader) {
  ModulePtr detached_module = ParseModule(R"(
low.func.def target<test.low.core> @detached_header(%seed: reg<test.i32>, %rhs: reg<test.i32>) asm {
  low.br ^loop(%seed: reg<test.i32>)
^loop(%cursor: reg<test.i32>):
  %view = copy %cursor {detached = true} : reg<test.i32> -> reg<test.i32>
  %alias = copy %view : reg<test.i32> -> reg<test.i32>
  %read = test.add.i32 %alias, %rhs
  %next = test.mul.i32 %cursor, %rhs
  test.event.memory.write.i32 %read
  low.br ^loop(%next: reg<test.i32>)
}
)");
  loom_low_emission_frame_t detached_frame = {};
  IREE_ASSERT_OK(BuildFrame(detached_module.get(), {}, &detached_frame));
  ExpectStorageLifetimePredecessors(detached_frame, /*consumer_node=*/4, {});

  ModulePtr partial_module = ParseModule(R"(
low.func.def target<test.low.core> @partial_header(%seed: reg<test.i32 x4>, %rhs: reg<test.i32>, %rhs_vector: reg<test.i32 x4>) asm {
  low.br ^loop(%seed: reg<test.i32 x4>)
^loop(%cursor: reg<test.i32 x4>):
  %lane = slice %cursor[0] : reg<test.i32 x4> -> reg<test.i32>
  %view = copy %lane : reg<test.i32> -> reg<test.i32>
  %read = test.add.i32 %view, %rhs
  %next = test.add.v4i32 %cursor, %rhs_vector
  test.event.memory.write.i32 %read
  low.br ^loop(%next: reg<test.i32 x4>)
}
)");
  loom_low_emission_frame_t partial_frame = {};
  IREE_ASSERT_OK(BuildFrame(partial_module.get(), {}, &partial_frame));
  ExpectStorageLifetimePredecessors(partial_frame, /*consumer_node=*/4, {});
}

TEST_F(LowEmissionFrameTest, TiedStorageRetainsAllReaders) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @tied_header(%seed: reg<test.i32>, %rhs: reg<test.i32>) asm {
  low.br ^loop(%seed: reg<test.i32>)
^loop(%cursor: reg<test.i32>):
  %view = copy %cursor : reg<test.i32> -> reg<test.i32>
  %before = test.add.i32 %view, %rhs
  %advanced = test.tied.any %view
  %read = test.add.i32 %advanced, %before
  %next = test.mul.i32 %cursor, %rhs
  test.event.memory.write.i32 %read
  low.br ^loop(%next: reg<test.i32>)
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/3, {2});
  ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/5, {2, 3, 4});
}

TEST_F(LowEmissionFrameTest, BackedgeRetainsProducerBlockReaders) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @forwarded_header(%condition: reg<test.i32>, %seed: reg<test.i32>, %rhs: reg<test.i32>) -> (reg<test.i32>) asm {
  low.br ^loop(%seed: reg<test.i32>)
^loop(%cursor: reg<test.i32>):
  %view = copy %cursor : reg<test.i32> -> reg<test.i32>
  %read = test.add.i32 %view, %rhs
  %next = test.mul.i32 %cursor, %rhs
  low.cond_br %condition, ^forward, ^exit : reg<test.i32>
^forward:
  low.br ^loop(%next: reg<test.i32>)
^exit:
  return %read
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/3, {2});
}

TEST_F(LowEmissionFrameTest, BackedgeRetainsAllHeaderReaders) {
  const auto expect_readers = [this](const char* source,
                                     int32_t minimum_issue_separation_cycles,
                                     loom_low_model_quality_t model_quality) {
    ModulePtr module = ParseModule(source);
    loom_low_emission_frame_t frame = {};
    IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

    ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/4, {2, 3});
    ExpectDependencyTiming(frame, 2, 4, LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE,
                           minimum_issue_separation_cycles,
                           LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_SCHEDULE_CLASS,
                           model_quality);
    ExpectDependencyTiming(frame, 3, 4, LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE,
                           minimum_issue_separation_cycles,
                           LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_SCHEDULE_CLASS,
                           model_quality);
  };

  expect_readers(R"(
low.func.def target<test.low.core> @scalar_backedge(%condition: reg<test.i32>, %seed: reg<test.i32>, %rhs0: reg<test.i32>, %rhs1: reg<test.i32>) -> (reg<test.i32>) asm {
  low.br ^loop(%seed: reg<test.i32>)
^loop(%state: reg<test.i32>):
  low.cond_br %condition, ^body, ^exit : reg<test.i32>
^body:
  %read0 = test.add.i32 %state, %rhs0
  %read1 = test.mul.i32 %state, %rhs1
  %next = test.add.i32 %state, %rhs1
  low.br ^loop(%next: reg<test.i32>)
^exit:
  return %state
}
)",
                 /*minimum_issue_separation_cycles=*/1,
                 LOOM_LOW_MODEL_QUALITY_EXACT);
  expect_readers(R"(
low.func.def target<test.low.core> @vector_backedge(%condition: reg<test.i32>, %seed: reg<test.i32 x4>, %rhs0: reg<test.i32 x4>, %rhs1: reg<test.i32 x4>) -> (reg<test.i32 x4>) asm {
  low.br ^loop(%seed: reg<test.i32 x4>)
^loop(%state: reg<test.i32 x4>):
  low.cond_br %condition, ^body, ^exit : reg<test.i32>
^body:
  %read0 = test.add.v4i32 %state, %rhs0
  %read1 = test.add.v4i32 %state, %rhs1
  %next = test.add.v4i32 %state, %rhs1
  low.br ^loop(%next: reg<test.i32 x4>)
^exit:
  return %state
}
)",
                 /*minimum_issue_separation_cycles=*/2,
                 LOOM_LOW_MODEL_QUALITY_ESTIMATED);
}

TEST_F(LowEmissionFrameTest, ComposedBackedgeRetainsHeaderReader) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @composed_backedge(%condition: reg<test.i32>, %seed: reg<test.i32 x4>, %rhs0: reg<test.i32 x4>, %rhs1: reg<test.i32 x4>) -> (reg<test.i32 x4>) asm {
  low.br ^loop(%seed: reg<test.i32 x4>)
^loop(%state: reg<test.i32 x4>):
  low.cond_br %condition, ^body, ^exit : reg<test.i32>
^body:
  %read = test.add.v4i32 %state, %rhs0
  %next_low = test.add.v4i32 %rhs0, %rhs1
  %next_high = test.add.v4i32 %rhs1, %rhs0
  %wide = concat(%next_low, %next_high) : (reg<test.i32 x4>, reg<test.i32 x4>) -> reg<test.i32 x8>
  %next_slice = slice %wide[0] : reg<test.i32 x8> -> reg<test.i32 x4>
  %next = copy %next_slice : reg<test.i32 x4> -> reg<test.i32 x4>
  low.br ^loop(%next: reg<test.i32 x4>)
^exit:
  return %state
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/3, {2});
  ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/5, {2});
  ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/6, {2});
  ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/7, {2});
}

TEST_F(LowEmissionFrameTest, TiedWritesRetainOverlappingReaders) {
  const auto expect_reader =
      [this](const char* source,
             std::initializer_list<uint32_t> expected_producer_nodes) {
        ModulePtr module = ParseModule(source);
        loom_low_emission_frame_t frame = {};
        IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));
        ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/1,
                                          expected_producer_nodes);
      };

  expect_reader(R"(
low.func.def target<test.low.core> @low_read_high_write(%state: reg<test.i32>, %ptr: reg<test.ptr>) -> (reg<test.i32>, reg<test.i32>) asm {
  %read = test.read.low16.i32 %state
  %next = test.write.high16.i32 %state, %ptr
  return %read, %next
}
)",
                {});
  expect_reader(R"(
low.func.def target<test.low.core> @high_read_high_write(%state: reg<test.i32>, %ptr: reg<test.ptr>) -> (reg<test.i32>, reg<test.i32>) asm {
  %read = test.read.high16.i32 %state
  %next = test.write.high16.i32 %state, %ptr
  return %read, %next
}
)",
                {0});
  expect_reader(R"(
low.func.def target<test.low.core> @high_read_low_write(%state: reg<test.i32>, %ptr: reg<test.ptr>) -> (reg<test.i32>, reg<test.i32>) asm {
  %read = test.read.high16.i32 %state
  %next = test.write.low16.tied.i32 %state, %ptr
  return %read, %next
}
)",
                {});
  expect_reader(R"(
low.func.def target<test.low.core> @low_read_low_write(%state: reg<test.i32>, %ptr: reg<test.ptr>) -> (reg<test.i32>, reg<test.i32>) asm {
  %read = test.read.low16.i32 %state
  %next = test.write.low16.tied.i32 %state, %ptr
  return %read, %next
}
)",
                {0});
  expect_reader(R"(
low.func.def target<test.low.core> @full_read_high_write(%state: reg<test.i32>, %rhs: reg<test.i32>, %ptr: reg<test.ptr>) -> (reg<test.i32>, reg<test.i32>) asm {
  %read = test.add.i32 %state, %rhs
  %next = test.write.high16.i32 %state, %ptr
  return %read, %next
}
)",
                {0});
  expect_reader(R"(
low.func.def target<test.low.core> @low_read_full_write(%state: reg<test.i32>) -> (reg<test.i32>, reg<test.i32>) asm {
  %read = test.read.low16.i32 %state
  %next = test.tied.any %state
  return %read, %next
}
)",
                {0});
}

TEST_F(LowEmissionFrameTest, WholeWriteRetainsEarlierDisjointReader) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @partial_then_whole(%state: reg<test.i32>, %ptr: reg<test.ptr>) -> (reg<test.i32>, reg<test.i32>) asm {
  %low_read = test.read.low16.i32 %state
  %high_next = test.write.high16.i32 %state, %ptr
  %full_next = test.tied.any %high_next
  return %low_read, %full_next
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame));

  ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/1, {});
  ExpectStorageLifetimePredecessors(frame, /*consumer_node=*/2, {0, 1});
  ExpectDependencyTiming(frame, 0, 2, LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE, 1,
                         LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_SCHEDULE_CLASS,
                         LOOM_LOW_MODEL_QUALITY_EXACT);
  ExpectDependencyTiming(frame, 1, 2, LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE, 4,
                         LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_SCHEDULE_CLASS,
                         LOOM_LOW_MODEL_QUALITY_FALLBACK);
}

TEST_F(LowEmissionFrameTest, OrderedEffectUsesDirectionalTimingEndpoints) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @directional_effect(%address: reg<test.ptr>, %payload: reg<test.i32 x4>) -> (reg<test.i32 x4>) asm {
  test.store.v4i32 %address, %payload
  test.barrier
  %loaded = test.load.v4i32 %address
  return %loaded
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame,
                            LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY));

  ASSERT_GE(frame.schedule.node_count, 3u);
  EXPECT_EQ(frame.schedule.nodes[0].issue_cycle, 0u);
  EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 2u);
  EXPECT_EQ(frame.schedule.nodes[2].issue_cycle, 4u);

  const loom_low_schedule_dependency_t* store_to_barrier = nullptr;
  const loom_low_schedule_dependency_t* barrier_to_load = nullptr;
  for (iree_host_size_t i = 0; i < frame.schedule.dependencies.count; ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&frame.schedule.dependencies, i);
    if (dependency->kind != LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT ||
        dependency->separation_source !=
            LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR) {
      continue;
    }
    if (dependency->producer_node == 0 && dependency->consumer_node == 1) {
      store_to_barrier = dependency;
    } else if (dependency->producer_node == 1 &&
               dependency->consumer_node == 2) {
      barrier_to_load = dependency;
    }
  }
  ASSERT_NE(store_to_barrier, nullptr);
  ASSERT_NE(barrier_to_load, nullptr);
  EXPECT_EQ(store_to_barrier->minimum_issue_separation_cycles, 2);
  EXPECT_EQ(barrier_to_load->minimum_issue_separation_cycles, 2);
  EXPECT_NE(store_to_barrier->consumer_event_id,
            barrier_to_load->producer_event_id);
}

TEST_F(LowEmissionFrameTest, EffectTimingCrossesDirectCfgEdge) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @direct_effect_edge(%address: reg<test.ptr>, %payload: reg<test.i32 x4>) -> (reg<test.i32 x4>) asm {
  test.store.v4i32 %address, %payload
  low.br ^consume
^consume:
  %loaded = test.load.v4i32 %address
  return %loaded
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame,
                            LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY));

  ASSERT_EQ(frame.schedule.node_count, 4u);
  EXPECT_TRUE(loom_low_br_isa(frame.schedule.nodes[1].op));
  EXPECT_EQ(frame.schedule.nodes[0].issue_cycle, 0u);
  EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 2u);
  EXPECT_EQ(frame.schedule.nodes[2].issue_cycle, 0u);

  const loom_low_schedule_dependency_t* boundary_dependency = nullptr;
  for (iree_host_size_t i = 0; i < frame.schedule.dependencies.count; ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&frame.schedule.dependencies, i);
    if (dependency->producer_node == 0 && dependency->consumer_node == 1 &&
        dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT &&
        dependency->separation_source ==
            LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR) {
      boundary_dependency = dependency;
      break;
    }
  }
  ASSERT_NE(boundary_dependency, nullptr);
  EXPECT_EQ(boundary_dependency->minimum_issue_separation_cycles, 2);
  EXPECT_EQ(boundary_dependency->producer_attachment_kind,
            LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT);
  EXPECT_EQ(boundary_dependency->consumer_attachment_kind,
            LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_NONE);
}

TEST_F(LowEmissionFrameTest, EffectTimingCrossesDiamondAndEmptyBlock) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @diamond_effect_edge(%condition: reg<test.i32>, %address: reg<test.ptr>, %payload: reg<test.i32 x4>) -> (reg<test.i32 x4>) asm {
  test.store.v4i32 %address, %payload
  low.cond_br %condition, ^empty, ^write : reg<test.i32>
^empty:
  low.br ^read
^read:
  %loaded = test.load.v4i32 %address
  return %loaded
^write:
  test.store.v4i32 %address, %payload
  return %payload
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame,
                            LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY));

  ASSERT_EQ(frame.schedule.node_count, 7u);
  EXPECT_TRUE(loom_low_cond_br_isa(frame.schedule.nodes[1].op));
  EXPECT_EQ(frame.schedule.nodes[0].issue_cycle, 0u);
  EXPECT_EQ(frame.schedule.nodes[1].issue_cycle, 2u);

  bool found_read_requirement = false;
  bool found_write_requirement = false;
  for (iree_host_size_t i = 0; i < frame.schedule.dependencies.count; ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&frame.schedule.dependencies, i);
    if (dependency->producer_node != 0 || dependency->consumer_node != 1 ||
        dependency->kind != LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT ||
        dependency->separation_source !=
            LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR) {
      continue;
    }
    found_read_requirement |= dependency->minimum_issue_separation_cycles == 2;
    found_write_requirement |= dependency->minimum_issue_separation_cycles == 1;
  }
  EXPECT_TRUE(found_read_requirement);
  EXPECT_TRUE(found_write_requirement);
}

TEST_F(LowEmissionFrameTest, EffectTimingCrossesCfgBackedge) {
  ModulePtr module = ParseModule(R"(
low.func.def target<test.low.core> @loop_effect_edge(%condition: reg<test.i32>, %address: reg<test.ptr>) -> (reg<test.i32 x4>) asm {
  low.br ^loop
^loop:
  %loaded = test.load.v4i32 %address
  test.barrier
  low.cond_br %condition, ^loop, ^exit : reg<test.i32>
^exit:
  return %loaded
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), {}, &frame,
                            LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY));

  ASSERT_EQ(frame.schedule.node_count, 5u);
  EXPECT_TRUE(loom_low_cond_br_isa(frame.schedule.nodes[3].op));
  EXPECT_EQ(frame.schedule.nodes[3].issue_cycle,
            frame.schedule.nodes[2].issue_cycle + 2u);

  const loom_low_schedule_dependency_t* backedge_dependency = nullptr;
  for (iree_host_size_t i = 0; i < frame.schedule.dependencies.count; ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&frame.schedule.dependencies, i);
    if (dependency->producer_node == 2 && dependency->consumer_node == 3 &&
        dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT &&
        dependency->separation_source ==
            LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR) {
      backedge_dependency = dependency;
      break;
    }
  }
  ASSERT_NE(backedge_dependency, nullptr);
  EXPECT_EQ(backedge_dependency->minimum_issue_separation_cycles, 2);
}

TEST_F(LowEmissionFrameTest, StructuralModelCarriesNativePacketTiming) {
  ModulePtr module = ParseModule();
  const loom_low_schedule_structural_model_t models[] = {
      {
          /*.op_kind=*/LOOM_OP_LOW_STORAGE_ADDRESS,
          /*.result_reg_class_id=*/TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR,
          /*.schedule_descriptor_ordinal=*/
          TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32,
      },
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(
      BuildFrame(module.get(), {models, IREE_ARRAYSIZE(models)}, &frame));

  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_low_descriptor_view_t* schedule_descriptor_view =
      loom_low_descriptor_set_descriptor_view_at(
          descriptor_set, TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32);
  const loom_low_schedule_class_t* expected_schedule_class =
      &descriptor_set
           ->schedule_classes[schedule_descriptor_view->schedule_class_id];
  ASSERT_EQ(expected_schedule_class->latency_cycles, 1u);
  ASSERT_EQ(expected_schedule_class->minimum_issue_separation_cycles, 1);
  ASSERT_EQ(expected_schedule_class->issue_use_count, 1u);

  const loom_low_schedule_node_t* address_node =
      FindNode(frame, LOOM_OP_LOW_STORAGE_ADDRESS);
  const loom_low_schedule_node_t* load_node = FindNode(frame, LOOM_OP_LOW_OP);
  ASSERT_NE(address_node, nullptr);
  ASSERT_NE(load_node, nullptr);
  EXPECT_EQ(address_node->kind, LOOM_LOW_SCHEDULE_NODE_STRUCTURAL);
  EXPECT_EQ(address_node->descriptor, nullptr);
  EXPECT_EQ(address_node->source_descriptor_ordinal,
            LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  EXPECT_EQ(address_node->schedule_class_id,
            schedule_descriptor_view->schedule_class_id);
  EXPECT_EQ(address_node->schedule_class, expected_schedule_class);
  EXPECT_GE(load_node->issue_cycle,
            address_node->issue_cycle +
                expected_schedule_class->minimum_issue_separation_cycles);

  const loom_low_schedule_model_summary_t* model_summary = nullptr;
  for (iree_host_size_t i = 0; i < frame.schedule.model_summary_count; ++i) {
    if (frame.schedule.model_summaries[i].schedule_class_id ==
        schedule_descriptor_view->schedule_class_id) {
      model_summary = &frame.schedule.model_summaries[i];
      break;
    }
  }
  ASSERT_NE(model_summary, nullptr);
  EXPECT_EQ(model_summary->use_count, 1u);

  const loom_low_issue_use_t* issue_use =
      &descriptor_set->issue_uses[expected_schedule_class->issue_use_start];
  const loom_low_schedule_resource_summary_t* resource_summary = nullptr;
  for (iree_host_size_t i = 0; i < frame.schedule.resource_summary_count; ++i) {
    if (frame.schedule.resource_summaries[i].resource_id ==
        issue_use->resource_id) {
      resource_summary = &frame.schedule.resource_summaries[i];
      break;
    }
  }
  ASSERT_NE(resource_summary, nullptr);
  EXPECT_EQ(resource_summary->use_count, 1u);
}

TEST_F(LowEmissionFrameTest, RejectsInvalidStructuralModelDescriptor) {
  ModulePtr module = ParseModule();
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_low_schedule_structural_model_t models[] = {
      {
          /*.op_kind=*/LOOM_OP_LOW_STORAGE_ADDRESS,
          /*.result_reg_class_id=*/TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR,
          /*.schedule_descriptor_ordinal=*/descriptor_set->descriptor_count,
      },
  };
  loom_low_emission_frame_t frame = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      BuildFrame(module.get(), {models, IREE_ARRAYSIZE(models)}, &frame));
}

TEST_F(LowEmissionFrameTest, RejectsOverlappingStructuralModels) {
  ModulePtr module = ParseModule();
  const loom_low_schedule_structural_model_t models[] = {
      {
          /*.op_kind=*/LOOM_OP_LOW_STORAGE_ADDRESS,
          /*.result_reg_class_id=*/LOOM_LOW_REG_CLASS_NONE,
          /*.schedule_descriptor_ordinal=*/
          TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32,
      },
      {
          /*.op_kind=*/LOOM_OP_LOW_STORAGE_ADDRESS,
          /*.result_reg_class_id=*/TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR,
          /*.schedule_descriptor_ordinal=*/
          TEST_LOW_CORE_DESCRIPTOR_REF_TEST_CONST_I32,
      },
  };
  loom_low_emission_frame_t frame = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      BuildFrame(module.get(), {models, IREE_ARRAYSIZE(models)}, &frame));
}

}  // namespace
}  // namespace loom
