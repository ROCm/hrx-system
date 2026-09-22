// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/frame.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/schedule/diagnostics.h"
#include "loom/codegen/low/schedule/physical_issue.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/low_registry.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class LowEmissionFrameTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_text_parse_options_t options = {};
    options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("frame_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  ModulePtr ParseModule() {
    return ParseModule(R"(
low.func.def target<test.low.core> @structural_model() -> (reg<test.i32 x4>) asm {
  %storage = storage {byte_alignment = 16, byte_length = 64} : low.storage<workgroup>
  %address = storage_address %storage : low.storage<workgroup> -> reg<test.ptr>
  %value = test.load.v4i32 %address
  return %value
}
)");
  }

  iree_status_t BuildFrame(
      loom_module_t* module,
      loom_low_schedule_structural_model_list_t structural_models,
      loom_low_emission_frame_t* out_frame,
      loom_low_schedule_strategy_t schedule_strategy =
          LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    loom_block_t* module_block = loom_module_block(module);
    IREE_ASSERT_EQ(module_block->op_count, 1);
    loom_low_emission_frame_options_t options = {};
    options.descriptor_registry = &registry_.registry;
    options.schedule_structural_models = structural_models;
    options.schedule_strategy = schedule_strategy;
    return loom_low_emission_frame_build(module, loom_block_op(module_block, 0),
                                         &options, &arena_, out_frame);
  }

  const loom_low_schedule_node_t* FindNode(
      const loom_low_emission_frame_t& frame, loom_op_kind_t op_kind) {
    for (iree_host_size_t i = 0; i < frame.schedule.node_count; ++i) {
      if (frame.schedule.nodes[i].op->kind == op_kind) {
        return &frame.schedule.nodes[i];
      }
    }
    return nullptr;
  }

  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t registry_ = {};
  iree_arena_allocator_t arena_ = {};
};

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
  IREE_ASSERT_OK(loom_low_emission_frame_build(module.get(), function, &options,
                                               &arena_, &frame));
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
    uint32_t cycle = 0;
    const uint32_t proposed_cycle =
        iree_max(iree_max(next_cycle, packet.node->issue_cycle),
                 loom_low_physical_issue_source_ready_cycle(&issue, i));
    IREE_ASSERT_OK(loom_low_physical_issue_place(&issue, &instruction, 1,
                                                 proposed_cycle, &cycle));
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
    IREE_ASSERT_OK(loom_low_emission_frame_build(
        module.get(), loom_block_op(loom_module_block(module.get()), 0),
        &options, &arena_, &frame));
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
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame));
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
  IREE_ASSERT_OK(loom_low_emission_frame_build(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &arena_, &frame));
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
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame));
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
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0),
      &quiet_options, &spill_free_options, &arena_, &quiet_frame));
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
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame));
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
  IREE_ASSERT_OK(loom_low_emission_frame_build_spill_free(
      module.get(), loom_block_op(loom_module_block(module.get()), 0), &options,
      &spill_free_options, &arena_, &frame));
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
