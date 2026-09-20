// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/sysv_frame.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/descriptors/low_registry.h"
#include "loom/target/arch/x86/ops/registry.h"
#include "loom/testing/context.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class X86SysvFrameTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_x86_ops_register_dialect(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_x86_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
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
                                  IREE_SV("sysv_frame_test.loom"), &context_,
                                  &pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_op_t* FindDefinition(loom_module_t* module) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_module_block(module), op) {
      if (loom_low_func_def_isa(op)) {
        return op;
      }
    }
    return nullptr;
  }

  iree_status_t BuildFrame(
      loom_module_t* module,
      const loom_low_allocation_fixed_value_t* fixed_values,
      iree_host_size_t fixed_value_count,
      loom_low_emission_frame_t* out_frame) {
    const loom_low_allocation_reserved_range_t reserved_range =
        loom_x86_sysv_frame_stack_pointer_reservation();
    loom_low_emission_frame_options_t options = {};
    options.descriptor_registry = &registry_.registry;
    options.schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY;
    options.allocation_fixed_values = fixed_values;
    options.allocation_fixed_value_count = fixed_value_count;
    options.allocation_reserved_ranges = &reserved_range;
    options.allocation_reserved_range_count = 1;
    return loom_low_emission_frame_build(module, FindDefinition(module),
                                         &options, &arena_, out_frame);
  }

  bool RangeContainsRegisterToSlot(const loom_x86_sysv_frame_plan_t& plan,
                                   uint32_t move_start, uint32_t move_count,
                                   uint32_t physical_register,
                                   loom_x86_sysv_frame_slot_kind_t slot_kind) {
    for (uint32_t i = 0; i < move_count; ++i) {
      const loom_low_move_t& move = plan.moves[move_start + i];
      if (move.source.location_kind !=
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
          move.source.location != physical_register ||
          move.destination.location_kind !=
              LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT) {
        continue;
      }
      if (plan.slots[move.destination.location].kind == slot_kind) {
        return true;
      }
    }
    return false;
  }

  bool RangeContainsSlotToRegister(const loom_x86_sysv_frame_plan_t& plan,
                                   uint32_t move_start, uint32_t move_count,
                                   loom_x86_sysv_frame_slot_kind_t slot_kind,
                                   uint32_t physical_register) {
    for (uint32_t i = 0; i < move_count; ++i) {
      const loom_low_move_t& move = plan.moves[move_start + i];
      if (move.source.location_kind !=
              LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT ||
          move.destination.location_kind !=
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
          move.destination.location != physical_register) {
        continue;
      }
      if (plan.slots[move.source.location].kind == slot_kind) {
        return true;
      }
    }
    return false;
  }

  iree_arena_block_pool_t pool_ = {};
  iree_arena_allocator_t arena_ = {};
  iree_arena_allocator_t scratch_arena_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(X86SysvFrameTest, PlansStackArgumentsCyclesAndSelectivePreservation) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.decl import(native, "external") target<x86.scalar.core>(@target) abi_layout({argument_locations = [7, 6, 2, 1, 8, 9, -1, -9], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 16}) @external(%a0: reg<x86.gpr64>, %a1: reg<x86.gpr64>, %a2: reg<x86.gpr64>, %a3: reg<x86.gpr64>, %a4: reg<x86.gpr64>, %a5: reg<x86.gpr64>, %a6: reg<x86.gpr64>, %a7: reg<x86.gpr64>) -> (reg<x86.gpr64>)

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7, 6, 2, 1, 8, 9, -1, -9], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 16}) @caller(%a0: reg<x86.gpr64>, %a1: reg<x86.gpr64>, %a2: reg<x86.gpr64>, %a3: reg<x86.gpr64>, %a4: reg<x86.gpr64>, %a5: reg<x86.gpr64>, %a6: reg<x86.gpr64>, %a7: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %volatile = copy %a2 : reg<x86.gpr64> -> reg<x86.gpr64>
  %preserved = copy %a3 : reg<x86.gpr64> -> reg<x86.gpr64>
  %incoming6 = low.func.stack_arg [6, 0] : reg<x86.gpr64>
  %outgoing6 = low.func.call_arg @external[6, 0](%incoming6) : reg<x86.gpr64> -> low.storage<stack>
  %incoming7 = low.func.stack_arg [7, 8] : reg<x86.gpr64>
  %outgoing7 = low.func.call_arg @external[7, 8](%incoming7) : reg<x86.gpr64> -> low.storage<stack>
  %called = low.func.call @external(%a0, %a1, %a2, %a3, %a4, %a5) : (reg<x86.gpr64>, reg<x86.gpr64>, reg<x86.gpr64>, reg<x86.gpr64>, reg<x86.gpr64>, reg<x86.gpr64>) [%outgoing6: low.storage<stack>, %outgoing7: low.storage<stack>] -> (reg<x86.gpr64>)
  %partial = add.gpr64 %called, %volatile
  %result = add.gpr64 %partial, %preserved
  return %result
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_value_id_t volatile_value =
      loom_op_const_results(loom_block_op(body, 0))[0];
  const loom_value_id_t preserved_value =
      loom_op_const_results(loom_block_op(body, 1))[0];
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RSI,
       1},
      {loom_block_arg_id(body, 1),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDI,
       1},
      {volatile_value, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_R10, 1},
      {preserved_value, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RBX, 1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);
  ASSERT_EQ(frame.allocation.spill_count, 0u);
  ASSERT_EQ(frame.schedule.return_node_count, 1u);
  EXPECT_TRUE(loom_low_return_isa(
      frame.schedule.nodes[frame.schedule.return_node_indices[0]].op));

  loom_x86_sysv_frame_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_x86_sysv_frame_plan_build(&frame, &arena_, &scratch_arena_, &plan));
  ASSERT_EQ(plan.call_count, 1u);
  ASSERT_EQ(plan.return_count, 1u);
  EXPECT_EQ(plan.frame_size % 16, 8u);
  EXPECT_GE(plan.frame_size, 40u);
  EXPECT_EQ(plan.abi_layout.stack_argument_bytes, 16u);

  bool has_move_scratch = false;
  for (uint32_t i = 0; i < plan.slot_count; ++i) {
    EXPECT_GE(plan.slots[i].byte_offset, 16u);
    has_move_scratch |=
        plan.slots[i].kind == LOOM_X86_SYSV_FRAME_SLOT_MOVE_SCRATCH;
  }
  EXPECT_TRUE(has_move_scratch);
  EXPECT_TRUE(RangeContainsRegisterToSlot(
      plan, plan.calls[0].move_start, plan.calls[0].save_move_count,
      LOOM_X86_SYSV_GPR_R10, LOOM_X86_SYSV_FRAME_SLOT_CALLER_SAVE));
  EXPECT_TRUE(RangeContainsRegisterToSlot(
      plan, plan.callee_save_moves.start, plan.callee_save_moves.count,
      LOOM_X86_SYSV_GPR_RBX, LOOM_X86_SYSV_FRAME_SLOT_CALLEE_SAVE));
  EXPECT_GE(plan.calls[0].argument_move_count, 5u);
  EXPECT_TRUE(RangeContainsSlotToRegister(
      plan,
      plan.calls[0].move_start + plan.calls[0].save_move_count +
          plan.calls[0].argument_move_count,
      plan.calls[0].result_and_restore_move_count,
      LOOM_X86_SYSV_FRAME_SLOT_CALLER_SAVE, LOOM_X86_SYSV_GPR_R10));
}

TEST_F(X86SysvFrameTest, SparseCfgAndBoundaryValuesNeedNoCallerSave) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.decl import(native, "external") target<x86.scalar.core>(@target) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @external(%value: reg<x86.gpr64>) -> (reg<x86.gpr64>)

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7, 6, 2], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @cfg(%condition: reg<x86.gpr32>, %call_value: reg<x86.gpr64>, %other_path_value: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  low.cond_br %condition, ^call_path, ^other_path : reg<x86.gpr32>
^call_path:
  %called = low.func.call @external(%call_value) : (reg<x86.gpr64>) -> (reg<x86.gpr64>)
  return %called
^other_path:
  return %other_path_value
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_region_t* body = loom_low_func_def_body(function);
  loom_block_t* entry = loom_region_entry_block(body);
  loom_block_t* call_path = loom_region_block(body, 1);
  const loom_value_id_t called_value =
      loom_op_const_results(loom_block_op(call_path, 0))[0];
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(entry, 1),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_R10,
       1},
      {loom_block_arg_id(entry, 2),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_R11,
       1},
      {called_value, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_R11, 1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);

  loom_x86_sysv_frame_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_x86_sysv_frame_plan_build(&frame, &arena_, &scratch_arena_, &plan));
  ASSERT_EQ(plan.call_count, 1u);
  ASSERT_EQ(plan.return_count, 2u);
  EXPECT_EQ(plan.calls[0].save_move_count, 0u);
  EXPECT_EQ(plan.calls[0].result_and_restore_move_count, 1u);
}

TEST_F(X86SysvFrameTest, LeafWithNoStorageHasNoFrameTables) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [], calling_convention = "sysv_x86_64", result_locations = [], stack_argument_bytes = 0}) @leaf() asm {
  return
}
)");
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), nullptr, 0, &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);

  loom_x86_sysv_frame_plan_t plan = {};
  IREE_ASSERT_OK(
      loom_x86_sysv_frame_plan_build(&frame, &arena_, &scratch_arena_, &plan));
  EXPECT_EQ(plan.frame_size, 0u);
  EXPECT_EQ(plan.slot_count, 0u);
  EXPECT_EQ(plan.move_count, 0u);
  EXPECT_EQ(plan.call_count, 0u);
  ASSERT_EQ(plan.return_count, 1u);
  EXPECT_EQ(plan.returns[0].result_moves.count, 0u);
  EXPECT_EQ(plan.callee_save_moves.count, 0u);
  EXPECT_EQ(plan.entry_moves.count, 0u);
  EXPECT_EQ(plan.callee_restore_moves.count, 0u);
}

}  // namespace
}  // namespace loom
