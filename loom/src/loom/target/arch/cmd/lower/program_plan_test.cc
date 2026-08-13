// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/pass/builtin_registry.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticCapture;
using ModulePtr = ::loom::testing::ModulePtr;

class CmdProgramPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseAndVerify(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    DiagnosticCapture capture;
    parse_options.diagnostic_sink = capture.sink();
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("cmd_program_plan_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    ModulePtr module_ptr(module);
    if (!module) {
      for (const auto& diagnostic : capture.diagnostics) {
        ADD_FAILURE() << diagnostic.error->summary << " at line "
                      << diagnostic.origin_line << ", column "
                      << diagnostic.origin_column;
      }
      return module_ptr;
    }

    loom_verify_options_t verify_options = {};
    verify_options.max_errors = 20;
    verify_options.sink = capture.sink();
    loom_verify_result_t result = {};
    IREE_CHECK_OK(loom_verify_module(module, &verify_options, &result));
    for (const auto& diagnostic : capture.diagnostics) {
      ADD_FAILURE() << diagnostic.error->summary << " at line "
                    << diagnostic.origin_line << ", column "
                    << diagnostic.origin_column;
    }
    EXPECT_EQ(result.error_count, 0u);
    return module_ptr;
  }

  loom_op_t* FindSymbol(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return module->symbols.entries[symbol_id].defining_op;
  }

  // Shared arena block pool backing source and prepared modules.
  iree_arena_block_pool_t block_pool_;

  // Source dialect context used by parsing, linking, and lowering.
  loom_context_t context_;
};

TEST_F(CmdProgramPlanTest, OwnsMultipleRootsAndDependencyTables) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.def @increment(%element_count: index) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %c1 = scalar.constant 1 : i32
  %source_view = buffer.view %source[%base] : buffer -> view<128xi32>
  %target_view = buffer.view %target[%base] : buffer -> view<128xi32>
  %value = view.load %source_view[%workgroup] : view<128xi32> -> i32
  %result = scalar.addi %value, %c1 : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32>
  kernel.return
}

kernel.def @double(%element_count: index) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %c2 = scalar.constant 2 : i32
  %source_view = buffer.view %source[%base] : buffer -> view<128xi32>
  %target_view = buffer.view %target[%base] : buffer -> view<128xi32>
  %value = view.load %source_view[%workgroup] : view<128xi32> -> i32
  %result = scalar.muli %value, %c2 : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32>
  kernel.return
}

command.program.def public @increment_then_double(%element_count: index) launch(%source: buffer, %scratch: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @increment[%element_count](%source, %scratch) : [index](buffer, buffer)
  kernel.launch @double[%element_count](%scratch, %target) : [index](buffer, buffer)
  command.return
}

command.program.def public @double_then_increment(%element_count: index) launch(%source: buffer, %scratch: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @double[%element_count](%source, %scratch) : [index](buffer, buffer)
  kernel.launch @increment[%element_count](%scratch, %target) : [index](buffer, buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_op_t* source_programs[] = {
      FindSymbol(source_module.get(), IREE_SV("increment_then_double")),
      FindSymbol(source_module.get(), IREE_SV("double_then_increment")),
  };

  loom_cmd_program_plan_t plan = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      source_module.get(), source_programs, IREE_ARRAYSIZE(source_programs),
      loom_pass_builtin_registry(),
      /*diagnostic_emitter=*/{}, &block_pool_, &valid, &plan,
      iree_allocator_system()));
  ASSERT_TRUE(valid);
  source_module.reset();

  ASSERT_EQ(plan.root_count, 2u);
  ASSERT_NE(plan.roots, nullptr);
  ASSERT_NE(plan.root_module, nullptr);
  ASSERT_NE(plan.launch_module, nullptr);
  const loom_cmd_program_root_t& forward = plan.roots[0];
  const loom_cmd_program_root_t& reverse = plan.roots[1];
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("increment_then_double")),
            forward.function_op);
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("double_then_increment")),
            reverse.function_op);
  EXPECT_TRUE(loom_low_func_def_isa(forward.function_op));
  EXPECT_TRUE(loom_low_func_def_isa(reverse.function_op));
  EXPECT_EQ(FindSymbol(plan.launch_module,
                       IREE_SV("increment_then_double.__launch_counts")),
            forward.launch_function_op);
  EXPECT_EQ(FindSymbol(plan.launch_module,
                       IREE_SV("double_then_increment.__launch_counts")),
            reverse.launch_function_op);
  EXPECT_EQ(forward.launch_tuple_count, 1u);
  EXPECT_EQ(reverse.launch_tuple_count, 1u);
  const loom_cmd_program_requirements_t forward_requirements =
      loom_cmd_program_root_requirements(&forward);
  EXPECT_EQ(forward_requirements.fixed_buffer_count, 0u);
  EXPECT_EQ(forward_requirements.rebindable_binding_count, 4u);
  EXPECT_EQ(forward_requirements.executable_count, 1u);
  EXPECT_EQ(forward_requirements.entry_count, 2u);
  EXPECT_EQ(forward_requirements.transient.binding_index, UINT32_MAX);
  EXPECT_EQ(forward_requirements.launch_counts.binding_index, 3u);
  EXPECT_EQ(forward_requirements.launch_counts.required_byte_length,
            LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_BYTE_LENGTH);
  EXPECT_EQ(forward_requirements.launch_counts.minimum_alignment,
            LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_ALIGNMENT);

  ASSERT_EQ(plan.dependency_count, 1u);
  ASSERT_NE(plan.dependency_units, nullptr);
  ASSERT_NE(plan.dependency_units[0].module, nullptr);
  ASSERT_EQ(plan.dependency_units[0].export_count, 2u);
  ASSERT_NE(plan.dependency_units[0].kernel_ops, nullptr);
  EXPECT_NE(plan.dependency_units[0].kernel_ops[0],
            plan.dependency_units[0].kernel_ops[1]);

  ASSERT_EQ(forward.executable_count, 1u);
  ASSERT_NE(forward.executable_unit_indices, nullptr);
  EXPECT_EQ(forward.executable_unit_indices[0], 0u);
  ASSERT_EQ(forward.entry_count, 2u);
  ASSERT_NE(forward.entries, nullptr);
  EXPECT_EQ(forward.entries[0].executable_index, 0u);
  EXPECT_EQ(forward.entries[0].unit_export_index, 0u);
  EXPECT_EQ(forward.entries[1].executable_index, 0u);
  EXPECT_EQ(forward.entries[1].unit_export_index, 1u);

  ASSERT_EQ(reverse.executable_count, 1u);
  ASSERT_NE(reverse.executable_unit_indices, nullptr);
  EXPECT_EQ(reverse.executable_unit_indices[0], 0u);
  ASSERT_EQ(reverse.entry_count, 2u);
  ASSERT_NE(reverse.entries, nullptr);
  EXPECT_EQ(reverse.entries[0].executable_index, 0u);
  EXPECT_EQ(reverse.entries[0].unit_export_index, 1u);
  EXPECT_EQ(reverse.entries[1].executable_index, 0u);
  EXPECT_EQ(reverse.entries[1].unit_export_index, 0u);

  loom_cmd_program_plan_deinitialize(&plan);
  EXPECT_EQ(plan.root_module, nullptr);
  EXPECT_EQ(plan.launch_module, nullptr);
  EXPECT_EQ(plan.roots, nullptr);
  EXPECT_EQ(plan.dependency_units, nullptr);
}

TEST_F(CmdProgramPlanTest, OwnsParameterRequirementTables) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.def @combine() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%lhs: view<3xi32>, %rhs: view<4xi32>, %target: buffer) {
  %element = index.constant 0 : index
  %base = index.constant 0 : offset
  %lhs_value = view.load %lhs[%element] : view<3xi32> -> i32
  %rhs_value = view.load %rhs[%element] : view<4xi32> -> i32
  %sum = scalar.addi %lhs_value, %rhs_value : i32
  %target_view = buffer.view %target[%base] : buffer -> view<1xi32>
  view.store %sum, %target_view[%element] : i32, view<1xi32>
  kernel.return
}

command.program.def public @parameterized() launch(%parameters: buffer, %target: buffer) {
  %layer = index.constant 3 : index
  %lhs = command.parameter %parameters, "blk.{}.lhs"[%layer] : view<3xi32>
  %rhs = command.parameter %parameters, "shared.rhs" : view<4xi32>
  kernel.launch @combine(%lhs, %rhs, %target) : (view<3xi32>, view<4xi32>, buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_op_t* source_programs[] = {
      FindSymbol(source_module.get(), IREE_SV("parameterized")),
  };

  loom_cmd_program_plan_t plan = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      source_module.get(), source_programs, IREE_ARRAYSIZE(source_programs),
      loom_pass_builtin_registry(),
      /*diagnostic_emitter=*/{}, &block_pool_, &valid, &plan,
      iree_allocator_system()));
  ASSERT_TRUE(valid);
  source_module.reset();

  ASSERT_EQ(plan.root_count, 1u);
  const loom_cmd_program_root_t& root = plan.roots[0];
  ASSERT_EQ(root.parameters.root_count, 1u);
  ASSERT_NE(root.parameters.roots, nullptr);
  EXPECT_EQ(root.parameters.roots[0].source_binding_ordinal, 0u);
  EXPECT_EQ(root.parameters.roots[0].fixed_buffer_index, 0u);
  EXPECT_EQ(root.parameters.roots[0].required_byte_length, 272u);
  EXPECT_EQ(root.parameters.roots[0].minimum_alignment, 256u);

  ASSERT_EQ(root.parameters.count, 2u);
  ASSERT_NE(root.parameters.entries, nullptr);
  const loom_cmd_parameter_requirement_t& lhs = root.parameters.entries[0];
  EXPECT_TRUE(iree_string_view_equal(lhs.key, IREE_SV("blk.3.lhs")));
  EXPECT_EQ(lhs.source_binding_ordinal, 0u);
  EXPECT_EQ(lhs.fixed_buffer_index, 0u);
  EXPECT_EQ(lhs.byte_offset, 0u);
  EXPECT_EQ(lhs.byte_length, 12u);
  EXPECT_EQ(lhs.minimum_alignment, 256u);
  const loom_cmd_parameter_requirement_t& rhs = root.parameters.entries[1];
  EXPECT_TRUE(iree_string_view_equal(rhs.key, IREE_SV("shared.rhs")));
  EXPECT_EQ(rhs.source_binding_ordinal, 0u);
  EXPECT_EQ(rhs.fixed_buffer_index, 0u);
  EXPECT_EQ(rhs.byte_offset, 256u);
  EXPECT_EQ(rhs.byte_length, 16u);
  EXPECT_EQ(rhs.minimum_alignment, 256u);
  EXPECT_EQ(root.transient.binding_index, UINT32_MAX);
  EXPECT_EQ(root.transient.required_byte_length, 0u);
  EXPECT_EQ(root.transient.minimum_alignment, 0u);

  loom_cmd_program_plan_deinitialize(&plan);
}

}  // namespace
}  // namespace loom
