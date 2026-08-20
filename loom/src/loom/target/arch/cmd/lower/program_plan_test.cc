// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/descriptors/descriptors.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/target/arch/cmd/program.h"
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
    const loom_low_descriptor_set_provider_t descriptor_set_providers[] = {
        loom_cmd_core_descriptor_set,
    };
    const loom_low_descriptor_registry_t descriptor_registry = {
        /*.descriptor_sets=*/{},
        /*.descriptor_set_count=*/{},
        /*.descriptor_set_providers=*/descriptor_set_providers,
        /*.descriptor_set_provider_count=*/
        IREE_ARRAYSIZE(descriptor_set_providers),
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &descriptor_registry, &parse_options.low_asm_environment);
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

command.program.def public @increment_twice(%element_count: index) launch(%source: buffer, %scratch: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @increment[%element_count](%source, %scratch) : [index](buffer, buffer)
  kernel.launch @increment[%element_count](%scratch, %target) : [index](buffer, buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_op_t* source_programs[] = {
      FindSymbol(source_module.get(), IREE_SV("increment_twice")),
      FindSymbol(source_module.get(), IREE_SV("increment_then_double")),
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
  const loom_cmd_program_root_t& twice = plan.roots[0];
  const loom_cmd_program_root_t& mixed = plan.roots[1];
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("increment_twice")),
            twice.function_op);
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("increment_then_double")),
            mixed.function_op);
  EXPECT_TRUE(loom_low_func_def_isa(twice.function_op));
  EXPECT_TRUE(loom_low_func_def_isa(mixed.function_op));
  EXPECT_EQ(FindSymbol(plan.launch_module, IREE_SV("increment_twice")),
            twice.launch_function_op);
  EXPECT_EQ(FindSymbol(plan.launch_module, IREE_SV("increment_then_double")),
            mixed.launch_function_op);
  EXPECT_EQ(twice.launch_tuple_count, 1u);
  EXPECT_EQ(mixed.launch_tuple_count, 1u);

  ASSERT_EQ(plan.dependency_count, 2u);
  ASSERT_NE(plan.dependency_units, nullptr);
  ASSERT_NE(plan.dependency_units[0].module, nullptr);
  ASSERT_NE(plan.dependency_units[1].module, nullptr);
  EXPECT_NE(plan.dependency_units[0].module, plan.dependency_units[1].module);
  ASSERT_EQ(twice.dependency_count, 1u);
  ASSERT_NE(twice.dependency_unit_indices, nullptr);
  EXPECT_EQ(twice.dependency_unit_indices[0], 0u);
  ASSERT_EQ(mixed.dependency_count, 2u);
  ASSERT_NE(mixed.dependency_unit_indices, nullptr);
  EXPECT_EQ(mixed.dependency_unit_indices[0], 0u);
  EXPECT_EQ(mixed.dependency_unit_indices[1], 1u);

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

TEST_F(CmdProgramPlanTest, SerializesMovedReferenceValues) {
  ModulePtr module = ParseAndVerify(R"(
low.func.def target<cmd.core> abi(command_program) @moved_binding() {
  %binding = low.resource<command_input> {index = 0, source_type = buffer} : reg<cmd.binding>
  %moved = low.move %binding : reg<cmd.binding> -> reg<cmd.binding>
  %zero_u32 = low.const<cmd.constant.u32> {value = 0} : reg<cmd.u32>
  %one_u32 = low.const<cmd.constant.u32> {value = 1} : reg<cmd.u32>
  %zero_u64 = low.const<cmd.constant.u64> {value = 0} : reg<cmd.u64>
  %length = low.const<cmd.constant.u64> {value = 16} : reg<cmd.u64>
  %target_ref = low.op<cmd.buffer.ref.binding>(%moved, %zero_u64, %length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  low.op<cmd.fill>(%target_ref, %zero_u32, %one_u32) : (reg<cmd.buffer_ref>, reg<cmd.u32>, reg<cmd.u32>)
  low.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_cmd_program_root_t root = {};
  root.function_op = FindSymbol(module.get(), IREE_SV("moved_binding"));
  root.abi_layout.rebindable_binding_count = 1;
  root.transient.binding_index = UINT32_MAX;
  loom_cmd_program_plan_t plan = {};
  plan.root_module = module.get();
  plan.roots = &root;
  plan.root_count = 1;

  iree_byte_span_t data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_plan_serialize_root(&plan, 0, &data,
                                                      iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(data.data, data.data_length), &program));
  EXPECT_EQ(program.requirements.rebindable_binding_count, 1u);
  EXPECT_EQ(program.commands.count, 1u);
  iree_allocator_free(iree_allocator_system(), data.data);
}

}  // namespace
}  // namespace loom
