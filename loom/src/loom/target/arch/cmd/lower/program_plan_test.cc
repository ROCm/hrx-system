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

  loom_symbol_ref_t FindSymbolRef(loom_module_t* module,
                                  iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return (loom_symbol_ref_t){/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  // Shared arena block pool backing source and prepared modules.
  iree_arena_block_pool_t block_pool_;

  // Source dialect context used by parsing, linking, and lowering.
  loom_context_t context_;
};

TEST_F(CmdProgramPlanTest, OwnsMultipleRootsAndDeduplicatesEntries) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.entry.decl @increment(%source: buffer, %target: buffer)
kernel.entry.decl @double(%source: buffer, %target: buffer)

command.program.def public @increment_then_double() launch(%source: buffer, %scratch: buffer, %target: buffer) {
  %count = index.constant 1 : index
  kernel.dispatch @increment[%count](%source, %scratch) : [index](buffer, buffer)
  kernel.dispatch @double[%count](%scratch, %target) : [index](buffer, buffer)
  command.return
}

command.program.def public @increment_twice() launch(%source: buffer, %scratch: buffer, %target: buffer) {
  %count = index.constant 1 : index
  kernel.dispatch @increment[%count](%source, %scratch) : [index](buffer, buffer)
  kernel.dispatch @increment[%count](%scratch, %target) : [index](buffer, buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_symbol_ref_t program_refs[] = {
      FindSymbolRef(source_module.get(), IREE_SV("increment_twice")),
      FindSymbolRef(source_module.get(), IREE_SV("increment_then_double")),
  };
  loom_link_plan_materialization_t materialization = {};
  materialization.module = source_module.release();

  loom_cmd_program_plan_t plan = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      &materialization, program_refs, IREE_ARRAYSIZE(program_refs),
      loom_pass_builtin_registry(),
      /*diagnostic_emitter=*/{}, &block_pool_, &valid, &plan,
      iree_allocator_system()));
  ASSERT_TRUE(valid);
  EXPECT_EQ(materialization.module, nullptr);

  ASSERT_EQ(plan.root_count, 2u);
  ASSERT_NE(plan.roots, nullptr);
  ASSERT_NE(plan.root_module, nullptr);
  const loom_cmd_program_root_t& twice = plan.roots[0];
  const loom_cmd_program_root_t& mixed = plan.roots[1];
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("increment_twice")),
            twice.function_op);
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("increment_then_double")),
            mixed.function_op);
  EXPECT_TRUE(loom_low_func_def_isa(twice.function_op));
  EXPECT_TRUE(loom_low_func_def_isa(mixed.function_op));
  EXPECT_EQ(twice.launch_counts.binding_index, UINT32_MAX);
  EXPECT_EQ(mixed.launch_counts.binding_index, UINT32_MAX);

  ASSERT_EQ(plan.entry_requirement_count, 2u);
  ASSERT_NE(plan.entry_requirements, nullptr);
  EXPECT_EQ(plan.entry_requirements[0].declaration_op,
            FindSymbol(plan.root_module, IREE_SV("increment")));
  EXPECT_EQ(plan.entry_requirements[1].declaration_op,
            FindSymbol(plan.root_module, IREE_SV("double")));
  ASSERT_EQ(twice.entry_requirement_count, 1u);
  ASSERT_NE(twice.entry_requirement_indices, nullptr);
  EXPECT_EQ(twice.entry_requirement_indices[0], 0u);
  ASSERT_EQ(mixed.entry_requirement_count, 2u);
  ASSERT_NE(mixed.entry_requirement_indices, nullptr);
  EXPECT_EQ(mixed.entry_requirement_indices[0], 0u);
  EXPECT_EQ(mixed.entry_requirement_indices[1], 1u);

  loom_cmd_program_plan_deinitialize(&plan);
  EXPECT_EQ(plan.root_module, nullptr);
  EXPECT_EQ(plan.roots, nullptr);
  EXPECT_EQ(plan.entry_requirements, nullptr);
}

TEST_F(CmdProgramPlanTest, OwnsParameterRequirementTables) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.entry.decl @combine(%lhs: view<3xi32>, %rhs: view<4xi32>, %target: buffer)

command.program.def public @parameterized() launch(%parameters: buffer, %target: buffer) {
  %layer = index.constant 3 : index
  %count = index.constant 1 : index
  %lhs = command.parameter %parameters, "blk.{}.lhs"[%layer] : view<3xi32>
  %rhs = command.parameter %parameters, "shared.rhs" : view<4xi32>
  kernel.dispatch @combine[%count](%lhs, %rhs, %target) : [index](view<3xi32>, view<4xi32>, buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_symbol_ref_t program_refs[] = {
      FindSymbolRef(source_module.get(), IREE_SV("parameterized")),
  };
  loom_link_plan_materialization_t materialization = {};
  materialization.module = source_module.release();

  loom_cmd_program_plan_t plan = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      &materialization, program_refs, IREE_ARRAYSIZE(program_refs),
      loom_pass_builtin_registry(),
      /*diagnostic_emitter=*/{}, &block_pool_, &valid, &plan,
      iree_allocator_system()));
  ASSERT_TRUE(valid);
  EXPECT_EQ(materialization.module, nullptr);

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

TEST_F(CmdProgramPlanTest, OwnsBodylessEntryRequirementsAndArtifact) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.entry.decl @configured_a(%scale: i8, %output: buffer)
kernel.entry.decl @configured_b(%output: buffer)

command.program.def public @bodyless() launch(%output: buffer) {
  %count_x = index.constant 7 : index
  %count_y = index.constant 5 : index
  %scale = scalar.constant -1 : i8
  command.concurrent {
    kernel.dispatch @configured_a[%count_x](%scale, %output) : [index](i8, buffer)
    kernel.dispatch @configured_a[%count_x](%scale, %output) : [index](i8, buffer)
  }
  kernel.dispatch @configured_b[%count_x, %count_y](%output) : [index, index](buffer)
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_symbol_ref_t program_refs[] = {
      FindSymbolRef(source_module.get(), IREE_SV("bodyless")),
  };
  loom_link_plan_materialization_t materialization = {};
  materialization.module = source_module.release();

  loom_cmd_program_plan_t plan = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      &materialization, program_refs, IREE_ARRAYSIZE(program_refs),
      loom_pass_builtin_registry(),
      /*diagnostic_emitter=*/{}, &block_pool_, &valid, &plan,
      iree_allocator_system()));
  ASSERT_TRUE(valid);
  EXPECT_EQ(materialization.module, nullptr);

  ASSERT_EQ(plan.root_count, 1u);
  ASSERT_NE(plan.root_module, nullptr);
  ASSERT_EQ(plan.entry_requirement_count, 2u);
  ASSERT_NE(plan.entry_requirements, nullptr);
  EXPECT_EQ(plan.entry_requirements[0].declaration_op,
            FindSymbol(plan.root_module, IREE_SV("configured_a")));
  EXPECT_EQ(plan.entry_requirements[1].declaration_op,
            FindSymbol(plan.root_module, IREE_SV("configured_b")));

  const loom_cmd_program_root_t& root = plan.roots[0];
  ASSERT_EQ(root.entry_requirement_count, 2u);
  ASSERT_NE(root.entry_requirement_indices, nullptr);
  EXPECT_EQ(root.entry_requirement_indices[0], 0u);
  EXPECT_EQ(root.entry_requirement_indices[1], 1u);
  EXPECT_EQ(root.launch_counts.binding_index, UINT32_MAX);
  EXPECT_EQ(root.launch_counts.required_byte_length, 0u);
  EXPECT_EQ(root.launch_counts.minimum_alignment, 0u);

  iree_byte_span_t data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_plan_serialize_root(&plan, 0, &data,
                                                      iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(data.data, data.data_length), &program));
  EXPECT_EQ(program.requirements.executable_count, 2u);
  EXPECT_EQ(program.requirements.entry_count, 2u);
  ASSERT_EQ(program.commands.count, 3u);
  const loom_cmd_program_command_t first =
      loom_cmd_program_command_at(&program, 0);
  const loom_cmd_program_command_t second =
      loom_cmd_program_command_at(&program, 1);
  const loom_cmd_program_command_t third =
      loom_cmd_program_command_at(&program, 2);
  EXPECT_EQ(first.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  EXPECT_EQ(second.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  EXPECT_EQ(third.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT_BARRIER);
  EXPECT_EQ(first.payload.dispatch_direct.executable_index, 0u);
  EXPECT_EQ(first.payload.dispatch_direct.entry_index, 0u);
  EXPECT_EQ(first.payload.dispatch_direct.workgroup_count_x, 7u);
  EXPECT_EQ(first.payload.dispatch_direct.workgroup_count_y, 1u);
  EXPECT_EQ(first.payload.dispatch_direct.workgroup_count_z, 1u);
  EXPECT_EQ(third.payload.dispatch_direct.executable_index, 1u);
  EXPECT_EQ(third.payload.dispatch_direct.entry_index, 1u);
  EXPECT_EQ(third.payload.dispatch_direct.workgroup_count_x, 7u);
  EXPECT_EQ(third.payload.dispatch_direct.workgroup_count_y, 5u);
  EXPECT_EQ(third.payload.dispatch_direct.workgroup_count_z, 1u);

  iree_allocator_free(iree_allocator_system(), data.data);
  loom_cmd_program_plan_deinitialize(&plan);
}

TEST_F(CmdProgramPlanTest, SerializesIndirectCountOrigins) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.entry.decl @stable_entry()
kernel.entry.decl @produce_counts(%counts: view<3xi32>)
kernel.entry.decl @dynamic_entry()

command.program.def public @stable_root() launch(%count_storage: buffer) {
  %c0 = index.constant 0 : offset
  %counts = buffer.view %count_storage[%c0] : buffer -> view<3xi32>
  kernel.dispatch @stable_entry[%counts]() : [view<3xi32>]()
  command.return
}

command.program.def public @dynamic_root() launch() {
  %c0 = index.constant 0 : offset
  %c1 = index.constant 1 : index
  %count_bytes = index.constant 12 : offset
  %count_storage = buffer.alloca<global> align(4) %count_bytes : buffer
  %counts = buffer.view %count_storage[%c0] : buffer -> view<3xi32>
  kernel.dispatch @produce_counts[%c1](%counts) : [index](view<3xi32>)
  kernel.dispatch @dynamic_entry[%counts]() : [view<3xi32>]()
  command.return
}
)");
  ASSERT_NE(source_module, nullptr);
  const loom_symbol_ref_t program_refs[] = {
      FindSymbolRef(source_module.get(), IREE_SV("stable_root")),
      FindSymbolRef(source_module.get(), IREE_SV("dynamic_root")),
  };
  loom_link_plan_materialization_t materialization = {};
  materialization.module = source_module.release();

  loom_cmd_program_plan_t plan = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      &materialization, program_refs, IREE_ARRAYSIZE(program_refs),
      loom_pass_builtin_registry(),
      /*diagnostic_emitter=*/{}, &block_pool_, &valid, &plan,
      iree_allocator_system()));
  ASSERT_TRUE(valid);
  EXPECT_EQ(materialization.module, nullptr);

  ASSERT_EQ(plan.root_count, 2u);
  const loom_cmd_program_root_t& stable_root = plan.roots[0];
  EXPECT_EQ(stable_root.launch_counts.binding_index, UINT32_MAX);
  EXPECT_EQ(stable_root.launch_counts.required_byte_length, 0u);
  EXPECT_EQ(stable_root.launch_counts.minimum_alignment, 0u);
  iree_byte_span_t stable_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_plan_serialize_root(&plan, 0, &stable_data,
                                                      iree_allocator_system()));
  loom_cmd_program_t stable_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(stable_data.data, stable_data.data_length),
      &stable_program));
  ASSERT_EQ(stable_program.commands.count, 1u);
  const loom_cmd_program_command_t stable_dispatch =
      loom_cmd_program_command_at(&stable_program, 0);
  EXPECT_EQ(stable_dispatch.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(
      loom_cmd_program_buffer_ref_at(
          &stable_program,
          stable_dispatch.payload.dispatch_indirect.workgroup_count_buffer_ref)
          .root_index,
      0u);

  const loom_cmd_program_root_t& dynamic_root = plan.roots[1];
  EXPECT_EQ(dynamic_root.launch_counts.binding_index, UINT32_MAX);
  EXPECT_EQ(dynamic_root.transient.binding_index, 0u);
  iree_byte_span_t dynamic_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_plan_serialize_root(&plan, 1, &dynamic_data,
                                                      iree_allocator_system()));
  loom_cmd_program_t dynamic_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(dynamic_data.data, dynamic_data.data_length),
      &dynamic_program));
  ASSERT_EQ(dynamic_program.commands.count, 2u);
  const loom_cmd_program_command_t dynamic_dispatch =
      loom_cmd_program_command_at(&dynamic_program, 1);
  EXPECT_EQ(dynamic_dispatch.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC_BARRIER);
  const loom_cmd_program_buffer_ref_t dynamic_count_ref =
      loom_cmd_program_buffer_ref_at(&dynamic_program,
                                     dynamic_dispatch.payload.dispatch_indirect
                                         .workgroup_count_buffer_ref);
  EXPECT_EQ(dynamic_count_ref.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(dynamic_count_ref.root_index,
            dynamic_program.requirements.transient.binding_index);

  iree_allocator_free(iree_allocator_system(), dynamic_data.data);
  iree_allocator_free(iree_allocator_system(), stable_data.data);
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
  root.launch_counts.binding_index = UINT32_MAX;
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
