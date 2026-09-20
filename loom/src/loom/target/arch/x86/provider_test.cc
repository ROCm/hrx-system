// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/provider.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/pass/registry.h"
#include "loom/pass/tooling.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class X86ProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_target_environment_initialize(
        &loom_x86_target_provider_set, &target_environment_));
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_target_environment_register_context(
        &target_environment_, &context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_target_environment_initialize_low_descriptor_registry(
        &target_environment_, &low_registry_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    loom_target_environment_deinitialize(&target_environment_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr Parse(iree_string_view_t source) {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("x86_provider_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  loom_op_t* FindSymbol(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return module->symbols.entries[symbol_id].defining_op;
  }

  void RunMaterialization(loom_module_t* module) {
    loom_low_pass_environment_storage_t environment_storage = {};
    const loom_pass_environment_t environment =
        loom_low_pass_environment_storage_initialize(
            &low_registry_.registry,
            /*lower_policy_registry=*/nullptr,
            /*legality_provider_list=*/nullptr,
            /*legalizer_registry=*/nullptr,
            /*math_policy_registry=*/nullptr,
            /*compile_report=*/nullptr, &target_environment_,
            /*function_versions=*/nullptr, &environment_storage);
    const loom_pass_tool_run_options_t options = {
        /*.registry=*/
        loom_target_environment_pass_registry(&target_environment_),
        /*.environment=*/environment,
        /*.function_versions=*/nullptr,
        /*.predicate_provider=*/{},
        /*.block_pool=*/&block_pool_,
    };
    loom_pass_run_result_t result = {};
    IREE_ASSERT_OK(loom_pass_tool_run_flat_pipeline(
        module, IREE_SV("x86-materialize-sysv-abi"), &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  const loom_named_attr_t* FindAttr(const loom_module_t* module,
                                    loom_named_attr_slice_t attrs,
                                    iree_string_view_t name) {
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      if (attrs.entries[i].name_id < module->strings.count &&
          iree_string_view_equal(
              module->strings.entries[attrs.entries[i].name_id], name)) {
        return &attrs.entries[i];
      }
    }
    return nullptr;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_environment_t target_environment_;
  loom_target_low_descriptor_registry_t low_registry_;
};

TEST_F(X86ProviderTest, ProvidesPassRegistry) {
  const loom_pass_registry_t* registry =
      loom_target_environment_pass_registry(&target_environment_);

  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(loom_pass_registry_lookup(
      registry, IREE_SV("x86-materialize-sysv-abi"), &descriptor));
  ASSERT_NE(descriptor, nullptr);
  ASSERT_NE(descriptor->info, nullptr);
  EXPECT_EQ(descriptor->info()->kind, LOOM_PASS_MODULE);
}

TEST_F(X86ProviderTest, MaterializesDefinitionsAndDeclarations) {
  static constexpr iree_string_view_t kSource = IREE_SVL(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) @identity(%value: reg<x86.gpr32>) -> (reg<x86.gpr32>) asm {
  return %value
}

low.func.decl import(native, "external") target<x86.scalar.core>(@target) @external(%a0: reg<x86.gpr64>, %a1: reg<x86.gpr64>, %a2: reg<x86.gpr64>, %a3: reg<x86.gpr64>, %a4: reg<x86.gpr64>, %a5: reg<x86.gpr64>, %a6: reg<x86.gpr64>, %a7: reg<x86.gpr64>) -> (reg<x86.gpr64>)
)");
  ModulePtr module = Parse(kSource);

  ASSERT_NO_FATAL_FAILURE(RunMaterialization(module.get()));

  loom_op_t* identity = FindSymbol(module.get(), IREE_SV("identity"));
  ASSERT_TRUE(loom_low_func_def_isa(identity));
  EXPECT_EQ(loom_low_func_def_abi(identity), LOOM_TARGET_ABI_OBJECT_FUNCTION);
  loom_named_attr_slice_t identity_layout =
      loom_low_func_def_abi_layout(identity);
  ASSERT_EQ(identity_layout.count, 4u);
  const loom_named_attr_t* identity_arguments =
      FindAttr(module.get(), identity_layout, IREE_SV("argument_locations"));
  ASSERT_NE(identity_arguments, nullptr);
  ASSERT_EQ(identity_arguments->value.kind, LOOM_ATTR_I64_ARRAY);
  ASSERT_EQ(identity_arguments->value.count, 1u);
  EXPECT_EQ(identity_arguments->value.i64_array[0], 7);

  loom_op_t* external = FindSymbol(module.get(), IREE_SV("external"));
  ASSERT_TRUE(loom_low_func_decl_isa(external));
  EXPECT_EQ(loom_low_func_decl_abi(external), LOOM_TARGET_ABI_OBJECT_FUNCTION);
  loom_named_attr_slice_t external_layout =
      loom_low_func_decl_abi_layout(external);
  ASSERT_EQ(external_layout.count, 4u);
  const loom_named_attr_t* external_arguments =
      FindAttr(module.get(), external_layout, IREE_SV("argument_locations"));
  ASSERT_NE(external_arguments, nullptr);
  ASSERT_EQ(external_arguments->value.kind, LOOM_ATTR_I64_ARRAY);
  ASSERT_EQ(external_arguments->value.count, 8u);
  const int64_t expected_locations[] = {7, 6, 2, 1, 8, 9, -1, -9};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected_locations); ++i) {
    EXPECT_EQ(external_arguments->value.i64_array[i], expected_locations[i]);
  }
  const loom_named_attr_t* stack_bytes =
      FindAttr(module.get(), external_layout, IREE_SV("stack_argument_bytes"));
  ASSERT_NE(stack_bytes, nullptr);
  EXPECT_EQ(loom_attr_as_i64(stack_bytes->value), 16);
}

TEST_F(X86ProviderTest, MaterializesStackArgumentsAtTheirUses) {
  static constexpr iree_string_view_t kSource = IREE_SVL(R"(
x86.target<scalar> @target

low.func.decl import(native, "external") target<x86.scalar.core>(@target) @external(%a0: reg<x86.gpr64>, %a1: reg<x86.gpr64>, %a2: reg<x86.gpr64>, %a3: reg<x86.gpr64>, %a4: reg<x86.gpr64>, %a5: reg<x86.gpr64>, %a6: reg<x86.gpr64>, %a7: reg<x86.gpr64>) -> (reg<x86.gpr64>)

low.func.def target<x86.scalar.core>(@target) @caller(%a0: reg<x86.gpr64>, %a1: reg<x86.gpr64>, %a2: reg<x86.gpr64>, %a3: reg<x86.gpr64>, %a4: reg<x86.gpr64>, %a5: reg<x86.gpr64>, %a6: reg<x86.gpr64>, %a7: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %result = low.func.call @external(%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7) : (reg<x86.gpr64>, reg<x86.gpr64>, reg<x86.gpr64>, reg<x86.gpr64>, reg<x86.gpr64>, reg<x86.gpr64>, reg<x86.gpr64>, reg<x86.gpr64>) -> (reg<x86.gpr64>)
  return %result
}
)");
  ModulePtr module = Parse(kSource);

  ASSERT_NO_FATAL_FAILURE(RunMaterialization(module.get()));

  loom_op_t* caller = FindSymbol(module.get(), IREE_SV("caller"));
  ASSERT_TRUE(loom_low_func_def_isa(caller));
  const loom_func_like_t function = loom_func_like_cast(module.get(), caller);
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  ASSERT_EQ(argument_count, 8u);
  EXPECT_TRUE(
      loom_value_has_no_uses(loom_module_value(module.get(), argument_ids[6])));
  EXPECT_TRUE(
      loom_value_has_no_uses(loom_module_value(module.get(), argument_ids[7])));

  const loom_block_t* block =
      loom_region_entry_block(loom_low_func_def_body(caller));
  const loom_op_t* op = block->first_op;
  ASSERT_TRUE(loom_low_func_stack_arg_isa(op));
  EXPECT_EQ(loom_low_func_stack_arg_ordinal(op), 6);
  EXPECT_EQ(loom_low_func_stack_arg_byte_offset(op), 0);
  op = op->next_op;
  ASSERT_TRUE(loom_low_func_call_arg_isa(op));
  EXPECT_EQ(loom_low_func_call_arg_ordinal(op), 6);
  EXPECT_EQ(loom_low_func_call_arg_byte_offset(op), 0);
  const loom_value_id_t outgoing6 = loom_low_func_call_arg_token(op);
  op = op->next_op;
  ASSERT_TRUE(loom_low_func_stack_arg_isa(op));
  EXPECT_EQ(loom_low_func_stack_arg_ordinal(op), 7);
  EXPECT_EQ(loom_low_func_stack_arg_byte_offset(op), 8);
  op = op->next_op;
  ASSERT_TRUE(loom_low_func_call_arg_isa(op));
  EXPECT_EQ(loom_low_func_call_arg_ordinal(op), 7);
  EXPECT_EQ(loom_low_func_call_arg_byte_offset(op), 8);
  const loom_value_id_t outgoing7 = loom_low_func_call_arg_token(op);
  op = op->next_op;
  ASSERT_TRUE(loom_low_func_call_isa(op));
  const loom_value_slice_t register_args = loom_low_func_call_operands(op);
  const loom_value_slice_t stack_args = loom_low_func_call_stack_args(op);
  ASSERT_EQ(register_args.count, 6u);
  ASSERT_EQ(stack_args.count, 2u);
  EXPECT_EQ(stack_args.values[0], outgoing6);
  EXPECT_EQ(stack_args.values[1], outgoing7);
  op = op->next_op;
  ASSERT_TRUE(loom_low_return_isa(op));
  EXPECT_EQ(op->next_op, nullptr);

  ASSERT_NO_FATAL_FAILURE(RunMaterialization(module.get()));
  uint32_t operation_count = 0;
  for (const loom_op_t* current = block->first_op; current != nullptr;
       current = current->next_op) {
    ++operation_count;
  }
  EXPECT_EQ(operation_count, 6u);
}

}  // namespace
}  // namespace loom
