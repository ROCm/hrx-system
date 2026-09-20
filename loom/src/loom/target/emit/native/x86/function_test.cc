// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/function.h"

#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/descriptors/low_registry.h"
#include "loom/target/arch/x86/ops/registry.h"
#include "loom/target/arch/x86/sysv_abi.h"
#include "loom/target/arch/x86/sysv_frame.h"
#include "loom/testing/context.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class X86FunctionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_x86_ops_register_dialect(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_x86_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
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
                                  IREE_SV("function_test.loom"), &context_,
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

  std::vector<uint8_t> Encode(const loom_low_emission_frame_t& frame) {
    loom_x86_function_encoding_t encoding = {};
    IREE_CHECK_OK(loom_x86_encode_sysv_function(&frame, &arena_, &encoding));
    return std::vector<uint8_t>(encoding.text.data,
                                encoding.text.data + encoding.text.data_length);
  }

  iree_arena_block_pool_t pool_ = {};
  iree_arena_allocator_t arena_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(X86FunctionTest, EncodesSysvFrameAndScalarArithmetic) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7, 6], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @add(%lhs: reg<x86.gpr64>, %rhs: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %sum = add.gpr64 %lhs, %rhs
  return %sum
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_value_id_t sum = loom_op_const_results(loom_block_op(body, 0))[0];
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_R12,
       1},
      {loom_block_arg_id(body, 1),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_R10,
       1},
      {sum, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_R12, 1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);

  const std::vector<uint8_t> expected = {
      0x48, 0x83, 0xEC, 0x08,  // sub rsp, 8
      0x4C, 0x89, 0x24, 0x24,  // mov [rsp], r12
      0x49, 0x89, 0xFC,        // mov r12, rdi
      0x49, 0x89, 0xF2,        // mov r10, rsi
      0x4D, 0x01, 0xD4,        // add r12, r10
      0x4C, 0x89, 0xE0,        // mov rax, r12
      0x4C, 0x8B, 0x24, 0x24,  // mov r12, [rsp]
      0x48, 0x83, 0xC4, 0x08,  // add rsp, 8
      0xC3,                    // ret
  };
  EXPECT_EQ(Encode(frame), expected);
}

TEST_F(X86FunctionTest, EncodesCompareAndSelectWithoutClobberingInputs) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7, 6, 2, 1], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @select_min(%lhs: reg<x86.gpr64>, %rhs: reg<x86.gpr64>, %true_value: reg<x86.gpr64>, %false_value: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %condition = cmp.slt.gpr64 %lhs, %rhs
  %result = select.gpr64 %condition, %true_value, %false_value
  return %result
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_value_id_t condition =
      loom_op_const_results(loom_block_op(body, 0))[0];
  const loom_value_id_t result =
      loom_op_const_results(loom_block_op(body, 1))[0];
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDI,
       1},
      {loom_block_arg_id(body, 1),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RSI,
       1},
      {loom_block_arg_id(body, 2),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDX,
       1},
      {loom_block_arg_id(body, 3),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RAX,
       1},
      {condition, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_R8, 1},
      {result, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);

  const std::vector<uint8_t> expected = {
      0x48, 0x89, 0xC8,        // mov rax, rcx
      0x48, 0x39, 0xF7,        // cmp rdi, rsi
      0x41, 0x0F, 0x9C, 0xC0,  // setl r8b
      0x45, 0x0F, 0xB6, 0xC0,  // movzx r8d, r8b
      0x45, 0x85, 0xC0,        // test r8d, r8d
      0x48, 0x0F, 0x45, 0xC2,  // cmovne rax, rdx
      0xC3,                    // ret
  };
  EXPECT_EQ(Encode(frame), expected);
}

TEST_F(X86FunctionTest, EncodesExtendedIndexedMemoryAddress) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7, 6], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @load(%base: reg<x86.gpr64>, %index: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %value = mov.load.indexed.gpr64 %base, %index {disp32 = -8, scale = 4}
  return %value
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_value_id_t value =
      loom_op_const_results(loom_block_op(body, 0))[0];
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_R10,
       1},
      {loom_block_arg_id(body, 1),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_R11,
       1},
      {value, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);

  const std::vector<uint8_t> expected = {
      0x49, 0x89, 0xFA,              // mov r10, rdi
      0x49, 0x89, 0xF3,              // mov r11, rsi
      0x4B, 0x8B, 0x44, 0x9A, 0xF8,  // mov rax, [r10+r11*4-8]
      0xC3,                          // ret
  };
  EXPECT_EQ(Encode(frame), expected);
}

TEST_F(X86FunctionTest, EncodesConditionalFallthroughAndRel32Branch) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7, 6, 2], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @choose(%condition: reg<x86.gpr32>, %true_value: reg<x86.gpr64>, %false_value: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  low.cond_br %condition, ^true, ^false : reg<x86.gpr32>
^true:
  return %true_value
^false:
  return %false_value
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_region_t* body = loom_low_func_def_body(function);
  loom_block_t* entry = loom_region_entry_block(body);
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(entry, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDI,
       1},
      {loom_block_arg_id(entry, 1),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RSI,
       1},
      {loom_block_arg_id(entry, 2),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDX,
       1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);

  const std::vector<uint8_t> expected = {
      0x85, 0xFF,                          // test edi, edi
      0x0F, 0x84, 0x04, 0x00, 0x00, 0x00,  // jz false
      0x48, 0x89, 0xF0,                    // mov rax, rsi
      0xC3,                                // ret
      0x48, 0x89, 0xD0,                    // mov rax, rdx
      0xC3,                                // ret
  };
  EXPECT_EQ(Encode(frame), expected);
}

TEST_F(X86FunctionTest, RetainsDirectCallRelocation) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.decl import(native, "external") target<x86.scalar.core>(@target) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @external(%value: reg<x86.gpr64>) -> (reg<x86.gpr64>)

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @caller(%value: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %result = low.func.call @external(%value) : (reg<x86.gpr64>) -> (reg<x86.gpr64>)
  return %result
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  loom_op_t* call_op = loom_block_op(body, 0);
  const loom_value_id_t result = loom_op_const_results(call_op)[0];
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDI,
       1},
      {result, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  ASSERT_EQ(frame.schedule.error_count, 0u);
  ASSERT_EQ(frame.allocation.error_count, 0u);

  loom_x86_function_encoding_t encoding = {};
  IREE_ASSERT_OK(loom_x86_encode_sysv_function(&frame, &arena_, &encoding));
  const std::vector<uint8_t> actual(
      encoding.text.data, encoding.text.data + encoding.text.data_length);
  const std::vector<uint8_t> expected = {
      0x48, 0x83, 0xEC, 0x08,        // sub rsp, 8
      0xE8, 0x00, 0x00, 0x00, 0x00,  // call external
      0x48, 0x83, 0xC4, 0x08,        // add rsp, 8
      0xC3,                          // ret
  };
  EXPECT_EQ(actual, expected);
  ASSERT_EQ(encoding.call_fixup_count, 1u);
  EXPECT_EQ(encoding.call_fixups[0].text_offset, 5u);
  EXPECT_EQ(encoding.call_fixups[0].target.symbol_id,
            loom_low_func_call_callee(call_op).symbol_id);
}

TEST_F(X86FunctionTest, EncodesSignedAndZeroExtensionDirections) {
  ModulePtr signed_module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @extend(%value: reg<x86.gpr32>) -> (reg<x86.gpr64>) asm {
  %result = movsxd.gpr64.gpr32 %value
  return %result
}
)");
  loom_op_t* signed_function = FindDefinition(signed_module.get());
  ASSERT_NE(signed_function, nullptr);
  loom_block_t* signed_body =
      loom_region_entry_block(loom_low_func_def_body(signed_function));
  const loom_value_id_t signed_result =
      loom_op_const_results(loom_block_op(signed_body, 0))[0];
  const loom_low_allocation_fixed_value_t signed_fixed_values[] = {
      {loom_block_arg_id(signed_body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDI,
       1},
      {signed_result, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
  };
  loom_low_emission_frame_t signed_frame = {};
  IREE_ASSERT_OK(BuildFrame(signed_module.get(), signed_fixed_values,
                            IREE_ARRAYSIZE(signed_fixed_values),
                            &signed_frame));
  EXPECT_EQ(Encode(signed_frame),
            (std::vector<uint8_t>{0x48, 0x63, 0xC7, 0xC3}));

  iree_arena_reset(&arena_);
  ModulePtr zero_module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @extend(%value: reg<x86.gpr32>) -> (reg<x86.gpr64>) asm {
  %result = movzx.gpr64.gpr32 %value
  return %result
}
)");
  loom_op_t* zero_function = FindDefinition(zero_module.get());
  ASSERT_NE(zero_function, nullptr);
  loom_block_t* zero_body =
      loom_region_entry_block(loom_low_func_def_body(zero_function));
  const loom_value_id_t zero_result =
      loom_op_const_results(loom_block_op(zero_body, 0))[0];
  const loom_low_allocation_fixed_value_t zero_fixed_values[] = {
      {loom_block_arg_id(zero_body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDI,
       1},
      {zero_result, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
  };
  loom_low_emission_frame_t zero_frame = {};
  IREE_ASSERT_OK(BuildFrame(zero_module.get(), zero_fixed_values,
                            IREE_ARRAYSIZE(zero_fixed_values), &zero_frame));
  EXPECT_EQ(Encode(zero_frame), (std::vector<uint8_t>{0x89, 0xF8, 0xC3}));
}

TEST_F(X86FunctionTest, EncodesImmediateArithmetic) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @scale(%value: reg<x86.gpr64>) -> (reg<x86.gpr32>) asm {
  %multiplied = imul.imm.gpr64 %value {imm32 = 7}
  %shifted = shl.imm.gpr64 %multiplied {shift = 3}
  %masked = and.imm.gpr64 %shifted {imm32 = -2147483648}
  %tagged = or.imm.gpr64 %masked {imm32 = 2147483647}
  %narrow = mov.trunc.gpr32.gpr64 %tagged
  %result = xor.imm.gpr32 %narrow {imm32 = -129}
  return %result
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_value_id_t multiplied =
      loom_op_const_results(loom_block_op(body, 0))[0];
  const loom_value_id_t shifted =
      loom_op_const_results(loom_block_op(body, 1))[0];
  const loom_value_id_t masked =
      loom_op_const_results(loom_block_op(body, 2))[0];
  const loom_value_id_t tagged =
      loom_op_const_results(loom_block_op(body, 3))[0];
  const loom_value_id_t narrow =
      loom_op_const_results(loom_block_op(body, 4))[0];
  const loom_value_id_t result =
      loom_op_const_results(loom_block_op(body, 5))[0];
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDI,
       1},
      {multiplied, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
      {shifted, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
      {masked, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
      {tagged, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
      {narrow, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
      {result, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  const std::vector<uint8_t> expected = {
      0x48, 0x69, 0xC7, 0x07, 0x00, 0x00, 0x00,  // imul rax, rdi, 7
      0x48, 0xC1, 0xE0, 0x03,                    // shl rax, 3
      0x48, 0x81, 0xE0, 0x00, 0x00, 0x00, 0x80,  // and rax, -2147483648
      0x48, 0x81, 0xC8, 0xFF, 0xFF, 0xFF, 0x7F,  // or rax, 2147483647
      0x89, 0xC0,                                // mov eax, eax
      0x81, 0xF0, 0x7F, 0xFF, 0xFF, 0xFF,        // xor eax, -129
      0xC3,                                      // ret
  };
  EXPECT_EQ(Encode(frame), expected);
}

TEST_F(X86FunctionTest, EncodesImmediateCompareAndConditionalSubtract) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @adjust(%value: reg<x86.gpr32>) -> (reg<x86.gpr32>) asm {
  %condition = cmp.uge.imm.gpr32 %value, -2147483647
  %adjusted = sub.if_uge.imm.gpr32 %value, -2147483647
  %result = add.gpr32 %adjusted, %condition
  return %result
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_value_id_t condition =
      loom_op_const_results(loom_block_op(body, 0))[0];
  const loom_value_id_t adjusted =
      loom_op_const_results(loom_block_op(body, 1))[0];
  const loom_value_id_t result =
      loom_op_const_results(loom_block_op(body, 2))[0];
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDI,
       1},
      {condition, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RCX, 1},
      {adjusted, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
      {result, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  const std::vector<uint8_t> expected = {
      0x81, 0xFF, 0x01, 0x00, 0x00, 0x80,  // cmp edi, -2147483647
      0x0F, 0x93, 0xC1,                    // setae cl
      0x0F, 0xB6, 0xC9,                    // movzx ecx, cl
      0x89, 0xF8,                          // mov eax, edi
      0x81, 0xE8, 0x01, 0x00, 0x00, 0x80,  // sub eax, -2147483647
      0x0F, 0x42, 0xC7,                    // cmovb eax, edi
      0x01, 0xC8,                          // add eax, ecx
      0xC3,                                // ret
  };
  EXPECT_EQ(Encode(frame), expected);
}

TEST_F(X86FunctionTest, EncodesImplicitHighProduct) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7, 6], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @high_product(%lhs: reg<x86.gpr64>, %rhs: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %accumulator = copy %lhs : reg<x86.gpr64> -> reg<x86.rax>
  %high = mul.high.gpr64 %accumulator, %rhs
  %result = copy %high : reg<x86.rdx> -> reg<x86.gpr64>
  return %result
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_value_id_t result =
      loom_op_const_results(loom_block_op(body, 2))[0];
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDI,
       1},
      {loom_block_arg_id(body, 1),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RSI,
       1},
      {result, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  const std::vector<uint8_t> expected = {
      0x48, 0x89, 0xF8,  // mov rax, rdi
      0x48, 0xF7, 0xE6,  // mul rsi
      0x48, 0x89, 0xD0,  // mov rax, rdx
      0xC3,              // ret
  };
  EXPECT_EQ(Encode(frame), expected);
}

TEST_F(X86FunctionTest, EncodesFullWidthConstantAndLea) {
  ModulePtr constant_module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @constant() -> (reg<x86.gpr64>) asm {
  %result = mov.imm64 4294967296
  return %result
}
)");
  loom_op_t* constant_function = FindDefinition(constant_module.get());
  ASSERT_NE(constant_function, nullptr);
  loom_block_t* constant_body =
      loom_region_entry_block(loom_low_func_def_body(constant_function));
  const loom_value_id_t constant_result =
      loom_op_const_results(loom_block_op(constant_body, 0))[0];
  const loom_low_allocation_fixed_value_t constant_fixed_values[] = {
      {constant_result, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
  };
  loom_low_emission_frame_t constant_frame = {};
  IREE_ASSERT_OK(BuildFrame(constant_module.get(), constant_fixed_values,
                            IREE_ARRAYSIZE(constant_fixed_values),
                            &constant_frame));
  EXPECT_EQ(Encode(constant_frame),
            (std::vector<uint8_t>{0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x01,
                                  0x00, 0x00, 0x00, 0xC3}));

  iree_arena_reset(&arena_);
  ModulePtr lea_module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7, 6], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @address(%base: reg<x86.gpr64>, %index: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %result = lea.add_scale %base, %index {disp32 = 32, scale = 8}
  return %result
}
)");
  loom_op_t* lea_function = FindDefinition(lea_module.get());
  ASSERT_NE(lea_function, nullptr);
  loom_block_t* lea_body =
      loom_region_entry_block(loom_low_func_def_body(lea_function));
  const loom_value_id_t lea_result =
      loom_op_const_results(loom_block_op(lea_body, 0))[0];
  const loom_low_allocation_fixed_value_t lea_fixed_values[] = {
      {loom_block_arg_id(lea_body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RDI,
       1},
      {loom_block_arg_id(lea_body, 1),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RSI,
       1},
      {lea_result, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_RAX, 1},
  };
  loom_low_emission_frame_t lea_frame = {};
  IREE_ASSERT_OK(BuildFrame(lea_module.get(), lea_fixed_values,
                            IREE_ARRAYSIZE(lea_fixed_values), &lea_frame));
  EXPECT_EQ(Encode(lea_frame),
            (std::vector<uint8_t>{0x48, 0x8D, 0x44, 0xF7, 0x20, 0xC3}));
}

TEST_F(X86FunctionTest, EncodesByteStoreWithRequiredRexPrefix) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7, 6], calling_convention = "sysv_x86_64", result_locations = [], stack_argument_bytes = 0}) @store(%value: reg<x86.gpr32>, %base: reg<x86.gpr64>) asm {
  mov.store.indexed.u8.gpr32 %value, %base, %base {disp32 = 0, scale = 1}
  return
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_RSI,
       1},
      {loom_block_arg_id(body, 1),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_R10,
       1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  EXPECT_EQ(Encode(frame),
            (std::vector<uint8_t>{0x49, 0x89, 0xF2, 0x89, 0xFE, 0x43, 0x88,
                                  0x34, 0x12, 0xC3}));
}

TEST_F(X86FunctionTest, EncodesStackStorageSpillAndReload) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @spill(%value: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %storage = storage {byte_alignment = 8, byte_length = 32} : low.storage<stack>
  %view = storage_view %storage {offset = 8, byte_length = 16} : low.storage<stack> -> low.storage<stack>
  low.spill %value, %view {offset = 8} : reg<x86.gpr64>, low.storage<stack>
  %reload = low.reload %view {offset = 8} : low.storage<stack> -> reg<x86.gpr64>
  return %reload
}
)");
  loom_op_t* function = FindDefinition(module.get());
  ASSERT_NE(function, nullptr);
  loom_block_t* body =
      loom_region_entry_block(loom_low_func_def_body(function));
  const loom_value_id_t reload =
      loom_op_const_results(loom_block_op(body, 3))[0];
  const loom_low_allocation_fixed_value_t fixed_values[] = {
      {loom_block_arg_id(body, 0),
       LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER, LOOM_X86_SYSV_GPR_R10,
       1},
      {reload, LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
       LOOM_X86_SYSV_GPR_R11, 1},
  };
  loom_low_emission_frame_t frame = {};
  IREE_ASSERT_OK(BuildFrame(module.get(), fixed_values,
                            IREE_ARRAYSIZE(fixed_values), &frame));
  EXPECT_EQ(Encode(frame),
            (std::vector<uint8_t>{0x48, 0x83, 0xEC, 0x28, 0x49, 0x89, 0xFA,
                                  0x4C, 0x89, 0x54, 0x24, 0x10, 0x4C, 0x8B,
                                  0x5C, 0x24, 0x10, 0x4C, 0x89, 0xD8, 0x48,
                                  0x83, 0xC4, 0x28, 0xC3}));
}

}  // namespace
}  // namespace loom
