// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/ireevm/candidate.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <utility>

#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/bytecode_module_def_reader.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/archive.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/native_module.h"
#include "iree/vm/shims.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/ireevm/descriptors/descriptors.h"
#include "loom/target/arch/ireevm/ops/registry.h"
#include "loom/target/arch/ireevm/provider.h"
#include "loom/target/emit/ireevm/function_bytecode.h"
#include "loom/target/provider.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/execution/ireevm/invocation.h"

namespace loom {
namespace {

iree_status_t RegisterContext(void* user_data, loom_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(context));
  const loom_target_environment_t* target_environment =
      static_cast<const loom_target_environment_t*>(user_data);
  return loom_target_environment_register_context(target_environment, context);
}

iree_status_t InitializeLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  const loom_target_environment_t* target_environment =
      static_cast<const loom_target_environment_t*>(user_data);
  return loom_target_environment_initialize_low_descriptor_registry(
      target_environment, out_registry);
}

constexpr char kPreparedVmSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "@double(%value: reg<ireevm.i32>) -> (reg<ireevm.i32>) {\n"
    "  %sum = low.op<ireevm.add.i32>(%value, %value) : "
    "(reg<ireevm.i32>, reg<ireevm.i32>) -> reg<ireevm.i32>\n"
    "  low.return %sum : reg<ireevm.i32>\n"
    "}\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"branchy\") "
    "@branchy(%lhs: reg<ireevm.i32>, %rhs: reg<ireevm.i32>) -> "
    "(reg<ireevm.i32>) {\n"
    "  %c0 = low.const<ireevm.const.i32> {i32_value = 0} : reg<ireevm.i32>\n"
    "  %is_zero = low.op<ireevm.cmp.eq.i32>(%lhs, %c0) : "
    "(reg<ireevm.i32>, reg<ireevm.i32>) -> reg<ireevm.i32>\n"
    "  low.cond_br %is_zero, ^then, ^else : reg<ireevm.i32>\n"
    "^then:\n"
    "  %sum = low.func.call @double(%rhs) : "
    "(reg<ireevm.i32>) -> (reg<ireevm.i32>)\n"
    "  low.return %sum : reg<ireevm.i32>\n"
    "^else:\n"
    "  %diff = low.op<ireevm.sub.i32>(%lhs, %rhs) : "
    "(reg<ireevm.i32>, reg<ireevm.i32>) -> reg<ireevm.i32>\n"
    "  low.return %diff : reg<ireevm.i32>\n"
    "}\n";

constexpr char kPreparedVmImportSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.decl import(vm, \"hal.buffer.length\") target(@vm_target) "
    "abi(vm_module_function) "
    "@hal_buffer_length(%value: reg<ireevm.i32>) -> (reg<ireevm.i32>)\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"length_identity\") "
    "@length_identity(%value: reg<ireevm.i32>) -> (reg<ireevm.i32>) {\n"
    "  %length = low.func.call @hal_buffer_length(%value) : "
    "(reg<ireevm.i32>) -> (reg<ireevm.i32>)\n"
    "  low.return %length : reg<ireevm.i32>\n"
    "}\n";

constexpr char kPreparedVmWideSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"wide_numeric\") "
    "@wide_numeric(%lhs: reg<ireevm.i64 x2>, %rhs: reg<ireevm.i64 x2>, "
    "%x: reg<ireevm.f32>, %y: reg<ireevm.f32>, "
    "%z: reg<ireevm.f64 x2>) -> "
    "(reg<ireevm.i64 x2>, reg<ireevm.f32>, reg<ireevm.f64 x2>, "
    "reg<ireevm.i32>) {\n"
    "  %sum = low.op<ireevm.add.i64>(%lhs, %rhs) : "
    "(reg<ireevm.i64 x2>, reg<ireevm.i64 x2>) -> reg<ireevm.i64 x2>\n"
    "  %product = low.op<ireevm.mul.f32>(%x, %y) : "
    "(reg<ireevm.f32>, reg<ireevm.f32>) -> reg<ireevm.f32>\n"
    "  %one_point_five = low.const<ireevm.const.f64> "
    "{f64_bits = 4609434218613702656} : reg<ireevm.f64 x2>\n"
    "  %wide_sum = low.op<ireevm.add.f64>(%z, %one_point_five) : "
    "(reg<ireevm.f64 x2>, reg<ireevm.f64 x2>) -> reg<ireevm.f64 x2>\n"
    "  %less = low.op<ireevm.cmp.lt.o.f64>(%z, %wide_sum) : "
    "(reg<ireevm.f64 x2>, reg<ireevm.f64 x2>) -> reg<ireevm.i32>\n"
    "  low.return %sum, %product, %wide_sum, %less : "
    "reg<ireevm.i64 x2>, reg<ireevm.f32>, reg<ireevm.f64 x2>, "
    "reg<ireevm.i32>\n"
    "}\n";

constexpr char kPreparedVmWideAbiLayoutSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"wide_abi_layout\") "
    "@wide_abi_layout(%wide0: reg<ireevm.i64 x2>, "
    "%narrow: reg<ireevm.i32>, %wide1: reg<ireevm.i64 x2>, "
    "%scalar: reg<ireevm.f32>, %wide2: reg<ireevm.f64 x2>) -> "
    "(reg<ireevm.i64 x2>, reg<ireevm.i32>, reg<ireevm.i64 x2>, "
    "reg<ireevm.f32>, reg<ireevm.f64 x2>) {\n"
    "  low.return %wide0, %narrow, %wide1, %scalar, %wide2 : "
    "reg<ireevm.i64 x2>, reg<ireevm.i32>, reg<ireevm.i64 x2>, "
    "reg<ireevm.f32>, reg<ireevm.f64 x2>\n"
    "}\n";

constexpr char kPreparedVmRefSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"retain_release\") "
    "@retain_release(%value: reg<ireevm.ref>) -> (reg<ireevm.ref>) {\n"
    "  %owned = low.op<ireevm.ref.retain>(%value) : "
    "(reg<ireevm.ref>) -> reg<ireevm.ref>\n"
    "  low.op<ireevm.ref.release>(%owned) : (reg<ireevm.ref>)\n"
    "  low.return %value : reg<ireevm.ref>\n"
    "}\n";

constexpr char kPreparedVmRefBranchSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"ref_branch\") "
    "@ref_branch(%value: reg<ireevm.ref>) {\n"
    "  low.br ^join(%value: reg<ireevm.ref>)\n"
    "^join(%arg: reg<ireevm.ref>):\n"
    "  low.op<ireevm.ref.release>(%arg) : (reg<ireevm.ref>)\n"
    "  low.return\n"
    "}\n";

constexpr char kPreparedVmBufferSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"buffer_ops\") "
    "@buffer_ops(%buffer: reg<ireevm.ref>, "
    "%offset: reg<ireevm.i64 x2>, "
    "%i32_value: reg<ireevm.i32>, "
    "%i64_value: reg<ireevm.i64 x2>, "
    "%f32_value: reg<ireevm.f32>, "
    "%f64_value: reg<ireevm.f64 x2>) -> "
    "(reg<ireevm.i64 x2>, reg<ireevm.i32>, reg<ireevm.i64 x2>, "
    "reg<ireevm.f32>, reg<ireevm.f64 x2>) {\n"
    "  %length = low.op<ireevm.buffer.length>(%buffer) : "
    "(reg<ireevm.ref>) -> reg<ireevm.i64 x2>\n"
    "  %load_i8u = low.op<ireevm.buffer.load.i8.u>(%buffer, %offset) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i32>\n"
    "  %load_i8s = low.op<ireevm.buffer.load.i8.s>(%buffer, %offset) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i32>\n"
    "  %load_i16u = low.op<ireevm.buffer.load.i16.u>(%buffer, %offset) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i32>\n"
    "  %load_i16s = low.op<ireevm.buffer.load.i16.s>(%buffer, %offset) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i32>\n"
    "  %load_i32 = low.op<ireevm.buffer.load.i32>(%buffer, %offset) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i32>\n"
    "  %load_i64 = low.op<ireevm.buffer.load.i64>(%buffer, %offset) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i64 x2>\n"
    "  %load_f32 = low.op<ireevm.buffer.load.f32>(%buffer, %offset) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.f32>\n"
    "  %load_f64 = low.op<ireevm.buffer.load.f64>(%buffer, %offset) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.f64 x2>\n"
    "  low.op<ireevm.buffer.store.i8>(%buffer, %offset, %i32_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.i32>)\n"
    "  low.op<ireevm.buffer.store.i16>(%buffer, %offset, %i32_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.i32>)\n"
    "  low.op<ireevm.buffer.store.i32>(%buffer, %offset, %i32_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.i32>)\n"
    "  low.op<ireevm.buffer.store.i64>(%buffer, %offset, %i64_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.i64 x2>)\n"
    "  low.op<ireevm.buffer.store.f32>(%buffer, %offset, %f32_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.f32>)\n"
    "  low.op<ireevm.buffer.store.f64>(%buffer, %offset, %f64_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.f64 x2>)\n"
    "  low.return %length, %load_i32, %load_i64, %load_f32, %load_f64 : "
    "reg<ireevm.i64 x2>, reg<ireevm.i32>, reg<ireevm.i64 x2>, "
    "reg<ireevm.f32>, reg<ireevm.f64 x2>\n"
    "}\n";

constexpr char kPreparedVmBufferExecutionSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"buffer_execute\") "
    "@buffer_execute(%buffer: reg<ireevm.ref>, "
    "%i32_value: reg<ireevm.i32>, "
    "%i64_value: reg<ireevm.i64 x2>, "
    "%f32_value: reg<ireevm.f32>, "
    "%f64_value: reg<ireevm.f64 x2>) -> "
    "(reg<ireevm.i64 x2>, reg<ireevm.i32>, reg<ireevm.i32>, "
    "reg<ireevm.i32>, reg<ireevm.i32>, reg<ireevm.i32>, "
    "reg<ireevm.i64 x2>, reg<ireevm.f32>, reg<ireevm.f64 x2>) {\n"
    "  %offset_i8 = low.const<ireevm.const.i64> {i64_value = 0} : "
    "reg<ireevm.i64 x2>\n"
    "  %offset_i16 = low.const<ireevm.const.i64> {i64_value = 1} : "
    "reg<ireevm.i64 x2>\n"
    "  %offset_i32 = low.const<ireevm.const.i64> {i64_value = 1} : "
    "reg<ireevm.i64 x2>\n"
    "  %offset_i64 = low.const<ireevm.const.i64> {i64_value = 1} : "
    "reg<ireevm.i64 x2>\n"
    "  %offset_f32 = low.const<ireevm.const.i64> {i64_value = 4} : "
    "reg<ireevm.i64 x2>\n"
    "  %offset_f64 = low.const<ireevm.const.i64> {i64_value = 3} : "
    "reg<ireevm.i64 x2>\n"
    "  %length = low.op<ireevm.buffer.length>(%buffer) : "
    "(reg<ireevm.ref>) -> reg<ireevm.i64 x2>\n"
    "  %load_i8u = low.op<ireevm.buffer.load.i8.u>(%buffer, %offset_i8) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i32>\n"
    "  %load_i8s = low.op<ireevm.buffer.load.i8.s>(%buffer, %offset_i8) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i32>\n"
    "  %load_i16u = low.op<ireevm.buffer.load.i16.u>(%buffer, %offset_i16) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i32>\n"
    "  %load_i16s = low.op<ireevm.buffer.load.i16.s>(%buffer, %offset_i16) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i32>\n"
    "  %load_i32 = low.op<ireevm.buffer.load.i32>(%buffer, %offset_i32) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i32>\n"
    "  %load_i64 = low.op<ireevm.buffer.load.i64>(%buffer, %offset_i64) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.i64 x2>\n"
    "  %load_f32 = low.op<ireevm.buffer.load.f32>(%buffer, %offset_f32) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.f32>\n"
    "  %load_f64 = low.op<ireevm.buffer.load.f64>(%buffer, %offset_f64) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>) -> reg<ireevm.f64 x2>\n"
    "  low.op<ireevm.buffer.store.i8>(%buffer, %offset_i8, %i32_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.i32>)\n"
    "  low.op<ireevm.buffer.store.i16>(%buffer, %offset_i16, %i32_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.i32>)\n"
    "  low.op<ireevm.buffer.store.i32>(%buffer, %offset_i32, %i32_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.i32>)\n"
    "  low.op<ireevm.buffer.store.i64>(%buffer, %offset_i64, %i64_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.i64 x2>)\n"
    "  low.op<ireevm.buffer.store.f32>(%buffer, %offset_f32, %f32_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.f32>)\n"
    "  low.op<ireevm.buffer.store.f64>(%buffer, %offset_f64, %f64_value) : "
    "(reg<ireevm.ref>, reg<ireevm.i64 x2>, reg<ireevm.f64 x2>)\n"
    "  low.op<ireevm.ref.release>(%buffer) : (reg<ireevm.ref>)\n"
    "  low.return %length, %load_i8u, %load_i8s, %load_i16u, %load_i16s, "
    "%load_i32, %load_i64, %load_f32, %load_f64 : "
    "reg<ireevm.i64 x2>, reg<ireevm.i32>, reg<ireevm.i32>, "
    "reg<ireevm.i32>, reg<ireevm.i32>, reg<ireevm.i32>, "
    "reg<ireevm.i64 x2>, reg<ireevm.f32>, reg<ireevm.f64 x2>\n"
    "}\n";

constexpr char kSourceVmOrchestrationSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "ireevm.import.decl target(@vm_target) "
    "symbol(\"loom_test.buffer_peek_i32\") "
    "@buffer_peek_i32(%buffer: ireevm.ref<ireevm.buffer>, "
    "%byte_offset: i64) -> (i32)\n"
    "ireevm.import.decl target(@vm_target) "
    "symbol(\"loom_test.retain_buffer\") "
    "@retain_buffer(%buffer: ireevm.ref<ireevm.buffer>) -> "
    "(ireevm.ref<ireevm.buffer>)\n"
    "ireevm.import.decl target(@vm_target) "
    "symbol(\"loom_test.scale_i32\") "
    "@scale_i32(%value: i32) -> (i32)\n"
    "\n"
    "func.def target(@vm_target) @double_i32(%value: i32) -> (i32) {\n"
    "  %doubled = scalar.addi %value, %value : i32\n"
    "  func.return %doubled : i32\n"
    "}\n"
    "\n"
    "func.def target(@vm_target) export(\"orchestrate\") "
    "@orchestrate(%buffer: ireevm.ref<ireevm.buffer>, "
    "%value: i32, %element_offset: i64) -> "
    "(i32, i32, i32, i64, ireevm.ref<ireevm.buffer>) {\n"
    "  %doubled = func.call @double_i32(%value) : (i32) -> (i32)\n"
    "  %scaled = func.call @scale_i32(%doubled) : (i32) -> (i32)\n"
    "  %direct = ireevm.buffer.load.i32 %buffer[%element_offset] : "
    "ireevm.ref<ireevm.buffer>, i64 -> i32\n"
    "  %double_offset = scalar.addi %element_offset, %element_offset : i64\n"
    "  %byte_offset = scalar.addi %double_offset, %double_offset : i64\n"
    "  %peeked = func.call @buffer_peek_i32(%buffer, %byte_offset) : "
    "(ireevm.ref<ireevm.buffer>, i64) -> (i32)\n"
    "  %length = ireevm.buffer.length %buffer : "
    "ireevm.ref<ireevm.buffer> -> i64\n"
    "  %owned_buffer = func.call @retain_buffer(%buffer) : "
    "(ireevm.ref<ireevm.buffer>) -> (ireevm.ref<ireevm.buffer>)\n"
    "  func.return %scaled, %direct, %peeked, %length, %owned_buffer : "
    "i32, i32, i32, i64, ireevm.ref<ireevm.buffer>\n"
    "}\n";

bool FlatbufferStringEquals(flatbuffers_string_t value,
                            iree_string_view_t expected) {
  return iree_string_view_equal(
      iree_make_string_view(value, flatbuffers_string_len(value)), expected);
}

uint16_t ReadU16LE(const uint8_t* data) {
  return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

void ExpectSingleBranchRemapCount(
    iree_vm_BytecodeModuleDef_table_t module_def,
    iree_vm_FunctionDescriptor_struct_t function_descriptor,
    uint16_t expected_remap_count) {
  constexpr uint8_t kBranchOp = 0x56;
  flatbuffers_uint8_vec_t bytecode_data =
      iree_vm_BytecodeModuleDef_bytecode_data(module_def);
  ASSERT_NE(bytecode_data, nullptr);
  const size_t bytecode_data_length = flatbuffers_uint8_vec_len(bytecode_data);
  const size_t bytecode_offset = function_descriptor->bytecode_offset;
  const size_t bytecode_length = function_descriptor->bytecode_length;
  ASSERT_LE(bytecode_offset, bytecode_data_length);
  ASSERT_LE(bytecode_length, bytecode_data_length - bytecode_offset);
  const uint8_t* function_bytecode = bytecode_data + bytecode_offset;
  const uint8_t* function_end = function_bytecode + bytecode_length;
  const uint8_t* branch_op =
      std::find(function_bytecode, function_end, kBranchOp);
  ASSERT_NE(branch_op, function_end);

  size_t remap_count_offset = (size_t)(branch_op - function_bytecode) + 1 + 4;
  if ((remap_count_offset & 1u) != 0) {
    ++remap_count_offset;
  }
  ASSERT_LE(remap_count_offset + sizeof(uint16_t), bytecode_length);
  EXPECT_EQ(ReadU16LE(function_bytecode + remap_count_offset),
            expected_remap_count);
}

void ExpectFunctionBytecodeContainsOpcodes(
    iree_vm_BytecodeModuleDef_table_t module_def,
    iree_vm_FunctionDescriptor_struct_t function_descriptor,
    std::initializer_list<uint32_t> opcodes) {
  flatbuffers_uint8_vec_t bytecode_data =
      iree_vm_BytecodeModuleDef_bytecode_data(module_def);
  ASSERT_NE(bytecode_data, nullptr);
  const size_t bytecode_data_length = flatbuffers_uint8_vec_len(bytecode_data);
  const size_t bytecode_offset = function_descriptor->bytecode_offset;
  const size_t bytecode_length = function_descriptor->bytecode_length;
  ASSERT_LE(bytecode_offset, bytecode_data_length);
  ASSERT_LE(bytecode_length, bytecode_data_length - bytecode_offset);
  const uint8_t* function_bytecode = bytecode_data + bytecode_offset;
  const uint8_t* function_end = function_bytecode + bytecode_length;
  const uint8_t* search_start = function_bytecode;

  for (uint32_t opcode : opcodes) {
    uint8_t encoded_opcode[2] = {
        (uint8_t)(opcode >> 8),
        (uint8_t)opcode,
    };
    const uint8_t* pattern =
        opcode <= UINT8_MAX ? &encoded_opcode[1] : encoded_opcode;
    const size_t pattern_length = opcode <= UINT8_MAX ? 1 : 2;
    const uint8_t* match = std::search(search_start, function_end, pattern,
                                       pattern + pattern_length);
    ASSERT_NE(match, function_end) << "missing opcode 0x" << std::hex << opcode;
    search_start = match + pattern_length;
  }
}

void ExpectImportedFunction(iree_vm_ImportFunctionDef_vec_t imported_functions,
                            iree_host_size_t index, iree_string_view_t name,
                            iree_string_view_t calling_convention) {
  ASSERT_LT(index, iree_vm_ImportFunctionDef_vec_len(imported_functions));
  iree_vm_ImportFunctionDef_table_t import_def =
      iree_vm_ImportFunctionDef_vec_at(imported_functions, index);
  EXPECT_TRUE(FlatbufferStringEquals(
      iree_vm_ImportFunctionDef_full_name(import_def), name));
  iree_vm_FunctionSignatureDef_table_t signature_def =
      iree_vm_ImportFunctionDef_signature(import_def);
  ASSERT_NE(signature_def, nullptr);
  EXPECT_TRUE(FlatbufferStringEquals(
      iree_vm_FunctionSignatureDef_calling_convention(signature_def),
      calling_convention));
}

void ExpectRegisterClassLayout(uint16_t register_class_id,
                               loom_ireevm_register_bank_t expected_bank,
                               uint16_t expected_unit_count,
                               uint16_t expected_alignment,
                               iree_string_view_t expected_class_name) {
  loom_ireevm_register_class_layout_t layout = {};
  IREE_ASSERT_OK(loom_ireevm_register_class_layout(register_class_id, &layout));
  EXPECT_EQ(layout.bank, expected_bank);
  EXPECT_EQ(layout.unit_count, expected_unit_count);
  EXPECT_EQ(layout.alignment, expected_alignment);
  EXPECT_TRUE(iree_string_view_equal(layout.class_name, expected_class_name));
}

void ExpectOutputI32(iree_vm_list_t* outputs, iree_host_size_t index,
                     int32_t expected_value) {
  iree_vm_value_t value = iree_vm_value_make_none();
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs, index, &value));
  ASSERT_EQ(value.type, IREE_VM_VALUE_TYPE_I32);
  EXPECT_EQ(value.i32, expected_value);
}

void ExpectOutputI64(iree_vm_list_t* outputs, iree_host_size_t index,
                     int64_t expected_value) {
  iree_vm_value_t value = iree_vm_value_make_none();
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs, index, &value));
  ASSERT_EQ(value.type, IREE_VM_VALUE_TYPE_I64);
  EXPECT_EQ(value.i64, expected_value);
}

void ExpectOutputF32(iree_vm_list_t* outputs, iree_host_size_t index,
                     float expected_value) {
  iree_vm_value_t value = iree_vm_value_make_none();
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs, index, &value));
  ASSERT_EQ(value.type, IREE_VM_VALUE_TYPE_F32);
  EXPECT_FLOAT_EQ(value.f32, expected_value);
}

void ExpectOutputF64(iree_vm_list_t* outputs, iree_host_size_t index,
                     double expected_value) {
  iree_vm_value_t value = iree_vm_value_make_none();
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs, index, &value));
  ASSERT_EQ(value.type, IREE_VM_VALUE_TYPE_F64);
  EXPECT_DOUBLE_EQ(value.f64, expected_value);
}

typedef struct loom_test_imports_state_t loom_test_imports_state_t;

IREE_VM_ABI_EXPORT(LoomTestBufferPeekI32, loom_test_imports_state_t, rI, i) {
  (void)stack;
  (void)module;
  (void)state;
  iree_vm_buffer_t* buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r0, &buffer));
  if (args->i1 < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "negative buffer byte offset");
  }
  iree_const_byte_span_t source_span = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_ro(buffer, (iree_host_size_t)args->i1,
                                             sizeof(rets->i0), sizeof(rets->i0),
                                             &source_span));
  std::memcpy(&rets->i0, source_span.data, sizeof(rets->i0));
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(LoomTestRetainBuffer, loom_test_imports_state_t, r, r) {
  (void)stack;
  (void)module;
  (void)state;
  iree_vm_ref_retain(&args->r0, &rets->r0);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(LoomTestScaleI32, loom_test_imports_state_t, i, i) {
  (void)stack;
  (void)module;
  (void)state;
  rets->i0 = args->i0 * 3;
  return iree_ok_status();
}

static const iree_vm_native_export_descriptor_t kLoomTestImportExports[] = {
    {IREE_SV("buffer_peek_i32"), IREE_SV("0rI_i"), 0, nullptr},
    {IREE_SV("retain_buffer"), IREE_SV("0r_r"), 0, nullptr},
    {IREE_SV("scale_i32"), IREE_SV("0i_i"), 0, nullptr},
};

static const iree_vm_native_function_ptr_t kLoomTestImportFunctions[] = {
    {(iree_vm_native_function_shim_t)iree_vm_shim_rI_i,
     (iree_vm_native_function_target_t)LoomTestBufferPeekI32},
    {(iree_vm_native_function_shim_t)iree_vm_shim_r_r,
     (iree_vm_native_function_target_t)LoomTestRetainBuffer},
    {(iree_vm_native_function_shim_t)iree_vm_shim_i_i,
     (iree_vm_native_function_target_t)LoomTestScaleI32},
};

static_assert(IREE_ARRAYSIZE(kLoomTestImportFunctions) ==
                  IREE_ARRAYSIZE(kLoomTestImportExports),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t kLoomTestImportModule = {
    /*name=*/IREE_SV("loom_test"),
    /*version=*/0,
    /*attr_count=*/0,
    /*attrs=*/nullptr,
    /*dependency_count=*/0,
    /*dependencies=*/nullptr,
    /*import_count=*/0,
    /*imports=*/nullptr,
    /*export_count=*/IREE_ARRAYSIZE(kLoomTestImportExports),
    /*exports=*/kLoomTestImportExports,
    /*function_count=*/IREE_ARRAYSIZE(kLoomTestImportFunctions),
    /*functions=*/kLoomTestImportFunctions,
};

iree_status_t CreateLoomTestImportModule(iree_vm_instance_t* instance,
                                         iree_allocator_t allocator,
                                         iree_vm_module_t** out_module) {
  iree_vm_module_t interface;
  IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, nullptr));
  return iree_vm_native_module_create(&interface, &kLoomTestImportModule,
                                      instance, allocator, out_module);
}

class IreeVmCandidateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_target_environment_initialize(
        &loom_ireevm_target_provider_set, &target_environment_));
    loom_run_session_options_t options = {};
    loom_run_session_options_initialize(&options);
    options.register_context = (loom_run_register_context_callback_t){
        /*.fn=*/RegisterContext,
        /*.user_data=*/&target_environment_,
    };
    options.initialize_low_descriptor_registry =
        (loom_run_initialize_low_descriptor_registry_callback_t){
            /*.fn=*/InitializeLowDescriptorRegistry,
            /*.user_data=*/&target_environment_,
        };
    IREE_ASSERT_OK(loom_run_session_initialize(&options, &session_));
  }

  void TearDown() override {
    loom_run_session_deinitialize(&session_);
    loom_target_environment_deinitialize(&target_environment_);
  }

  iree_status_t Parse(iree_string_view_t source,
                      loom_run_module_t* out_module) {
    loom_run_module_parse_options_t options = {};
    loom_run_module_parse_options_initialize(&options);
    options.filename = IREE_SV("ireevm_candidate_test.loom");
    options.source = source;
    return loom_run_module_parse(&session_, &options, out_module);
  }

  void InitializeCandidateOptions(
      loom_run_module_t* run_module,
      loom_run_candidate_compile_options_t* out_options) {
    loom_run_candidate_compile_options_initialize(out_options);
    out_options->source_resolver = loom_run_module_source_resolver(run_module);
  }

  iree_status_t LowerToPreparedLow(loom_run_module_t* run_module) {
    loom_compile_pipeline_options_t options = {};
    loom_compile_pipeline_options_initialize(&options);
    options.target_environment = &target_environment_;
    options.low_descriptor_registry =
        loom_run_session_low_descriptor_registry(&session_);
    options.source_resolver = loom_run_module_source_resolver(run_module);

    loom_pass_run_result_t run_result = {};
    IREE_RETURN_IF_ERROR(loom_compile_run_pipeline(
        run_module->module, &options, loom_run_session_block_pool(&session_),
        &run_result));
    if (run_result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source-to-low pipeline emitted %u errors",
                              run_result.error_count);
    }
    return iree_ok_status();
  }

  void ParseArchive(const loom_ireevm_module_archive_t* archive,
                    iree_vm_BytecodeModuleDef_table_t* out_module_def) {
    iree_const_byte_span_t flatbuffer = iree_const_byte_span_empty();
    IREE_ASSERT_OK(iree_vm_bytecode_archive_parse_header(
        iree_make_const_byte_span(archive->data, archive->data_length),
        &flatbuffer, nullptr));
    *out_module_def = iree_vm_BytecodeModuleDef_as_root(flatbuffer.data);
    ASSERT_NE(*out_module_def, nullptr);
  }

  loom_run_session_t session_ = {};
  loom_target_environment_t target_environment_ = {};
};

TEST(IreeVmRegisterClassLayoutTest, MapsCoreClassesToVmBytecodeBanks) {
  ExpectRegisterClassLayout(IREEVM_CORE_REG_CLASS_ID_I32,
                            LOOM_IREEVM_REGISTER_BANK_I32, 1, 1,
                            IREE_SV("ireevm.i32"));
  ExpectRegisterClassLayout(IREEVM_CORE_REG_CLASS_ID_I64,
                            LOOM_IREEVM_REGISTER_BANK_I32, 2, 2,
                            IREE_SV("ireevm.i64"));
  ExpectRegisterClassLayout(IREEVM_CORE_REG_CLASS_ID_F32,
                            LOOM_IREEVM_REGISTER_BANK_I32, 1, 1,
                            IREE_SV("ireevm.f32"));
  ExpectRegisterClassLayout(IREEVM_CORE_REG_CLASS_ID_F64,
                            LOOM_IREEVM_REGISTER_BANK_I32, 2, 2,
                            IREE_SV("ireevm.f64"));
  ExpectRegisterClassLayout(IREEVM_CORE_REG_CLASS_ID_REF,
                            LOOM_IREEVM_REGISTER_BANK_REF, 1, 1,
                            IREE_SV("ireevm.ref"));
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS;
  options.report = &report;

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);
  EXPECT_EQ(candidate.compile_report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE);
  EXPECT_EQ(candidate.compile_report.status_code, IREE_STATUS_OK);
  EXPECT_TRUE(
      iree_all_bits_set(candidate.compile_report.detail_flags,
                        LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE));
  EXPECT_TRUE(iree_all_bits_set(candidate.compile_report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE));
  EXPECT_TRUE(iree_all_bits_set(candidate.compile_report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION));
  EXPECT_TRUE(iree_all_bits_set(candidate.compile_report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION));
  EXPECT_TRUE(
      iree_all_bits_set(candidate.compile_report.detail_flags,
                        LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS));
  EXPECT_FALSE(
      iree_string_view_is_empty(candidate.compile_report.target_bundle_name));
  EXPECT_FALSE(
      iree_string_view_is_empty(candidate.compile_report.lowered_symbol));
  EXPECT_GT(candidate.compile_report.schedule_node_count, 0u);
  EXPECT_GT(candidate.compile_report.scheduled_node_count, 0u);
  EXPECT_GT(candidate.compile_report.allocation_assignment_count, 0u);
  EXPECT_GT(candidate.compile_report.emitted_instruction_count, 0u);
  EXPECT_GT(candidate.compile_report.emitted_code_byte_count, 0u);
  EXPECT_GE(candidate.compile_report.emitted_code_storage_byte_count,
            candidate.compile_report.emitted_code_byte_count);
  EXPECT_GT(candidate.compile_report.pressure_rows.count, 0u);
  ASSERT_NE(candidate.compile_report.pressure_rows.head, nullptr);
  const auto* pressure_rows =
      static_cast<const loom_target_compile_report_pressure_row_t*>(
          loom_target_compile_report_vec_const_rows(
              candidate.compile_report.pressure_rows.head));
  EXPECT_GT(pressure_rows[0].peak_live_units, 0u);
  EXPECT_EQ(candidate.compile_report.source_low_selected_op_count, 0u);
  EXPECT_EQ(candidate.compile_report.source_low_emitted_op_count, 0u);
  EXPECT_EQ(candidate.compile_report.source_low_rows.count, 0u);
  EXPECT_EQ(candidate.compile_report.artifact_size,
            candidate.archive.data_length);
  EXPECT_EQ(report.artifact_kind, candidate.compile_report.artifact_kind);
  EXPECT_EQ(report.artifact_size, candidate.compile_report.artifact_size);
  EXPECT_EQ(report.pressure_rows.count,
            candidate.compile_report.pressure_rows.count);
  EXPECT_EQ(report.source_low_rows.count, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  EXPECT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 2u);
  iree_vm_ImportFunctionDef_vec_t imported_functions =
      iree_vm_BytecodeModuleDef_imported_functions(module_def);
  EXPECT_EQ(iree_vm_ImportFunctionDef_vec_len(imported_functions), 0u);
  iree_vm_ExportFunctionDef_vec_t exported_functions =
      iree_vm_BytecodeModuleDef_exported_functions(module_def);
  ASSERT_EQ(iree_vm_ExportFunctionDef_vec_len(exported_functions), 1u);
  iree_vm_ExportFunctionDef_table_t export_def =
      iree_vm_ExportFunctionDef_vec_at(exported_functions, 0);
  EXPECT_TRUE(FlatbufferStringEquals(
      iree_vm_ExportFunctionDef_local_name(export_def), IREE_SV("branchy")));
  EXPECT_EQ(iree_vm_ExportFunctionDef_internal_ordinal(export_def), 1);

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_target_compile_report_deinitialize(&report);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithNullReportAllocator) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_null());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS;
  options.report = &report;

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);
  EXPECT_EQ(candidate.compile_report.status_code, IREE_STATUS_OK);
  EXPECT_TRUE(
      iree_all_bits_set(candidate.compile_report.detail_flags,
                        LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE));
  EXPECT_TRUE(iree_all_bits_set(candidate.compile_report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE));
  EXPECT_TRUE(
      iree_all_bits_set(candidate.compile_report.detail_flags,
                        LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS));
  EXPECT_GT(candidate.compile_report.schedule_node_count, 0u);
  EXPECT_EQ(candidate.compile_report.artifact_size,
            candidate.archive.data_length);
  EXPECT_EQ(candidate.compile_report.pressure_rows.count, 0u);
  EXPECT_EQ(candidate.compile_report.pressure_rows.head, nullptr);
  EXPECT_EQ(report.detail_flags, candidate.compile_report.detail_flags);
  EXPECT_EQ(report.artifact_size, candidate.compile_report.artifact_size);
  EXPECT_EQ(report.pressure_rows.count, 0u);
  EXPECT_EQ(report.pressure_rows.head, nullptr);

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_target_compile_report_deinitialize(&report);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithoutReport) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);
  EXPECT_EQ(candidate.compile_report.detail_flags,
            LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE);
  EXPECT_EQ(candidate.compile_report.artifact_size, 0u);
  EXPECT_EQ(candidate.compile_report.pressure_rows.count, 0u);
  EXPECT_EQ(candidate.compile_report.source_low_rows.count, 0u);

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithImport) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmImportSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  EXPECT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 1u);
  iree_vm_ImportFunctionDef_vec_t imported_functions =
      iree_vm_BytecodeModuleDef_imported_functions(module_def);
  ASSERT_EQ(iree_vm_ImportFunctionDef_vec_len(imported_functions), 1u);
  iree_vm_ImportFunctionDef_table_t import_def =
      iree_vm_ImportFunctionDef_vec_at(imported_functions, 0);
  EXPECT_TRUE(
      FlatbufferStringEquals(iree_vm_ImportFunctionDef_full_name(import_def),
                             IREE_SV("hal.buffer.length")));
  iree_vm_ExportFunctionDef_vec_t exported_functions =
      iree_vm_BytecodeModuleDef_exported_functions(module_def);
  ASSERT_EQ(iree_vm_ExportFunctionDef_vec_len(exported_functions), 1u);
  iree_vm_ExportFunctionDef_table_t export_def =
      iree_vm_ExportFunctionDef_vec_at(exported_functions, 0);
  EXPECT_TRUE(
      FlatbufferStringEquals(iree_vm_ExportFunctionDef_local_name(export_def),
                             IREE_SV("length_identity")));

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithWideScalars) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmWideSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  constexpr iree_vm_FeatureBits_enum_t kFeatureRequirements =
      (iree_vm_FeatureBits_enum_t)(iree_vm_FeatureBits_EXT_F32 |
                                   iree_vm_FeatureBits_EXT_F64);
  EXPECT_EQ(iree_vm_BytecodeModuleDef_requirements(module_def),
            kFeatureRequirements);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  ASSERT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 1u);
  iree_vm_FunctionDescriptor_struct_t function_descriptor =
      iree_vm_FunctionDescriptor_vec_at(function_descriptors, 0);
  EXPECT_EQ(function_descriptor->requirements, kFeatureRequirements);
  iree_vm_FunctionSignatureDef_vec_t function_signatures =
      iree_vm_BytecodeModuleDef_function_signatures(module_def);
  ASSERT_EQ(iree_vm_FunctionSignatureDef_vec_len(function_signatures), 1u);
  iree_vm_FunctionSignatureDef_table_t signature_def =
      iree_vm_FunctionSignatureDef_vec_at(function_signatures, 0);
  EXPECT_TRUE(FlatbufferStringEquals(
      iree_vm_FunctionSignatureDef_calling_convention(signature_def),
      IREE_SV("0IIffF_IfFi")));

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithWideAbiLayout) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmWideAbiLayoutSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  ASSERT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 1u);
  iree_vm_FunctionDescriptor_struct_t function_descriptor =
      iree_vm_FunctionDescriptor_vec_at(function_descriptors, 0);
  EXPECT_GT(function_descriptor->bytecode_length, 0);
  EXPECT_EQ(function_descriptor->i32_register_count, 10);
  EXPECT_EQ(function_descriptor->ref_register_count, 0);
  iree_vm_FunctionSignatureDef_vec_t function_signatures =
      iree_vm_BytecodeModuleDef_function_signatures(module_def);
  ASSERT_EQ(iree_vm_FunctionSignatureDef_vec_len(function_signatures), 1u);
  iree_vm_FunctionSignatureDef_table_t signature_def =
      iree_vm_FunctionSignatureDef_vec_at(function_signatures, 0);
  EXPECT_TRUE(FlatbufferStringEquals(
      iree_vm_FunctionSignatureDef_calling_convention(signature_def),
      IREE_SV("0IiIfF_IiIfF")));

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithRefs) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmRefSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  ASSERT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 1u);
  iree_vm_FunctionDescriptor_struct_t function_descriptor =
      iree_vm_FunctionDescriptor_vec_at(function_descriptors, 0);
  EXPECT_GT(function_descriptor->bytecode_length, 0);
  EXPECT_EQ(function_descriptor->i32_register_count, 0);
  EXPECT_EQ(function_descriptor->ref_register_count, 2);
  iree_vm_FunctionSignatureDef_vec_t function_signatures =
      iree_vm_BytecodeModuleDef_function_signatures(module_def);
  ASSERT_EQ(iree_vm_FunctionSignatureDef_vec_len(function_signatures), 1u);
  iree_vm_FunctionSignatureDef_table_t signature_def =
      iree_vm_FunctionSignatureDef_vec_at(function_signatures, 0);
  EXPECT_TRUE(FlatbufferStringEquals(
      iree_vm_FunctionSignatureDef_calling_convention(signature_def),
      IREE_SV("0r_r")));

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithRefBranch) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmRefBranchSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  ASSERT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 1u);
  iree_vm_FunctionDescriptor_struct_t function_descriptor =
      iree_vm_FunctionDescriptor_vec_at(function_descriptors, 0);
  EXPECT_GT(function_descriptor->bytecode_length, 0);
  EXPECT_EQ(function_descriptor->i32_register_count, 0);
  EXPECT_EQ(function_descriptor->ref_register_count, 1);
  ExpectSingleBranchRemapCount(module_def, function_descriptor,
                               /*expected_remap_count=*/0);

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithBufferOps) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmBufferSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  constexpr iree_vm_FeatureBits_enum_t kFeatureRequirements =
      (iree_vm_FeatureBits_enum_t)(iree_vm_FeatureBits_EXT_F32 |
                                   iree_vm_FeatureBits_EXT_F64);
  EXPECT_EQ(iree_vm_BytecodeModuleDef_requirements(module_def),
            kFeatureRequirements);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  ASSERT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 1u);
  iree_vm_FunctionDescriptor_struct_t function_descriptor =
      iree_vm_FunctionDescriptor_vec_at(function_descriptors, 0);
  EXPECT_EQ(function_descriptor->requirements, kFeatureRequirements);
  EXPECT_EQ(function_descriptor->ref_register_count, 1);
  EXPECT_GT(function_descriptor->i32_register_count, 0);
  ExpectFunctionBytecodeContainsOpcodes(module_def, function_descriptor,
                                        {
                                            0x6E,
                                            0x62,
                                            0x63,
                                            0x64,
                                            0x65,
                                            0x66,
                                            0x67,
                                            0xE033,
                                            0xE139,
                                            0x68,
                                            0x69,
                                            0x6A,
                                            0x6B,
                                            0xE034,
                                            0xE13A,
                                        });

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, ExecuteVmArchiveCandidateWithHostBuffer) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmBufferExecutionSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));

  iree_vm_instance_t* instance = nullptr;
  iree_vm_module_t* module = nullptr;
  iree_vm_context_t* context = nullptr;
  iree_vm_buffer_t* buffer = nullptr;
  iree_vm_list_t* inputs = nullptr;
  iree_vm_list_t* outputs = nullptr;

  IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                         iree_allocator_system(), &instance));
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      iree_make_const_byte_span(candidate.archive.data,
                                candidate.archive.data_length),
      iree_allocator_null(), iree_allocator_system(), &module));
  iree_vm_module_t* modules[] = {module};
  IREE_ASSERT_OK(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
      iree_allocator_system(), &context));

  constexpr iree_host_size_t kBufferLength = 40;
  uint8_t initial_data[kBufferLength] = {};
  const uint16_t initial_i16 = 0xFF80u;
  const int32_t initial_i32 = 0x01020304;
  const int64_t initial_i64 = 0x0102030405060708ll;
  const float initial_f32 = 1.25f;
  const double initial_f64 = 2.5;
  initial_data[0] = 0xFEu;
  std::memcpy(initial_data + 2, &initial_i16, sizeof(initial_i16));
  std::memcpy(initial_data + 4, &initial_i32, sizeof(initial_i32));
  std::memcpy(initial_data + 8, &initial_i64, sizeof(initial_i64));
  std::memcpy(initial_data + 16, &initial_f32, sizeof(initial_f32));
  std::memcpy(initial_data + 24, &initial_f64, sizeof(initial_f64));
  IREE_ASSERT_OK(iree_vm_buffer_create(
      IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_HOST,
      kBufferLength, /*alignment=*/8, iree_allocator_system(), &buffer));
  std::memcpy(iree_vm_buffer_data(buffer), initial_data, sizeof(initial_data));

  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 5,
                                     iree_allocator_system(), &inputs));
  iree_vm_ref_t buffer_ref = iree_vm_buffer_retain_ref(buffer);
  IREE_ASSERT_OK(iree_vm_list_push_ref_retain(inputs, &buffer_ref));
  iree_vm_ref_release(&buffer_ref);
  iree_vm_value_t input_i32 = iree_vm_value_make_i32(0x12345678);
  iree_vm_value_t input_i64 = iree_vm_value_make_i64(0x102030405060708ll);
  iree_vm_value_t input_f32 = iree_vm_value_make_f32(9.5f);
  iree_vm_value_t input_f64 = iree_vm_value_make_f64(123.25);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs, &input_i32));
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs, &input_i64));
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs, &input_f32));
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs, &input_f64));

  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 9,
                                     iree_allocator_system(), &outputs));
  iree_vm_function_t function = {};
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT, IREE_SV("buffer_execute"),
      &function));
  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                /*policy=*/nullptr, inputs, outputs,
                                iree_allocator_system()));

  ASSERT_EQ(iree_vm_list_size(outputs), 9u);
  ExpectOutputI64(outputs, 0, kBufferLength);
  ExpectOutputI32(outputs, 1, 254);
  ExpectOutputI32(outputs, 2, -2);
  ExpectOutputI32(outputs, 3, 65408);
  ExpectOutputI32(outputs, 4, -128);
  ExpectOutputI32(outputs, 5, initial_i32);
  ExpectOutputI64(outputs, 6, initial_i64);
  ExpectOutputF32(outputs, 7, initial_f32);
  ExpectOutputF64(outputs, 8, initial_f64);

  const uint8_t* buffer_data = iree_vm_buffer_data(buffer);
  uint8_t stored_i8 = 0;
  uint16_t stored_i16 = 0;
  int32_t stored_i32 = 0;
  int64_t stored_i64 = 0;
  float stored_f32 = 0.0f;
  double stored_f64 = 0.0;
  std::memcpy(&stored_i8, buffer_data + 0, sizeof(stored_i8));
  std::memcpy(&stored_i16, buffer_data + 2, sizeof(stored_i16));
  std::memcpy(&stored_i32, buffer_data + 4, sizeof(stored_i32));
  std::memcpy(&stored_i64, buffer_data + 8, sizeof(stored_i64));
  std::memcpy(&stored_f32, buffer_data + 16, sizeof(stored_f32));
  std::memcpy(&stored_f64, buffer_data + 24, sizeof(stored_f64));
  EXPECT_EQ(stored_i8, (uint8_t)input_i32.i32);
  EXPECT_EQ(stored_i16, (uint16_t)input_i32.i32);
  EXPECT_EQ(stored_i32, input_i32.i32);
  EXPECT_EQ(stored_i64, input_i64.i64);
  EXPECT_FLOAT_EQ(stored_f32, input_f32.f32);
  EXPECT_DOUBLE_EQ(stored_f64, input_f64.f64);

  iree_vm_buffer_t* readonly_buffer = nullptr;
  iree_vm_list_t* readonly_inputs = nullptr;
  iree_vm_list_t* readonly_outputs = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_ORIGIN_HOST, buffer, 0, kBufferLength,
      /*alignment=*/8, iree_allocator_system(), &readonly_buffer));
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 5,
                                     iree_allocator_system(),
                                     &readonly_inputs));
  iree_vm_ref_t readonly_buffer_ref =
      iree_vm_buffer_retain_ref(readonly_buffer);
  IREE_ASSERT_OK(
      iree_vm_list_push_ref_retain(readonly_inputs, &readonly_buffer_ref));
  iree_vm_ref_release(&readonly_buffer_ref);
  IREE_ASSERT_OK(iree_vm_list_push_value(readonly_inputs, &input_i32));
  IREE_ASSERT_OK(iree_vm_list_push_value(readonly_inputs, &input_i64));
  IREE_ASSERT_OK(iree_vm_list_push_value(readonly_inputs, &input_f32));
  IREE_ASSERT_OK(iree_vm_list_push_value(readonly_inputs, &input_f64));
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 9,
                                     iree_allocator_system(),
                                     &readonly_outputs));
  iree_status_t readonly_status =
      iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                     /*policy=*/nullptr, readonly_inputs, readonly_outputs,
                     iree_allocator_system());
  EXPECT_THAT(
      iree::Status(std::move(readonly_status)),
      iree::testing::status::StatusIs(iree::StatusCode::kPermissionDenied));

  iree_vm_list_release(readonly_outputs);
  iree_vm_list_release(readonly_inputs);
  iree_vm_buffer_release(readonly_buffer);
  iree_vm_list_release(outputs);
  iree_vm_list_release(inputs);
  iree_vm_buffer_release(buffer);
  iree_vm_context_release(context);
  iree_vm_module_release(module);
  iree_vm_instance_release(instance);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, ExecuteSourceVmArchiveCandidateWithImports) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kSourceVmOrchestrationSource), &run_module));
  IREE_ASSERT_OK(LowerToPreparedLow(&run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  EXPECT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 2u);
  iree_vm_ImportFunctionDef_vec_t imported_functions =
      iree_vm_BytecodeModuleDef_imported_functions(module_def);
  ASSERT_EQ(iree_vm_ImportFunctionDef_vec_len(imported_functions), 3u);
  ExpectImportedFunction(imported_functions, 0,
                         IREE_SV("loom_test.buffer_peek_i32"),
                         IREE_SV("0rI_i"));
  ExpectImportedFunction(imported_functions, 1,
                         IREE_SV("loom_test.retain_buffer"), IREE_SV("0r_r"));
  ExpectImportedFunction(imported_functions, 2, IREE_SV("loom_test.scale_i32"),
                         IREE_SV("0i_i"));
  iree_vm_ExportFunctionDef_vec_t exported_functions =
      iree_vm_BytecodeModuleDef_exported_functions(module_def);
  ASSERT_EQ(iree_vm_ExportFunctionDef_vec_len(exported_functions), 1u);
  iree_vm_ExportFunctionDef_table_t export_def =
      iree_vm_ExportFunctionDef_vec_at(exported_functions, 0);
  EXPECT_TRUE(
      FlatbufferStringEquals(iree_vm_ExportFunctionDef_local_name(export_def),
                             IREE_SV("orchestrate")));
  EXPECT_EQ(iree_vm_ExportFunctionDef_internal_ordinal(export_def), 1);

  loom_run_vm_runtime_t runtime = {};
  iree_vm_module_t* import_module = nullptr;
  loom_run_vm_prepared_candidate_t prepared_candidate = {};
  loom_run_vm_invocation_plan_t plan = {};
  loom_run_vm_iteration_t iteration = {};
  iree_vm_buffer_t* buffer = nullptr;

  IREE_ASSERT_OK(
      loom_run_vm_runtime_initialize(iree_allocator_system(), &runtime));
  IREE_ASSERT_OK(CreateLoomTestImportModule(
      runtime.instance, iree_allocator_system(), &import_module));

  iree_vm_module_t* dependency_modules[] = {import_module};
  loom_run_vm_prepared_candidate_options_t candidate_options = {};
  loom_run_vm_prepared_candidate_options_initialize(&candidate_options);
  candidate_options.function_name = IREE_SV("orchestrate");
  candidate_options.dependency_modules = dependency_modules;
  candidate_options.dependency_module_count =
      IREE_ARRAYSIZE(dependency_modules);
  IREE_ASSERT_OK(loom_run_vm_prepared_candidate_prepare(
      &runtime, &candidate.archive, &candidate_options, iree_allocator_system(),
      &prepared_candidate));

  constexpr iree_host_size_t kBufferLength = 32;
  constexpr int32_t kElementValue = 1234567;
  constexpr int64_t kElementOffset = 3;
  IREE_ASSERT_OK(iree_vm_buffer_create(
      IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_HOST,
      kBufferLength, /*alignment=*/8, iree_allocator_system(), &buffer));
  std::memcpy(iree_vm_buffer_data(buffer) + sizeof(int32_t) * kElementOffset,
              &kElementValue, sizeof(kElementValue));

  loom_run_vm_invocation_plan_initialize(&plan);
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 3,
                                     iree_allocator_system(), &plan.inputs));
  iree_vm_ref_t buffer_ref = iree_vm_buffer_retain_ref(buffer);
  IREE_ASSERT_OK(iree_vm_list_push_ref_retain(plan.inputs, &buffer_ref));
  iree_vm_ref_release(&buffer_ref);
  iree_vm_value_t input_value = iree_vm_value_make_i32(7);
  iree_vm_value_t input_offset = iree_vm_value_make_i64(kElementOffset);
  IREE_ASSERT_OK(iree_vm_list_push_value(plan.inputs, &input_value));
  IREE_ASSERT_OK(iree_vm_list_push_value(plan.inputs, &input_offset));

  IREE_ASSERT_OK(loom_run_vm_invocation_invoke_plan(
      &prepared_candidate, &plan, iree_allocator_system(), &iteration));
  ASSERT_EQ(iree_vm_list_size(iteration.outputs), 5u);
  ExpectOutputI32(iteration.outputs, 0, 42);
  ExpectOutputI32(iteration.outputs, 1, kElementValue);
  ExpectOutputI32(iteration.outputs, 2, kElementValue);
  ExpectOutputI64(iteration.outputs, 3, kBufferLength);
  iree_vm_ref_t output_ref = iree_vm_ref_null();
  IREE_ASSERT_OK(
      iree_vm_list_get_ref_retain(iteration.outputs, 4, &output_ref));
  EXPECT_EQ(output_ref.ptr, buffer);
  EXPECT_EQ(output_ref.type, iree_vm_buffer_type());
  iree_vm_ref_release(&output_ref);

  loom_run_vm_iteration_deinitialize(&iteration);
  loom_run_vm_invocation_plan_deinitialize(&plan);
  iree_vm_buffer_release(buffer);
  loom_run_vm_prepared_candidate_deinitialize(&prepared_candidate);
  iree_vm_module_release(import_module);
  loom_run_vm_runtime_deinitialize(&runtime);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

}  // namespace
}  // namespace loom
