// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_builder.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/spirv/records/target_records.h"
#include "loom/target/emit/spirv/binary_format.h"

namespace {

struct Instruction {
  uint16_t opcode = 0;
  std::vector<uint32_t> operands;
  iree_host_size_t word_offset = 0;
};

std::vector<Instruction> ParseInstructions(
    const loom_spirv_module_binary_t& module) {
  std::vector<Instruction> instructions;
  for (iree_host_size_t offset = 5; offset < module.word_count;) {
    uint32_t header = module.words[offset];
    uint16_t word_count = (uint16_t)(header >> 16);
    uint16_t opcode = (uint16_t)(header & 0xFFFFu);
    IREE_ASSERT_GT(word_count, 0u);
    IREE_ASSERT_LE(offset + word_count, module.word_count);
    Instruction instruction;
    instruction.opcode = opcode;
    instruction.word_offset = offset;
    instruction.operands.assign(module.words + offset + 1,
                                module.words + offset + word_count);
    instructions.push_back(std::move(instruction));
    offset += word_count;
  }
  return instructions;
}

std::string DecodeStringOperand(const std::vector<uint32_t>& operands,
                                iree_host_size_t operand_index,
                                iree_host_size_t* out_next_operand_index) {
  std::string result;
  for (iree_host_size_t i = operand_index; i < operands.size(); ++i) {
    uint32_t word = operands[i];
    for (uint32_t byte_index = 0; byte_index < 4; ++byte_index) {
      char c = (char)((word >> (byte_index * 8)) & 0xFFu);
      if (c == '\0') {
        *out_next_operand_index = i + 1;
        return result;
      }
      result.push_back(c);
    }
  }
  *out_next_operand_index = operands.size();
  return result;
}

bool HasInstruction(const std::vector<Instruction>& instructions,
                    uint16_t opcode, std::initializer_list<uint32_t> operands) {
  for (const Instruction& instruction : instructions) {
    if (instruction.opcode == opcode &&
        instruction.operands == std::vector<uint32_t>(operands)) {
      return true;
    }
  }
  return false;
}

const Instruction* FindInstruction(const std::vector<Instruction>& instructions,
                                   uint16_t opcode,
                                   std::initializer_list<uint32_t> prefix) {
  std::vector<uint32_t> prefix_vector(prefix);
  for (const Instruction& instruction : instructions) {
    if (instruction.opcode != opcode ||
        instruction.operands.size() < prefix_vector.size()) {
      continue;
    }
    bool matches = true;
    for (iree_host_size_t i = 0; i < prefix_vector.size(); ++i) {
      if (instruction.operands[i] != prefix_vector[i]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return &instruction;
    }
  }
  return nullptr;
}

iree_status_t WriteInstruction(loom_spirv_module_builder_t* builder,
                               loom_spirv_module_section_t section,
                               uint16_t opcode,
                               std::initializer_list<uint32_t> operands) {
  return loom_spirv_binary_write_instruction(
      loom_spirv_module_builder_section(builder, section), opcode,
      operands.begin(), operands.size());
}

iree_status_t WriteStringInstruction(
    loom_spirv_module_builder_t* builder, loom_spirv_module_section_t section,
    uint16_t opcode, std::initializer_list<uint32_t> prefix_operands,
    iree_string_view_t string,
    std::initializer_list<uint32_t> suffix_operands) {
  return loom_spirv_binary_write_string_instruction(
      loom_spirv_module_builder_section(builder, section), opcode,
      prefix_operands.begin(), prefix_operands.size(), string,
      suffix_operands.begin(), suffix_operands.size());
}

enum StorageBufferAddIds : uint32_t {
  kVoidType = 1,
  kFunctionType = 2,
  kU32Type = 3,
  kI32Type = 4,
  kV3U32Type = 5,
  kU32Zero = 6,
  kRuntimeArrayI32Type = 7,
  kBufferStructType = 8,
  kPtrStorageBufferStructType = 9,
  kPtrStorageBufferI32Type = 10,
  kPtrInputV3U32Type = 11,
  kPtrInputU32Type = 12,
  kGlobalInvocationId = 13,
  kInput0 = 14,
  kInput1 = 15,
  kOutput = 16,
  kMainFunction = 17,
  kEntryLabel = 18,
  kGlobalInvocationXPointer = 19,
  kGlobalInvocationX = 20,
  kInput0Pointer = 21,
  kInput1Pointer = 22,
  kOutputPointer = 23,
  kInput0Value = 24,
  kInput1Value = 25,
  kSumValue = 26,
  kIdBound = 27,
};

iree_status_t EmitStorageBufferI32Add(loom_spirv_module_builder_t* builder,
                                      iree_string_view_t entry_point_name) {
  loom_spirv_module_builder_require_id_bound(builder, kIdBound);

  IREE_RETURN_IF_ERROR(WriteStringInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ENTRY_POINT, LOOM_SPIRV_OP_ENTRY_POINT,
      {LOOM_SPIRV_EXECUTION_MODEL_GL_COMPUTE, kMainFunction}, entry_point_name,
      {kGlobalInvocationId}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_EXECUTION_MODE,
      LOOM_SPIRV_OP_EXECUTION_MODE,
      {kMainFunction, LOOM_SPIRV_EXECUTION_MODE_LOCAL_SIZE, 1, 1, 1}));

  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION, LOOM_SPIRV_OP_DECORATE,
      {kGlobalInvocationId, LOOM_SPIRV_DECORATION_BUILT_IN,
       LOOM_SPIRV_BUILT_IN_GLOBAL_INVOCATION_ID}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION, LOOM_SPIRV_OP_DECORATE,
      {kRuntimeArrayI32Type, LOOM_SPIRV_DECORATION_ARRAY_STRIDE, 4}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION,
      LOOM_SPIRV_OP_MEMBER_DECORATE,
      {kBufferStructType, 0, LOOM_SPIRV_DECORATION_OFFSET, 0}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION, LOOM_SPIRV_OP_DECORATE,
      {kBufferStructType, LOOM_SPIRV_DECORATION_BLOCK}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION, LOOM_SPIRV_OP_DECORATE,
      {kInput0, LOOM_SPIRV_DECORATION_DESCRIPTOR_SET, 0}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION, LOOM_SPIRV_OP_DECORATE,
      {kInput0, LOOM_SPIRV_DECORATION_BINDING, 0}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION, LOOM_SPIRV_OP_DECORATE,
      {kInput1, LOOM_SPIRV_DECORATION_DESCRIPTOR_SET, 0}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION, LOOM_SPIRV_OP_DECORATE,
      {kInput1, LOOM_SPIRV_DECORATION_BINDING, 1}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION, LOOM_SPIRV_OP_DECORATE,
      {kOutput, LOOM_SPIRV_DECORATION_DESCRIPTOR_SET, 0}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_ANNOTATION, LOOM_SPIRV_OP_DECORATE,
      {kOutput, LOOM_SPIRV_DECORATION_BINDING, 2}));

  IREE_RETURN_IF_ERROR(WriteInstruction(builder,
                                        LOOM_SPIRV_MODULE_SECTION_DECLARATION,
                                        LOOM_SPIRV_OP_TYPE_VOID, {kVoidType}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION,
      LOOM_SPIRV_OP_TYPE_FUNCTION, {kFunctionType, kVoidType}));
  IREE_RETURN_IF_ERROR(
      WriteInstruction(builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION,
                       LOOM_SPIRV_OP_TYPE_INT, {kU32Type, 32, 0}));
  IREE_RETURN_IF_ERROR(
      WriteInstruction(builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION,
                       LOOM_SPIRV_OP_TYPE_INT, {kI32Type, 32, 1}));
  IREE_RETURN_IF_ERROR(
      WriteInstruction(builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION,
                       LOOM_SPIRV_OP_TYPE_VECTOR, {kV3U32Type, kU32Type, 3}));
  IREE_RETURN_IF_ERROR(
      WriteInstruction(builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION,
                       LOOM_SPIRV_OP_CONSTANT, {kU32Type, kU32Zero, 0}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION,
      LOOM_SPIRV_OP_TYPE_RUNTIME_ARRAY, {kRuntimeArrayI32Type, kI32Type}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION, LOOM_SPIRV_OP_TYPE_STRUCT,
      {kBufferStructType, kRuntimeArrayI32Type}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION,
      LOOM_SPIRV_OP_TYPE_POINTER,
      {kPtrStorageBufferStructType, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER,
       kBufferStructType}));
  IREE_RETURN_IF_ERROR(
      WriteInstruction(builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION,
                       LOOM_SPIRV_OP_TYPE_POINTER,
                       {kPtrStorageBufferI32Type,
                        LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, kI32Type}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION,
      LOOM_SPIRV_OP_TYPE_POINTER,
      {kPtrInputV3U32Type, LOOM_SPIRV_STORAGE_CLASS_INPUT, kV3U32Type}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION,
      LOOM_SPIRV_OP_TYPE_POINTER,
      {kPtrInputU32Type, LOOM_SPIRV_STORAGE_CLASS_INPUT, kU32Type}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION, LOOM_SPIRV_OP_VARIABLE,
      {kPtrInputV3U32Type, kGlobalInvocationId,
       LOOM_SPIRV_STORAGE_CLASS_INPUT}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION, LOOM_SPIRV_OP_VARIABLE,
      {kPtrStorageBufferStructType, kInput0,
       LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION, LOOM_SPIRV_OP_VARIABLE,
      {kPtrStorageBufferStructType, kInput1,
       LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_DECLARATION, LOOM_SPIRV_OP_VARIABLE,
      {kPtrStorageBufferStructType, kOutput,
       LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER}));

  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION, LOOM_SPIRV_OP_FUNCTION,
      {kVoidType, kMainFunction, LOOM_SPIRV_FUNCTION_CONTROL_NONE,
       kFunctionType}));
  IREE_RETURN_IF_ERROR(WriteInstruction(builder,
                                        LOOM_SPIRV_MODULE_SECTION_FUNCTION,
                                        LOOM_SPIRV_OP_LABEL, {kEntryLabel}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION, LOOM_SPIRV_OP_ACCESS_CHAIN,
      {kPtrInputU32Type, kGlobalInvocationXPointer, kGlobalInvocationId,
       kU32Zero}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION, LOOM_SPIRV_OP_LOAD,
      {kU32Type, kGlobalInvocationX, kGlobalInvocationXPointer}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION, LOOM_SPIRV_OP_ACCESS_CHAIN,
      {kPtrStorageBufferI32Type, kInput0Pointer, kInput0, kU32Zero,
       kGlobalInvocationX}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION, LOOM_SPIRV_OP_ACCESS_CHAIN,
      {kPtrStorageBufferI32Type, kInput1Pointer, kInput1, kU32Zero,
       kGlobalInvocationX}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION, LOOM_SPIRV_OP_ACCESS_CHAIN,
      {kPtrStorageBufferI32Type, kOutputPointer, kOutput, kU32Zero,
       kGlobalInvocationX}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION, LOOM_SPIRV_OP_LOAD,
      {kI32Type, kInput0Value, kInput0Pointer}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION, LOOM_SPIRV_OP_LOAD,
      {kI32Type, kInput1Value, kInput1Pointer}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION, LOOM_SPIRV_OP_I_ADD,
      {kI32Type, kSumValue, kInput0Value, kInput1Value}));
  IREE_RETURN_IF_ERROR(
      WriteInstruction(builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION,
                       LOOM_SPIRV_OP_STORE, {kOutputPointer, kSumValue}));
  IREE_RETURN_IF_ERROR(WriteInstruction(
      builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION, LOOM_SPIRV_OP_RETURN, {}));
  return WriteInstruction(builder, LOOM_SPIRV_MODULE_SECTION_FUNCTION,
                          LOOM_SPIRV_OP_FUNCTION_END, {});
}

TEST(SpirvModuleBuilderTest, BuildsStorageBufferI32AddModule) {
  loom_spirv_module_builder_t builder;
  IREE_ASSERT_OK(loom_spirv_module_builder_initialize(
      &loom_spirv_low_target_bundle_vulkan1_3, iree_allocator_system(),
      &builder));
  IREE_ASSERT_OK(EmitStorageBufferI32Add(&builder, IREE_SV("vector_add_i32")));

  loom_spirv_module_binary_t module;
  IREE_ASSERT_OK(loom_spirv_module_builder_finalize(&builder, &module));
  loom_spirv_module_builder_deinitialize(&builder);

  ASSERT_GE(module.word_count, 5u);
  EXPECT_EQ(module.words[0], LOOM_SPIRV_MAGIC_NUMBER);
  EXPECT_EQ(module.words[1], UINT32_C(0x00010300));
  EXPECT_EQ(module.words[2], LOOM_SPIRV_GENERATOR_LOOM);
  EXPECT_EQ(module.words[3], kIdBound);
  EXPECT_EQ(module.words[4], LOOM_SPIRV_SCHEMA_RESERVED);
  EXPECT_EQ(loom_spirv_module_binary_byte_span(&module).data_length,
            module.word_count * sizeof(uint32_t));

  std::vector<Instruction> instructions = ParseInstructions(module);
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_CAPABILITY,
                             {LOOM_SPIRV_CAPABILITY_SHADER}));
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_CAPABILITY,
                             {LOOM_SPIRV_CAPABILITY_VULKAN_MEMORY_MODEL}));
  EXPECT_TRUE(HasInstruction(
      instructions, LOOM_SPIRV_OP_CAPABILITY,
      {LOOM_SPIRV_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES}));
  EXPECT_TRUE(
      HasInstruction(instructions, LOOM_SPIRV_OP_MEMORY_MODEL,
                     {LOOM_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64,
                      LOOM_SPIRV_MEMORY_MODEL_VULKAN}));

  const Instruction* vulkan_memory_extension =
      FindInstruction(instructions, LOOM_SPIRV_OP_EXTENSION, {});
  ASSERT_NE(vulkan_memory_extension, nullptr);
  iree_host_size_t next_operand_index = 0;
  EXPECT_EQ(DecodeStringOperand(vulkan_memory_extension->operands, 0,
                                &next_operand_index),
            "SPV_KHR_vulkan_memory_model");

  const Instruction* physical_buffer_extension = nullptr;
  for (const Instruction& instruction : instructions) {
    if (instruction.opcode != LOOM_SPIRV_OP_EXTENSION) {
      continue;
    }
    iree_host_size_t next_index = 0;
    if (DecodeStringOperand(instruction.operands, 0, &next_index) ==
        "SPV_KHR_physical_storage_buffer") {
      physical_buffer_extension = &instruction;
      break;
    }
  }
  ASSERT_NE(physical_buffer_extension, nullptr);

  const Instruction* entry_point =
      FindInstruction(instructions, LOOM_SPIRV_OP_ENTRY_POINT,
                      {LOOM_SPIRV_EXECUTION_MODEL_GL_COMPUTE, kMainFunction});
  ASSERT_NE(entry_point, nullptr);
  EXPECT_EQ(DecodeStringOperand(entry_point->operands, 2, &next_operand_index),
            "vector_add_i32");
  ASSERT_EQ(next_operand_index + 1, entry_point->operands.size());
  EXPECT_EQ(entry_point->operands[next_operand_index], kGlobalInvocationId);

  EXPECT_TRUE(HasInstruction(
      instructions, LOOM_SPIRV_OP_EXECUTION_MODE,
      {kMainFunction, LOOM_SPIRV_EXECUTION_MODE_LOCAL_SIZE, 1, 1, 1}));
  EXPECT_TRUE(
      HasInstruction(instructions, LOOM_SPIRV_OP_DECORATE,
                     {kGlobalInvocationId, LOOM_SPIRV_DECORATION_BUILT_IN,
                      LOOM_SPIRV_BUILT_IN_GLOBAL_INVOCATION_ID}));
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_DECORATE,
                             {kInput0, LOOM_SPIRV_DECORATION_BINDING, 0}));
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_DECORATE,
                             {kInput1, LOOM_SPIRV_DECORATION_BINDING, 1}));
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_DECORATE,
                             {kOutput, LOOM_SPIRV_DECORATION_BINDING, 2}));
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_TYPE_RUNTIME_ARRAY,
                             {kRuntimeArrayI32Type, kI32Type}));
  EXPECT_TRUE(
      HasInstruction(instructions, LOOM_SPIRV_OP_TYPE_POINTER,
                     {kPtrStorageBufferI32Type,
                      LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, kI32Type}));
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_ACCESS_CHAIN,
                             {kPtrStorageBufferI32Type, kInput0Pointer, kInput0,
                              kU32Zero, kGlobalInvocationX}));
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_ACCESS_CHAIN,
                             {kPtrStorageBufferI32Type, kInput1Pointer, kInput1,
                              kU32Zero, kGlobalInvocationX}));
  EXPECT_TRUE(
      HasInstruction(instructions, LOOM_SPIRV_OP_I_ADD,
                     {kI32Type, kSumValue, kInput0Value, kInput1Value}));
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_STORE,
                             {kOutputPointer, kSumValue}));

  loom_spirv_module_binary_deinitialize(&module, iree_allocator_system());
}

TEST(SpirvModuleBuilderTest, EmitsRawBdaHalKernelPreamble) {
  const loom_target_snapshot_t snapshot = {
      /*.name=*/IREE_SVL("spirv-vulkan1.3"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_SPIRV,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
  };
  const loom_target_export_plan_t export_plan = {
      /*.name=*/IREE_SVL("hal-kernel"),
      /*.export_symbol=*/{},
      /*.abi_kind=*/LOOM_TARGET_ABI_HAL_KERNEL,
  };
  const loom_target_config_t config = {
      /*.name=*/IREE_SVL("spirv.logical.core"),
      /*.contract_set_key=*/{},
      /*.contract_feature_bits=*/LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA,
  };
  const loom_target_bundle_t target = {
      /*.name=*/IREE_SVL("spirv-vulkan1.3-hal"),
      /*.snapshot=*/&snapshot,
      /*.export_plan=*/&export_plan,
      /*.config=*/&config,
  };

  loom_spirv_module_builder_t builder;
  IREE_ASSERT_OK(loom_spirv_module_builder_initialize(
      &target, iree_allocator_system(), &builder));

  loom_spirv_module_binary_t module;
  IREE_ASSERT_OK(loom_spirv_module_builder_finalize(&builder, &module));
  loom_spirv_module_builder_deinitialize(&builder);

  std::vector<Instruction> instructions = ParseInstructions(module);
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_CAPABILITY,
                             {LOOM_SPIRV_CAPABILITY_SHADER}));
  EXPECT_TRUE(HasInstruction(
      instructions, LOOM_SPIRV_OP_CAPABILITY,
      {LOOM_SPIRV_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES}));
  EXPECT_FALSE(HasInstruction(instructions, LOOM_SPIRV_OP_CAPABILITY,
                              {LOOM_SPIRV_CAPABILITY_VULKAN_MEMORY_MODEL}));
  EXPECT_TRUE(
      HasInstruction(instructions, LOOM_SPIRV_OP_MEMORY_MODEL,
                     {LOOM_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64,
                      LOOM_SPIRV_MEMORY_MODEL_GLSL450}));

  bool has_physical_storage_buffer_extension = false;
  bool has_vulkan_memory_model_extension = false;
  for (const Instruction& instruction : instructions) {
    if (instruction.opcode != LOOM_SPIRV_OP_EXTENSION) {
      continue;
    }
    iree_host_size_t next_index = 0;
    const std::string extension =
        DecodeStringOperand(instruction.operands, 0, &next_index);
    has_physical_storage_buffer_extension |=
        extension == "SPV_KHR_physical_storage_buffer";
    has_vulkan_memory_model_extension |=
        extension == "SPV_KHR_vulkan_memory_model";
  }
  EXPECT_TRUE(has_physical_storage_buffer_extension);
  EXPECT_FALSE(has_vulkan_memory_model_extension);

  loom_spirv_module_binary_deinitialize(&module, iree_allocator_system());
}

TEST(SpirvModuleBuilderTest, EmitsCooperativeMatrixRawBdaHalKernelPreamble) {
  const loom_target_snapshot_t snapshot = {
      /*.name=*/IREE_SVL("spirv-vulkan1.3"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_SPIRV,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
  };
  const loom_target_export_plan_t export_plan = {
      /*.name=*/IREE_SVL("hal-kernel"),
      /*.export_symbol=*/{},
      /*.abi_kind=*/LOOM_TARGET_ABI_HAL_KERNEL,
  };
  const loom_target_config_t config = {
      /*.name=*/IREE_SVL("spirv.logical.core"),
      /*.contract_set_key=*/{},
      /*.contract_feature_bits=*/LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA,
  };
  const loom_target_bundle_t target = {
      /*.name=*/IREE_SVL("spirv-vulkan1.3-hal-coop"),
      /*.snapshot=*/&snapshot,
      /*.export_plan=*/&export_plan,
      /*.config=*/&config,
  };

  loom_spirv_module_builder_t builder;
  IREE_ASSERT_OK(loom_spirv_module_builder_initialize(
      &target, iree_allocator_system(), &builder));
  loom_spirv_module_builder_require_feature_bits(
      &builder, LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR);

  loom_spirv_module_binary_t module;
  IREE_ASSERT_OK(loom_spirv_module_builder_finalize(&builder, &module));
  loom_spirv_module_builder_deinitialize(&builder);

  std::vector<Instruction> instructions = ParseInstructions(module);
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_CAPABILITY,
                             {LOOM_SPIRV_CAPABILITY_SHADER}));
  EXPECT_TRUE(HasInstruction(
      instructions, LOOM_SPIRV_OP_CAPABILITY,
      {LOOM_SPIRV_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES}));
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_CAPABILITY,
                             {LOOM_SPIRV_CAPABILITY_COOPERATIVE_MATRIX_KHR}));
  EXPECT_TRUE(HasInstruction(instructions, LOOM_SPIRV_OP_CAPABILITY,
                             {LOOM_SPIRV_CAPABILITY_VULKAN_MEMORY_MODEL}));
  EXPECT_TRUE(
      HasInstruction(instructions, LOOM_SPIRV_OP_MEMORY_MODEL,
                     {LOOM_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64,
                      LOOM_SPIRV_MEMORY_MODEL_VULKAN}));

  bool has_physical_storage_buffer_extension = false;
  bool has_cooperative_matrix_extension = false;
  bool has_vulkan_memory_model_extension = false;
  for (const Instruction& instruction : instructions) {
    if (instruction.opcode != LOOM_SPIRV_OP_EXTENSION) {
      continue;
    }
    iree_host_size_t next_index = 0;
    const std::string extension =
        DecodeStringOperand(instruction.operands, 0, &next_index);
    has_physical_storage_buffer_extension |=
        extension == "SPV_KHR_physical_storage_buffer";
    has_cooperative_matrix_extension |=
        extension == "SPV_KHR_cooperative_matrix";
    has_vulkan_memory_model_extension |=
        extension == "SPV_KHR_vulkan_memory_model";
  }
  EXPECT_TRUE(has_physical_storage_buffer_extension);
  EXPECT_TRUE(has_cooperative_matrix_extension);
  EXPECT_TRUE(has_vulkan_memory_model_extension);

  loom_spirv_module_binary_deinitialize(&module, iree_allocator_system());
}

TEST(SpirvModuleBuilderTest, RejectsNonSpirvTargetBundle) {
  const loom_target_snapshot_t snapshot = {
      /*.name=*/IREE_SVL("not-spirv"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
  };
  const loom_target_export_plan_t export_plan = {
      /*.name=*/IREE_SVL("shader-entry"),
      /*.export_symbol=*/{},
      /*.abi_kind=*/LOOM_TARGET_ABI_SHADER_ENTRY_POINT,
  };
  const loom_target_config_t config = {
      /*.name=*/IREE_SVL("config"),
      /*.contract_set_key=*/{},
      /*.contract_feature_bits=*/LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA,
  };
  const loom_target_bundle_t target = {
      /*.name=*/IREE_SVL("wrong-target"),
      /*.snapshot=*/&snapshot,
      /*.export_plan=*/&export_plan,
      /*.config=*/&config,
  };

  loom_spirv_module_builder_t builder;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_spirv_module_builder_initialize(
                            &target, iree_allocator_system(), &builder));
}

}  // namespace
