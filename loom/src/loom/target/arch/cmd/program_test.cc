// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/program.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/cmd/format.h"

namespace loom {
namespace {

static void StoreLaunchBinding(std::vector<uint8_t>& data,
                               uint32_t table_offset, uint32_t index,
                               loom_cmd_program_buffer_role_t role,
                               uint32_t root_index) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_LAUNCH_BINDING_SIZE;
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_LAUNCH_BINDING_ROLE_OFFSET, role);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_LAUNCH_BINDING_ROOT_INDEX_OFFSET, root_index);
}

static void StoreBufferRef(std::vector<uint8_t>& data, uint32_t table_offset,
                           uint32_t index, loom_cmd_program_buffer_role_t role,
                           uint32_t root_index, uint64_t byte_offset,
                           uint64_t byte_length) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
  iree_unaligned_store_le_u32(record + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET,
                              role);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET, root_index);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET, byte_offset);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET, byte_length);
}

static void StoreEntrySchema(std::vector<uint8_t>& data, uint32_t table_offset,
                             uint32_t index, uint32_t entry_index,
                             uint32_t kind_offset, uint32_t argument_count,
                             uint32_t argument_byte_length) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_ENTRY_SCHEMA_SIZE;
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ENTRY_INDEX_OFFSET, entry_index);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_KIND_OFFSET_OFFSET, kind_offset);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ARGUMENT_COUNT_OFFSET,
      argument_count);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ARGUMENT_BYTE_LENGTH_OFFSET,
      argument_byte_length);
}

static void StoreCommand(std::vector<uint8_t>& data, uint32_t table_offset,
                         uint32_t index, loom_cmd_program_command_kind_t kind,
                         uint32_t argument_offset,
                         uint32_t argument_schema_index, uint32_t operand_0 = 0,
                         uint32_t operand_1 = 0, uint32_t operand_2 = 0,
                         uint32_t operand_3 = 0, uint32_t operand_4 = 0) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_COMMAND_SIZE;
  iree_unaligned_store_le_u32(record + LOOM_CMD_PROGRAM_COMMAND_KIND_OFFSET,
                              (uint32_t)kind);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_OFFSET_OFFSET,
      argument_offset);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_SCHEMA_INDEX_OFFSET,
      argument_schema_index);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_0_OFFSET, operand_0);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_1_OFFSET, operand_1);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_2_OFFSET, operand_2);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_3_OFFSET, operand_3);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_4_OFFSET, operand_4);
}

static void StoreParameterRoot(std::vector<uint8_t>& data,
                               uint32_t table_offset, uint32_t index,
                               uint32_t fixed_buffer_index,
                               uint64_t required_byte_length,
                               uint64_t minimum_alignment) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_PARAMETER_ROOT_SIZE;
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_FIXED_BUFFER_INDEX_OFFSET,
      fixed_buffer_index);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_REQUIRED_BYTE_LENGTH_OFFSET,
      required_byte_length);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_MINIMUM_ALIGNMENT_OFFSET,
      minimum_alignment);
}

static void StoreParameter(std::vector<uint8_t>& data, uint32_t table_offset,
                           uint32_t index, uint32_t key_offset,
                           uint32_t key_length, uint32_t fixed_buffer_index,
                           uint64_t byte_offset, uint64_t byte_length,
                           uint64_t minimum_alignment) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_PARAMETER_SIZE;
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_KEY_OFFSET_OFFSET, key_offset);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_KEY_LENGTH_OFFSET, key_length);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_FIXED_BUFFER_INDEX_OFFSET,
      fixed_buffer_index);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_PARAMETER_BYTE_OFFSET_OFFSET, byte_offset);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_PARAMETER_BYTE_LENGTH_OFFSET, byte_length);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_PARAMETER_MINIMUM_ALIGNMENT_OFFSET,
      minimum_alignment);
}

static constexpr uint32_t kValidArgumentBufferOffset = 4 + 8 + 8 + 8;
static constexpr uint32_t kValidArgumentByteLength =
    kValidArgumentBufferOffset + LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;

static std::vector<uint8_t> BuildValidProgram(uint32_t dependency_count = 0) {
  static constexpr uint32_t kLaunchBindingCount = 2;
  static constexpr uint32_t kBufferRefCount = 2;
  static constexpr uint32_t kEntrySchemaCount = 1;
  static constexpr uint32_t kEntrySchemaKindCount = 5;
  static constexpr uint32_t kArgumentDataLength = kValidArgumentByteLength * 3;
  static constexpr uint32_t kCommandCount = 6;
  static constexpr uint32_t kParameterRootCount = 1;
  static constexpr uint32_t kParameterCount = 2;
  static constexpr char kFirstKey[] = "alpha";
  static constexpr char kSecondKey[] = "blk.3.weight";
  static constexpr uint32_t kFirstKeyLength = sizeof(kFirstKey) - 1;
  static constexpr uint32_t kSecondKeyLength = sizeof(kSecondKey) - 1;
  static constexpr uint32_t kParameterKeyLength =
      kFirstKeyLength + kSecondKeyLength;
  loom_cmd_program_format_layout_t layout = {};
  IREE_CHECK_OK(loom_cmd_program_format_calculate_layout(
      kLaunchBindingCount, kBufferRefCount, dependency_count, kEntrySchemaCount,
      kEntrySchemaKindCount, kArgumentDataLength, kCommandCount,
      kParameterRootCount, kParameterCount, kParameterKeyLength, &layout));
  std::vector<uint8_t> data(layout.total_length, 0);

  memcpy(data.data() + LOOM_CMD_PROGRAM_HEADER_MAGIC_OFFSET,
         LOOM_CMD_PROGRAM_FORMAT_MAGIC, LOOM_CMD_PROGRAM_FORMAT_MAGIC_LENGTH);
  iree_unaligned_store_le_u16(
      data.data() + LOOM_CMD_PROGRAM_HEADER_VERSION_OFFSET,
      LOOM_CMD_PROGRAM_FORMAT_VERSION);
  iree_unaligned_store_le_u16(data.data() + LOOM_CMD_PROGRAM_HEADER_SIZE_OFFSET,
                              LOOM_CMD_PROGRAM_HEADER_SIZE);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_TOTAL_LENGTH_OFFSET,
      layout.total_length);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_FIXED_BUFFER_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_REBINDABLE_BINDING_COUNT_OFFSET, 2);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_EXECUTABLE_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ENTRY_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_COUNT_OFFSET,
      kBufferRefCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_MEMORY_DEPENDENCY_COUNT_OFFSET,
      dependency_count);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_DATA_LENGTH_OFFSET,
      kArgumentDataLength);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_COUNT_OFFSET,
      kCommandCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_COUNT_OFFSET,
      kParameterRootCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_COUNT_OFFSET,
      kParameterCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_LENGTH_OFFSET,
      kParameterKeyLength);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_COUNT_OFFSET,
      kEntrySchemaCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_KIND_COUNT_OFFSET,
      kEntrySchemaKindCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_BINDING_COUNT_OFFSET,
      kLaunchBindingCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_BINDING_TABLE_OFFSET,
      layout.launch_binding_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET,
      layout.buffer_ref_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_MEMORY_DEPENDENCY_TABLE_OFFSET,
      layout.memory_dependency_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_TABLE_OFFSET,
      layout.entry_schema_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_KIND_TABLE_OFFSET,
      layout.entry_schema_kind_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_DATA_OFFSET,
      layout.argument_data_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET,
      layout.command_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_TABLE_OFFSET,
      layout.parameter_root_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_TABLE_OFFSET,
      layout.parameter_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_TABLE_OFFSET,
      layout.parameter_key_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BINDING_INDEX_OFFSET,
      UINT32_MAX);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BINDING_INDEX_OFFSET,
      1);
  iree_unaligned_store_le_u64(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BYTE_LENGTH_OFFSET,
      12);
  iree_unaligned_store_le_u64(
      data.data() +
          LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_MINIMUM_ALIGNMENT_OFFSET,
      4);

  StoreLaunchBinding(data, layout.launch_binding_offset, 0,
                     LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED, 0);
  StoreLaunchBinding(data, layout.launch_binding_offset, 1,
                     LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE, 0);
  StoreBufferRef(data, layout.buffer_ref_offset, 0,
                 LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED, 0, 0, 256);
  StoreBufferRef(data, layout.buffer_ref_offset, 1,
                 LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE, 1, 0, 12);
  StoreEntrySchema(data, layout.entry_schema_offset, 0, /*entry_index=*/0,
                   /*kind_offset=*/0, /*argument_count=*/5,
                   kValidArgumentByteLength);
  data[layout.entry_schema_kind_offset + 0] =
      LOOM_CMD_PROGRAM_ARGUMENT_KIND_B32;
  data[layout.entry_schema_kind_offset + 1] =
      LOOM_CMD_PROGRAM_ARGUMENT_KIND_B64;
  data[layout.entry_schema_kind_offset + 2] =
      LOOM_CMD_PROGRAM_ARGUMENT_KIND_INDEX;
  data[layout.entry_schema_kind_offset + 3] =
      LOOM_CMD_PROGRAM_ARGUMENT_KIND_OFFSET;
  data[layout.entry_schema_kind_offset + 4] =
      LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER;
  for (uint32_t i = 0; i < 3; ++i) {
    uint8_t* argument_data = data.data() + layout.argument_data_offset +
                             i * kValidArgumentByteLength;
    iree_unaligned_store_le_u32(argument_data, 7);
    iree_unaligned_store_le_u64(argument_data + 4, UINT64_C(0x123456789));
    iree_unaligned_store_le_u64(argument_data + 12, 64);
    iree_unaligned_store_le_u64(argument_data + 20, 4096);
    iree_unaligned_store_le_u32(argument_data + kValidArgumentBufferOffset +
                                    LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET,
                                LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
    iree_unaligned_store_le_u32(
        argument_data + kValidArgumentBufferOffset +
            LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET,
        0);
    iree_unaligned_store_le_u64(
        argument_data + kValidArgumentBufferOffset +
            LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET,
        4);
    iree_unaligned_store_le_u64(
        argument_data + kValidArgumentBufferOffset +
            LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET,
        8);
  }
  StoreCommand(data, layout.command_offset, 0,
               LOOM_CMD_PROGRAM_COMMAND_KIND_FILL, 0, 0, 0, 0x12345678, 4);
  StoreCommand(data, layout.command_offset, 1,
               LOOM_CMD_PROGRAM_COMMAND_KIND_COPY, 0, 0, 0, 1);
  StoreCommand(data, layout.command_offset, 2,
               LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT,
               /*argument_offset=*/0, /*argument_schema_index=*/0, 0, 0, 1, 2,
               3);
  StoreCommand(data, layout.command_offset, 3,
               LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC,
               kValidArgumentByteLength, /*argument_schema_index=*/0, 0, 0, 1);
  StoreCommand(data, layout.command_offset, 4,
               LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC,
               kValidArgumentByteLength * 2, /*argument_schema_index=*/0, 0, 0,
               1);
  StoreCommand(data, layout.command_offset, 5,
               LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION, 0, 0);
  StoreParameterRoot(data, layout.parameter_root_offset, 0,
                     /*fixed_buffer_index=*/0, /*required_byte_length=*/512,
                     /*minimum_alignment=*/256);
  StoreParameter(data, layout.parameter_offset, 0, /*key_offset=*/0,
                 kFirstKeyLength, /*fixed_buffer_index=*/0, /*byte_offset=*/0,
                 /*byte_length=*/64, /*minimum_alignment=*/256);
  StoreParameter(data, layout.parameter_offset, 1,
                 /*key_offset=*/kFirstKeyLength, kSecondKeyLength,
                 /*fixed_buffer_index=*/0, /*byte_offset=*/256,
                 /*byte_length=*/128, /*minimum_alignment=*/256);
  memcpy(data.data() + layout.parameter_key_offset, kFirstKey, kFirstKeyLength);
  memcpy(data.data() + layout.parameter_key_offset + kFirstKeyLength,
         kSecondKey, kSecondKeyLength);
  return data;
}

static iree_const_byte_span_t AsByteSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

TEST(CmdProgramTest, TimestampUsesOrdinaryBufferReferences) {
  for (auto kind : {LOOM_CMD_PROGRAM_COMMAND_KIND_TIMESTAMP,
                    LOOM_CMD_PROGRAM_COMMAND_KIND_TIMESTAMP_BARRIER}) {
    SCOPED_TRACE(kind);
    std::vector<uint8_t> data = BuildValidProgram();
    const uint32_t command_offset = iree_unaligned_load_le_u32(
        data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET);
    const uint32_t reference_offset = iree_unaligned_load_le_u32(
        data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET);
    StoreBufferRef(data, reference_offset, 0,
                   LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE, 0, 8, 8);
    StoreCommand(data, command_offset, 0, kind, 0, 0, 0);
    loom_cmd_program_t program = {};
    IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));
    const auto capture = loom_cmd_program_command_at(&program, 0);
    EXPECT_EQ(capture.kind, kind);
    const auto target = loom_cmd_program_buffer_ref_at(
        &program, capture.payload.timestamp.target_buffer_ref);
    EXPECT_EQ(target.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
    EXPECT_EQ(target.root_index, 0u);
    EXPECT_EQ(target.byte_offset, 8u);
    EXPECT_EQ(target.byte_length, 8u);
  }
}

TEST(CmdProgramTest, ExplicitDependenciesPreserveScopesAndRanges) {
  std::vector<uint8_t> data = BuildValidProgram(2);
  const uint32_t dependency_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_MEMORY_DEPENDENCY_TABLE_OFFSET);
  const uint32_t command_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET);
  uint8_t* global = data.data() + dependency_offset;
  iree_unaligned_store_le_u32(
      global + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_SOURCE_SCOPE_OFFSET,
      LOOM_CMD_PROGRAM_ACCESS_DISPATCH_WRITE);
  iree_unaligned_store_le_u32(
      global + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_TARGET_SCOPE_OFFSET,
      LOOM_CMD_PROGRAM_ACCESS_INDIRECT_COMMAND_READ);
  iree_unaligned_store_le_u32(
      global + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_BUFFER_REF_OFFSET,
      LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_GLOBAL);
  uint8_t* ranged = global + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_SIZE;
  iree_unaligned_store_le_u32(
      ranged + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_SOURCE_SCOPE_OFFSET,
      LOOM_CMD_PROGRAM_ACCESS_TRANSFER_WRITE);
  iree_unaligned_store_le_u32(
      ranged + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_TARGET_SCOPE_OFFSET,
      LOOM_CMD_PROGRAM_ACCESS_DISPATCH_READ);
  iree_unaligned_store_le_u32(
      ranged + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_BUFFER_REF_OFFSET, 1);
  StoreCommand(
      data, command_offset, 0, LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER, 0, 0,
      LOOM_CMD_PROGRAM_STAGE_DISPATCH | LOOM_CMD_PROGRAM_STAGE_TRANSFER,
      LOOM_CMD_PROGRAM_STAGE_COMMAND_PROCESS | LOOM_CMD_PROGRAM_STAGE_DISPATCH,
      0, 0, 2);

  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));
  EXPECT_EQ(program.maximum_barrier_dependency_count, 2u);
  const auto boundary = loom_cmd_program_command_at(&program, 0);
  EXPECT_EQ(boundary.payload.barrier.source_stages,
            LOOM_CMD_PROGRAM_STAGE_DISPATCH | LOOM_CMD_PROGRAM_STAGE_TRANSFER);
  EXPECT_EQ(
      boundary.payload.barrier.target_stages,
      LOOM_CMD_PROGRAM_STAGE_COMMAND_PROCESS | LOOM_CMD_PROGRAM_STAGE_DISPATCH);
  EXPECT_EQ(boundary.payload.barrier.flags, 0u);
  EXPECT_EQ(boundary.payload.barrier.first_dependency, 0u);
  EXPECT_EQ(boundary.payload.barrier.dependency_count, 2u);
  const auto first = loom_cmd_program_memory_dependency_at(&program, 0);
  EXPECT_EQ(first.source_scope, LOOM_CMD_PROGRAM_ACCESS_DISPATCH_WRITE);
  EXPECT_EQ(first.target_scope, LOOM_CMD_PROGRAM_ACCESS_INDIRECT_COMMAND_READ);
  EXPECT_EQ(first.buffer_ref, LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_GLOBAL);
  const auto second = loom_cmd_program_memory_dependency_at(&program, 1);
  EXPECT_EQ(second.source_scope, LOOM_CMD_PROGRAM_ACCESS_TRANSFER_WRITE);
  EXPECT_EQ(second.target_scope, LOOM_CMD_PROGRAM_ACCESS_DISPATCH_READ);
  const auto range =
      loom_cmd_program_buffer_ref_at(&program, second.buffer_ref);
  EXPECT_EQ(range.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(range.root_index, 1u);
  EXPECT_EQ(range.byte_offset, 0u);
  EXPECT_EQ(range.byte_length, 12u);

  // These bytes are an untrusted artifact boundary, not a compiler-owned plan.
  const uint32_t invalid_offsets[] = {
      dependency_offset +
          LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_SOURCE_SCOPE_OFFSET,
      dependency_offset +
          LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_TARGET_SCOPE_OFFSET,
      dependency_offset + LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_SIZE +
          LOOM_CMD_PROGRAM_MEMORY_DEPENDENCY_BUFFER_REF_OFFSET,
      command_offset + LOOM_CMD_PROGRAM_COMMAND_OPERAND_0_OFFSET,
      command_offset + LOOM_CMD_PROGRAM_COMMAND_OPERAND_1_OFFSET,
      command_offset + LOOM_CMD_PROGRAM_COMMAND_OPERAND_2_OFFSET,
      command_offset + LOOM_CMD_PROGRAM_COMMAND_OPERAND_3_OFFSET,
      command_offset + LOOM_CMD_PROGRAM_COMMAND_OPERAND_4_OFFSET,
  };
  for (uint32_t offset : invalid_offsets) {
    SCOPED_TRACE(offset);
    const uint32_t valid_value =
        iree_unaligned_load_le_u32(data.data() + offset);
    iree_unaligned_store_le_u32(data.data() + offset, 1u << 31);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_cmd_program_parse(AsByteSpan(data), &program));
    iree_unaligned_store_le_u32(data.data() + offset, valid_value);
  }
}

TEST(CmdProgramTest, OnlyFullExecutionJoinsSplitWaves) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t command_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET);
  StoreCommand(data, command_offset, 0, LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER,
               0, 0, LOOM_CMD_PROGRAM_STAGE_TRANSFER,
               LOOM_CMD_PROGRAM_STAGE_DISPATCH);
  StoreCommand(data, command_offset, 1, LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER,
               0, 0, LOOM_CMD_PROGRAM_STAGE_WORK, LOOM_CMD_PROGRAM_STAGE_WORK);
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));
  loom_cmd_program_barrier_wave_iterator_t iterator;
  loom_cmd_program_barrier_wave_iterator_initialize(&program, &iterator);
  loom_cmd_program_barrier_wave_t wave = {};
  ASSERT_TRUE(loom_cmd_program_barrier_wave_iterator_next(&iterator, &wave));
  EXPECT_EQ(wave.ordinal, 0u);
  EXPECT_EQ(wave.commands.command_count, 1u);
  EXPECT_EQ(wave.work_count, 0u);
  ASSERT_TRUE(loom_cmd_program_barrier_wave_iterator_next(&iterator, &wave));
  EXPECT_EQ(wave.ordinal, 1u);
  EXPECT_EQ(wave.commands.first_command, 1u);
  EXPECT_EQ(wave.commands.command_count, 4u);
  EXPECT_EQ(wave.work_count, 3u);
}

TEST(CmdProgramTest, TimestampRejectsInvalidStorageAndPayload) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t command_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET);
  const uint32_t reference_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET);
  StoreCommand(data, command_offset, 0, LOOM_CMD_PROGRAM_COMMAND_KIND_TIMESTAMP,
               0, 0, 0);
  loom_cmd_program_t program = {};
  StoreBufferRef(data, reference_offset, 0,
                 LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE, 0, 4, 8);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
  StoreBufferRef(data, reference_offset, 0,
                 LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE, 0, 8, 16);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
  StoreBufferRef(data, reference_offset, 0,
                 LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE, 0, 8, 8);
  StoreCommand(data, command_offset, 0, LOOM_CMD_PROGRAM_COMMAND_KIND_TIMESTAMP,
               0, 0, 0, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, ParsesCanonicalProgram) {
  const std::vector<uint8_t> data = BuildValidProgram();
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));

  EXPECT_EQ(program.requirements.fixed_buffer_count, 1u);
  EXPECT_EQ(program.requirements.rebindable_binding_count, 2u);
  EXPECT_EQ(program.requirements.transient.binding_index, UINT32_MAX);
  EXPECT_EQ(program.requirements.transient.required_byte_length, 0u);
  EXPECT_EQ(program.requirements.transient.minimum_alignment, 0u);
  EXPECT_EQ(program.requirements.launch_counts.binding_index, 1u);
  EXPECT_EQ(program.requirements.launch_counts.required_byte_length, 12u);
  EXPECT_EQ(program.requirements.launch_counts.minimum_alignment, 4u);
  EXPECT_EQ(program.requirements.executable_count, 1u);
  EXPECT_EQ(program.requirements.entry_count, 1u);
  ASSERT_EQ(program.launch_bindings.count, 2u);
  const loom_cmd_program_launch_binding_t fixed_launch_binding =
      loom_cmd_program_launch_binding_at(&program, 0);
  EXPECT_EQ(fixed_launch_binding.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED);
  EXPECT_EQ(fixed_launch_binding.root_index, 0u);
  const loom_cmd_program_launch_binding_t rebindable_launch_binding =
      loom_cmd_program_launch_binding_at(&program, 1);
  EXPECT_EQ(rebindable_launch_binding.role,
            LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(rebindable_launch_binding.root_index, 0u);
  EXPECT_EQ(program.buffer_refs.count, 2u);
  ASSERT_EQ(program.entry_schemas.count, 1u);
  EXPECT_EQ(program.entry_schema_kinds.count, 5u);
  EXPECT_EQ(program.argument_data.data_length, 156u);
  EXPECT_EQ(program.commands.count, 6u);
  ASSERT_EQ(program.parameter_roots.count, 1u);
  ASSERT_EQ(program.parameters.count, 2u);
  const loom_cmd_program_buffer_ref_t binding =
      loom_cmd_program_buffer_ref_at(&program, 1);
  EXPECT_EQ(binding.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(binding.root_index, 1u);
  EXPECT_EQ(binding.byte_offset, 0u);
  EXPECT_EQ(binding.byte_length, 12u);
  const loom_cmd_program_entry_schema_t schema =
      loom_cmd_program_entry_schema_at(&program, 0);
  EXPECT_EQ(schema.entry_index, 0u);
  EXPECT_EQ(schema.argument_count, 5u);
  EXPECT_EQ(schema.argument_byte_length, 52u);
  EXPECT_EQ(loom_cmd_program_entry_schema_kind_at(&program, &schema, 1),
            LOOM_CMD_PROGRAM_ARGUMENT_KIND_B64);
  EXPECT_EQ(loom_cmd_program_entry_schema_kind_at(&program, &schema, 2),
            LOOM_CMD_PROGRAM_ARGUMENT_KIND_INDEX);
  EXPECT_EQ(loom_cmd_program_entry_schema_kind_at(&program, &schema, 3),
            LOOM_CMD_PROGRAM_ARGUMENT_KIND_OFFSET);
  const loom_cmd_program_command_t command =
      loom_cmd_program_command_at(&program, 2);
  EXPECT_EQ(command.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  EXPECT_EQ(command.argument_schema_index, 0u);
  const iree_const_byte_span_t argument_data =
      loom_cmd_program_command_argument_data(&program, &command);
  ASSERT_EQ(argument_data.data_length, 52u);
  EXPECT_EQ(iree_unaligned_load_le_u64(argument_data.data + 4),
            UINT64_C(0x123456789));
  EXPECT_EQ(iree_unaligned_load_le_u64(argument_data.data + 12), 64u);
  EXPECT_EQ(iree_unaligned_load_le_u64(argument_data.data + 20), 4096u);
  EXPECT_EQ(command.payload.dispatch_direct.workgroup_count_x, 1u);
  EXPECT_EQ(command.payload.dispatch_direct.workgroup_count_y, 2u);
  EXPECT_EQ(command.payload.dispatch_direct.workgroup_count_z, 3u);
  const loom_cmd_program_parameter_root_t parameter_root =
      loom_cmd_program_parameter_root_at(&program, 0);
  EXPECT_EQ(parameter_root.fixed_buffer_index, 0u);
  EXPECT_EQ(parameter_root.required_byte_length, 512u);
  EXPECT_EQ(parameter_root.minimum_alignment, 256u);
  const loom_cmd_program_parameter_t parameter =
      loom_cmd_program_parameter_at(&program, 1);
  EXPECT_TRUE(iree_string_view_equal(parameter.key, IREE_SV("blk.3.weight")));
  EXPECT_EQ(parameter.fixed_buffer_index, 0u);
  EXPECT_EQ(parameter.byte_offset, 256u);
  EXPECT_EQ(parameter.byte_length, 128u);
  EXPECT_EQ(parameter.minimum_alignment, 256u);
}

TEST(CmdProgramTest, IteratesCanonicalBarrierWavesOnce) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t command_table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET);
  StoreCommand(data, command_table_offset, 0,
               LOOM_CMD_PROGRAM_COMMAND_KIND_FILL_BARRIER, 0, 0, 0, 0x12345678,
               4);
  StoreCommand(data, command_table_offset, 1,
               LOOM_CMD_PROGRAM_COMMAND_KIND_COPY_BARRIER, 0, 0, 0, 1);
  StoreCommand(data, command_table_offset, 3,
               LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC_BARRIER,
               /*argument_offset=*/kValidArgumentByteLength,
               /*argument_schema_index=*/0, 0, 0, 1);
  StoreCommand(data, command_table_offset, 4,
               LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC_BARRIER,
               /*argument_offset=*/kValidArgumentByteLength * 2,
               /*argument_schema_index=*/0, 0, 0, 1);

  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));
  const loom_cmd_program_command_range_t all_commands =
      loom_cmd_program_command_range_all(&program);
  EXPECT_EQ(all_commands.first_command, 0u);
  EXPECT_EQ(all_commands.command_count, 6u);

  loom_cmd_program_barrier_wave_iterator_t iterator;
  loom_cmd_program_barrier_wave_iterator_initialize(&program, &iterator);
  const loom_cmd_program_barrier_wave_t expected[] = {
      {.ordinal = 1, .commands = {.first_command = 0, .command_count = 1}},
      {.ordinal = 2, .commands = {.first_command = 1, .command_count = 2}},
      {.ordinal = 3, .commands = {.first_command = 3, .command_count = 1}},
      {.ordinal = 4, .commands = {.first_command = 4, .command_count = 1}},
      {.ordinal = 5, .commands = {.first_command = 5, .command_count = 1}},
  };
  for (const loom_cmd_program_barrier_wave_t& expected_wave : expected) {
    loom_cmd_program_barrier_wave_t wave = {};
    ASSERT_TRUE(loom_cmd_program_barrier_wave_iterator_next(&iterator, &wave));
    EXPECT_EQ(wave.ordinal, expected_wave.ordinal);
    EXPECT_EQ(wave.commands.first_command,
              expected_wave.commands.first_command);
    EXPECT_EQ(wave.commands.command_count,
              expected_wave.commands.command_count);
  }
  loom_cmd_program_barrier_wave_t wave = {};
  EXPECT_FALSE(loom_cmd_program_barrier_wave_iterator_next(&iterator, &wave));
}

TEST(CmdProgramTest, ParsesTransientRequirement) {
  std::vector<uint8_t> data = BuildValidProgram();
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_REBINDABLE_BINDING_COUNT_OFFSET, 3);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BINDING_INDEX_OFFSET, 2);
  iree_unaligned_store_le_u64(
      data.data() + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BYTE_LENGTH_OFFSET, 4096);
  iree_unaligned_store_le_u64(
      data.data() + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_MINIMUM_ALIGNMENT_OFFSET,
      256);

  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));
  EXPECT_EQ(program.requirements.transient.binding_index, 2u);
  EXPECT_EQ(program.requirements.transient.required_byte_length, 4096u);
  EXPECT_EQ(program.requirements.transient.minimum_alignment, 256u);
}

TEST(CmdProgramTest, RejectsMalformedLaunchCountRequirement) {
  std::vector<uint8_t> data = BuildValidProgram();
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BINDING_INDEX_OFFSET,
      0);

  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsUnknownLaunchBindingRole) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_BINDING_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      data.data() + table_offset + LOOM_CMD_PROGRAM_LAUNCH_BINDING_ROLE_OFFSET,
      0xFF);

  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsNoncanonicalLaunchBindingIndex) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_BINDING_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      data.data() + table_offset + LOOM_CMD_PROGRAM_LAUNCH_BINDING_SIZE +
          LOOM_CMD_PROGRAM_LAUNCH_BINDING_ROOT_INDEX_OFFSET,
      1);

  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedLaunchCountTuple) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET);
  iree_unaligned_store_le_u64(
      data.data() + table_offset + LOOM_CMD_PROGRAM_BUFFER_REF_SIZE +
          LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET,
      8);

  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, AcceptsStableIndirectWithoutHostCountStorage) {
  std::vector<uint8_t> data = BuildValidProgram();
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BINDING_INDEX_OFFSET,
      UINT32_MAX);
  iree_unaligned_store_le_u64(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BYTE_LENGTH_OFFSET, 0);
  iree_unaligned_store_le_u64(
      data.data() +
          LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_MINIMUM_ALIGNMENT_OFFSET,
      0);

  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));
  EXPECT_EQ(program.requirements.launch_counts.binding_index, UINT32_MAX);
}

TEST(CmdProgramTest, RejectsMalformedHeader) {
  std::vector<uint8_t> data = BuildValidProgram();
  data[LOOM_CMD_PROGRAM_HEADER_MAGIC_OFFSET] ^= 0xFF;
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsStaleFormatVersion) {
  std::vector<uint8_t> data = BuildValidProgram();
  iree_unaligned_store_le_u16(
      data.data() + LOOM_CMD_PROGRAM_HEADER_VERSION_OFFSET,
      LOOM_CMD_PROGRAM_FORMAT_VERSION - 1);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedBufferReference) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET);
  iree_unaligned_store_le_u32(data.data() + table_offset +
                                  LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET,
                              1);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedArgumentBuffer) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_DATA_OFFSET);
  iree_unaligned_store_le_u32(data.data() + table_offset +
                                  kValidArgumentBufferOffset +
                                  LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET,
                              2);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedEntrySchema) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_KIND_TABLE_OFFSET);
  data[table_offset + 1] = 0xFF;
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedCommand) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      data.data() + table_offset + 2 * LOOM_CMD_PROGRAM_COMMAND_SIZE +
          LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_SCHEMA_INDEX_OFFSET,
      1);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, ParsesBarrierDispatchKind) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      data.data() + table_offset + 2 * LOOM_CMD_PROGRAM_COMMAND_SIZE +
          LOOM_CMD_PROGRAM_COMMAND_KIND_OFFSET,
      LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT_BARRIER);
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));
  const loom_cmd_program_command_t command =
      loom_cmd_program_command_at(&program, 2);
  EXPECT_EQ(command.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT_BARRIER);
  EXPECT_EQ(command.payload.dispatch_direct.workgroup_count_x, 1u);
}

TEST(CmdProgramTest, RejectsUnknownCommandKind) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      data.data() + table_offset + LOOM_CMD_PROGRAM_COMMAND_KIND_OFFSET,
      LOOM_CMD_PROGRAM_COMMAND_KIND_FILL | (2u << 8));
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsUnsupportedBarrierKind) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET);
  iree_unaligned_store_le_u32(data.data() + table_offset +
                                  5 * LOOM_CMD_PROGRAM_COMMAND_SIZE +
                                  LOOM_CMD_PROGRAM_COMMAND_KIND_OFFSET,
                              LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION |
                                  LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_BIT);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedParameterRoot) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      data.data() + table_offset +
          LOOM_CMD_PROGRAM_PARAMETER_ROOT_FIXED_BUFFER_INDEX_OFFSET,
      1);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedParameter) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      data.data() + table_offset + LOOM_CMD_PROGRAM_PARAMETER_KEY_OFFSET_OFFSET,
      1);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

}  // namespace
}  // namespace loom
