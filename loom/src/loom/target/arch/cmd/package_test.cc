// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/package.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/cmd/format.h"
#include "loom/target/arch/cmd/package_format.h"

namespace loom {
namespace {

static std::vector<uint8_t> BuildProgram(bool indirect) {
  const uint32_t buffer_ref_count = indirect ? 1 : 0;
  loom_cmd_program_format_layout_t layout = {};
  IREE_CHECK_OK(loom_cmd_program_format_calculate_layout(
      buffer_ref_count, /*entry_schema_count=*/1,
      /*entry_schema_kind_count=*/0, /*argument_data_length=*/0,
      /*command_count=*/1, /*parameter_root_count=*/0,
      /*parameter_count=*/0, /*parameter_key_length=*/0, &layout));
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
      data.data() + LOOM_CMD_PROGRAM_HEADER_FIXED_BUFFER_COUNT_OFFSET, 0);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BINDING_COUNT_OFFSET,
      indirect ? 1 : 0);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_EXECUTABLE_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ENTRY_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_COUNT_OFFSET,
      buffer_ref_count);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_DATA_LENGTH_OFFSET, 0);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_COUNT_OFFSET, 0);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_COUNT_OFFSET, 0);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_LENGTH_OFFSET, 0);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_KIND_COUNT_OFFSET, 0);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET,
      layout.buffer_ref_offset);
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
      indirect ? 0 : UINT32_MAX);
  iree_unaligned_store_le_u64(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BYTE_LENGTH_OFFSET,
      indirect ? LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_BYTE_LENGTH : 0);
  iree_unaligned_store_le_u64(
      data.data() +
          LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_MINIMUM_ALIGNMENT_OFFSET,
      indirect ? LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_ALIGNMENT : 0);

  if (indirect) {
    uint8_t* buffer_ref = data.data() + layout.buffer_ref_offset;
    iree_unaligned_store_le_u32(
        buffer_ref + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET,
        LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
    iree_unaligned_store_le_u32(
        buffer_ref + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET, 0);
    iree_unaligned_store_le_u64(
        buffer_ref + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET, 0);
    iree_unaligned_store_le_u64(
        buffer_ref + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET,
        LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_BYTE_LENGTH);
  }

  uint8_t* entry_schema = data.data() + layout.entry_schema_offset;
  iree_unaligned_store_le_u32(
      entry_schema + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ENTRY_INDEX_OFFSET, 0);
  iree_unaligned_store_le_u32(
      entry_schema + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_KIND_OFFSET_OFFSET, 0);
  iree_unaligned_store_le_u32(
      entry_schema + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ARGUMENT_COUNT_OFFSET, 0);
  iree_unaligned_store_le_u32(
      entry_schema + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ARGUMENT_BYTE_LENGTH_OFFSET,
      0);

  uint8_t* command = data.data() + layout.command_offset;
  iree_unaligned_store_le_u32(
      command + LOOM_CMD_PROGRAM_COMMAND_KIND_OFFSET,
      indirect ? LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC
               : LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  iree_unaligned_store_le_u32(
      command + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_OFFSET_OFFSET, 0);
  iree_unaligned_store_le_u32(
      command + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_SCHEMA_INDEX_OFFSET, 0);
  iree_unaligned_store_le_u32(
      command + LOOM_CMD_PROGRAM_COMMAND_OPERAND_0_OFFSET, 0);
  iree_unaligned_store_le_u32(
      command + LOOM_CMD_PROGRAM_COMMAND_OPERAND_1_OFFSET, 0);
  iree_unaligned_store_le_u32(
      command + LOOM_CMD_PROGRAM_COMMAND_OPERAND_2_OFFSET, indirect ? 0 : 1);
  iree_unaligned_store_le_u32(
      command + LOOM_CMD_PROGRAM_COMMAND_OPERAND_3_OFFSET, indirect ? 0 : 1);
  iree_unaligned_store_le_u32(
      command + LOOM_CMD_PROGRAM_COMMAND_OPERAND_4_OFFSET, indirect ? 0 : 1);
  return data;
}

static iree_const_byte_span_t AsByteSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

static void BuildPackageInputs(
    const loom_cmd_program_t* direct_program,
    const loom_cmd_program_t* indirect_program,
    const std::vector<uint8_t>& host_program,
    loom_cmd_program_package_source_entry_t (&entries)[2],
    loom_cmd_program_package_source_export_t (&exports)[2]) {
  entries[0] = {/*.executable_index=*/0, /*.name=*/IREE_SV("prefill_qkv")};
  entries[1] = {/*.executable_index=*/0, /*.name=*/IREE_SV("decode_qkv")};
  exports[0] = {/*.name=*/IREE_SV("prefill"),
                /*.program=*/direct_program,
                /*.host_program_data=*/iree_const_byte_span_empty(),
                /*.entries=*/&entries[0],
                /*.entry_count=*/1};
  exports[1] = {/*.name=*/IREE_SV("decode"),
                /*.program=*/indirect_program,
                /*.host_program_data=*/AsByteSpan(host_program),
                /*.entries=*/&entries[1],
                /*.entry_count=*/1};
}

TEST(CmdProgramPackageTest, BuildsAndParsesCanonicalMultiRootPackage) {
  const std::vector<uint8_t> direct_data = BuildProgram(false);
  const std::vector<uint8_t> indirect_data = BuildProgram(true);
  const std::vector<uint8_t> host_program = {0x4C, 0x4F, 0x4F, 0x4D};
  loom_cmd_program_t direct_program = {};
  loom_cmd_program_t indirect_program = {};
  IREE_ASSERT_OK(
      loom_cmd_program_parse(AsByteSpan(direct_data), &direct_program));
  IREE_ASSERT_OK(
      loom_cmd_program_parse(AsByteSpan(indirect_data), &indirect_program));

  loom_cmd_program_package_source_entry_t entries[2] = {};
  loom_cmd_program_package_source_export_t exports[2] = {};
  BuildPackageInputs(&direct_program, &indirect_program, host_program, entries,
                     exports);

  iree_byte_span_t package_data = iree_byte_span_empty();
  loom_cmd_program_package_t built_package = {};
  IREE_ASSERT_OK(loom_cmd_program_package_build(
      exports, IREE_ARRAYSIZE(exports), iree_allocator_system(), &package_data,
      &built_package));
  EXPECT_EQ(built_package.export_count, 2u);
  EXPECT_EQ(built_package.entry_count, 2u);

  loom_cmd_program_package_t parsed_package = {};
  IREE_EXPECT_OK(loom_cmd_program_package_parse(
      iree_make_const_byte_span(package_data.data, package_data.data_length),
      &parsed_package));
  const loom_cmd_program_package_export_t prefill =
      loom_cmd_program_package_export_at(&parsed_package, 0);
  EXPECT_TRUE(iree_string_view_equal(prefill.name, IREE_SV("prefill")));
  EXPECT_EQ(prefill.host_program_data.data_length, 0u);
  EXPECT_EQ(prefill.entry_count, 1u);
  loom_cmd_program_t parsed_prefill = {};
  IREE_EXPECT_OK(loom_cmd_program_parse(prefill.program_data, &parsed_prefill));

  loom_cmd_program_package_export_t decode = {};
  ASSERT_TRUE(loom_cmd_program_package_lookup_export(
      &parsed_package, IREE_SV("decode"), &decode));
  EXPECT_EQ(decode.host_program_data.data_length, host_program.size());
  EXPECT_EQ(memcmp(decode.host_program_data.data, host_program.data(),
                   host_program.size()),
            0);
  const loom_cmd_program_package_entry_t decode_entry =
      loom_cmd_program_package_export_entry_at(&parsed_package, &decode, 0);
  EXPECT_EQ(decode_entry.executable_index, 0u);
  EXPECT_TRUE(iree_string_view_equal(decode_entry.name, IREE_SV("decode_qkv")));
  iree_allocator_free(iree_allocator_system(), package_data.data);
}

TEST(CmdProgramPackageTest, RejectsInconsistentBuildAssociations) {
  const std::vector<uint8_t> direct_data = BuildProgram(false);
  const std::vector<uint8_t> indirect_data = BuildProgram(true);
  const std::vector<uint8_t> host_program = {0x4C};
  loom_cmd_program_t direct_program = {};
  loom_cmd_program_t indirect_program = {};
  IREE_ASSERT_OK(
      loom_cmd_program_parse(AsByteSpan(direct_data), &direct_program));
  IREE_ASSERT_OK(
      loom_cmd_program_parse(AsByteSpan(indirect_data), &indirect_program));
  loom_cmd_program_package_source_entry_t entries[2] = {};
  loom_cmd_program_package_source_export_t exports[2] = {};
  BuildPackageInputs(&direct_program, &indirect_program, host_program, entries,
                     exports);

  iree_byte_span_t package_data = iree_byte_span_empty();
  loom_cmd_program_package_t package = {};
  exports[1].host_program_data = iree_const_byte_span_empty();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_package_build(
                            exports, IREE_ARRAYSIZE(exports),
                            iree_allocator_system(), &package_data, &package));

  exports[1].host_program_data = AsByteSpan(host_program);
  exports[1].name = exports[0].name;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        loom_cmd_program_package_build(
                            exports, IREE_ARRAYSIZE(exports),
                            iree_allocator_system(), &package_data, &package));

  exports[1].name = IREE_SV("decode");
  entries[1].executable_index = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_cmd_program_package_build(
                            exports, IREE_ARRAYSIZE(exports),
                            iree_allocator_system(), &package_data, &package));
}

TEST(CmdProgramPackageTest, RejectsMalformedCanonicalStorage) {
  const std::vector<uint8_t> direct_data = BuildProgram(false);
  const std::vector<uint8_t> indirect_data = BuildProgram(true);
  const std::vector<uint8_t> host_program = {0x4C};
  loom_cmd_program_t direct_program = {};
  loom_cmd_program_t indirect_program = {};
  IREE_ASSERT_OK(
      loom_cmd_program_parse(AsByteSpan(direct_data), &direct_program));
  IREE_ASSERT_OK(
      loom_cmd_program_parse(AsByteSpan(indirect_data), &indirect_program));
  loom_cmd_program_package_source_entry_t entries[2] = {};
  loom_cmd_program_package_source_export_t exports[2] = {};
  BuildPackageInputs(&direct_program, &indirect_program, host_program, entries,
                     exports);
  iree_byte_span_t package_data = iree_byte_span_empty();
  loom_cmd_program_package_t package = {};
  IREE_ASSERT_OK(loom_cmd_program_package_build(
      exports, IREE_ARRAYSIZE(exports), iree_allocator_system(), &package_data,
      &package));

  std::vector<uint8_t> malformed(package_data.data,
                                 package_data.data + package_data.data_length);
  iree_allocator_free(iree_allocator_system(), package_data.data);
  loom_cmd_program_package_t parsed_package = {};

  const uint32_t export_table_offset = iree_unaligned_load_le_u32(
      malformed.data() + LOOM_CMD_PROGRAM_PACKAGE_HEADER_EXPORT_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      malformed.data() + export_table_offset +
          LOOM_CMD_PROGRAM_PACKAGE_EXPORT_NAME_OFFSET_OFFSET,
      1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_cmd_program_package_parse(AsByteSpan(malformed), &parsed_package));

  iree_unaligned_store_le_u32(
      malformed.data() + export_table_offset +
          LOOM_CMD_PROGRAM_PACKAGE_EXPORT_NAME_OFFSET_OFFSET,
      0);
  const uint32_t entry_table_offset = iree_unaligned_load_le_u32(
      malformed.data() + LOOM_CMD_PROGRAM_PACKAGE_HEADER_ENTRY_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      malformed.data() + entry_table_offset +
          LOOM_CMD_PROGRAM_PACKAGE_ENTRY_EXECUTABLE_INDEX_OFFSET,
      1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_cmd_program_package_parse(AsByteSpan(malformed), &parsed_package));

  iree_unaligned_store_le_u32(
      malformed.data() + entry_table_offset +
          LOOM_CMD_PROGRAM_PACKAGE_ENTRY_EXECUTABLE_INDEX_OFFSET,
      0);
  const uint32_t program_offset = iree_unaligned_load_le_u32(
      malformed.data() + export_table_offset +
      LOOM_CMD_PROGRAM_PACKAGE_EXPORT_PROGRAM_OFFSET_OFFSET);
  malformed[program_offset] ^= 0xFF;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_cmd_program_package_parse(AsByteSpan(malformed), &parsed_package));
}

}  // namespace
}  // namespace loom
