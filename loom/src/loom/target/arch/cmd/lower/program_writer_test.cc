// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_writer.h"

#include <cstring>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/command/program.h"

namespace loom {
namespace {

TEST(CmdProgramWriterTest, OwnsInputTablesAndPayloads) {
  loom_cmd_program_launch_binding_t bindings[] = {
      {LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE, 0},
      {LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED, 0},
  };
  loom_cmd_program_buffer_ref_t buffer_refs[] = {
      {LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED, 0, 32, 16},
      {LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE, 0, 8, 16},
  };
  loom_cmd_program_entry_schema_t schemas[] = {{0, 0, 2, 16}};
  uint8_t schema_kinds[] = {LOOM_CMD_PROGRAM_ARGUMENT_KIND_INDEX,
                            LOOM_CMD_PROGRAM_ARGUMENT_KIND_OFFSET};
  uint8_t arguments[16];
  iree_unaligned_store_le_u64(arguments, UINT64_MAX - 6);
  iree_unaligned_store_le_u64(arguments + 8, UINT64_C(0x100000040));
  loom_cmd_program_command_t commands[2] = {};
  commands[0].kind = LOOM_CMD_PROGRAM_COMMAND_KIND_COPY;
  commands[0].payload.copy.source_buffer_ref = 0;
  commands[0].payload.copy.target_buffer_ref = 1;
  commands[1].kind = LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT_BARRIER;
  commands[1].payload.dispatch_direct.workgroup_count_x = 7;
  commands[1].payload.dispatch_direct.workgroup_count_y = 3;
  commands[1].payload.dispatch_direct.workgroup_count_z = 1;
  char key[] = "weights";
  loom_cmd_program_parameter_root_t roots[] = {{0, 48, 16}};
  loom_cmd_program_parameter_t parameters[] = {
      {iree_make_string_view(key, sizeof(key) - 1), 0, 32, 16, 16},
  };
  loom_cmd_program_tables_t tables = {};
  tables.requirements.fixed_buffer_count = 1;
  tables.requirements.rebindable_binding_count = 1;
  tables.requirements.executable_count = 1;
  tables.requirements.entry_count = 1;
  tables.requirements.transient.binding_index = UINT32_MAX;
  tables.requirements.launch_counts.binding_index = UINT32_MAX;
  tables.launch_bindings = bindings;
  tables.launch_binding_count = IREE_ARRAYSIZE(bindings);
  tables.buffer_refs = buffer_refs;
  tables.buffer_ref_count = IREE_ARRAYSIZE(buffer_refs);
  tables.entry_schemas = schemas;
  tables.entry_schema_count = IREE_ARRAYSIZE(schemas);
  tables.entry_schema_kinds = schema_kinds;
  tables.entry_schema_kind_count = IREE_ARRAYSIZE(schema_kinds);
  tables.argument_data = arguments;
  tables.argument_data_length = sizeof(arguments);
  tables.commands = commands;
  tables.command_count = IREE_ARRAYSIZE(commands);
  tables.parameter_roots = roots;
  tables.parameter_root_count = IREE_ARRAYSIZE(roots);
  tables.parameters = parameters;
  tables.parameter_count = IREE_ARRAYSIZE(parameters);
  tables.parameter_key_length = sizeof(key) - 1;

  iree_byte_span_t data = iree_byte_span_empty();
  IREE_ASSERT_OK(
      loom_cmd_program_write(&tables, iree_allocator_system(), &data));
  std::memset(bindings, 0, sizeof(bindings));
  std::memset(buffer_refs, 0, sizeof(buffer_refs));
  std::memset(schemas, 0, sizeof(schemas));
  std::memset(schema_kinds, 0, sizeof(schema_kinds));
  std::memset(arguments, 0, sizeof(arguments));
  std::memset(commands, 0, sizeof(commands));
  std::memset(roots, 0, sizeof(roots));
  std::memset(parameters, 0, sizeof(parameters));
  std::memset(key, 0, sizeof(key));
  std::memset(&tables, 0, sizeof(tables));

  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(data.data, data.data_length), &program));
  EXPECT_EQ(program.requirements.transient.binding_index, UINT32_MAX);
  EXPECT_EQ(program.requirements.launch_counts.binding_index, UINT32_MAX);
  EXPECT_EQ(loom_cmd_program_launch_binding_at(&program, 0).role,
            LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(loom_cmd_program_launch_binding_at(&program, 1).role,
            LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED);
  const auto source = loom_cmd_program_buffer_ref_at(&program, 0);
  EXPECT_EQ(source.byte_offset, 32u);
  EXPECT_EQ(source.byte_length, 16u);
  const auto parameter = loom_cmd_program_parameter_at(&program, 0);
  EXPECT_TRUE(iree_string_view_equal(parameter.key, IREE_SV("weights")));
  EXPECT_EQ(parameter.byte_offset, 32u);
  EXPECT_EQ(parameter.byte_length, 16u);
  EXPECT_EQ(
      loom_cmd_program_parameter_root_at(&program, 0).required_byte_length,
      48u);
  const auto schema = loom_cmd_program_entry_schema_at(&program, 0);
  EXPECT_EQ(loom_cmd_program_entry_schema_kind_at(&program, &schema, 0),
            LOOM_CMD_PROGRAM_ARGUMENT_KIND_INDEX);
  EXPECT_EQ(loom_cmd_program_entry_schema_kind_at(&program, &schema, 1),
            LOOM_CMD_PROGRAM_ARGUMENT_KIND_OFFSET);
  const auto dispatch = loom_cmd_program_command_at(&program, 1);
  EXPECT_EQ(dispatch.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT_BARRIER);
  EXPECT_EQ(dispatch.payload.dispatch_direct.workgroup_count_x, 7u);
  EXPECT_EQ(dispatch.payload.dispatch_direct.workgroup_count_y, 3u);
  EXPECT_EQ(dispatch.payload.dispatch_direct.workgroup_count_z, 1u);
  const auto payload =
      loom_cmd_program_command_argument_data(&program, &dispatch);
  EXPECT_EQ(iree_unaligned_load_le_u64(payload.data), UINT64_MAX - 6);
  EXPECT_EQ(iree_unaligned_load_le_u64(payload.data + 8),
            UINT64_C(0x100000040));
  iree_allocator_free(iree_allocator_system(), data.data);
}

TEST(CmdProgramWriterTest, AllocationFailureLeavesEmptyOutput) {
  loom_cmd_program_tables_t tables = {};
  tables.requirements.transient.binding_index = UINT32_MAX;
  tables.requirements.launch_counts.binding_index = UINT32_MAX;
  uint8_t sentinel = 0;
  iree_byte_span_t data = iree_make_byte_span(&sentinel, sizeof(sentinel));
  IREE_ALLOCATOR_INLINE_STORAGE(storage, 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_cmd_program_write(
          &tables, iree_allocator_inline_arena(&storage.header), &data));
  EXPECT_EQ(data.data, nullptr);
  EXPECT_EQ(data.data_length, 0u);
}

}  // namespace
}  // namespace loom
