// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_writer.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "loom/runtime/command/format.h"

static void loom_cmd_program_write_header(
    const loom_cmd_program_tables_t* tables,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  memcpy(data.data + LOOM_CMD_PROGRAM_HEADER_MAGIC_OFFSET,
         LOOM_CMD_PROGRAM_FORMAT_MAGIC, LOOM_CMD_PROGRAM_FORMAT_MAGIC_LENGTH);
  iree_unaligned_store_le_u16(
      data.data + LOOM_CMD_PROGRAM_HEADER_VERSION_OFFSET,
      LOOM_CMD_PROGRAM_FORMAT_VERSION);
  iree_unaligned_store_le_u16(data.data + LOOM_CMD_PROGRAM_HEADER_SIZE_OFFSET,
                              LOOM_CMD_PROGRAM_HEADER_SIZE);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_TOTAL_LENGTH_OFFSET,
      layout->total_length);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_FIXED_BUFFER_COUNT_OFFSET,
      tables->requirements.fixed_buffer_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_REBINDABLE_BINDING_COUNT_OFFSET,
      tables->requirements.rebindable_binding_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_EXECUTABLE_COUNT_OFFSET,
      tables->requirements.executable_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_COUNT_OFFSET,
      tables->requirements.entry_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_COUNT_OFFSET,
      tables->buffer_ref_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_DATA_LENGTH_OFFSET,
      tables->argument_data_length);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_COMMAND_COUNT_OFFSET,
      tables->command_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_COUNT_OFFSET,
      tables->parameter_root_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_COUNT_OFFSET,
      tables->parameter_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_LENGTH_OFFSET,
      tables->parameter_key_length);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_COUNT_OFFSET,
      tables->entry_schema_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_KIND_COUNT_OFFSET,
      tables->entry_schema_kind_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_LAUNCH_BINDING_COUNT_OFFSET,
      tables->launch_binding_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_LAUNCH_BINDING_TABLE_OFFSET,
      layout->launch_binding_offset);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET,
      layout->buffer_ref_offset);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_TABLE_OFFSET,
      layout->entry_schema_offset);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_KIND_TABLE_OFFSET,
      layout->entry_schema_kind_offset);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_DATA_OFFSET,
      layout->argument_data_offset);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET,
      layout->command_offset);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_TABLE_OFFSET,
      layout->parameter_root_offset);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_TABLE_OFFSET,
      layout->parameter_offset);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_TABLE_OFFSET,
      layout->parameter_key_offset);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BINDING_INDEX_OFFSET,
      tables->requirements.transient.binding_index);
  iree_unaligned_store_le_u64(
      data.data + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BYTE_LENGTH_OFFSET,
      tables->requirements.transient.required_byte_length);
  iree_unaligned_store_le_u64(
      data.data + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_MINIMUM_ALIGNMENT_OFFSET,
      tables->requirements.transient.minimum_alignment);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BINDING_INDEX_OFFSET,
      tables->requirements.launch_counts.binding_index);
  iree_unaligned_store_le_u64(
      data.data + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BYTE_LENGTH_OFFSET,
      tables->requirements.launch_counts.required_byte_length);
  iree_unaligned_store_le_u64(
      data.data + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_MINIMUM_ALIGNMENT_OFFSET,
      tables->requirements.launch_counts.minimum_alignment);
}

static void loom_cmd_program_write_launch_bindings(
    const loom_cmd_program_tables_t* tables,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  for (uint32_t i = 0; i < tables->launch_binding_count; ++i) {
    uint8_t* record = data.data + layout->launch_binding_offset +
                      i * LOOM_CMD_PROGRAM_LAUNCH_BINDING_SIZE;
    const loom_cmd_program_launch_binding_t binding =
        tables->launch_bindings[i];
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_LAUNCH_BINDING_ROLE_OFFSET, binding.role);
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_LAUNCH_BINDING_ROOT_INDEX_OFFSET,
        binding.root_index);
  }
}

static void loom_cmd_program_write_buffer_refs(
    const loom_cmd_program_tables_t* tables,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  for (uint32_t i = 0; i < tables->buffer_ref_count; ++i) {
    uint8_t* record = data.data + layout->buffer_ref_offset +
                      i * LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
    const loom_cmd_program_buffer_ref_t buffer_ref = tables->buffer_refs[i];
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET, buffer_ref.role);
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET,
        buffer_ref.root_index);
    iree_unaligned_store_le_u64(
        record + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET,
        buffer_ref.byte_offset);
    iree_unaligned_store_le_u64(
        record + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET,
        buffer_ref.byte_length);
  }
}

static void loom_cmd_program_write_entry_schemas(
    const loom_cmd_program_tables_t* tables,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  for (uint32_t i = 0; i < tables->entry_schema_count; ++i) {
    uint8_t* record = data.data + layout->entry_schema_offset +
                      i * LOOM_CMD_PROGRAM_ENTRY_SCHEMA_SIZE;
    const loom_cmd_program_entry_schema_t schema = tables->entry_schemas[i];
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ENTRY_INDEX_OFFSET,
        schema.entry_index);
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_KIND_OFFSET_OFFSET,
        schema.kind_offset);
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ARGUMENT_COUNT_OFFSET,
        schema.argument_count);
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ARGUMENT_BYTE_LENGTH_OFFSET,
        schema.argument_byte_length);
  }
  if (tables->entry_schema_kind_count != 0) {
    memcpy(data.data + layout->entry_schema_kind_offset,
           tables->entry_schema_kinds, tables->entry_schema_kind_count);
  }
  if (tables->argument_data_length != 0) {
    memcpy(data.data + layout->argument_data_offset, tables->argument_data,
           tables->argument_data_length);
  }
}

static void loom_cmd_program_write_commands(
    const loom_cmd_program_tables_t* tables,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  for (uint32_t i = 0; i < tables->command_count; ++i) {
    uint8_t* record =
        data.data + layout->command_offset + i * LOOM_CMD_PROGRAM_COMMAND_SIZE;
    const loom_cmd_program_command_t command = tables->commands[i];
    iree_unaligned_store_le_u32(record + LOOM_CMD_PROGRAM_COMMAND_KIND_OFFSET,
                                (uint32_t)command.kind);
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_OFFSET_OFFSET,
        command.argument_offset);
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_SCHEMA_INDEX_OFFSET,
        command.argument_schema_index);
    uint32_t operands[5] = {0};
    switch ((uint32_t)loom_cmd_program_command_kind_base(command.kind)) {
      case LOOM_CMD_PROGRAM_COMMAND_KIND_TIMESTAMP:
        operands[0] = command.payload.timestamp.target_buffer_ref;
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL:
        operands[0] = command.payload.fill.target_buffer_ref;
        operands[1] = command.payload.fill.pattern;
        operands[2] = command.payload.fill.pattern_length;
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY:
        operands[0] = command.payload.copy.source_buffer_ref;
        operands[1] = command.payload.copy.target_buffer_ref;
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT:
        operands[0] = command.payload.dispatch_direct.executable_index;
        operands[1] = command.payload.dispatch_direct.entry_index;
        operands[2] = command.payload.dispatch_direct.workgroup_count_x;
        operands[3] = command.payload.dispatch_direct.workgroup_count_y;
        operands[4] = command.payload.dispatch_direct.workgroup_count_z;
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC:
        operands[0] = command.payload.dispatch_indirect.executable_index;
        operands[1] = command.payload.dispatch_indirect.entry_index;
        operands[2] =
            command.payload.dispatch_indirect.workgroup_count_buffer_ref;
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
        break;
    }
    for (uint32_t operand_index = 0; operand_index < IREE_ARRAYSIZE(operands);
         ++operand_index) {
      iree_unaligned_store_le_u32(
          record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_0_OFFSET +
              operand_index * sizeof(uint32_t),
          operands[operand_index]);
    }
  }
}

static void loom_cmd_program_write_parameters(
    const loom_cmd_program_tables_t* tables,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  for (uint32_t i = 0; i < tables->parameter_root_count; ++i) {
    uint8_t* record = data.data + layout->parameter_root_offset +
                      i * LOOM_CMD_PROGRAM_PARAMETER_ROOT_SIZE;
    const loom_cmd_program_parameter_root_t* root = &tables->parameter_roots[i];
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_FIXED_BUFFER_INDEX_OFFSET,
        root->fixed_buffer_index);
    iree_unaligned_store_le_u64(
        record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_REQUIRED_BYTE_LENGTH_OFFSET,
        root->required_byte_length);
    iree_unaligned_store_le_u64(
        record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_MINIMUM_ALIGNMENT_OFFSET,
        root->minimum_alignment);
  }

  uint32_t key_offset = 0;
  for (uint32_t i = 0; i < tables->parameter_count; ++i) {
    uint8_t* record = data.data + layout->parameter_offset +
                      i * LOOM_CMD_PROGRAM_PARAMETER_SIZE;
    const loom_cmd_program_parameter_t* parameter = &tables->parameters[i];
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_PARAMETER_KEY_OFFSET_OFFSET, key_offset);
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_PARAMETER_KEY_LENGTH_OFFSET,
        (uint32_t)parameter->key.size);
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_PARAMETER_FIXED_BUFFER_INDEX_OFFSET,
        parameter->fixed_buffer_index);
    iree_unaligned_store_le_u64(
        record + LOOM_CMD_PROGRAM_PARAMETER_BYTE_OFFSET_OFFSET,
        parameter->byte_offset);
    iree_unaligned_store_le_u64(
        record + LOOM_CMD_PROGRAM_PARAMETER_BYTE_LENGTH_OFFSET,
        parameter->byte_length);
    iree_unaligned_store_le_u64(
        record + LOOM_CMD_PROGRAM_PARAMETER_MINIMUM_ALIGNMENT_OFFSET,
        parameter->minimum_alignment);
    memcpy(data.data + layout->parameter_key_offset + key_offset,
           parameter->key.data, parameter->key.size);
    key_offset += (uint32_t)parameter->key.size;
  }
  IREE_ASSERT_EQ(key_offset, tables->parameter_key_length);
}

iree_status_t loom_cmd_program_write(const loom_cmd_program_tables_t* tables,
                                     iree_allocator_t host_allocator,
                                     iree_byte_span_t* out_data) {
  *out_data = iree_make_byte_span(NULL, 0);
  loom_cmd_program_format_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_cmd_program_format_calculate_layout(
      tables->launch_binding_count, tables->buffer_ref_count,
      tables->entry_schema_count, tables->entry_schema_kind_count,
      tables->argument_data_length, tables->command_count,
      tables->parameter_root_count, tables->parameter_count,
      tables->parameter_key_length, &layout));
  uint8_t* data = NULL;
  // The allocation's zero initialization also clears every reserved field.
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, layout.total_length, (void**)&data));
  *out_data = iree_make_byte_span(data, layout.total_length);
  loom_cmd_program_write_header(tables, &layout, *out_data);
  loom_cmd_program_write_launch_bindings(tables, &layout, *out_data);
  loom_cmd_program_write_buffer_refs(tables, &layout, *out_data);
  loom_cmd_program_write_entry_schemas(tables, &layout, *out_data);
  loom_cmd_program_write_commands(tables, &layout, *out_data);
  loom_cmd_program_write_parameters(tables, &layout, *out_data);
  return iree_ok_status();
}
