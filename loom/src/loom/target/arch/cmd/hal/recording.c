// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/hal/recording.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "loom/target/arch/cmd/format.h"

typedef struct loom_cmd_hal_recording_workspace_t {
  // Parsed artifact being recorded.
  const loom_cmd_program_t* program;
  // Live package resources supplied by the application.
  const loom_cmd_hal_recording_inputs_t* inputs;
  // Begun command buffer receiving portable commands.
  iree_hal_command_buffer_t* command_buffer;
  // Resolved ranges used by fills, copies, and indirect launch counts.
  iree_hal_buffer_ref_t* buffer_refs;
  // Scratch storage for one dispatch constant block.
  uint8_t* constants;
  // Scratch storage for one dispatch binding table.
  iree_hal_buffer_ref_t* bindings;
} loom_cmd_hal_recording_workspace_t;

static iree_status_t loom_cmd_hal_record_execution_barrier(
    iree_hal_command_buffer_t* command_buffer) {
  const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                      IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ |
                      IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                      IREE_HAL_ACCESS_SCOPE_MEMORY_READ |
                      IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
  };
  return iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &memory_barrier, 0, NULL);
}

static uint16_t loom_cmd_hal_scalar_byte_length(
    loom_cmd_program_argument_kind_t kind) {
  switch (kind) {
    case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B8:
      return 1;
    case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B16:
      return 2;
    case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B32:
      return 4;
    case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B64:
      return 8;
    default:
      return 0;
  }
}

static iree_status_t loom_cmd_hal_analyze_entry_schemas(
    const loom_cmd_program_t* program,
    const loom_cmd_hal_recording_inputs_t* inputs,
    iree_host_size_t* out_max_constant_byte_length,
    iree_host_size_t* out_max_binding_count) {
  *out_max_constant_byte_length = 0;
  *out_max_binding_count = 0;
  if (program->requirements.rebindable_binding_count > UINT32_C(0x1000000)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command program binding count exceeds the HAL slot namespace");
  }
  for (uint32_t i = 0; i < program->requirements.entry_count; ++i) {
    const iree_hal_executable_function_info_t* info = &inputs->entries[i].info;
    *out_max_constant_byte_length =
        iree_max(*out_max_constant_byte_length,
                 (iree_host_size_t)info->constant_byte_length);
    *out_max_binding_count =
        iree_max(*out_max_binding_count, (iree_host_size_t)info->binding_count);
  }
  for (uint32_t i = 0; i < program->entry_schemas.count; ++i) {
    const loom_cmd_program_entry_schema_t schema =
        loom_cmd_program_entry_schema_at(program, i);
    const loom_cmd_hal_entry_t* entry = &inputs->entries[schema.entry_index];
    if (schema.argument_count != entry->info.parameter_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command entry %" PRIu32 " expects %" PRIu16
                              " parameters but its program supplies %" PRIu32,
                              schema.entry_index, entry->info.parameter_count,
                              schema.argument_count);
    }
    for (uint32_t j = 0; j < schema.argument_count; ++j) {
      const loom_cmd_program_argument_kind_t kind =
          loom_cmd_program_entry_schema_kind_at(program, &schema, j);
      const iree_hal_executable_function_parameter_t* parameter =
          &entry->parameters[j];
      if (kind == LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER) {
        if (parameter->type ==
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING) {
          continue;
        }
        if (parameter->type ==
            IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR) {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "HAL raw buffer-pointer parameters cannot preserve command "
              "program rebinding");
        }
      }
      const uint16_t scalar_byte_length = loom_cmd_hal_scalar_byte_length(kind);
      if (scalar_byte_length == 0 ||
          parameter->type !=
              IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT ||
          parameter->size != scalar_byte_length) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "command entry %" PRIu32 " argument %" PRIu32
                                " is incompatible with its live executable",
                                schema.entry_index, j);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_hal_resolve_buffer_ref(
    const loom_cmd_hal_recording_workspace_t* workspace,
    loom_cmd_program_buffer_ref_t source,
    iree_hal_buffer_ref_t* out_buffer_ref) {
  if (source.role == LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE) {
    *out_buffer_ref = iree_hal_make_indirect_buffer_ref(
        source.root_index, source.byte_offset, source.byte_length);
    return iree_ok_status();
  }

  const iree_hal_buffer_ref_t root =
      workspace->inputs->fixed_buffers[source.root_index];
  if (root.buffer == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixed-buffer root %" PRIu32 " is not bound",
                            source.root_index);
  }
  iree_device_size_t adjusted_offset = 0;
  iree_device_size_t adjusted_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      root.offset, root.length, source.byte_offset, source.byte_length,
      &adjusted_offset, &adjusted_length));
  *out_buffer_ref =
      iree_hal_make_buffer_ref(root.buffer, adjusted_offset, adjusted_length);
  return iree_ok_status();
}

static loom_cmd_program_buffer_ref_t loom_cmd_hal_decode_buffer_argument(
    const uint8_t* data) {
  return (loom_cmd_program_buffer_ref_t){
      .role = (loom_cmd_program_buffer_role_t)iree_unaligned_load_le_u32(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET),
      .root_index = iree_unaligned_load_le_u32(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET),
      .byte_offset = iree_unaligned_load_le_u64(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET),
      .byte_length = iree_unaligned_load_le_u64(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET),
  };
}

static iree_status_t loom_cmd_hal_pack_arguments(
    loom_cmd_hal_recording_workspace_t* workspace,
    const loom_cmd_program_command_t* command,
    const loom_cmd_hal_entry_t* entry, iree_const_byte_span_t* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings) {
  if (entry->info.constant_byte_length != 0) {
    memset(workspace->constants, 0, entry->info.constant_byte_length);
  }
  if (entry->info.binding_count != 0) {
    memset(workspace->bindings, 0,
           entry->info.binding_count * sizeof(*workspace->bindings));
  }
  const loom_cmd_program_entry_schema_t schema =
      loom_cmd_program_entry_schema_at(workspace->program,
                                       command->argument_schema_index);
  const iree_const_byte_span_t argument_data =
      loom_cmd_program_command_argument_data(workspace->program, command);
  const uint8_t* cursor = argument_data.data;
  for (uint32_t i = 0; i < schema.argument_count; ++i) {
    const loom_cmd_program_argument_kind_t kind =
        loom_cmd_program_entry_schema_kind_at(workspace->program, &schema, i);
    const iree_hal_executable_function_parameter_t* parameter =
        &entry->parameters[i];
    if (kind == LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER) {
      const loom_cmd_program_buffer_ref_t source =
          loom_cmd_hal_decode_buffer_argument(cursor);
      IREE_RETURN_IF_ERROR(loom_cmd_hal_resolve_buffer_ref(
          workspace, source, &workspace->bindings[parameter->offset]));
      cursor += LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
    } else {
      memcpy(workspace->constants + parameter->offset, cursor, parameter->size);
      cursor += parameter->size;
    }
  }

  *out_constants = iree_make_const_byte_span(workspace->constants,
                                             entry->info.constant_byte_length);
  *out_bindings = (iree_hal_buffer_ref_list_t){
      .count = entry->info.binding_count,
      .values = workspace->bindings,
  };
  return iree_ok_status();
}

static iree_status_t loom_cmd_hal_record_dispatch(
    loom_cmd_hal_recording_workspace_t* workspace,
    const loom_cmd_program_command_t* command) {
  const loom_cmd_program_command_kind_t kind =
      loom_cmd_program_command_kind_base(command->kind);
  const bool is_direct = kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT;
  const uint32_t entry_index =
      is_direct ? command->payload.dispatch_direct.entry_index
                : command->payload.dispatch_indirect.entry_index;
  const loom_cmd_hal_entry_t* entry = &workspace->inputs->entries[entry_index];

  iree_const_byte_span_t constants = iree_const_byte_span_empty();
  iree_hal_buffer_ref_list_t bindings = iree_hal_buffer_ref_list_empty();
  IREE_RETURN_IF_ERROR(loom_cmd_hal_pack_arguments(workspace, command, entry,
                                                   &constants, &bindings));
  iree_hal_dispatch_config_t config = {0};
  iree_hal_dispatch_flags_t flags = IREE_HAL_DISPATCH_FLAG_NONE;
  if (is_direct) {
    config = iree_hal_make_static_dispatch_config(
        command->payload.dispatch_direct.workgroup_count_x,
        command->payload.dispatch_direct.workgroup_count_y,
        command->payload.dispatch_direct.workgroup_count_z);
  } else {
    config.workgroup_count_ref =
        workspace->buffer_refs[command->payload.dispatch_indirect
                                   .workgroup_count_buffer_ref];
    flags = kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC
                ? IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS
                : IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS;
  }
  return iree_hal_command_buffer_dispatch(
      workspace->command_buffer,
      workspace->inputs->executables[entry->executable_index], entry->function,
      config, constants, bindings, flags);
}

static iree_status_t loom_cmd_hal_record_command(
    loom_cmd_hal_recording_workspace_t* workspace,
    const loom_cmd_program_command_t* command) {
  if (loom_cmd_program_command_kind_has_barrier(command->kind)) {
    IREE_RETURN_IF_ERROR(
        loom_cmd_hal_record_execution_barrier(workspace->command_buffer));
  }
  switch (loom_cmd_program_command_kind_base(command->kind)) {
    case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL: {
      const uint32_t pattern = command->payload.fill.pattern;
      return iree_hal_command_buffer_fill_buffer(
          workspace->command_buffer,
          workspace->buffer_refs[command->payload.fill.target_buffer_ref],
          &pattern, command->payload.fill.pattern_length,
          IREE_HAL_FILL_FLAG_NONE);
    }
    case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY:
      return iree_hal_command_buffer_copy_buffer(
          workspace->command_buffer,
          workspace->buffer_refs[command->payload.copy.source_buffer_ref],
          workspace->buffer_refs[command->payload.copy.target_buffer_ref],
          IREE_HAL_COPY_FLAG_NONE);
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT:
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC:
      return loom_cmd_hal_record_dispatch(workspace, command);
    case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
      return loom_cmd_hal_record_execution_barrier(workspace->command_buffer);
    default:
      IREE_ASSERT_UNREACHABLE("validated command kind");
      return iree_ok_status();
  }
}

iree_status_t loom_cmd_hal_record_program(
    const loom_cmd_program_t* program,
    const loom_cmd_hal_recording_inputs_t* inputs,
    iree_hal_command_buffer_t* command_buffer,
    iree_allocator_t host_allocator) {
  iree_host_size_t max_constant_byte_length = 0;
  iree_host_size_t max_binding_count = 0;
  IREE_RETURN_IF_ERROR(loom_cmd_hal_analyze_entry_schemas(
      program, inputs, &max_constant_byte_length, &max_binding_count));

  iree_host_size_t total_size = 0;
  iree_host_size_t buffer_refs_offset = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t bindings_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          program->buffer_refs.count, iree_hal_buffer_ref_t,
          iree_alignof(iree_hal_buffer_ref_t), &buffer_refs_offset),
      IREE_STRUCT_FIELD_ALIGNED(max_binding_count, iree_hal_buffer_ref_t,
                                iree_alignof(iree_hal_buffer_ref_t),
                                &bindings_offset),
      IREE_STRUCT_FIELD(max_constant_byte_length, uint8_t, &constants_offset)));

  uint8_t* storage = NULL;
  iree_status_t status = iree_ok_status();
  if (total_size != 0) {
    status = iree_allocator_malloc_uninitialized(host_allocator, total_size,
                                                 (void**)&storage);
  }
  loom_cmd_hal_recording_workspace_t workspace = {
      .program = program,
      .inputs = inputs,
      .command_buffer = command_buffer,
      .buffer_refs =
          storage != NULL
              ? (iree_hal_buffer_ref_t*)(storage + buffer_refs_offset)
              : NULL,
      .constants = storage != NULL ? storage + constants_offset : NULL,
      .bindings = storage != NULL
                      ? (iree_hal_buffer_ref_t*)(storage + bindings_offset)
                      : NULL,
  };
  for (uint32_t i = 0;
       i < program->buffer_refs.count && iree_status_is_ok(status); ++i) {
    status = loom_cmd_hal_resolve_buffer_ref(
        &workspace, loom_cmd_program_buffer_ref_at(program, i),
        &workspace.buffer_refs[i]);
  }
  for (uint32_t i = 0; i < program->commands.count && iree_status_is_ok(status);
       ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, i);
    status = loom_cmd_hal_record_command(&workspace, &command);
  }

  iree_allocator_free(host_allocator, storage);
  return status;
}
