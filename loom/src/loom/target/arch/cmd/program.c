// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/program.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "loom/target/arch/cmd/format.h"

static loom_cmd_program_buffer_ref_t loom_cmd_program_decode_buffer_ref(
    const uint8_t* record) {
  return (loom_cmd_program_buffer_ref_t){
      .role = (loom_cmd_program_buffer_role_t)iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET),
      .root_index = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET),
      .byte_offset = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET),
      .byte_length = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET),
  };
}

loom_cmd_program_buffer_ref_t loom_cmd_program_buffer_ref_at(
    const loom_cmd_program_t* program, uint32_t index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_LT(index, program->buffer_refs.count);
  return loom_cmd_program_decode_buffer_ref(
      program->buffer_refs.data + index * LOOM_CMD_PROGRAM_BUFFER_REF_SIZE);
}

loom_cmd_program_entry_schema_t loom_cmd_program_entry_schema_at(
    const loom_cmd_program_t* program, uint32_t index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_LT(index, program->entry_schemas.count);
  const uint8_t* record =
      program->entry_schemas.data + index * LOOM_CMD_PROGRAM_ENTRY_SCHEMA_SIZE;
  return (loom_cmd_program_entry_schema_t){
      .entry_index = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ENTRY_INDEX_OFFSET),
      .kind_offset = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_KIND_OFFSET_OFFSET),
      .argument_count = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ARGUMENT_COUNT_OFFSET),
      .argument_byte_length = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_ENTRY_SCHEMA_ARGUMENT_BYTE_LENGTH_OFFSET),
  };
}

loom_cmd_program_argument_kind_t loom_cmd_program_entry_schema_kind_at(
    const loom_cmd_program_t* program,
    const loom_cmd_program_entry_schema_t* schema, uint32_t argument_index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_ARGUMENT(schema);
  IREE_ASSERT_LT(argument_index, schema->argument_count);
  IREE_ASSERT_LE(schema->kind_offset, program->entry_schema_kinds.count);
  IREE_ASSERT_LE(schema->argument_count,
                 program->entry_schema_kinds.count - schema->kind_offset);
  return (loom_cmd_program_argument_kind_t)
      program->entry_schema_kinds.data[schema->kind_offset + argument_index];
}

iree_const_byte_span_t loom_cmd_program_command_argument_data(
    const loom_cmd_program_t* program,
    const loom_cmd_program_command_t* command) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_ARGUMENT(command);
  IREE_ASSERT_LT(command->argument_schema_index, program->entry_schemas.count);
  const loom_cmd_program_entry_schema_t schema =
      loom_cmd_program_entry_schema_at(program, command->argument_schema_index);
  IREE_ASSERT_LE(command->argument_offset, program->argument_data.data_length);
  IREE_ASSERT_LE(schema.argument_byte_length,
                 program->argument_data.data_length - command->argument_offset);
  return iree_make_const_byte_span(
      program->argument_data.data + command->argument_offset,
      schema.argument_byte_length);
}

loom_cmd_program_command_t loom_cmd_program_command_at(
    const loom_cmd_program_t* program, uint32_t index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_LT(index, program->commands.count);
  const uint8_t* record =
      program->commands.data + index * LOOM_CMD_PROGRAM_COMMAND_SIZE;
  const loom_cmd_program_command_kind_t kind =
      (loom_cmd_program_command_kind_t)iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_COMMAND_KIND_OFFSET);
  const uint32_t operand_0 = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_0_OFFSET);
  const uint32_t operand_1 = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_1_OFFSET);
  const uint32_t operand_2 = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_2_OFFSET);
  const uint32_t operand_3 = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_3_OFFSET);
  const uint32_t operand_4 = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_4_OFFSET);
  loom_cmd_program_command_t command = {
      .kind = kind,
      .argument_offset = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_OFFSET_OFFSET),
      .argument_schema_index = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_SCHEMA_INDEX_OFFSET),
  };
  switch ((uint32_t)loom_cmd_program_command_kind_base(kind)) {
    case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL:
      command.payload.fill.target_buffer_ref = operand_0;
      command.payload.fill.pattern = operand_1;
      command.payload.fill.pattern_length = operand_2;
      break;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY:
      command.payload.copy.source_buffer_ref = operand_0;
      command.payload.copy.target_buffer_ref = operand_1;
      break;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT:
      command.payload.dispatch_direct.executable_index = operand_0;
      command.payload.dispatch_direct.entry_index = operand_1;
      command.payload.dispatch_direct.workgroup_count_x = operand_2;
      command.payload.dispatch_direct.workgroup_count_y = operand_3;
      command.payload.dispatch_direct.workgroup_count_z = operand_4;
      break;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC:
      command.payload.dispatch_indirect.executable_index = operand_0;
      command.payload.dispatch_indirect.entry_index = operand_1;
      command.payload.dispatch_indirect.workgroup_count_buffer_ref = operand_2;
      break;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
      break;
  }
  return command;
}

loom_cmd_program_command_range_t loom_cmd_program_command_range_all(
    const loom_cmd_program_t* program) {
  IREE_ASSERT_ARGUMENT(program);
  return (loom_cmd_program_command_range_t){
      .first_command = 0,
      .command_count = program->commands.count,
  };
}

void loom_cmd_program_barrier_wave_iterator_initialize(
    const loom_cmd_program_t* program,
    loom_cmd_program_barrier_wave_iterator_t* iterator) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_ARGUMENT(iterator);
  *iterator = (loom_cmd_program_barrier_wave_iterator_t){
      .program = program,
  };
}

bool loom_cmd_program_barrier_wave_iterator_next(
    loom_cmd_program_barrier_wave_iterator_t* iterator,
    loom_cmd_program_barrier_wave_t* out_wave) {
  IREE_ASSERT_ARGUMENT(iterator);
  IREE_ASSERT_ARGUMENT(iterator->program);
  IREE_ASSERT_ARGUMENT(out_wave);
  const loom_cmd_program_t* program = iterator->program;
  if (iterator->next_command == program->commands.count) return false;

  const uint32_t first_command = iterator->next_command;
  loom_cmd_program_command_t command =
      loom_cmd_program_command_at(program, first_command);
  if (loom_cmd_program_command_kind_begins_barrier_wave(command.kind)) {
    ++iterator->barrier_wave_ordinal;
  }

  uint32_t next_command = first_command + 1;
  while (next_command < program->commands.count) {
    command = loom_cmd_program_command_at(program, next_command);
    if (loom_cmd_program_command_kind_begins_barrier_wave(command.kind)) break;
    ++next_command;
  }
  *out_wave = (loom_cmd_program_barrier_wave_t){
      .ordinal = iterator->barrier_wave_ordinal,
      .commands =
          {
              .first_command = first_command,
              .command_count = next_command - first_command,
          },
  };
  iterator->next_command = next_command;
  return true;
}

loom_cmd_program_parameter_root_t loom_cmd_program_parameter_root_at(
    const loom_cmd_program_t* program, uint32_t index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_LT(index, program->parameter_roots.count);
  const uint8_t* record = program->parameter_roots.data +
                          index * LOOM_CMD_PROGRAM_PARAMETER_ROOT_SIZE;
  return (loom_cmd_program_parameter_root_t){
      .fixed_buffer_index = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_FIXED_BUFFER_INDEX_OFFSET),
      .required_byte_length = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_REQUIRED_BYTE_LENGTH_OFFSET),
      .minimum_alignment = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_MINIMUM_ALIGNMENT_OFFSET),
  };
}

loom_cmd_program_parameter_t loom_cmd_program_parameter_at(
    const loom_cmd_program_t* program, uint32_t index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_LT(index, program->parameters.count);
  const uint8_t* record =
      program->parameters.data + index * LOOM_CMD_PROGRAM_PARAMETER_SIZE;
  const uint32_t key_offset = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_KEY_OFFSET_OFFSET);
  const uint32_t key_length = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_KEY_LENGTH_OFFSET);
  IREE_ASSERT_LE(key_offset, program->parameter_keys.data_length);
  IREE_ASSERT_LE(key_length, program->parameter_keys.data_length - key_offset);
  return (loom_cmd_program_parameter_t){
      .key = iree_make_string_view(
          (const char*)program->parameter_keys.data + key_offset, key_length),
      .fixed_buffer_index = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_PARAMETER_FIXED_BUFFER_INDEX_OFFSET),
      .byte_offset = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_PARAMETER_BYTE_OFFSET_OFFSET),
      .byte_length = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_PARAMETER_BYTE_LENGTH_OFFSET),
      .minimum_alignment = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_PARAMETER_MINIMUM_ALIGNMENT_OFFSET),
  };
}

static iree_status_t loom_cmd_program_validate_buffer_refs(
    const loom_cmd_program_t* program) {
  for (uint32_t i = 0; i < program->buffer_refs.count; ++i) {
    const loom_cmd_program_buffer_ref_t buffer_ref =
        loom_cmd_program_buffer_ref_at(program, i);
    if (buffer_ref.role == LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED) {
      if (buffer_ref.root_index >= program->requirements.fixed_buffer_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "command buffer reference %" PRIu32
                                " selects fixed root %" PRIu32
                                " outside the fixed-buffer table",
                                i, buffer_ref.root_index);
      }
    } else if (buffer_ref.role == LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE) {
      if (buffer_ref.root_index >=
          program->requirements.rebindable_binding_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "command buffer reference %" PRIu32
                                " selects binding root %" PRIu32
                                " outside the rebindable table",
                                i, buffer_ref.root_index);
      }
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command buffer reference %" PRIu32
                              " has unknown role %u",
                              i, (unsigned)buffer_ref.role);
    }
  }
  return iree_ok_status();
}

static uint32_t loom_cmd_program_argument_byte_length(
    loom_cmd_program_argument_kind_t kind) {
  switch (kind) {
    case LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER:
      return LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
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

static iree_status_t loom_cmd_program_validate_entry_schemas(
    const loom_cmd_program_t* program) {
  uint32_t expected_kind_offset = 0;
  for (uint32_t i = 0; i < program->entry_schemas.count; ++i) {
    const loom_cmd_program_entry_schema_t schema =
        loom_cmd_program_entry_schema_at(program, i);
    if (schema.entry_index >= program->requirements.entry_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command entry schema %" PRIu32
                              " selects entry %" PRIu32
                              " outside the requirement table",
                              i, schema.entry_index);
    }
    if (schema.argument_count == 0) {
      if (schema.kind_offset != 0 || schema.argument_byte_length != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "empty command entry schema %" PRIu32
                                " has a noncanonical payload",
                                i);
      }
      continue;
    }
    if (schema.kind_offset != expected_kind_offset ||
        schema.argument_count >
            program->entry_schema_kinds.count - expected_kind_offset) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command entry schema %" PRIu32 " has a noncanonical kind slice", i);
    }
    uint64_t argument_byte_length = 0;
    for (uint32_t argument_index = 0; argument_index < schema.argument_count;
         ++argument_index) {
      const loom_cmd_program_argument_kind_t kind =
          loom_cmd_program_entry_schema_kind_at(program, &schema,
                                                argument_index);
      const uint32_t byte_length = loom_cmd_program_argument_byte_length(kind);
      if (byte_length == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "command entry schema %" PRIu32
                                " argument %" PRIu32 " has unknown kind %u",
                                i, argument_index, (unsigned)kind);
      }
      argument_byte_length += byte_length;
    }
    if (argument_byte_length != schema.argument_byte_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command entry schema %" PRIu32 " declares %" PRIu32
          " argument bytes but its "
          "kinds require %" PRIu64,
          i, schema.argument_byte_length, argument_byte_length);
    }
    expected_kind_offset += schema.argument_count;
  }
  if (expected_kind_offset != program->entry_schema_kinds.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command entry-schema kind storage has unreferenced trailing bytes");
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_argument_buffer(
    const loom_cmd_program_t* program, uint32_t command_index,
    uint32_t argument_index, const uint8_t* data) {
  const loom_cmd_program_buffer_ref_t buffer_ref =
      loom_cmd_program_decode_buffer_ref(data);
  if (buffer_ref.role == LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED) {
    if (buffer_ref.root_index < program->requirements.fixed_buffer_count) {
      return iree_ok_status();
    }
  } else if (buffer_ref.role == LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE) {
    if (buffer_ref.root_index <
        program->requirements.rebindable_binding_count) {
      return iree_ok_status();
    }
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command dispatch %" PRIu32 " argument %" PRIu32
                            " has unknown buffer role %u",
                            command_index, argument_index,
                            (unsigned)buffer_ref.role);
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "command dispatch %" PRIu32 " argument %" PRIu32
                          " selects buffer root %" PRIu32
                          " outside its requirement table",
                          command_index, argument_index, buffer_ref.root_index);
}

static iree_status_t loom_cmd_program_validate_dispatch(
    const loom_cmd_program_t* program, uint32_t command_index,
    const loom_cmd_program_command_t* command,
    uint32_t* expected_argument_offset) {
  const bool is_direct = loom_cmd_program_command_kind_base(command->kind) ==
                         LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT;
  const uint32_t executable_index =
      is_direct ? command->payload.dispatch_direct.executable_index
                : command->payload.dispatch_indirect.executable_index;
  const uint32_t entry_index =
      is_direct ? command->payload.dispatch_direct.entry_index
                : command->payload.dispatch_indirect.entry_index;
  if (executable_index >= program->requirements.executable_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command dispatch %" PRIu32
                            " selects executable %" PRIu32
                            " outside the requirement table",
                            command_index, executable_index);
  }
  if (entry_index >= program->requirements.entry_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command dispatch %" PRIu32
                            " selects entry %" PRIu32
                            " outside the requirement table",
                            command_index, entry_index);
  }
  if (command->argument_schema_index >= program->entry_schemas.count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command dispatch %" PRIu32
                            " selects entry schema %" PRIu32
                            " outside the schema table",
                            command_index, command->argument_schema_index);
  }
  const loom_cmd_program_entry_schema_t schema =
      loom_cmd_program_entry_schema_at(program, command->argument_schema_index);
  if (schema.entry_index != entry_index) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command dispatch %" PRIu32 " pairs entry %" PRIu32
                            " with schema for entry %" PRIu32,
                            command_index, entry_index, schema.entry_index);
  }
  if (schema.argument_byte_length == 0) {
    if (command->argument_offset != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty command dispatch %" PRIu32
                              " has a nonzero argument offset",
                              command_index);
    }
  } else if (command->argument_offset != *expected_argument_offset ||
             schema.argument_byte_length > program->argument_data.data_length -
                                               *expected_argument_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command dispatch %" PRIu32
                            " has a noncanonical argument byte slice",
                            command_index);
  } else {
    const uint8_t* cursor =
        program->argument_data.data + command->argument_offset;
    for (uint32_t argument_index = 0; argument_index < schema.argument_count;
         ++argument_index) {
      const loom_cmd_program_argument_kind_t kind =
          loom_cmd_program_entry_schema_kind_at(program, &schema,
                                                argument_index);
      if (kind == LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER) {
        IREE_RETURN_IF_ERROR(loom_cmd_program_validate_argument_buffer(
            program, command_index, argument_index, cursor));
      }
      cursor += loom_cmd_program_argument_byte_length(kind);
    }
    *expected_argument_offset += schema.argument_byte_length;
  }
  if (!is_direct &&
      command->payload.dispatch_indirect.workgroup_count_buffer_ref >=
          program->buffer_refs.count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command dispatch %" PRIu32
        " selects workgroup-count buffer reference %" PRIu32
        " outside the table",
        command_index,
        command->payload.dispatch_indirect.workgroup_count_buffer_ref);
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_commands(
    const loom_cmd_program_t* program) {
  uint32_t expected_argument_offset = 0;
  for (uint32_t i = 0; i < program->commands.count; ++i) {
    const uint8_t* record =
        program->commands.data + i * LOOM_CMD_PROGRAM_COMMAND_SIZE;
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, i);
    const uint32_t operand_3 = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_3_OFFSET);
    const uint32_t operand_4 = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_4_OFFSET);
    const loom_cmd_program_command_kind_t base_kind =
        loom_cmd_program_command_kind_base(command.kind);
    if (loom_cmd_program_command_kind_has_barrier(command.kind) &&
        base_kind == LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command kind %u is unsupported", command.kind);
    }
    switch ((uint32_t)base_kind) {
      case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL:
        if (command.argument_offset != 0 ||
            command.argument_schema_index != 0 ||
            command.payload.fill.target_buffer_ref >=
                program->buffer_refs.count ||
            (command.payload.fill.pattern_length != 1 &&
             command.payload.fill.pattern_length != 2 &&
             command.payload.fill.pattern_length != 4) ||
            operand_3 != 0 || operand_4 != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "command fill %" PRIu32 " has a noncanonical payload", i);
        }
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY:
        if (command.argument_offset != 0 ||
            command.argument_schema_index != 0 ||
            command.payload.copy.source_buffer_ref >=
                program->buffer_refs.count ||
            command.payload.copy.target_buffer_ref >=
                program->buffer_refs.count ||
            iree_unaligned_load_le_u32(
                record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_2_OFFSET) != 0 ||
            operand_3 != 0 || operand_4 != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "command copy %" PRIu32 " has a noncanonical payload", i);
        }
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT: {
        IREE_RETURN_IF_ERROR(loom_cmd_program_validate_dispatch(
            program, i, &command, &expected_argument_offset));
        break;
      }
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC:
        if (operand_3 != 0 || operand_4 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "indirect command dispatch %" PRIu32
                                  " has a noncanonical payload",
                                  i);
        }
        IREE_RETURN_IF_ERROR(loom_cmd_program_validate_dispatch(
            program, i, &command, &expected_argument_offset));
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
        if (command.argument_offset != 0 ||
            command.argument_schema_index != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "command barrier %" PRIu32 " has a noncanonical payload", i);
        }
        for (uint32_t operand_index = 0; operand_index < 5; ++operand_index) {
          if (iree_unaligned_load_le_u32(
                  record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_0_OFFSET +
                  operand_index * sizeof(uint32_t)) != 0) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "command barrier %" PRIu32 " has a noncanonical payload", i);
          }
        }
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "command %" PRIu32 " has unknown kind %u", i,
                                (unsigned)command.kind);
    }
  }
  if (expected_argument_offset != program->argument_data.data_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command argument storage has unreferenced trailing bytes");
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_parameter_roots(
    const loom_cmd_program_t* program) {
  uint32_t previous_fixed_buffer_index = 0;
  for (uint32_t i = 0; i < program->parameter_roots.count; ++i) {
    const uint8_t* record = program->parameter_roots.data +
                            i * LOOM_CMD_PROGRAM_PARAMETER_ROOT_SIZE;
    const loom_cmd_program_parameter_root_t root =
        loom_cmd_program_parameter_root_at(program, i);
    if (iree_unaligned_load_le_u32(
            record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_RESERVED_OFFSET) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command parameter root %" PRIu32 " has reserved fields set", i);
    }
    if (root.fixed_buffer_index >= program->requirements.fixed_buffer_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command parameter root %" PRIu32
                              " selects fixed buffer %" PRIu32
                              " outside the fixed-buffer table",
                              i, root.fixed_buffer_index);
    }
    if (i != 0 && root.fixed_buffer_index <= previous_fixed_buffer_index) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command parameter roots are not in canonical ascending order");
    }
    if (!iree_is_power_of_two_uint64(root.minimum_alignment)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command parameter root %" PRIu32
                              " has invalid minimum alignment %" PRIu64,
                              i, root.minimum_alignment);
    }
    previous_fixed_buffer_index = root.fixed_buffer_index;
  }
  return iree_ok_status();
}

static bool loom_cmd_program_find_parameter_root(
    const loom_cmd_program_t* program, uint32_t fixed_buffer_index,
    loom_cmd_program_parameter_root_t* out_root) {
  uint32_t begin = 0;
  uint32_t end = program->parameter_roots.count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    const loom_cmd_program_parameter_root_t root =
        loom_cmd_program_parameter_root_at(program, middle);
    if (root.fixed_buffer_index < fixed_buffer_index) {
      begin = middle + 1;
    } else if (root.fixed_buffer_index > fixed_buffer_index) {
      end = middle;
    } else {
      *out_root = root;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_cmd_program_validate_parameters(
    const loom_cmd_program_t* program) {
  uint32_t expected_key_offset = 0;
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const uint8_t* record =
        program->parameters.data + i * LOOM_CMD_PROGRAM_PARAMETER_SIZE;
    const uint32_t key_offset = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_PARAMETER_KEY_OFFSET_OFFSET);
    const uint32_t key_length = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_PARAMETER_KEY_LENGTH_OFFSET);
    if (iree_unaligned_load_le_u32(
            record + LOOM_CMD_PROGRAM_PARAMETER_RESERVED_OFFSET) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command parameter %" PRIu32 " has reserved fields set", i);
    }
    if (key_length == 0 || key_offset != expected_key_offset ||
        key_offset > program->parameter_keys.data_length ||
        key_length > program->parameter_keys.data_length - key_offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command parameter %" PRIu32
                              " has a noncanonical key slice [%" PRIu32
                              ", %" PRIu64 ")",
                              i, key_offset, (uint64_t)key_offset + key_length);
    }
    expected_key_offset += key_length;

    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    loom_cmd_program_parameter_root_t root = {0};
    if (!loom_cmd_program_find_parameter_root(
            program, parameter.fixed_buffer_index, &root)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command parameter %" PRIu32
                              " selects fixed buffer %" PRIu32
                              " without a parameter-root requirement",
                              i, parameter.fixed_buffer_index);
    }
    if (!iree_is_power_of_two_uint64(parameter.minimum_alignment) ||
        parameter.byte_offset % parameter.minimum_alignment != 0 ||
        root.minimum_alignment < parameter.minimum_alignment) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command parameter %" PRIu32 " has inconsistent alignment", i);
    }
    uint64_t byte_end = 0;
    if (!iree_checked_add_u64(parameter.byte_offset, parameter.byte_length,
                              &byte_end) ||
        byte_end > root.required_byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command parameter %" PRIu32
                              " exceeds fixed buffer %" PRIu32 " requirement",
                              i, parameter.fixed_buffer_index);
    }
  }
  if (expected_key_offset != program->parameter_keys.data_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command parameter key storage has unreferenced trailing bytes");
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_transient(
    const loom_cmd_program_t* program) {
  const loom_cmd_program_transient_requirement_t transient =
      program->requirements.transient;
  if (transient.binding_index == UINT32_MAX) {
    if (transient.required_byte_length != 0 ||
        transient.minimum_alignment != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command program without a transient binding declares storage");
    }
    return iree_ok_status();
  }
  if (transient.binding_index >=
      program->requirements.rebindable_binding_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command transient selects binding %" PRIu32
                            " outside the rebindable table",
                            transient.binding_index);
  }
  if (transient.required_byte_length == 0 ||
      !iree_is_power_of_two_uint64(transient.minimum_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command transient requires a positive length and power-of-two "
        "alignment");
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_launch_counts(
    const loom_cmd_program_t* program) {
  const loom_cmd_program_launch_count_requirement_t requirement =
      program->requirements.launch_counts;
  if (requirement.binding_index == UINT32_MAX) {
    if (requirement.required_byte_length != 0 ||
        requirement.minimum_alignment != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command program without host launch counts declares storage");
    }
  } else if (requirement.binding_index >=
                 program->requirements.rebindable_binding_count ||
             requirement.required_byte_length == 0 ||
             requirement.minimum_alignment !=
                 LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_ALIGNMENT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command host launch-count requirement is not representable");
  }

  bool found_host_count_dispatch = requirement.binding_index == UINT32_MAX;
  for (uint32_t i = 0; i < program->commands.count; ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, i);
    const loom_cmd_program_command_kind_t base_kind =
        loom_cmd_program_command_kind_base(command.kind);
    if (base_kind != LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC &&
        base_kind != LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC) {
      continue;
    }
    const loom_cmd_program_buffer_ref_t buffer_ref =
        loom_cmd_program_buffer_ref_at(
            program,
            command.payload.dispatch_indirect.workgroup_count_buffer_ref);
    if (buffer_ref.byte_length !=
            LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_BYTE_LENGTH ||
        buffer_ref.byte_offset %
                LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_ALIGNMENT !=
            0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "indirect dispatch references a malformed workgroup-count tuple");
    }
    if (base_kind != LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC ||
        buffer_ref.role != LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE ||
        buffer_ref.root_index != requirement.binding_index) {
      continue;
    }
    uint64_t byte_end = 0;
    if (!iree_checked_add_u64(buffer_ref.byte_offset, buffer_ref.byte_length,
                              &byte_end) ||
        byte_end > requirement.required_byte_length ||
        buffer_ref.byte_offset % requirement.minimum_alignment != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "static indirect dispatch references inconsistent host launch "
          "counts");
    }
    found_host_count_dispatch = true;
  }
  if (!found_host_count_dispatch) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host launch-count storage has no static indirect dispatch");
  }
  return iree_ok_status();
}

iree_status_t loom_cmd_program_parse(iree_const_byte_span_t data,
                                     loom_cmd_program_t* out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = (loom_cmd_program_t){0};
  if (data.data_length < LOOM_CMD_PROGRAM_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program is shorter than its header");
  }
  if (memcmp(data.data, LOOM_CMD_PROGRAM_FORMAT_MAGIC,
             LOOM_CMD_PROGRAM_FORMAT_MAGIC_LENGTH) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program has invalid magic bytes");
  }
  const uint16_t version = iree_unaligned_load_le_u16(
      data.data + LOOM_CMD_PROGRAM_HEADER_VERSION_OFFSET);
  if (version != LOOM_CMD_PROGRAM_FORMAT_VERSION) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "command program format version %" PRIu16 " is unsupported", version);
  }
  const uint16_t header_size = iree_unaligned_load_le_u16(
      data.data + LOOM_CMD_PROGRAM_HEADER_SIZE_OFFSET);
  if (header_size != LOOM_CMD_PROGRAM_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program header size %" PRIu16
                            " is not canonical",
                            header_size);
  }
  const uint32_t total_length = iree_unaligned_load_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_TOTAL_LENGTH_OFFSET);
  if (total_length != data.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program declares %" PRIu32
                            " bytes but received %" PRIhsz,
                            total_length, data.data_length);
  }
  if (iree_unaligned_load_le_u32(
          data.data + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_RESERVED_OFFSET) !=
          0 ||
      iree_unaligned_load_le_u32(
          data.data + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_RESERVED_OFFSET) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program header reserved fields are set");
  }

  loom_cmd_program_t program = {
      .storage = data,
      .requirements =
          {
              .fixed_buffer_count = iree_unaligned_load_le_u32(
                  data.data +
                  LOOM_CMD_PROGRAM_HEADER_FIXED_BUFFER_COUNT_OFFSET),
              .rebindable_binding_count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_BINDING_COUNT_OFFSET),
              .executable_count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_EXECUTABLE_COUNT_OFFSET),
              .entry_count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_COUNT_OFFSET),
              .transient =
                  {
                      .binding_index = iree_unaligned_load_le_u32(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BINDING_INDEX_OFFSET),
                      .required_byte_length = iree_unaligned_load_le_u64(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BYTE_LENGTH_OFFSET),
                      .minimum_alignment = iree_unaligned_load_le_u64(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_TRANSIENT_MINIMUM_ALIGNMENT_OFFSET),
                  },
              .launch_counts =
                  {
                      .binding_index = iree_unaligned_load_le_u32(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BINDING_INDEX_OFFSET),
                      .required_byte_length = iree_unaligned_load_le_u64(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BYTE_LENGTH_OFFSET),
                      .minimum_alignment = iree_unaligned_load_le_u64(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_MINIMUM_ALIGNMENT_OFFSET),
                  },
          },
      .buffer_refs =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_COUNT_OFFSET),
          },
      .entry_schemas =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data +
                  LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_COUNT_OFFSET),
          },
      .entry_schema_kinds =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data +
                  LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_KIND_COUNT_OFFSET),
          },
      .commands =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_COMMAND_COUNT_OFFSET),
          },
      .parameter_roots =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data +
                  LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_COUNT_OFFSET),
          },
      .parameters =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_COUNT_OFFSET),
          },
  };
  const uint32_t parameter_key_length = iree_unaligned_load_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_LENGTH_OFFSET);
  const uint32_t argument_data_length = iree_unaligned_load_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_DATA_LENGTH_OFFSET);
  loom_cmd_program_format_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_cmd_program_format_calculate_layout(
      program.buffer_refs.count, program.entry_schemas.count,
      program.entry_schema_kinds.count, argument_data_length,
      program.commands.count, program.parameter_roots.count,
      program.parameters.count, parameter_key_length, &layout));
  if (layout.total_length != total_length ||
      layout.buffer_ref_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET) ||
      layout.entry_schema_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_TABLE_OFFSET) ||
      layout.entry_schema_kind_offset !=
          iree_unaligned_load_le_u32(
              data.data +
              LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_KIND_TABLE_OFFSET) ||
      layout.argument_data_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_DATA_OFFSET) ||
      layout.command_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET) ||
      layout.parameter_root_offset !=
          iree_unaligned_load_le_u32(
              data.data +
              LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_TABLE_OFFSET) ||
      layout.parameter_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_TABLE_OFFSET) ||
      layout.parameter_key_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_TABLE_OFFSET)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program table layout is not canonical");
  }
  program.buffer_refs.data = data.data + layout.buffer_ref_offset;
  program.entry_schemas.data = data.data + layout.entry_schema_offset;
  program.entry_schema_kinds.data = data.data + layout.entry_schema_kind_offset;
  program.argument_data = iree_make_const_byte_span(
      data.data + layout.argument_data_offset, argument_data_length);
  program.commands.data = data.data + layout.command_offset;
  program.parameter_roots.data = data.data + layout.parameter_root_offset;
  program.parameters.data = data.data + layout.parameter_offset;
  program.parameter_keys = iree_make_const_byte_span(
      data.data + layout.parameter_key_offset, parameter_key_length);

  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_transient(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_buffer_refs(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_entry_schemas(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_commands(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_launch_counts(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_parameter_roots(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_parameters(&program));
  *out_program = program;
  return iree_ok_status();
}
