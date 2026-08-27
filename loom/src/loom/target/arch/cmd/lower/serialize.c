// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/serialize.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/cmd/abi_layout.h"
#include "loom/target/arch/cmd/descriptors/descriptors.h"
#include "loom/target/arch/cmd/format.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/target/registers.h"

enum { LOOM_CMD_SERIALIZE_INITIAL_CAPACITY = 32 };

typedef enum loom_cmd_serialize_value_kind_e {
  // Value has not been materialized by a supported low operation.
  LOOM_CMD_SERIALIZE_VALUE_KIND_NONE = 0,
  // Exact unsigned 32-bit scalar.
  LOOM_CMD_SERIALIZE_VALUE_KIND_U32 = 1,
  // Exact unsigned 64-bit scalar.
  LOOM_CMD_SERIALIZE_VALUE_KIND_U64 = 2,
  // Exact 8-bit kernel argument payload.
  LOOM_CMD_SERIALIZE_VALUE_KIND_B8 = 3,
  // Exact 16-bit kernel argument payload.
  LOOM_CMD_SERIALIZE_VALUE_KIND_B16 = 4,
  // Exact 32-bit kernel argument payload.
  LOOM_CMD_SERIALIZE_VALUE_KIND_B32 = 5,
  // Exact 64-bit kernel argument payload.
  LOOM_CMD_SERIALIZE_VALUE_KIND_B64 = 6,
  // Fixed buffer-root table index.
  LOOM_CMD_SERIALIZE_VALUE_KIND_FIXED_BUFFER = 7,
  // Rebindable buffer-root table index.
  LOOM_CMD_SERIALIZE_VALUE_KIND_BINDING = 8,
  // Executable requirement table index.
  LOOM_CMD_SERIALIZE_VALUE_KIND_EXECUTABLE = 9,
  // Program entry requirement table index.
  LOOM_CMD_SERIALIZE_VALUE_KIND_ENTRY = 10,
  // Serialized buffer-reference table index.
  LOOM_CMD_SERIALIZE_VALUE_KIND_BUFFER_REF = 11,
} loom_cmd_serialize_value_kind_t;

typedef struct loom_cmd_serialize_value_t {
  // Active payload kind.
  loom_cmd_serialize_value_kind_t kind;
  // Kind-specific value payload.
  union {
    // Exact scalar bits for structural and argument values.
    uint64_t scalar;
    // Dense resource or serialized table index.
    uint32_t index;
  } payload;
} loom_cmd_serialize_value_t;

typedef struct loom_cmd_serialize_buffer_ref_table_t {
  // Arena-owned semantic buffer-reference rows.
  loom_cmd_program_buffer_ref_t* values;
  // Number of populated rows.
  iree_host_size_t count;
  // Number of allocated rows.
  iree_host_size_t capacity;
} loom_cmd_serialize_buffer_ref_table_t;

typedef struct loom_cmd_serialize_byte_table_t {
  // Arena-owned bytes.
  uint8_t* values;
  // Number of populated bytes.
  iree_host_size_t count;
  // Number of allocated bytes.
  iree_host_size_t capacity;
} loom_cmd_serialize_byte_table_t;

typedef struct loom_cmd_serialize_entry_schema_table_t {
  // Arena-owned executable-entry schema rows.
  loom_cmd_program_entry_schema_t* values;
  // Number of populated rows.
  iree_host_size_t count;
  // Number of allocated rows.
  iree_host_size_t capacity;
} loom_cmd_serialize_entry_schema_table_t;

typedef struct loom_cmd_serialize_command_table_t {
  // Arena-owned decoded command rows.
  loom_cmd_program_command_t* values;
  // Number of populated rows.
  iree_host_size_t count;
  // Number of allocated rows.
  iree_host_size_t capacity;
} loom_cmd_serialize_command_table_t;

typedef struct loom_cmd_serialize_build_t {
  // Verified module containing the closed low function.
  const loom_module_t* module;
  // Generated descriptor set governing the function body.
  const loom_low_descriptor_set_t* descriptor_set;
  // Scratch arena owning semantic build tables.
  iree_arena_allocator_t* arena;
  // Function-local mapping from module value IDs to dense ordinals.
  const loom_local_value_domain_t* value_domain;
  // Low SSA value state indexed by function-local value ordinal.
  loom_cmd_serialize_value_t* values;
  // Number of entries in |values|.
  iree_host_size_t value_count;
  // Immutable external resource counts declared by the function ABI.
  loom_cmd_program_requirements_t requirements;
  // Buffer ranges used by fills, copies, and indirect launch counts.
  loom_cmd_serialize_buffer_ref_table_t buffer_refs;
  // Executable-entry logical argument schemas.
  loom_cmd_serialize_entry_schema_table_t entry_schemas;
  // Flattened logical argument kinds referenced by entry schemas.
  loom_cmd_serialize_byte_table_t entry_schema_kinds;
  // Tagless dispatch argument payloads.
  loom_cmd_serialize_byte_table_t argument_data;
  // Schema index by dense executable entry, or UINT32_MAX before first use.
  uint32_t* entry_schema_indices;
  // Ordered command rows.
  loom_cmd_serialize_command_table_t commands;
  // Compiler-owned named parameter requirements to persist.
  const loom_cmd_parameter_requirement_table_t* parameter_requirements;
  // Compiler-owned aggregate transient requirement to persist.
  const loom_cmd_transient_requirement_t* transient_requirement;
  // Total concatenated byte length of all parameter keys.
  uint32_t parameter_key_length;
} loom_cmd_serialize_build_t;

static bool loom_cmd_serialize_packet_is(
    const loom_cmd_serialize_build_t* build,
    const loom_low_descriptor_packet_t* packet, uint32_t descriptor_ordinal) {
  IREE_ASSERT_LT(descriptor_ordinal, build->descriptor_set->descriptor_count);
  return packet->descriptor_ordinal == descriptor_ordinal;
}

static uint64_t loom_cmd_serialize_constant_value(const loom_op_t* op) {
  const loom_named_attr_slice_t attrs = loom_low_const_attrs(op);
  IREE_ASSERT_EQ(attrs.count, 1u);
  IREE_ASSERT_EQ(attrs.entries[0].value.kind, LOOM_ATTR_I64);
  const int64_t signed_value = loom_attr_as_i64(attrs.entries[0].value);
  uint64_t value = 0;
  memcpy(&value, &signed_value, sizeof(value));
  return value;
}

static loom_cmd_serialize_value_t* loom_cmd_serialize_result(
    loom_cmd_serialize_build_t* build, loom_value_id_t value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(build->value_domain, value_id);
  return &build->values[value_ordinal];
}

static const loom_cmd_serialize_value_t* loom_cmd_serialize_operand(
    const loom_cmd_serialize_build_t* build, loom_value_id_t value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(build->value_domain, value_id);
  IREE_ASSERT_NE(build->values[value_ordinal].kind,
                 LOOM_CMD_SERIALIZE_VALUE_KIND_NONE);
  return &build->values[value_ordinal];
}

static iree_status_t loom_cmd_serialize_reserve_buffer_refs(
    loom_cmd_serialize_build_t* build, iree_host_size_t additional_count) {
  const iree_host_size_t required_count =
      build->buffer_refs.count + additional_count;
  if (required_count <= build->buffer_refs.capacity) return iree_ok_status();
  return iree_arena_grow_array(
      build->arena, build->buffer_refs.count,
      iree_max(required_count,
               (iree_host_size_t)LOOM_CMD_SERIALIZE_INITIAL_CAPACITY),
      sizeof(*build->buffer_refs.values), &build->buffer_refs.capacity,
      (void**)&build->buffer_refs.values);
}

static iree_status_t loom_cmd_serialize_reserve_bytes(
    loom_cmd_serialize_build_t* build, loom_cmd_serialize_byte_table_t* table,
    iree_host_size_t additional_count) {
  const iree_host_size_t required_count = table->count + additional_count;
  if (required_count <= table->capacity) return iree_ok_status();
  return iree_arena_grow_array(
      build->arena, table->count,
      iree_max(required_count,
               (iree_host_size_t)LOOM_CMD_SERIALIZE_INITIAL_CAPACITY),
      sizeof(*table->values), &table->capacity, (void**)&table->values);
}

static iree_status_t loom_cmd_serialize_reserve_entry_schemas(
    loom_cmd_serialize_build_t* build, iree_host_size_t additional_count) {
  const iree_host_size_t required_count =
      build->entry_schemas.count + additional_count;
  if (required_count <= build->entry_schemas.capacity) return iree_ok_status();
  return iree_arena_grow_array(
      build->arena, build->entry_schemas.count,
      iree_max(required_count,
               (iree_host_size_t)LOOM_CMD_SERIALIZE_INITIAL_CAPACITY),
      sizeof(*build->entry_schemas.values), &build->entry_schemas.capacity,
      (void**)&build->entry_schemas.values);
}

static iree_status_t loom_cmd_serialize_reserve_commands(
    loom_cmd_serialize_build_t* build, iree_host_size_t additional_count) {
  const iree_host_size_t required_count =
      build->commands.count + additional_count;
  if (required_count <= build->commands.capacity) return iree_ok_status();
  return iree_arena_grow_array(
      build->arena, build->commands.count,
      iree_max(required_count,
               (iree_host_size_t)LOOM_CMD_SERIALIZE_INITIAL_CAPACITY),
      sizeof(*build->commands.values), &build->commands.capacity,
      (void**)&build->commands.values);
}

static iree_status_t loom_cmd_serialize_append_buffer_ref(
    loom_cmd_serialize_build_t* build, loom_cmd_program_buffer_ref_t buffer_ref,
    uint32_t* out_index) {
  *out_index = 0;
  if (build->buffer_refs.count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command program exceeds the buffer-reference table limit");
  }
  IREE_RETURN_IF_ERROR(loom_cmd_serialize_reserve_buffer_refs(build, 1));
  *out_index = (uint32_t)build->buffer_refs.count;
  build->buffer_refs.values[build->buffer_refs.count++] = buffer_ref;
  return iree_ok_status();
}

static iree_status_t loom_cmd_serialize_append_command(
    loom_cmd_serialize_build_t* build, loom_cmd_program_command_t command) {
  if (build->commands.count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command program exceeds the command table limit");
  }
  IREE_RETURN_IF_ERROR(loom_cmd_serialize_reserve_commands(build, 1));
  build->commands.values[build->commands.count++] = command;
  return iree_ok_status();
}

static iree_status_t loom_cmd_serialize_import_resource(
    loom_cmd_serialize_build_t* build, const loom_op_t* op) {
  const loom_value_id_t result_id = loom_low_resource_result(op);
  const loom_type_t result_type =
      loom_module_value_type(build->module, result_id);
  const uint16_t register_class_id =
      loom_low_register_type_class_id(result_type);
  const int64_t signed_resource_index = loom_low_resource_index(op);
  IREE_ASSERT_GE(signed_resource_index, 0);
  const uint64_t resource_index = (uint64_t)signed_resource_index;
  loom_cmd_serialize_value_t* result =
      loom_cmd_serialize_result(build, result_id);
  switch (register_class_id) {
    case CMD_CORE_REG_CLASS_ID_BUFFER: {
      IREE_ASSERT_LT(resource_index, build->requirements.fixed_buffer_count);
      result->kind = LOOM_CMD_SERIALIZE_VALUE_KIND_FIXED_BUFFER;
      break;
    }
    case CMD_CORE_REG_CLASS_ID_BINDING: {
      IREE_ASSERT_LT(resource_index,
                     build->requirements.rebindable_binding_count);
      result->kind = LOOM_CMD_SERIALIZE_VALUE_KIND_BINDING;
      break;
    }
    case CMD_CORE_REG_CLASS_ID_EXECUTABLE: {
      IREE_ASSERT_LT(resource_index, build->requirements.executable_count);
      result->kind = LOOM_CMD_SERIALIZE_VALUE_KIND_EXECUTABLE;
      break;
    }
    case CMD_CORE_REG_CLASS_ID_ENTRY: {
      IREE_ASSERT_LT(resource_index, build->requirements.entry_count);
      result->kind = LOOM_CMD_SERIALIZE_VALUE_KIND_ENTRY;
      break;
    }
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "command_input resource has unsupported cmd register class %" PRIu16,
          register_class_id);
  }
  result->payload.index = (uint32_t)resource_index;
  return iree_ok_status();
}

static void loom_cmd_serialize_constant(loom_cmd_serialize_build_t* build,
                                        const loom_op_t* op,
                                        loom_cmd_serialize_value_kind_t kind) {
  loom_cmd_serialize_value_t* result =
      loom_cmd_serialize_result(build, loom_low_const_result(op));
  result->kind = kind;
  result->payload.scalar = loom_cmd_serialize_constant_value(op);
}

static void loom_cmd_serialize_transfer(loom_cmd_serialize_build_t* build,
                                        const loom_op_t* op) {
  IREE_ASSERT_EQ(op->operand_count, 1u);
  IREE_ASSERT_EQ(op->result_count, 1u);
  const loom_cmd_serialize_value_t source =
      *loom_cmd_serialize_operand(build, loom_op_const_operands(op)[0]);
  *loom_cmd_serialize_result(build, loom_op_const_results(op)[0]) = source;
}

static iree_status_t loom_cmd_serialize_buffer_ref(
    loom_cmd_serialize_build_t* build, const loom_op_t* op,
    loom_cmd_program_buffer_role_t role,
    loom_cmd_serialize_value_kind_t expected_root_kind) {
  const loom_value_slice_t operands = loom_low_op_operands(op);
  const loom_value_slice_t results = loom_low_op_results(op);
  const loom_cmd_serialize_value_t* root =
      loom_cmd_serialize_operand(build, operands.values[0]);
  const loom_cmd_serialize_value_t* byte_offset =
      loom_cmd_serialize_operand(build, operands.values[1]);
  const loom_cmd_serialize_value_t* byte_length =
      loom_cmd_serialize_operand(build, operands.values[2]);
  IREE_ASSERT_EQ(root->kind, expected_root_kind);
  IREE_ASSERT_EQ(byte_offset->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_U64);
  IREE_ASSERT_EQ(byte_length->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_U64);
  const loom_cmd_program_buffer_ref_t buffer_ref = {
      .role = role,
      .root_index = root->payload.index,
      .byte_offset = byte_offset->payload.scalar,
      .byte_length = byte_length->payload.scalar,
  };
  loom_cmd_serialize_value_t* result =
      loom_cmd_serialize_result(build, results.values[0]);
  result->kind = LOOM_CMD_SERIALIZE_VALUE_KIND_BUFFER_REF;
  return loom_cmd_serialize_append_buffer_ref(build, buffer_ref,
                                              &result->payload.index);
}

static iree_status_t loom_cmd_serialize_append_argument_bytes(
    loom_cmd_serialize_build_t* build, iree_host_size_t byte_length,
    uint8_t** out_data) {
  *out_data = NULL;
  if (byte_length > UINT32_MAX ||
      build->argument_data.count > UINT32_MAX - byte_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command program exceeds the argument payload limit");
  }
  IREE_RETURN_IF_ERROR(loom_cmd_serialize_reserve_bytes(
      build, &build->argument_data, byte_length));
  *out_data = build->argument_data.values + build->argument_data.count;
  build->argument_data.count += byte_length;
  return iree_ok_status();
}

static iree_status_t loom_cmd_serialize_append_schema_kind(
    loom_cmd_serialize_build_t* build, loom_cmd_program_argument_kind_t kind) {
  if (build->entry_schema_kinds.count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command program exceeds the entry-schema kind limit");
  }
  IREE_RETURN_IF_ERROR(
      loom_cmd_serialize_reserve_bytes(build, &build->entry_schema_kinds, 1));
  build->entry_schema_kinds.values[build->entry_schema_kinds.count++] =
      (uint8_t)kind;
  return iree_ok_status();
}

static iree_status_t loom_cmd_serialize_append_scalar_argument(
    loom_cmd_serialize_build_t* build, loom_cmd_program_argument_kind_t kind,
    uint64_t value, uint32_t* out_byte_length) {
  uint32_t byte_length = 0;
  switch (kind) {
    case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B8:
      byte_length = 1;
      break;
    case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B16:
      byte_length = 2;
      break;
    case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B32:
      byte_length = 4;
      break;
    case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B64:
      byte_length = 8;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("scalar argument kind");
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      loom_cmd_serialize_append_argument_bytes(build, byte_length, &data));
  switch (byte_length) {
    case 1:
      data[0] = (uint8_t)value;
      break;
    case 2:
      iree_unaligned_store_le_u16(data, (uint16_t)value);
      break;
    case 4:
      iree_unaligned_store_le_u32(data, (uint32_t)value);
      break;
    case 8:
      iree_unaligned_store_le_u64(data, value);
      break;
  }
  *out_byte_length = byte_length;
  return iree_ok_status();
}

static iree_status_t loom_cmd_serialize_append_buffer_argument(
    loom_cmd_serialize_build_t* build, loom_cmd_program_buffer_role_t role,
    const loom_cmd_serialize_value_t* root,
    const loom_cmd_serialize_value_t* byte_offset,
    const loom_cmd_serialize_value_t* byte_length) {
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(loom_cmd_serialize_append_argument_bytes(
      build, LOOM_CMD_PROGRAM_BUFFER_REF_SIZE, &data));
  iree_unaligned_store_le_u32(data + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET,
                              role);
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET,
      root->payload.index);
  iree_unaligned_store_le_u64(
      data + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET,
      byte_offset->payload.scalar);
  iree_unaligned_store_le_u64(
      data + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET,
      byte_length->payload.scalar);
  return iree_ok_status();
}

static iree_status_t loom_cmd_serialize_flatten_arguments(
    loom_cmd_serialize_build_t* build, loom_value_slice_t operands,
    uint16_t argument_start, uint32_t entry_index,
    uint32_t* out_argument_offset, uint32_t* out_argument_schema_index) {
  *out_argument_offset = 0;
  *out_argument_schema_index = 0;
  IREE_ASSERT_LE(argument_start, operands.count);
  IREE_ASSERT_LT(entry_index, build->requirements.entry_count);

  uint32_t schema_index = build->entry_schema_indices[entry_index];
  const bool is_new_schema = schema_index == UINT32_MAX;
  if (is_new_schema) {
    if (build->entry_schemas.count >= UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "command program exceeds the entry-schema table limit");
    }
    IREE_RETURN_IF_ERROR(loom_cmd_serialize_reserve_entry_schemas(build, 1));
    schema_index = (uint32_t)build->entry_schemas.count++;
    build->entry_schema_indices[entry_index] = schema_index;
    build->entry_schemas.values[schema_index] =
        (loom_cmd_program_entry_schema_t){
            .entry_index = entry_index,
        };
  }
  loom_cmd_program_entry_schema_t* schema =
      &build->entry_schemas.values[schema_index];
  if (is_new_schema && argument_start < operands.count) {
    schema->kind_offset = (uint32_t)build->entry_schema_kinds.count;
  }

  const uint32_t argument_offset = argument_start == operands.count
                                       ? 0
                                       : (uint32_t)build->argument_data.count;
  uint32_t logical_index = 0;
  uint32_t argument_byte_length = 0;
  uint16_t operand_index = argument_start;
  while (operand_index < operands.count) {
    const loom_cmd_serialize_value_t* value =
        loom_cmd_serialize_operand(build, operands.values[operand_index]);
    loom_cmd_program_argument_kind_t kind = 0;
    uint32_t byte_length = 0;
    switch (value->kind) {
      case LOOM_CMD_SERIALIZE_VALUE_KIND_B8:
        kind = LOOM_CMD_PROGRAM_ARGUMENT_KIND_B8;
        break;
      case LOOM_CMD_SERIALIZE_VALUE_KIND_B16:
        kind = LOOM_CMD_PROGRAM_ARGUMENT_KIND_B16;
        break;
      case LOOM_CMD_SERIALIZE_VALUE_KIND_B32:
        kind = LOOM_CMD_PROGRAM_ARGUMENT_KIND_B32;
        break;
      case LOOM_CMD_SERIALIZE_VALUE_KIND_B64:
        kind = LOOM_CMD_PROGRAM_ARGUMENT_KIND_B64;
        break;
      case LOOM_CMD_SERIALIZE_VALUE_KIND_FIXED_BUFFER:
      case LOOM_CMD_SERIALIZE_VALUE_KIND_BINDING: {
        if ((iree_host_size_t)operand_index + 3 > operands.count) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "command buffer argument %" PRIu32
                                  " is missing its byte offset or byte length",
                                  logical_index);
        }
        const loom_cmd_serialize_value_t* byte_offset =
            loom_cmd_serialize_operand(build,
                                       operands.values[operand_index + 1]);
        const loom_cmd_serialize_value_t* byte_length_value =
            loom_cmd_serialize_operand(build,
                                       operands.values[operand_index + 2]);
        if (byte_offset->kind != LOOM_CMD_SERIALIZE_VALUE_KIND_U64 ||
            byte_length_value->kind != LOOM_CMD_SERIALIZE_VALUE_KIND_U64) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "command buffer argument %" PRIu32
              " requires u64 byte offset and byte length values",
              logical_index);
        }
        kind = LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER;
        IREE_RETURN_IF_ERROR(loom_cmd_serialize_append_buffer_argument(
            build,
            value->kind == LOOM_CMD_SERIALIZE_VALUE_KIND_FIXED_BUFFER
                ? LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED
                : LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE,
            value, byte_offset, byte_length_value));
        byte_length = LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
        operand_index += 3;
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "command argument %" PRIu32
                                " has no portable payload encoding",
                                logical_index);
    }
    if (kind != LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER) {
      IREE_RETURN_IF_ERROR(loom_cmd_serialize_append_scalar_argument(
          build, kind, value->payload.scalar, &byte_length));
      ++operand_index;
    }

    if (is_new_schema) {
      IREE_RETURN_IF_ERROR(loom_cmd_serialize_append_schema_kind(build, kind));
    } else if (logical_index >= schema->argument_count ||
               build->entry_schema_kinds
                       .values[schema->kind_offset + logical_index] != kind) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command entry %" PRIu32
          " is dispatched with inconsistent logical argument kinds",
          entry_index);
    }
    argument_byte_length += byte_length;
    ++logical_index;
  }

  if (is_new_schema) {
    schema->argument_count = logical_index;
    schema->argument_byte_length = argument_byte_length;
  } else if (logical_index != schema->argument_count ||
             argument_byte_length != schema->argument_byte_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command entry %" PRIu32
        " is dispatched with an inconsistent logical argument shape",
        entry_index);
  }
  *out_argument_offset = argument_offset;
  *out_argument_schema_index = schema_index;
  return iree_ok_status();
}

static iree_status_t loom_cmd_serialize_fill(
    loom_cmd_serialize_build_t* build, const loom_op_t* op,
    loom_cmd_program_command_kind_t kind) {
  const loom_value_slice_t operands = loom_low_op_operands(op);
  const loom_cmd_serialize_value_t* target =
      loom_cmd_serialize_operand(build, operands.values[0]);
  const loom_cmd_serialize_value_t* pattern =
      loom_cmd_serialize_operand(build, operands.values[1]);
  const loom_cmd_serialize_value_t* pattern_length =
      loom_cmd_serialize_operand(build, operands.values[2]);
  IREE_ASSERT_EQ(target->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_BUFFER_REF);
  IREE_ASSERT_EQ(pattern->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_U32);
  IREE_ASSERT_EQ(pattern_length->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_U32);
  const loom_cmd_program_command_t command = {
      .kind = kind,
      .payload.fill =
          {
              .target_buffer_ref = target->payload.index,
              .pattern = (uint32_t)pattern->payload.scalar,
              .pattern_length = (uint32_t)pattern_length->payload.scalar,
          },
  };
  return loom_cmd_serialize_append_command(build, command);
}

static iree_status_t loom_cmd_serialize_copy(
    loom_cmd_serialize_build_t* build, const loom_op_t* op,
    loom_cmd_program_command_kind_t kind) {
  const loom_value_slice_t operands = loom_low_op_operands(op);
  const loom_cmd_serialize_value_t* source =
      loom_cmd_serialize_operand(build, operands.values[0]);
  const loom_cmd_serialize_value_t* target =
      loom_cmd_serialize_operand(build, operands.values[1]);
  IREE_ASSERT_EQ(source->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_BUFFER_REF);
  IREE_ASSERT_EQ(target->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_BUFFER_REF);
  const loom_cmd_program_command_t command = {
      .kind = kind,
      .payload.copy =
          {
              .source_buffer_ref = source->payload.index,
              .target_buffer_ref = target->payload.index,
          },
  };
  return loom_cmd_serialize_append_command(build, command);
}

static iree_status_t loom_cmd_serialize_dispatch(
    loom_cmd_serialize_build_t* build, const loom_op_t* op,
    loom_cmd_program_command_kind_t kind) {
  const loom_value_slice_t operands = loom_low_op_operands(op);
  const loom_cmd_serialize_value_t* executable =
      loom_cmd_serialize_operand(build, operands.values[0]);
  const loom_cmd_serialize_value_t* entry =
      loom_cmd_serialize_operand(build, operands.values[1]);
  IREE_ASSERT_EQ(executable->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_EXECUTABLE);
  IREE_ASSERT_EQ(entry->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_ENTRY);
  loom_cmd_program_command_t command = {
      .kind = kind,
  };
  uint16_t argument_start = 0;
  const loom_cmd_program_command_kind_t base_kind =
      loom_cmd_program_command_kind_base(kind);
  const bool is_direct =
      base_kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT;
  if (is_direct) {
    const loom_cmd_serialize_value_t* workgroup_count_x =
        loom_cmd_serialize_operand(build, operands.values[2]);
    const loom_cmd_serialize_value_t* workgroup_count_y =
        loom_cmd_serialize_operand(build, operands.values[3]);
    const loom_cmd_serialize_value_t* workgroup_count_z =
        loom_cmd_serialize_operand(build, operands.values[4]);
    IREE_ASSERT_EQ(workgroup_count_x->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_U32);
    IREE_ASSERT_EQ(workgroup_count_y->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_U32);
    IREE_ASSERT_EQ(workgroup_count_z->kind, LOOM_CMD_SERIALIZE_VALUE_KIND_U32);
    command.payload.dispatch_direct.executable_index =
        executable->payload.index;
    command.payload.dispatch_direct.entry_index = entry->payload.index;
    command.payload.dispatch_direct.workgroup_count_x =
        (uint32_t)workgroup_count_x->payload.scalar;
    command.payload.dispatch_direct.workgroup_count_y =
        (uint32_t)workgroup_count_y->payload.scalar;
    command.payload.dispatch_direct.workgroup_count_z =
        (uint32_t)workgroup_count_z->payload.scalar;
    argument_start = 5;
  } else {
    const loom_cmd_serialize_value_t* workgroup_count =
        loom_cmd_serialize_operand(build, operands.values[2]);
    IREE_ASSERT_EQ(workgroup_count->kind,
                   LOOM_CMD_SERIALIZE_VALUE_KIND_BUFFER_REF);
    command.payload.dispatch_indirect.executable_index =
        executable->payload.index;
    command.payload.dispatch_indirect.entry_index = entry->payload.index;
    command.payload.dispatch_indirect.workgroup_count_buffer_ref =
        workgroup_count->payload.index;
    const loom_cmd_program_buffer_ref_t buffer_ref =
        build->buffer_refs.values[workgroup_count->payload.index];
    IREE_ASSERT_EQ(buffer_ref.byte_length,
                   LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_BYTE_LENGTH);
    IREE_ASSERT_EQ(
        buffer_ref.byte_offset % LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_ALIGNMENT,
        0u);
    argument_start = 3;
  }
  IREE_RETURN_IF_ERROR(loom_cmd_serialize_flatten_arguments(
      build, operands, argument_start, entry->payload.index,
      &command.argument_offset, &command.argument_schema_index));
  return loom_cmd_serialize_append_command(build, command);
}

static iree_status_t loom_cmd_serialize_packet(
    loom_cmd_serialize_build_t* build,
    const loom_low_descriptor_packet_t* packet) {
  const loom_op_t* op = packet->op;
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_CONSTANT_U32)) {
    loom_cmd_serialize_constant(build, op, LOOM_CMD_SERIALIZE_VALUE_KIND_U32);
    return iree_ok_status();
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_CONSTANT_U64)) {
    loom_cmd_serialize_constant(build, op, LOOM_CMD_SERIALIZE_VALUE_KIND_U64);
    return iree_ok_status();
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_CONSTANT_B8)) {
    loom_cmd_serialize_constant(build, op, LOOM_CMD_SERIALIZE_VALUE_KIND_B8);
    return iree_ok_status();
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_CONSTANT_B16)) {
    loom_cmd_serialize_constant(build, op, LOOM_CMD_SERIALIZE_VALUE_KIND_B16);
    return iree_ok_status();
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_CONSTANT_B32)) {
    loom_cmd_serialize_constant(build, op, LOOM_CMD_SERIALIZE_VALUE_KIND_B32);
    return iree_ok_status();
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_CONSTANT_B64)) {
    loom_cmd_serialize_constant(build, op, LOOM_CMD_SERIALIZE_VALUE_KIND_B64);
    return iree_ok_status();
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_BUFFER_REF_DIRECT)) {
    return loom_cmd_serialize_buffer_ref(
        build, op, LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED,
        LOOM_CMD_SERIALIZE_VALUE_KIND_FIXED_BUFFER);
  }
  if (loom_cmd_serialize_packet_is(
          build, packet, CMD_CORE_DESCRIPTOR_REF_BUFFER_REF_BINDING)) {
    return loom_cmd_serialize_buffer_ref(
        build, op, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE,
        LOOM_CMD_SERIALIZE_VALUE_KIND_BINDING);
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_FILL)) {
    return loom_cmd_serialize_fill(build, op,
                                   LOOM_CMD_PROGRAM_COMMAND_KIND_FILL);
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_FILL_BARRIER)) {
    return loom_cmd_serialize_fill(build, op,
                                   LOOM_CMD_PROGRAM_COMMAND_KIND_FILL_BARRIER);
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_COPY)) {
    return loom_cmd_serialize_copy(build, op,
                                   LOOM_CMD_PROGRAM_COMMAND_KIND_COPY);
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_COPY_BARRIER)) {
    return loom_cmd_serialize_copy(build, op,
                                   LOOM_CMD_PROGRAM_COMMAND_KIND_COPY_BARRIER);
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_DISPATCH_DIRECT)) {
    return loom_cmd_serialize_dispatch(
        build, op, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  }
  if (loom_cmd_serialize_packet_is(
          build, packet, CMD_CORE_DESCRIPTOR_REF_DISPATCH_DIRECT_BARRIER)) {
    return loom_cmd_serialize_dispatch(
        build, op, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT_BARRIER);
  }
  if (loom_cmd_serialize_packet_is(
          build, packet, CMD_CORE_DESCRIPTOR_REF_DISPATCH_INDIRECT_STATIC)) {
    return loom_cmd_serialize_dispatch(
        build, op, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  }
  if (loom_cmd_serialize_packet_is(
          build, packet,
          CMD_CORE_DESCRIPTOR_REF_DISPATCH_INDIRECT_STATIC_BARRIER)) {
    return loom_cmd_serialize_dispatch(
        build, op,
        LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC_BARRIER);
  }
  if (loom_cmd_serialize_packet_is(
          build, packet, CMD_CORE_DESCRIPTOR_REF_DISPATCH_INDIRECT_DYNAMIC)) {
    return loom_cmd_serialize_dispatch(
        build, op, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC);
  }
  if (loom_cmd_serialize_packet_is(
          build, packet,
          CMD_CORE_DESCRIPTOR_REF_DISPATCH_INDIRECT_DYNAMIC_BARRIER)) {
    return loom_cmd_serialize_dispatch(
        build, op,
        LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC_BARRIER);
  }
  if (loom_cmd_serialize_packet_is(build, packet,
                                   CMD_CORE_DESCRIPTOR_REF_BARRIER_EXECUTION)) {
    const loom_cmd_program_command_t command = {
        .kind = LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION,
    };
    return loom_cmd_serialize_append_command(build, command);
  }
  const iree_string_view_t descriptor_key =
      loom_low_descriptor_packet_diagnostic_key(build->descriptor_set, packet);
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "cmd low descriptor `%.*s` cannot be serialized as a portable command",
      (int)descriptor_key.size, descriptor_key.data);
}

static iree_status_t loom_cmd_serialize_function_body(
    loom_cmd_serialize_build_t* build, const loom_region_t* body) {
  const loom_block_t* block = loom_region_const_entry_block(body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    loom_low_descriptor_packet_t packet = {0};
    loom_low_descriptor_packet_initialize(build->descriptor_set, op, &packet);
    if (packet.kind != LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
      IREE_RETURN_IF_ERROR(loom_cmd_serialize_packet(build, &packet));
    } else if (loom_low_resource_isa(op)) {
      IREE_ASSERT_EQ(loom_low_resource_import_kind(op),
                     LOOM_LOW_RESOURCE_IMPORT_KIND_COMMAND_INPUT);
      IREE_RETURN_IF_ERROR(loom_cmd_serialize_import_resource(build, op));
    } else if (loom_low_copy_isa(op) || loom_low_move_isa(op)) {
      loom_cmd_serialize_transfer(build, op);
    } else if (loom_low_return_isa(op)) {
      IREE_ASSERT_EQ(op->operand_count, 0u);
    } else {
      const iree_string_view_t op_name = loom_op_name(build->module, op);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "low operation `%.*s` cannot be serialized as a portable command",
          (int)op_name.size, op_name.data);
    }
  }
  return iree_ok_status();
}

static void loom_cmd_serialize_write_header(
    const loom_cmd_serialize_build_t* build,
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
      build->requirements.fixed_buffer_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_BINDING_COUNT_OFFSET,
      build->requirements.rebindable_binding_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_EXECUTABLE_COUNT_OFFSET,
      build->requirements.executable_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_COUNT_OFFSET,
      build->requirements.entry_count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_COUNT_OFFSET,
      (uint32_t)build->buffer_refs.count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_DATA_LENGTH_OFFSET,
      (uint32_t)build->argument_data.count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_COMMAND_COUNT_OFFSET,
      (uint32_t)build->commands.count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_COUNT_OFFSET,
      build->parameter_requirements
          ? (uint32_t)build->parameter_requirements->root_count
          : 0);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_COUNT_OFFSET,
      build->parameter_requirements
          ? (uint32_t)build->parameter_requirements->count
          : 0);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_LENGTH_OFFSET,
      build->parameter_key_length);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_COUNT_OFFSET,
      (uint32_t)build->entry_schemas.count);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_SCHEMA_KIND_COUNT_OFFSET,
      (uint32_t)build->entry_schema_kinds.count);
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
      build->transient_requirement ? build->transient_requirement->binding_index
                                   : UINT32_MAX);
  iree_unaligned_store_le_u64(
      data.data + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BYTE_LENGTH_OFFSET,
      build->transient_requirement
          ? build->transient_requirement->required_byte_length
          : 0);
  iree_unaligned_store_le_u64(
      data.data + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_MINIMUM_ALIGNMENT_OFFSET,
      build->transient_requirement
          ? build->transient_requirement->minimum_alignment
          : 0);
  iree_unaligned_store_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BINDING_INDEX_OFFSET,
      build->requirements.launch_counts.binding_index);
  iree_unaligned_store_le_u64(
      data.data + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BYTE_LENGTH_OFFSET,
      build->requirements.launch_counts.required_byte_length);
  iree_unaligned_store_le_u64(
      data.data + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_MINIMUM_ALIGNMENT_OFFSET,
      build->requirements.launch_counts.minimum_alignment);
}

static void loom_cmd_serialize_write_buffer_refs(
    const loom_cmd_serialize_build_t* build,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  for (uint32_t i = 0; i < build->buffer_refs.count; ++i) {
    uint8_t* record = data.data + layout->buffer_ref_offset +
                      i * LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
    const loom_cmd_program_buffer_ref_t buffer_ref =
        build->buffer_refs.values[i];
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

static void loom_cmd_serialize_write_entry_schemas(
    const loom_cmd_serialize_build_t* build,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  for (uint32_t i = 0; i < build->entry_schemas.count; ++i) {
    uint8_t* record = data.data + layout->entry_schema_offset +
                      i * LOOM_CMD_PROGRAM_ENTRY_SCHEMA_SIZE;
    const loom_cmd_program_entry_schema_t schema =
        build->entry_schemas.values[i];
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
  if (build->entry_schema_kinds.count != 0) {
    memcpy(data.data + layout->entry_schema_kind_offset,
           build->entry_schema_kinds.values, build->entry_schema_kinds.count);
  }
  if (build->argument_data.count != 0) {
    memcpy(data.data + layout->argument_data_offset,
           build->argument_data.values, build->argument_data.count);
  }
}

static void loom_cmd_serialize_write_commands(
    const loom_cmd_serialize_build_t* build,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  for (uint32_t i = 0; i < build->commands.count; ++i) {
    uint8_t* record =
        data.data + layout->command_offset + i * LOOM_CMD_PROGRAM_COMMAND_SIZE;
    const loom_cmd_program_command_t command = build->commands.values[i];
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

static void loom_cmd_serialize_write_parameters(
    const loom_cmd_serialize_build_t* build,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  if (!build->parameter_requirements) return;
  for (uint32_t i = 0; i < build->parameter_requirements->root_count; ++i) {
    uint8_t* record = data.data + layout->parameter_root_offset +
                      i * LOOM_CMD_PROGRAM_PARAMETER_ROOT_SIZE;
    const loom_cmd_parameter_root_requirement_t* root =
        &build->parameter_requirements->roots[i];
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
  for (uint32_t i = 0; i < build->parameter_requirements->count; ++i) {
    uint8_t* record = data.data + layout->parameter_offset +
                      i * LOOM_CMD_PROGRAM_PARAMETER_SIZE;
    const loom_cmd_parameter_requirement_t* parameter =
        &build->parameter_requirements->entries[i];
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
  IREE_ASSERT_EQ(key_offset, build->parameter_key_length);
}

static void loom_cmd_serialize_write_program(
    const loom_cmd_serialize_build_t* build,
    const loom_cmd_program_format_layout_t* layout, iree_byte_span_t data) {
  IREE_ASSERT_EQ(data.data_length, layout->total_length);
  memset(data.data, 0, data.data_length);
  loom_cmd_serialize_write_header(build, layout, data);
  loom_cmd_serialize_write_buffer_refs(build, layout, data);
  loom_cmd_serialize_write_entry_schemas(build, layout, data);
  loom_cmd_serialize_write_commands(build, layout, data);
  loom_cmd_serialize_write_parameters(build, layout, data);
}

iree_status_t loom_cmd_program_plan_serialize_root(
    const loom_cmd_program_plan_t* plan, iree_host_size_t root_index,
    iree_byte_span_t* out_data, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_LT(root_index, plan->root_count);
  IREE_ASSERT_ARGUMENT(out_data);
  *out_data = iree_make_byte_span(NULL, 0);
  loom_module_t* module = plan->root_module;
  const loom_cmd_program_root_t* root = &plan->roots[root_index];
  const loom_op_t* function_op = root->function_op;
  const loom_cmd_parameter_requirement_table_t* parameter_requirements =
      &root->parameters;
  const loom_cmd_transient_requirement_t* transient_requirement =
      &root->transient;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_cmd_core_descriptor_set();
  IREE_ASSERT(loom_low_func_def_isa(function_op));
  IREE_ASSERT_EQ(loom_low_func_def_abi(function_op),
                 LOOM_TARGET_ABI_COMMAND_PROGRAM);
  const loom_region_t* body = loom_low_function_const_body(function_op);
  IREE_ASSERT(body != NULL);
  IREE_ASSERT_EQ(body->block_count, 1u);
  IREE_ASSERT_EQ(loom_region_const_entry_block(body)->arg_count, 0u);
  IREE_ASSERT_EQ(function_op->result_count, 0u);

  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);
  loom_local_value_domain_t value_domain = {0};
  iree_status_t status = loom_local_value_domain_acquire_for_region(
      module, body, &arena, &value_domain);
  loom_cmd_serialize_build_t build = {
      .module = module,
      .descriptor_set = descriptor_set,
      .arena = &arena,
      .value_domain = &value_domain,
      .value_count = value_domain.value_count,
      .requirements =
          {
              .fixed_buffer_count = root->abi_layout.fixed_buffer_count,
              .rebindable_binding_count =
                  root->abi_layout.rebindable_binding_count,
              .executable_count = root->abi_layout.executable_count,
              .entry_count = root->abi_layout.entry_count,
              .launch_counts = root->launch_counts,
          },
      .parameter_requirements = parameter_requirements,
      .transient_requirement = transient_requirement,
  };
  if (iree_status_is_ok(status) && build.value_count != 0) {
    status =
        iree_arena_allocate_array(&arena, build.value_count,
                                  sizeof(*build.values), (void**)&build.values);
  }
  if (iree_status_is_ok(status) && build.value_count != 0) {
    memset(build.values, 0, build.value_count * sizeof(*build.values));
  }
  if (iree_status_is_ok(status) && build.requirements.entry_count != 0) {
    status = iree_arena_allocate_array(&arena, build.requirements.entry_count,
                                       sizeof(*build.entry_schema_indices),
                                       (void**)&build.entry_schema_indices);
  }
  if (iree_status_is_ok(status) && build.requirements.entry_count != 0) {
    for (uint32_t i = 0; i < build.requirements.entry_count; ++i) {
      build.entry_schema_indices[i] = UINT32_MAX;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_serialize_function_body(&build, body);
  }
  if (iree_status_is_ok(status) && parameter_requirements) {
    if (parameter_requirements->root_count > UINT32_MAX ||
        parameter_requirements->count > UINT32_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "command parameter requirement table exceeds the format limit");
    }
    uint64_t parameter_key_length = 0;
    for (iree_host_size_t i = 0;
         i < parameter_requirements->count && iree_status_is_ok(status); ++i) {
      if (!iree_checked_add_u64(parameter_key_length,
                                parameter_requirements->entries[i].key.size,
                                &parameter_key_length) ||
          parameter_key_length > UINT32_MAX) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "command parameter keys exceed the format limit");
      }
    }
    build.parameter_key_length = (uint32_t)parameter_key_length;
  }

  loom_cmd_program_format_layout_t layout = {0};
  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_format_calculate_layout(
        (uint32_t)build.buffer_refs.count, (uint32_t)build.entry_schemas.count,
        (uint32_t)build.entry_schema_kinds.count,
        (uint32_t)build.argument_data.count, (uint32_t)build.commands.count,
        parameter_requirements ? (uint32_t)parameter_requirements->root_count
                               : 0,
        parameter_requirements ? (uint32_t)parameter_requirements->count : 0,
        build.parameter_key_length, &layout);
  }
  uint8_t* data = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, layout.total_length,
                                   (void**)&data);
  }
  if (iree_status_is_ok(status)) {
    *out_data = iree_make_byte_span(data, layout.total_length);
    loom_cmd_serialize_write_program(&build, &layout, *out_data);
  }

  if (loom_local_value_domain_is_acquired(&value_domain)) {
    loom_local_value_domain_release(&value_domain);
  }
  iree_arena_deinitialize(&arena);
  return status;
}
