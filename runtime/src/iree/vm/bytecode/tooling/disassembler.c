// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/tooling/disassembler.h"

#include <string.h>

typedef uint8_t iree_vm_bytecode_tooling_field_encoding_t;
enum iree_vm_bytecode_tooling_field_encoding_e {
  IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_U8 = 1u,
  IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_I16 = 2u,
  IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_U16 = 3u,
  IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_I32 = 4u,
  IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_U32 = 5u,
};

typedef uint8_t iree_vm_bytecode_tooling_field_role_t;
enum iree_vm_bytecode_tooling_field_role_e {
  IREE_VM_BYTECODE_TOOLING_FIELD_ROLE_RESULT = 1u,
  IREE_VM_BYTECODE_TOOLING_FIELD_ROLE_OPERAND = 2u,
  IREE_VM_BYTECODE_TOOLING_FIELD_ROLE_IMMEDIATE = 3u,
  IREE_VM_BYTECODE_TOOLING_FIELD_ROLE_RANGE_BASE = 4u,
  IREE_VM_BYTECODE_TOOLING_FIELD_ROLE_RANGE_COUNT = 5u,
  IREE_VM_BYTECODE_TOOLING_FIELD_ROLE_CONSTRAINT_MEMBER = 6u,
  IREE_VM_BYTECODE_TOOLING_FIELD_ROLE_PADDING = 7u,
};

typedef struct iree_vm_bytecode_tooling_field_descriptor_t {
  // BSTRING offset of the encoded field name.
  uint16_t name_offset;
  // Byte offset from the beginning of the instruction record.
  uint8_t byte_offset;
  // Number of adjacent encoded scalars.
  uint8_t array_length;
  // Encoded scalar representation.
  iree_vm_bytecode_tooling_field_encoding_t encoding;
  // Presentation and semantic role.
  iree_vm_bytecode_tooling_field_role_t role;
} iree_vm_bytecode_tooling_field_descriptor_t;

typedef struct iree_vm_bytecode_tooling_instruction_descriptor_t {
  // BSTRING offset of the canonical instruction mnemonic.
  uint16_t mnemonic_offset;
  // First field in the flat field descriptor table.
  uint16_t field_base;
  // Exact encoded instruction record length.
  uint8_t byte_length;
  // Number of fields beginning at |field_base|.
  uint8_t field_count;
} iree_vm_bytecode_tooling_instruction_descriptor_t;

typedef struct iree_vm_bytecode_tooling_page_descriptor_t {
  // First instruction in the flat instruction descriptor table.
  uint16_t instruction_base;
  // Architectural page byte, or zero for Core.
  uint8_t page_id;
  // Number of instructions owned by the page.
  uint8_t instruction_count;
} iree_vm_bytecode_tooling_page_descriptor_t;

#include "iree/vm/bytecode/tooling/isa_tables.c.inc"

static iree_string_view_t iree_vm_bytecode_disassembly_string(uint16_t offset) {
  const uint8_t* value = iree_vm_bytecode_tooling_isa_string_table + offset;
  return iree_make_string_view((const char*)value + 1, value[0]);
}

static iree_status_t iree_vm_bytecode_disassembly_emit(
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_string_builder_t* builder) {
  iree_status_t status = write_callback.fn(write_callback.user_data,
                                           iree_string_builder_view(builder));
  iree_string_builder_reset(builder);
  return status;
}

static const iree_vm_bytecode_tooling_page_descriptor_t*
iree_vm_bytecode_disassembly_find_page(uint8_t page_id,
                                       iree_host_size_t* out_page_index) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_vm_bytecode_tooling_pages); ++i) {
    if (iree_vm_bytecode_tooling_pages[i].page_id == page_id) {
      *out_page_index = i;
      return &iree_vm_bytecode_tooling_pages[i];
    }
  }
  return NULL;
}

static iree_host_size_t iree_vm_bytecode_disassembly_encoding_size(
    iree_vm_bytecode_tooling_field_encoding_t encoding) {
  switch (encoding) {
    case IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_U8:
      return 1;
    case IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_I16:
    case IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_U16:
      return 2;
    case IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_I32:
    case IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_U32:
      return 4;
    default:
      return 0;
  }
}

static iree_status_t iree_vm_bytecode_disassembly_append_scalar(
    iree_string_builder_t* builder,
    iree_vm_bytecode_tooling_field_encoding_t encoding, const uint8_t* data) {
  switch (encoding) {
    case IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_U8:
      return iree_string_builder_append_format(builder, "%u",
                                               (unsigned int)data[0]);
    case IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_I16: {
      int16_t value = 0;
      memcpy(&value, data, sizeof(value));
      return iree_string_builder_append_format(builder, "%" PRId16, value);
    }
    case IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_U16: {
      uint16_t value = 0;
      memcpy(&value, data, sizeof(value));
      return iree_string_builder_append_format(builder, "%" PRIu16, value);
    }
    case IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_I32: {
      int32_t value = 0;
      memcpy(&value, data, sizeof(value));
      return iree_string_builder_append_format(builder, "%" PRId32, value);
    }
    case IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_U32: {
      uint32_t value = 0;
      memcpy(&value, data, sizeof(value));
      return iree_string_builder_append_format(builder, "%" PRIu32, value);
    }
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "invalid generated instruction field encoding");
  }
}

static iree_status_t iree_vm_bytecode_disassembly_append_field(
    iree_string_builder_t* builder,
    const iree_vm_bytecode_tooling_field_descriptor_t* field,
    const uint8_t* record) {
  const iree_string_view_t name =
      iree_vm_bytecode_disassembly_string(field->name_offset);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " %.*s=", (int)name.size, name.data));
  if (field->array_length > 1) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "["));
  }
  const iree_host_size_t element_size =
      iree_vm_bytecode_disassembly_encoding_size(field->encoding);
  for (uint8_t i = 0; i < field->array_length; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembly_append_scalar(
        builder, field->encoding,
        record + field->byte_offset + i * element_size));
  }
  if (field->array_length > 1) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "]"));
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_disassembly_verify_field(
    uint32_t function_ordinal, iree_host_size_t record_offset,
    const iree_vm_bytecode_tooling_instruction_descriptor_t* instruction,
    const iree_vm_bytecode_tooling_field_descriptor_t* field,
    const uint8_t* record) {
  const iree_host_size_t element_size =
      iree_vm_bytecode_disassembly_encoding_size(field->encoding);
  if (element_size == 0 || field->array_length == 0 ||
      field->byte_offset > instruction->byte_length ||
      element_size * field->array_length >
          instruction->byte_length - field->byte_offset) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "generated instruction field layout exceeds its record");
  }
  if (field->role != IREE_VM_BYTECODE_TOOLING_FIELD_ROLE_PADDING) {
    return iree_ok_status();
  }
  const iree_host_size_t byte_length = element_size * field->array_length;
  for (iree_host_size_t i = 0; i < byte_length; ++i) {
    if (record[field->byte_offset + i] != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function %" PRIu32 " record at byte %" PRIhsz
                              " has nonzero canonical padding",
                              function_ordinal, record_offset);
    }
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_disassemble_function(
    uint32_t function_ordinal, iree_const_byte_span_t bytecode,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_string_builder_t* builder) {
  iree_host_size_t offset = 0;
  while (offset < bytecode.data_length) {
    const uint8_t* record = bytecode.data + offset;
    uint8_t page_id = 0;
    uint8_t opcode = record[0];
    if (opcode >= 0xF0 && opcode <= 0xFD) {
      page_id = opcode;
      if (bytecode.data_length - offset < 2) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "function %" PRIu32
                                " has a truncated page escape at byte %" PRIhsz,
                                function_ordinal, offset);
      }
      opcode = record[1];
    } else if (opcode == 0 || opcode >= 0xFE) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "function %" PRIu32 " has reserved opcode 0x%02X at byte %" PRIhsz,
          function_ordinal, (unsigned int)record[0], offset);
    }

    iree_host_size_t page_index = 0;
    const iree_vm_bytecode_tooling_page_descriptor_t* page =
        iree_vm_bytecode_disassembly_find_page(page_id, &page_index);
    if (!page) {
      return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                              "function %" PRIu32
                              " uses unknown page 0x%02X at byte %" PRIhsz,
                              function_ordinal, (unsigned int)page_id, offset);
    }
    const uint8_t local_instruction =
        iree_vm_bytecode_tooling_opcode_maps[page_index][opcode];
    if (local_instruction == 0) {
      return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                              "function %" PRIu32
                              " uses unknown opcode 0x%02X on page 0x%02X"
                              " at byte %" PRIhsz,
                              function_ordinal, (unsigned int)opcode,
                              (unsigned int)page_id, offset);
    }
    if (local_instruction > page->instruction_count) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "generated opcode map index is out of range");
    }
    const uint16_t instruction_ordinal =
        page->instruction_base + local_instruction - 1;
    if (instruction_ordinal >=
        IREE_ARRAYSIZE(iree_vm_bytecode_tooling_instructions)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "generated instruction descriptor index is out of range");
    }
    const iree_vm_bytecode_tooling_instruction_descriptor_t* instruction =
        &iree_vm_bytecode_tooling_instructions[instruction_ordinal];
    if (instruction->byte_length == 0 ||
        instruction->byte_length > bytecode.data_length - offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "function %" PRIu32
                              " has a truncated instruction at byte %" PRIhsz,
                              function_ordinal, offset);
    }

    const iree_string_view_t mnemonic =
        iree_vm_bytecode_disassembly_string(instruction->mnemonic_offset);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "    +%04" PRIhsz "  %.*s", offset, (int)mnemonic.size,
        mnemonic.data));
    for (uint8_t i = 0; i < instruction->field_count; ++i) {
      const uint32_t field_ordinal = instruction->field_base + i;
      if (field_ordinal >= IREE_ARRAYSIZE(iree_vm_bytecode_tooling_fields)) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "generated instruction field index is out of range");
      }
      const iree_vm_bytecode_tooling_field_descriptor_t* field =
          &iree_vm_bytecode_tooling_fields[field_ordinal];
      IREE_RETURN_IF_ERROR(iree_vm_bytecode_disassembly_verify_field(
          function_ordinal, offset, instruction, field, record));
      if (field->role != IREE_VM_BYTECODE_TOOLING_FIELD_ROLE_PADDING) {
        IREE_RETURN_IF_ERROR(
            iree_vm_bytecode_disassembly_append_field(builder, field, record));
      }
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    IREE_RETURN_IF_ERROR(
        iree_vm_bytecode_disassembly_emit(write_callback, builder));
    offset += instruction->byte_length;
  }
  return iree_ok_status();
}
