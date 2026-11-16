// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/function.h"

typedef struct loom_x86_branch_fixup_t {
  // Output displacement field offset.
  iree_io_stream_pos_t offset;
  // Shared CFG, common epilogue, or function-local label ordinal.
  uint32_t target;
} loom_x86_branch_fixup_t;

// The writer accumulates physical instruction counts where bytes are emitted.
// Branch relocation overwrites existing fields and contributes no instructions.
typedef struct loom_x86_function_writer_t {
  // Borrowed output stream for the current function.
  iree_io_stream_t* stream;
  // Physical instructions emitted, including the preservation envelope.
  uint64_t instruction_count;
} loom_x86_function_writer_t;

static iree_status_t loom_x86_function_write_encoding(
    loom_x86_function_writer_t* writer, loom_x86_encoding_form_t form,
    uint16_t encoding_id, loom_x86_encoding_operands_t operands) {
  loom_x86_encoded_instruction_t instruction;
  loom_x86_encode_instruction(form, encoding_id, &operands, &instruction);
  writer->instruction_count += instruction.instruction_count;
  return iree_io_stream_write(writer->stream, instruction.length,
                              instruction.bytes);
}

static iree_status_t loom_x86_function_write_stack(
    loom_x86_function_writer_t* writer, loom_x86_encoding_form_t form,
    uint8_t reg) {
  return loom_x86_function_write_encoding(
      writer, form, 0, (loom_x86_encoding_operands_t){.inputs = {reg}});
}

static iree_status_t loom_x86_function_write_stack_enter(
    const loom_x86_function_t* function, loom_x86_function_writer_t* writer) {
  if (function->stack.realignment.mask) {
    IREE_RETURN_IF_ERROR(loom_x86_function_write_encoding(
        writer, LOOM_X86_ENCODING_FORM_UNARY_R_RM,
        0x8b | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .result = function->stack.realignment.scratch_register,
            .inputs = {4}}));
  }
  if (function->stack.realignment.mask) {
    IREE_RETURN_IF_ERROR(loom_x86_function_write_encoding(
        writer, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
        0x81 | (4u << 9) | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .immediate = function->stack.realignment.mask, .result = 4}));
  }
  if (function->stack.allocation_size) {
    IREE_RETURN_IF_ERROR(loom_x86_function_write_encoding(
        writer, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
        0x81 | (5u << 9) | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .immediate = function->stack.allocation_size, .result = 4}));
  }
  if (function->stack.realignment.mask) {
    IREE_RETURN_IF_ERROR(loom_x86_function_write_encoding(
        writer, LOOM_X86_ENCODING_FORM_STORE, 0x89 | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .immediate = function->stack.realignment.saved_pointer_offset,
            .inputs = {function->stack.realignment.scratch_register, 4}}));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_write_stack_leave(
    const loom_x86_function_t* function, loom_x86_function_writer_t* writer) {
  if (function->stack.realignment.mask) {
    return loom_x86_function_write_encoding(
        writer, LOOM_X86_ENCODING_FORM_LOAD, 0x8b | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .immediate = function->stack.realignment.saved_pointer_offset,
            .result = 4,
            .inputs = {4}});
  }
  if (function->stack.allocation_size) {
    return loom_x86_function_write_encoding(
        writer, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
        0x81 | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){
            .immediate = function->stack.allocation_size, .result = 4});
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_function_write_upper_vector_state_cleanup(
    loom_x86_function_writer_t* writer) {
  static const uint8_t bytes[] = {0xc5, 0xf8, 0x77};
  ++writer->instruction_count;
  return iree_io_stream_write(writer->stream, sizeof(bytes), bytes);
}

iree_status_t loom_x86_function_write(const loom_x86_function_t* function,
                                      const uint32_t* symbol_indices,
                                      iree_host_size_t section_index,
                                      loom_native_object_fixup_t* symbol_fixups,
                                      iree_io_stream_t* stream,
                                      iree_arena_allocator_t* arena,
                                      uint64_t* out_instruction_count) {
  if (out_instruction_count) {
    *out_instruction_count = 0;
  }
  loom_x86_function_writer_t writer = {.stream = stream};
  const iree_io_stream_pos_t function_start = iree_io_stream_offset(stream);
  iree_io_stream_pos_t* block_offsets = NULL;
  loom_x86_branch_fixup_t* fixups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, function->block_count + 1 + function->local_label_count,
      sizeof(*block_offsets), (void**)&block_offsets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena,
      2 * function->block_count + function->failure_branch_count +
          function->local_fixup_count,
      sizeof(*fixups), (void**)&fixups));
  iree_status_t status = iree_ok_status();
  for (uint8_t reg = 0; reg < 16 && iree_status_is_ok(status); ++reg) {
    if (function->saved_registers & (1u << reg)) {
      status = loom_x86_function_write_stack(&writer,
                                             LOOM_X86_ENCODING_FORM_PUSH, reg);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_function_write_stack_enter(function, &writer);
  }
  iree_host_size_t fixup_count = 0;
  iree_host_size_t symbol_fixup_index = 0;
  iree_host_size_t call_cleanup_index = 0;
  uint32_t block = 0;
  for (iree_host_size_t i = 0;
       i < function->instruction_count && iree_status_is_ok(status); ++i) {
    while (block < function->block_count &&
           function->block_starts[block] == i) {
      block_offsets[block++] = iree_io_stream_offset(stream);
    }
    const loom_x86_instruction_t* prepared = &function->instructions[i];
    if (prepared->encoding_format_id == LOOM_X86_ENCODING_FORM_LABEL) {
      block_offsets[prepared->reference] = iree_io_stream_offset(stream);
      continue;
    }
    if (function->may_dirty_upper_vector_state &&
        call_cleanup_index < function->upper_vector_call_cleanup_count &&
        function->upper_vector_call_cleanup_indices[call_cleanup_index] == i) {
      status = loom_x86_function_write_upper_vector_state_cleanup(&writer);
      ++call_cleanup_index;
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    loom_x86_encoded_instruction_t instruction;
    loom_x86_encode_instruction(prepared->encoding_format_id,
                                prepared->encoding_id, &prepared->operands,
                                &instruction);
    writer.instruction_count += instruction.instruction_count;
    const bool is_call =
        prepared->encoding_format_id == LOOM_X86_ENCODING_FORM_CALL;
    const bool is_symbol_address =
        prepared->encoding_format_id ==
            LOOM_X86_ENCODING_FORM_ADDRESS_PC_RELATIVE ||
        (prepared->encoding_format_id & LOOM_X86_ENCODING_FORMAT_VECTOR &&
         ((prepared->encoding_format_id >> 12) & 7) ==
             LOOM_X86_VECTOR_ENCODING_RIP_LOAD);
    if ((is_call || is_symbol_address) && prepared->reference != UINT32_MAX) {
      symbol_fixups[symbol_fixup_index++] = (loom_native_object_fixup_t){
          .section_contribution_index = section_index,
          .section_offset = iree_io_stream_offset(stream) + instruction.length -
                            4 - function_start,
          .relocation_kind =
              is_call ? LOOM_X86_RELOCATION_CALL : LOOM_X86_RELOCATION_ADDRESS,
          .target_symbol_index = symbol_indices[prepared->reference],
          .addend = -4,
      };
    } else if (prepared->reference != UINT32_MAX) {
      fixups[fixup_count++] = (loom_x86_branch_fixup_t){
          .offset = iree_io_stream_offset(stream) + instruction.length - 4,
          .target = prepared->reference,
      };
    }
    status =
        iree_io_stream_write(stream, instruction.length, instruction.bytes);
  }
  while (block <= function->block_count) {
    block_offsets[block++] = iree_io_stream_offset(stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_function_write_stack_leave(function, &writer);
  }
  for (uint8_t i = 16; i > 0 && iree_status_is_ok(status); --i) {
    const uint8_t reg = i - 1;
    if (function->saved_registers & (1u << reg)) {
      status = loom_x86_function_write_stack(&writer,
                                             LOOM_X86_ENCODING_FORM_POP, reg);
    }
  }
  if (iree_status_is_ok(status) && function->may_dirty_upper_vector_state &&
      !function->has_upper_vector_result) {
    status = loom_x86_function_write_upper_vector_state_cleanup(&writer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_function_write_stack(&writer,
                                           LOOM_X86_ENCODING_FORM_RETURN, 0);
  }
  const iree_io_stream_pos_t end = iree_io_stream_offset(stream);
  for (iree_host_size_t i = 0; i < fixup_count && iree_status_is_ok(status);
       ++i) {
    const int64_t displacement =
        block_offsets[fixups[i].target] - fixups[i].offset - 4;
    if (displacement < INT32_MIN || displacement > INT32_MAX) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "x86 branch exceeds signed 32-bit displacement");
    } else {
      uint8_t bytes[4];
      iree_unaligned_store_le_u32(bytes, (uint32_t)displacement);
      status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                                   fixups[i].offset);
      if (iree_status_is_ok(status)) {
        status = iree_io_stream_write(stream, 4, bytes);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, end);
  }
  if (iree_status_is_ok(status) && out_instruction_count) {
    *out_instruction_count = writer.instruction_count;
  }
  return status;
}
