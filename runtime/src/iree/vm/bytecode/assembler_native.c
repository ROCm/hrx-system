// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler_native.h"

enum {
  IREE_VM_ASSEMBLER_NATIVE_RECORD = 1,
  IREE_VM_ASSEMBLER_NATIVE_RANGE = 2,
  IREE_VM_ASSEMBLER_NATIVE_VALUE = 3,
  IREE_VM_ASSEMBLER_NATIVE_REF = 4,
  IREE_VM_ASSEMBLER_NATIVE_RESULT = 0x10,
  IREE_VM_ASSEMBLER_NATIVE_LOCAL = 0x20,
};

struct iree_vm_bytecode_assembler_native_operand_t {
  // Record/range/value/ref kind and selector direction/storage flags.
  uint32_t kind;
  // Enclosing group index, or UINT32_MAX for the root.
  uint32_t parent;
  // First index after this entire subtree.
  uint32_t end;
  // Number of fields in a record or records in a range.
  uint32_t count;
  // Selector index/offset, or the fixed word stride for a record.
  uint32_t value;
};

static iree_status_t iree_vm_bytecode_assembler_native_expect(
    iree_vm_bytecode_assembler_lexer_t* lexer, iree_string_view_t literal) {
  if (!iree_vm_bytecode_assembler_lexer_try_literal(lexer, literal)) {
    return iree_vm_bytecode_assembler_lexer_error(
        lexer, "expected native operand delimiter");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_native_parse(
    iree_vm_bytecode_assembler_lexer_t* lexer, iree_allocator_t host_allocator,
    iree_vm_bytecode_assembler_native_call_t** out_call) {
  *out_call = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_assembler_native_expect(lexer, IREE_SV("(")));
  iree_host_size_t capacity = 16;
  iree_vm_bytecode_assembler_native_operand_t* operands = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, capacity * sizeof(*operands), (void**)&operands));
  operands[0] = (iree_vm_bytecode_assembler_native_operand_t){
      .kind = IREE_VM_ASSEMBLER_NATIVE_RECORD, .parent = UINT32_MAX};
  uint32_t count = 1, frame = 0, word_count = 0;
  iree_status_t status = iree_ok_status();
  while (frame != UINT32_MAX && iree_status_is_ok(status)) {
    const bool is_range =
        operands[frame].kind == IREE_VM_ASSEMBLER_NATIVE_RANGE;
    const iree_string_view_t close = is_range ? IREE_SV("]") : IREE_SV(")");
    if (is_range && operands[frame].count == 1 &&
        iree_vm_bytecode_assembler_lexer_try_literal(lexer, IREE_SV("*"))) {
      uint64_t repetitions = 0;
      if (operands[frame + 1].count != 0) {
        status = iree_vm_bytecode_assembler_lexer_error(
            lexer, "only zero-field records may use repetition syntax");
      } else {
        status = iree_vm_bytecode_assembler_lexer_parse_unsigned(
            lexer, 4, IREE_VM_BYTECODE_ASSEMBLER_RADIX_DECIMAL, &repetitions);
      }
      if (iree_status_is_ok(status)) {
        status = iree_vm_bytecode_assembler_native_expect(lexer, close);
        operands[frame].count = (uint32_t)repetitions;
      }
      operands[frame].end = count;
      frame = operands[frame].parent;
      continue;
    }
    if (iree_vm_bytecode_assembler_lexer_try_literal(lexer, close)) {
      const uint32_t parent = operands[frame].parent;
      if (!is_range && parent != UINT32_MAX && frame != parent + 1 &&
          (operands[frame].count != operands[parent + 1].count ||
           operands[frame].value != operands[parent + 1].value)) {
        status = iree_vm_bytecode_assembler_lexer_error(
            lexer, "native range elements require the same fixed record shape");
      }
      operands[frame].end = count;
      frame = parent;
      continue;
    }
    if (operands[frame].count) {
      status = iree_vm_bytecode_assembler_native_expect(lexer, IREE_SV(","));
    }
    if (iree_status_is_ok(status) && count == capacity) {
      iree_host_size_t next_capacity = capacity * 2;
      if (next_capacity > UINT32_MAX ||
          next_capacity > IREE_HOST_SIZE_MAX / sizeof(*operands)) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "native operand tree exceeds host size");
      } else {
        status = iree_allocator_realloc(host_allocator,
                                        next_capacity * sizeof(*operands),
                                        (void**)&operands);
        capacity = next_capacity;
      }
    }
    if (!iree_status_is_ok(status)) {
      continue;
    }
    const uint32_t index = count++;
    iree_vm_bytecode_assembler_native_operand_t* node = &operands[index];
    *node = (iree_vm_bytecode_assembler_native_operand_t){.parent = frame,
                                                          .end = index + 1};
    ++operands[frame].count;
    if (is_range) {
      status = iree_vm_bytecode_assembler_native_expect(lexer, IREE_SV("("));
      node->kind = IREE_VM_ASSEMBLER_NATIVE_RECORD;
      frame = index;
      continue;
    }
    if (iree_vm_bytecode_assembler_lexer_try_literal(lexer, IREE_SV("["))) {
      node->kind = IREE_VM_ASSEMBLER_NATIVE_RANGE;
    } else {
      uint32_t flags = 0;
      if (iree_vm_bytecode_assembler_lexer_try_literal(lexer,
                                                       IREE_SV("out "))) {
        flags |= IREE_VM_ASSEMBLER_NATIVE_RESULT;
      }
      if (iree_vm_bytecode_assembler_lexer_try_literal(lexer, IREE_SV("#"))) {
        flags |= IREE_VM_ASSEMBLER_NATIVE_LOCAL;
      } else {
        status = iree_vm_bytecode_assembler_native_expect(lexer, IREE_SV("%"));
      }
      if (iree_status_is_ok(status)) {
        if (iree_vm_bytecode_assembler_lexer_try_literal(lexer, IREE_SV("v"))) {
          node->kind = IREE_VM_ASSEMBLER_NATIVE_VALUE | flags;
        } else if (iree_vm_bytecode_assembler_lexer_try_literal(lexer,
                                                                IREE_SV("r"))) {
          node->kind = IREE_VM_ASSEMBLER_NATIVE_REF | flags;
        } else {
          status = iree_vm_bytecode_assembler_lexer_error(
              lexer, "expected native scalar or ref selector");
        }
      }
      if (iree_status_is_ok(status)) {
        uint64_t value = 0;
        status = iree_vm_bytecode_assembler_lexer_parse_unsigned(
            lexer, 4, IREE_VM_BYTECODE_ASSEMBLER_RADIX_DECIMAL, &value);
        node->value = (uint32_t)value;
      }
    }
    const uint32_t width = node->kind == IREE_VM_ASSEMBLER_NATIVE_RANGE ? 2 : 1;
    if (iree_status_is_ok(status) && width > UINT32_MAX - word_count) {
      status = iree_vm_bytecode_assembler_lexer_error(
          lexer, "native operand word count exceeds u32");
    }
    if (iree_status_is_ok(status)) {
      operands[frame].value += width;
      word_count += width;
      if (node->kind == IREE_VM_ASSEMBLER_NATIVE_RANGE) {
        frame = index;
      }
    }
  }
  iree_vm_bytecode_assembler_native_call_t* call = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, sizeof(*call), (void**)&call);
  }
  if (iree_status_is_ok(status)) {
    call->operands = operands;
    call->word_count = word_count;
    *out_call = call;
  } else {
    iree_allocator_free(host_allocator, operands);
  }
  return status;
}

void iree_vm_bytecode_assembler_native_free(
    iree_vm_bytecode_assembler_native_call_t* call,
    iree_allocator_t host_allocator) {
  if (!call) {
    return;
  }
  iree_allocator_free(host_allocator, call->operands);
  iree_allocator_free(host_allocator, call);
}

typedef struct iree_vm_bytecode_assembler_native_frame_t {
  // Expected layout of each record.
  const iree_vm_native_call_layout_t* layout;
  // Current record node in the parsed operand tree.
  uint32_t record;
  // Next field node of the current record.
  uint32_t operand;
  // Next expected field index.
  uint32_t field;
  // Word offset of the next field in the current record.
  uint32_t offset;
  // Number of records left, including the current one.
  uint32_t remaining;
} iree_vm_bytecode_assembler_native_frame_t;

static iree_status_t iree_vm_bytecode_assembler_native_selector(
    const iree_vm_bytecode_assembler_native_operand_t* operand,
    iree_vm_native_call_field_t field,
    const iree_vm_bytecode_v0_function_row_t* function, uint32_t* out_word) {
  const bool is_ref = (field.kind & ~IREE_VM_NATIVE_CALL_FIELD_RESULT) ==
                      IREE_VM_NATIVE_CALL_FIELD_REF;
  const uint32_t kind = operand->kind & ~IREE_VM_ASSEMBLER_NATIVE_LOCAL;
  const uint32_t expected =
      (is_ref ? IREE_VM_ASSEMBLER_NATIVE_REF : IREE_VM_ASSEMBLER_NATIVE_VALUE) |
      ((field.kind & IREE_VM_NATIVE_CALL_FIELD_RESULT)
           ? IREE_VM_ASSEMBLER_NATIVE_RESULT
           : 0);
  if (kind != expected) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native selector domain or direction differs from its layout");
  }
  const bool is_local = (operand->kind & IREE_VM_ASSEMBLER_NATIVE_LOCAL) != 0;
  const uint32_t limit = is_ref
                             ? (is_local ? function->local_ref_count_u32
                                         : function->ref_register_count_u16)
                             : (is_local ? function->local_byte_length_u16
                                         : function->value_register_count_u16);
  const uint32_t width = is_local && !is_ref ? sizeof(uint64_t) : 1;
  if (operand->value > limit || width > limit - operand->value) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "native selector is outside its declared storage bank");
  }
  *out_word = operand->value;
  if (is_local) {
    if (is_ref) {
      *out_word += function->ref_register_count_u16;
    } else {
      *out_word |= IREE_VM_NATIVE_CALL_LOCAL_VALUE;
    }
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_assembler_native_encode(
    const iree_vm_bytecode_assembler_native_call_t* call,
    const iree_vm_native_call_schema_t* schema,
    const iree_vm_bytecode_v0_function_row_t* function, uint32_t* words,
    iree_allocator_t scratch_allocator) {
  const iree_vm_native_call_layout_t* root =
      &schema->layouts[schema->root_ordinal];
  iree_vm_bytecode_assembler_native_frame_t* stack = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      scratch_allocator, (root->depth + 1u) * sizeof(*stack), (void**)&stack));
  stack[0] = (iree_vm_bytecode_assembler_native_frame_t){
      .layout = root, .operand = 1, .remaining = 1};
  uint32_t depth = 1, cursor = root->word_count;
  iree_status_t status = iree_ok_status();
  while (depth && iree_status_is_ok(status)) {
    iree_vm_bytecode_assembler_native_frame_t* frame = &stack[depth - 1];
    const iree_vm_bytecode_assembler_native_operand_t* record =
        &call->operands[frame->record];
    if (record->count != frame->layout->field_count ||
        record->value != frame->layout->word_count) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "native record field count differs from its layout");
      continue;
    }
    if (frame->field == frame->layout->field_count) {
      if (--frame->remaining == 0) {
        --depth;
      } else {
        frame->record = record->end;
        frame->operand = frame->record + 1;
        frame->field = 0;
      }
      continue;
    }
    const uint32_t operand_index = frame->operand;
    const iree_vm_bytecode_assembler_native_operand_t* operand =
        &call->operands[operand_index];
    frame->operand = operand->end;
    const iree_vm_native_call_field_t field =
        schema->fields[frame->layout->field_base + frame->field++];
    if (field.kind == IREE_VM_NATIVE_CALL_FIELD_RANGE) {
      if (operand->kind != IREE_VM_ASSEMBLER_NATIVE_RANGE) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "native range requires grouped records");
        continue;
      }
      const iree_vm_native_call_layout_t* child =
          &schema->layouts[field.type_ordinal];
      const uint64_t length = (uint64_t)operand->count * child->word_count;
      if (cursor > call->word_count || length > call->word_count - cursor) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "native range shape differs from its layout");
        continue;
      }
      words[frame->offset++] = operand->count;
      words[frame->offset++] = cursor;
      const uint32_t offset = cursor;
      cursor += (uint32_t)length;
      if (child->field_count && operand->count) {
        stack[depth++] = (iree_vm_bytecode_assembler_native_frame_t){
            .layout = child,
            .record = operand_index + 1,
            .operand = operand_index + 2,
            .offset = offset,
            .remaining = operand->count};
      } else if (operand->end != operand_index + 1 &&
                 call->operands[operand_index + 1].count) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "native range requires zero-field records");
      }
    } else {
      status = iree_vm_bytecode_assembler_native_selector(
          operand, field, function, &words[frame->offset++]);
    }
  }
  iree_allocator_free(scratch_allocator, stack);
  return status;
}
