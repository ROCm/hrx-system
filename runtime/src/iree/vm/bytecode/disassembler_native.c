// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/disassembler_native.h"

#include <inttypes.h>
#include <stdio.h>

typedef struct iree_vm_bytecode_disassembler_native_frame_t {
  // Layout of each record in this range.
  const iree_vm_native_call_layout_t* layout;
  // Word offset of the current fixed record.
  uint32_t offset;
  // Number of records left, including the current record.
  uint32_t remaining;
  // Next field to print in the current record.
  uint32_t field;
  // Word offset of that field within the current record.
  uint32_t field_offset;
} iree_vm_bytecode_disassembler_native_frame_t;

iree_status_t iree_vm_bytecode_disassembler_native_call(
    const iree_vm_native_call_schema_t* schema,
    const iree_vm_bytecode_v0_function_row_t* function, const uint32_t* words,
    iree_vm_bytecode_disassembler_write_callback_t output,
    iree_allocator_t scratch_allocator) {
  const iree_vm_native_call_layout_t* root =
      &schema->layouts[schema->root_ordinal];
  iree_vm_bytecode_disassembler_native_frame_t inline_stack[16];
  iree_vm_bytecode_disassembler_native_frame_t* stack = inline_stack;
  if (root->depth >= IREE_ARRAYSIZE(inline_stack)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        scratch_allocator, (root->depth + 1u) * sizeof(*stack),
        (void**)&stack));
  }
  stack[0] = (iree_vm_bytecode_disassembler_native_frame_t){.layout = root,
                                                            .remaining = 1};
  uint32_t depth = 1;
  iree_status_t status = output.fn(output.user_data, IREE_SV("("));
  while (depth && iree_status_is_ok(status)) {
    iree_vm_bytecode_disassembler_native_frame_t* frame = &stack[depth - 1];
    if (frame->field == frame->layout->field_count) {
      status = output.fn(output.user_data, IREE_SV(")"));
      if (--frame->remaining == 0) {
        --depth;
        if (depth && iree_status_is_ok(status)) {
          status = output.fn(output.user_data, IREE_SV("]"));
        }
      } else {
        frame->offset += frame->layout->word_count;
        frame->field = 0;
        frame->field_offset = 0;
        if (iree_status_is_ok(status)) {
          status = output.fn(output.user_data, IREE_SV(", ("));
        }
      }
      continue;
    }
    if (frame->field) {
      status = output.fn(output.user_data, IREE_SV(", "));
    }
    if (!iree_status_is_ok(status)) {
      continue;
    }
    const iree_vm_native_call_field_t field =
        schema->fields[frame->layout->field_base + frame->field++];
    const uint32_t* value = words + frame->offset + frame->field_offset;
    if (field.kind == IREE_VM_NATIVE_CALL_FIELD_RANGE) {
      frame->field_offset += 2;
      const iree_vm_native_call_layout_t* child =
          &schema->layouts[field.type_ordinal];
      if (!value[0]) {
        status = output.fn(output.user_data, IREE_SV("[]"));
      } else if (!child->field_count) {
        char text[32];
        const int length =
            snprintf(text, sizeof(text), "[() * %" PRIu32 "]", value[0]);
        status =
            output.fn(output.user_data, iree_make_string_view(text, length));
      } else {
        status = output.fn(output.user_data, IREE_SV("[("));
        stack[depth++] = (iree_vm_bytecode_disassembler_native_frame_t){
            .layout = child, .offset = value[1], .remaining = value[0]};
      }
    } else {
      ++frame->field_offset;
      uint32_t selector = *value;
      char bank = 'v', storage = '%';
      if ((field.kind & ~IREE_VM_NATIVE_CALL_FIELD_RESULT) ==
          IREE_VM_NATIVE_CALL_FIELD_REF) {
        bank = 'r';
        if (selector >= function->ref_register_count_u16) {
          storage = '#';
          selector -= function->ref_register_count_u16;
        }
      } else if (selector & IREE_VM_NATIVE_CALL_LOCAL_VALUE) {
        storage = '#';
        selector &= ~IREE_VM_NATIVE_CALL_LOCAL_VALUE;
      }
      char text[32];
      const int length =
          snprintf(text, sizeof(text), "%s%c%c%" PRIu32,
                   field.kind & IREE_VM_NATIVE_CALL_FIELD_RESULT ? "out " : "",
                   storage, bank, selector);
      status = output.fn(output.user_data, iree_make_string_view(text, length));
    }
  }
  if (stack != inline_stack) {
    iree_allocator_free(scratch_allocator, stack);
  }
  return status;
}
