// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/native_call_plan.h"

#include <string.h>

#include "iree/vm/native_call.h"
#include "loom/ir/module.h"

static bool loom_vm_native_type_named(const loom_module_t* module,
                                      loom_type_t type,
                                      iree_string_view_t name) {
  return loom_type_is_dialect(type) &&
         iree_string_view_equal(
             loom_string_table_get(&module->strings,
                                   loom_type_dialect_name_id(type)),
             name);
}

iree_status_t loom_vm_native_call_template_build(
    const loom_module_t* module, loom_named_attr_slice_t layout,
    uint16_t argument_count, uint16_t result_count,
    iree_arena_allocator_t* arena,
    loom_vm_native_call_template_t** out_template) {
  *out_template = NULL;
  loom_attribute_t record = {0}, arguments = {0}, results = {0}, version = {0},
                   signature = {0};
  for (iree_host_size_t i = 0; i < layout.count; ++i) {
    const loom_named_attr_t* attr = &layout.entries[i];
    const iree_string_view_t name =
        loom_string_table_get(&module->strings, attr->name_id);
    if (iree_string_view_equal(name, IREE_SV("record"))) {
      record = attr->value;
    } else if (iree_string_view_equal(name, IREE_SV("arguments"))) {
      arguments = attr->value;
    } else if (iree_string_view_equal(name, IREE_SV("results"))) {
      results = attr->value;
    } else if (iree_string_view_equal(name, IREE_SV("version"))) {
      version = attr->value;
    } else if (iree_string_view_equal(name, IREE_SV("signature"))) {
      signature = attr->value;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown vm_native ABI layout field '%.*s'",
                              (int)name.size, name.data);
    }
  }
  if (signature.kind != LOOM_ATTR_TYPE ||
      !loom_vm_native_type_named(
          module, loom_type_table_get(&module->types, signature.type_id),
          IREE_SV("vm.record"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vm_native ABI requires a vm.record signature");
  }
  if (record.kind != LOOM_ATTR_I64_ARRAY ||
      arguments.kind != LOOM_ATTR_I64_ARRAY ||
      results.kind != LOOM_ATTR_I64_ARRAY || version.kind != LOOM_ATTR_I64 ||
      version.i64 < 0 || version.i64 > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "vm_native ABI layout requires record, arguments, results arrays and "
        "a u32 version");
  }
  if (arguments.count != argument_count || results.count != result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vm_native ABI layout must map every argument and "
                            "result exactly once");
  }
  const uint32_t operand_count = (uint32_t)argument_count + result_count;
  iree_host_size_t storage_size = 0, words_offset = 0, operands_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_vm_native_call_template_t), &storage_size,
      IREE_STRUCT_FIELD(record.count, uint32_t, &words_offset),
      IREE_STRUCT_FIELD(operand_count, uint32_t, &operands_offset)));
  loom_vm_native_call_template_t* call_template = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, storage_size, (void**)&call_template));
  uint32_t* words = (uint32_t*)((uint8_t*)call_template + words_offset);
  uint32_t* operands = (uint32_t*)((uint8_t*)call_template + operands_offset);
  *call_template = (loom_vm_native_call_template_t){
      .signature = signature.type_id,
      .words = words,
      .word_count = record.count,
      .operand_offsets = operands,
      .operand_count = operand_count,
      .version = (uint32_t)version.i64,
  };
  // Prototype words double as the cold patch-occupancy bitmap. A selector
  // placeholder must start at zero and is restored after all offsets pass.
  for (uint32_t i = 0; i < record.count; ++i) {
    if (record.i64_array[i] < 0 || record.i64_array[i] > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "vm_native record word exceeds u32");
    }
    words[i] = (uint32_t)record.i64_array[i];
  }
  for (uint32_t i = 0; i < operand_count; ++i) {
    const int64_t offset = i < argument_count
                               ? arguments.i64_array[i]
                               : results.i64_array[i - argument_count];
    if (offset < 0 || offset >= record.count || words[offset] != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "vm_native operand offsets must name distinct zero record words");
    }
    operands[i] = (uint32_t)offset;
    words[offset] = UINT32_MAX;
  }
  for (uint32_t i = 0; i < operand_count; ++i) {
    words[operands[i]] = 0;
  }
  *out_template = call_template;
  return iree_ok_status();
}

typedef struct loom_vm_native_layout_frame_t {
  // Enclosing declaration being constructed, or NULL for the root.
  struct loom_vm_native_layout_frame_t* parent;
  // Canonical type whose ordered parameters define this record.
  loom_type_t type;
  // Retained layout and field storage in the program arena.
  loom_vm_native_layout_plan_t* layout;
  // Next field to visit; a pending child leaves this index unchanged.
  uint32_t field_index;
} loom_vm_native_layout_frame_t;

static iree_status_t loom_vm_native_layout_push(
    loom_vm_native_call_plan_t* plan, loom_type_t type,
    iree_arena_allocator_t* scratch_arena,
    loom_vm_native_layout_frame_t** inout_frame) {
  loom_vm_native_layout_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(scratch_arena, sizeof(*frame), (void**)&frame));
  const uint16_t field_count = loom_type_dialect_param_count(type);
  iree_host_size_t storage_size = 0, fields_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_vm_native_layout_plan_t), &storage_size,
      IREE_STRUCT_FIELD(field_count, iree_vm_bytecode_v0_native_field_row_t,
                        &fields_offset)));
  loom_vm_native_layout_plan_t* layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, storage_size, (void**)&layout));
  *layout = (loom_vm_native_layout_plan_t){
      .row = {.field_count_u16 = field_count},
      .fields = (iree_vm_bytecode_v0_native_field_row_t*)((uint8_t*)layout +
                                                          fields_offset),
  };
  *frame = (loom_vm_native_layout_frame_t){
      .parent = *inout_frame, .type = type, .layout = layout};
  *inout_frame = frame;
  return iree_ok_status();
}

iree_status_t loom_vm_native_call_layout_bind(
    loom_vm_native_call_plan_t* plan, const loom_module_t* module,
    loom_vm_native_call_template_t* call_template,
    loom_vm_native_layout_plan_t** layouts_by_type,
    loom_vm_reference_plan_t* references,
    iree_arena_allocator_t* scratch_arena) {
  if (layouts_by_type[call_template->signature]) {
    call_template->layout_ordinal =
        layouts_by_type[call_template->signature]->ordinal;
    return iree_ok_status();
  }
  loom_vm_native_layout_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(loom_vm_native_layout_push(
      plan, loom_type_table_get(&module->types, call_template->signature),
      scratch_arena, &frame));
  iree_status_t status = iree_ok_status();
  while (frame && iree_status_is_ok(status)) {
    loom_vm_native_layout_plan_t* layout = frame->layout;
    if (frame->field_index == layout->row.field_count_u16) {
      if (plan->layout_count == 65536 ||
          layout->row.field_count_u16 > UINT32_MAX - plan->field_count) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "VM native declaration table exceeds wire limits");
        continue;
      }
      layout->ordinal = (uint16_t)plan->layout_count++;
      layout->row.field_base_u32 = plan->field_count;
      plan->field_count += layout->row.field_count_u16;
      if (plan->layout_tail) {
        plan->layout_tail->next = layout;
      } else {
        plan->layouts = layout;
      }
      plan->layout_tail = layout;
      layouts_by_type[loom_module_lookup_type_id(module, frame->type)] = layout;
      frame = frame->parent;
      continue;
    }
    loom_type_t type =
        loom_type_dialect_params(frame->type)[frame->field_index];
    iree_vm_bytecode_v0_native_field_row_t* field =
        &layout->fields[frame->field_index];
    *field = (iree_vm_bytecode_v0_native_field_row_t){0};
    if (loom_vm_native_type_named(module, type, IREE_SV("vm.range"))) {
      const loom_type_id_t type_id = loom_module_lookup_type_id(module, type);
      const loom_vm_native_layout_plan_t* child = layouts_by_type[type_id];
      if (!child) {
        status = loom_vm_native_layout_push(plan, type, scratch_arena, &frame);
        continue;
      }
      field->kind_u16 = IREE_VM_NATIVE_CALL_FIELD_RANGE;
      field->type_ordinal_u16 = child->ordinal;
      layout->row.word_count_u32 += 2;
      layout->row.depth_u16 =
          iree_max(layout->row.depth_u16, child->row.depth_u16 + 1u);
    } else {
      if (loom_vm_native_type_named(module, type, IREE_SV("vm.out"))) {
        if (loom_type_dialect_param_count(type) != 1) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "vm.out requires exactly one scalar or ref type");
          continue;
        }
        field->kind_u16 = IREE_VM_NATIVE_CALL_FIELD_RESULT;
        type = loom_type_dialect_params(type)[0];
      }
      const uint16_t kind = loom_vm_type_scalar_kind(type);
      if (kind) {
        field->kind_u16 |= kind;
      } else if (loom_type_is_buffer(type) ||
                 (loom_type_is_dialect(type) &&
                  !loom_type_dialect_param_count(type))) {
        field->kind_u16 |= IREE_VM_NATIVE_CALL_FIELD_REF;
        status = loom_vm_reference_plan_bind(
            references, module, type, scratch_arena, &field->type_ordinal_u16);
      } else {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "native record leaf requires a supported "
                                  "scalar or managed reference type");
      }
      ++layout->row.word_count_u32;
    }
    ++frame->field_index;
  }
  if (iree_status_is_ok(status)) {
    call_template->layout_ordinal =
        layouts_by_type[call_template->signature]->ordinal;
  }
  return status;
}

typedef struct loom_vm_native_record_frame_t {
  // Layout of each fixed record in this range.
  const loom_vm_native_layout_plan_t* layout;
  // Word offset of the current fixed record.
  uint32_t offset;
  // Number of records remaining, including the current record.
  uint32_t remaining;
  // Next field within the current record.
  uint32_t field_index;
  // Word offset of that field within the current record.
  uint32_t field_offset;
} loom_vm_native_record_frame_t;

iree_status_t loom_vm_native_call_template_validate(
    const loom_vm_native_call_template_t* call_template,
    const loom_vm_native_layout_plan_t* const* layouts,
    const iree_vm_bytecode_v0_signature_descriptor_row_t* operands,
    uint16_t argument_count, iree_arena_allocator_t* scratch_arena) {
  const loom_vm_native_layout_plan_t* root =
      layouts[call_template->layout_ordinal];
  uint32_t cursor = root->row.word_count_u32;
  if (cursor > call_template->word_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vm_native record is truncated");
  }
  uint32_t* operand_by_word = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, iree_max(call_template->word_count, 1u),
      sizeof(*operand_by_word), (void**)&operand_by_word));
  memset(operand_by_word, 0xFF,
         call_template->word_count * sizeof(*operand_by_word));
  for (uint32_t i = 0; i < call_template->operand_count; ++i) {
    operand_by_word[call_template->operand_offsets[i]] = i;
  }
  loom_vm_native_record_frame_t* stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, root->row.depth_u16 + 1u, sizeof(*stack), (void**)&stack));
  stack[0] = (loom_vm_native_record_frame_t){.layout = root, .remaining = 1};
  uint32_t depth = 1, leaf_count = 0;
  iree_status_t status = iree_ok_status();
  while (depth && iree_status_is_ok(status)) {
    loom_vm_native_record_frame_t* frame = &stack[depth - 1];
    if (frame->field_index == frame->layout->row.field_count_u16) {
      if (--frame->remaining == 0 || !frame->layout->row.field_count_u16) {
        --depth;
      } else {
        frame->offset += frame->layout->row.word_count_u32;
        frame->field_index = 0;
        frame->field_offset = 0;
      }
      continue;
    }
    const iree_vm_bytecode_v0_native_field_row_t field =
        frame->layout->fields[frame->field_index++];
    const uint32_t offset = frame->offset + frame->field_offset;
    if (field.kind_u16 == IREE_VM_NATIVE_CALL_FIELD_RANGE) {
      frame->field_offset += 2;
      const uint32_t count = call_template->words[offset];
      const uint32_t begin = call_template->words[offset + 1];
      const loom_vm_native_layout_plan_t* child =
          layouts[field.type_ordinal_u16];
      const uint64_t length = (uint64_t)count * child->row.word_count_u32;
      if (begin != cursor || length > call_template->word_count - cursor ||
          operand_by_word[offset] != UINT32_MAX ||
          operand_by_word[offset + 1] != UINT32_MAX) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "vm_native range has noncanonical extent or "
                                  "an operand in its header");
      } else {
        cursor += (uint32_t)length;
        if (count && child->row.field_count_u16) {
          stack[depth++] = (loom_vm_native_record_frame_t){
              .layout = child, .offset = begin, .remaining = count};
        }
      }
    } else {
      ++frame->field_offset;
      const uint32_t operand = operand_by_word[offset];
      const uint16_t kind = operand < call_template->operand_count
                                ? operands[operand].kind_u16 |
                                      (operand >= argument_count
                                           ? IREE_VM_NATIVE_CALL_FIELD_RESULT
                                           : 0)
                                : 0;
      if (operand == UINT32_MAX || field.kind_u16 != kind ||
          field.type_ordinal_u16 != operands[operand].type_ordinal_u16) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "vm_native record leaf does not match its "
                                  "logical operand type and direction");
      }
      ++leaf_count;
    }
  }
  if (iree_status_is_ok(status) &&
      (cursor != call_template->word_count ||
       leaf_count != call_template->operand_count)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "vm_native record does not cover exactly its logical operands");
  }
  return status;
}

iree_status_t loom_vm_native_call_plan_append(
    loom_vm_native_call_plan_t* plan,
    const loom_vm_native_call_template_t* call_template,
    uint16_t function_ordinal, uint32_t* out_ordinal, uint32_t** out_words) {
  if (plan->count == UINT32_MAX ||
      call_template->word_count > UINT32_MAX - plan->word_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM native callsite extent exceeds u32");
  }
  iree_host_size_t storage_size = 0, words_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_vm_native_callsite_plan_t), &storage_size,
      IREE_STRUCT_FIELD(call_template->word_count, uint32_t, &words_offset)));
  loom_vm_native_callsite_plan_t* callsite = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, storage_size, (void**)&callsite));
  uint32_t* words = (uint32_t*)((uint8_t*)callsite + words_offset);
  if (call_template->word_count) {
    memcpy(words, call_template->words,
           call_template->word_count * sizeof(*words));
  }
  *callsite = (loom_vm_native_callsite_plan_t){
      .row = {.function_ordinal_u16 = function_ordinal,
              .import_ordinal_u16 = call_template->import_ordinal,
              .word_offset_u32 = plan->word_count,
              .word_count_u32 = call_template->word_count},
      .words = words,
  };
  if (plan->tail) {
    plan->tail->next = callsite;
  } else {
    plan->head = callsite;
  }
  plan->tail = callsite;
  *out_ordinal = plan->count++;
  plan->word_count += call_template->word_count;
  *out_words = words;
  return iree_ok_status();
}
