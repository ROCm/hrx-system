// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/native_call_plan.h"

#include <string.h>

#include "loom/ir/module.h"

iree_status_t loom_vm_native_call_template_build(
    const loom_module_t* module, loom_named_attr_slice_t layout,
    uint16_t argument_count, uint16_t result_count,
    iree_arena_allocator_t* arena,
    loom_vm_native_call_template_t** out_template) {
  *out_template = NULL;
  loom_attribute_t record = {0}, arguments = {0}, results = {0}, version = {0};
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
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown vm_native ABI layout field '%.*s'",
                              (int)name.size, name.data);
    }
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
              .module_name_u16 = call_template->module_name,
              .export_name_u16 = call_template->export_name,
              .version_u32 = call_template->version,
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
