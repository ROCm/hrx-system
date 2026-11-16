// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/signature_flags.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

iree_const_byte_span_t loom_func_like_signature_flags(
    loom_func_like_t function) {
  if (!function.vtable ||
      function.vtable->signature_flags_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_const_byte_span_empty();
  }
  return loom_attr_as_bytes(loom_op_const_attrs(
      function.op)[function.vtable->signature_flags_attr_index]);
}

iree_status_t loom_func_like_prune_signature_flags(
    loom_module_t* module, loom_func_like_t function, uint16_t argument_count,
    uint16_t result_count, const bool* removed_arguments,
    const bool* removed_results) {
  const iree_const_byte_span_t source =
      loom_func_like_signature_flags(function);
  if (!source.data_length) {
    return iree_ok_status();
  }
  uint8_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, source.data_length, sizeof(*values), (void**)&values));
  uint32_t count = 0;
  uint8_t present = 0;
  for (uint32_t i = 0; i < argument_count; ++i) {
    if (!removed_arguments || !removed_arguments[i]) {
      values[count++] = source.data[i];
      present |= source.data[i];
    }
  }
  for (uint32_t i = 0; i < result_count; ++i) {
    if (!removed_results || !removed_results[i]) {
      values[count++] = source.data[argument_count + i];
      present |= source.data[argument_count + i];
    }
  }
  return loom_op_set_attr(
      module, function.op, function.vtable->signature_flags_attr_index,
      present ? loom_attr_signature_flags(values, count) : loom_attr_absent());
}

static void loom_signature_flags_project_values(
    const loom_module_t* module, const uint8_t* source, uint16_t source_count,
    const uint16_t* indices, const loom_value_id_t* target_values,
    uint16_t target_count, uint8_t* target, uint8_t* present) {
  for (uint32_t i = 0; i < source_count; ++i) {
    const uint16_t begin = indices[i];
    const uint16_t end = i + 1 < source_count ? indices[i + 1] : target_count;
    for (uint32_t j = begin; j < end; ++j) {
      const loom_type_t type = loom_module_value_type(module, target_values[j]);
      target[j] =
          loom_type_is_buffer(type) || loom_type_is_view(type) ? source[i] : 0;
      *present |= target[j];
    }
  }
}

iree_status_t loom_func_like_project_signature_flags(
    loom_module_t* module, loom_func_like_t source, loom_func_like_t target,
    uint16_t source_argument_count, uint16_t source_result_count,
    const uint16_t* argument_indices, const uint16_t* result_indices) {
  const iree_const_byte_span_t flags = loom_func_like_signature_flags(source);
  if (!flags.data_length) {
    return iree_ok_status();
  }
  uint16_t target_argument_count = 0;
  const loom_value_id_t* target_arguments =
      loom_func_like_arg_ids(target, &target_argument_count);
  const uint32_t count =
      (uint32_t)target_argument_count + target.op->result_count;
  uint8_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, count, sizeof(*values), (void**)&values));
  uint8_t present = 0;
  loom_signature_flags_project_values(module, flags.data, source_argument_count,
                                      argument_indices, target_arguments,
                                      target_argument_count, values, &present);
  loom_signature_flags_project_values(
      module, flags.data + source_argument_count, source_result_count,
      result_indices, loom_op_const_results(target.op), target.op->result_count,
      values + target_argument_count, &present);
  return loom_op_set_attr(
      module, target.op, target.vtable->signature_flags_attr_index,
      present ? loom_attr_signature_flags(values, count) : loom_attr_absent());
}
