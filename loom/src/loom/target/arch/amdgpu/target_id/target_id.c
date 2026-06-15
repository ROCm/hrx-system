// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/target_id/target_id.h"

#include <string.h>

#include "loom/codegen/low/target_binding.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/target.h"

const iree_string_view_t loom_amdgpu_amdhsa_target_id_prefix =
    IREE_SVL("amdgcn-amd-amdhsa--");

const loom_amdgpu_processor_info_t* loom_amdgpu_target_processor_from_op(
    const loom_module_t* module, const loom_op_t* target_op) {
  if (!loom_amdgpu_target_isa(target_op)) {
    return NULL;
  }
  return loom_amdgpu_target_record_processor(module, target_op);
}

const loom_amdgpu_processor_info_t* loom_amdgpu_target_processor_from_ref(
    const loom_module_t* module, loom_symbol_ref_t target_ref) {
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
      target_ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[target_ref.symbol_id];
  return loom_amdgpu_target_processor_from_op(module, symbol->defining_op);
}

const loom_amdgpu_processor_info_t*
loom_amdgpu_target_processor_from_resolved_target(
    const loom_module_t* module, const loom_low_resolved_target_t* target) {
  return loom_amdgpu_target_processor_from_op(module, target->target_op);
}

iree_status_t loom_amdgpu_amdhsa_target_id_append(
    const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t feature_suffix, iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, loom_amdgpu_amdhsa_target_id_prefix));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, processor->name));
  if (!iree_string_view_is_empty(feature_suffix)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, IREE_SV(":")));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, feature_suffix));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_amdhsa_target_id_format(
    const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t feature_suffix, iree_arena_allocator_t* arena,
    iree_string_view_t* out_target_id) {
  *out_target_id = iree_string_view_empty();

  const iree_host_size_t separator_length =
      iree_string_view_is_empty(feature_suffix) ? 0 : 1;
  iree_host_size_t prefix_processor_length = 0;
  iree_host_size_t prefix_processor_separator_length = 0;
  iree_host_size_t target_id_length = 0;
  if (!iree_host_size_checked_add(loom_amdgpu_amdhsa_target_id_prefix.size,
                                  processor->name.size,
                                  &prefix_processor_length) ||
      !iree_host_size_checked_add(prefix_processor_length, separator_length,
                                  &prefix_processor_separator_length) ||
      !iree_host_size_checked_add(prefix_processor_separator_length,
                                  feature_suffix.size, &target_id_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU AMDHSA target-id length overflows");
  }

  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, target_id_length, (void**)&data));
  char* cursor = data;
  memcpy(cursor, loom_amdgpu_amdhsa_target_id_prefix.data,
         loom_amdgpu_amdhsa_target_id_prefix.size);
  cursor += loom_amdgpu_amdhsa_target_id_prefix.size;
  memcpy(cursor, processor->name.data, processor->name.size);
  cursor += processor->name.size;
  if (!iree_string_view_is_empty(feature_suffix)) {
    *cursor++ = ':';
    memcpy(cursor, feature_suffix.data, feature_suffix.size);
  }

  *out_target_id = iree_make_string_view(data, target_id_length);
  return iree_ok_status();
}
