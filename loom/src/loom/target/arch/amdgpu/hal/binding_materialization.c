// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal/binding_materialization.h"

#include <inttypes.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/hal/binding_descriptor.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

#define LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_POINTER_HIGH_MASK UINT32_C(0x0000FFFF)
#define LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_ENABLE_BIT \
  UINT32_C(0x4000)
#define LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_WORD_SHIFT 16u

enum loom_amdgpu_hal_binding_load_flag_bits_e {
  // Allows a final same-width kernarg load to reuse the kernarg pointer SGPRs.
  LOOM_AMDGPU_HAL_BINDING_LOAD_FLAG_REUSE_KERNARG_STORAGE = 1u << 0,
};
typedef uint32_t loom_amdgpu_hal_binding_load_flags_t;

static iree_status_t loom_amdgpu_hal_binding_make_sgpr_type(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t unit_count, loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, unit_count, &type));
  return loom_module_intern_type(module, type, out_type);
}

static uint32_t loom_amdgpu_hal_binding_descriptor_range_word(int64_t extent) {
  if (extent > UINT32_MAX) {
    return UINT32_MAX;
  }
  return (uint32_t)extent;
}

static iree_string_view_t loom_amdgpu_hal_binding_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static const loom_low_descriptor_t*
loom_amdgpu_hal_binding_resolve_low_op_descriptor(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op) {
  const int64_t packet_ordinal = loom_low_op_descriptor_ordinal(op);
  if (packet_ordinal >= 0 && (uint64_t)packet_ordinal <= UINT32_MAX) {
    return loom_low_descriptor_set_descriptor_at(descriptor_set,
                                                 (uint32_t)packet_ordinal);
  }
  if (packet_ordinal != -1) {
    return NULL;
  }
  iree_string_view_t key =
      loom_amdgpu_hal_binding_module_string(module, loom_low_op_opcode(op));
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return NULL;
  }
  return loom_low_descriptor_set_descriptor_at(descriptor_set,
                                               descriptor_ordinal);
}

static iree_status_t loom_amdgpu_hal_binding_cache_swizzle_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_buffer_resource_cache_swizzle_t* out_kind) {
  *out_kind = LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE;
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
      descriptor_set->descriptor_set_ordinal, &descriptor_set_info));
  *out_kind = descriptor_set_info->buffer_resource.cache_swizzle;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_insert_kernarg_live_in(
    loom_rewriter_t* rewriter, loom_op_t* function_op, loom_type_t sgpr_x2_type,
    loom_value_id_t* out_value, loom_op_t** out_live_in_op) {
  *out_value = LOOM_VALUE_ID_INVALID;
  *out_live_in_op = NULL;
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      rewriter->module,
      IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE),
      &source_id));
  loom_block_t* entry_block =
      loom_region_entry_block(loom_low_function_body(function_op));
  if (entry_block->first_op != NULL) {
    loom_builder_set_before(&rewriter->builder, entry_block->first_op);
  } else {
    loom_builder_set_block(&rewriter->builder, entry_block);
  }
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_live_in_build(
      &rewriter->builder, source_id, loom_make_named_attr_slice(NULL, 0),
      sgpr_x2_type, function_op->location, &live_in_op));
  *out_value = loom_low_live_in_result(live_in_op);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(rewriter->module,
                                                 IREE_SV("kernarg"), &name_id));
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_name(rewriter->module, *out_value, name_id));
  *out_live_in_op = live_in_op;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_get_kernarg_live_in(
    loom_rewriter_t* rewriter, loom_op_t* function_op, loom_type_t sgpr_x2_type,
    loom_value_id_t* out_value, bool* out_inserted) {
  *out_value = LOOM_VALUE_ID_INVALID;
  *out_inserted = false;
  loom_block_t* entry_block =
      loom_region_entry_block(loom_low_function_body(function_op));
  loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (!loom_low_live_in_isa(op)) {
      continue;
    }
    const loom_value_id_t live_in_value = loom_low_live_in_result(op);
    if (loom_amdgpu_hal_kernel_abi_is_kernarg_segment_ptr_live_in(
            rewriter->module, live_in_value)) {
      *out_value = live_in_value;
      return iree_ok_status();
    }
  }
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_insert_kernarg_live_in(
      rewriter, function_op, sgpr_x2_type, out_value, &live_in_op));
  *out_inserted = true;
  return iree_ok_status();
}

static loom_op_t* loom_amdgpu_hal_binding_first_non_live_in(
    loom_op_t* function_op) {
  loom_block_t* entry_block =
      loom_region_entry_block(loom_low_function_body(function_op));
  loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (!loom_low_live_in_isa(op)) {
      return op;
    }
  }
  return NULL;
}

static void loom_amdgpu_hal_binding_set_entry_insertion_point(
    loom_rewriter_t* rewriter, loom_op_t* function_op) {
  loom_op_t* first_non_live_in =
      loom_amdgpu_hal_binding_first_non_live_in(function_op);
  if (first_non_live_in != NULL) {
    loom_builder_set_before(&rewriter->builder, first_non_live_in);
    return;
  }
  loom_builder_set_block(
      &rewriter->builder,
      loom_region_entry_block(loom_low_function_body(function_op)));
}

static iree_status_t loom_amdgpu_hal_binding_i64_attr(
    loom_module_t* module, iree_string_view_t name, int64_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_resolve_descriptor_ref(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor,
    loom_string_id_t* out_opcode_id) {
  *out_descriptor = NULL;
  *out_opcode_id = LOOM_STRING_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  if (descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AMDGPU HAL binding materialization references missing descriptor ref "
        "%" PRIu16,
        descriptor_ref);
  }
  iree_string_view_t key = loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_offset);
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL binding materialization descriptor ref %" PRIu16
        " has no descriptor key",
        descriptor_ref);
  }
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(rewriter->module, key, out_opcode_id));
  *out_descriptor = descriptor;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_s_mov_b32(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    int64_t value, loom_type_t sgpr_type, loom_location_id_t location,
    loom_value_id_t* out_value, loom_op_t** out_op) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (out_op != NULL) {
    *out_op = NULL;
  }
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_i64_attr(
      rewriter->module, IREE_SV("imm32"), value, &attr));
  loom_op_t* const_op = NULL;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_resolve_descriptor_ref(
      rewriter, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      &descriptor, &opcode_id));
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      &rewriter->builder, descriptor_set, descriptor, opcode_id,
      loom_make_named_attr_slice(&attr, 1), sgpr_type, location, &const_op));
  *out_value = loom_low_const_result(const_op);
  if (out_op != NULL) {
    *out_op = const_op;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_low_slice(
    loom_rewriter_t* rewriter, loom_value_id_t source, uint32_t offset,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(&rewriter->builder, source, offset,
                                            result_type, location, &slice_op));
  *out_value = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_s_binary_b32(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  const loom_type_t result_types[] = {result_type};
  loom_op_t* binary_op = NULL;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_resolve_descriptor_ref(
      rewriter, descriptor_set, descriptor_ref, &descriptor, &opcode_id));
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set, descriptor, opcode_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &binary_op));
  *out_value = loom_value_slice_get(loom_low_op_results(binary_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_s_load_dwordx2(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t kernarg_ptr, uint32_t kernarg_offset,
    loom_type_t sgpr_x2_type, loom_amdgpu_hal_binding_load_flags_t load_flags,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_i64_attr(
      rewriter->module, IREE_SV("offset"), kernarg_offset, &attr));
  const loom_value_id_t operands[] = {kernarg_ptr};
  const loom_type_t result_types[] = {sgpr_x2_type};
  const loom_tied_result_t tied_kernarg_storage[] = {{
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  }};
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  if (iree_any_bit_set(
          load_flags,
          LOOM_AMDGPU_HAL_BINDING_LOAD_FLAG_REUSE_KERNARG_STORAGE)) {
    tied_results = tied_kernarg_storage;
    tied_result_count = IREE_ARRAYSIZE(tied_kernarg_storage);
  }
  loom_op_t* load_op = NULL;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_resolve_descriptor_ref(
      rewriter, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, &descriptor,
      &opcode_id));
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set, descriptor, opcode_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&attr, 1),
      result_types, IREE_ARRAYSIZE(result_types), tied_results,
      tied_result_count, location, &load_op));
  *out_value = loom_low_op_results(load_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_s_load_dwordx4(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t kernarg_ptr, uint32_t kernarg_offset,
    loom_type_t sgpr_x4_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_i64_attr(
      rewriter->module, IREE_SV("offset"), kernarg_offset, &attr));
  const loom_value_id_t operands[] = {kernarg_ptr};
  const loom_type_t result_types[] = {sgpr_x4_type};
  loom_op_t* load_op = NULL;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_resolve_descriptor_ref(
      rewriter, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX4_OFFSET_ONLY, &descriptor,
      &opcode_id));
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set, descriptor, opcode_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&attr, 1),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_low_op_results(load_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_scalar_load(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t kernarg_ptr, uint32_t kernarg_offset, loom_type_t sgpr_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_i64_attr(
      rewriter->module, IREE_SV("offset"), kernarg_offset, &attr));
  const loom_value_id_t operands[] = {kernarg_ptr};
  const loom_type_t result_types[] = {sgpr_type};
  loom_op_t* load_op = NULL;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_resolve_descriptor_ref(
      rewriter, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, &descriptor,
      &opcode_id));
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set, descriptor, opcode_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&attr, 1),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_low_op_results(load_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_descriptor_pointer(
    loom_rewriter_t* rewriter, bool has_cache_swizzle,
    uint32_t cache_swizzle_stride,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t loaded_pointer, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type, loom_location_id_t location,
    loom_value_id_t* out_pointer) {
  *out_pointer = LOOM_VALUE_ID_INVALID;
  loom_value_id_t pointer_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
      rewriter, loaded_pointer, 0, sgpr_type, location, &pointer_low));
  loom_value_id_t pointer_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
      rewriter, loaded_pointer, 1, sgpr_type, location, &pointer_high));

  loom_value_id_t pointer_high_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
      rewriter, descriptor_set,
      LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_POINTER_HIGH_MASK, sgpr_type, location,
      &pointer_high_mask, NULL));
  loom_value_id_t masked_pointer_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_binary_b32(
      rewriter, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
      pointer_high, pointer_high_mask, sgpr_type, location,
      &masked_pointer_high));

  loom_value_id_t descriptor_pointer_high = masked_pointer_high;
  if (has_cache_swizzle) {
    loom_amdgpu_buffer_resource_cache_swizzle_t cache_swizzle_kind =
        LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_cache_swizzle_kind(
        descriptor_set, &cache_swizzle_kind));
    if (cache_swizzle_kind !=
        LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU HAL binding materialization reached an unverified cache "
          "swizzle descriptor");
    }

    const uint32_t cache_swizzle_word =
        ((cache_swizzle_stride |
          LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_ENABLE_BIT)
         << LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_WORD_SHIFT);
    loom_value_id_t cache_swizzle = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
        rewriter, descriptor_set, cache_swizzle_word, sgpr_type, location,
        &cache_swizzle, NULL));
    loom_value_id_t swizzled_pointer_high = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_binary_b32(
        rewriter, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32,
        masked_pointer_high, cache_swizzle, sgpr_type, location,
        &swizzled_pointer_high));
    descriptor_pointer_high = swizzled_pointer_high;
  }

  const loom_value_id_t sources[] = {pointer_low, descriptor_pointer_high};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      &rewriter->builder, sources, IREE_ARRAYSIZE(sources), sgpr_x2_type,
      location, &concat_op));
  *out_pointer = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_materialize_one(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_amdgpu_hal_kernarg_resource_t* resource,
    loom_value_id_t kernarg_ptr,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_x2_type,
    loom_amdgpu_hal_binding_load_flags_t load_flags) {
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_set_before(&rewriter->builder, op);

  loom_value_id_t pointer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx2(
      rewriter, descriptor_set, kernarg_ptr, resource->kernarg_offset,
      sgpr_x2_type, load_flags, op->location, &pointer));

  loom_value_id_t replacement = pointer;
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static bool loom_amdgpu_hal_binding_resource_is_used(
    const loom_module_t* module,
    const loom_amdgpu_hal_kernarg_resource_t* resource) {
  const loom_value_id_t result =
      loom_low_resource_result(resource->resource_op);
  const loom_value_t* value = loom_module_value(module, result);
  return value->use_count != 0 ||
         loom_module_value_has_type_uses(module, result);
}

static bool loom_amdgpu_hal_binding_has_used_resources_from(
    const loom_module_t* module,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_host_size_t start_index) {
  for (iree_host_size_t i = start_index; i < layout->resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource = &layout->resources[i];
    const loom_op_t* resource_op = resource->resource_op;
    if (iree_any_bit_set(resource_op->flags, LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    if (loom_amdgpu_hal_binding_resource_is_used(module, resource)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_hal_binding_can_pair_resource_load(
    const loom_module_t* module,
    const loom_amdgpu_hal_kernarg_resource_t* first_resource,
    const loom_amdgpu_hal_kernarg_resource_t* second_resource,
    loom_type_t sgpr_x2_type) {
  if (!loom_amdgpu_hal_binding_resource_is_used(module, first_resource) ||
      !loom_amdgpu_hal_binding_resource_is_used(module, second_resource)) {
    return false;
  }
  if (!loom_type_equal(first_resource->abi_type, sgpr_x2_type) ||
      !loom_type_equal(second_resource->abi_type, sgpr_x2_type)) {
    return false;
  }
  const loom_op_t* first_op = first_resource->resource_op;
  const loom_op_t* second_op = second_resource->resource_op;
  if (first_op->parent_block == NULL ||
      first_op->parent_block != second_op->parent_block ||
      first_op->block_ordinal >= second_op->block_ordinal) {
    return false;
  }
  return first_resource->kernarg_size ==
             LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE &&
         second_resource->kernarg_size ==
             LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE &&
         first_resource->kernarg_offset + first_resource->kernarg_size ==
             second_resource->kernarg_offset;
}

static iree_status_t loom_amdgpu_hal_binding_materialize_resource_value(
    loom_rewriter_t* rewriter, loom_op_t* resource_op,
    loom_value_id_t replacement, loom_value_id_t value_checkpoint) {
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, resource_op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, resource_op,
                                                  &replacement, 1);
}

static iree_status_t loom_amdgpu_hal_binding_materialize_resource_pair(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_hal_kernarg_resource_t* first_resource,
    const loom_amdgpu_hal_kernarg_resource_t* second_resource,
    loom_value_id_t kernarg_ptr, loom_type_t sgpr_x2_type,
    loom_type_t sgpr_x4_type) {
  loom_op_t* first_op = (loom_op_t*)first_resource->resource_op;
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_set_before(&rewriter->builder, first_op);

  loom_value_id_t loaded_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx4(
      rewriter, descriptor_set, kernarg_ptr, first_resource->kernarg_offset,
      sgpr_x4_type, first_op->location, &loaded_pair));

  loom_value_id_t first_pointer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
      rewriter, loaded_pair, /*offset=*/0, sgpr_x2_type, first_op->location,
      &first_pointer));

  loom_value_id_t second_pointer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
      rewriter, loaded_pair, /*offset=*/2, sgpr_x2_type,
      second_resource->resource_op->location, &second_pointer));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_resource_value(
      rewriter, first_op, first_pointer, value_checkpoint));
  return loom_amdgpu_hal_binding_materialize_resource_value(
      rewriter, (loom_op_t*)second_resource->resource_op, second_pointer,
      value_checkpoint);
}

static iree_status_t loom_amdgpu_hal_binding_move_value_name(
    loom_module_t* module, loom_value_id_t source, loom_value_id_t target) {
  return loom_module_move_value_name(module, source, target);
}

static bool loom_amdgpu_hal_binding_direct_arg_is_used(
    const loom_module_t* module,
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg) {
  const loom_value_t* arg_value = loom_module_value(module, direct_arg->arg_id);
  return arg_value->use_count != 0 ||
         loom_module_value_has_type_uses(module, direct_arg->arg_id);
}

static bool loom_amdgpu_hal_binding_has_used_direct_args_from(
    const loom_module_t* module,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_host_size_t start_index) {
  for (iree_host_size_t i = start_index; i < layout->direct_arg_count; ++i) {
    if (loom_amdgpu_hal_binding_direct_arg_is_used(module,
                                                   &layout->direct_args[i])) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_hal_binding_can_pair_direct_arg_load(
    const loom_module_t* module,
    const loom_amdgpu_hal_kernarg_direct_arg_t* first_arg,
    const loom_amdgpu_hal_kernarg_direct_arg_t* second_arg,
    loom_type_t sgpr_type) {
  if (!loom_amdgpu_hal_binding_direct_arg_is_used(module, first_arg) ||
      !loom_amdgpu_hal_binding_direct_arg_is_used(module, second_arg)) {
    return false;
  }
  if (!loom_type_equal(first_arg->abi_type, sgpr_type) ||
      !loom_type_equal(second_arg->abi_type, sgpr_type)) {
    return false;
  }
  return first_arg->kernarg_size == sizeof(uint32_t) &&
         second_arg->kernarg_size == sizeof(uint32_t) &&
         first_arg->kernarg_offset + first_arg->kernarg_size ==
             second_arg->kernarg_offset;
}

static iree_status_t loom_amdgpu_hal_binding_materialize_direct_arg_value(
    loom_rewriter_t* rewriter,
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg,
    loom_value_id_t loaded) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_move_value_name(
      rewriter->module, direct_arg->arg_id, loaded));
  return loom_value_replace_all_uses_with(rewriter->module, direct_arg->arg_id,
                                          loaded);
}

static iree_status_t loom_amdgpu_hal_binding_materialize_direct_arg_pair(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_hal_kernarg_direct_arg_t* first_arg,
    const loom_amdgpu_hal_kernarg_direct_arg_t* second_arg,
    loom_value_id_t kernarg_ptr, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type, loom_amdgpu_hal_binding_load_flags_t load_flags,
    loom_location_id_t location) {
  loom_value_id_t loaded_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx2(
      rewriter, descriptor_set, kernarg_ptr, first_arg->kernarg_offset,
      sgpr_x2_type, load_flags, location, &loaded_pair));

  loom_value_id_t first_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
      rewriter, loaded_pair, /*offset=*/0, sgpr_type, location, &first_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_direct_arg_value(
      rewriter, first_arg, first_value));

  loom_value_id_t second_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
      rewriter, loaded_pair, /*offset=*/1, sgpr_type, location, &second_value));
  return loom_amdgpu_hal_binding_materialize_direct_arg_value(
      rewriter, second_arg, second_value);
}

static iree_status_t loom_amdgpu_hal_binding_materialize_direct_arg_load(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg,
    loom_value_id_t kernarg_ptr, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type, loom_amdgpu_hal_binding_load_flags_t load_flags,
    loom_location_id_t location) {
  loom_value_id_t loaded = LOOM_VALUE_ID_INVALID;
  if (direct_arg->kernarg_size == sizeof(uint32_t) &&
      loom_type_equal(direct_arg->abi_type, sgpr_type)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_scalar_load(
        rewriter, descriptor_set, kernarg_ptr, direct_arg->kernarg_offset,
        sgpr_type, location, &loaded));
  } else if (direct_arg->kernarg_size == 2u * sizeof(uint32_t) &&
             loom_type_equal(direct_arg->abi_type, sgpr_x2_type)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx2(
        rewriter, descriptor_set, kernarg_ptr, direct_arg->kernarg_offset,
        sgpr_x2_type, load_flags, location, &loaded));
  } else {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU HAL binding materialization reached an unverified direct "
        "argument layout");
  }
  return loom_amdgpu_hal_binding_materialize_direct_arg_value(
      rewriter, direct_arg, loaded);
}

static iree_status_t loom_amdgpu_hal_binding_materialize_direct_args(
    loom_rewriter_t* rewriter, loom_op_t* function_op,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_value_id_t kernarg_ptr,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type,
    loom_amdgpu_hal_binding_load_flags_t available_load_flags,
    iree_host_size_t* out_materialized_count) {
  *out_materialized_count = 0;
  if (layout->direct_arg_count == 0) {
    return iree_ok_status();
  }

  loom_block_t* entry_block =
      loom_region_entry_block(loom_low_function_body(function_op));
  const bool has_resource_loads =
      loom_amdgpu_hal_binding_has_used_resources_from(rewriter->module, layout,
                                                      /*start_index=*/0);
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg =
        &layout->direct_args[i];
    if (!loom_amdgpu_hal_binding_direct_arg_is_used(rewriter->module,
                                                    direct_arg)) {
      continue;
    }
    if (kernarg_ptr == LOOM_VALUE_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU HAL binding materialization direct argument requires an "
          "available kernarg segment pointer");
    }

    if (i + 1 < layout->direct_arg_count &&
        loom_amdgpu_hal_binding_can_pair_direct_arg_load(
            rewriter->module, direct_arg, &layout->direct_args[i + 1],
            sgpr_type)) {
      loom_amdgpu_hal_binding_load_flags_t load_flags = 0;
      if (available_load_flags != 0 && !has_resource_loads &&
          !loom_amdgpu_hal_binding_has_used_direct_args_from(rewriter->module,
                                                             layout, i + 2)) {
        load_flags = available_load_flags;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_direct_arg_pair(
          rewriter, descriptor_set, direct_arg, &layout->direct_args[i + 1],
          kernarg_ptr, sgpr_type, sgpr_x2_type, load_flags,
          function_op->location));
      *out_materialized_count += 2;
      ++i;
      continue;
    }

    loom_amdgpu_hal_binding_load_flags_t load_flags = 0;
    if (available_load_flags != 0 && !has_resource_loads &&
        !loom_amdgpu_hal_binding_has_used_direct_args_from(rewriter->module,
                                                           layout, i + 1)) {
      load_flags = available_load_flags;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_direct_arg_load(
        rewriter, descriptor_set, direct_arg, kernarg_ptr, sgpr_type,
        sgpr_x2_type, load_flags, function_op->location));
    ++*out_materialized_count;
  }

  for (iree_host_size_t i = layout->direct_arg_count; i > 0; --i) {
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg =
        &layout->direct_args[i - 1];
    if (direct_arg->argument_index >= entry_block->arg_count) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU HAL binding materialization direct argument index was "
          "invalidated before removal");
    }
    if (loom_block_arg_id(entry_block, direct_arg->argument_index) !=
        direct_arg->arg_id) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU HAL binding materialization direct argument ordering changed "
          "before removal");
    }
    IREE_RETURN_IF_ERROR(loom_block_remove_arg(rewriter->module, entry_block,
                                               direct_arg->argument_index));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_materialize_resources(
    loom_rewriter_t* rewriter,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_value_id_t kernarg_ptr,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_x2_type,
    loom_type_t sgpr_x4_type,
    loom_amdgpu_hal_binding_load_flags_t available_load_flags,
    iree_host_size_t* out_materialized_count) {
  *out_materialized_count = 0;
  for (iree_host_size_t i = 0; i < layout->resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource = &layout->resources[i];
    loom_op_t* resource_op = (loom_op_t*)resource->resource_op;
    if (iree_any_bit_set(resource_op->flags, LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    if (!loom_amdgpu_hal_binding_resource_is_used(rewriter->module, resource)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, resource_op));
      continue;
    }
    if (kernarg_ptr == LOOM_VALUE_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU HAL binding materialization resource requires an available "
          "kernarg segment pointer");
    }

    if (i + 1 < layout->resource_count &&
        loom_amdgpu_hal_binding_can_pair_resource_load(
            rewriter->module, resource, &layout->resources[i + 1],
            sgpr_x2_type)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_resource_pair(
          rewriter, descriptor_set, resource, &layout->resources[i + 1],
          kernarg_ptr, sgpr_x2_type, sgpr_x4_type));
      *out_materialized_count += 2;
      ++i;
      continue;
    }

    loom_amdgpu_hal_binding_load_flags_t load_flags = 0;
    if (available_load_flags != 0 &&
        !loom_amdgpu_hal_binding_has_used_resources_from(rewriter->module,
                                                         layout, i + 1)) {
      load_flags = available_load_flags;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_one(
        rewriter, resource_op, resource, kernarg_ptr, descriptor_set,
        sgpr_x2_type, load_flags));
    ++*out_materialized_count;
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_hal_binding_materialize_buffer_descriptor_pseudo(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_target_bundle_t* target_bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type) {
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_set_before(&rewriter->builder, op);

  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  const int64_t cache_swizzle_stride_attr =
      attrs.entries[LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE]
          .value.i64;
  const uint32_t cache_swizzle_stride = (uint32_t)cache_swizzle_stride_attr;
  const bool has_cache_swizzle = cache_swizzle_stride != 0;

  loom_value_id_t pointer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_descriptor_pointer(
      rewriter, has_cache_swizzle, cache_swizzle_stride, descriptor_set,
      operands.values[0], sgpr_type, sgpr_x2_type, op->location, &pointer));
  loom_value_id_t range = LOOM_VALUE_ID_INVALID;
  const bool has_dynamic_extent =
      descriptor ==
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR_EXTENT);
  if (has_dynamic_extent) {
    range = operands.values[1];
  } else {
    const int64_t extent =
        attrs.entries[LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_EXTENT].value.i64;
    const uint32_t range_word =
        loom_amdgpu_hal_binding_descriptor_range_word(extent);
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
        rewriter, descriptor_set, range_word, sgpr_type, op->location, &range,
        NULL));
  }
  loom_value_id_t flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
      rewriter, descriptor_set,
      target_bundle->export_plan->hal_kernel.buffer_resource_flags, sgpr_type,
      op->location, &flags, NULL));

  const loom_value_id_t sources[] = {pointer, range, flags};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      &rewriter->builder, sources, IREE_ARRAYSIZE(sources),
      loom_module_value_type(rewriter->module, results.values[0]), op->location,
      &concat_op));

  loom_value_id_t replacement = loom_low_concat_result(concat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t
loom_amdgpu_hal_binding_materialize_buffer_descriptors_with_types(
    loom_rewriter_t* rewriter, loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type, iree_host_size_t* out_materialized_count) {
  *out_materialized_count = 0;
  loom_region_t* body = loom_low_function_body(function_op);
  if (body == NULL) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU HAL binding materialization reached a low function without a "
        "body");
  }
  iree_status_t status = iree_ok_status();
  const loom_low_descriptor_t* static_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR);
  const loom_low_descriptor_t* dynamic_extent_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR_EXTENT);
  for (uint16_t block_index = 0;
       iree_status_is_ok(status) && block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(body, block_index);
    loom_op_t* op = block->first_op;
    while (iree_status_is_ok(status) && op != NULL) {
      loom_op_t* next_op = op->next_op;
      if (loom_low_op_isa(op)) {
        const loom_low_descriptor_t* descriptor =
            loom_amdgpu_hal_binding_resolve_low_op_descriptor(
                rewriter->module, descriptor_set, op);
        if (descriptor == NULL || (descriptor != static_descriptor &&
                                   descriptor != dynamic_extent_descriptor)) {
          op = next_op;
          continue;
        }
        status = loom_amdgpu_hal_binding_materialize_buffer_descriptor_pseudo(
            rewriter, op, target_bundle, descriptor_set, descriptor, sgpr_type,
            sgpr_x2_type);
        if (iree_status_is_ok(status)) {
          ++*out_materialized_count;
        }
      }
      op = next_op;
    }
  }
  return status;
}

iree_status_t loom_amdgpu_hal_binding_materialize_buffer_descriptors(
    loom_module_t* module, loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t* out_materialized_count,
    iree_arena_allocator_t* scratch_arena) {
  if (module == NULL || function_op == NULL || target_bundle == NULL ||
      descriptor_set == NULL || out_materialized_count == NULL ||
      scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL descriptor materialization requires a module, function, "
        "target bundle, descriptor set, output count, and scratch arena");
  }
  *out_materialized_count = 0;
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL descriptor materialization requires low.func.def or "
        "low.kernel.def");
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_make_sgpr_type(
      module, descriptor_set, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_make_sgpr_type(
      module, descriptor_set, 2, &sgpr_x2_type));
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  iree_status_t status =
      loom_amdgpu_hal_binding_materialize_buffer_descriptors_with_types(
          &rewriter, function_op, target_bundle, descriptor_set, sgpr_type,
          sgpr_x2_type, out_materialized_count);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_amdgpu_hal_binding_materialize(
    loom_module_t* module, loom_op_t* function_op,
    const loom_target_bundle_t* target_bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_hal_binding_materialization_result_t* out_result,
    iree_arena_allocator_t* scratch_arena) {
  if (module == NULL || function_op == NULL || target_bundle == NULL ||
      descriptor_set == NULL || out_result == NULL || scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL binding materialization requires a module, function, "
        "target bundle, descriptor set, output result, and scratch arena");
  }
  *out_result = (loom_amdgpu_hal_binding_materialization_result_t){0};
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL binding materialization requires low.func.def or "
        "low.kernel.def");
  }

  loom_amdgpu_hal_kernel_abi_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_low(
      module, function_op, &layout, scratch_arena));
  out_result->abi_layout = layout;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_make_sgpr_type(
      module, descriptor_set, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_make_sgpr_type(
      module, descriptor_set, 2, &sgpr_x2_type));
  loom_type_t sgpr_x4_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_make_sgpr_type(
      module, descriptor_set, 4, &sgpr_x4_type));
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  iree_status_t status = iree_ok_status();
  if (loom_low_kernel_def_isa(function_op) &&
      (layout.resource_count != 0 || layout.direct_arg_count != 0 ||
       layout.uses_kernarg_segment_ptr)) {
    loom_attribute_t abi_layout_attr = {0};
    status = loom_amdgpu_hal_kernel_abi_make_layout_attr(
        module, &layout, scratch_arena, &abi_layout_attr);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_set_attr(&rewriter, function_op,
                                      loom_low_kernel_def_abi_layout_ATTR_INDEX,
                                      abi_layout_attr);
    }
  }
  loom_value_id_t kernarg_ptr = LOOM_VALUE_ID_INVALID;
  bool inserted_live_in = false;

  loom_region_t* body = loom_low_function_body(function_op);
  if (body == NULL) {
    loom_rewriter_deinitialize(&rewriter);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU HAL binding materialization reached a low function without a "
        "body");
  }
  if (iree_status_is_ok(status) && layout.uses_kernarg_segment_ptr) {
    status = loom_amdgpu_hal_binding_get_kernarg_live_in(
        &rewriter, function_op, sgpr_x2_type, &kernarg_ptr, &inserted_live_in);
    if (iree_status_is_ok(status)) {
      loom_amdgpu_hal_binding_set_entry_insertion_point(&rewriter, function_op);
    }
  }
  const loom_amdgpu_hal_binding_load_flags_t available_load_flags =
      inserted_live_in ? LOOM_AMDGPU_HAL_BINDING_LOAD_FLAG_REUSE_KERNARG_STORAGE
                       : 0;
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_hal_binding_materialize_direct_args(
        &rewriter, function_op, &layout, kernarg_ptr, descriptor_set, sgpr_type,
        sgpr_x2_type, available_load_flags,
        &out_result->materialized_direct_arg_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_hal_binding_materialize_resources(
        &rewriter, &layout, kernarg_ptr, descriptor_set, sgpr_x2_type,
        sgpr_x4_type, available_load_flags,
        &out_result->materialized_binding_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_hal_binding_materialize_buffer_descriptors_with_types(
        &rewriter, function_op, target_bundle, descriptor_set, sgpr_type,
        sgpr_x2_type, &out_result->materialized_descriptor_count);
  }

  out_result->inserted_kernarg_segment_ptr_live_in = inserted_live_in;
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
