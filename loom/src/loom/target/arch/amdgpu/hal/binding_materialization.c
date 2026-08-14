// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal/binding_materialization.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/buffer_resource.h"
#include "loom/target/arch/amdgpu/hal/binding_descriptor.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

#define LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_ENABLE_BIT \
  UINT32_C(0x4000)
#define LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_CACHE_SWIZZLE_WORD_SHIFT 16u

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

static const loom_low_descriptor_t* loom_amdgpu_hal_binding_low_op_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op) {
  return &descriptor_set->descriptors[loom_low_op_descriptor(op)];
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
      &rewriter->builder, 0, source_id, loom_make_named_attr_slice(NULL, 0),
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
    if (loom_amdgpu_hal_kernel_abi_live_in_source_kind(rewriter->module,
                                                       live_in_value) ==
        LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR) {
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

static const loom_low_descriptor_t* loom_amdgpu_hal_binding_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  IREE_ASSERT(descriptor != NULL,
              "generated AMDGPU HAL materialization descriptor refs exist");
  return descriptor;
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
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_hal_binding_descriptor_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32);
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      &rewriter->builder, descriptor_set, descriptor,
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
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_hal_binding_descriptor_ref(descriptor_set, descriptor_ref);
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set, descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &binary_op));
  *out_value = loom_value_slice_get(loom_low_op_results(binary_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_s_binary_b32_rhs_inline(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    uint32_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_i64_attr(
      rewriter->module, IREE_SV("imm32"), rhs, &attr));
  const loom_value_id_t operands[] = {lhs};
  const loom_type_t result_types[] = {result_type};
  loom_op_t* binary_op = NULL;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_hal_binding_descriptor_ref(descriptor_set, descriptor_ref);
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set, descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&attr, 1),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &binary_op));
  *out_value = loom_value_slice_get(loom_low_op_results(binary_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_s_load_offset_only(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t kernarg_ptr,
    uint32_t kernarg_offset, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_i64_attr(
      rewriter->module, IREE_SV("offset"), kernarg_offset, &attr));
  const loom_value_id_t operands[] = {kernarg_ptr};
  const loom_type_t result_types[] = {result_type};
  loom_op_t* load_op = NULL;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_hal_binding_descriptor_ref(descriptor_set, descriptor_ref);
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set, descriptor, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&attr, 1),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_low_op_results(load_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_build_s_load_dwordx2(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t kernarg_ptr, uint32_t kernarg_offset,
    loom_type_t sgpr_x2_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_amdgpu_hal_binding_build_s_load_offset_only(
      rewriter, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, kernarg_ptr,
      kernarg_offset, sgpr_x2_type, location, out_value);
}

static iree_status_t loom_amdgpu_hal_binding_build_s_load_dwordx4(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t kernarg_ptr, uint32_t kernarg_offset,
    loom_type_t sgpr_x4_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_amdgpu_hal_binding_build_s_load_offset_only(
      rewriter, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX4_OFFSET_ONLY, kernarg_ptr,
      kernarg_offset, sgpr_x4_type, location, out_value);
}

static iree_status_t loom_amdgpu_hal_binding_build_s_load_dwordx8(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t kernarg_ptr, uint32_t kernarg_offset,
    loom_type_t sgpr_x8_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_amdgpu_hal_binding_build_s_load_offset_only(
      rewriter, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX8_OFFSET_ONLY, kernarg_ptr,
      kernarg_offset, sgpr_x8_type, location, out_value);
}

static iree_status_t loom_amdgpu_hal_binding_build_scalar_load(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t kernarg_ptr, uint32_t kernarg_offset, loom_type_t sgpr_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  return loom_amdgpu_hal_binding_build_s_load_offset_only(
      rewriter, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, kernarg_ptr,
      kernarg_offset, sgpr_type, location, out_value);
}

typedef struct loom_amdgpu_hal_binding_descriptor_pointer_words_t {
  loom_value_id_t low;
  loom_value_id_t high;
} loom_amdgpu_hal_binding_descriptor_pointer_words_t;

static iree_status_t loom_amdgpu_hal_binding_build_descriptor_pointer_words(
    loom_rewriter_t* rewriter,
    const loom_amdgpu_descriptor_set_buffer_resource_info_t*
        buffer_resource_info,
    const loom_amdgpu_buffer_resource_record_encoding_info_t*
        record_encoding_info,
    uint32_t cache_swizzle_stride,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t loaded_pointer, loom_type_t sgpr_type,
    loom_location_id_t location,
    loom_amdgpu_hal_binding_descriptor_pointer_words_t* out_words) {
  *out_words = (loom_amdgpu_hal_binding_descriptor_pointer_words_t){
      .low = LOOM_VALUE_ID_INVALID,
      .high = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
      rewriter, loaded_pointer, 0, sgpr_type, location, &out_words->low));
  loom_value_id_t pointer_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
      rewriter, loaded_pointer, 1, sgpr_type, location, &pointer_high));

  loom_value_id_t descriptor_pointer_high = pointer_high;
  if (record_encoding_info->pointer_high_mask != 0) {
    loom_value_id_t pointer_high_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
        rewriter, descriptor_set, record_encoding_info->pointer_high_mask,
        sgpr_type, location, &pointer_high_mask, NULL));
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_binary_b32(
        rewriter, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
        pointer_high, pointer_high_mask, sgpr_type, location,
        &descriptor_pointer_high));
  }

  if (cache_swizzle_stride != 0) {
    const loom_amdgpu_buffer_resource_cache_swizzle_t cache_swizzle_kind =
        buffer_resource_info->cache_swizzle;
    if (cache_swizzle_kind !=
        LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT) {
      IREE_ASSERT_UNREACHABLE(
          "verified AMDGPU HAL buffer descriptor cache swizzle");
      IREE_BUILTIN_UNREACHABLE();
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
        descriptor_pointer_high, cache_swizzle, sgpr_type, location,
        &swizzled_pointer_high));
    descriptor_pointer_high = swizzled_pointer_high;
  }

  out_words->high = descriptor_pointer_high;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_materialize_one(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_amdgpu_hal_kernarg_resource_t* resource,
    loom_value_id_t kernarg_ptr,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_x2_type) {
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_set_before(&rewriter->builder, op);

  loom_value_id_t pointer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx2(
      rewriter, descriptor_set, kernarg_ptr, resource->kernarg_offset,
      sgpr_x2_type, op->location, &pointer));

  loom_value_id_t replacement = pointer;
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static bool loom_amdgpu_hal_binding_resource_is_used(
    const loom_module_t* module,
    const loom_amdgpu_hal_kernarg_resource_t* resource) {
  if (resource->resource_op == NULL) {
    return false;
  }
  const loom_value_id_t result =
      loom_low_resource_result(resource->resource_op);
  const loom_value_t* value = loom_module_value(module, result);
  return value->use_count != 0 ||
         loom_module_value_has_type_uses(module, result);
}

static bool loom_amdgpu_hal_binding_can_group_resource_load(
    const loom_module_t* module,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_host_size_t start_index, iree_host_size_t group_count,
    loom_type_t sgpr_x2_type) {
  if (group_count == 0 || start_index + group_count > layout->resource_count) {
    return false;
  }
  const loom_amdgpu_hal_kernarg_resource_t* first_resource =
      &layout->resources[start_index];
  const loom_op_t* previous_op = NULL;
  uint32_t previous_end_offset = first_resource->kernarg_offset;
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource =
        &layout->resources[start_index + i];
    if (!loom_amdgpu_hal_binding_resource_is_used(module, resource)) {
      return false;
    }
    if (!loom_type_equal(resource->abi_type, sgpr_x2_type)) {
      return false;
    }
    const loom_op_t* resource_op = resource->resource_op;
    if (resource_op == NULL || resource_op->parent_block == NULL) {
      return false;
    }
    if (iree_any_bit_set(resource_op->flags, LOOM_OP_FLAG_DEAD)) {
      return false;
    }
    if (previous_op != NULL &&
        (resource_op->parent_block != previous_op->parent_block ||
         previous_op->block_ordinal >= resource_op->block_ordinal)) {
      return false;
    }
    if (resource->kernarg_size !=
        LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE) {
      return false;
    }
    if (i != 0 && resource->kernarg_offset != previous_end_offset) {
      return false;
    }
    previous_op = resource_op;
    previous_end_offset = resource->kernarg_offset + resource->kernarg_size;
  }
  return true;
}

static iree_status_t loom_amdgpu_hal_binding_materialize_resource_value(
    loom_rewriter_t* rewriter, loom_op_t* resource_op,
    loom_value_id_t replacement, loom_value_id_t value_checkpoint) {
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, resource_op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, resource_op,
                                                  &replacement, 1);
}

static iree_status_t loom_amdgpu_hal_binding_materialize_resource_group(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_host_size_t start_index, iree_host_size_t group_count,
    loom_value_id_t kernarg_ptr, loom_type_t sgpr_x2_type,
    loom_type_t group_type) {
  const loom_amdgpu_hal_kernarg_resource_t* first_resource =
      &layout->resources[start_index];
  loom_op_t* first_op = (loom_op_t*)first_resource->resource_op;
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_set_before(&rewriter->builder, first_op);

  loom_value_id_t loaded_group = LOOM_VALUE_ID_INVALID;
  if (group_count == 4) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx8(
        rewriter, descriptor_set, kernarg_ptr, first_resource->kernarg_offset,
        group_type, first_op->location, &loaded_group));
  } else {
    IREE_ASSERT(group_count == 2);
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx4(
        rewriter, descriptor_set, kernarg_ptr, first_resource->kernarg_offset,
        group_type, first_op->location, &loaded_group));
  }

  IREE_ASSERT(group_count <= 4);
  loom_op_t* resource_ops[4] = {0};
  loom_value_id_t pointers[4] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource =
        &layout->resources[start_index + i];
    resource_ops[i] = (loom_op_t*)resource->resource_op;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
        rewriter, loaded_group, /*offset=*/(uint32_t)(2 * i), sgpr_x2_type,
        resource_ops[i]->location, &pointers[i]));
  }
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_resource_value(
        rewriter, resource_ops[i], pointers[i], value_checkpoint));
  }
  return iree_ok_status();
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

static bool loom_amdgpu_hal_binding_layout_uses_kernarg_segment_ptr(
    const loom_module_t* module,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout) {
  for (iree_host_size_t i = 0; i < layout->resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource = &layout->resources[i];
    if (resource->resource_op != NULL &&
        !iree_any_bit_set(resource->resource_op->flags, LOOM_OP_FLAG_DEAD) &&
        loom_amdgpu_hal_binding_resource_is_used(module, resource)) {
      return true;
    }
  }
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    if (loom_amdgpu_hal_binding_direct_arg_is_used(module,
                                                   &layout->direct_args[i])) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_hal_binding_direct_arg_unit_count(
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg,
    loom_type_t sgpr_type, loom_type_t sgpr_x2_type, uint32_t* out_unit_count) {
  *out_unit_count = 0;
  if (direct_arg->kernarg_size == sizeof(uint32_t) &&
      loom_type_equal(direct_arg->abi_type, sgpr_type)) {
    *out_unit_count = 1;
    return true;
  }
  if (direct_arg->kernarg_size == 2u * sizeof(uint32_t) &&
      loom_type_equal(direct_arg->abi_type, sgpr_x2_type)) {
    *out_unit_count = 2;
    return true;
  }
  return false;
}

static bool loom_amdgpu_hal_binding_can_group_direct_arg_load(
    const loom_module_t* module,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_host_size_t start_index, iree_host_size_t group_count,
    loom_type_t sgpr_type, loom_type_t sgpr_x2_type, uint32_t* out_unit_count) {
  *out_unit_count = 0;
  if (group_count == 0 ||
      start_index + group_count > layout->direct_arg_count) {
    return false;
  }
  const loom_amdgpu_hal_kernarg_direct_arg_t* first_arg =
      &layout->direct_args[start_index];
  uint32_t expected_unit_count = 0;
  if (!loom_amdgpu_hal_binding_direct_arg_unit_count(
          first_arg, sgpr_type, sgpr_x2_type, &expected_unit_count)) {
    return false;
  }
  uint32_t previous_end_offset = first_arg->kernarg_offset;
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg =
        &layout->direct_args[start_index + i];
    if (!loom_amdgpu_hal_binding_direct_arg_is_used(module, direct_arg)) {
      return false;
    }
    uint32_t unit_count = 0;
    if (!loom_amdgpu_hal_binding_direct_arg_unit_count(
            direct_arg, sgpr_type, sgpr_x2_type, &unit_count) ||
        unit_count != expected_unit_count) {
      return false;
    }
    if (i != 0 && direct_arg->kernarg_offset != previous_end_offset) {
      return false;
    }
    previous_end_offset = direct_arg->kernarg_offset + direct_arg->kernarg_size;
  }
  *out_unit_count = expected_unit_count;
  return true;
}

static iree_status_t loom_amdgpu_hal_binding_materialize_direct_arg_group(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_host_size_t start_index, iree_host_size_t group_count,
    uint32_t unit_count, loom_value_id_t kernarg_ptr, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type, loom_type_t group_type,
    loom_location_id_t location, loom_value_id_t* materialized_values) {
  const loom_amdgpu_hal_kernarg_direct_arg_t* first_arg =
      &layout->direct_args[start_index];
  const uint32_t total_unit_count = unit_count * (uint32_t)group_count;
  loom_value_id_t loaded_group = LOOM_VALUE_ID_INVALID;
  if (total_unit_count == 2) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx2(
        rewriter, descriptor_set, kernarg_ptr, first_arg->kernarg_offset,
        group_type, location, &loaded_group));
  } else if (total_unit_count == 4) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx4(
        rewriter, descriptor_set, kernarg_ptr, first_arg->kernarg_offset,
        group_type, location, &loaded_group));
  } else {
    IREE_ASSERT(total_unit_count == 8);
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx8(
        rewriter, descriptor_set, kernarg_ptr, first_arg->kernarg_offset,
        group_type, location, &loaded_group));
  }
  IREE_ASSERT(group_count <= 4);
  loom_value_id_t values[4] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  loom_type_t slice_type = unit_count == 1 ? sgpr_type : sgpr_x2_type;
  uint32_t slice_offset = 0;
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_low_slice(
        rewriter, loaded_group, slice_offset, slice_type, location,
        &values[i]));
    slice_offset += unit_count;
  }
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg =
        &layout->direct_args[start_index + i];
    materialized_values[direct_arg->argument_index] = values[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_materialize_direct_arg_load(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg,
    loom_value_id_t kernarg_ptr, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type, loom_location_id_t location,
    loom_value_id_t* out_loaded) {
  *out_loaded = LOOM_VALUE_ID_INVALID;
  if (direct_arg->kernarg_size == sizeof(uint32_t) &&
      loom_type_equal(direct_arg->abi_type, sgpr_type)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_scalar_load(
        rewriter, descriptor_set, kernarg_ptr, direct_arg->kernarg_offset,
        sgpr_type, location, out_loaded));
  } else if (direct_arg->kernarg_size == 2u * sizeof(uint32_t) &&
             loom_type_equal(direct_arg->abi_type, sgpr_x2_type)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_load_dwordx2(
        rewriter, descriptor_set, kernarg_ptr, direct_arg->kernarg_offset,
        sgpr_x2_type, location, out_loaded));
  } else {
    IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI direct argument layout");
    IREE_BUILTIN_UNREACHABLE();
  }
  return iree_ok_status();
}

static bool loom_amdgpu_hal_binding_try_direct_arg_index(
    const loom_module_t* module, const loom_block_t* entry_block,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout, loom_value_id_t value_id,
    uint16_t* out_argument_index) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (!loom_value_is_block_arg(value) ||
      loom_value_def_block(value) != entry_block) {
    return false;
  }
  const uint16_t argument_index = loom_value_def_index(value);
  if (argument_index >= layout->direct_arg_count ||
      layout->direct_args[argument_index].arg_id != value_id) {
    return false;
  }
  *out_argument_index = argument_index;
  return true;
}

static iree_status_t loom_amdgpu_hal_binding_snapshot_function_predicates(
    loom_rewriter_t* rewriter, loom_func_like_t function,
    loom_predicate_t** out_predicates, uint16_t* out_predicate_count) {
  *out_predicates = NULL;
  *out_predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function, out_predicate_count);
  if (*out_predicate_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, *out_predicate_count, sizeof(**out_predicates),
      (void**)out_predicates));
  memcpy(*out_predicates, predicates,
         (iree_host_size_t)*out_predicate_count * sizeof(**out_predicates));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_set_function_predicates(
    loom_rewriter_t* rewriter, loom_func_like_t function,
    const loom_predicate_t* predicates, uint16_t predicate_count) {
  IREE_ASSERT_NE(function.vtable->predicates_attr_index, LOOM_ATTR_INDEX_NONE);
  loom_attribute_t predicate_attr = loom_attr_absent();
  if (predicate_count != 0) {
    loom_predicate_t* predicate_storage = NULL;
    IREE_RETURN_IF_ERROR(loom_builder_copy_predicate_list_attr_storage(
        &rewriter->builder, predicates, predicate_count,
        IREE_SV("AMDGPU HAL retained function predicates"),
        &predicate_storage));
    predicate_attr =
        loom_attr_predicate_list(predicate_storage, predicate_count);
  }
  return loom_rewriter_set_attr(rewriter, function.op,
                                function.vtable->predicates_attr_index,
                                predicate_attr);
}

static iree_status_t loom_amdgpu_hal_binding_transfer_function_predicates(
    loom_rewriter_t* rewriter, loom_func_like_t function,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    const loom_block_t* entry_block, loom_predicate_t* predicates,
    uint16_t predicate_count, loom_value_id_t* materialized_values) {
  uint16_t retained_count = 0;
  bool has_direct_predicate = false;
  for (uint16_t i = 0; i < predicate_count; ++i) {
    loom_predicate_t predicate = predicates[i];
    uint16_t argument_indices[IREE_ARRAYSIZE(predicate.args)] = {0};
    uint16_t argument_count = 0;
    bool references_direct_arg = false;
    bool is_transferable = true;
    for (uint8_t j = 0; j < predicate.arg_count; ++j) {
      if (predicate.arg_tags[j] != LOOM_PRED_ARG_VALUE) continue;
      const loom_value_id_t value_id = (loom_value_id_t)predicate.args[j];
      uint16_t argument_index = 0;
      if (!loom_amdgpu_hal_binding_try_direct_arg_index(
              rewriter->module, entry_block, layout, value_id,
              &argument_index)) {
        is_transferable = false;
        continue;
      }
      references_direct_arg = true;
      if (materialized_values[argument_index] == LOOM_VALUE_ID_INVALID) {
        is_transferable = false;
      }
      bool already_listed = false;
      for (uint16_t k = 0; k < argument_count; ++k) {
        already_listed |= argument_indices[k] == argument_index;
      }
      if (!already_listed) {
        argument_indices[argument_count++] = argument_index;
      }
    }

    if (!references_direct_arg) {
      predicates[retained_count++] = predicate;
      continue;
    }
    has_direct_predicate = true;
    if (!is_transferable) {
      // Predicates are compile-time facts, not runtime uses. Do not introduce a
      // kernarg load solely to keep a predicate whose values otherwise vanish.
      continue;
    }

    loom_value_id_t values[IREE_ARRAYSIZE(predicate.args)] = {0};
    loom_type_t result_types[IREE_ARRAYSIZE(predicate.args)] = {0};
    for (uint16_t j = 0; j < argument_count; ++j) {
      const uint16_t argument_index = argument_indices[j];
      values[j] = materialized_values[argument_index];
      result_types[j] = loom_module_value_type(rewriter->module, values[j]);
    }
    for (uint8_t j = 0; j < predicate.arg_count; ++j) {
      if (predicate.arg_tags[j] != LOOM_PRED_ARG_VALUE) continue;
      uint16_t argument_index = 0;
      const bool is_direct_arg = loom_amdgpu_hal_binding_try_direct_arg_index(
          rewriter->module, entry_block, layout,
          (loom_value_id_t)predicate.args[j], &argument_index);
      IREE_ASSERT(is_direct_arg);
      predicate.args[j] = materialized_values[argument_index];
    }

    loom_op_t* assume_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_assume_build(
        &rewriter->builder, values, argument_count, &predicate,
        /*predicates_count=*/1, result_types, argument_count,
        function.op->location, &assume_op));
    const loom_value_slice_t results = loom_low_assume_results(assume_op);
    for (uint16_t j = 0; j < argument_count; ++j) {
      materialized_values[argument_indices[j]] = results.values[j];
    }
  }

  if (!has_direct_predicate) return iree_ok_status();
  return loom_amdgpu_hal_binding_set_function_predicates(
      rewriter, function, predicates, retained_count);
}

static iree_status_t loom_amdgpu_hal_binding_replace_direct_arg_uses(
    loom_rewriter_t* rewriter,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    const loom_value_id_t* materialized_values) {
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    const loom_value_id_t replacement =
        materialized_values[layout->direct_args[i].argument_index];
    if (replacement == LOOM_VALUE_ID_INVALID) continue;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_move_value_name(
        rewriter->module, layout->direct_args[i].arg_id, replacement));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter, layout->direct_args[i].arg_id, replacement));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_materialize_direct_args(
    loom_rewriter_t* rewriter, loom_op_t* function_op,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_value_id_t kernarg_ptr,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type, loom_type_t sgpr_x4_type,
    loom_type_t sgpr_x8_type, iree_host_size_t* out_materialized_count) {
  *out_materialized_count = 0;
  if (layout->direct_arg_count == 0) {
    return iree_ok_status();
  }

  loom_block_t* entry_block =
      loom_region_entry_block(loom_low_function_body(function_op));
  loom_func_like_t function =
      loom_func_like_cast(rewriter->module, function_op);
  IREE_ASSERT(loom_func_like_isa(function));
  loom_predicate_t* predicates = NULL;
  uint16_t predicate_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_snapshot_function_predicates(
      rewriter, function, &predicates, &predicate_count));
  loom_value_id_t* materialized_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, layout->direct_arg_count, sizeof(*materialized_values),
      (void**)&materialized_values));
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    materialized_values[i] = LOOM_VALUE_ID_INVALID;
  }
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg =
        &layout->direct_args[i];
    if (!loom_amdgpu_hal_binding_direct_arg_is_used(rewriter->module,
                                                    direct_arg)) {
      continue;
    }
    if (kernarg_ptr == LOOM_VALUE_ID_INVALID) {
      IREE_ASSERT_UNREACHABLE(
          "verified AMDGPU HAL ABI direct argument kernarg dependency");
      IREE_BUILTIN_UNREACHABLE();
    }

    uint32_t arg_unit_count = 0;
    if (loom_amdgpu_hal_binding_can_group_direct_arg_load(
            rewriter->module, layout, i, /*group_count=*/4, sgpr_type,
            sgpr_x2_type, &arg_unit_count)) {
      const uint32_t load_unit_count = arg_unit_count * 4;
      loom_type_t group_type =
          load_unit_count == 4 ? sgpr_x4_type : sgpr_x8_type;
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_direct_arg_group(
          rewriter, descriptor_set, layout, i, /*group_count=*/4,
          arg_unit_count, kernarg_ptr, sgpr_type, sgpr_x2_type, group_type,
          function_op->location, materialized_values));
      *out_materialized_count += 4;
      i += 3;
      continue;
    }

    if (loom_amdgpu_hal_binding_can_group_direct_arg_load(
            rewriter->module, layout, i, /*group_count=*/2, sgpr_type,
            sgpr_x2_type, &arg_unit_count)) {
      const uint32_t load_unit_count = arg_unit_count * 2;
      loom_type_t group_type =
          load_unit_count == 2 ? sgpr_x2_type : sgpr_x4_type;
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_direct_arg_group(
          rewriter, descriptor_set, layout, i, /*group_count=*/2,
          arg_unit_count, kernarg_ptr, sgpr_type, sgpr_x2_type, group_type,
          function_op->location, materialized_values));
      *out_materialized_count += 2;
      ++i;
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_direct_arg_load(
        rewriter, descriptor_set, direct_arg, kernarg_ptr, sgpr_type,
        sgpr_x2_type, function_op->location,
        &materialized_values[direct_arg->argument_index]));
    ++*out_materialized_count;
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_transfer_function_predicates(
      rewriter, function, layout, entry_block, predicates, predicate_count,
      materialized_values));
  // Delay replacement until every assumption has been built. The new
  // assumptions refer only to materialized values, so the complete replacement
  // updates all original body, type, and attribute uses without rewriting the
  // assumptions into self-references.
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_replace_direct_arg_uses(
      rewriter, layout, materialized_values));

  for (iree_host_size_t i = layout->direct_arg_count; i > 0; --i) {
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg =
        &layout->direct_args[i - 1];
    if (direct_arg->argument_index >= entry_block->arg_count) {
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU HAL direct argument entry-block index is stable");
      IREE_BUILTIN_UNREACHABLE();
    }
    if (loom_block_arg_id(entry_block, direct_arg->argument_index) !=
        direct_arg->arg_id) {
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU HAL direct argument entry-block order is stable");
      IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(loom_block_remove_arg(rewriter->module, entry_block,
                                               direct_arg->argument_index));
    rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_binding_materialize_resources(
    loom_rewriter_t* rewriter,
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_value_id_t kernarg_ptr,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_x2_type,
    loom_type_t sgpr_x4_type, loom_type_t sgpr_x8_type,
    iree_host_size_t* out_materialized_count) {
  *out_materialized_count = 0;
  for (iree_host_size_t i = 0; i < layout->resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource = &layout->resources[i];
    loom_op_t* resource_op = (loom_op_t*)resource->resource_op;
    if (resource_op == NULL) {
      continue;
    }
    if (iree_any_bit_set(resource_op->flags, LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    if (!loom_amdgpu_hal_binding_resource_is_used(rewriter->module, resource)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, resource_op));
      continue;
    }
    if (kernarg_ptr == LOOM_VALUE_ID_INVALID) {
      IREE_ASSERT_UNREACHABLE(
          "verified AMDGPU HAL ABI resource kernarg dependency");
      IREE_BUILTIN_UNREACHABLE();
    }

    if (loom_amdgpu_hal_binding_can_group_resource_load(
            rewriter->module, layout, i, /*group_count=*/4, sgpr_x2_type)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_resource_group(
          rewriter, descriptor_set, layout, i, /*group_count=*/4, kernarg_ptr,
          sgpr_x2_type, sgpr_x8_type));
      *out_materialized_count += 4;
      i += 3;
      continue;
    }

    if (loom_amdgpu_hal_binding_can_group_resource_load(
            rewriter->module, layout, i, /*group_count=*/2, sgpr_x2_type)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_resource_group(
          rewriter, descriptor_set, layout, i, /*group_count=*/2, kernarg_ptr,
          sgpr_x2_type, sgpr_x4_type));
      *out_materialized_count += 2;
      ++i;
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize_one(
        rewriter, resource_op, resource, kernarg_ptr, descriptor_set,
        sgpr_x2_type));
    ++*out_materialized_count;
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_hal_binding_materialize_buffer_descriptor_pseudo(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_descriptor_set_buffer_resource_info_t*
        buffer_resource_info,
    const loom_amdgpu_buffer_resource_record_encoding_info_t*
        record_encoding_info,
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

  loom_amdgpu_hal_binding_descriptor_pointer_words_t pointer_words = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_descriptor_pointer_words(
      rewriter, buffer_resource_info, record_encoding_info,
      cache_swizzle_stride, descriptor_set, operands.values[0], sgpr_type,
      op->location, &pointer_words));

  const uint8_t num_records_word1_bit_count =
      record_encoding_info->num_records_word1_bit_count;
  loom_value_id_t pointer = LOOM_VALUE_ID_INVALID;
  if (num_records_word1_bit_count == 0) {
    const loom_value_id_t pointer_sources[] = {
        pointer_words.low,
        pointer_words.high,
    };
    loom_op_t* pointer_concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        &rewriter->builder, pointer_sources, IREE_ARRAYSIZE(pointer_sources),
        sgpr_x2_type, op->location, &pointer_concat_op));
    pointer = loom_low_concat_result(pointer_concat_op);
  }

  loom_value_id_t num_records_word2 = LOOM_VALUE_ID_INVALID;
  const bool has_dynamic_extent =
      descriptor ==
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR_EXTENT);
  uint32_t static_range_word = 0;
  if (has_dynamic_extent) {
    num_records_word2 = operands.values[1];
  } else {
    const int64_t extent =
        attrs.entries[LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_EXTENT].value.i64;
    static_range_word = loom_amdgpu_hal_binding_descriptor_range_word(extent);
    const uint32_t encoded_word2 =
        static_range_word >> num_records_word1_bit_count;
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
        rewriter, descriptor_set, encoded_word2, sgpr_type, op->location,
        &num_records_word2, NULL));
  }
  if (num_records_word1_bit_count != 0) {
    const uint32_t word1_shift = 32u - num_records_word1_bit_count;
    loom_value_id_t shifted_num_records = LOOM_VALUE_ID_INVALID;
    if (has_dynamic_extent) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_hal_binding_build_s_binary_b32_rhs_inline(
              rewriter, descriptor_set,
              LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL_B32_RHS_INLINE,
              num_records_word2, word1_shift, sgpr_type, op->location,
              &shifted_num_records));
    } else {
      const uint32_t shifted_range = static_range_word << word1_shift;
      if (shifted_range != 0) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
            rewriter, descriptor_set, shifted_range, sgpr_type, op->location,
            &shifted_num_records, NULL));
      }
    }
    if (shifted_num_records != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_binary_b32(
          rewriter, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32,
          pointer_words.high, shifted_num_records, sgpr_type, op->location,
          &pointer_words.high));
    }
    if (has_dynamic_extent) {
      loom_value_id_t shifted_num_records_word2 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_hal_binding_build_s_binary_b32_rhs_inline(
              rewriter, descriptor_set,
              LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B32_RHS_INLINE,
              num_records_word2, num_records_word1_bit_count, sgpr_type,
              op->location, &shifted_num_records_word2));
      num_records_word2 = shifted_num_records_word2;
    }
  }

  loom_value_id_t word3 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_build_s_mov_b32(
      rewriter, descriptor_set,
      record_encoding_info->raw_bounds_checked_word3_control, sgpr_type,
      op->location, &word3, NULL));

  if (pointer == LOOM_VALUE_ID_INVALID) {
    const loom_value_id_t pointer_sources[] = {
        pointer_words.low,
        pointer_words.high,
    };
    loom_op_t* pointer_concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        &rewriter->builder, pointer_sources, IREE_ARRAYSIZE(pointer_sources),
        sgpr_x2_type, op->location, &pointer_concat_op));
    pointer = loom_low_concat_result(pointer_concat_op);
  }

  const loom_value_id_t sources[] = {pointer, num_records_word2, word3};
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
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_type,
    loom_type_t sgpr_x2_type, iree_host_size_t* out_materialized_count) {
  *out_materialized_count = 0;
  loom_region_t* body = loom_low_function_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  const loom_low_descriptor_t* static_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR);
  const loom_low_descriptor_t* dynamic_extent_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR_EXTENT);
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  IREE_ASSERT(descriptor_set_info != NULL);
  const loom_amdgpu_descriptor_set_buffer_resource_info_t*
      buffer_resource_info = &descriptor_set_info->buffer_resource;
  const loom_amdgpu_buffer_resource_record_encoding_info_t*
      record_encoding_info = loom_amdgpu_buffer_resource_record_encoding_info(
          buffer_resource_info->record_encoding);
  for (uint16_t block_index = 0;
       iree_status_is_ok(status) && block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(body, block_index);
    loom_op_t* op = block->first_op;
    while (iree_status_is_ok(status) && op != NULL) {
      loom_op_t* next_op = op->next_op;
      if (loom_low_op_isa(op)) {
        const loom_low_descriptor_t* descriptor =
            loom_amdgpu_hal_binding_low_op_descriptor(descriptor_set, op);
        if (descriptor != static_descriptor &&
            descriptor != dynamic_extent_descriptor) {
          op = next_op;
          continue;
        }
        status = loom_amdgpu_hal_binding_materialize_buffer_descriptor_pseudo(
            rewriter, op, descriptor_set, buffer_resource_info,
            record_encoding_info, descriptor, sgpr_type, sgpr_x2_type);
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
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t* out_materialized_count,
    iree_arena_allocator_t* scratch_arena) {
  if (module == NULL || function_op == NULL || descriptor_set == NULL ||
      out_materialized_count == NULL || scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL descriptor materialization requires a module, function, "
        "descriptor set, output count, and scratch arena");
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
          &rewriter, function_op, descriptor_set, sgpr_type, sgpr_x2_type,
          out_materialized_count);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_amdgpu_hal_binding_materialize(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_hal_binding_materialization_result_t* out_result,
    iree_arena_allocator_t* scratch_arena) {
  if (module == NULL || function_op == NULL || descriptor_set == NULL ||
      out_result == NULL || scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL binding materialization requires a module, function, "
        "descriptor set, output result, and scratch arena");
  }
  *out_result = (loom_amdgpu_hal_binding_materialization_result_t){0};
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL binding materialization requires low.func.def or "
        "low.kernel.def");
  }
  loom_region_t* body = loom_low_function_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_hal_kernel_abi_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_low(
      module, function_op, &layout, scratch_arena));
  layout.uses_kernarg_segment_ptr =
      loom_amdgpu_hal_binding_layout_uses_kernarg_segment_ptr(module, &layout);
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
  loom_type_t sgpr_x8_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_make_sgpr_type(
      module, descriptor_set, 8, &sgpr_x8_type));
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

  if (iree_status_is_ok(status) && layout.uses_kernarg_segment_ptr) {
    status = loom_amdgpu_hal_binding_get_kernarg_live_in(
        &rewriter, function_op, sgpr_x2_type, &kernarg_ptr, &inserted_live_in);
    if (iree_status_is_ok(status)) {
      loom_amdgpu_hal_binding_set_entry_insertion_point(&rewriter, function_op);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_hal_binding_materialize_direct_args(
        &rewriter, function_op, &layout, kernarg_ptr, descriptor_set, sgpr_type,
        sgpr_x2_type, sgpr_x4_type, sgpr_x8_type,
        &out_result->materialized_direct_arg_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_hal_binding_materialize_resources(
        &rewriter, &layout, kernarg_ptr, descriptor_set, sgpr_x2_type,
        sgpr_x4_type, sgpr_x8_type, &out_result->materialized_binding_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_hal_binding_materialize_buffer_descriptors_with_types(
        &rewriter, function_op, descriptor_set, sgpr_type, sgpr_x2_type,
        &out_result->materialized_descriptor_count);
  }

  out_result->changed =
      rewriter.created_op_count != 0 || rewriter.erased_op_count != 0 ||
      iree_any_bit_set(rewriter.flags, LOOM_REWRITER_FLAG_CHANGED);
  out_result->inserted_kernarg_segment_ptr_live_in = inserted_live_in;
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
