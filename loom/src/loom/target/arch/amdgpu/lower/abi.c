// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/abi.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/types.h"

static iree_status_t loom_amdgpu_make_hal_buffer_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_string_id_t hal_buffer_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("hal.buffer"), &hal_buffer_id));
  *out_type = loom_type_dialect_opaque(hal_buffer_id);
  return iree_ok_status();
}

static uint32_t loom_amdgpu_hal_binding_index(loom_low_lower_context_t* context,
                                              uint16_t source_argument_index) {
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids = loom_func_like_arg_ids(
      loom_low_lower_context_source_function(context), &argument_count);
  uint32_t resource_index = 0;
  for (uint16_t i = 0; i < source_argument_index && i < argument_count; ++i) {
    loom_type_t type = loom_module_value_type(
        loom_low_lower_context_module(context), argument_ids[i]);
    if (loom_type_is_buffer(type)) {
      ++resource_index;
    }
  }
  return resource_index;
}

typedef struct loom_amdgpu_buffer_argument_extent_t {
  // Module containing the source function being inspected.
  const loom_module_t* module;
  // True once a buffer.view derived from the source argument is found.
  bool found_view;
  // True when a derived view has no exact static dense byte extent.
  bool found_unbounded_view;
  // Maximum byte extent required by all statically boundable derived views.
  int64_t extent;
} loom_amdgpu_buffer_argument_extent_t;

static bool loom_amdgpu_view_static_dense_byte_extent(
    const loom_module_t* module, loom_type_t view_type,
    int64_t* out_byte_extent) {
  *out_byte_extent = 0;
  if (!loom_type_is_view(view_type)) {
    return false;
  }

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_type_address_layout(
          /*context=*/NULL, module, view_type, stride_storage,
          IREE_ARRAYSIZE(stride_storage), &layout) ||
      layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    return false;
  }

  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  if (element_bit_count <= 0 || (element_bit_count % 8) != 0) {
    return false;
  }

  int64_t element_count = 1;
  const uint8_t rank = loom_type_rank(view_type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(view_type, i)) {
      return false;
    }
    const int64_t dim_size = loom_type_dim_static_size_at(view_type, i);
    if (dim_size < 0 ||
        !iree_checked_mul_i64(element_count, dim_size, &element_count)) {
      return false;
    }
  }

  const int64_t element_byte_count = element_bit_count / 8;
  return iree_checked_mul_i64(element_count, element_byte_count,
                              out_byte_extent);
}

static void loom_amdgpu_buffer_argument_extent_include_view(
    loom_amdgpu_buffer_argument_extent_t* state,
    const loom_op_t* buffer_view_op) {
  if (state->found_unbounded_view) {
    return;
  }
  state->found_view = true;
  int64_t base_byte_offset = 0;
  int64_t view_byte_extent = 0;
  int64_t extent = 0;
  if (!loom_amdgpu_module_value_as_exact_index_constant(
          state->module, loom_buffer_view_byte_offset(buffer_view_op),
          &base_byte_offset) ||
      base_byte_offset < 0 ||
      !loom_amdgpu_view_static_dense_byte_extent(
          state->module,
          loom_module_value_type(state->module,
                                 loom_buffer_view_result(buffer_view_op)),
          &view_byte_extent) ||
      !iree_checked_add_i64(base_byte_offset, view_byte_extent, &extent)) {
    state->found_unbounded_view = true;
    return;
  }
  state->extent = iree_max(state->extent, extent);
}

static void loom_amdgpu_buffer_argument_extent_include_uses(
    loom_amdgpu_buffer_argument_extent_t* state, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(state->module, value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    const uint16_t operand_index = loom_use_operand_index(*use);
    if (operand_index == 0 && loom_buffer_view_isa(user_op)) {
      loom_amdgpu_buffer_argument_extent_include_view(state, user_op);
      continue;
    }
    if (operand_index < user_op->result_count &&
        loom_traits_are_fact_identity(
            loom_op_effective_traits(state->module, user_op))) {
      loom_amdgpu_buffer_argument_extent_include_uses(
          state, loom_op_results(user_op)[operand_index]);
      continue;
    }
    state->found_unbounded_view = true;
  }
}

static bool loom_amdgpu_source_buffer_argument_extent(
    loom_low_lower_context_t* context, loom_value_id_t source_argument_id,
    int64_t* out_extent) {
  *out_extent = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_amdgpu_buffer_argument_extent_t state = {
      .module = module,
  };
  loom_amdgpu_buffer_argument_extent_include_uses(&state, source_argument_id);
  if (!state.found_view || state.found_unbounded_view) {
    return false;
  }
  *out_extent = state.extent;
  return true;
}

iree_status_t loom_amdgpu_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)user_data;
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL &&
      loom_type_is_buffer(source_type)) {
    loom_type_t binding_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &binding_type));
    loom_type_t source_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_hal_buffer_type(context, &source_type));
    loom_low_resource_build_flags_t resource_build_flags = 0;
    int64_t resource_extent = 0;
    if (loom_amdgpu_source_buffer_argument_extent(context, source_argument_id,
                                                  &resource_extent)) {
      resource_build_flags |= LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT;
    }
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = binding_type,
        .resource_import_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING,
        .resource_index =
            loom_amdgpu_hal_binding_index(context, source_argument_index),
        .resource_source_type = source_type,
        .resource_build_flags = resource_build_flags,
        .resource_extent = resource_extent,
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL &&
      loom_kernel_def_isa(source_function_op) &&
      loom_amdgpu_type_is_f32(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, &out_argument->abi_type);
  }
  return loom_amdgpu_map_value(user_data, context, source_function_op,
                               source_argument_id, source_type,
                               &out_argument->abi_type);
}
