// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/address_materialization.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static iree_status_t loom_amdgpu_address_make_sgpr_type(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &type));
  return loom_module_intern_type(module, type, out_type);
}

static const loom_low_descriptor_t* loom_amdgpu_address_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  IREE_ASSERT(descriptor != NULL,
              "generated AMDGPU address materialization descriptors exist");
  return descriptor;
}

static iree_status_t loom_amdgpu_address_i64_attr(loom_module_t* module,
                                                  iree_string_view_t name,
                                                  int64_t value,
                                                  loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_address_build_s_mov_b32(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_type_t sgpr_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_i64_attr(
      rewriter->module, IREE_SV("imm32"), value, &attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      &rewriter->builder, descriptor_set,
      loom_amdgpu_address_descriptor_ref(descriptor_set,
                                         LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32),
      loom_make_named_attr_slice(&attr, 1), sgpr_type, location, &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_address_build_s_binary_b32(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t sgpr_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  const loom_type_t result_types[] = {sgpr_type};
  loom_op_t* binary_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set,
      loom_amdgpu_address_descriptor_ref(descriptor_set, descriptor_ref),
      operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &binary_op));
  *out_value = loom_value_slice_get(loom_low_op_results(binary_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_address_build_s_lshl_b32_rhs_inline(
    loom_rewriter_t* rewriter, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint32_t byte_shift, loom_type_t sgpr_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_i64_attr(
      rewriter->module, IREE_SV("imm32"), byte_shift, &attr));
  const loom_value_id_t operands[] = {value};
  const loom_type_t result_types[] = {sgpr_type};
  loom_op_t* shift_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &rewriter->builder, descriptor_set,
      loom_amdgpu_address_descriptor_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL_B32_RHS_INLINE),
      operands, IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&attr, 1),
      result_types, IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &shift_op));
  *out_value = loom_value_slice_get(loom_low_op_results(shift_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_address_materialize_expression(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t sgpr_type) {
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_set_before(&rewriter->builder, op);

  const uint32_t byte_shift =
      (uint32_t)loom_amdgpu_address_add_scaled_u32_byte_shift(op);
  const loom_value_id_t base = loom_amdgpu_address_add_scaled_u32_base(op);
  loom_value_id_t byte_offset = loom_amdgpu_address_add_scaled_u32_offset(op);
  if (byte_shift != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_address_build_s_lshl_b32_rhs_inline(
        rewriter, descriptor_set, byte_offset, byte_shift, sgpr_type,
        op->location, &byte_offset));
  }

  loom_op_t* base_low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(&rewriter->builder, base,
                                            /*offset=*/0, sgpr_type,
                                            op->location, &base_low_op));
  const loom_value_id_t base_low = loom_low_slice_result(base_low_op);
  loom_op_t* base_high_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(&rewriter->builder, base,
                                            /*offset=*/1, sgpr_type,
                                            op->location, &base_high_op));
  const loom_value_id_t base_high = loom_low_slice_result(base_high_op);
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_build_s_mov_b32(
      rewriter, descriptor_set, 0, sgpr_type, op->location, &zero));

  loom_value_id_t address_low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_build_s_binary_b32(
      rewriter, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, base_low,
      byte_offset, sgpr_type, op->location, &address_low));
  loom_value_id_t address_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_build_s_binary_b32(
      rewriter, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32,
      base_high, zero, sgpr_type, op->location, &address_high));

  const loom_value_id_t parts[] = {address_low, address_high};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      &rewriter->builder, parts, IREE_ARRAYSIZE(parts),
      loom_module_value_type(rewriter->module,
                             loom_amdgpu_address_add_scaled_u32_result(op)),
      op->location, &concat_op));
  loom_value_id_t replacement = loom_low_concat_result(concat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

iree_status_t loom_amdgpu_address_materialize_expressions(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t* out_materialized_count,
    iree_arena_allocator_t* scratch_arena) {
  *out_materialized_count = 0;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_address_make_sgpr_type(module, descriptor_set, &sgpr_type));
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));

  iree_status_t status = iree_ok_status();
  loom_region_t* body = loom_low_function_body(function_op);
  for (uint16_t block_index = 0;
       iree_status_is_ok(status) && block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(body, block_index);
    loom_op_t* op = block->first_op;
    while (iree_status_is_ok(status) && op != NULL) {
      loom_op_t* next_op = op->next_op;
      if (loom_amdgpu_address_add_scaled_u32_isa(op)) {
        status = loom_amdgpu_address_materialize_expression(
            &rewriter, op, descriptor_set, sgpr_type);
        if (iree_status_is_ok(status)) {
          ++*out_materialized_count;
        }
      }
      op = next_op;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
