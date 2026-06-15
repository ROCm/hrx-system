// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/bitfield.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_bitfield_extract_plan_from_op(
    const loom_op_t* source_op, loom_amdgpu_bitfield_extract_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_bitfield_extract_plan_t){0};

  int64_t offset = 0;
  int64_t width = 0;
  if (loom_vector_bitfield_extractu_isa(source_op)) {
    offset = loom_vector_bitfield_extractu_offset(source_op);
    width = loom_vector_bitfield_extractu_width(source_op);
    out_plan->source = loom_vector_bitfield_extractu_source(source_op);
    out_plan->result = loom_vector_bitfield_extractu_result(source_op);
    out_plan->is_signed = false;
  } else if (loom_vector_bitfield_extracts_isa(source_op)) {
    offset = loom_vector_bitfield_extracts_offset(source_op);
    width = loom_vector_bitfield_extracts_width(source_op);
    out_plan->source = loom_vector_bitfield_extracts_source(source_op);
    out_plan->result = loom_vector_bitfield_extracts_result(source_op);
    out_plan->is_signed = true;
  } else {
    return false;
  }

  if (offset < 0 || offset > 31 || width < 1 || width > 32 ||
      offset + width > 32) {
    return false;
  }
  out_plan->offset = (uint32_t)offset;
  out_plan->width = (uint32_t)width;
  return true;
}

static bool loom_amdgpu_bitfield_insert_plan_from_op(
    const loom_op_t* source_op, loom_amdgpu_bitfield_insert_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_bitfield_insert_plan_t){0};
  if (!loom_vector_bitfield_insert_isa(source_op)) {
    return false;
  }

  const int64_t offset = loom_vector_bitfield_insert_offset(source_op);
  const int64_t width = loom_vector_bitfield_insert_width(source_op);
  if (offset < 0 || offset > 31 || width < 1 || width > 32 ||
      offset + width > 32) {
    return false;
  }
  out_plan->field = loom_vector_bitfield_insert_field(source_op);
  out_plan->base = loom_vector_bitfield_insert_base(source_op);
  out_plan->result = loom_vector_bitfield_insert_result(source_op);
  out_plan->offset = (uint32_t)offset;
  out_plan->width = (uint32_t)width;
  return true;
}

iree_status_t loom_amdgpu_select_vector_bitfield_extract_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitfield_extract_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  if (!loom_amdgpu_bitfield_extract_plan_from_op(source_op, out_plan)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, out_plan->source);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);
  const uint32_t source_lane_count =
      loom_amdgpu_vector_i32_lane_count(source_type);
  *out_selected =
      source_lane_count != 0 &&
      loom_amdgpu_vector_i32_lane_count(result_type) == source_lane_count;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_bitfield_insert_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitfield_insert_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  if (!loom_amdgpu_bitfield_insert_plan_from_op(source_op, out_plan)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t field_type =
      loom_module_value_type(module, out_plan->field);
  const loom_type_t base_type = loom_module_value_type(module, out_plan->base);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);
  const uint32_t base_lane_count = loom_amdgpu_vector_i32_lane_count(base_type);
  *out_selected =
      base_lane_count != 0 &&
      loom_amdgpu_vector_i32_lane_count(field_type) == base_lane_count &&
      loom_amdgpu_vector_i32_lane_count(result_type) == base_lane_count;
  return iree_ok_status();
}

static uint32_t loom_amdgpu_bitfield_low_mask(uint32_t width) {
  return width == 32 ? UINT32_MAX : (UINT32_C(1) << width) - 1u;
}

static uint32_t loom_amdgpu_bitfield_insert_target_mask(
    const loom_amdgpu_bitfield_insert_plan_t* plan) {
  return loom_amdgpu_bitfield_low_mask(plan->width) << plan->offset;
}

static bool loom_amdgpu_bitfield_extract_prefers_bfe(
    const loom_amdgpu_bitfield_extract_plan_t* plan) {
  if (plan->is_signed) {
    return plan->offset + plan->width < 32;
  }
  return plan->offset != 0;
}

static iree_status_t loom_amdgpu_try_emit_bitfield_extract_bfe(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_extract_plan_t* plan,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane, bool* out_selected) {
  const loom_amdgpu_vgpr_bfe_extract_flags_t flags =
      plan->is_signed ? LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_SIGN_EXTEND
                      : LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_NONE;
  return loom_amdgpu_try_emit_vgpr_b32_bfe_extract(
      context, source_op, source_lane, plan->offset, plan->width, flags,
      lane_type, out_lane, out_selected);
}

static iree_status_t loom_amdgpu_emit_extractu_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_extract_plan_t* plan,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (plan->offset == 0 && plan->width == 32) {
    *out_lane = source_lane;
    return iree_ok_status();
  }

  if (loom_amdgpu_bitfield_extract_prefers_bfe(plan)) {
    bool selected_bfe = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_bitfield_extract_bfe(
        context, source_op, plan, source_lane, lane_type, out_lane,
        &selected_bfe));
    if (selected_bfe) {
      return iree_ok_status();
    }
  }

  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      plan->offset, source_lane, lane_type, &shifted));

  if (plan->width == 32) {
    *out_lane = shifted;
    return iree_ok_status();
  }

  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, shifted,
      loom_amdgpu_bitfield_low_mask(plan->width), lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_extracts_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_extract_plan_t* plan,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (plan->offset == 0 && plan->width == 32) {
    *out_lane = source_lane;
    return iree_ok_status();
  }

  if (loom_amdgpu_bitfield_extract_prefers_bfe(plan)) {
    bool selected_bfe = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_bitfield_extract_bfe(
        context, source_op, plan, source_lane, lane_type, out_lane,
        &selected_bfe));
    if (selected_bfe) {
      return iree_ok_status();
    }
  }

  loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      32u - plan->offset - plan->width, source_lane, lane_type, &shifted_left));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
      32u - plan->width, shifted_left, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bitfield_extract_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_extract_plan_t* plan,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  if (plan->is_signed) {
    return loom_amdgpu_emit_extracts_lane(context, source_op, plan, source_lane,
                                          lane_type, out_lane);
  }
  return loom_amdgpu_emit_extractu_lane(context, source_op, plan, source_lane,
                                        lane_type, out_lane);
}

static iree_status_t loom_amdgpu_try_emit_bitfield_insert_bfi(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_insert_plan_t* plan, loom_value_id_t field_lane,
    loom_value_id_t base_lane, loom_type_t lane_type, loom_value_id_t* out_lane,
    bool* out_selected) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  *out_selected = false;

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  bool present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT, &descriptor,
      &present));
  if (!present) {
    return iree_ok_status();
  }

  loom_value_id_t shifted_field = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      plan->offset, field_lane, lane_type, &shifted_field));

  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("imm32"), loom_amdgpu_bitfield_insert_target_mask(plan),
      attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  loom_value_id_t operands[] = {shifted_field, base_lane};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_lane = loom_value_slice_get(loom_low_op_results(low_op), 0);
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_insert_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_insert_plan_t* plan, loom_value_id_t field_lane,
    loom_value_id_t base_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (plan->offset == 0 && plan->width == 32) {
    *out_lane = field_lane;
    return iree_ok_status();
  }

  bool selected_bfi = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_bitfield_insert_bfi(
      context, source_op, plan, field_lane, base_lane, lane_type, out_lane,
      &selected_bfi));
  if (selected_bfi) {
    return iree_ok_status();
  }

  loom_value_id_t field_low_bits = field_lane;
  if (plan->width < 32) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        field_lane, loom_amdgpu_bitfield_low_mask(plan->width), lane_type,
        &field_low_bits));
  }

  loom_value_id_t shifted_field = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      plan->offset, field_low_bits, lane_type, &shifted_field));

  const uint32_t target_mask = loom_amdgpu_bitfield_insert_target_mask(plan);
  const uint32_t clear_mask = ~target_mask;
  if (clear_mask == 0) {
    *out_lane = shifted_field;
    return iree_ok_status();
  }

  loom_value_id_t cleared_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, base_lane,
      clear_mask, lane_type, &cleared_base));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, cleared_base,
      shifted_field, lane_type, out_lane);
}

iree_status_t loom_amdgpu_lower_vector_bitfield_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_extract_plan_t* plan) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, plan->source);
  const uint32_t lane_count = loom_amdgpu_vector_i32_lane_count(source_type);
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  if (lane_count == 1) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bitfield_extract_lane(
        context, source_op, plan, low_source, lane_type, &low_result));
    return loom_low_lower_bind_value(context, plan->result, low_result);
  }

  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source, i, lane_type, &source_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bitfield_extract_lane(
        context, source_op, plan, source_lane, lane_type, &lane_results[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_vector_bitfield_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_insert_plan_t* plan) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t base_type = loom_module_value_type(module, plan->base);
  const uint32_t lane_count = loom_amdgpu_vector_i32_lane_count(base_type);
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_value_id_t low_field = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->field, &low_field));
  loom_value_id_t low_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->base, &low_base));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  if (lane_count == 1) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_insert_lane(
        context, source_op, plan, low_field, low_base, lane_type, &low_result));
    return loom_low_lower_bind_value(context, plan->result, low_result);
  }

  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t field_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_field, i, lane_type, &field_lane));
    loom_value_id_t base_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_base, i, lane_type, &base_lane));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_insert_lane(context, source_op, plan, field_lane,
                                     base_lane, lane_type, &lane_results[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}
