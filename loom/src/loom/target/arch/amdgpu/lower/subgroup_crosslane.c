// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared AMDGPU subgroup cross-lane packet emission helpers.

#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

enum {
  LOOM_AMDGPU_DS_SWIZZLE_BITMASK_AND = 0x1Fu,
  LOOM_AMDGPU_DS_SWIZZLE_BITMASK_XOR_SHIFT = 10u,
};

typedef struct loom_amdgpu_dpp_xor_control_t {
  // DPP control supported by legacy and DPP16 descriptors.
  uint16_t legacy_control;
  // DPP control supported only by DPP16 descriptors.
  uint16_t dpp16_control;
} loom_amdgpu_dpp_xor_control_t;

typedef struct loom_amdgpu_dpp_descriptor_family_t {
  // Descriptor moving one source register through a DPP control.
  loom_amdgpu_descriptor_ref_t move_ref;
  // Descriptor applying the same DPP control to a conditional false operand.
  loom_amdgpu_descriptor_ref_t conditional_ref;
  // Descriptor updating selected destination banks while preserving the rest.
  loom_amdgpu_descriptor_ref_t masked_move_ref;
} loom_amdgpu_dpp_descriptor_family_t;

static const loom_amdgpu_dpp_xor_control_t kLoomAmdgpuDppXorControls[] = {
    [1] = {.legacy_control = LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1,
           .dpp16_control = LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1},
    [2] = {.legacy_control = LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_2,
           .dpp16_control = LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_2},
    [4] = {.dpp16_control = LOOM_AMDGPU_DPP_CTRL_ROW_XMASK_BASE + 4},
    [8] = {.dpp16_control = LOOM_AMDGPU_DPP_CTRL_ROW_XMASK_BASE + 8},
};

static const loom_amdgpu_dpp_descriptor_family_t
    kLoomAmdgpuDppDescriptorFamilies[] = {
        {
            .move_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP16,
            .conditional_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_DPP16,
            .masked_move_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP16_MASKED,
        },
        {
            .move_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP,
            .conditional_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_DPP,
            .masked_move_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP_MASKED,
        },
};

static const loom_amdgpu_dpp_descriptor_family_t*
loom_amdgpu_select_dpp_descriptor_family(
    const loom_low_descriptor_set_t* descriptor_set) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuDppDescriptorFamilies); ++i) {
    const loom_amdgpu_dpp_descriptor_family_t* family =
        &kLoomAmdgpuDppDescriptorFamilies[i];
    if (loom_amdgpu_descriptor_set_has_ref(descriptor_set, family->move_ref)) {
      return family;
    }
  }
  return NULL;
}

loom_amdgpu_descriptor_ref_t loom_amdgpu_select_dpp_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_dpp_descriptor_family_t* family =
      loom_amdgpu_select_dpp_descriptor_family(descriptor_set);
  return family == NULL ? LOOM_AMDGPU_DESCRIPTOR_REF_NONE : family->move_ref;
}

bool loom_amdgpu_select_direct_xor_lane_recipe(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t width,
    uint32_t lane_xor, loom_amdgpu_direct_xor_lane_recipe_t* out_recipe) {
  *out_recipe = (loom_amdgpu_direct_xor_lane_recipe_t){
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .conditional_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .masked_move_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  if (!loom_amdgpu_u32_is_power_of_two(width) || lane_xor == 0 ||
      lane_xor >= width) {
    return false;
  }

  const loom_amdgpu_dpp_descriptor_family_t* dpp_family =
      loom_amdgpu_select_dpp_descriptor_family(descriptor_set);
  const loom_amdgpu_descriptor_ref_t dpp_descriptor_ref =
      dpp_family == NULL ? LOOM_AMDGPU_DESCRIPTOR_REF_NONE
                         : dpp_family->move_ref;
  uint32_t dpp_ctrl = 0;
  if (lane_xor < IREE_ARRAYSIZE(kLoomAmdgpuDppXorControls)) {
    const loom_amdgpu_dpp_xor_control_t* control =
        &kLoomAmdgpuDppXorControls[lane_xor];
    dpp_ctrl = dpp_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP16
                   ? control->dpp16_control
                   : control->legacy_control;
  }
  if (dpp_ctrl != 0 && dpp_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    *out_recipe = (loom_amdgpu_direct_xor_lane_recipe_t){
        .descriptor_ref = dpp_descriptor_ref,
        .conditional_ref = loom_amdgpu_descriptor_set_has_ref(
                               descriptor_set, dpp_family->conditional_ref)
                               ? dpp_family->conditional_ref
                               : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
        .masked_move_ref = loom_amdgpu_descriptor_set_has_ref(
                               descriptor_set, dpp_family->masked_move_ref)
                               ? dpp_family->masked_move_ref
                               : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
        .immediate = dpp_ctrl,
        .crosslane_kind = LOOM_AMDGPU_CROSSLANE_DPP,
    };
    return true;
  }

  if (lane_xor < 32 &&
      loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_SWIZZLE_B32)) {
    *out_recipe = (loom_amdgpu_direct_xor_lane_recipe_t){
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_SWIZZLE_B32,
        .conditional_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
        .masked_move_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
        .immediate = LOOM_AMDGPU_DS_SWIZZLE_BITMASK_AND |
                     (lane_xor << LOOM_AMDGPU_DS_SWIZZLE_BITMASK_XOR_SHIFT),
        .crosslane_kind = LOOM_AMDGPU_CROSSLANE_SWIZZLE,
    };
    return true;
  }

  return false;
}

iree_status_t loom_amdgpu_emit_subgroup_bpermute_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_byte_offset, uint32_t static_byte_offset,
    loom_value_id_t low_source_value, loom_type_t lane_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      low_source_byte_offset,
      low_source_value,
  };
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  if (static_byte_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset"), static_byte_offset, attrs,
        IREE_ARRAYSIZE(attrs), &attr_count));
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_subgroup_readlane_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t source_value, uint32_t lane, loom_type_t result_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("lane"), lane, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, &source_value, 1,
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_subgroup_readlane_sgpr_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t source_value, loom_value_id_t source_lane,
    loom_type_t result_type, loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      source_value,
      source_lane,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_direct_crosslane_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_amdgpu_crosslane_kind_t crosslane_kind,
    loom_value_id_t low_source_value, uint32_t immediate, loom_type_t lane_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(crosslane_kind == LOOM_AMDGPU_CROSSLANE_DPP ||
              crosslane_kind == LOOM_AMDGPU_CROSSLANE_SWIZZLE);
  loom_named_attr_t attrs[1];
  iree_host_size_t attr_count = 0;
  const iree_string_view_t immediate_attr_name =
      crosslane_kind == LOOM_AMDGPU_CROSSLANE_DPP ? IREE_SV("dpp_ctrl")
                                                  : IREE_SV("offset");
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, immediate_attr_name, immediate,
                                  attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, &low_source_value, 1,
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_subgroup_lane_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_lane, loom_type_t lane_type,
    loom_value_id_t* out_low_source_byte_offset) {
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 2,
      low_source_lane, lane_type, out_low_source_byte_offset);
}
