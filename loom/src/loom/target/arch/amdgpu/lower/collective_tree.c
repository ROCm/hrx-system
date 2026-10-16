// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/collective_tree.h"

#include "loom/target/arch/amdgpu/lower/collective_combine.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/types.h"

#define LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS 6u

static const loom_amdgpu_collective_combine_dpp_form_t
    kLoomAmdgpuCollectiveDppCombineForms[] = {
        LOOM_AMDGPU_COLLECTIVE_COMBINE_DPP_FORM_DPP16,
        LOOM_AMDGPU_COLLECTIVE_COMBINE_DPP_FORM_LEGACY,
};

static bool loom_amdgpu_subgroup_reduce_dpp_row_is_applicable(
    uint32_t active_lane_count) {
  if (active_lane_count <= LOOM_AMDGPU_DPP_ROW_LANE_COUNT) {
    return true;
  }
  return loom_amdgpu_u32_is_power_of_two(active_lane_count);
}

loom_amdgpu_subgroup_reduce_crosslane_kind_t
loom_amdgpu_subgroup_reduce_choose_crosslane_kind(
    uint32_t active_lane_count,
    const loom_low_lower_resolved_descriptor_t* dpp_move,
    const loom_low_lower_resolved_descriptor_t* dpp_combine,
    const loom_low_lower_resolved_descriptor_t* permlanex16) {
  if (!loom_amdgpu_subgroup_reduce_dpp_row_is_applicable(active_lane_count) ||
      (!dpp_move->descriptor && !dpp_combine->descriptor)) {
    return LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_BPERMUTE;
  }
  if (active_lane_count > LOOM_AMDGPU_DPP_ROW_LANE_COUNT &&
      permlanex16->descriptor) {
    return LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_PERMLANEX16;
  }
  return LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_BPERMUTE;
}

static iree_status_t loom_amdgpu_resolve_collective_dpp_combine_descriptor(
    loom_low_lower_context_t* context, loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_low_lower_resolved_descriptor_t* out_descriptor, bool* out_present) {
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_present = false;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuCollectiveDppCombineForms); ++i) {
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_collective_combine_dpp_descriptor_ref(
            kind, payload_kind, kLoomAmdgpuCollectiveDppCombineForms[i],
            &descriptor_ref)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, descriptor_ref, out_descriptor, out_present));
    if (*out_present) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_resolve_collective_dpp_descriptors(
    loom_low_lower_context_t* context, loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_low_lower_resolved_descriptor_t* dpp_move,
    loom_low_lower_resolved_descriptor_t* dpp_combine) {
  bool present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_collective_dpp_combine_descriptor(
      context, kind, payload_kind, dpp_combine, &present));
  return loom_amdgpu_resolve_descriptor_ref_if_present(
      context,
      loom_amdgpu_select_dpp_descriptor_ref(
          loom_low_lower_context_descriptor_set(context)),
      dpp_move, &present);
}

iree_status_t loom_amdgpu_select_subgroup_reduce_crosslane_kind(
    loom_low_lower_context_t* context, uint32_t active_lane_count,
    loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_low_lower_resolved_descriptor_t* dpp_move,
    loom_low_lower_resolved_descriptor_t* dpp_combine,
    loom_low_lower_resolved_descriptor_t* permlanex16,
    loom_amdgpu_subgroup_reduce_crosslane_kind_t* out_crosslane_kind) {
  *dpp_move = (loom_low_lower_resolved_descriptor_t){0};
  *dpp_combine = (loom_low_lower_resolved_descriptor_t){0};
  *permlanex16 = (loom_low_lower_resolved_descriptor_t){0};
  *out_crosslane_kind = LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_BPERMUTE;
  if (!loom_amdgpu_subgroup_reduce_dpp_row_is_applicable(active_lane_count)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_collective_dpp_descriptors(
      context, kind, payload_kind, dpp_move, dpp_combine));
  if (!dpp_combine->descriptor && !dpp_move->descriptor) {
    return iree_ok_status();
  }

  if (active_lane_count > LOOM_AMDGPU_DPP_ROW_LANE_COUNT) {
    bool permlanex16_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERMLANEX16_B32_SRC12_INLINE,
        permlanex16, &permlanex16_descriptor_present));
  }
  *out_crosslane_kind = loom_amdgpu_subgroup_reduce_choose_crosslane_kind(
      active_lane_count, dpp_move, dpp_combine, permlanex16);
  return iree_ok_status();
}

bool loom_amdgpu_subgroup_reduce_dpp_row_descriptor_is_present(
    const loom_low_descriptor_set_t* descriptor_set, loom_combining_kind_t kind,
    loom_amdgpu_subgroup_payload_kind_t payload_kind) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuCollectiveDppCombineForms); ++i) {
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (loom_amdgpu_collective_combine_dpp_descriptor_ref(
            kind, payload_kind, kLoomAmdgpuCollectiveDppCombineForms[i],
            &descriptor_ref) &&
        loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref)) {
      return true;
    }
  }
  return loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP16) ||
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP);
}

static iree_status_t loom_amdgpu_emit_subgroup_permlanex16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t source_value, loom_type_t lane_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[2];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("selector_low"), 0, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("selector_high"), 0, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, &source_value, 1,
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_subgroup_dpp_combine_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, uint32_t dpp_ctrl, loom_type_t lane_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("dpp_ctrl"), dpp_ctrl, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_xor_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane_id, uint32_t offset, loom_type_t lane_type,
    loom_value_id_t* out_source_lane) {
  *out_source_lane = LOOM_VALUE_ID_INVALID;
  if (offset == 0) {
    *out_source_lane = lane_id;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT, lane_id,
      offset, lane_type, out_source_lane);
}

iree_status_t loom_amdgpu_emit_subgroup_combine(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static uint32_t loom_amdgpu_subgroup_reduce_first_offset(
    uint32_t active_lane_count) {
  if (active_lane_count <= 1) {
    return 0;
  }
  uint32_t offset = 1;
  while ((offset << 1) < active_lane_count) {
    offset <<= 1;
  }
  return offset;
}

iree_status_t loom_amdgpu_emit_subgroup_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* select_descriptor,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t guard, loom_type_t lane_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      false_value,
      true_value,
      guard,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, select_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_subgroup_lane_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_guard) {
  *out_guard = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_guard = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_xor_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  const bool precompute_step_values = plan->register_count > 1;
  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  uint32_t step_count = 0;
  const uint32_t first_offset =
      loom_amdgpu_subgroup_reduce_first_offset(plan->active_lane_count);
  if (precompute_step_values) {
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      IREE_ASSERT_LT(step_count, IREE_ARRAYSIZE(source_byte_offsets));
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane(
          context, source_op, lane_id, offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, source_lane, lane_type,
          &source_byte_offsets[step_count]));
      ++step_count;
    }
  }

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];
    uint32_t step_index = 0;
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
      if (precompute_step_values) {
        low_source_byte_offset = source_byte_offsets[step_index++];
      } else {
        loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane(
            context, source_op, lane_id, offset, lane_type, &source_lane));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
            context, source_op, source_lane, lane_type,
            &low_source_byte_offset));
      }
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
          context, source_op, &plan->bpermute_descriptor,
          low_source_byte_offset, /*static_byte_offset=*/0, accumulator,
          lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &accumulator));
    }
    inout_registers[i] = accumulator;
  }

  return iree_ok_status();
}

static uint32_t loom_amdgpu_subgroup_reduce_dpp_ctrl(uint32_t lane_count) {
  switch (lane_count) {
    case 2:
      return LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1;
    case 4:
      return LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_2;
    case 8:
      return LOOM_AMDGPU_DPP_CTRL_ROW_HALF_MIRROR;
    case 16:
      return LOOM_AMDGPU_DPP_CTRL_ROW_MIRROR;
    default:
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU subgroup DPP reduce lowering requires a supported step");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_amdgpu_subgroup_reduce_cross_row_uses_permlanex16(
    const loom_amdgpu_subgroup_reduce_plan_t* plan, uint32_t offset) {
  return plan->crosslane_kind ==
             LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_PERMLANEX16 &&
         offset == LOOM_AMDGPU_DPP_ROW_LANE_COUNT;
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_cross_row_xor_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  const bool precompute_step_values = plan->register_count > 1;
  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  uint32_t step_count = 0;
  if (precompute_step_values) {
    for (uint32_t offset = LOOM_AMDGPU_DPP_ROW_LANE_COUNT;
         offset < plan->active_lane_count; offset <<= 1) {
      if (loom_amdgpu_subgroup_reduce_cross_row_uses_permlanex16(plan,
                                                                 offset)) {
        continue;
      }
      IREE_ASSERT_LT(step_count, IREE_ARRAYSIZE(source_byte_offsets));
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane(
          context, source_op, lane_id, offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, source_lane, lane_type,
          &source_byte_offsets[step_count]));
      ++step_count;
    }
  }

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];
    uint32_t step_index = 0;
    for (uint32_t offset = LOOM_AMDGPU_DPP_ROW_LANE_COUNT;
         offset < plan->active_lane_count; offset <<= 1) {
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      if (loom_amdgpu_subgroup_reduce_cross_row_uses_permlanex16(plan,
                                                                 offset)) {
        IREE_ASSERT(plan->permlanex16_descriptor.descriptor != NULL);
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_permlanex16_register(
            context, source_op, &plan->permlanex16_descriptor, accumulator,
            lane_type, &peer));
      } else {
        loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
        if (precompute_step_values) {
          low_source_byte_offset = source_byte_offsets[step_index++];
        } else {
          loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_xor_lane(
              context, source_op, lane_id, offset, lane_type, &source_lane));
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
              context, source_op, source_lane, lane_type,
              &low_source_byte_offset));
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
            context, source_op, &plan->bpermute_descriptor,
            low_source_byte_offset, /*static_byte_offset=*/0, accumulator,
            lane_type, &peer));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &accumulator));
    }
    inout_registers[i] = accumulator;
  }

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_dpp_row_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  const uint32_t row_lane_count =
      iree_min(plan->active_lane_count, LOOM_AMDGPU_DPP_ROW_LANE_COUNT);
  // Independent components provide useful work between dependent DPP stages.
  // The target hazard planner supplies any remaining required spacing.
  for (uint32_t lane_count = 2; lane_count <= row_lane_count;
       lane_count <<= 1) {
    const uint32_t dpp_ctrl = loom_amdgpu_subgroup_reduce_dpp_ctrl(lane_count);
    for (uint32_t i = 0; i < plan->register_count; ++i) {
      loom_value_id_t accumulator = inout_registers[i];
      if (plan->dpp_combine_descriptor.descriptor != NULL) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_dpp_combine_register(
            context, source_op, &plan->dpp_combine_descriptor, accumulator,
            accumulator, dpp_ctrl, lane_type, &inout_registers[i]));
        continue;
      }
      IREE_ASSERT(plan->dpp_descriptor.descriptor != NULL);
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_direct_crosslane_register(
          context, source_op, &plan->dpp_descriptor, LOOM_AMDGPU_CROSSLANE_DPP,
          accumulator, dpp_ctrl, lane_type, &peer));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &inout_registers[i]));
    }
  }

  return loom_amdgpu_emit_subgroup_reduce_cross_row_xor_tree(
      context, source_op, plan, lane_id, lane_type, inout_registers);
}

static iree_status_t loom_amdgpu_emit_subgroup_add_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lane_id, uint32_t offset, loom_type_t lane_type,
    loom_value_id_t* out_source_lane) {
  *out_source_lane = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, lane_id,
      offset, lane_type, out_source_lane);
}

static iree_status_t loom_amdgpu_emit_subgroup_reduce_down_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t dynamic_active_lane_count,
    loom_value_id_t* inout_registers) {
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t active_lane_count = dynamic_active_lane_count;
  if (active_lane_count == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->active_lane_count, lane_type, &active_lane_count));
  }
  loom_value_id_t first_lane_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
      &first_lane_offset));

  const bool precompute_step_values = plan->register_count > 1;
  loom_value_id_t source_byte_offsets[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {
      0};
  loom_value_id_t guards[LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS] = {0};
  uint32_t step_count = 0;
  const uint32_t first_offset =
      loom_amdgpu_subgroup_reduce_first_offset(plan->active_lane_count);
  if (precompute_step_values) {
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      IREE_ASSERT_LT(step_count, IREE_ARRAYSIZE(source_byte_offsets));
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_add_lane(
          context, source_op, lane_id, offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, source_lane, lane_type,
          &source_byte_offsets[step_count]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
          context, source_op, &plan->guard_descriptor, source_lane,
          active_lane_count, mask_type, &guards[step_count]));
      ++step_count;
    }
  }

  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t accumulator = inout_registers[i];
    uint32_t step_index = 0;
    for (uint32_t offset = first_offset; offset != 0; offset >>= 1) {
      loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
      loom_value_id_t guard = LOOM_VALUE_ID_INVALID;
      if (precompute_step_values) {
        low_source_byte_offset = source_byte_offsets[step_index++];
        guard = guards[step_index - 1];
      } else {
        loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_add_lane(
            context, source_op, lane_id, offset, lane_type, &source_lane));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
            context, source_op, source_lane, lane_type,
            &low_source_byte_offset));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_compare(
            context, source_op, &plan->guard_descriptor, source_lane,
            active_lane_count, mask_type, &guard));
      }
      loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
          context, source_op, &plan->bpermute_descriptor,
          low_source_byte_offset, /*static_byte_offset=*/0, accumulator,
          lane_type, &peer));
      // An absent peer leaves the accumulator unchanged. Substituting an
      // identity before combining can destroy NaNs or the sign of zero.
      loom_value_id_t combined = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
          context, source_op, &plan->combine_descriptor, accumulator, peer,
          lane_type, &combined));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_select(
          context, source_op, &plan->select_descriptor, accumulator, combined,
          guard, lane_type, &accumulator));
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
        context, source_op, &plan->bpermute_descriptor, first_lane_offset,
        /*static_byte_offset=*/0, accumulator, lane_type, &inout_registers[i]));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_subgroup_reduce_tree(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t dynamic_active_lane_count,
    loom_value_id_t* inout_registers) {
  if (dynamic_active_lane_count == LOOM_VALUE_ID_INVALID &&
      (plan->active_lane_count <= 1 ||
       loom_amdgpu_u32_is_power_of_two(plan->active_lane_count))) {
    if (plan->crosslane_kind ==
            LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_BPERMUTE ||
        plan->crosslane_kind ==
            LOOM_AMDGPU_SUBGROUP_REDUCE_CROSSLANE_DPP_ROW_PERMLANEX16) {
      return loom_amdgpu_emit_subgroup_reduce_dpp_row_tree(
          context, source_op, plan, lane_id, lane_type, inout_registers);
    }
    return loom_amdgpu_emit_subgroup_reduce_xor_tree(
        context, source_op, plan, lane_id, lane_type, inout_registers);
  }
  return loom_amdgpu_emit_subgroup_reduce_down_tree(
      context, source_op, plan, lane_id, lane_type, dynamic_active_lane_count,
      inout_registers);
}

iree_status_t loom_amdgpu_emit_subgroup_reduce_scalar_broadcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* inout_registers) {
  IREE_ASSERT_EQ(plan->wavefront_size, 64);
  IREE_ASSERT_EQ(plan->active_lane_count, 64);
  IREE_ASSERT(plan->readlane_descriptor.descriptor != NULL);

  loom_amdgpu_subgroup_reduce_plan_t half_wave_plan = *plan;
  half_wave_plan.active_lane_count = plan->wavefront_size / 2u;
  half_wave_plan.publication_kind =
      LOOM_AMDGPU_SUBGROUP_REDUCE_PUBLICATION_ALL_LANES;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_reduce_tree(
      context, source_op, &half_wave_plan, lane_id, lane_type,
      LOOM_VALUE_ID_INVALID, inout_registers));

  // Lane 0 holds each low-half result and lane 63 receives the full result
  // after combining it with the corresponding high-half result.
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_half = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_readlane_register(
        context, source_op, &plan->readlane_descriptor, inout_registers[i], 0,
        sgpr_type, &low_half));

    loom_value_id_t full_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_combine(
        context, source_op, &plan->combine_descriptor, low_half,
        inout_registers[i], lane_type, &full_result));

    loom_value_id_t full_result_scalar = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_readlane_register(
        context, source_op, &plan->readlane_descriptor, full_result, 63,
        sgpr_type, &full_result_scalar));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
        context, source_op, full_result_scalar, &inout_registers[i]));
  }
  return iree_ok_status();
}
