// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/control_mask.h"

#include <string.h>

#include "loom/codegen/low/lower/control_plan.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

bool loom_amdgpu_low_type_is_sgpr_pair(loom_low_lower_context_t* context,
                                       loom_type_t low_type) {
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  return is_sgpr && loom_low_register_type_unit_count(low_type) == 2;
}

bool loom_amdgpu_control_argument_is_lane_mask(
    loom_low_lower_context_t* context, loom_low_lower_block_ref_t block,
    uint16_t argument_index) {
  const loom_block_t* source =
      loom_low_lower_control_signature_source_block(context, block);
  return argument_index < source->arg_count &&
         loom_amdgpu_native_type_is_lane_mask(
             loom_block_arg_type(loom_low_lower_context_module(context), source,
                                 argument_index),
             loom_low_lower_control_argument_type(context, block,
                                                  argument_index));
}

static loom_amdgpu_mask_merge_flags_t loom_amdgpu_mask_merge_constant(
    loom_low_lower_context_t* context, loom_value_slice_t args, uint16_t index,
    loom_amdgpu_mask_merge_flags_t zero_flag,
    loom_amdgpu_mask_merge_flags_t one_flag) {
  bool value = false;
  if (index < args.count &&
      loom_amdgpu_value_as_i1_constant(context, args.values[index], &value)) {
    return value ? one_flag : zero_flag;
  }
  return 0;
}

iree_status_t loom_amdgpu_prepare_masked_merge(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_block_t* source_merge_dest,
    loom_low_lower_block_ref_t merge_low_dest, loom_value_slice_t false_args,
    loom_value_slice_t true_args,
    const loom_amdgpu_mask_merge_flags_t** out_flags) {
  *out_flags = NULL;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint16_t count =
      loom_low_lower_control_carrier_argument_offset(context, merge_low_dest);
  loom_amdgpu_mask_merge_flags_t* flags = NULL;
  const uint32_t previous_error_count =
      loom_low_lower_context_error_count(context);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       i < count && iree_status_is_ok(status) &&
       loom_low_lower_context_error_count(context) == previous_error_count;
       ++i) {
    const loom_type_t type =
        loom_low_lower_control_argument_type(context, merge_low_dest, i);
    const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
        context, type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
        context, type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    const bool is_lane_mask =
        i < source_merge_dest->arg_count &&
        loom_amdgpu_native_type_is_lane_mask(
            loom_block_arg_type(module, source_merge_dest, i), type);
    const uint32_t word_count = loom_low_register_type_unit_count(type);
    const uint32_t maximum_word_count =
        LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES * (is_lane_mask ? 2u : 1u);
    if ((!is_sgpr && !is_vgpr) || word_count == 0 ||
        word_count > maximum_word_count) {
      status = loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_merge_register_values"));
      continue;
    }
    loom_amdgpu_mask_merge_flags_t choice = 0;
    if (is_sgpr && !is_lane_mask) {
      choice = LOOM_AMDGPU_MASK_MERGE_UNIFORM;
    } else if (is_lane_mask && word_count == 2) {
      choice = loom_amdgpu_mask_merge_constant(
                   context, false_args, i, LOOM_AMDGPU_MASK_MERGE_FALSE_ZERO,
                   LOOM_AMDGPU_MASK_MERGE_FALSE_ONE) |
               loom_amdgpu_mask_merge_constant(context, true_args, i,
                                               LOOM_AMDGPU_MASK_MERGE_TRUE_ZERO,
                                               LOOM_AMDGPU_MASK_MERGE_TRUE_ONE);
    }
    if (!choice) {
      continue;
    }
    if (flags == NULL) {
      status = loom_low_lower_allocate_function_array(
          context, count, sizeof(*flags), (void**)&flags);
      if (iree_status_is_ok(status)) {
        memset(flags, 0, count * sizeof(*flags));
      }
    }
    if (iree_status_is_ok(status)) {
      flags[i] = choice;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_flags = flags;
  }
  return status;
}

iree_status_t loom_amdgpu_emit_zero_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, value_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, value_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (!is_sgpr && !is_vgpr) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_register_values"));
  }

  const uint32_t lane_count = loom_low_register_type_unit_count(value_type);
  const uint32_t maximum_word_count =
      LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES * (is_sgpr ? 2u : 1u);
  if (lane_count == 0 || lane_count > maximum_word_count) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_register_values"));
  }
  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(value_type, 1);
  const loom_amdgpu_descriptor_ref_t descriptor =
      is_sgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32
              : LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32;
  if (lane_count == 1) {
    return loom_amdgpu_emit_const_u32(context, source_op, descriptor, 0,
                                      lane_type, out_value);
  }

  loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, descriptor, 0, lane_type, &zero_lane));
  loom_value_id_t lanes[2 * LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    lanes[i] = zero_lane;
  }
  return loom_amdgpu_build_low_register_range(
      context, source_op, lanes, lane_count, value_type, out_value);
}

iree_status_t loom_amdgpu_emit_zero_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(value_type, 1);
  loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, lane_type,
      &zero_lane));
  const loom_value_id_t lanes[] = {zero_lane, zero_lane};
  return loom_amdgpu_build_low_register_range(
      context, source_op, lanes, IREE_ARRAYSIZE(lanes), value_type, out_value);
}

static iree_status_t loom_amdgpu_emit_masked_merge_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t lane_type, bool is_uniform,
    loom_value_id_t* out_merged_value) {
  const loom_value_id_t select_operands[] = {
      is_uniform ? low_true_value : low_false_value,
      is_uniform ? low_false_value : low_true_value,
      low_condition,
  };
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op,
      is_uniform ? LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32
                 : LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      select_operands, IREE_ARRAYSIZE(select_operands),
      loom_named_attr_slice_empty(), &lane_type, 1, &select_op));
  *out_merged_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_native_i1_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &op));
  *out_result = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_native_i1_mask_merge(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_amdgpu_mask_merge_flags_t flags,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t mask_type,
    loom_value_id_t* out_merged_value) {
  *out_merged_value = LOOM_VALUE_ID_INVALID;
  const bool true_is_constant =
      iree_any_bit_set(flags, LOOM_AMDGPU_MASK_MERGE_TRUE_ZERO |
                                  LOOM_AMDGPU_MASK_MERGE_TRUE_ONE);
  const bool true_constant =
      iree_any_bit_set(flags, LOOM_AMDGPU_MASK_MERGE_TRUE_ONE);
  const bool false_is_constant =
      iree_any_bit_set(flags, LOOM_AMDGPU_MASK_MERGE_FALSE_ZERO |
                                  LOOM_AMDGPU_MASK_MERGE_FALSE_ONE);
  const bool false_constant =
      iree_any_bit_set(flags, LOOM_AMDGPU_MASK_MERGE_FALSE_ONE);
  if (true_is_constant && false_is_constant) {
    if (true_constant && false_constant) {
      *out_merged_value = saved_exec;
      return iree_ok_status();
    }
    if (!true_constant && !false_constant) {
      return loom_amdgpu_emit_zero_native_i1_mask(context, source_op, mask_type,
                                                  out_merged_value);
    }
  }

  if (low_true_value == low_false_value) {
    *out_merged_value = low_true_value;
    return iree_ok_status();
  }

  if (true_is_constant && true_constant && false_is_constant &&
      !false_constant) {
    return loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, saved_exec,
        low_condition, mask_type, out_merged_value);
  }

  loom_value_id_t true_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, saved_exec,
      low_condition, mask_type, &true_condition));

  if (true_is_constant && !true_constant && false_is_constant &&
      false_constant) {
    return loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
        true_condition, saved_exec, mask_type, out_merged_value);
  }

  loom_value_id_t true_mask = LOOM_VALUE_ID_INVALID;
  bool true_mask_is_zero = false;
  if (true_is_constant && true_constant) {
    true_mask = true_condition;
  } else if (true_is_constant && !true_constant) {
    true_mask_is_zero = true;
  } else if (low_true_value == low_condition) {
    true_mask = true_condition;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        true_condition, low_true_value, mask_type, &true_mask));
  }

  if ((false_is_constant && !false_constant) ||
      low_false_value == low_condition) {
    if (true_mask_is_zero) {
      return loom_amdgpu_emit_zero_native_i1_mask(context, source_op, mask_type,
                                                  out_merged_value);
    }
    *out_merged_value = true_mask;
    return iree_ok_status();
  }

  loom_value_id_t false_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, true_condition,
      saved_exec, mask_type, &false_condition));
  loom_value_id_t false_mask = LOOM_VALUE_ID_INVALID;
  if (false_is_constant && false_constant) {
    false_mask = false_condition;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        false_condition, low_false_value, mask_type, &false_mask));
  }
  if (true_mask_is_zero) {
    *out_merged_value = false_mask;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_native_i1_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64, true_mask,
      false_mask, mask_type, out_merged_value);
}

iree_status_t loom_amdgpu_emit_masked_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t participating_lanes, loom_amdgpu_mask_merge_flags_t flags,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_value_id_t* inout_uniform_condition,
    loom_type_t value_type, loom_value_id_t* out_merged_value) {
  *out_merged_value = LOOM_VALUE_ID_INVALID;
  const bool is_uniform =
      iree_any_bit_set(flags, LOOM_AMDGPU_MASK_MERGE_UNIFORM);
  const uint32_t lane_count = loom_low_register_type_unit_count(value_type);
  if (!is_uniform && loom_amdgpu_low_type_is_register_class(
                         context, value_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR)) {
    if (lane_count == 2) {
      return loom_amdgpu_emit_native_i1_mask_merge(
          context, source_op, participating_lanes, flags, low_false_value,
          low_true_value, low_condition, value_type, out_merged_value);
    }
    const loom_type_t mask_type =
        loom_low_register_carrier_type_with_unit_count(value_type, 2);
    const uint32_t mask_count = lane_count / 2;
    loom_value_id_t masks[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    for (uint32_t i = 0; i < mask_count; ++i) {
      loom_value_id_t false_mask = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_false_value, 2 * i, mask_type, &false_mask));
      loom_value_id_t true_mask = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_true_value, 2 * i, mask_type, &true_mask));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_merge(
          context, source_op, participating_lanes, flags, false_mask, true_mask,
          low_condition, mask_type, &masks[i]));
    }
    return loom_amdgpu_build_low_register_range(
        context, source_op, masks, mask_count, value_type, out_merged_value);
  }

  if (is_uniform && *inout_uniform_condition == LOOM_VALUE_ID_INVALID) {
    // Any surviving true lane supplies the uniform result. A retired lane's
    // predicate must not select an arm whose payload no longer participates.
    loom_type_t scc_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
    const loom_type_t condition_types[] = {
        loom_module_value_type(loom_low_lower_context_module(context),
                               low_condition),
        scc_type,
    };
    const loom_value_id_t condition_operands[] = {low_condition,
                                                  participating_lanes};
    loom_op_t* condition_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64_SCC,
        condition_operands, IREE_ARRAYSIZE(condition_operands),
        loom_named_attr_slice_empty(), condition_types,
        IREE_ARRAYSIZE(condition_types), &condition_op));
    *inout_uniform_condition = loom_op_const_results(condition_op)[1];
  }
  if (is_uniform) {
    low_condition = *inout_uniform_condition;
  }

  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(value_type, 1);
  if (lane_count == 1) {
    if (!is_uniform) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, low_false_value, &low_false_value));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, low_true_value, &low_true_value));
    }
    return loom_amdgpu_emit_masked_merge_lane(
        context, source_op, low_false_value, low_true_value, low_condition,
        lane_type, is_uniform, out_merged_value);
  }

  if (!is_uniform) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_false_value, &low_false_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_true_value, &low_true_value));
  }

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t false_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_false_value, i, lane_type, &false_lane));
    loom_value_id_t true_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_true_value, i, lane_type, &true_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_masked_merge_lane(
        context, source_op, false_lane, true_lane, low_condition, lane_type,
        is_uniform, &lanes[i]));
  }

  return loom_amdgpu_build_low_register_range(
      context, source_op, lanes, lane_count, value_type, out_merged_value);
}
