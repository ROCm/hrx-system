// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/collective_payload.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_subgroup_shuffle_width_is_supported(
    int64_t width, uint32_t wavefront_size) {
  return width > 0 && width <= (int64_t)wavefront_size &&
         loom_amdgpu_u32_is_power_of_two((uint32_t)width);
}

typedef enum loom_amdgpu_subgroup_shuffle_shape_failure_e {
  LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_NONE = 0,
  LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_EXACT_WIDTH = 1,
  LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_POWER_OF_TWO_WIDTH = 2,
  LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_LANE_RANGE = 3,
} loom_amdgpu_subgroup_shuffle_shape_failure_t;

typedef struct loom_amdgpu_subgroup_shuffle_shape_t {
  // Exact cluster width in lanes.
  uint32_t width;
  // Exact source-lane offset within the cluster, or UINT32_MAX when dynamic.
  uint32_t exact_offset;
} loom_amdgpu_subgroup_shuffle_shape_t;

static bool loom_amdgpu_subgroup_shuffle_resolve_width(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* op, uint32_t wavefront_size,
    loom_amdgpu_subgroup_shuffle_shape_t* out_shape,
    loom_amdgpu_subgroup_shuffle_shape_failure_t* out_failure) {
  out_shape->width = 0;
  *out_failure = LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_NONE;

  int64_t width = 0;
  if (!loom_amdgpu_value_as_exact_i32(
          module, fact_table, loom_kernel_subgroup_shuffle_width(op), &width)) {
    *out_failure = LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_EXACT_WIDTH;
    return false;
  }
  if (!loom_amdgpu_subgroup_shuffle_width_is_supported(width, wavefront_size)) {
    *out_failure =
        LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_POWER_OF_TWO_WIDTH;
    return false;
  }

  out_shape->width = (uint32_t)width;
  return true;
}

static bool loom_amdgpu_subgroup_shuffle_resolve_offset(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* op, loom_amdgpu_subgroup_shuffle_shape_t* out_shape,
    loom_amdgpu_subgroup_shuffle_shape_failure_t* out_failure) {
  out_shape->exact_offset = UINT32_MAX;
  *out_failure = LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_NONE;

  const loom_value_id_t source_offset = loom_kernel_subgroup_shuffle_offset(op);
  int64_t offset = 0;
  if (loom_amdgpu_value_as_exact_i32(module, fact_table, source_offset,
                                     &offset)) {
    if (offset < 0 || offset >= (int64_t)out_shape->width) {
      *out_failure = LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_LANE_RANGE;
      return false;
    }
    out_shape->exact_offset = (uint32_t)offset;
    return true;
  }

  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, source_offset);
  if (loom_value_facts_is_float(facts) || facts.range_lo < 0 ||
      facts.range_hi >= (int64_t)out_shape->width) {
    *out_failure = LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_LANE_RANGE;
    return false;
  }
  return true;
}

static iree_string_view_t loom_amdgpu_subgroup_shuffle_shape_failure_key(
    loom_amdgpu_subgroup_shuffle_shape_failure_t failure) {
  switch (failure) {
    case LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_EXACT_WIDTH:
      return IREE_SV("subgroup_shuffle.exact_width");
    case LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_POWER_OF_TWO_WIDTH:
      return IREE_SV("subgroup_shuffle.power_of_two_width");
    case LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_LANE_RANGE:
      return IREE_SV("subgroup_shuffle.lane_range");
    case LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE(
      "AMDGPU subgroup shuffle shape failure requires reason");
  return IREE_SV("subgroup_shuffle.exact_width");
}

iree_status_t loom_amdgpu_select_kernel_subgroup_shuffle_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_shuffle_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_shuffle_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_shuffle_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_shuffle_value(source_op);
  loom_amdgpu_subgroup_payload_kind_t payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(module, value, &payload_kind,
                                                   &register_count)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  if (!loom_amdgpu_select_subgroup_wavefront_size(context, &wavefront_size)) {
    return iree_ok_status();
  }

  loom_amdgpu_subgroup_shuffle_shape_t shape = {0};
  loom_amdgpu_subgroup_shuffle_shape_failure_t shape_failure =
      LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_NONE;
  if (!loom_amdgpu_subgroup_shuffle_resolve_width(
          module, loom_low_lower_context_fact_table(context), source_op,
          wavefront_size, &shape, &shape_failure)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_select_direct_subgroup_width(context, wavefront_size,
                                                shape.width)) {
    return iree_ok_status();
  }

  if (!loom_amdgpu_subgroup_shuffle_resolve_offset(
          module, loom_low_lower_context_fact_table(context), source_op, &shape,
          &shape_failure)) {
    return iree_ok_status();
  }

  const loom_kernel_subgroup_shuffle_mode_t mode =
      loom_kernel_subgroup_shuffle_mode(source_op);
  loom_amdgpu_crosslane_kind_t crosslane_kind = LOOM_AMDGPU_CROSSLANE_BPERMUTE;
  uint32_t crosslane_immediate = 0;
  bool descriptor_present = false;
  loom_amdgpu_direct_xor_lane_recipe_t xor_recipe = {0};
  if (mode == LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_XOR &&
      shape.exact_offset != UINT32_MAX &&
      loom_amdgpu_select_direct_xor_lane_recipe(
          loom_low_lower_context_descriptor_set(context), shape.width,
          shape.exact_offset, &xor_recipe)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, xor_recipe.descriptor_ref, &out_plan->descriptor));
    crosslane_kind = xor_recipe.crosslane_kind;
    crosslane_immediate = xor_recipe.immediate;
    descriptor_present = true;
  }
  if (!descriptor_present) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
        &out_plan->descriptor, &descriptor_present));
    if (!descriptor_present) {
      return iree_ok_status();
    }
  }

  out_plan->value = value;
  out_plan->source_offset = loom_kernel_subgroup_shuffle_offset(source_op);
  out_plan->result = loom_kernel_subgroup_shuffle_result(source_op);
  out_plan->valid = loom_kernel_subgroup_shuffle_valid(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  out_plan->crosslane_kind = crosslane_kind;
  out_plan->mode = mode;
  out_plan->crosslane_immediate = crosslane_immediate;
  out_plan->exact_offset = shape.exact_offset;
  out_plan->width = shape.width;
  out_plan->wavefront_size = wavefront_size;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_mask_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &mask_type, 1, &low_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_valid_true(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t lane_type, loom_type_t valid_type, loom_value_id_t* out_valid) {
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
      &zero));
  return loom_amdgpu_emit_subgroup_mask_compare(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, zero, zero,
      valid_type, out_valid);
}

static iree_status_t loom_amdgpu_emit_subgroup_cluster_base(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_shuffle_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* out_cluster_base) {
  *out_cluster_base = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, lane_id,
      ~(plan->width - 1u), lane_type, out_cluster_base);
}

static iree_status_t loom_amdgpu_emit_subgroup_lane_relative_to_width(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_shuffle_plan_t* plan, loom_value_id_t lane_id,
    loom_type_t lane_type, loom_value_id_t* out_lane_relative) {
  *out_lane_relative = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, lane_id,
      plan->width - 1u, lane_type, out_lane_relative);
}

static iree_status_t loom_amdgpu_emit_subgroup_shuffle_index_source_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_shuffle_plan_t* plan, loom_value_id_t lane_id,
    loom_value_id_t low_source_offset, loom_type_t lane_type,
    loom_value_id_t* out_source_lane) {
  *out_source_lane = LOOM_VALUE_ID_INVALID;
  if (plan->width == plan->wavefront_size) {
    if (plan->exact_offset == UINT32_MAX) {
      *out_source_lane = low_source_offset;
      return iree_ok_status();
    }
    return loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->exact_offset, lane_type, out_source_lane);
  }

  loom_value_id_t cluster_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_cluster_base(
      context, source_op, plan, lane_id, lane_type, &cluster_base));
  if (plan->exact_offset == 0) {
    *out_source_lane = cluster_base;
    return iree_ok_status();
  }
  if (plan->exact_offset == UINT32_MAX) {
    return loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, cluster_base,
        low_source_offset, lane_type, out_source_lane);
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT, cluster_base,
      plan->exact_offset, lane_type, out_source_lane);
}

static iree_status_t loom_amdgpu_emit_subgroup_shuffle_source_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_shuffle_plan_t* plan, loom_type_t lane_type,
    loom_type_t valid_type, loom_value_id_t* out_source_byte_offset,
    loom_value_id_t* out_valid) {
  *out_source_byte_offset = LOOM_VALUE_ID_INVALID;
  *out_valid = LOOM_VALUE_ID_INVALID;

  loom_value_id_t low_source_offset = LOOM_VALUE_ID_INVALID;
  if (plan->exact_offset == UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, plan->source_offset, &low_source_offset));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, low_source_offset, &low_source_offset));
  }

  if (plan->mode == LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_INDEX &&
      plan->width == plan->wavefront_size) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_valid_true(
        context, source_op, lane_type, valid_type, out_valid));
    if (plan->exact_offset == UINT32_MAX) {
      return loom_amdgpu_emit_subgroup_lane_byte_offset(
          context, source_op, low_source_offset, lane_type,
          out_source_byte_offset);
    }
    return loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->exact_offset * 4u, lane_type, out_source_byte_offset);
  }

  loom_value_id_t lane_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, lane_type, &lane_id));

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  switch (plan->mode) {
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_INDEX: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_shuffle_index_source_lane(
          context, source_op, plan, lane_id, low_source_offset, lane_type,
          &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_valid_true(
          context, source_op, lane_type, valid_type, out_valid));
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_XOR:
      if (plan->exact_offset == 0) {
        source_lane = lane_id;
      } else if (plan->exact_offset == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32, lane_id,
            low_source_offset, lane_type, &source_lane));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
            lane_id, plan->exact_offset, lane_type, &source_lane));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_valid_true(
          context, source_op, lane_type, valid_type, out_valid));
      break;
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_UP: {
      loom_value_id_t offset = low_source_offset;
      if (plan->exact_offset != UINT32_MAX) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
            plan->exact_offset, lane_type, &offset));
      }
      loom_value_id_t lane_for_compare = lane_id;
      if (plan->width != plan->wavefront_size) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_relative_to_width(
            context, source_op, plan, lane_id, lane_type, &lane_for_compare));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_U32, lane_id,
          offset, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_mask_compare(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32,
          lane_for_compare, offset, valid_type, out_valid));
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_DOWN: {
      loom_value_id_t offset = low_source_offset;
      if (plan->exact_offset != UINT32_MAX) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
            plan->exact_offset, lane_type, &offset));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, lane_id,
          offset, lane_type, &source_lane));
      loom_value_id_t source_lane_for_compare = source_lane;
      if (plan->width != plan->wavefront_size) {
        loom_value_id_t lane_relative = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_relative_to_width(
            context, source_op, plan, lane_id, lane_type, &lane_relative));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
            lane_relative, offset, lane_type, &source_lane_for_compare));
      }
      loom_value_id_t width = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, plan->width,
          lane_type, &width));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_mask_compare(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
          source_lane_for_compare, width, valid_type, out_valid));
      break;
    }
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_COUNT_:
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU subgroup shuffle lowering requires a supported mode");
      IREE_BUILTIN_UNREACHABLE();
  }

  return loom_amdgpu_emit_subgroup_lane_byte_offset(
      context, source_op, source_lane, lane_type, out_source_byte_offset);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_shuffle(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_shuffle_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_type_t valid_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->valid, &valid_type));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collective_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t low_valid = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
  if (plan->crosslane_kind != LOOM_AMDGPU_CROSSLANE_BPERMUTE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_valid_true(
        context, source_op, lane_type, valid_type, &low_valid));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_shuffle_source_byte_offset(
        context, source_op, plan, lane_type, valid_type,
        &low_source_byte_offset, &low_valid));
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_value(context, plan->valid, low_valid));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &low_source_register));
    switch (plan->crosslane_kind) {
      case LOOM_AMDGPU_CROSSLANE_DPP:
      case LOOM_AMDGPU_CROSSLANE_SWIZZLE: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_direct_crosslane_register(
            context, source_op, &plan->descriptor, plan->crosslane_kind,
            low_source_register, plan->crosslane_immediate, lane_type,
            &result_registers[i]));
        break;
      }
      case LOOM_AMDGPU_CROSSLANE_BPERMUTE: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
            context, source_op, &plan->descriptor, low_source_byte_offset,
            /*static_byte_offset=*/0, low_source_register, lane_type,
            &result_registers[i]));
        break;
      }
    }
  }

  return loom_amdgpu_collective_bind_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

void loom_amdgpu_mark_subgroup_shuffle_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_shuffle_plan_t* plan) {
  (void)source_op;
  loom_low_lower_require_source_value_storage(context, plan->value);
  if (plan->exact_offset == UINT32_MAX) {
    loom_low_lower_require_source_value_storage(context, plan->source_offset);
  }
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_shuffle(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_shuffle_value(op);
  loom_amdgpu_subgroup_payload_kind_t unused_payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(
          module, value, &unused_payload_kind, &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("subgroup_shuffle.payload"));
  }

  const uint32_t wavefront_size = loom_amdgpu_target_wavefront_size(bundle);
  if (!loom_amdgpu_wavefront_size_is_valid(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_shuffle.wavefront_size"));
  }

  loom_amdgpu_subgroup_shuffle_shape_t shape = {0};
  loom_amdgpu_subgroup_shuffle_shape_failure_t shape_failure =
      LOOM_AMDGPU_SUBGROUP_SHUFFLE_SHAPE_FAILURE_NONE;
  if (!loom_amdgpu_subgroup_shuffle_resolve_width(
          module, loom_target_low_legality_fact_table(context), op,
          wavefront_size, &shape, &shape_failure)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_subgroup_shuffle_shape_failure_key(shape_failure));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_direct_subgroup_width(
      context, op, wavefront_size, shape.width,
      IREE_SV("subgroup_shuffle.native_width")));

  if (!loom_amdgpu_subgroup_shuffle_resolve_offset(
          module, loom_target_low_legality_fact_table(context), op, &shape,
          &shape_failure)) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_subgroup_shuffle_shape_failure_key(shape_failure));
  }

  loom_amdgpu_direct_xor_lane_recipe_t unused_xor_recipe = {0};
  if (loom_kernel_subgroup_shuffle_mode(op) ==
          LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_XOR &&
      shape.exact_offset != UINT32_MAX &&
      loom_amdgpu_select_direct_xor_lane_recipe(
          loom_target_low_legality_descriptor_set(context), shape.width,
          shape.exact_offset, &unused_xor_recipe)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      IREE_SV("descriptor.ds_bpermute_b32"));
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_match(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  return loom_amdgpu_low_legality_reject(
      context, op, IREE_SV("subgroup_match.target_legalization"));
}

#undef LOOM_AMDGPU_MAX_SUBGROUP_TREE_STEPS
