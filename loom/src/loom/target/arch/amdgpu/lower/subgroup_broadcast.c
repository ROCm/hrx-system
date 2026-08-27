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

typedef struct loom_amdgpu_subgroup_broadcast_shape_t {
  // Exact source lane when known during planning, or UINT32_MAX when dynamic.
  uint32_t exact_source_lane;
  // Whether every active lane observes the same source lane.
  bool source_lane_is_subgroup_uniform;
} loom_amdgpu_subgroup_broadcast_shape_t;

static bool loom_amdgpu_subgroup_broadcast_resolve_shape(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_lane, uint32_t wavefront_size,
    loom_amdgpu_subgroup_broadcast_shape_t* out_shape) {
  out_shape->exact_source_lane = UINT32_MAX;

  int64_t exact_value = 0;
  if (loom_amdgpu_value_as_exact_i32(module, fact_table, source_lane,
                                     &exact_value)) {
    if (exact_value < 0 || exact_value >= (int64_t)wavefront_size) {
      return false;
    }
    out_shape->exact_source_lane = (uint32_t)exact_value;
    out_shape->source_lane_is_subgroup_uniform = true;
    return true;
  }

  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, source_lane);
  out_shape->source_lane_is_subgroup_uniform =
      loom_value_facts_is_subgroup_uniform(facts);
  return !loom_value_facts_is_float(facts) && facts.range_lo >= 0 &&
         facts.range_hi < (int64_t)wavefront_size;
}

iree_status_t loom_amdgpu_select_kernel_subgroup_broadcast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_broadcast_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_broadcast_plan_t){0};
  out_plan->exact_source_lane = UINT32_MAX;
  *out_selected = false;
  if (!loom_kernel_subgroup_broadcast_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_broadcast_value(source_op);
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

  const loom_value_id_t source_lane =
      loom_kernel_subgroup_broadcast_lane(source_op);
  loom_amdgpu_subgroup_broadcast_shape_t shape = {0};
  if (!loom_amdgpu_subgroup_broadcast_resolve_shape(
          module, loom_low_lower_context_fact_table(context), source_lane,
          wavefront_size, &shape)) {
    return iree_ok_status();
  }

  bool descriptors_present = false;
  if (loom_amdgpu_select_direct_subgroup_width(context, wavefront_size,
                                               wavefront_size)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
        &out_plan->exchange_descriptor, &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
    out_plan->strategy = LOOM_AMDGPU_SUBGROUP_BROADCAST_STRATEGY_BPERMUTE;
  } else {
    if (shape.exact_source_lane == UINT32_MAX &&
        !shape.source_lane_is_subgroup_uniform) {
      return iree_ok_status();
    }
    const loom_amdgpu_descriptor_ref_t exchange_descriptor_ref =
        shape.exact_source_lane == UINT32_MAX
            ? LOOM_AMDGPU_DESCRIPTOR_REF_V_READLANE_B32_SRC1_SGPR
            : LOOM_AMDGPU_DESCRIPTOR_REF_V_READLANE_B32_SRC1_INLINE;
    const loom_amdgpu_descriptor_resolution_t resolutions[] = {
        {
            .descriptor_ref = exchange_descriptor_ref,
            .out_descriptor = &out_plan->exchange_descriptor,
        },
        {
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
            .out_descriptor = &out_plan->scalar_copy_descriptor,
        },
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
        context, resolutions, IREE_ARRAYSIZE(resolutions),
        &descriptors_present));
    if (!descriptors_present) {
      return iree_ok_status();
    }
    out_plan->strategy =
        LOOM_AMDGPU_SUBGROUP_BROADCAST_STRATEGY_SCALAR_READLANE;
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_broadcast_result(source_op);
  out_plan->source_lane = source_lane;
  out_plan->exact_source_lane = shape.exact_source_lane;
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_broadcast_first_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_broadcast_first_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_broadcast_first_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_broadcast_first_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t value =
      loom_kernel_subgroup_broadcast_first_value(source_op);
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

  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
      &out_plan->descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_broadcast_first_result(source_op);
  out_plan->payload_kind = payload_kind;
  out_plan->register_count = register_count;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_readfirstlane_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_value, loom_type_t result_type,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      low_source_value,
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

iree_status_t loom_amdgpu_lower_kernel_subgroup_broadcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_broadcast_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collective_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t low_source_byte_offset = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source_lane = LOOM_VALUE_ID_INVALID;
  loom_type_t scalar_type = loom_type_none();
  switch (plan->strategy) {
    case LOOM_AMDGPU_SUBGROUP_BROADCAST_STRATEGY_BPERMUTE:
      if (plan->exact_source_lane != UINT32_MAX) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
            plan->exact_source_lane * 4u, lane_type, &low_source_byte_offset));
      } else {
        IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
            context, plan->source_lane, &low_source_lane));
        IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
            context, source_op, low_source_lane, &low_source_lane));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
            context, source_op, low_source_lane, lane_type,
            &low_source_byte_offset));
      }
      break;
    case LOOM_AMDGPU_SUBGROUP_BROADCAST_STRATEGY_SCALAR_READLANE: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &scalar_type));
      if (plan->exact_source_lane == UINT32_MAX) {
        IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
            context, plan->source_lane, &low_source_lane));
      }
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported AMDGPU subgroup broadcast strategy");
  }

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &low_source_register));
    switch (plan->strategy) {
      case LOOM_AMDGPU_SUBGROUP_BROADCAST_STRATEGY_BPERMUTE: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
            context, source_op, &plan->exchange_descriptor,
            low_source_byte_offset, /*static_byte_offset=*/0,
            low_source_register, lane_type, &result_registers[i]));
        break;
      }
      case LOOM_AMDGPU_SUBGROUP_BROADCAST_STRATEGY_SCALAR_READLANE: {
        loom_value_id_t scalar_register = LOOM_VALUE_ID_INVALID;
        if (plan->exact_source_lane != UINT32_MAX) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_readlane_register(
              context, source_op, &plan->exchange_descriptor,
              low_source_register, plan->exact_source_lane, scalar_type,
              &scalar_register));
        } else {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_readlane_sgpr_register(
              context, source_op, &plan->exchange_descriptor,
              low_source_register, low_source_lane, scalar_type,
              &scalar_register));
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_unary(
            context, source_op, &plan->scalar_copy_descriptor, scalar_register,
            lane_type, &result_registers[i]));
        break;
      }
      default:
        IREE_ASSERT_UNREACHABLE(
            "unsupported AMDGPU subgroup broadcast strategy");
    }
  }

  return loom_amdgpu_collective_bind_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

void loom_amdgpu_mark_subgroup_broadcast_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_broadcast_plan_t* plan) {
  (void)source_op;
  loom_low_lower_require_source_value_storage(context, plan->value);
  if (plan->exact_source_lane == UINT32_MAX) {
    loom_low_lower_require_source_value_storage(context, plan->source_lane);
  }
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_broadcast_first(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_broadcast_first_plan_t* plan) {
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t read_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &read_type));

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collective_lookup_payload(
      context, source_op, plan->value, plan->payload_kind, &low_value));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &low_source_register));

    loom_value_id_t low_read_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_readfirstlane_register(
        context, source_op, &plan->descriptor, low_source_register, read_type,
        &low_read_register));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
        context, source_op, low_read_register, &result_registers[i]));
  }

  return loom_amdgpu_collective_bind_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_broadcast_value(op);
  loom_amdgpu_subgroup_payload_kind_t unused_payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(
          module, value, &unused_payload_kind, &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast.payload"));
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, IREE_SV("subgroup_broadcast.wavefront_size"),
      &wavefront_size));

  const loom_value_id_t source_lane = loom_kernel_subgroup_broadcast_lane(op);
  loom_amdgpu_subgroup_broadcast_shape_t shape = {0};
  if (!loom_amdgpu_subgroup_broadcast_resolve_shape(
          module, loom_target_low_legality_fact_table(context), source_lane,
          wavefront_size, &shape)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast.lane_range"));
  }

  if (loom_amdgpu_target_supports_direct_subgroup_width(
          loom_amdgpu_target_facts_cast(
              loom_target_low_legality_target_facts(context)),
          wavefront_size, wavefront_size)) {
    return loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
        IREE_SV("descriptor.ds_bpermute_b32"));
  }
  if (shape.exact_source_lane == UINT32_MAX &&
      !shape.source_lane_is_subgroup_uniform) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast.uniform_lane"));
  }
  const loom_amdgpu_descriptor_ref_t exchange_descriptor_ref =
      shape.exact_source_lane == UINT32_MAX
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_READLANE_B32_SRC1_SGPR
          : LOOM_AMDGPU_DESCRIPTOR_REF_V_READLANE_B32_SRC1_INLINE;
  const loom_amdgpu_low_legality_descriptor_requirement_t requirements[] = {
      {
          .constraint_key = IREE_SVL("descriptor.v_readlane_b32"),
          .descriptor_ref = exchange_descriptor_ref,
      },
      {
          .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
          .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
      },
  };
  return loom_amdgpu_low_legality_verify_descriptor_requirements(
      context, op, requirements, IREE_ARRAYSIZE(requirements));
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast_first(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_value_id_t value = loom_kernel_subgroup_broadcast_first_value(op);
  loom_amdgpu_subgroup_payload_kind_t unused_payload_kind =
      LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  uint32_t unused_register_count = 0;
  if (!loom_amdgpu_collective_payload_is_supported(
          module, value, &unused_payload_kind, &unused_register_count)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast_first.payload"));
  }

  uint32_t unused_wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, IREE_SV("subgroup_broadcast_first.wavefront_size"),
      &unused_wavefront_size));
  return loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
      IREE_SV("descriptor.v_readfirstlane_b32"));
}
