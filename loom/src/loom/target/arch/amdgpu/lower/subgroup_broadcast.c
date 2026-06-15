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

static bool loom_amdgpu_subgroup_wavefront_size_is_supported(
    uint32_t wavefront_size) {
  return wavefront_size == 32 || wavefront_size == 64;
}
static bool loom_amdgpu_subgroup_exact_i32_value(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, int64_t* out_value) {
  *out_value = 0;

  int64_t fact_value = 0;
  if (loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &fact_value) &&
      fact_value >= INT32_MIN && fact_value <= INT32_MAX) {
    *out_value = fact_value;
    return true;
  }

  return loom_amdgpu_module_value_as_i32_constant(module, value_id, out_value);
}
static bool loom_amdgpu_subgroup_i32_lane_is_in_range(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, uint32_t wavefront_size) {
  int64_t exact_value = 0;
  if (loom_amdgpu_subgroup_exact_i32_value(module, fact_table, value_id,
                                           &exact_value)) {
    return exact_value >= 0 && exact_value < (int64_t)wavefront_size;
  }

  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, value_id);
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return iree_ok_status();
  }

  const loom_value_id_t source_lane =
      loom_kernel_subgroup_broadcast_lane(source_op);
  int64_t exact_source_lane = 0;
  const bool source_lane_is_exact = loom_amdgpu_subgroup_exact_i32_value(
      module, loom_low_lower_context_fact_table(context), source_lane,
      &exact_source_lane);
  if (!loom_amdgpu_subgroup_i32_lane_is_in_range(
          module, loom_low_lower_context_fact_table(context), source_lane,
          wavefront_size)) {
    return iree_ok_status();
  }

  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      &out_plan->descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  out_plan->value = value;
  out_plan->result = loom_kernel_subgroup_broadcast_result(source_op);
  out_plan->source_lane = source_lane;
  if (source_lane_is_exact) {
    out_plan->exact_source_lane = (uint32_t)exact_source_lane;
  }
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
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

static iree_status_t loom_amdgpu_emit_subgroup_bpermute_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_byte_offset, loom_value_id_t low_source_value,
    loom_type_t lane_type, loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      low_source_byte_offset,
      low_source_value,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
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

static iree_status_t loom_amdgpu_emit_subgroup_lane_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_lane, loom_type_t lane_type,
    loom_value_id_t* out_low_source_byte_offset) {
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 2,
      low_source_lane, lane_type, out_low_source_byte_offset);
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
  if (plan->exact_source_lane != UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        plan->exact_source_lane * 4u, lane_type, &low_source_byte_offset));
  } else {
    loom_value_id_t low_source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->source_lane,
                                                     &low_source_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, low_source_lane, &low_source_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
        context, source_op, low_source_lane, lane_type,
        &low_source_byte_offset));
  }

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_collective_payload_register(
        context, source_op, plan->register_count, low_value, i, lane_type,
        &low_source_register));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
        context, source_op, &plan->descriptor, low_source_byte_offset,
        low_source_register, lane_type, &result_registers[i]));
  }

  return loom_amdgpu_collective_bind_payload_result(
      context, source_op, plan->result, plan->register_count, result_registers);
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
static iree_status_t loom_amdgpu_low_legality_verify_subgroup_wavefront(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t constraint_key, uint32_t* out_wavefront_size) {
  *out_wavefront_size = 0;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, out_wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(*out_wavefront_size)) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
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
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_wavefront_size(bundle, &wavefront_size));
  if (!loom_amdgpu_subgroup_wavefront_size_is_supported(wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast.wavefront_size"));
  }

  const loom_value_id_t source_lane = loom_kernel_subgroup_broadcast_lane(op);
  if (!loom_amdgpu_subgroup_i32_lane_is_in_range(
          module, loom_target_low_legality_fact_table(context), source_lane,
          wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_broadcast.lane_range"));
  }

  return loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      IREE_SV("descriptor.ds_bpermute_b32"));
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
