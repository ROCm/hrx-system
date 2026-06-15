// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/collective_payload.h"

#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"

static loom_amdgpu_subgroup_payload_kind_t loom_amdgpu_collective_payload_kind(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = 0;
  if (loom_amdgpu_type_is_i32(type)) {
    *out_register_count = 1;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR;
  }
  if (loom_amdgpu_type_is_f32(type)) {
    *out_register_count = 1;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR;
  }
  const uint32_t i32_lane_count = loom_amdgpu_vector_i32_lane_count(type);
  if (i32_lane_count != 0) {
    *out_register_count = i32_lane_count;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR;
  }
  const uint32_t f32_lane_count = loom_amdgpu_vector_f32_lane_count(type);
  if (f32_lane_count != 0) {
    *out_register_count = f32_lane_count;
    return LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR;
  }
  return LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
}

bool loom_amdgpu_collective_payload_is_supported(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_amdgpu_subgroup_payload_kind_t* out_kind,
    uint32_t* out_register_count) {
  *out_kind = LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
  const loom_type_t type = loom_module_value_type(module, value_id);
  *out_kind = loom_amdgpu_collective_payload_kind(type, out_register_count);
  return *out_kind != LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE;
}

bool loom_amdgpu_collective_payload_is_integer(
    loom_amdgpu_subgroup_payload_kind_t payload_kind) {
  return payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR ||
         payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR;
}

bool loom_amdgpu_collective_payload_is_float(
    loom_amdgpu_subgroup_payload_kind_t payload_kind) {
  return payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR ||
         payload_kind == LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR;
}

iree_status_t loom_amdgpu_collective_lookup_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_value_id_t* out_low_value) {
  switch (payload_kind) {
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_SCALAR:
      return loom_amdgpu_lookup_or_materialize_vgpr_i32(context, source_op,
                                                        value, out_low_value);
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_SCALAR:
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_I32_VECTOR:
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_F32_VECTOR:
      return loom_low_lower_lookup_value(context, value, out_low_value);
    case LOOM_AMDGPU_SUBGROUP_PAYLOAD_NONE:
      break;
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "AMDGPU collective lowering has no payload kind");
}

iree_status_t loom_amdgpu_collective_payload_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t register_count, loom_value_id_t low_value, uint32_t register_index,
    loom_type_t lane_type, loom_value_id_t* out_register) {
  *out_register = LOOM_VALUE_ID_INVALID;
  if (register_count == 1) {
    *out_register = low_value;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_value,
                                    register_index, lane_type, out_register);
}

iree_status_t loom_amdgpu_collective_bind_payload_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, uint32_t register_count,
    const loom_value_id_t* result_registers) {
  if (register_count == 1) {
    return loom_low_lower_bind_value(context, source_result,
                                     result_registers[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), result_registers, register_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}
