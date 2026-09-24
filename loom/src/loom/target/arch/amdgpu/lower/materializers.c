// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/materializers.h"

#include <stdint.h>

#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_low_type_can_materialize_as_vgpr_registers(
    loom_low_lower_context_t* context, loom_type_t low_type) {
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  if (unit_count == 0 || unit_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return false;
  }
  return loom_amdgpu_low_type_is_register_class(
             context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR) ||
         loom_amdgpu_low_type_is_register_class(context, low_type,
                                                LOOM_AMDGPU_REG_CLASS_ID_SGPR);
}

iree_status_t loom_amdgpu_value_can_materialize_as_vgpr_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize) {
  *out_can_materialize = false;
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, value_id, &low_type));
  *out_can_materialize =
      loom_amdgpu_low_type_can_materialize_as_vgpr_registers(context, low_type);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_value, out_low_value);
}

iree_status_t loom_amdgpu_value_can_materialize_as_vgpr_i32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize) {
  *out_can_materialize = false;
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), value_id);
  if (!loom_amdgpu_type_is_i32(type) &&
      loom_amdgpu_vector_i32_register_count(type) == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_value_can_materialize_as_vgpr_registers(
      context, source_op, value_id, out_can_materialize);
}

iree_status_t loom_amdgpu_value_can_materialize_as_vgpr_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize) {
  *out_can_materialize = false;
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), value_id);
  if (!loom_amdgpu_type_is_f32(type) &&
      loom_amdgpu_vector_f32_register_count(type) == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_value_can_materialize_as_vgpr_registers(
      context, source_op, value_id, out_can_materialize);
}

iree_status_t loom_amdgpu_value_can_materialize_as_vgpr_i64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize) {
  *out_can_materialize = false;
  if (!loom_amdgpu_type_is_i64(loom_module_value_type(
          loom_low_lower_context_module(context), value_id))) {
    return iree_ok_status();
  }
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, value_id, &low_type));
  if (loom_low_register_type_unit_count(low_type) != 2) {
    return iree_ok_status();
  }
  *out_can_materialize =
      loom_amdgpu_low_type_is_register_class(context, low_type,
                                             LOOM_AMDGPU_REG_CLASS_ID_VGPR) ||
      loom_amdgpu_low_type_is_register_class(context, low_type,
                                             LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_value_can_materialize_as_vgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize) {
  *out_can_materialize = false;
  if (!loom_amdgpu_value_is_address_scalar(context, value_id)) {
    return iree_ok_status();
  }
  int64_t constant = 0;
  if (loom_amdgpu_value_as_address_constant(context, value_id, &constant) &&
      constant >= 0 && constant <= UINT32_MAX) {
    *out_can_materialize = true;
    return iree_ok_status();
  }
  return loom_amdgpu_value_can_materialize_as_vgpr_registers(
      context, source_op, value_id, out_can_materialize);
}

iree_status_t loom_amdgpu_value_can_materialize_as_sgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize) {
  *out_can_materialize = false;
  if (!loom_amdgpu_value_is_address_scalar(context, value_id)) {
    return iree_ok_status();
  }
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, value_id, &low_type));
  if (!loom_amdgpu_low_type_is_register_class(context, low_type,
                                              LOOM_AMDGPU_REG_CLASS_ID_SGPR)) {
    return iree_ok_status();
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  *out_can_materialize = unit_count == 1 || unit_count == 2;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_value_can_materialize_as_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize) {
  *out_can_materialize = false;
  if (!loom_amdgpu_type_is_i1(loom_module_value_type(
          loom_low_lower_context_module(context), value_id))) {
    return iree_ok_status();
  }
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, value_id, &low_type));
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  const bool is_scc = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC);
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  *out_can_materialize = (is_sgpr && (unit_count == 1 || unit_count == 2)) ||
                         (is_scc && unit_count == 1);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lookup_or_materialize_i1_integer(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, uint32_t register_class_id,
    loom_value_id_t* out_low_value) {
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));
  const loom_type_t low_type =
      loom_module_value_type(loom_low_lower_context_module(context), low_value);
  if (loom_low_register_type_unit_count(low_type) == 2) {
    // A native mask describes individual lanes, not a uniform 64-bit integer.
    IREE_ASSERT_EQ(register_class_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    loom_type_t lane_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
    loom_value_id_t low_false = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_true = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
        &low_false));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 1, lane_type,
        &low_true));
    return loom_amdgpu_emit_vgpr_select(context, source_op, low_false, low_true,
                                        low_value, lane_type, out_low_value);
  }
  if (loom_low_register_type_class_id(low_type) ==
      LOOM_AMDGPU_REG_CLASS_ID_SCC) {
    loom_type_t word_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &word_type));
    loom_value_id_t low_false = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_true = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, word_type,
        &low_false));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 1, word_type,
        &low_true));
    const loom_value_id_t operands[] = {low_true, low_false, low_value};
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32, operands,
        IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &word_type, 1,
        &select_op));
    low_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  }
  if (register_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_value, out_low_value);
  }
  *out_low_value = low_value;
  return iree_ok_status();
}
