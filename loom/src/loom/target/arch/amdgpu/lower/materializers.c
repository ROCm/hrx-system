// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/materializers.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

iree_status_t loom_amdgpu_prepare_vgpr_literal(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void** out_plan) {
  *out_plan = NULL;
  const loom_type_t actual_type =
      loom_low_lower_value_binding_type(context, source_value);
  if (loom_amdgpu_low_type_is_register_class(context, actual_type,
                                             LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
    return iree_ok_status();
  }
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_value);
  uint32_t bits = 0;
  if (loom_amdgpu_type_is_f32(source_type)) {
    if (!loom_amdgpu_value_as_f32_constant(context, source_value, &bits)) {
      return iree_ok_status();
    }
  } else {
    int64_t value = 0;
    if (loom_amdgpu_type_is_i32(source_type)) {
      if (!loom_amdgpu_value_as_i32_constant(context, source_value, &value)) {
        return iree_ok_status();
      }
    } else if (!loom_amdgpu_type_is_address_scalar(source_type) ||
               !loom_amdgpu_value_as_address_constant(context, source_value,
                                                      &value) ||
               value < 0 || value > UINT32_MAX) {
      return iree_ok_status();
    }
    bits = (uint32_t)value;
  }
  uint32_t* literal = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*literal), (void**)&literal));
  *literal = bits;
  *out_plan = literal;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_prepared_vgpr_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* plan,
    loom_value_id_t* out_low_value) {
  if (!plan) {
    return loom_amdgpu_lookup_or_materialize_vgpr_registers(
        context, source_op, source_value, out_low_value);
  }
  loom_type_t type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &type));
  return loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      *(const uint32_t*)plan, type, out_low_value);
}

iree_status_t loom_amdgpu_prepare_native_i1_mask(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void** out_plan) {
  static const bool kFalse = false;
  static const bool kTrue = true;
  *out_plan = NULL;
  bool constant = false;
  if (loom_amdgpu_value_as_i1_constant(context, source_value, &constant)) {
    *out_plan = constant ? &kTrue : &kFalse;
  }
  return iree_ok_status();
}

loom_type_t loom_amdgpu_materialized_vgpr_register_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value) {
  const loom_type_t type =
      loom_low_lower_value_binding_type(context, source_value);
  if (loom_low_register_type_class_id(type) == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return type;
  }
  return loom_low_register_type(
      loom_low_lower_context_descriptor_set(context)->stable_id,
      LOOM_AMDGPU_REG_CLASS_ID_VGPR, loom_low_register_type_unit_count(type));
}

loom_type_t loom_amdgpu_materialized_vop3_binary_rhs_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value) {
  const loom_amdgpu_descriptor_set_info_t* info =
      loom_amdgpu_target_info_descriptor_set_at(
          loom_low_lower_context_descriptor_set(context)
              ->descriptor_set_ordinal);
  if (loom_amdgpu_descriptor_set_info_has_flags(
          info, LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOP3_TWO_SCALAR_SOURCES)) {
    return loom_low_lower_value_binding_type(context, source_value);
  }
  return loom_amdgpu_materialized_vgpr_register_type(context, source_value);
}

loom_type_t loom_amdgpu_materialized_vgpr_address_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value) {
  (void)source_value;
  return loom_low_register_type(
      loom_low_lower_context_descriptor_set(context)->stable_id,
      LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
}

loom_type_t loom_amdgpu_materialized_sgpr_address_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value) {
  (void)source_value;
  return loom_low_register_type(
      loom_low_lower_context_descriptor_set(context)->stable_id,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
}

loom_type_t loom_amdgpu_materialized_native_i1_mask_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value) {
  (void)source_value;
  return loom_low_register_type(
      loom_low_lower_context_descriptor_set(context)->stable_id,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
}

static bool loom_amdgpu_low_type_can_materialize_as_vgpr_registers(
    loom_low_lower_context_t* context, loom_type_t low_type) {
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  if (unit_count == 0) {
    return false;
  }
  if (loom_amdgpu_low_type_is_register_class(context, low_type,
                                             LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
    return true;
  }
  return unit_count <= LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES &&
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
  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_value, out_low_value);
}

iree_status_t loom_amdgpu_value_can_materialize_as_vop3_binary_rhs(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize) {
  return loom_amdgpu_value_can_materialize_as_vgpr_registers(
      context, source_op, value_id, out_can_materialize);
}

iree_status_t loom_amdgpu_lookup_or_materialize_vop3_binary_rhs(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  if (loom_amdgpu_descriptor_set_info_has_flags(
          descriptor_set_info,
          LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOP3_TWO_SCALAR_SOURCES)) {
    *out_low_value = loom_low_lower_lookup_value(context, source_value);
    return iree_ok_status();
  }
  return loom_amdgpu_lookup_or_materialize_vgpr_registers(
      context, source_op, source_value, out_low_value);
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
  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);
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

iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (!loom_low_type_is_register(low_type) ||
      loom_low_register_type_unit_count(low_type) != 2) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU i64 VGPR materializer selected an unsupported low value");
    IREE_BUILTIN_UNREACHABLE();
  }

  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!is_sgpr) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU i64 VGPR materializer selected an unsupported register class");
    IREE_BUILTIN_UNREACHABLE();
  }
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_value, out_low_value);
}

iree_status_t loom_amdgpu_emit_prepared_vgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* plan,
    loom_value_id_t* out_low_value) {
  if (plan) {
    return loom_amdgpu_emit_prepared_vgpr_value(
        context, source_op, source_value, plan, out_low_value);
  }
  *out_low_value = LOOM_VALUE_ID_INVALID;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (loom_amdgpu_type_is_i1(loom_module_value_type(module, source_value))) {
    return loom_amdgpu_lookup_or_materialize_i1_integer(
        context, source_op, source_value, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
        out_low_value);
  }
  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);

  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
    if (unit_count == 1) {
      *out_low_value = low_value;
      return iree_ok_status();
    }
    if (unit_count == 2) {
      const loom_type_t lane_type =
          loom_amdgpu_low_register_lane_type(module, low_value);
      return loom_amdgpu_emit_low_slice(context, source_op, low_value,
                                        /*offset=*/0, lane_type, out_low_value);
    }
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address materializer selected a wrong-width VGPR value");
    IREE_BUILTIN_UNREACHABLE();
  }

  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (is_sgpr) {
    loom_value_id_t low_lane = low_value;
    if (loom_low_register_type_unit_count(low_type) == 2) {
      const loom_type_t lane_type =
          loom_amdgpu_low_register_lane_type(module, low_value);
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, /*offset=*/0, lane_type, &low_lane));
    }
    return loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_lane, out_low_value);
  }
  IREE_ASSERT_UNREACHABLE(
      "AMDGPU address materializer selected an unsupported low value");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_amdgpu_lookup_or_materialize_sgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (loom_amdgpu_type_is_i1(loom_module_value_type(module, source_value))) {
    return loom_amdgpu_lookup_or_materialize_i1_integer(
        context, source_op, source_value, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
        out_low_value);
  }
  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);

  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!is_sgpr) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address materializer selected an incompatible SGPR value");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  if (unit_count != 1 && unit_count != 2) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address materializer selected a wrong-width SGPR value");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (unit_count == 1) {
    *out_low_value = low_value;
    return iree_ok_status();
  }
  const loom_type_t lane_type =
      loom_amdgpu_low_register_lane_type(module, low_value);
  return loom_amdgpu_emit_low_slice(context, source_op, low_value,
                                    /*offset=*/0, lane_type, out_low_value);
}

iree_status_t loom_amdgpu_materialize_uniform_sgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  IREE_ASSERT(unit_count == 1 || unit_count == 2);
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  IREE_ASSERT(is_sgpr || is_vgpr);

  const loom_type_t source_lane_type =
      loom_low_register_carrier_type_with_unit_count(low_type, 1);
  loom_value_id_t low_lane = low_value;
  if (unit_count == 2) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_low_slice(context, source_op, low_value, /*offset=*/0,
                                   source_lane_type, &low_lane));
  }
  if (is_sgpr) {
    *out_low_value = low_lane;
    return iree_ok_status();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  const loom_value_id_t operands[] = {low_lane};
  loom_op_t* readfirstlane_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
      operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      &sgpr_type, 1, &readfirstlane_op));
  *out_low_value =
      loom_value_slice_get(loom_low_op_results(readfirstlane_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_materialize_low_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (is_sgpr && loom_low_register_type_unit_count(low_type) == 2) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));

  loom_value_id_t condition = low_value;
  const bool is_scc = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC);
  if (is_sgpr && loom_low_register_type_unit_count(low_type) == 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr32_nonzero_scc(
        context, source_op, low_value, &condition));
  } else if (!is_scc || loom_low_register_type_unit_count(low_type) != 1) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU native mask materializer selected an unsupported low value");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_op_t* exec_read_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      &mask_type, 1, &exec_read_op));
  const loom_value_id_t exec_mask =
      loom_value_slice_get(loom_low_op_results(exec_read_op), 0);

  loom_value_id_t exec_lanes[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(exec_lanes); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, exec_mask, i, sgpr_type, &exec_lanes[i]));
  }

  loom_value_id_t low_zero32 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, sgpr_type,
      &low_zero32));

  loom_value_id_t mask_lanes[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(mask_lanes); ++i) {
    const loom_value_id_t operands[] = {
        exec_lanes[i],
        low_zero32,
        condition,
    };
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32, operands,
        IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &sgpr_type, 1,
        &select_op));
    mask_lanes[i] = loom_value_slice_get(loom_low_op_results(select_op), 0);
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), mask_lanes,
      IREE_ARRAYSIZE(mask_lanes), mask_type, source_op->location, &concat_op));
  *out_low_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_prepared_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* plan,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  if (plan) {
    if (!*(const bool*)plan) {
      return loom_amdgpu_emit_sgpr64_constant_u64(context, source_op, 0,
                                                  out_low_value);
    }
    // A true predicate covers the lanes active at this use, including when
    // its canonical lowering was produced before a divergent branch.
    loom_type_t mask_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
    loom_op_t* exec_read_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
        /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
        &mask_type, 1, &exec_read_op));
    *out_low_value = loom_value_slice_get(loom_low_op_results(exec_read_op), 0);
    return iree_ok_status();
  }

  loom_value_id_t low_value =
      loom_low_lower_lookup_value(context, source_value);

  return loom_amdgpu_materialize_low_native_i1_mask(context, source_op,
                                                    low_value, out_low_value);
}
