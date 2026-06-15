// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"
#include "loom/util/math.h"

bool loom_amdgpu_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

bool loom_amdgpu_type_is_i16(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I16;
}

bool loom_amdgpu_type_is_i64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I64;
}

bool loom_amdgpu_type_is_i8(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I8;
}

bool loom_amdgpu_type_is_i1(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

bool loom_amdgpu_type_is_address_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET);
}

bool loom_amdgpu_type_is_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

bool loom_amdgpu_type_is_f64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F64;
}

bool loom_amdgpu_type_is_16bit_float(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  return element_type == LOOM_SCALAR_TYPE_F16 ||
         element_type == LOOM_SCALAR_TYPE_BF16;
}

bool loom_amdgpu_type_vector_storage(
    loom_type_t type, loom_amdgpu_vector_storage_t* out_storage) {
  *out_storage = (loom_amdgpu_vector_storage_t){0};
  if (!loom_type_is_vector(type) || !loom_type_is_all_static(type)) {
    return false;
  }
  uint64_t element_count = 0;
  if (!loom_type_static_element_count(type, &element_count) ||
      element_count == 0 || element_count > UINT32_MAX) {
    return false;
  }

  const loom_scalar_type_t element_type = loom_type_element_type(type);
  switch (element_type) {
    case LOOM_SCALAR_TYPE_I1: {
      if (loom_amdgpu_vector_i1_lane_count(type) == 0) {
        return false;
      }
      *out_storage = (loom_amdgpu_vector_storage_t){
          .kind = LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK,
          .element_type = element_type,
          .element_count = (uint32_t)element_count,
          .register_count = (uint32_t)element_count * 2u,
          .element_register_count = 2,
          .element_bit_count = 1,
      };
      return true;
    }
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F32: {
      const uint32_t register_count =
          loom_amdgpu_vector_32bit_register_count(type);
      if (register_count == 0) {
        return false;
      }
      *out_storage = (loom_amdgpu_vector_storage_t){
          .kind = LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT,
          .element_type = element_type,
          .element_count = (uint32_t)element_count,
          .register_count = register_count,
          .element_register_count = 1,
          .element_bit_count = 32,
      };
      return true;
    }
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F64: {
      if (element_count >
          LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / UINT32_C(2)) {
        return false;
      }
      *out_storage = (loom_amdgpu_vector_storage_t){
          .kind = LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT,
          .element_type = element_type,
          .element_count = (uint32_t)element_count,
          .register_count = (uint32_t)element_count * 2u,
          .element_register_count = 2,
          .element_bit_count = 64,
      };
      return true;
    }
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16: {
      uint32_t payload_bit_count = 0;
      uint32_t register_count = 0;
      if (!loom_amdgpu_type_packed_16bit_float_storage(type, &payload_bit_count,
                                                       &register_count)) {
        return false;
      }
      *out_storage = (loom_amdgpu_vector_storage_t){
          .kind = LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT,
          .element_type = element_type,
          .element_count = payload_bit_count / 16u,
          .register_count = register_count,
          .element_register_count = 1,
          .element_bit_count = 16,
      };
      return true;
    }
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16: {
      uint32_t payload_bit_count = 0;
      uint32_t register_count = 0;
      if (!loom_amdgpu_type_packed_integer_storage(type, &payload_bit_count,
                                                   &register_count)) {
        return false;
      }
      const int32_t element_bit_count = loom_scalar_type_bitwidth(element_type);
      if (element_bit_count <= 0) {
        return false;
      }
      *out_storage = (loom_amdgpu_vector_storage_t){
          .kind = LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER,
          .element_type = element_type,
          .element_count = (uint32_t)element_count,
          .register_count = register_count,
          .element_register_count = 1,
          .element_bit_count = (uint32_t)element_bit_count,
      };
      return true;
    }
    default:
      return false;
  }
}

uint32_t loom_amdgpu_static_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type,
                                              uint32_t max_lane_count) {
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type) ||
      loom_type_element_type(type) != element_type) {
    return 0;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 || lane_count > (int64_t)max_lane_count) {
    return 0;
  }
  return (uint32_t)lane_count;
}

uint32_t loom_amdgpu_static_vector_register_count(
    loom_type_t type, loom_scalar_type_t element_type,
    uint32_t max_register_count) {
  if (!loom_type_is_vector(type) || !loom_type_is_all_static(type) ||
      loom_type_element_type(type) != element_type) {
    return 0;
  }
  uint64_t element_count = 0;
  if (!loom_type_static_element_count(type, &element_count) ||
      element_count < 1 || element_count > max_register_count) {
    return 0;
  }
  return (uint32_t)element_count;
}

static uint32_t loom_amdgpu_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type) {
  return loom_amdgpu_static_vector_lane_count(
      type, element_type, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

static uint32_t loom_amdgpu_vector_register_count(
    loom_type_t type, loom_scalar_type_t element_type) {
  return loom_amdgpu_static_vector_register_count(
      type, element_type, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

static bool loom_amdgpu_type_is_vector_32bit_register_range(loom_type_t type) {
  return loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_I32) != 0 ||
         loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_F32) != 0;
}

bool loom_amdgpu_type_is_32bit_memory_payload(loom_type_t type) {
  return loom_amdgpu_type_is_i32(type) || loom_amdgpu_type_is_f32(type) ||
         loom_amdgpu_static_vector_lane_count(
             type, LOOM_SCALAR_TYPE_I32, LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) !=
             0 ||
         loom_amdgpu_static_vector_lane_count(
             type, LOOM_SCALAR_TYPE_F32, LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) !=
             0;
}

static bool loom_amdgpu_type_is_offset(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
}

uint32_t loom_amdgpu_vector_32bit_lane_count(loom_type_t type) {
  const uint32_t i32_lane_count =
      loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_I32);
  return i32_lane_count != 0
             ? i32_lane_count
             : loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_32bit_register_count(loom_type_t type) {
  const uint32_t i32_register_count =
      loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_I32);
  return i32_register_count != 0
             ? i32_register_count
             : loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_F32);
}

bool loom_amdgpu_static_vector_flat_register_from_indices(
    loom_type_t type, const int64_t* indices, uint32_t* out_ordinal) {
  uint32_t ordinal = 0;
  const uint8_t rank = loom_type_rank(type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    const int64_t dimension_size = loom_type_dim_static_size_at(type, axis);
    if (dimension_size < 1 || indices[axis] < 0 ||
        indices[axis] >= dimension_size ||
        ordinal >
            (UINT32_MAX - (uint32_t)indices[axis]) / (uint32_t)dimension_size) {
      return false;
    }
    ordinal = ordinal * (uint32_t)dimension_size + (uint32_t)indices[axis];
  }
  *out_ordinal = ordinal;
  return true;
}

uint32_t loom_amdgpu_vector_i32_lane_count(loom_type_t type) {
  return loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_I32);
}

uint32_t loom_amdgpu_vector_i32_register_count(loom_type_t type) {
  return loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_I32);
}

uint32_t loom_amdgpu_vector_f32_lane_count(loom_type_t type) {
  return loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_f32_register_count(loom_type_t type) {
  return loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_i1_lane_count(loom_type_t type) {
  return loom_amdgpu_static_vector_lane_count(
      type, LOOM_SCALAR_TYPE_I1, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

uint32_t loom_amdgpu_vector_i8_lane_count(loom_type_t type) {
  return loom_amdgpu_static_vector_lane_count(type, LOOM_SCALAR_TYPE_I8,
                                              LOOM_AMDGPU_MAX_PACKED_I8_LANES);
}

bool loom_amdgpu_type_packed_integer_storage(loom_type_t type,
                                             uint32_t* out_payload_bit_count,
                                             uint32_t* out_register_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 || lane_count > INT32_MAX) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (!loom_scalar_type_is_integer(element_type)) {
    return false;
  }
  const int32_t element_bit_count = loom_scalar_type_bitwidth(element_type);
  if (element_bit_count <= 0) {
    return false;
  }
  int64_t total_bit_count = 0;
  if (!iree_checked_mul_i64(lane_count, element_bit_count, &total_bit_count) ||
      total_bit_count <= 0) {
    return false;
  }
  const int64_t register_count = (total_bit_count + 31) / 32;
  if (register_count < 1 ||
      register_count > (int64_t)LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }
  *out_payload_bit_count = (uint32_t)total_bit_count;
  *out_register_count = (uint32_t)register_count;
  return true;
}

bool loom_amdgpu_type_packed_16bit_float_storage(
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 ||
      lane_count > (int64_t)LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (element_type != LOOM_SCALAR_TYPE_F16 &&
      element_type != LOOM_SCALAR_TYPE_BF16) {
    return false;
  }
  const uint32_t register_count = (uint32_t)((lane_count + 1) / 2);
  *out_payload_bit_count = (uint32_t)lane_count * 16u;
  *out_register_count = register_count;
  return true;
}

bool loom_amdgpu_type_is_byte_addressable_view(loom_type_t type) {
  if (!loom_type_is_view(type)) {
    return false;
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  return element_bit_count > 0 && (element_bit_count % 8) == 0;
}

static iree_string_view_t loom_amdgpu_nonempty(iree_string_view_t value,
                                               iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

void loom_amdgpu_low_legality_make_context_params(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_diagnostic_param_t* params) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  params[0] =
      loom_param_string(loom_amdgpu_nonempty(bundle->name, IREE_SV("<empty>")));
  params[1] = loom_param_string(
      loom_amdgpu_nonempty(bundle->export_plan->name, IREE_SV("<empty>")));
  params[2] = loom_param_string(
      loom_amdgpu_nonempty(bundle->config->name, IREE_SV("<empty>")));
  params[3] =
      loom_param_string(loom_target_low_legality_function_name(context));
  params[4] = loom_param_string(
      loom_op_name(loom_target_low_legality_module(context), op));
}

iree_status_t loom_amdgpu_low_legality_reject(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t constraint_key) {
  loom_diagnostic_param_t
      params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 1];
  loom_amdgpu_low_legality_make_context_params(context, op, params);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT] =
      loom_param_string(constraint_key);
  return loom_target_low_legality_emit_error_ref(
      context, op, LOOM_ERR_AMDGPU_023_REF, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_amdgpu_low_legality_verify_descriptor_requirement(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t constraint_key) {
  if (!loom_amdgpu_descriptor_set_has_ref(
          loom_target_low_legality_descriptor_set(context), descriptor_ref)) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_descriptor_requirements(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_amdgpu_low_legality_descriptor_requirement_t* requirements,
    iree_host_size_t requirement_count) {
  for (iree_host_size_t i = 0; i < requirement_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, requirements[i].descriptor_ref,
        requirements[i].constraint_key));
  }
  return iree_ok_status();
}

bool loom_amdgpu_value_is_i32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_address_scalar(loom_low_lower_context_t* context,
                                         loom_value_id_t value_id) {
  return loom_amdgpu_type_is_address_scalar(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_f32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id) {
  return loom_amdgpu_type_is_f32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_16bit_float(loom_low_lower_context_t* context,
                                      loom_value_id_t value_id) {
  return loom_amdgpu_type_is_16bit_float(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_byte_addressable_view(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_byte_addressable_view(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static iree_status_t loom_amdgpu_make_register_type(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(context, reg_class_id, unit_count,
                                           out_type);
}

iree_status_t loom_amdgpu_make_sgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                                        1, out_type);
}

iree_status_t loom_amdgpu_make_sgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                                        unit_count, out_type);
}

iree_status_t loom_amdgpu_make_vgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                        1, out_type);
}

iree_status_t loom_amdgpu_make_scc_type(loom_low_lower_context_t* context,
                                        loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_SCC,
                                        1, out_type);
}

iree_status_t loom_amdgpu_make_vgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                        unit_count, out_type);
}

iree_status_t loom_amdgpu_make_descriptor_row_implicit_resource_type(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  IREE_ASSERT(descriptor != NULL);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (operand->role != LOOM_LOW_OPERAND_ROLE_RESOURCE ||
        !iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      continue;
    }
    for (uint16_t j = 0; j < operand->reg_class_alt_count; ++j) {
      const uint32_t alt_index = operand->reg_class_alt_start + j;
      if (alt_index >= descriptor_set->reg_class_alt_count) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU implicit-resource descriptor has an invalid register-class "
            "alternative span");
      }
      const loom_low_reg_class_alt_t* alt =
          &descriptor_set->reg_class_alts[alt_index];
      if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
        continue;
      }
      if (alt->reg_class_id >= descriptor_set->reg_class_count) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "AMDGPU implicit-resource descriptor references an invalid "
            "register class");
      }
      return loom_amdgpu_make_register_type(context, alt->reg_class_id,
                                            operand->unit_count, out_type);
    }
  }
  return iree_make_status(
      IREE_STATUS_INTERNAL,
      "AMDGPU descriptor has no explicit implicit resource operand");
}

iree_status_t loom_amdgpu_low_type_register_class_is(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    bool* out_match) {
  *out_match = false;
  if (!loom_low_type_is_register(type)) {
    return iree_ok_status();
  }
  *out_match = loom_low_register_type_descriptor_set_stable_id(type) ==
                   loom_low_lower_context_descriptor_set(context)->stable_id &&
               loom_low_register_type_class_id(type) == reg_class_id;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_type_is_register_class_count(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    uint32_t register_unit_count, bool* out_match) {
  *out_match = false;
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_unit_count(type) != register_unit_count) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_type_register_class_is(context, type, reg_class_id,
                                                out_match);
}

iree_status_t loom_amdgpu_low_value_is_register_class_count(
    loom_low_lower_context_t* context, loom_value_id_t low_value,
    uint16_t reg_class_id, uint32_t register_unit_count, bool* out_match) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  return loom_amdgpu_low_type_is_register_class_count(
      context, loom_module_value_type(module, low_value), reg_class_id,
      register_unit_count, out_match);
}

static bool loom_amdgpu_source_memory_root_is_read_only(
    const loom_low_source_memory_access_plan_t* plan,
    const loom_view_region_table_t* view_regions) {
  if (view_regions == NULL ||
      plan->alias_scope_id == LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    return false;
  }
  const loom_view_access_flags_t access_flags =
      loom_view_region_table_root_access_flags(view_regions,
                                               plan->root_value_id);
  return iree_all_bits_set(access_flags, LOOM_VIEW_ACCESS_READ) &&
         !iree_any_bit_set(access_flags, LOOM_VIEW_ACCESS_WRITE);
}

static bool loom_amdgpu_source_memory_terms_prefer_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_low_source_memory_access_plan_t* plan) {
  for (uint8_t i = 0; i < plan->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term = &plan->dynamic_terms[i];
    switch (term->source) {
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE:
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID:
        break;
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID:
        return true;
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE:
        if (loom_amdgpu_source_value_prefers_vgpr(module, fact_table,
                                                  view_regions, term->index)) {
          return true;
        }
        break;
    }
    for (uint8_t stride_ordinal = 0; stride_ordinal < term->stride_value_count;
         ++stride_ordinal) {
      if (loom_amdgpu_source_value_prefers_vgpr(
              module, fact_table, view_regions,
              term->stride_values[stride_ordinal])) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_amdgpu_source_memory_access_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions, const loom_op_t* source_op,
    loom_type_t source_type) {
  if (fact_table == NULL || view_regions == NULL) {
    return true;
  }
  if (!loom_amdgpu_type_is_32bit_memory_payload(source_type)) {
    return true;
  }
  loom_low_source_memory_access_plan_t plan = {0};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build(module, fact_table, source_op,
                                                &plan, &diagnostic)) {
    return true;
  }
  if (plan.operation_kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD ||
      (plan.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
       plan.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT) ||
      iree_any_bit_set(
          plan.cache_policy.build_flags,
          LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE |
              LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL) ||
      !loom_amdgpu_source_memory_root_is_read_only(&plan, view_regions) ||
      loom_amdgpu_source_memory_terms_prefer_vgpr(module, fact_table,
                                                  view_regions, &plan)) {
    return true;
  }
  return false;
}

static bool loom_amdgpu_vector_extract_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions, const loom_op_t* source_op) {
  const loom_attribute_t static_indices =
      loom_vector_extract_static_indices(source_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] == INT64_MIN) {
    return true;
  }
  return loom_amdgpu_source_value_prefers_vgpr(
      module, fact_table, view_regions, loom_vector_extract_source(source_op));
}

static bool loom_amdgpu_source_value_naturally_prefers_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id) {
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  loom_amdgpu_vector_storage_t vector_storage = {0};
  return loom_amdgpu_type_is_i8(source_type) ||
         loom_amdgpu_type_is_i16(source_type) ||
         loom_amdgpu_type_is_f32(source_type) ||
         loom_amdgpu_type_is_f64(source_type) ||
         loom_amdgpu_type_is_16bit_float(source_type) ||
         (loom_amdgpu_type_vector_storage(source_type, &vector_storage) &&
          vector_storage.kind != LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK);
}

static bool loom_amdgpu_source_value_facts_prefer_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  if (fact_table == NULL || source_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  if (loom_amdgpu_type_is_i1(source_type)) {
    return false;
  }
  return loom_value_facts_is_lane_varying(
      loom_value_fact_table_lookup(fact_table, source_value_id));
}

static bool loom_amdgpu_source_value_known_distribution_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id, loom_value_facts_t* out_facts) {
  if (out_facts != NULL) {
    *out_facts = loom_value_facts_unknown();
  }
  if (fact_table == NULL || source_value_id >= module->values.count ||
      !loom_value_fact_table_has_entry(fact_table, source_value_id)) {
    return false;
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, source_value_id);
  if ((facts.flags & LOOM_VALUE_FACT_DISTRIBUTION_MASK) == 0) {
    return false;
  }
  if (out_facts != NULL) {
    *out_facts = facts;
  }
  return true;
}

static bool loom_amdgpu_distribution_transfer_result_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id, const loom_op_t* defining_op) {
  loom_value_facts_t distribution_facts = loom_value_facts_unknown();
  if (loom_amdgpu_source_value_known_distribution_facts(
          module, fact_table, source_value_id, &distribution_facts)) {
    return loom_value_facts_is_lane_varying(distribution_facts);
  }

  const loom_value_id_t* operands = loom_op_const_operands(defining_op);
  for (uint16_t i = 0; i < defining_op->operand_count; ++i) {
    const loom_value_id_t operand = operands[i];
    if (loom_amdgpu_source_value_is_native_i1_mask(module, fact_table,
                                                   view_regions, operand) ||
        loom_amdgpu_source_value_prefers_vgpr(module, fact_table, view_regions,
                                              operand)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_scalar_conversion_result_requires_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id,
    const loom_op_t* defining_op) {
  if (loom_value_def_index(loom_module_value(module, source_value_id)) != 0) {
    return false;
  }
  const loom_type_t value_type =
      loom_module_value_type(module, source_value_id);
  switch (defining_op->kind) {
    case LOOM_OP_SCALAR_FPTOSI:
    case LOOM_OP_SCALAR_FPTOUI:
      return loom_amdgpu_type_is_i8(value_type) ||
             loom_amdgpu_type_is_i16(value_type) ||
             loom_amdgpu_type_is_i32(value_type);
    default:
      return false;
  }
}

static bool loom_amdgpu_scalar_conversion_result_follows_operand_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id,
    const loom_op_t* defining_op, loom_value_id_t* out_operand) {
  *out_operand = LOOM_VALUE_ID_INVALID;
  if (loom_value_def_index(loom_module_value(module, source_value_id)) != 0 ||
      defining_op->operand_count != 1) {
    return false;
  }
  switch (defining_op->kind) {
    case LOOM_OP_SCALAR_EXTSI:
    case LOOM_OP_SCALAR_EXTUI:
      *out_operand = loom_op_const_operands(defining_op)[0];
      return *out_operand != source_value_id;
    default:
      return false;
  }
}

static bool loom_amdgpu_source_value_facts_are_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  if (fact_table == NULL || source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  return loom_value_facts_is_lane_predicate(
      loom_value_fact_table_lookup(fact_table, source_value_id));
}

static bool loom_amdgpu_source_value_facts_are_uniform_i1(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  if (fact_table == NULL || source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  return loom_value_facts_is_uniform(
      loom_value_fact_table_lookup(fact_table, source_value_id));
}

static bool loom_amdgpu_source_value_facts_are_subgroup_lane_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id, loom_value_facts_t* out_facts) {
  if (out_facts != NULL) {
    *out_facts = loom_value_facts_unknown();
  }
  if (fact_table == NULL || source_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  if (!loom_amdgpu_type_is_i32(source_type) &&
      !loom_amdgpu_type_is_i64(source_type)) {
    return false;
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, source_value_id);
  if (!loom_value_facts_is_subgroup_lane_mask(facts)) {
    return false;
  }
  if (out_facts != NULL) {
    *out_facts = facts;
  }
  return true;
}

static bool loom_amdgpu_block_branches_to_with_arg(
    const loom_block_t* source_block, const loom_block_t* target_block,
    uint16_t target_arg_index, loom_value_id_t* out_arg) {
  *out_arg = LOOM_VALUE_ID_INVALID;
  if (source_block == NULL || source_block->op_count == 0) {
    return false;
  }
  const loom_op_t* terminator = loom_block_const_last_op(source_block);
  if (!loom_cfg_br_isa(terminator) ||
      loom_cfg_br_dest(terminator) != target_block) {
    return false;
  }
  const loom_value_slice_t args = loom_cfg_br_args(terminator);
  if (target_arg_index >= args.count) {
    return false;
  }
  *out_arg = args.values[target_arg_index];
  return true;
}

static bool loom_amdgpu_source_value_directly_prefers_vgpr(
    const loom_module_t* module, loom_value_id_t source_value_id,
    loom_value_id_t excluded_value_id) {
  while (true) {
    if (source_value_id == excluded_value_id ||
        source_value_id >= module->values.count) {
      return false;
    }
    if (loom_amdgpu_source_value_naturally_prefers_vgpr(module,
                                                        source_value_id)) {
      return true;
    }
    const loom_value_t* value = loom_module_value(module, source_value_id);
    if (loom_value_is_block_arg(value)) {
      return false;
    }
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) {
      return false;
    }

    if (loom_traits_are_fact_identity(
            loom_op_effective_traits(module, defining_op))) {
      const uint16_t result_index = loom_value_def_index(value);
      if (result_index >= defining_op->operand_count) {
        return false;
      }
      const loom_value_id_t next_value_id =
          loom_op_const_operands(defining_op)[result_index];
      if (next_value_id == source_value_id) {
        return false;
      }
      source_value_id = next_value_id;
      continue;
    }

    switch (defining_op->kind) {
      case LOOM_OP_KERNEL_WORKITEM_ID:
      case LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID:
        return true;
      case LOOM_OP_KERNEL_SUBGROUP_ID:
      case LOOM_OP_KERNEL_SUBGROUP_LANE_ID:
        return true;
      case LOOM_OP_KERNEL_SUBGROUP_BROADCAST:
      case LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST:
      case LOOM_OP_KERNEL_SUBGROUP_REDUCE:
      case LOOM_OP_KERNEL_SUBGROUP_SCAN:
      case LOOM_OP_KERNEL_SUBGROUP_SHUFFLE:
        return loom_value_def_index(value) == 0;
      case LOOM_OP_INDEX_CAST: {
        const loom_value_id_t next_value_id =
            loom_index_cast_input(defining_op);
        if (next_value_id == source_value_id) {
          return false;
        }
        source_value_id = next_value_id;
        continue;
      }
      case LOOM_OP_VECTOR_EXTRACT:
        return loom_amdgpu_source_value_naturally_prefers_vgpr(
            module, loom_vector_extract_source(defining_op));
      case LOOM_OP_VECTOR_REDUCE:
        return loom_amdgpu_type_is_vector_32bit_register_range(
            loom_module_value_type(module,
                                   loom_vector_reduce_input(defining_op)));
      default: {
        if (loom_amdgpu_scalar_conversion_result_requires_vgpr(
                module, source_value_id, defining_op)) {
          return true;
        }
        loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
        return loom_amdgpu_scalar_conversion_result_follows_operand_vgpr(
                   module, source_value_id, defining_op, &operand) &&
               loom_amdgpu_source_value_directly_prefers_vgpr(module, operand,
                                                              source_value_id);
      }
    }
  }
}

static bool loom_amdgpu_scf_select_payload_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id, loom_value_id_t condition_value_id) {
  if (source_value_id == condition_value_id ||
      source_value_id >= module->values.count) {
    return false;
  }
  return loom_amdgpu_source_value_facts_prefer_vgpr(module, fact_table,
                                                    source_value_id) ||
         loom_amdgpu_source_value_directly_prefers_vgpr(module, source_value_id,
                                                        condition_value_id);
}

static bool loom_amdgpu_scalar_cmpi_i64_requires_native_mask(
    const loom_module_t* module, const loom_op_t* source_op) {
  if (!loom_scalar_cmpi_isa(source_op)) {
    return false;
  }
  const loom_value_id_t lhs = loom_scalar_cmpi_lhs(source_op);
  const loom_value_id_t rhs = loom_scalar_cmpi_rhs(source_op);
  if (lhs >= module->values.count || rhs >= module->values.count ||
      !loom_amdgpu_type_is_i64(loom_module_value_type(module, lhs)) ||
      !loom_amdgpu_type_is_i64(loom_module_value_type(module, rhs))) {
    return false;
  }
  switch (loom_scalar_cmpi_predicate(source_op)) {
    case LOOM_SCALAR_CMPI_PREDICATE_EQ:
    case LOOM_SCALAR_CMPI_PREDICATE_NE:
      return false;
    default:
      return true;
  }
}

static bool loom_amdgpu_source_value_is_direct_native_i1_mask_except(
    const loom_module_t* module, loom_value_id_t source_value_id,
    loom_value_id_t excluded_value_id) {
  if (source_value_id == excluded_value_id ||
      source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return false;
  }

  if (loom_scalar_cmpi_isa(defining_op) && loom_value_def_index(value) == 0) {
    return loom_amdgpu_scalar_cmpi_i64_requires_native_mask(module,
                                                            defining_op) ||
           loom_amdgpu_source_value_directly_prefers_vgpr(
               module, loom_scalar_cmpi_lhs(defining_op), excluded_value_id) ||
           loom_amdgpu_source_value_directly_prefers_vgpr(
               module, loom_scalar_cmpi_rhs(defining_op), excluded_value_id);
  }

  if (loom_scalar_cmpf_isa(defining_op) && loom_value_def_index(value) == 0) {
    return loom_amdgpu_source_value_directly_prefers_vgpr(
               module, loom_scalar_cmpf_lhs(defining_op), excluded_value_id) ||
           loom_amdgpu_source_value_directly_prefers_vgpr(
               module, loom_scalar_cmpf_rhs(defining_op), excluded_value_id);
  }

  if (loom_index_cmp_isa(defining_op) && loom_value_def_index(value) == 0) {
    return loom_amdgpu_source_value_directly_prefers_vgpr(
               module, loom_index_cmp_lhs(defining_op), excluded_value_id) ||
           loom_amdgpu_source_value_directly_prefers_vgpr(
               module, loom_index_cmp_rhs(defining_op), excluded_value_id);
  }

  return loom_kernel_subgroup_shuffle_isa(defining_op) &&
         loom_value_def_index(value) == 1;
}

static bool loom_amdgpu_block_arg_merges_native_mask_diamond(
    const loom_module_t* module, const loom_block_t* block,
    uint16_t arg_index) {
  const loom_region_t* region = block->parent_region;
  if (region == NULL || block == loom_region_const_entry_block(region)) {
    return false;
  }

  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* guard = loom_region_const_block(region, block_index);
    if (guard == NULL || guard->op_count == 0) {
      continue;
    }
    const loom_op_t* terminator = loom_block_const_last_op(guard);
    if (!loom_cfg_cond_br_isa(terminator) ||
        !loom_amdgpu_source_value_is_direct_native_i1_mask_except(
            module, loom_cfg_cond_br_condition(terminator),
            LOOM_VALUE_ID_INVALID)) {
      continue;
    }

    loom_value_id_t true_arg = LOOM_VALUE_ID_INVALID;
    loom_value_id_t false_arg = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_block_branches_to_with_arg(
            loom_cfg_cond_br_true_dest(terminator), block, arg_index,
            &true_arg) &&
        loom_amdgpu_block_branches_to_with_arg(
            loom_cfg_cond_br_false_dest(terminator), block, arg_index,
            &false_arg)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_value_feeds_native_mask_merge_arg(
    const loom_module_t* module, const loom_value_t* value,
    loom_value_id_t value_id) {
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    const uint16_t operand_index = loom_use_operand_index(uses[i]);
    if (!loom_cfg_br_isa(user_op)) {
      continue;
    }
    loom_block_t* dest = loom_cfg_br_dest(user_op);
    if (operand_index >= dest->arg_count) {
      continue;
    }

    const loom_region_t* region = dest->parent_region;
    bool feeds_native_mask_merge = false;
    for (uint16_t block_index = 0;
         region != NULL && block_index < region->block_count; ++block_index) {
      const loom_block_t* guard = loom_region_const_block(region, block_index);
      if (guard == NULL || guard->op_count == 0) {
        continue;
      }
      const loom_op_t* terminator = loom_block_const_last_op(guard);
      if (!loom_cfg_cond_br_isa(terminator)) {
        continue;
      }
      loom_value_id_t true_arg = LOOM_VALUE_ID_INVALID;
      loom_value_id_t false_arg = LOOM_VALUE_ID_INVALID;
      if (loom_amdgpu_block_branches_to_with_arg(
              loom_cfg_cond_br_true_dest(terminator), dest, operand_index,
              &true_arg) &&
          loom_amdgpu_block_branches_to_with_arg(
              loom_cfg_cond_br_false_dest(terminator), dest, operand_index,
              &false_arg) &&
          loom_amdgpu_source_value_is_direct_native_i1_mask_except(
              module, loom_cfg_cond_br_condition(terminator), value_id)) {
        feeds_native_mask_merge = true;
        break;
      }
    }
    if (feeds_native_mask_merge) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_block_arg_has_cfg_predecessor(const loom_block_t* block,
                                                      uint16_t arg_index) {
  const loom_region_t* region = block != NULL ? block->parent_region : NULL;
  if (region == NULL || arg_index >= block->arg_count) {
    return false;
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* predecessor =
        loom_region_const_block(region, block_index);
    if (predecessor == NULL || predecessor->op_count == 0) {
      continue;
    }
    const loom_op_t* terminator = loom_block_const_last_op(predecessor);
    if (!loom_cfg_br_isa(terminator) || loom_cfg_br_dest(terminator) != block) {
      continue;
    }
    if (arg_index < loom_cfg_br_args(terminator).count) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_scf_select_use_needs_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  const loom_value_t* value = loom_module_value(module, source_value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    if (loom_use_operand_index(*use) != 0) {
      continue;
    }
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (!loom_scf_select_isa(user_op)) {
      continue;
    }
    if (loom_amdgpu_scf_select_payload_prefers_vgpr(
            module, fact_table, loom_scf_select_result(user_op),
            source_value_id) ||
        loom_amdgpu_scf_select_payload_prefers_vgpr(
            module, fact_table, loom_scf_select_true_value(user_op),
            source_value_id) ||
        loom_amdgpu_scf_select_payload_prefers_vgpr(
            module, fact_table, loom_scf_select_false_value(user_op),
            source_value_id)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_vector_constructor_use_needs_native_i1_mask(
    const loom_module_t* module, loom_value_id_t source_value_id) {
  const loom_value_t* value = loom_module_value(module, source_value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (loom_vector_splat_isa(user_op) && loom_use_operand_index(*use) == 0 &&
        loom_amdgpu_vector_i1_lane_count(loom_module_value_type(
            module, loom_vector_splat_result(user_op))) != 0) {
      return true;
    }
    if (loom_vector_from_elements_isa(user_op) &&
        loom_amdgpu_vector_i1_lane_count(loom_module_value_type(
            module, loom_vector_from_elements_result(user_op))) != 0) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_source_i1_value_has_cross_block_use(
    const loom_module_t* module, loom_value_id_t source_value_id) {
  if (source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || defining_op->parent_block == NULL) {
    return false;
  }

  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (user_op != NULL && user_op->parent_block != defining_op->parent_block) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_source_value_is_durable_i1_bool(
    const loom_module_t* module, loom_value_id_t source_value_id) {
  if (!loom_amdgpu_source_i1_value_has_cross_block_use(module,
                                                       source_value_id)) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  const loom_op_t* defining_op = loom_value_def_op(value);
  return defining_op != NULL && loom_value_def_index(value) == 0 &&
         loom_scalar_cmpi_isa(defining_op) &&
         loom_amdgpu_type_is_i32(
             loom_module_value_type(module, loom_scalar_cmpi_lhs(defining_op)));
}

static bool loom_amdgpu_source_value_can_lower_as_scc_i1(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id, loom_value_id_t excluded_value_id) {
  if (source_value_id == excluded_value_id ||
      source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  const loom_value_id_t next_excluded_value_id =
      excluded_value_id == LOOM_VALUE_ID_INVALID ? source_value_id
                                                 : excluded_value_id;
  if (loom_amdgpu_source_value_facts_are_native_i1_mask(module, fact_table,
                                                        source_value_id)) {
    return false;
  }
  if (loom_amdgpu_source_value_facts_are_uniform_i1(module, fact_table,
                                                    source_value_id)) {
    return true;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    const loom_block_t* block = loom_value_def_block(value);
    return !loom_amdgpu_block_arg_has_cfg_predecessor(
        block, loom_value_def_index(value));
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL) {
    return true;
  }
  if (loom_value_def_index(value) != 0) {
    return false;
  }

  if (loom_traits_are_fact_identity(
          loom_op_effective_traits(module, defining_op))) {
    if (defining_op->operand_count == 0) {
      return false;
    }
    const loom_value_id_t source_identity_value_id =
        loom_op_const_operands(defining_op)[0];
    return source_identity_value_id != source_value_id &&
           loom_amdgpu_source_value_can_lower_as_scc_i1(
               module, fact_table, view_regions, source_identity_value_id,
               next_excluded_value_id);
  }

  if (loom_scalar_constant_isa(defining_op)) {
    return true;
  }

  if (loom_scalar_cmpi_isa(defining_op)) {
    return !loom_amdgpu_scalar_cmpi_i64_requires_native_mask(module,
                                                             defining_op) &&
           !loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpi_lhs(defining_op)) &&
           !loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpi_rhs(defining_op));
  }

  if (loom_scalar_cmpf_isa(defining_op)) {
    return !loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpf_lhs(defining_op)) &&
           !loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpf_rhs(defining_op));
  }

  if (loom_index_cmp_isa(defining_op)) {
    return !loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_index_cmp_lhs(defining_op)) &&
           !loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_index_cmp_rhs(defining_op));
  }

  if (loom_scalar_andi_isa(defining_op)) {
    return loom_amdgpu_source_value_can_lower_as_scc_i1(
               module, fact_table, view_regions,
               loom_scalar_andi_lhs(defining_op), next_excluded_value_id) &&
           loom_amdgpu_source_value_can_lower_as_scc_i1(
               module, fact_table, view_regions,
               loom_scalar_andi_rhs(defining_op), next_excluded_value_id);
  }

  if (loom_scalar_ori_isa(defining_op)) {
    return loom_amdgpu_source_value_can_lower_as_scc_i1(
               module, fact_table, view_regions,
               loom_scalar_ori_lhs(defining_op), next_excluded_value_id) &&
           loom_amdgpu_source_value_can_lower_as_scc_i1(
               module, fact_table, view_regions,
               loom_scalar_ori_rhs(defining_op), next_excluded_value_id);
  }

  if (loom_scalar_xori_isa(defining_op)) {
    return loom_amdgpu_source_value_can_lower_as_scc_i1(
               module, fact_table, view_regions,
               loom_scalar_xori_lhs(defining_op), next_excluded_value_id) &&
           loom_amdgpu_source_value_can_lower_as_scc_i1(
               module, fact_table, view_regions,
               loom_scalar_xori_rhs(defining_op), next_excluded_value_id);
  }

  return false;
}

static bool loom_amdgpu_source_value_is_native_i1_mask_excluding(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id, loom_value_id_t excluded_value_id) {
  if (source_value_id == excluded_value_id ||
      source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  const loom_value_id_t next_excluded_value_id =
      excluded_value_id == LOOM_VALUE_ID_INVALID ? source_value_id
                                                 : excluded_value_id;
  if (loom_amdgpu_source_value_facts_are_native_i1_mask(module, fact_table,
                                                        source_value_id)) {
    return true;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_amdgpu_scf_select_use_needs_native_i1_mask(module, fact_table,
                                                      source_value_id)) {
    return true;
  }
  if (loom_amdgpu_vector_constructor_use_needs_native_i1_mask(
          module, source_value_id)) {
    return true;
  }
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    if (loom_amdgpu_source_value_can_lower_as_scc_i1(
            module, fact_table, view_regions, source_value_id,
            next_excluded_value_id)) {
      continue;
    }
    const loom_op_t* user_op = loom_use_user_op(*use);
    const uint16_t operand_index = loom_use_operand_index(*use);
    if (!loom_cfg_br_isa(user_op)) {
      continue;
    }
    const loom_block_t* dest = loom_cfg_br_dest(user_op);
    if (operand_index < dest->arg_count) {
      const loom_value_id_t dest_arg = loom_block_arg_id(dest, operand_index);
      if (dest_arg != source_value_id &&
          loom_amdgpu_source_value_is_native_i1_mask_excluding(
              module, fact_table, view_regions, dest_arg,
              next_excluded_value_id)) {
        return true;
      }
    }
  }
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return false;
  }

  if (loom_traits_are_fact_identity(
          loom_op_effective_traits(module, defining_op))) {
    const uint16_t result_index = loom_value_def_index(value);
    if (result_index >= defining_op->operand_count) {
      return false;
    }
    const loom_value_id_t source_identity_value_id =
        loom_op_const_operands(defining_op)[result_index];
    return source_identity_value_id != source_value_id &&
           loom_amdgpu_source_value_is_native_i1_mask_excluding(
               module, fact_table, view_regions, source_identity_value_id,
               next_excluded_value_id);
  }

  if (loom_scalar_cmpi_isa(defining_op) && loom_value_def_index(value) == 0) {
    return loom_amdgpu_scalar_cmpi_i64_requires_native_mask(module,
                                                            defining_op) ||
           loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpi_lhs(defining_op)) ||
           loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpi_rhs(defining_op));
  }

  if (loom_scalar_cmpf_isa(defining_op) && loom_value_def_index(value) == 0) {
    return loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpf_lhs(defining_op)) ||
           loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpf_rhs(defining_op));
  }

  if (loom_index_cmp_isa(defining_op) && loom_value_def_index(value) == 0) {
    return loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_index_cmp_lhs(defining_op)) ||
           loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_index_cmp_rhs(defining_op));
  }

  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  if (loom_scalar_andi_isa(defining_op) && loom_value_def_index(value) == 0) {
    lhs = loom_scalar_andi_lhs(defining_op);
    rhs = loom_scalar_andi_rhs(defining_op);
  } else if (loom_scalar_ori_isa(defining_op) &&
             loom_value_def_index(value) == 0) {
    lhs = loom_scalar_ori_lhs(defining_op);
    rhs = loom_scalar_ori_rhs(defining_op);
  } else if (loom_scalar_xori_isa(defining_op) &&
             loom_value_def_index(value) == 0) {
    lhs = loom_scalar_xori_lhs(defining_op);
    rhs = loom_scalar_xori_rhs(defining_op);
  }
  if (lhs != LOOM_VALUE_ID_INVALID) {
    const bool lhs_is_mask =
        loom_amdgpu_source_value_is_native_i1_mask_excluding(
            module, fact_table, view_regions, lhs, next_excluded_value_id);
    const bool rhs_is_mask =
        loom_amdgpu_source_value_is_native_i1_mask_excluding(
            module, fact_table, view_regions, rhs, next_excluded_value_id);
    if (!lhs_is_mask && !rhs_is_mask) {
      return false;
    }
    return (lhs_is_mask || loom_amdgpu_source_value_can_lower_as_scc_i1(
                               module, fact_table, view_regions, lhs,
                               next_excluded_value_id)) &&
           (rhs_is_mask ||
            loom_amdgpu_source_value_can_lower_as_scc_i1(
                module, fact_table, view_regions, rhs, next_excluded_value_id));
  }

  return loom_kernel_subgroup_shuffle_isa(defining_op) &&
         loom_value_def_index(value) == 1;
}

bool loom_amdgpu_source_value_is_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id) {
  return loom_amdgpu_source_value_is_native_i1_mask_excluding(
      module, fact_table, view_regions, source_value_id, LOOM_VALUE_ID_INVALID);
}

bool loom_amdgpu_source_value_is_uniform_subgroup_lane_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  return loom_amdgpu_source_value_facts_are_subgroup_lane_mask(
             module, fact_table, source_value_id, &facts) &&
         loom_value_facts_is_uniform(facts);
}

bool loom_amdgpu_source_value_is_divergent_subgroup_lane_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  return loom_amdgpu_source_value_facts_are_subgroup_lane_mask(
             module, fact_table, source_value_id, &facts) &&
         loom_value_facts_is_lane_varying(facts);
}

static bool loom_amdgpu_source_value_store_use_requires_vgpr(
    loom_value_id_t source_value_id, const loom_op_t* user_op) {
  if (loom_vector_store_isa(user_op)) {
    return loom_vector_store_value(user_op) == source_value_id;
  }
  if (loom_view_store_isa(user_op)) {
    return loom_view_store_value(user_op) == source_value_id;
  }
  return false;
}

static bool loom_amdgpu_source_value_scf_select_payload_use_requires_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id, const loom_op_t* user_op) {
  if (!loom_scf_select_isa(user_op) ||
      (loom_scf_select_true_value(user_op) != source_value_id &&
       loom_scf_select_false_value(user_op) != source_value_id)) {
    return false;
  }
  return loom_amdgpu_source_value_is_native_i1_mask_excluding(
      module, fact_table, view_regions, loom_scf_select_condition(user_op),
      source_value_id);
}

static bool loom_amdgpu_scf_select_result_requires_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id, const loom_op_t* defining_op) {
  if (!loom_scf_select_isa(defining_op) ||
      loom_scf_select_result(defining_op) != source_value_id) {
    return false;
  }
  const loom_value_id_t condition = loom_scf_select_condition(defining_op);
  return loom_amdgpu_source_value_is_native_i1_mask_excluding(
      module, fact_table, view_regions, condition, source_value_id);
}

static bool loom_amdgpu_index_arithmetic_result_follows_operand_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id, const loom_op_t* defining_op) {
  if (loom_value_def_index(loom_module_value(module, source_value_id)) != 0) {
    return false;
  }

  uint16_t operand_count = 0;
  switch (defining_op->kind) {
    case LOOM_OP_INDEX_ADD:
    case LOOM_OP_INDEX_SUB:
    case LOOM_OP_INDEX_MUL:
    case LOOM_OP_INDEX_SCALE:
    case LOOM_OP_INDEX_DIV:
    case LOOM_OP_INDEX_REM:
    case LOOM_OP_INDEX_MIN:
    case LOOM_OP_INDEX_MAX:
    case LOOM_OP_INDEX_ANDI:
    case LOOM_OP_INDEX_ORI:
    case LOOM_OP_INDEX_XORI:
    case LOOM_OP_INDEX_SHLI:
    case LOOM_OP_INDEX_SHRSI:
    case LOOM_OP_INDEX_SHRUI:
    case LOOM_OP_INDEX_ROTLI:
    case LOOM_OP_INDEX_ROTRI:
      operand_count = 2;
      break;
    case LOOM_OP_INDEX_MADD:
      operand_count = 3;
      break;
    default:
      return false;
  }
  if (operand_count > defining_op->operand_count) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(defining_op);
  for (uint16_t i = 0; i < operand_count; ++i) {
    if (operands[i] != source_value_id &&
        loom_amdgpu_source_value_prefers_vgpr(module, fact_table, view_regions,
                                              operands[i])) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_source_value_has_vgpr_payload_use(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id) {
  const loom_value_t* value = loom_module_value(module, source_value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (loom_amdgpu_source_value_store_use_requires_vgpr(source_value_id,
                                                         user_op) ||
        loom_amdgpu_source_value_scf_select_payload_use_requires_vgpr(
            module, fact_table, view_regions, source_value_id, user_op)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_scalar_i32_to_i64_conversion_consumers_require_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id, const loom_op_t* defining_op) {
  loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_scalar_conversion_result_follows_operand_vgpr(
          module, source_value_id, defining_op, &operand) ||
      !loom_amdgpu_type_is_i32(loom_module_value_type(module, operand)) ||
      !loom_amdgpu_type_is_i64(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }
  return loom_amdgpu_source_value_has_vgpr_payload_use(
      module, fact_table, view_regions, source_value_id);
}

bool loom_amdgpu_source_value_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id) {
  if (source_value_id >= module->values.count) {
    return false;
  }
  if (loom_amdgpu_source_value_facts_prefer_vgpr(module, fact_table,
                                                 source_value_id)) {
    return true;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    if (loom_amdgpu_source_value_naturally_prefers_vgpr(module,
                                                        source_value_id)) {
      return true;
    }
    const loom_block_t* block = loom_value_def_block(value);
    const uint16_t arg_index = loom_value_def_index(value);
    return loom_amdgpu_block_arg_merges_native_mask_diamond(module, block,
                                                            arg_index);
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL) {
    return loom_amdgpu_source_value_naturally_prefers_vgpr(module,
                                                           source_value_id);
  }
  const loom_trait_flags_t defining_op_traits =
      loom_op_effective_traits(module, defining_op);

  if (loom_amdgpu_value_feeds_native_mask_merge_arg(module, value,
                                                    source_value_id)) {
    return true;
  }

  if (loom_amdgpu_scalar_i32_to_i64_conversion_consumers_require_vgpr(
          module, fact_table, view_regions, source_value_id, defining_op)) {
    return true;
  }

  if (loom_amdgpu_scf_select_result_requires_vgpr(
          module, fact_table, view_regions, source_value_id, defining_op)) {
    return true;
  }

  if (loom_amdgpu_index_arithmetic_result_follows_operand_vgpr(
          module, fact_table, view_regions, source_value_id, defining_op)) {
    return true;
  }

  if (loom_scalar_bitcast_isa(defining_op)) {
    const loom_value_id_t input_value_id =
        loom_scalar_bitcast_input(defining_op);
    return loom_amdgpu_source_value_prefers_vgpr(module, fact_table,
                                                 view_regions, input_value_id);
  }

  loom_type_t source_type = loom_module_value_type(module, source_value_id);
  if (loom_amdgpu_type_is_f64(source_type) ||
      loom_amdgpu_type_is_16bit_float(source_type)) {
    return true;
  }

  if (loom_value_def_index(value) == 0 &&
      loom_traits_have_distribution_transfer(defining_op_traits)) {
    return loom_amdgpu_distribution_transfer_result_prefers_vgpr(
        module, fact_table, view_regions, source_value_id, defining_op);
  }

  if (loom_traits_are_fact_identity(defining_op_traits)) {
    const uint16_t result_index = loom_value_def_index(value);
    if (result_index >= defining_op->operand_count) {
      return false;
    }
    const loom_value_id_t source_identity_value_id =
        loom_op_const_operands(defining_op)[result_index];
    if (source_identity_value_id == source_value_id) {
      return false;
    }
    return loom_amdgpu_source_value_prefers_vgpr(
        module, fact_table, view_regions, source_identity_value_id);
  }

  switch (defining_op->kind) {
    case LOOM_OP_KERNEL_WORKITEM_ID:
      return loom_kernel_workitem_id_dimension(defining_op) <
             LOOM_KERNEL_DIMENSION_COUNT_;
    case LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID:
      return loom_kernel_workitem_dispatch_id_dimension(defining_op) <
             LOOM_KERNEL_DIMENSION_COUNT_;
    case LOOM_OP_KERNEL_SUBGROUP_ID:
    case LOOM_OP_KERNEL_SUBGROUP_LANE_ID:
      return true;
    case LOOM_OP_KERNEL_SUBGROUP_BROADCAST:
    case LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST:
    case LOOM_OP_KERNEL_SUBGROUP_REDUCE:
    case LOOM_OP_KERNEL_SUBGROUP_SCAN:
      return loom_value_def_index(value) == 0;
    case LOOM_OP_KERNEL_SUBGROUP_SHUFFLE:
      return loom_value_def_index(value) == 0;
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
    case LOOM_OP_VECTOR_SELECT:
    case LOOM_OP_VECTOR_SPLAT: {
      loom_amdgpu_vector_storage_t storage = {0};
      return loom_value_def_index(value) == 0 &&
             loom_amdgpu_type_vector_storage(source_type, &storage) &&
             storage.kind != LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK;
    }
    case LOOM_OP_VECTOR_EXTRACT:
      return loom_amdgpu_vector_extract_prefers_vgpr(module, fact_table,
                                                     view_regions, defining_op);
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VIEW_LOAD:
      return loom_amdgpu_source_memory_access_prefers_vgpr(
          module, fact_table, view_regions, defining_op, source_type);
    case LOOM_OP_VECTOR_REDUCE:
      return loom_amdgpu_type_is_vector_32bit_register_range(
          loom_module_value_type(module,
                                 loom_vector_reduce_input(defining_op)));
    case LOOM_OP_VIEW_ATOMIC_CMPXCHG:
    case LOOM_OP_VIEW_ATOMIC_RMW:
      return true;
    default: {
      if (loom_amdgpu_scalar_conversion_result_requires_vgpr(
              module, source_value_id, defining_op) ||
          loom_amdgpu_type_is_f32(source_type) ||
          loom_amdgpu_type_is_vector_32bit_register_range(source_type)) {
        return true;
      }
      loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
      return loom_amdgpu_scalar_conversion_result_follows_operand_vgpr(
                 module, source_value_id, defining_op, &operand) &&
             loom_amdgpu_source_value_prefers_vgpr(module, fact_table,
                                                   view_regions, operand);
    }
  }
}

iree_status_t loom_amdgpu_context_value_prefers_vgpr(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_prefers_vgpr) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  *out_prefers_vgpr = loom_amdgpu_source_value_prefers_vgpr(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context), view_regions,
      source_value_id);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_context_value_is_native_i1_mask(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_is_native_mask) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  *out_is_native_mask = loom_amdgpu_source_value_is_native_i1_mask(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context), view_regions,
      source_value_id);
  return iree_ok_status();
}

static bool loom_amdgpu_offset_value_needs_64bit(
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    loom_value_id_t source_value_id, loom_type_t source_type) {
  if (!loom_amdgpu_type_is_offset(source_type)) {
    return false;
  }
  if (fact_table == NULL || module == NULL ||
      source_value_id >= module->values.count) {
    return true;
  }
  return !loom_value_facts_fit_unsigned_bit_count(
      loom_value_fact_table_lookup(fact_table, source_value_id), 32);
}

typedef struct loom_amdgpu_register_shape_t {
  // AMDGPU descriptor-set register class selected for the value.
  uint16_t class_id;
  // Number of 32-bit register units occupied by the value.
  uint32_t unit_count;
} loom_amdgpu_register_shape_t;

static loom_amdgpu_register_shape_t loom_amdgpu_register_shape(
    uint16_t class_id, uint32_t unit_count) {
  return (loom_amdgpu_register_shape_t){
      .class_id = class_id,
      .unit_count = unit_count,
  };
}

typedef enum loom_amdgpu_scalar_value_register_policy_e {
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_NONE = 0,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED = 1,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_I1 = 2,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK = 3,
  LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_OFFSET_WIDTH = 4,
} loom_amdgpu_scalar_value_register_policy_t;

typedef struct loom_amdgpu_scalar_value_register_mapping_t {
  // True when this scalar type has an AMDGPU source-value placement policy.
  bool is_valid;
  // Default register class before value-specific facts adjust placement.
  uint16_t default_class_id;
  // Default number of 32-bit register units before value-specific facts apply.
  uint32_t default_unit_count;
  // Value-sensitive placement policy for this scalar type.
  loom_amdgpu_scalar_value_register_policy_t policy;
} loom_amdgpu_scalar_value_register_mapping_t;

#define LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FIXED(class_id, unit_count) \
  {                                                                   \
      .is_valid = true,                                               \
      .default_class_id = class_id,                                   \
      .default_unit_count = unit_count,                               \
      .policy = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED,       \
  }

#define LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY(class_id, unit_count, \
                                                 selected_policy)      \
  {                                                                    \
      .is_valid = true,                                                \
      .default_class_id = class_id,                                    \
      .default_unit_count = unit_count,                                \
      .policy = selected_policy,                                       \
  }

static const loom_amdgpu_scalar_value_register_mapping_t
    loom_amdgpu_scalar_value_register_mappings[LOOM_SCALAR_TYPE_COUNT_] = {
        [LOOM_SCALAR_TYPE_INDEX] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_OFFSET_WIDTH),
        [LOOM_SCALAR_TYPE_OFFSET] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_OFFSET_WIDTH),
        [LOOM_SCALAR_TYPE_I1] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY(
            LOOM_AMDGPU_REG_CLASS_ID_SCC, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_I1),
        [LOOM_SCALAR_TYPE_I8] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FIXED(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1),
        [LOOM_SCALAR_TYPE_I16] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FIXED(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1),
        [LOOM_SCALAR_TYPE_I32] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK),
        [LOOM_SCALAR_TYPE_I64] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK),
        [LOOM_SCALAR_TYPE_F16] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FIXED(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1),
        [LOOM_SCALAR_TYPE_BF16] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FIXED(
            LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1),
        [LOOM_SCALAR_TYPE_F32] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK),
        [LOOM_SCALAR_TYPE_F64] = LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY(
            LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
            LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK),
};

#undef LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY
#undef LOOM_AMDGPU_SCALAR_VALUE_REGISTER_FIXED

static bool loom_amdgpu_source_scalar_value_register_shape(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id, loom_type_t source_type,
    loom_amdgpu_register_shape_t* out_shape) {
  if (!loom_type_is_scalar(source_type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(source_type);
  if (element_type >= LOOM_SCALAR_TYPE_COUNT_) {
    return false;
  }
  const loom_amdgpu_scalar_value_register_mapping_t* mapping =
      &loom_amdgpu_scalar_value_register_mappings[element_type];
  if (!mapping->is_valid) {
    return false;
  }
  *out_shape = loom_amdgpu_register_shape(mapping->default_class_id,
                                          mapping->default_unit_count);
  switch (mapping->policy) {
    case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED:
      return true;
    case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_I1:
      if (loom_amdgpu_source_value_is_native_i1_mask(
              module, fact_table, view_regions, source_value_id)) {
        *out_shape =
            loom_amdgpu_register_shape(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
      } else if (loom_amdgpu_source_value_is_durable_i1_bool(module,
                                                             source_value_id)) {
        *out_shape =
            loom_amdgpu_register_shape(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
      }
      return true;
    case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK:
      if (loom_amdgpu_source_value_prefers_vgpr(
              module, fact_table, view_regions, source_value_id)) {
        out_shape->class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
      }
      return true;
    case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_OFFSET_WIDTH:
      if (loom_amdgpu_source_value_prefers_vgpr(
              module, fact_table, view_regions, source_value_id)) {
        out_shape->class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
      }
      if (loom_amdgpu_offset_value_needs_64bit(fact_table, module,
                                               source_value_id, source_type)) {
        out_shape->unit_count = 2;
      }
      return true;
    case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_NONE:
      return false;
  }
  return false;
}

static bool loom_amdgpu_source_vector_value_register_shape(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id, loom_type_t source_type,
    loom_amdgpu_register_shape_t* out_shape) {
  loom_amdgpu_vector_storage_t vector_storage = {0};
  if (!loom_amdgpu_type_vector_storage(source_type, &vector_storage)) {
    return false;
  }
  switch (vector_storage.kind) {
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK:
      *out_shape = loom_amdgpu_register_shape(LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                                              vector_storage.register_count);
      return true;
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
      *out_shape = loom_amdgpu_register_shape(
          loom_amdgpu_source_value_prefers_vgpr(module, fact_table,
                                                view_regions, source_value_id)
              ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
              : LOOM_AMDGPU_REG_CLASS_ID_SGPR,
          vector_storage.register_count);
      return true;
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER:
      *out_shape = loom_amdgpu_register_shape(LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                              vector_storage.register_count);
      return true;
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_NONE:
      return false;
  }
  return false;
}

static bool loom_amdgpu_source_value_register_shape_needs_analysis(
    loom_type_t source_type) {
  if (loom_type_is_scalar(source_type)) {
    const loom_scalar_type_t element_type = loom_type_element_type(source_type);
    if (element_type >= LOOM_SCALAR_TYPE_COUNT_) {
      return false;
    }
    const loom_amdgpu_scalar_value_register_mapping_t* mapping =
        &loom_amdgpu_scalar_value_register_mappings[element_type];
    switch (mapping->policy) {
      case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_I1:
      case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_PREFERRED_BANK:
      case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_OFFSET_WIDTH:
        return true;
      case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_NONE:
      case LOOM_AMDGPU_SCALAR_VALUE_REGISTER_POLICY_FIXED:
        return false;
    }
    return false;
  }

  loom_amdgpu_vector_storage_t vector_storage = {0};
  if (!loom_amdgpu_type_vector_storage(source_type, &vector_storage)) {
    return false;
  }
  return vector_storage.kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT ||
         vector_storage.kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT;
}

static bool loom_amdgpu_source_value_register_shape(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id, loom_type_t source_type,
    loom_amdgpu_register_shape_t* out_shape) {
  return loom_amdgpu_source_scalar_value_register_shape(
             module, fact_table, view_regions, source_value_id, source_type,
             out_shape) ||
         loom_amdgpu_source_vector_value_register_shape(
             module, fact_table, view_regions, source_value_id, source_type,
             out_shape);
}

iree_status_t loom_amdgpu_map_type(void* user_data,
                                   loom_low_lower_context_t* context,
                                   const loom_op_t* source_op,
                                   loom_type_t source_type,
                                   loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_amdgpu_type_is_i1(source_type)) {
    return loom_amdgpu_make_scc_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_i32(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_i8(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_i16(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_i64(source_type) ||
      loom_amdgpu_type_is_f64(source_type)) {
    return loom_amdgpu_make_sgpr_range_type(context, 2, out_low_type);
  }
  if (loom_amdgpu_type_is_address_scalar(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_16bit_float(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  loom_amdgpu_vector_storage_t vector_storage = {0};
  if (loom_amdgpu_type_vector_storage(source_type, &vector_storage)) {
    if (vector_storage.kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK) {
      return loom_amdgpu_make_sgpr_range_type(
          context, vector_storage.register_count, out_low_type);
    }
    return loom_amdgpu_make_register_type(
        context, LOOM_AMDGPU_REG_CLASS_ID_VGPR, vector_storage.register_count,
        out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

iree_status_t loom_amdgpu_map_value(void* user_data,
                                    loom_low_lower_context_t* context,
                                    const loom_op_t* source_op,
                                    loom_value_id_t source_value_id,
                                    loom_type_t source_type,
                                    loom_type_t* out_low_type) {
  (void)user_data;
  const loom_view_region_table_t* view_regions = NULL;
  if (loom_amdgpu_source_value_register_shape_needs_analysis(source_type)) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_context_view_regions(context, &view_regions));
  }
  loom_amdgpu_register_shape_t shape = {0};
  if (loom_amdgpu_source_value_register_shape(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), view_regions,
          source_value_id, source_type, &shape)) {
    return loom_amdgpu_make_register_type(context, shape.class_id,
                                          shape.unit_count, out_low_type);
  }
  return loom_amdgpu_map_type(user_data, context, source_op, source_type,
                              out_low_type);
}

static iree_status_t loom_amdgpu_map_contract_register(
    const loom_target_contract_query_environment_t* environment,
    uint16_t descriptor_register_class_id, uint32_t register_unit_count,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  if (descriptor_register_class_id >=
      environment->descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU contract register class %" PRIu16
                            " is outside the selected descriptor set",
                            descriptor_register_class_id);
  }
  *out_mapped_value = loom_low_lower_rule_mapped_value_register(
      descriptor_register_class_id, register_unit_count);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_map_contract_value(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  const loom_type_t source_type =
      loom_module_value_type(environment->module, source_value_id);
  loom_amdgpu_register_shape_t shape = {0};
  if (loom_amdgpu_source_value_register_shape(
          environment->module, environment->fact_table,
          environment->view_regions, source_value_id, source_type, &shape)) {
    return loom_amdgpu_map_contract_register(
        environment, shape.class_id, shape.unit_count, out_mapped_value);
  }
  return iree_ok_status();
}

bool loom_amdgpu_module_value_as_exact_index_constant(
    const loom_module_t* module, loom_value_id_t value_id, int64_t* out_value) {
  *out_value = 0;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_index_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_index_constant_value(defining_op);
  if (attr.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

bool loom_amdgpu_module_value_as_i32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              int64_t* out_value) {
  *out_value = 0;
  if (!loom_amdgpu_type_is_i32(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_scalar_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_scalar_constant_value(defining_op);
  if (!loom_amdgpu_attr_is_i32_immediate(attr)) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

bool loom_amdgpu_module_value_as_f32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              uint32_t* out_bit_pattern) {
  *out_bit_pattern = 0;
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_scalar_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_scalar_constant_value(defining_op);
  if (!loom_amdgpu_attr_is_f32_immediate(attr)) {
    return false;
  }
  *out_bit_pattern = loom_amdgpu_attr_f32_bit_pattern(attr);
  return true;
}

bool loom_amdgpu_value_facts_as_exact_non_negative_i64(loom_value_facts_t facts,
                                                       int64_t* out_value) {
  *out_value = 0;
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts) ||
      facts.range_lo < 0) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

bool loom_amdgpu_value_facts_as_exact_i32(loom_value_facts_t facts,
                                          int64_t* out_value) {
  *out_value = 0;
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &value) || value < INT32_MIN ||
      value > INT32_MAX) {
    return false;
  }
  *out_value = value;
  return true;
}

bool loom_amdgpu_value_facts_as_f32_bit_pattern(loom_value_facts_t facts,
                                                uint32_t* out_bit_pattern) {
  *out_bit_pattern = 0;
  if (!loom_value_facts_is_exact(facts) || !loom_value_facts_is_float(facts)) {
    return false;
  }
  const float f32_value = (float)loom_value_facts_as_f64(facts);
  memcpy(out_bit_pattern, &f32_value, sizeof(*out_bit_pattern));
  return true;
}

static bool loom_amdgpu_i64_value_as_u32_bits(int64_t value,
                                              uint32_t* out_bits) {
  if (value < INT32_MIN || value > UINT32_MAX) {
    return false;
  }
  *out_bits = (uint32_t)value;
  return true;
}

bool loom_amdgpu_value_facts_as_u32_bits(loom_value_facts_t facts,
                                         uint32_t* out_bits) {
  *out_bits = 0;
  if (loom_value_facts_is_exact(facts) && loom_value_facts_is_float(facts)) {
    return loom_amdgpu_value_facts_as_f32_bit_pattern(facts, out_bits);
  }
  int64_t value = 0;
  return loom_value_facts_as_exact_i64(facts, &value) &&
         loom_amdgpu_i64_value_as_u32_bits(value, out_bits);
}

bool loom_amdgpu_source_lane_as_u32_bits(
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    loom_value_id_t source, uint32_t lane, uint32_t* out_bits) {
  *out_bits = 0;
  if (fact_table == NULL || module == NULL) {
    return false;
  }

  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, source);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const bool source_is_integer_scalar =
      loom_type_is_scalar(source_type) &&
      loom_scalar_type_is_integer(loom_type_element_type(source_type));
  const int32_t source_bit_count =
      source_is_integer_scalar
          ? loom_scalar_type_bitwidth(loom_type_element_type(source_type))
          : 0;
  const uint32_t lane_offset = lane * 32u;
  int64_t exact_value = 0;
  if (source_is_integer_scalar && lane_offset < (uint32_t)source_bit_count &&
      loom_value_facts_as_exact_i64(facts, &exact_value)) {
    *out_bits = (uint32_t)((uint64_t)exact_value >> lane_offset);
    return true;
  }
  // A non-negative scalar range contained in 32 bits has zero high lanes.
  if (source_is_integer_scalar && lane_offset < (uint32_t)source_bit_count &&
      facts.range_lo >= 0 && facts.range_hi <= UINT32_MAX && lane > 0) {
    *out_bits = 0;
    return true;
  }

  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                             &uniform)) {
    return loom_amdgpu_value_facts_as_u32_bits(uniform.element, out_bits);
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_value_facts_query_small_static_lanes(&fact_table->context, facts,
                                                &lanes)) {
    return lane < lanes.count &&
           loom_amdgpu_value_facts_as_u32_bits(lanes.lanes[lane], out_bits);
  }

  loom_value_fact_vector_iota_t iota = {0};
  if (loom_value_facts_query_vector_iota(&fact_table->context, facts, &iota)) {
    int64_t base = 0;
    int64_t step = 0;
    int64_t delta = 0;
    int64_t value = 0;
    return loom_value_facts_as_exact_i64(iota.base, &base) &&
           loom_value_facts_as_exact_i64(iota.step, &step) &&
           loom_checked_mul_i64((int64_t)lane, step, &delta) &&
           loom_checked_add_i64(base, delta, &value) &&
           loom_amdgpu_i64_value_as_u32_bits(value, out_bits);
  }

  return loom_amdgpu_value_facts_as_u32_bits(facts, out_bits);
}

bool loom_amdgpu_u32_is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool loom_amdgpu_attr_is_i32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_I64 && value.i64 >= INT32_MIN &&
         value.i64 <= INT32_MAX;
}

bool loom_amdgpu_attr_is_u32_address_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_I64 && value.i64 >= 0 &&
         value.i64 <= UINT32_MAX;
}

bool loom_amdgpu_attr_is_f32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_F64;
}

uint32_t loom_amdgpu_attr_f32_bit_pattern(loom_attribute_t value) {
  const float f32_value = (float)loom_attr_as_f64(value);
  uint32_t bit_pattern = 0;
  memcpy(&bit_pattern, &f32_value, sizeof(bit_pattern));
  return bit_pattern;
}

bool loom_amdgpu_attr_is_16bit_float_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_F64;
}

uint32_t loom_amdgpu_attr_16bit_float_bit_pattern(loom_scalar_type_t type,
                                                  loom_attribute_t value) {
  const float f32_value = (float)loom_attr_as_f64(value);
  switch (type) {
    case LOOM_SCALAR_TYPE_F16:
      return iree_math_f32_to_f16(f32_value);
    case LOOM_SCALAR_TYPE_BF16:
      return iree_math_f32_to_bf16(f32_value);
    default:
      IREE_ASSERT_UNREACHABLE("expected f16 or bf16");
      return 0;
  }
}

bool loom_amdgpu_value_as_i32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       int64_t* out_value) {
  *out_value = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_type_is_i32(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  return fact_table != NULL &&
         loom_amdgpu_value_facts_as_exact_i32(
             loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

bool loom_amdgpu_value_as_i1_constant(loom_low_lower_context_t* context,
                                      loom_value_id_t value_id,
                                      bool* out_value) {
  *out_value = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_type_is_i1(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  return fact_table != NULL &&
         loom_value_facts_as_exact_bool(
             loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

bool loom_amdgpu_value_as_f32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       uint32_t* out_bit_pattern) {
  *out_bit_pattern = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) {
    return false;
  }
  return loom_amdgpu_value_facts_as_f32_bit_pattern(
      loom_value_fact_table_lookup(fact_table, value_id), out_bit_pattern);
}

bool loom_amdgpu_value_as_address_constant(loom_low_lower_context_t* context,
                                           loom_value_id_t value_id,
                                           int64_t* out_value) {
  *out_value = 0;
  if (!loom_amdgpu_value_is_address_scalar(context, value_id)) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  return fact_table != NULL &&
         loom_value_facts_as_exact_i64(
             loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

bool loom_amdgpu_value_can_materialize_as_vgpr_i32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), value_id);
  return loom_amdgpu_type_is_i32(type) ||
         loom_amdgpu_vector_i32_register_count(type) != 0;
}

bool loom_amdgpu_value_can_materialize_as_vgpr_f32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), value_id);
  return loom_amdgpu_type_is_f32(type) ||
         loom_amdgpu_vector_f32_register_count(type) != 0;
}

bool loom_amdgpu_value_can_materialize_as_vgpr_address(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_value_is_address_scalar(context, value_id);
}

bool loom_amdgpu_value_can_materialize_as_native_i1_mask(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i1(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}
