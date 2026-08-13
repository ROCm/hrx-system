// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

static iree_status_t loom_amdgpu_address_emit(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_amdgpu_address_emit_operand_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint16_t operand_index, iree_string_view_t operand_name,
    loom_type_t actual_type, iree_string_view_t expected_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(operand_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                    operand_index)),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_amdgpu_address_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                                  IREE_ARRAYSIZE(params));
}

static bool loom_amdgpu_address_type_is_sgpr_units(loom_type_t type,
                                                   uint32_t unit_count) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_class_id(type) ==
             LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
         loom_low_register_type_unit_count(type) == unit_count;
}

iree_status_t loom_amdgpu_address_add_scaled_u32_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const int64_t byte_shift = loom_amdgpu_address_add_scaled_u32_byte_shift(op);
  if (byte_shift < 0 || byte_shift > 31) {
    const loom_diagnostic_param_t params[] = {
        loom_param_with_field_ref(
            loom_param_string(IREE_SV("byte_shift")),
            loom_diagnostic_field_ref(
                LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                loom_amdgpu_address_add_scaled_u32_byte_shift_ATTR_INDEX)),
        loom_param_i64(byte_shift),
        loom_param_string(IREE_SV("an integer in the inclusive range [0, 31]")),
    };
    return loom_amdgpu_address_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                                    IREE_ARRAYSIZE(params));
  }

  const loom_type_t base_type = loom_module_value_type(
      module, loom_amdgpu_address_add_scaled_u32_base(op));
  if (!loom_amdgpu_address_type_is_sgpr_units(base_type, 2)) {
    return loom_amdgpu_address_emit_operand_constraint(
        emitter, op, 0, IREE_SV("base"), base_type,
        IREE_SV("a two-unit AMDGPU SGPR address"));
  }
  const loom_type_t offset_type = loom_module_value_type(
      module, loom_amdgpu_address_add_scaled_u32_offset(op));
  if (!loom_amdgpu_address_type_is_sgpr_units(offset_type, 1)) {
    return loom_amdgpu_address_emit_operand_constraint(
        emitter, op, 1, IREE_SV("offset"), offset_type,
        IREE_SV("a one-unit AMDGPU SGPR unsigned offset"));
  }
  return iree_ok_status();
}
