// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/types.h"

#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"

iree_status_t loom_amdgpu_make_sgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, out_type);
}

iree_status_t loom_amdgpu_make_sgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_SGPR, unit_count, out_type);
}

iree_status_t loom_amdgpu_make_vgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, out_type);
}

iree_status_t loom_amdgpu_make_scc_type(loom_low_lower_context_t* context,
                                        loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, out_type);
}

iree_status_t loom_amdgpu_make_vgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, LOOM_AMDGPU_REG_CLASS_ID_VGPR, unit_count, out_type);
}

iree_status_t loom_amdgpu_make_descriptor_row_implicit_resource_type(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    loom_type_t* out_type) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  return loom_amdgpu_make_descriptor_implicit_resource_type(
      descriptor_set, descriptor, out_type);
}

bool loom_amdgpu_low_type_is_register_class(loom_low_lower_context_t* context,
                                            loom_type_t type,
                                            uint16_t reg_class_id) {
  if (!loom_low_type_is_register(type)) {
    return false;
  }
  return loom_low_register_type_descriptor_set_stable_id(type) ==
             loom_low_lower_context_descriptor_set(context)->stable_id &&
         loom_low_register_type_class_id(type) == reg_class_id;
}

bool loom_amdgpu_low_type_is_register_class_count(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    uint32_t register_unit_count) {
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_unit_count(type) != register_unit_count) {
    return false;
  }
  return loom_amdgpu_low_type_is_register_class(context, type, reg_class_id);
}

bool loom_amdgpu_low_value_is_register_class_count(
    loom_low_lower_context_t* context, loom_value_id_t low_value,
    uint16_t reg_class_id, uint32_t register_unit_count) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  return loom_amdgpu_low_type_is_register_class_count(
      context, loom_module_value_type(module, low_value), reg_class_id,
      register_unit_count);
}
