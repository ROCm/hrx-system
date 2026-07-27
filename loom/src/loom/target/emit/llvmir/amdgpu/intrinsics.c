// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/amdgpu/intrinsics.h"

#include <stdio.h>

static iree_status_t loom_llvmir_declare_amdgcn_workitem_id(
    loom_llvmir_module_t* module, iree_string_view_t name,
    loom_llvmir_function_t** out_function) {
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));
  loom_llvmir_function_desc_t desc = {
      .kind = LOOM_LLVMIR_FUNCTION_DECLARATION,
      .name = name,
      .return_type = i32_type,
      .linkage = LOOM_LLVMIR_LINKAGE_DEFAULT,
      .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
      .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
  };
  return loom_llvmir_module_add_function(module, &desc, out_function);
}

iree_status_t loom_llvmir_declare_amdgcn_workitem_id_x(
    loom_llvmir_module_t* module, loom_llvmir_function_t** out_function) {
  return loom_llvmir_declare_amdgcn_workitem_id(
      module, IREE_SV("llvm.amdgcn.workitem.id.x"), out_function);
}

iree_status_t loom_llvmir_declare_amdgcn_workitem_id_y(
    loom_llvmir_module_t* module, loom_llvmir_function_t** out_function) {
  return loom_llvmir_declare_amdgcn_workitem_id(
      module, IREE_SV("llvm.amdgcn.workitem.id.y"), out_function);
}

iree_status_t loom_llvmir_declare_amdgcn_workitem_id_z(
    loom_llvmir_module_t* module, loom_llvmir_function_t** out_function) {
  return loom_llvmir_declare_amdgcn_workitem_id(
      module, IREE_SV("llvm.amdgcn.workitem.id.z"), out_function);
}

iree_status_t loom_llvmir_declare_amdgcn_make_buffer_rsrc(
    loom_llvmir_module_t* module, uint32_t result_address_space,
    uint32_t base_address_space, loom_llvmir_function_t** out_function) {
  loom_llvmir_type_id_t result_ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t base_ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i16_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, result_address_space, &result_ptr_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, base_address_space, &base_ptr_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 16, &i16_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 64, &i64_type));

  char name_buffer[64];
  int name_length = snprintf(name_buffer, sizeof(name_buffer),
                             "llvm.amdgcn.make.buffer.rsrc.p%u.p%u",
                             result_address_space, base_address_space);
  if (name_length <= 0 || (size_t)name_length >= sizeof(name_buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU buffer resource intrinsic name overflow");
  }
  loom_llvmir_function_desc_t desc = {
      .kind = LOOM_LLVMIR_FUNCTION_DECLARATION,
      .name = iree_make_string_view(name_buffer, (iree_host_size_t)name_length),
      .return_type = result_ptr_type,
      .linkage = LOOM_LLVMIR_LINKAGE_DEFAULT,
      .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
      .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_add_function(module, &desc, out_function));

  loom_llvmir_attr_t readnone_attr = {
      .kind = LOOM_LLVMIR_ATTR_READNONE,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
  loom_llvmir_parameter_desc_t base_param = {
      .type_id = base_ptr_type,
      .attrs = &readnone_attr,
      .attr_count = 1,
  };
  loom_llvmir_value_id_t ignored = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_function_add_parameter(*out_function, &base_param, &ignored));
  loom_llvmir_parameter_desc_t stride_param = {.type_id = i16_type};
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function, &stride_param, &ignored));
  loom_llvmir_parameter_desc_t records_param = {.type_id = i64_type};
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function, &records_param, &ignored));
  loom_llvmir_parameter_desc_t flags_param = {.type_id = i32_type};
  return loom_llvmir_function_add_parameter(*out_function, &flags_param,
                                            &ignored);
}
