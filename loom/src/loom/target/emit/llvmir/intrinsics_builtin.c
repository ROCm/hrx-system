// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/intrinsics_builtin.h"

#include <stdio.h>

static loom_llvmir_attr_t loom_llvmir_builtin_attr(
    loom_llvmir_attr_kind_t kind) {
  return (loom_llvmir_attr_t){
      .kind = kind,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
}

static iree_status_t loom_llvmir_builtin_declare_void_intrinsic(
    loom_llvmir_module_t* module, iree_string_view_t name,
    loom_llvmir_function_t** out_function) {
  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_void_type(module, &void_type));
  loom_llvmir_function_desc_t desc = {
      .kind = LOOM_LLVMIR_FUNCTION_DECLARATION,
      .name = name,
      .return_type = void_type,
      .linkage = LOOM_LLVMIR_LINKAGE_DEFAULT,
      .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
      .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
  };
  return loom_llvmir_module_add_function(module, &desc, out_function);
}

iree_status_t loom_llvmir_declare_memcpy(
    loom_llvmir_module_t* module, uint32_t target_address_space,
    uint32_t source_address_space, uint32_t length_bit_width,
    loom_llvmir_function_t** out_function) {
  loom_llvmir_type_id_t target_ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t source_ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t length_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i1_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, target_address_space, &target_ptr_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, source_address_space, &source_ptr_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_integer_type(
      module, length_bit_width, &length_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 1, &i1_type));

  char name_buffer[64];
  int name_length =
      snprintf(name_buffer, sizeof(name_buffer), "llvm.memcpy.p%u.p%u.i%u",
               target_address_space, source_address_space, length_bit_width);
  if (name_length <= 0 || (size_t)name_length >= sizeof(name_buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM memcpy intrinsic name overflow");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_builtin_declare_void_intrinsic(
      module, iree_make_string_view(name_buffer, (iree_host_size_t)name_length),
      out_function));

  loom_llvmir_attr_t target_attrs[] = {
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_NOALIAS),
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_NOCAPTURE),
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_WRITEONLY),
  };
  loom_llvmir_attr_t source_attrs[] = {
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_NOALIAS),
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_NOCAPTURE),
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_READONLY),
  };
  loom_llvmir_attr_t immarg_attrs[] = {
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_IMMARG),
  };
  loom_llvmir_value_id_t ignored = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = target_ptr_type,
          .attrs = target_attrs,
          .attr_count = IREE_ARRAYSIZE(target_attrs),
      },
      &ignored));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = source_ptr_type,
          .attrs = source_attrs,
          .attr_count = IREE_ARRAYSIZE(source_attrs),
      },
      &ignored));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function, &(loom_llvmir_parameter_desc_t){.type_id = length_type},
      &ignored));
  return loom_llvmir_function_add_parameter(
      *out_function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = i1_type,
          .attrs = immarg_attrs,
          .attr_count = IREE_ARRAYSIZE(immarg_attrs),
      },
      &ignored);
}

iree_status_t loom_llvmir_declare_memset(
    loom_llvmir_module_t* module, uint32_t target_address_space,
    uint32_t length_bit_width, loom_llvmir_function_t** out_function) {
  loom_llvmir_type_id_t target_ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i8_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t length_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i1_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, target_address_space, &target_ptr_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 8, &i8_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_integer_type(
      module, length_bit_width, &length_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 1, &i1_type));

  char name_buffer[64];
  int name_length =
      snprintf(name_buffer, sizeof(name_buffer), "llvm.memset.p%u.i%u",
               target_address_space, length_bit_width);
  if (name_length <= 0 || (size_t)name_length >= sizeof(name_buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM memset intrinsic name overflow");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_builtin_declare_void_intrinsic(
      module, iree_make_string_view(name_buffer, (iree_host_size_t)name_length),
      out_function));

  loom_llvmir_attr_t target_attrs[] = {
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_NOCAPTURE),
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_WRITEONLY),
  };
  loom_llvmir_attr_t immarg_attrs[] = {
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_IMMARG),
  };
  loom_llvmir_value_id_t ignored = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = target_ptr_type,
          .attrs = target_attrs,
          .attr_count = IREE_ARRAYSIZE(target_attrs),
      },
      &ignored));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function, &(loom_llvmir_parameter_desc_t){.type_id = i8_type},
      &ignored));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function, &(loom_llvmir_parameter_desc_t){.type_id = length_type},
      &ignored));
  return loom_llvmir_function_add_parameter(
      *out_function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = i1_type,
          .attrs = immarg_attrs,
          .attr_count = IREE_ARRAYSIZE(immarg_attrs),
      },
      &ignored);
}

iree_status_t loom_llvmir_declare_prefetch(
    loom_llvmir_module_t* module, uint32_t pointer_address_space,
    loom_llvmir_function_t** out_function) {
  loom_llvmir_type_id_t pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, pointer_address_space, &pointer_type));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 32, &i32_type));

  char name_buffer[64];
  int name_length = snprintf(name_buffer, sizeof(name_buffer),
                             "llvm.prefetch.p%u", pointer_address_space);
  if (name_length <= 0 || (size_t)name_length >= sizeof(name_buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM prefetch intrinsic name overflow");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_builtin_declare_void_intrinsic(
      module, iree_make_string_view(name_buffer, (iree_host_size_t)name_length),
      out_function));

  loom_llvmir_attr_t pointer_attrs[] = {
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_READONLY),
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_NOCAPTURE),
  };
  loom_llvmir_attr_t immarg_attrs[] = {
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_IMMARG),
  };
  loom_llvmir_value_id_t ignored = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = pointer_type,
          .attrs = pointer_attrs,
          .attr_count = IREE_ARRAYSIZE(pointer_attrs),
      },
      &ignored));
  loom_llvmir_parameter_desc_t immarg_param = {
      .type_id = i32_type,
      .attrs = immarg_attrs,
      .attr_count = IREE_ARRAYSIZE(immarg_attrs),
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function, &immarg_param, &ignored));
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function, &immarg_param, &ignored));
  return loom_llvmir_function_add_parameter(*out_function, &immarg_param,
                                            &ignored);
}

static iree_status_t loom_llvmir_declare_lifetime_marker(
    loom_llvmir_module_t* module, iree_string_view_t marker_name,
    uint32_t pointer_address_space, loom_llvmir_function_t** out_function) {
  loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 64, &i64_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, pointer_address_space, &pointer_type));

  char name_buffer[64];
  int name_length =
      snprintf(name_buffer, sizeof(name_buffer), "llvm.lifetime.%.*s.p%u",
               (int)marker_name.size, marker_name.data, pointer_address_space);
  if (name_length <= 0 || (size_t)name_length >= sizeof(name_buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM lifetime intrinsic name overflow");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_builtin_declare_void_intrinsic(
      module, iree_make_string_view(name_buffer, (iree_host_size_t)name_length),
      out_function));

  loom_llvmir_attr_t immarg_attrs[] = {
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_IMMARG),
  };
  loom_llvmir_attr_t pointer_attrs[] = {
      loom_llvmir_builtin_attr(LOOM_LLVMIR_ATTR_NOCAPTURE),
  };
  loom_llvmir_value_id_t ignored = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      *out_function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = i64_type,
          .attrs = immarg_attrs,
          .attr_count = IREE_ARRAYSIZE(immarg_attrs),
      },
      &ignored));
  return loom_llvmir_function_add_parameter(
      *out_function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = pointer_type,
          .attrs = pointer_attrs,
          .attr_count = IREE_ARRAYSIZE(pointer_attrs),
      },
      &ignored);
}

iree_status_t loom_llvmir_declare_lifetime_start(
    loom_llvmir_module_t* module, uint32_t pointer_address_space,
    loom_llvmir_function_t** out_function) {
  return loom_llvmir_declare_lifetime_marker(
      module, IREE_SV("start"), pointer_address_space, out_function);
}

iree_status_t loom_llvmir_declare_lifetime_end(
    loom_llvmir_module_t* module, uint32_t pointer_address_space,
    loom_llvmir_function_t** out_function) {
  return loom_llvmir_declare_lifetime_marker(
      module, IREE_SV("end"), pointer_address_space, out_function);
}
