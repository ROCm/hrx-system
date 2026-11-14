// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/registers.h"

#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/registers.h"

bool loom_spirv_value_type_from_low_register_type(
    loom_type_t type, loom_spirv_value_type_t* out_value_type) {
  *out_value_type = (loom_spirv_value_type_t){0};
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          SPIRV_LOGICAL_CORE_DESCRIPTOR_SET_ID) {
    return false;
  }

  const uint16_t register_class_id = loom_low_register_type_class_id(type);
  const loom_type_t* value_type = loom_type_register_value_type(type);
  if (register_class_id == SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID) {
    return value_type != NULL &&
           loom_spirv_value_type_from_loom_type(*value_type, out_value_type);
  }
  if (value_type != NULL) {
    return false;
  }
  return loom_spirv_value_type_from_reg_class_id(register_class_id,
                                                 out_value_type);
}

// One scalar vocabulary feeds both pointer directions and both storage classes.
#define LOOM_SPIRV_LOGICAL_SCALARS(M, space) \
  M(space, F16, F16)                         \
  M(space, F32, F32)                         \
  M(space, F64, F64)                         \
  M(space, BF16, BF16)                       \
  M(space, I8, S8)                           \
  M(space, I16, S16)                         \
  M(space, I32, S32)                         \
  M(space, I64, S64)                         \
  M(space, U8, U8)                           \
  M(space, U16, U16)                         \
  M(space, U32, U32)                         \
  M(space, U64, U64)

#define LOOM_SPIRV_POINTER_REGISTER(space, suffix, scalar) \
  [LOOM_SPIRV_SCALAR_TYPE_##scalar] =                      \
      SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_##space##_##suffix,

uint16_t loom_spirv_logical_pointer_reg_class_id(
    loom_spirv_value_class_t value_class,
    loom_spirv_scalar_type_t scalar_type) {
  static const uint16_t kClasses[][LOOM_SPIRV_SCALAR_TYPE_U64 + 1] = {
      [LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP -
       LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP] =
          {[LOOM_SPIRV_SCALAR_TYPE_UNKNOWN] = LOOM_LOW_REG_CLASS_NONE,
           LOOM_SPIRV_LOGICAL_SCALARS(LOOM_SPIRV_POINTER_REGISTER, WORKGROUP)},
      [LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY -
       LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP] =
          {[LOOM_SPIRV_SCALAR_TYPE_UNKNOWN] = LOOM_LOW_REG_CLASS_NONE,
           LOOM_SPIRV_LOGICAL_SCALARS(LOOM_SPIRV_POINTER_REGISTER,
                                      WORKGROUP_ARRAY)},
      [LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION -
       LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP] =
          {[LOOM_SPIRV_SCALAR_TYPE_UNKNOWN] = LOOM_LOW_REG_CLASS_NONE,
           LOOM_SPIRV_LOGICAL_SCALARS(LOOM_SPIRV_POINTER_REGISTER, FUNCTION)},
      [LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION_ARRAY -
       LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP] =
          {[LOOM_SPIRV_SCALAR_TYPE_UNKNOWN] = LOOM_LOW_REG_CLASS_NONE,
           LOOM_SPIRV_LOGICAL_SCALARS(LOOM_SPIRV_POINTER_REGISTER,
                                      FUNCTION_ARRAY)},
  };
  if ((unsigned)scalar_type > LOOM_SPIRV_SCALAR_TYPE_U64 ||
      (value_class != LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP &&
       value_class != LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY &&
       value_class != LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION &&
       value_class != LOOM_SPIRV_VALUE_CLASS_PTR_FUNCTION_ARRAY)) {
    return LOOM_LOW_REG_CLASS_NONE;
  }
  return kClasses[value_class - LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP]
                 [scalar_type];
}
#undef LOOM_SPIRV_POINTER_REGISTER

#define LOOM_SPIRV_POINTER_VALUE_TYPE(space, suffix, scalar)               \
  [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_##space##_##                        \
      suffix] = {.value_class = LOOM_SPIRV_VALUE_CLASS_PTR_##space,        \
                 .scalar_type = LOOM_SPIRV_SCALAR_TYPE_##scalar},          \
 [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_##space##_ARRAY_##                   \
     suffix] = {.value_class = LOOM_SPIRV_VALUE_CLASS_PTR_##space##_ARRAY, \
                .scalar_type = LOOM_SPIRV_SCALAR_TYPE_##scalar},

bool loom_spirv_value_type_from_reg_class_id(
    uint16_t reg_class_id, loom_spirv_value_type_t* out_value_type) {
  static const loom_spirv_value_type_t kRegValueTypes[] = {
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64] =
          {.value_class = LOOM_SPIRV_VALUE_CLASS_OFFSET64,
           .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U64},
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER] =
          {.value_class = LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS},
      LOOM_SPIRV_LOGICAL_SCALARS(LOOM_SPIRV_POINTER_VALUE_TYPE, WORKGROUP)
          LOOM_SPIRV_LOGICAL_SCALARS(LOOM_SPIRV_POINTER_VALUE_TYPE, FUNCTION)};
  *out_value_type = (loom_spirv_value_type_t){0};
  if (reg_class_id >= IREE_ARRAYSIZE(kRegValueTypes) ||
      kRegValueTypes[reg_class_id].value_class ==
          LOOM_SPIRV_VALUE_CLASS_UNKNOWN) {
    return false;
  }
  *out_value_type = kRegValueTypes[reg_class_id];
  return true;
}
#undef LOOM_SPIRV_POINTER_VALUE_TYPE
#undef LOOM_SPIRV_LOGICAL_SCALARS
