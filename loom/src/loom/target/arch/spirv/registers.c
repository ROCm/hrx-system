// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/registers.h"

#include "loom/target/arch/spirv/descriptors/descriptors.h"

uint16_t loom_spirv_ptr_workgroup_reg_class_id(
    loom_spirv_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SPIRV_SCALAR_TYPE_F16:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_F16;
    case LOOM_SPIRV_SCALAR_TYPE_F32:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_F32;
    case LOOM_SPIRV_SCALAR_TYPE_F64:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_F64;
    case LOOM_SPIRV_SCALAR_TYPE_BF16:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_BF16;
    case LOOM_SPIRV_SCALAR_TYPE_S8:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_I8;
    case LOOM_SPIRV_SCALAR_TYPE_S16:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_I16;
    case LOOM_SPIRV_SCALAR_TYPE_S32:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_I32;
    case LOOM_SPIRV_SCALAR_TYPE_S64:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_I64;
    case LOOM_SPIRV_SCALAR_TYPE_U8:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_U8;
    case LOOM_SPIRV_SCALAR_TYPE_U16:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_U16;
    case LOOM_SPIRV_SCALAR_TYPE_U32:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_U32;
    case LOOM_SPIRV_SCALAR_TYPE_U64:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_U64;
    case LOOM_SPIRV_SCALAR_TYPE_UNKNOWN:
      return LOOM_LOW_REG_CLASS_NONE;
  }
  return LOOM_LOW_REG_CLASS_NONE;
}

uint16_t loom_spirv_ptr_workgroup_array_reg_class_id(
    loom_spirv_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SPIRV_SCALAR_TYPE_F16:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_F16;
    case LOOM_SPIRV_SCALAR_TYPE_F32:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_F32;
    case LOOM_SPIRV_SCALAR_TYPE_F64:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_F64;
    case LOOM_SPIRV_SCALAR_TYPE_BF16:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_BF16;
    case LOOM_SPIRV_SCALAR_TYPE_S8:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_I8;
    case LOOM_SPIRV_SCALAR_TYPE_S16:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_I16;
    case LOOM_SPIRV_SCALAR_TYPE_S32:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_I32;
    case LOOM_SPIRV_SCALAR_TYPE_S64:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_I64;
    case LOOM_SPIRV_SCALAR_TYPE_U8:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_U8;
    case LOOM_SPIRV_SCALAR_TYPE_U16:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_U16;
    case LOOM_SPIRV_SCALAR_TYPE_U32:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_U32;
    case LOOM_SPIRV_SCALAR_TYPE_U64:
      return SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_U64;
    case LOOM_SPIRV_SCALAR_TYPE_UNKNOWN:
      return LOOM_LOW_REG_CLASS_NONE;
  }
  return LOOM_LOW_REG_CLASS_NONE;
}

bool loom_spirv_value_type_from_reg_class_id(
    uint16_t reg_class_id, loom_spirv_value_type_t* out_value_type) {
  static const loom_spirv_value_type_t kRegValueTypes[] = {
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_OFFSET64] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_OFFSET64,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U64,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_STORAGE_BUFFER] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_F16] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_F16,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_F16] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_F16,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_F32] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_F32,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_F32] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_F32,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_F64] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_F64,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_F64] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_F64,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_BF16] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_BF16,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_BF16] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_BF16,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_I8] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_S8,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_I8] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_S8,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_I16] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_S16,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_I16] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_S16,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_I32] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_S32,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_I32] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_S32,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_I64] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_S64,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_I64] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_S64,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_U8] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U8,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_U8] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U8,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_U16] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U16,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_U16] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U16,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_U32] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U32,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_U32] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U32,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_ARRAY_U64] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U64,
          },
      [SPIRV_LOGICAL_CORE_REG_CLASS_ID_PTR_WORKGROUP_U64] =
          {
              .value_class = LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP,
              .scalar_type = LOOM_SPIRV_SCALAR_TYPE_U64,
          },
  };
  *out_value_type = (loom_spirv_value_type_t){0};
  if (reg_class_id >= IREE_ARRAYSIZE(kRegValueTypes) ||
      kRegValueTypes[reg_class_id].value_class ==
          LOOM_SPIRV_VALUE_CLASS_UNKNOWN) {
    return false;
  }
  *out_value_type = kRegValueTypes[reg_class_id];
  return true;
}
