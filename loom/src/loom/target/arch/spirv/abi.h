// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V low boundary ABI metadata.

#ifndef LOOM_TARGET_ARCH_SPIRV_ABI_H_
#define LOOM_TARGET_ARCH_SPIRV_ABI_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"
#include "loom/target/arch/spirv/value_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_SPIRV_ABI_ARG_VALUE_TYPES_ATTR_NAME "spirv_arg_value_types"
#define LOOM_SPIRV_ABI_RESULT_VALUE_TYPES_ATTR_NAME "spirv_result_value_types"

static inline bool loom_spirv_abi_value_type_from_source_type(
    loom_type_t source_type, loom_spirv_value_type_t* out_value_type) {
  *out_value_type = (loom_spirv_value_type_t){0};
  if (!loom_type_is_scalar(source_type)) {
    return false;
  }
  switch (loom_type_element_type(source_type)) {
    case LOOM_SCALAR_TYPE_I1:
      *out_value_type = (loom_spirv_value_type_t){
          /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_BOOL,
      };
      return true;
    case LOOM_SCALAR_TYPE_I8:
      *out_value_type = (loom_spirv_value_type_t){
          /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
          /*.scalar_type=*/LOOM_SPIRV_SCALAR_TYPE_S8,
      };
      return true;
    case LOOM_SCALAR_TYPE_I16:
      *out_value_type = (loom_spirv_value_type_t){
          /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
          /*.scalar_type=*/LOOM_SPIRV_SCALAR_TYPE_S16,
      };
      return true;
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_I32:
      *out_value_type = (loom_spirv_value_type_t){
          /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
          /*.scalar_type=*/LOOM_SPIRV_SCALAR_TYPE_S32,
      };
      return true;
    case LOOM_SCALAR_TYPE_I64:
      *out_value_type = (loom_spirv_value_type_t){
          /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
          /*.scalar_type=*/LOOM_SPIRV_SCALAR_TYPE_S64,
      };
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_value_type = (loom_spirv_value_type_t){
          /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
          /*.scalar_type=*/LOOM_SPIRV_SCALAR_TYPE_F16,
      };
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_value_type = (loom_spirv_value_type_t){
          /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
          /*.scalar_type=*/LOOM_SPIRV_SCALAR_TYPE_BF16,
      };
      return true;
    case LOOM_SCALAR_TYPE_F32:
      *out_value_type = (loom_spirv_value_type_t){
          /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
          /*.scalar_type=*/LOOM_SPIRV_SCALAR_TYPE_F32,
      };
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_value_type = (loom_spirv_value_type_t){
          /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
          /*.scalar_type=*/LOOM_SPIRV_SCALAR_TYPE_F64,
      };
      return true;
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_COUNT_:
      return false;
  }
  return false;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_ABI_H_
