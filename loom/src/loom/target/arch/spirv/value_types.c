// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/value_types.h"

#include "loom/target/arch/spirv/ops/types.h"

static bool loom_spirv_numeric_scalar_type_from_loom_scalar_type(
    loom_scalar_type_t type, loom_spirv_scalar_type_t* out_scalar_type) {
  *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  switch (type) {
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_S8;
      return true;
    case LOOM_SCALAR_TYPE_I16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_S16;
      return true;
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_I32:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_S32;
      return true;
    case LOOM_SCALAR_TYPE_I64:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_S64;
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_F16;
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_BF16;
      return true;
    case LOOM_SCALAR_TYPE_F32:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_F32;
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_F64;
      return true;
    case LOOM_SCALAR_TYPE_OFFSET:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_U64;
      return true;
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_COUNT_:
      return false;
  }
  return false;
}

bool loom_spirv_value_type_from_loom_type(
    loom_type_t type, loom_spirv_value_type_t* out_value_type) {
  *out_value_type = (loom_spirv_value_type_t){
      /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_UNKNOWN,
  };
  if (loom_spirv_cooperative_matrix_type_isa(type)) {
    const int64_t rows = loom_spirv_cooperative_matrix_type_rows(type);
    const int64_t columns = loom_spirv_cooperative_matrix_type_columns(type);
    const loom_spirv_scalar_type_t component_type =
        loom_spirv_cooperative_matrix_type_component_type(type);
    const loom_spirv_scope_t scope =
        loom_spirv_cooperative_matrix_type_scope(type);
    const loom_spirv_cooperative_matrix_use_t use =
        loom_spirv_cooperative_matrix_type_use(type);
    if (rows <= 0 || rows > UINT16_MAX || columns <= 0 ||
        columns > UINT16_MAX) {
      return false;
    }
    *out_value_type = (loom_spirv_value_type_t){
        .value_class = LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX,
        .scalar_type = component_type,
        .cooperative_matrix =
            {
                .rows = (uint16_t)rows,
                .columns = (uint16_t)columns,
                .scope = scope,
                .use = use,
            },
    };
    return true;
  }
  if (loom_type_is_scalar(type)) {
    if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1) {
      *out_value_type = (loom_spirv_value_type_t){
          /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_BOOL,
      };
      return true;
    }
    if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET) {
      return false;
    }
    loom_spirv_scalar_type_t scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
    if (!loom_spirv_numeric_scalar_type_from_loom_scalar_type(
            loom_type_element_type(type), &scalar_type)) {
      return false;
    }
    *out_value_type = (loom_spirv_value_type_t){
        /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
        /*.scalar_type=*/scalar_type,
    };
    return true;
  }
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      loom_type_dim_is_dynamic_at(type, 0)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 2 || lane_count > 4) {
    return false;
  }
  if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1) {
    *out_value_type = (loom_spirv_value_type_t){
        /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_BOOL_VECTOR,
        .vector =
            {
                .lane_count = (uint16_t)lane_count,
            },
    };
    return true;
  }
  if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_F8E4M3 ||
      loom_type_element_type(type) == LOOM_SCALAR_TYPE_F8E5M2) {
    return false;
  }
  loom_spirv_scalar_type_t scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  if (!loom_spirv_numeric_scalar_type_from_loom_scalar_type(
          loom_type_element_type(type), &scalar_type)) {
    return false;
  }
  *out_value_type = (loom_spirv_value_type_t){
      /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_VECTOR,
      /*.scalar_type=*/scalar_type,
      .vector =
          {
              .lane_count = (uint16_t)lane_count,
          },
  };
  return true;
}
