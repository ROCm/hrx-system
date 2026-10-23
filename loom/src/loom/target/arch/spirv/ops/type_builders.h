// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// clang-format off
#ifndef LOOM_TARGET_ARCH_SPIRV_OPS_TYPE_BUILDERS_H_
#define LOOM_TARGET_ARCH_SPIRV_OPS_TYPE_BUILDERS_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/ir.h"
#include "loom/ir/types.h"
#include "loom/target/arch/spirv/isa.h"
#include "loom/target/arch/spirv/scalar_type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_module_t loom_module_t;

// Exact-build family keys; serialized types use their public names.
enum {
  LOOM_PARAMETERIZED_TYPE_SPIRV_COOPERATIVE_MATRIX = LOOM_PARAMETERIZED_TYPE_KIND(LOOM_DIALECT_SPIRV, 0),
};

static inline bool loom_spirv_cooperative_matrix_type_isa(loom_type_t type) {
  return loom_type_is_parameterized(type) && loom_type_parameterized_kind(type) == LOOM_PARAMETERIZED_TYPE_SPIRV_COOPERATIVE_MATRIX;
}
enum { LOOM_SPIRV_COOPERATIVE_MATRIX_TYPE_ROWS_PARAMETER_INDEX = 0 };
static inline int64_t loom_spirv_cooperative_matrix_type_rows(loom_type_t type) {
  return loom_attr_as_i64(loom_type_parameterized_parameters(type)[LOOM_SPIRV_COOPERATIVE_MATRIX_TYPE_ROWS_PARAMETER_INDEX]);
}
enum { LOOM_SPIRV_COOPERATIVE_MATRIX_TYPE_COLUMNS_PARAMETER_INDEX = 1 };
static inline int64_t loom_spirv_cooperative_matrix_type_columns(loom_type_t type) {
  return loom_attr_as_i64(loom_type_parameterized_parameters(type)[LOOM_SPIRV_COOPERATIVE_MATRIX_TYPE_COLUMNS_PARAMETER_INDEX]);
}
enum { LOOM_SPIRV_COOPERATIVE_MATRIX_TYPE_COMPONENT_TYPE_PARAMETER_INDEX = 2 };
static inline loom_spirv_scalar_type_t loom_spirv_cooperative_matrix_type_component_type(loom_type_t type) {
  return (loom_spirv_scalar_type_t)loom_attr_as_enum(loom_type_parameterized_parameters(type)[LOOM_SPIRV_COOPERATIVE_MATRIX_TYPE_COMPONENT_TYPE_PARAMETER_INDEX]);
}
enum { LOOM_SPIRV_COOPERATIVE_MATRIX_TYPE_SCOPE_PARAMETER_INDEX = 3 };
static inline loom_spirv_scope_t loom_spirv_cooperative_matrix_type_scope(loom_type_t type) {
  return (loom_spirv_scope_t)loom_attr_as_enum(loom_type_parameterized_parameters(type)[LOOM_SPIRV_COOPERATIVE_MATRIX_TYPE_SCOPE_PARAMETER_INDEX]);
}
enum { LOOM_SPIRV_COOPERATIVE_MATRIX_TYPE_USE_PARAMETER_INDEX = 4 };
static inline loom_spirv_cooperative_matrix_use_t loom_spirv_cooperative_matrix_type_use(loom_type_t type) {
  return (loom_spirv_cooperative_matrix_use_t)loom_attr_as_enum(loom_type_parameterized_parameters(type)[LOOM_SPIRV_COOPERATIVE_MATRIX_TYPE_USE_PARAMETER_INDEX]);
}
iree_status_t loom_spirv_cooperative_matrix_type_make(
    loom_module_t* module,
    int64_t rows,
    int64_t columns,
    loom_spirv_scalar_type_t component_type,
    loom_spirv_scope_t scope,
    loom_spirv_cooperative_matrix_use_t use,
    loom_type_t* out_type);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_OPS_TYPE_BUILDERS_H_
