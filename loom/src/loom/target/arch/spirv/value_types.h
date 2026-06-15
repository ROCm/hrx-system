// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V target-local value typing.
//
// Logical SPIR-V low registers describe storage classes such as `spirv.id`;
// they deliberately do not encode every scalar payload carried by an SSA ID.
// Target-owned descriptor rows, ABI metadata, and emission tables use this
// compact value-type record when a specific payload is required.

#ifndef LOOM_TARGET_ARCH_SPIRV_VALUE_TYPES_H_
#define LOOM_TARGET_ARCH_SPIRV_VALUE_TYPES_H_

#include "iree/base/api.h"
#include "loom/target/arch/spirv/isa.h"
#include "loom/target/arch/spirv/scalar_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_spirv_value_class_e {
  // Unknown or uninitialized value class.
  LOOM_SPIRV_VALUE_CLASS_UNKNOWN = 0,
  // SPIR-V SSA scalar ID with a concrete scalar type.
  LOOM_SPIRV_VALUE_CLASS_SCALAR = 1,
  // 64-bit byte offset used by address calculations.
  LOOM_SPIRV_VALUE_CLASS_OFFSET64 = 2,
  // Raw 64-bit PhysicalStorageBuffer address before scalar pointer typing.
  LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS = 3,
  // PhysicalStorageBuffer pointer with a concrete scalar pointee type.
  LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER = 4,
  // SPIR-V boolean SSA ID produced by compare packets.
  LOOM_SPIRV_VALUE_CLASS_BOOL = 5,
  // KHR cooperative matrix SSA ID with concrete component and use operands.
  LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX = 6,
  // Workgroup pointer with a concrete scalar pointee type.
  LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP = 7,
  // Workgroup array pointer with a concrete scalar element type.
  LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY = 8,
} loom_spirv_value_class_t;

typedef struct loom_spirv_value_type_t {
  // Target-local value class consumed by lowering and emission.
  loom_spirv_value_class_t value_class;
  // Scalar component type for scalar, pointer, and cooperative matrix classes.
  loom_spirv_scalar_type_t scalar_type;
  // Cooperative matrix row count.
  uint16_t rows;
  // Cooperative matrix column count.
  uint16_t columns;
  // Cooperative matrix scope operand.
  loom_spirv_scope_t scope;
  // Cooperative matrix use operand.
  loom_spirv_cooperative_matrix_use_t cooperative_matrix_use;
} loom_spirv_value_type_t;

enum loom_spirv_abi_value_type_code_e {
  // No payload metadata is attached to this ABI position.
  LOOM_SPIRV_ABI_VALUE_TYPE_NONE = 0,
  // Boolean payload represented by an i32 ABI field when materialized through
  // descriptor or push-constant storage.
  LOOM_SPIRV_ABI_VALUE_TYPE_BOOL = 1,
  // First code value reserved for scalar payloads.
  LOOM_SPIRV_ABI_VALUE_TYPE_SCALAR_BASE = 16,
};

static inline int64_t loom_spirv_abi_value_type_code_from_scalar(
    loom_spirv_scalar_type_t scalar_type) {
  return LOOM_SPIRV_ABI_VALUE_TYPE_SCALAR_BASE + (int64_t)scalar_type;
}

static inline bool loom_spirv_abi_value_type_encode(
    loom_spirv_value_type_t value_type, int64_t* out_code) {
  *out_code = LOOM_SPIRV_ABI_VALUE_TYPE_NONE;
  switch (value_type.value_class) {
    case LOOM_SPIRV_VALUE_CLASS_SCALAR:
      if (loom_spirv_scalar_type_descriptor(value_type.scalar_type) == NULL) {
        return false;
      }
      *out_code =
          loom_spirv_abi_value_type_code_from_scalar(value_type.scalar_type);
      return true;
    case LOOM_SPIRV_VALUE_CLASS_BOOL:
      *out_code = LOOM_SPIRV_ABI_VALUE_TYPE_BOOL;
      return true;
    case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
    case LOOM_SPIRV_VALUE_CLASS_OFFSET64:
    case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS:
    case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP:
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY:
      return true;
  }
  return false;
}

static inline bool loom_spirv_abi_value_type_decode(
    int64_t code, loom_spirv_value_type_t* out_value_type) {
  *out_value_type = (loom_spirv_value_type_t){
      /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_UNKNOWN,
  };
  if (code == LOOM_SPIRV_ABI_VALUE_TYPE_NONE) {
    return true;
  }
  if (code == LOOM_SPIRV_ABI_VALUE_TYPE_BOOL) {
    *out_value_type = (loom_spirv_value_type_t){
        /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_BOOL,
    };
    return true;
  }
  if (code <= LOOM_SPIRV_ABI_VALUE_TYPE_SCALAR_BASE) {
    return false;
  }
  const int64_t scalar_code = code - LOOM_SPIRV_ABI_VALUE_TYPE_SCALAR_BASE;
  if (scalar_code > UINT8_MAX) {
    return false;
  }
  const loom_spirv_scalar_type_t scalar_type =
      (loom_spirv_scalar_type_t)scalar_code;
  if (loom_spirv_scalar_type_descriptor(scalar_type) == NULL) {
    return false;
  }
  *out_value_type = (loom_spirv_value_type_t){
      /*.value_class=*/LOOM_SPIRV_VALUE_CLASS_SCALAR,
      /*.scalar_type=*/scalar_type,
  };
  return true;
}

// Returns true when |lhs| and |rhs| name the same target-local SPIR-V value
// type.
static inline bool loom_spirv_value_type_equal(loom_spirv_value_type_t lhs,
                                               loom_spirv_value_type_t rhs) {
  if (lhs.value_class != rhs.value_class) {
    return false;
  }
  switch (lhs.value_class) {
    case LOOM_SPIRV_VALUE_CLASS_SCALAR:
    case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP:
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY:
      return lhs.scalar_type == rhs.scalar_type;
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
      return lhs.scalar_type == rhs.scalar_type && lhs.rows == rhs.rows &&
             lhs.columns == rhs.columns && lhs.scope == rhs.scope &&
             lhs.cooperative_matrix_use == rhs.cooperative_matrix_use;
    case LOOM_SPIRV_VALUE_CLASS_OFFSET64:
    case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS:
    case LOOM_SPIRV_VALUE_CLASS_BOOL:
    case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
      return true;
  }
  return false;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_VALUE_TYPES_H_
