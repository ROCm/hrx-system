// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V target-local value typing.
//
// Logical SPIR-V low registers describe storage classes such as `spirv.id`.
// Structural register payloads and target-owned packet rows map into this
// compact value-type record for verification and binary emission.

#ifndef LOOM_TARGET_ARCH_SPIRV_VALUE_TYPES_H_
#define LOOM_TARGET_ARCH_SPIRV_VALUE_TYPES_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"
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
  // Ordinary numeric vector SSA ID with a component type and lane count.
  LOOM_SPIRV_VALUE_CLASS_VECTOR = 9,
  // Ordinary Boolean vector SSA ID with a lane count.
  LOOM_SPIRV_VALUE_CLASS_BOOL_VECTOR = 10,
} loom_spirv_value_class_t;

typedef struct loom_spirv_value_type_t {
  // Target-local value class consumed by lowering and emission.
  loom_spirv_value_class_t value_class;
  // Scalar component type for scalar, numeric vector, pointer, and cooperative
  // matrix classes.
  loom_spirv_scalar_type_t scalar_type;
  union {
    // Ordinary-vector shape metadata.
    struct {
      // Number of vector components.
      uint16_t lane_count;
    } vector;
    // Cooperative-matrix shape and use metadata.
    struct {
      // Cooperative matrix row count.
      uint16_t rows;
      // Cooperative matrix column count.
      uint16_t columns;
      // Cooperative matrix scope operand.
      loom_spirv_scope_t scope;
      // Cooperative matrix use operand.
      loom_spirv_cooperative_matrix_use_t use;
    } cooperative_matrix;
  };
} loom_spirv_value_type_t;

static_assert(sizeof(loom_spirv_value_type_t) == 20,
              "SPIR-V value types must remain compact");

// Maps a public Loom scalar, native ordinary-vector, or SPIR-V aggregate type
// to its canonical target-local value type. Returns false when the public type
// is malformed or has no logical SPIR-V representation.
bool loom_spirv_value_type_from_loom_type(
    loom_type_t type, loom_spirv_value_type_t* out_value_type);

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
    case LOOM_SPIRV_VALUE_CLASS_VECTOR:
      return lhs.scalar_type == rhs.scalar_type &&
             lhs.vector.lane_count == rhs.vector.lane_count;
    case LOOM_SPIRV_VALUE_CLASS_BOOL_VECTOR:
      return lhs.vector.lane_count == rhs.vector.lane_count;
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
      return lhs.scalar_type == rhs.scalar_type &&
             lhs.cooperative_matrix.rows == rhs.cooperative_matrix.rows &&
             lhs.cooperative_matrix.columns == rhs.cooperative_matrix.columns &&
             lhs.cooperative_matrix.scope == rhs.cooperative_matrix.scope &&
             lhs.cooperative_matrix.use == rhs.cooperative_matrix.use;
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
