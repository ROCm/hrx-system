// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Abstract type constraints shared by IR schemas.

#ifndef LOOM_IR_TYPE_CONSTRAINT_H_
#define LOOM_IR_TYPE_CONSTRAINT_H_

#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_type_constraint_e {
  LOOM_TYPE_CONSTRAINT_TILE = 0,
  LOOM_TYPE_CONSTRAINT_TENSOR,
  LOOM_TYPE_CONSTRAINT_INTEGER,
  LOOM_TYPE_CONSTRAINT_FLOAT,
  LOOM_TYPE_CONSTRAINT_SCALAR,
  LOOM_TYPE_CONSTRAINT_INDEX,
  LOOM_TYPE_CONSTRAINT_OFFSET,
  LOOM_TYPE_CONSTRAINT_ADDRESS,
  LOOM_TYPE_CONSTRAINT_ANY,
  LOOM_TYPE_CONSTRAINT_ANY_ENCODING,
  LOOM_TYPE_CONSTRAINT_POOL,
  LOOM_TYPE_CONSTRAINT_REGISTER,
  // Exactly i1. Used for comparison results and boolean predicates.
  LOOM_TYPE_CONSTRAINT_I1,
  // Exactly i32.
  LOOM_TYPE_CONSTRAINT_I32,
  LOOM_TYPE_CONSTRAINT_VECTOR,
  // Vector type with rank 1.
  LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR,
  // Vector type with an all-static shape.
  LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR,
  // Vector type with an all-static rank-1 shape.
  LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR,
  LOOM_TYPE_CONSTRAINT_VIEW,
  LOOM_TYPE_CONSTRAINT_BUFFER,
  // Shaped type with an integer element type.
  LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT,
  // Shaped type with a floating-point element type.
  LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT,
  // Scalar index type or non-i1 integer type.
  LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR,
  // Shaped type with index or non-i1 integer element type.
  LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT,
  // Shaped type with element type i1.
  LOOM_TYPE_CONSTRAINT_I1_ELEMENT,
  // Shaped type with element type i8.
  LOOM_TYPE_CONSTRAINT_I8_ELEMENT,
  // Shaped type with element type i32.
  LOOM_TYPE_CONSTRAINT_I32_ELEMENT,
  // Shaped type with element type f16 or bf16.
  LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT,
  // Shaped type with element type f32.
  LOOM_TYPE_CONSTRAINT_F32_ELEMENT,
  // Encoding type with address-layout role.
  LOOM_TYPE_CONSTRAINT_ENCODING_LAYOUT,
  // Encoding type with storage-schema role.
  LOOM_TYPE_CONSTRAINT_ENCODING_SCHEMA,
  // Encoding type with physical-storage role.
  LOOM_TYPE_CONSTRAINT_ENCODING_STORAGE,
  // Encoding type with numeric-transform role.
  LOOM_TYPE_CONSTRAINT_ENCODING_TRANSFORM,
  // Function-local byte storage handle.
  LOOM_TYPE_CONSTRAINT_STORAGE,
  // Scalar index, non-i1 integer, or floating-point bitwise payload.
  LOOM_TYPE_CONSTRAINT_BITWISE_SCALAR,
  // Scalar 8/16/32/64-bit integer or floating-point byte pattern.
  LOOM_TYPE_CONSTRAINT_BYTE_PATTERN_SCALAR,
  // Shaped type with index, non-i1 integer, or floating-point element type.
  LOOM_TYPE_CONSTRAINT_BITWISE_ELEMENT,
  LOOM_TYPE_CONSTRAINT_COUNT_,
} loom_type_constraint_t;

// Returns the display name for a type constraint (e.g., "tile", "integer").
const char* loom_type_constraint_name(loom_type_constraint_t constraint);

// Returns true if |type| satisfies the abstract |constraint|.
bool loom_type_satisfies_constraint(loom_type_t type,
                                    loom_type_constraint_t constraint);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_TYPE_CONSTRAINT_H_
