// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scalar element type vocabulary for Loom IR.
//
// These are internal compiler values, not bytecode-stable wire enums. The
// bytecode format has its own independent encoding and versioning.

#ifndef LOOM_IR_SCALAR_TYPE_H_
#define LOOM_IR_SCALAR_TYPE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scalar element type kind.
//
// Ordered: address types, integers by width, floats by width.
enum loom_scalar_type_e {
  // Signed target-width integer for loop bounds, dimension sizes, and general
  // indexing arithmetic. Arithmetic follows signed semantics.
  LOOM_SCALAR_TYPE_INDEX = 0,
  // Unsigned target-width integer for buffer byte offsets and addressing.
  // Arithmetic follows unsigned semantics. Same bitwidth as index.
  LOOM_SCALAR_TYPE_OFFSET = 1,
  // 1-bit integer. Boolean results (comparisons, predicates).
  LOOM_SCALAR_TYPE_I1 = 2,
  // 8-bit signed integer. Quantized weights, byte-level data.
  LOOM_SCALAR_TYPE_I8 = 3,
  // 16-bit signed integer. Intermediate quantized computations.
  LOOM_SCALAR_TYPE_I16 = 4,
  // 32-bit signed integer. General-purpose integer arithmetic.
  LOOM_SCALAR_TYPE_I32 = 5,
  // 64-bit signed integer. Large counts, hash values.
  LOOM_SCALAR_TYPE_I64 = 6,
  // 8-bit float, E4M3 variant.
  LOOM_SCALAR_TYPE_F8E4M3 = 7,
  // 8-bit float, E5M2 variant.
  LOOM_SCALAR_TYPE_F8E5M2 = 8,
  // IEEE 754 binary16 half-precision.
  LOOM_SCALAR_TYPE_F16 = 9,
  // bfloat16.
  LOOM_SCALAR_TYPE_BF16 = 10,
  // IEEE 754 binary32 single-precision.
  LOOM_SCALAR_TYPE_F32 = 11,
  // IEEE 754 binary64 double-precision.
  LOOM_SCALAR_TYPE_F64 = 12,
  LOOM_SCALAR_TYPE_COUNT_,
};
// Raw scalar type storage. Parsed and bytecode-loaded types may carry invalid
// ordinals until validation reports them.
typedef uint8_t loom_scalar_type_t;

// Returns the name string for a scalar type (e.g., "f32", "index").
// Returns NULL if |type| is out of range.
const char* loom_scalar_type_name(loom_scalar_type_t type);

// Returns the bitwidth of a scalar type, or 0 if |type| is out of range.
// Index and offset return 64.
int32_t loom_scalar_type_bitwidth(loom_scalar_type_t type);

// Returns the signed i64 value domain used to reason about integer-like scalar
// values of |type|. Offset is represented as the non-negative address domain
// because IR integer literals are signed i64 payloads.
bool loom_scalar_type_integer_domain(loom_scalar_type_t type, int64_t* out_lo,
                                     int64_t* out_hi);

// Parses a scalar type name. Returns true on success.
bool loom_scalar_type_parse(iree_string_view_t name,
                            loom_scalar_type_t* out_type);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_SCALAR_TYPE_H_
