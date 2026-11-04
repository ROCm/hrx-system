// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_COLLECTIVE_ARITHMETIC_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_COLLECTIVE_ARITHMETIC_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/ops/combining.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native subgroup topology shared by product and extremum arithmetic.
typedef enum loom_spirv_collective_mode_e {
  LOOM_SPIRV_COLLECTIVE_REDUCE,
  LOOM_SPIRV_COLLECTIVE_CLUSTERED_REDUCE,
  LOOM_SPIRV_COLLECTIVE_INCLUSIVE_SCAN,
  LOOM_SPIRV_COLLECTIVE_EXCLUSIVE_SCAN,
} loom_spirv_collective_mode_t;

// Immutable native-carrier instructions shared by subgroup and workgroup
// extrema. Ordered keys preserve subnormals and signed zeros without floating
// arithmetic; the combining kind selects number preference or NaN propagation.
typedef struct loom_spirv_extrema_row_t loom_spirv_extrema_row_t;

typedef struct loom_spirv_collective_extrema_t {
  // Registry-owned instructions for the selected native carrier.
  const loom_spirv_extrema_row_t* row;
  // Floating-point source and decoded-result register type.
  loom_type_t float_type;
  // Signed sortable-key register type.
  loom_type_t integer_type;
  // Component-wise comparison register type.
  loom_type_t predicate_type;
  // Scalar integer register type used to construct splatted constants.
  loom_type_t constant_type;
  // Immediate name for the scalar integer literal.
  loom_string_id_t constant_key;
  // Source extremum semantics, including its NaN policy.
  loom_combining_kind_t kind;
} loom_spirv_collective_extrema_t;

typedef struct loom_spirv_extrema_constants_t {
  // Mask separating magnitude from sign, reused during decoding.
  loom_value_id_t magnitude_mask;
  // Shift count used to sign-extend the floating-point sign.
  loom_value_id_t sign_shift;
  // Positive infinity bits, also the minimum scan's empty-prefix key.
  loom_value_id_t infinity;
} loom_spirv_extrema_constants_t;

// Returns the native arithmetic row, or NULL for an unsupported source carrier.
const loom_spirv_extrema_row_t* loom_spirv_collective_extrema_row(
    loom_type_t source_type);

// Retains typed carriers for a supported row and floating-point extremum kind.
iree_status_t loom_spirv_collective_extrema_initialize(
    loom_low_lower_context_t* context, const loom_spirv_extrema_row_t* row,
    loom_combining_kind_t kind,
    loom_spirv_collective_extrema_t* out_arithmetic);

// Encodes a floating value as a signed sortable key. Exclusive scans reserve
// the integer identity for an empty prefix, distinct from every encoded NaN.
iree_status_t loom_spirv_collective_extrema_encode(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_collective_extrema_t* arithmetic, loom_value_id_t value,
    bool exclusive, loom_spirv_extrema_constants_t* out_constants,
    loom_value_id_t* out_key);

// Decodes a reduced key, mapping an exclusive scan's empty prefix to infinity.
iree_status_t loom_spirv_collective_extrema_decode(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_collective_extrema_t* arithmetic,
    loom_spirv_extrema_constants_t constants, loom_value_id_t key,
    bool exclusive, loom_value_id_t* out_value);

// Selects and emits subgroup extrema through the same arithmetic as workgroups.
iree_status_t loom_spirv_select_subgroup_extrema(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);
iree_status_t loom_spirv_emit_subgroup_extrema(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Scalar floating instructions for magnitude and sign-parity products.
typedef struct loom_spirv_product_row_t loom_spirv_product_row_t;

typedef struct loom_spirv_collective_product_t {
  // Immutable instructions for the native floating component.
  const loom_spirv_product_row_t* row;
  // Native floating operand and result register type.
  loom_type_t float_type;
  // Boolean sign and sign-parity register type.
  loom_type_t predicate_type;
  // Immediate name for the floating zero constant's bits.
  loom_string_id_t constant_key;
} loom_spirv_collective_product_t;

// Returns whether a native F16/F32 product has subgroup-uniform input. Such
// products keep the sign outside floating collective strength reduction.
bool loom_spirv_collective_product_needed(loom_low_lower_context_t* context,
                                          loom_combining_kind_t kind,
                                          loom_value_id_t source_value);
iree_status_t loom_spirv_collective_product_initialize(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    loom_spirv_collective_product_t* out_arithmetic);

// Emits one scalar component with identical magnitude and sign participation.
// Cluster attributes are empty except for a contiguous clustered reduction.
iree_status_t loom_spirv_collective_product_emit(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_collective_product_t* arithmetic,
    loom_spirv_collective_mode_t mode, loom_named_attr_slice_t attributes,
    loom_value_id_t value, loom_value_id_t* out_value);

iree_status_t loom_spirv_select_subgroup_product(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);
iree_status_t loom_spirv_emit_subgroup_product(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_COLLECTIVE_ARITHMETIC_H_
