// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Numeric transform encoding descriptor decoding.
//
// `#numeric_transform` is an encoding<transform> family used by register-level
// vector transforms and by higher-level storage schemas that reference those
// transforms. This helper decodes the descriptor shape once so verifiers,
// reference lowerings, and fact inference agree on family names, optional
// normalization, and dynamic operand parameters.

#ifndef LOOM_OPS_ENCODING_NUMERIC_TRANSFORM_H_
#define LOOM_OPS_ENCODING_NUMERIC_TRANSFORM_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Numeric transform family selected by the static `family` parameter.
typedef enum loom_encoding_numeric_transform_family_e {
  LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_UNKNOWN = 0,
  LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD = 1,
  LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD_SIGN = 2,
  LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD = 3,
  LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_JL_DENSE = 4,
} loom_encoding_numeric_transform_family_t;

// Scaling convention selected by the optional static `normalization` parameter.
typedef enum loom_encoding_numeric_transform_normalization_e {
  LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_NONE = 0,
  LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_ORTHONORMAL = 1,
} loom_encoding_numeric_transform_normalization_t;

// Static-or-dynamic element extent selected by a numeric transform.
typedef struct loom_encoding_numeric_transform_extent_t {
  // Positive static extent, or zero when not statically specified.
  int64_t static_value;

  // Dynamic index value carrying the extent, or invalid when absent.
  loom_value_id_t dynamic_value;
} loom_encoding_numeric_transform_extent_t;

// Decoded #numeric_transform descriptor.
typedef struct loom_encoding_numeric_transform_descriptor_t {
  // Static transform family.
  loom_encoding_numeric_transform_family_t family;

  // Static normalization convention. Defaults to none when omitted.
  loom_encoding_numeric_transform_normalization_t normalization;

  // Number of input elements transformed along the last vector axis.
  loom_encoding_numeric_transform_extent_t input_extent;

  // Number of output elements produced along the last vector axis.
  loom_encoding_numeric_transform_extent_t output_extent;

  // Optional dynamic i1 vector carrying per-lane negative-sign bits.
  loom_value_id_t signs;

  // Optional dynamic integer/index vector carrying per-lane source indices.
  loom_value_id_t permutation;

  // Optional dynamic floating-point matrix for dense projection transforms.
  loom_value_id_t matrix;

  // Optional dynamic seed for deterministic sign/permutation generation.
  loom_value_id_t seed;
} loom_encoding_numeric_transform_descriptor_t;

// Returns the canonical encoding family spelling.
iree_string_view_t loom_encoding_numeric_transform_name(void);

// Returns the parsed family for |name|, or UNKNOWN for unsupported names.
loom_encoding_numeric_transform_family_t
loom_encoding_numeric_transform_family_from_name(iree_string_view_t name);

// Returns true for transform families whose matrix is a Hadamard-like map over
// the last vector axis.
bool loom_encoding_numeric_transform_family_is_hadamard_like(
    loom_encoding_numeric_transform_family_t family);

// Returns the parsed normalization for |name|, or false for unsupported names.
bool loom_encoding_numeric_transform_normalization_from_name(
    iree_string_view_t name,
    loom_encoding_numeric_transform_normalization_t* out_normalization);

// Returns true when the descriptor carries each optional dynamic parameter.
bool loom_encoding_numeric_transform_has_signs(
    const loom_encoding_numeric_transform_descriptor_t* descriptor);
bool loom_encoding_numeric_transform_has_permutation(
    const loom_encoding_numeric_transform_descriptor_t* descriptor);
bool loom_encoding_numeric_transform_has_matrix(
    const loom_encoding_numeric_transform_descriptor_t* descriptor);
bool loom_encoding_numeric_transform_has_seed(
    const loom_encoding_numeric_transform_descriptor_t* descriptor);

// Evaluates the deterministic seed-derived sign bit used by hadamard_sign.
// Uses unsigned SplitMix64 wraparound on seed + input_index and returns false
// when the input lane is invalid.
bool loom_encoding_numeric_transform_seed_sign_bit(int64_t seed,
                                                   int64_t input_index,
                                                   bool* out_negate);

// Attempts to decode the numeric transform descriptor produced by |value_id|
// into |out_descriptor|. Returns false without modifying the output or
// emitting diagnostics for unsupported, non-local, or malformed producers.
// Callers use that conservative result at their own invariant boundary.
bool loom_encoding_numeric_transform_try_read_unverified_descriptor(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_encoding_numeric_transform_descriptor_t* out_descriptor);

// Attempts to decode a descriptor from verified IR. Valid non-local values
// still return false, while locally defined numeric transforms consume their
// established family and operand invariants without revalidation.
bool loom_encoding_numeric_transform_try_read_verified_descriptor(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_encoding_numeric_transform_descriptor_t* out_descriptor);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_NUMERIC_TRANSFORM_H_
