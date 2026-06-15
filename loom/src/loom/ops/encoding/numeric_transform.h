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

// Conservative decode result. Non-OK results let each consumer choose the
// right behavior: verifier silence, unknown facts, or lowering failure.
typedef enum loom_encoding_numeric_transform_read_code_e {
  // Descriptor was decoded successfully.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK = 0,
  // The transform SSA value is not in the module value table.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_VALUE_OUT_OF_RANGE = 1,
  // The transform SSA value is a block argument or otherwise non-local.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NOT_LOCALLY_DEFINED = 2,
  // The transform SSA value is not produced by encoding.define.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NOT_ENCODING_DEFINE = 3,
  // The encoding.define spec is not #numeric_transform.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NOT_NUMERIC_TRANSFORM = 4,
  // The required static family parameter is absent.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_MISSING_FAMILY = 5,
  // The family parameter exists but is not a static string.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NON_STRING_FAMILY = 6,
  // The family string is not part of the supported family vocabulary.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_UNKNOWN_FAMILY = 7,
  // The normalization parameter exists but is not a static string.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NON_STRING_NORMALIZATION = 8,
  // The normalization string is not part of the supported vocabulary.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_UNKNOWN_NORMALIZATION = 9,
  // A dynamic parameter name does not map to a valid operand ordinal.
  LOOM_ENCODING_NUMERIC_TRANSFORM_READ_MALFORMED_DYNAMIC_PARAM = 10,
} loom_encoding_numeric_transform_read_code_t;

// Decoded #numeric_transform descriptor.
typedef struct loom_encoding_numeric_transform_descriptor_t {
  // Static transform family.
  loom_encoding_numeric_transform_family_t family;

  // Static normalization convention. Defaults to none when omitted.
  loom_encoding_numeric_transform_normalization_t normalization;

  // Optional dynamic i1 vector carrying per-lane negative-sign bits.
  loom_value_id_t signs;

  // Optional dynamic integer/index vector carrying per-lane source indices.
  loom_value_id_t permutation;

  // Optional dynamic floating-point matrix for dense projection transforms.
  loom_value_id_t matrix;

  // Optional dynamic seed for deterministic sign/permutation generation.
  loom_value_id_t seed;
} loom_encoding_numeric_transform_descriptor_t;

// Descriptor read result including the offending parameter when applicable.
typedef struct loom_encoding_numeric_transform_read_t {
  // Decoded descriptor. Only complete when code is READ_OK.
  loom_encoding_numeric_transform_descriptor_t descriptor;

  // Decode result category.
  loom_encoding_numeric_transform_read_code_t code;

  // Parameter name associated with code, or empty when no single parameter
  // caused the result.
  iree_string_view_t parameter_name;
} loom_encoding_numeric_transform_read_t;

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

// Decodes the numeric transform descriptor produced by |value_id|. This never
// emits diagnostics or returns iree_status_t: unsupported/non-local/malformed
// cases are ordinary query results that consumers interpret at their own
// invariant boundary.
loom_encoding_numeric_transform_read_t
loom_encoding_numeric_transform_read_descriptor(const loom_module_t* module,
                                                loom_value_id_t value_id);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_NUMERIC_TRANSFORM_H_
