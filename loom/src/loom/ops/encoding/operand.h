// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Encoded-operand fact conversion.

#ifndef LOOM_OPS_ENCODING_OPERAND_H_
#define LOOM_OPS_ENCODING_OPERAND_H_

#include "iree/base/api.h"
#include "loom/ops/encoding/ops.h"
#include "loom/util/fact_table.h"
#include "loom/util/numeric_format.h"

#ifdef __cplusplus
extern "C" {
#endif

static_assert(LOOM_ENCODING_NUMERIC_FORMAT_COUNT_ <= 65,
              "numeric format enum must fit its 64-bit fact field");
static_assert(LOOM_ENCODING_PAYLOAD_PACKING_COUNT_ <= 33,
              "payload packing enum must fit its 32-bit fact field");
static_assert(LOOM_ENCODING_SCALE_TOPOLOGY_COUNT_ <= 33,
              "scale topology enum must fit its 32-bit fact field");
static_assert(LOOM_ENCODING_AFFINE_POLICY_COUNT_ <= 33,
              "affine policy enum must fit its 32-bit fact field");
static_assert(LOOM_ENCODING_ROUNDING_POLICY_COUNT_ <= 33,
              "rounding policy enum must fit its 32-bit fact field");
static_assert(LOOM_ENCODING_CODEBOOK_POLICY_COUNT_ <= 33,
              "codebook policy enum must fit its 32-bit fact field");
static_assert(LOOM_ENCODING_SPARSITY_POLICY_COUNT_ <= 33,
              "sparsity policy enum must fit its 32-bit fact field");

// Converts a dense descriptor enum ordinal to its one-hot fact value. Enum
// ordinal zero maps to the semantic none/unknown fact value zero.
static inline uint64_t loom_encoding_one_hot_enum_fact(uint8_t value) {
  return value == 0 ? 0 : UINT64_C(1) << (value - 1);
}

static inline loom_value_fact_numeric_format_flags_t
loom_encoding_numeric_format_fact(loom_encoding_numeric_format_t value) {
  return loom_encoding_one_hot_enum_fact((uint8_t)value);
}

static inline loom_value_fact_payload_packing_flags_t
loom_encoding_payload_packing_fact(loom_encoding_payload_packing_t value) {
  return (uint32_t)loom_encoding_one_hot_enum_fact((uint8_t)value);
}

static inline loom_value_fact_scale_topology_flags_t
loom_encoding_scale_topology_fact(loom_encoding_scale_topology_t value) {
  return (uint32_t)loom_encoding_one_hot_enum_fact((uint8_t)value);
}

static inline loom_value_fact_affine_policy_flags_t
loom_encoding_affine_policy_fact(loom_encoding_affine_policy_t value) {
  return (uint32_t)loom_encoding_one_hot_enum_fact((uint8_t)value);
}

static inline loom_value_fact_rounding_policy_flags_t
loom_encoding_rounding_policy_fact(loom_encoding_rounding_policy_t value) {
  if (value == LOOM_ENCODING_ROUNDING_POLICY_FINITE_FLUSH_SUBNORMAL) {
    return LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY |
           LOOM_VALUE_FACT_ROUNDING_POLICY_FLUSH_SUBNORMAL;
  }
  return (uint32_t)loom_encoding_one_hot_enum_fact((uint8_t)value);
}

static inline loom_value_fact_codebook_policy_flags_t
loom_encoding_codebook_policy_fact(loom_encoding_codebook_policy_t value) {
  return (uint32_t)loom_encoding_one_hot_enum_fact((uint8_t)value);
}

static inline loom_value_fact_sparsity_policy_flags_t
loom_encoding_sparsity_policy_fact(loom_encoding_sparsity_policy_t value) {
  return (uint32_t)loom_encoding_one_hot_enum_fact((uint8_t)value);
}

// Returns the authored enum keyword for an exact encoded-operand parameter fact
// value, or an empty string when the fact cannot be represented by one enum
// case. This is a cold reporting path; compilation consumes numeric facts.
iree_string_view_t loom_encoding_operand_fact_name(
    loom_encoding_operand_parameter_t parameter, uint64_t value);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_OPERAND_H_
