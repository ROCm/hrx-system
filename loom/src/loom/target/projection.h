// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Authored target-op projection metadata.
//
// This boundary maps target-like operation attributes into immutable target
// facts. Facts-only consumers must not depend on it.

#ifndef LOOM_TARGET_PROJECTION_H_
#define LOOM_TARGET_PROJECTION_H_

#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_fact_projector_t loom_target_fact_projector_t;

enum loom_target_projection_value_bits_e {
  // Enum attr projected into a uint8_t target enum field.
  LOOM_TARGET_PROJECTION_VALUE_ENUM_U8 = 1,
  // I64 attr projected into a uint32_t field after verification.
  LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32 = 2,
  // I64 attr projected into a uint64_t field after verification.
  LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64 = 3,
  // String attr projected into an iree_string_view_t field.
  LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW = 4,
};
typedef uint8_t loom_target_projection_value_kind_t;

typedef struct loom_target_projection_t {
  // Byte offset into loom_target_bundle_storage_t for the destination field.
  uint16_t storage_offset;
  // Attribute index on the target-like op.
  uint8_t attr_index;
  // Target-neutral fact field receiving the projected value.
  loom_target_fact_field_t fact_field;
  // Projection operation used to copy the present attr payload.
  loom_target_projection_value_kind_t value_kind;
} loom_target_projection_t;

static_assert(sizeof(loom_target_projection_t) == 6,
              "loom_target_projection_t must be exactly 6 bytes");

typedef struct loom_target_like_descriptor_t {
  // Direct selector-indexed bundle table for a target-like op family.
  const loom_target_bundle_table_t* bundle_table;
  // Optional projection rows for typed attrs that override the selected bundle.
  const loom_target_projection_t* projections;
  // Number of entries in |projections|.
  uint8_t projection_count;
  // Static type of facts projected from this target-like op.
  const loom_target_fact_type_t* fact_type;
  // Optional family-owned projector for authored target-specific attributes.
  const loom_target_fact_projector_t* fact_projector;
} loom_target_like_descriptor_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PROJECTION_H_
