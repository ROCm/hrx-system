// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable storage relations and target-provided operation-pair recipes.

#ifndef LOOM_CODEGEN_LOW_PLACEMENT_PAIR_H_
#define LOOM_CODEGEN_LOW_PLACEMENT_PAIR_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_placement_relation_kind_bits_e {
  // Unknown or uninitialized placement relation kind.
  LOOM_LOW_PLACEMENT_RELATION_UNKNOWN = 0,
  // Result and source unit ranges should occupy identical storage units.
  LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE = 1,
  // Result units should occupy a subrange of the source storage units.
  LOOM_LOW_PLACEMENT_RELATION_SUBRANGE = 2,
  // Result units should occupy a contiguous packed range of source values.
  LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART = 3,
  // Result and source locations should differ under location_mask.
  LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION = 4,
  // Result and source unit ranges should occupy disjoint storage.
  LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE = 5,
} loom_low_placement_relation_kind_bits_t;
typedef uint8_t loom_low_placement_relation_kind_t;

// Sentinel for pair affinities without a placement recipe. Nonzero indexes are
// stored as recipe index + 1 so zero-initialized affinity rows stay
// recipe-free.
#define LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE 0

typedef enum loom_low_placement_pair_component_bits_e {
  // Unknown or uninitialized pair component.
  LOOM_LOW_PLACEMENT_PAIR_COMPONENT_UNKNOWN = 0,
  // First scheduled operation in the pair.
  LOOM_LOW_PLACEMENT_PAIR_COMPONENT_FIRST = 1,
  // Second scheduled operation in the pair.
  LOOM_LOW_PLACEMENT_PAIR_COMPONENT_SECOND = 2,
} loom_low_placement_pair_component_bits_t;
typedef uint8_t loom_low_placement_pair_component_t;

typedef enum loom_low_placement_pair_value_kind_bits_e {
  // Unknown or uninitialized pair value kind.
  LOOM_LOW_PLACEMENT_PAIR_VALUE_UNKNOWN = 0,
  // Operation operand selected by index.
  LOOM_LOW_PLACEMENT_PAIR_VALUE_OPERAND = 1,
  // Operation result selected by index.
  LOOM_LOW_PLACEMENT_PAIR_VALUE_RESULT = 2,
} loom_low_placement_pair_value_kind_bits_t;
typedef uint8_t loom_low_placement_pair_value_kind_t;

// One operation value and unit offset referenced by a pair recipe.
typedef struct loom_low_placement_pair_value_ref_t {
  // Pair component containing the value.
  loom_low_placement_pair_component_t component;
  // Whether index selects an operand or result.
  loom_low_placement_pair_value_kind_t kind;
  // Operand or result index within the selected operation.
  uint16_t index;
  // Allocation-unit offset within the selected value.
  uint16_t unit_offset;
} loom_low_placement_pair_value_ref_t;

// One target-provided location relation within a scheduled pair recipe.
typedef struct loom_low_placement_pair_relation_t {
  // First value participating in the relation.
  loom_low_placement_pair_value_ref_t result;
  // Second value participating in the relation.
  loom_low_placement_pair_value_ref_t source;
  // Number of contiguous allocation units covered by the relation.
  uint16_t unit_count;
  // Location relation applied to the selected values.
  loom_low_placement_relation_kind_t kind;
  // Low location bits compared by DIFFERENT_MASKED_LOCATION.
  uint32_t location_mask;
} loom_low_placement_pair_relation_t;

// Target-provided placement recipe shared by compatible descriptor pairs.
// Each alternative is a conjunction of |relation_count| rows. Alternatives
// are ordered by preference; placement selects the first one that is not
// structurally impossible for the concrete pair values.
typedef struct loom_low_placement_pair_recipe_t {
  // Borrowed relation rows grouped contiguously by alternative.
  const loom_low_placement_pair_relation_t* relations;
  // Number of relation rows in each alternative.
  uint16_t relation_count;
  // Number of ordered alternative relation conjunctions.
  uint16_t alternative_count;
  // Native packet count saved when one concrete use satisfies this recipe.
  uint16_t packet_savings;
} loom_low_placement_pair_recipe_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PLACEMENT_PAIR_H_
