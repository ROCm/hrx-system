// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact target-independent facts derived from native coordinate maps.
//
// Exact finite maps are generation-time proofs. These immutable records retain
// the placement and source-owner relations needed by shipping compiler policy
// and compile reports without carrying or enumerating either finite domain.

#ifndef LOOM_ANALYSIS_NATIVE_LAYOUT_H_
#define LOOM_ANALYSIS_NATIVE_LAYOUT_H_

#include "iree/base/api.h"
#include "loom/analysis/contract_roles.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_native_layout_evidence_t;
enum loom_native_layout_evidence_e {
  // No native layout evidence is present.
  LOOM_NATIVE_LAYOUT_EVIDENCE_NONE = 0,
  // A finite map proves every physical and logical coordinate relationship.
  LOOM_NATIVE_LAYOUT_EVIDENCE_EXACT = 1,
  // Runtime metadata completes a statically known physical grouping.
  LOOM_NATIVE_LAYOUT_EVIDENCE_METADATA_DEPENDENT = 2,
  // A symbolic parameter controls one or more layout extents.
  LOOM_NATIVE_LAYOUT_EVIDENCE_PARAMETRIC = 3,
  // The target carrier does not expose its physical coordinate layout.
  LOOM_NATIVE_LAYOUT_EVIDENCE_OPAQUE = 4,
};

// Exact logical shape computed by one target-native contraction primitive.
typedef struct loom_native_contraction_shape_t {
  // Number of independent matrix blocks.
  uint32_t block_count;
  // Number of result rows in each block.
  uint32_t result_row_count;
  // Number of result columns in each block.
  uint32_t result_column_count;
  // Reduction depth consumed by the primitive.
  uint32_t reduction_count;
} loom_native_contraction_shape_t;

// Compact physical placement facts for one semantic contraction role.
typedef struct loom_native_contraction_role_facts_t {
  // Semantic contraction role described by this placement.
  loom_contract_operand_role_t role;
  // Strength of the retained placement evidence.
  loom_native_layout_evidence_t evidence;
  // Bit width of one logical payload element.
  uint16_t element_bit_count;
  // Number of target registers in one participant's payload.
  uint16_t register_count;
  // Number of source-level payload elements held by one participant.
  uint16_t payload_element_count;
  // Number of physical coordinate-bearing positions across all participants.
  uint32_t physical_position_count;
  // Number of logical coordinates represented by the role.
  uint32_t logical_coordinate_count;
  // Minimum exact physical owner count, or zero for non-exact evidence.
  uint16_t owner_multiplicity_minimum;
  // Maximum exact physical owner count, or zero for non-exact evidence.
  uint16_t owner_multiplicity_maximum;
} loom_native_contraction_role_facts_t;

// Compact placement facts for one target-native contraction primitive.
typedef struct loom_native_contraction_facts_t {
  // Exact logical primitive shape.
  loom_native_contraction_shape_t shape;
  // Number of physical execution participants. Targets with no explicit
  // participant axis use one.
  uint16_t participant_count;
  // Physical placement of the left-hand source role.
  loom_native_contraction_role_facts_t lhs;
  // Physical placement of the right-hand source role.
  loom_native_contraction_role_facts_t rhs;
  // Physical placement of the accumulator input role.
  loom_native_contraction_role_facts_t accumulator;
  // Physical placement of the result role.
  loom_native_contraction_role_facts_t result;
} loom_native_contraction_facts_t;

typedef uint8_t loom_native_physical_dimension_t;
enum loom_native_physical_dimension_e {
  // No physical coordinate dimension.
  LOOM_NATIVE_PHYSICAL_DIMENSION_NONE = 0,
  // Execution participant coordinate, such as a subgroup lane.
  LOOM_NATIVE_PHYSICAL_DIMENSION_PARTICIPANT = 1,
  // Participant-local payload position coordinate.
  LOOM_NATIVE_PHYSICAL_DIMENSION_POSITION = 2,
};

// One destination-coordinate digit contributing to a source owner coordinate.
// Evaluation adds `((destination / divisor) % modulus) * multiplier` to the
// selected source-owner dimension. A zero modulus retains the full quotient.
typedef struct loom_native_transition_owner_factor_t {
  // Destination physical dimension from which the digit is extracted.
  loom_native_physical_dimension_t destination_dimension;
  // Source-owner physical dimension receiving the scaled digit.
  loom_native_physical_dimension_t source_owner_dimension;
  // Divisor applied to the destination coordinate.
  uint32_t destination_divisor;
  // Modulus applied to the quotient, or zero to retain the full quotient.
  uint32_t destination_modulus;
  // Multiplier applied before accumulating into the source-owner coordinate.
  uint32_t source_owner_multiplier;
} loom_native_transition_owner_factor_t;

// Exact compact source-owner movement for one native layout transition.
typedef struct loom_native_transition_facts_t {
  // Semantic role providing the source physical payload.
  loom_contract_operand_role_t source_role;
  // Semantic role required by the destination physical payload.
  loom_contract_operand_role_t destination_role;
  // Number of destination physical positions materialized by the transition.
  uint32_t destination_position_count;
  // Destination positions whose selected source has another participant owner.
  uint32_t participant_change_count;
  // Destination positions whose selected source has another local position.
  uint32_t local_position_change_count;
  // Minimum destination replication count for one selected source position.
  uint16_t destination_positions_per_source_minimum;
  // Maximum destination replication count for one selected source position.
  uint16_t destination_positions_per_source_maximum;
  // Generated destination-to-source-owner factor terms.
  const loom_native_transition_owner_factor_t* source_owner_factors;
  // Number of generated source-owner factor terms.
  uint8_t source_owner_factor_count;
} loom_native_transition_facts_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_NATIVE_LAYOUT_H_
