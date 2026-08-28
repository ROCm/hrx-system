// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent matrix fragment lane/register layout facts.
//
// These records describe how a target-shaped matrix fragment payload maps from
// subgroup lanes and lane-local payload registers back to logical M/N/K
// coordinates. Targets own their descriptor identities and numeric legality,
// while this layer owns the reusable coordinate formulas that reference
// legalization, diagnostics, tests, and target lowering all need to agree on.

#ifndef LOOM_ANALYSIS_MATRIX_FRAGMENT_LAYOUT_H_
#define LOOM_ANALYSIS_MATRIX_FRAGMENT_LAYOUT_H_

#include "iree/base/api.h"
#include "loom/analysis/contract_roles.h"
#include "loom/analysis/native_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_matrix_fragment_coordinate_flag_bits_e {
  // Coordinate carries an independent block or batch value.
  LOOM_MATRIX_FRAGMENT_COORDINATE_BLOCK = 1u << 0,
  // Coordinate carries an M/result-row value.
  LOOM_MATRIX_FRAGMENT_COORDINATE_ROW = 1u << 1,
  // Coordinate carries an N/result-column value.
  LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN = 1u << 2,
  // Coordinate carries a K/reduction value.
  LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION = 1u << 3,
} loom_matrix_fragment_coordinate_flag_bits_t;

// Bitset of loom_matrix_fragment_coordinate_flag_bits_t values.
typedef uint32_t loom_matrix_fragment_coordinate_flags_t;

// Canonical semantic axes carried by matrix fragment layouts. The ordering is
// also the row-major delinearization order for per-lane payload elements.
typedef enum loom_matrix_fragment_axis_e {
  LOOM_MATRIX_FRAGMENT_AXIS_BLOCK = 0,
  LOOM_MATRIX_FRAGMENT_AXIS_ROW = 1,
  LOOM_MATRIX_FRAGMENT_AXIS_COLUMN = 2,
  LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION = 3,
  LOOM_MATRIX_FRAGMENT_AXIS_COUNT = 4,
} loom_matrix_fragment_axis_t;

// Named physical and semantic dimensions used by generated coordinate plans.
typedef enum loom_matrix_fragment_coordinate_dimension_e {
  LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT = 0,
  LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE = 1,
  LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_BLOCK = 2,
  LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW = 3,
  LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN = 4,
  LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_REDUCTION = 5,
  LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT = 6,
} loom_matrix_fragment_coordinate_dimension_t;

// One mixed-radix term in a generated coordinate projection. Evaluation adds
// `((source / source_divisor) % source_modulus) * destination_multiplier`.
// A zero source modulus omits the remainder operation.
typedef struct loom_matrix_fragment_coordinate_projection_term_t {
  // Dimension from which the source digit is extracted.
  uint8_t source_dimension;
  // Dimension into which the scaled digit is accumulated.
  uint8_t destination_dimension;
  // Divisor applied to the source coordinate before optional reduction.
  uint16_t source_divisor;
  // Modulus applied to the quotient, or zero to retain the full quotient.
  uint16_t source_modulus;
  // Multiplier applied before accumulating into the destination coordinate.
  uint16_t destination_multiplier;
} loom_matrix_fragment_coordinate_projection_term_t;

// Direct generated projection for one fixed fragment role. Forward terms map
// participant/value coordinates to stored role coordinates. Inverse terms map
// stored coordinates to the canonical participant/value owner. For compressed
// roles, the reduction coordinate names the physical storage group; runtime
// metadata owns its expansion into logical reduction positions. The terms
// array stores the forward slice followed by the inverse slice.
typedef struct loom_matrix_fragment_coordinate_projection_plan_t {
  // Concatenated forward and inverse coordinate terms.
  const loom_matrix_fragment_coordinate_projection_term_t* terms;
  // Number of forward terms at the beginning of |terms|.
  uint8_t forward_term_count;
  // Number of inverse terms following the forward terms.
  uint8_t inverse_term_count;
} loom_matrix_fragment_coordinate_projection_plan_t;

typedef struct loom_matrix_fragment_tile_shape_t {
  // Independent matrix blocks computed by one target-native instruction.
  uint16_t block_count;
  // Contracted result rows in the target-native tile.
  uint16_t result_row_count;
  // Contracted result columns in the target-native tile.
  uint16_t result_column_count;
  // Contracted K depth consumed by one target-native instruction.
  uint16_t reduction_count;
} loom_matrix_fragment_tile_shape_t;

// Exact executable source-owner projection for one packed B16 publication.
// Participants matching the AND/equality predicate own the low-axis elements
// and publish one 32-bit packet. The participant selected by xor owns the
// adjacent high-axis element at the same participant-local payload position. A
// zero AND mask means that this packed publication is unavailable.
typedef struct loom_matrix_fragment_packed_b16_publication_t {
  // Participant bit mask selecting the packet publishers.
  uint16_t publishing_participant_and_mask;
  // Participant value selected after applying the publisher bit mask.
  uint16_t publishing_participant_equal_value;
  // XOR mask selecting the adjacent high-axis source participant.
  uint16_t paired_participant_xor_mask;
} loom_matrix_fragment_packed_b16_publication_t;

typedef struct loom_matrix_fragment_role_layout_t {
  // Contract operand role described by this role layout.
  loom_contract_operand_role_t role;
  // Number of 32-bit payload registers held by each participating lane.
  uint16_t register_count;
  // Bit width of each logical scalar element. Elements are densely packed
  // across the role's register bitstream and may straddle register boundaries.
  uint16_t element_bit_count;
  // Number of scalar elements in the role's source-level payload vector.
  uint16_t payload_element_count;
  // Number of distinct stored coordinates represented by the payload.
  uint16_t coordinate_element_count;
  // Reserved zero-valued storage for future generated role facts.
  uint16_t reserved;
  // Payload element stride between distinct logical coordinates.
  uint16_t coordinate_element_stride;
  // Exact packed-B16 publication projections derived for this role.
  struct {
    // Projection pairing adjacent result rows.
    loom_matrix_fragment_packed_b16_publication_t row;
    // Projection pairing adjacent result columns.
    loom_matrix_fragment_packed_b16_publication_t column;
  } packed_b16_publications;
  // Coordinate axes produced by this role layout.
  loom_matrix_fragment_coordinate_flags_t coordinate_flags;
  // Single coordinate flag for the axis densely packed within each register,
  // or zero when elements are not densely packed along one semantic axis.
  loom_matrix_fragment_coordinate_flags_t packed_element_coordinate_flag;
  // Physical-to-logical grouping for a compressed reduction axis.
  struct {
    // Number of elements physically stored in each reduction group.
    uint16_t storage_element_count;

    // Number of logical reduction elements represented by each group.
    uint16_t logical_element_count;
  } reduction_group;
  // Direct generated storage-coordinate plan.
  const loom_matrix_fragment_coordinate_projection_plan_t*
      coordinate_projection_plan;
} loom_matrix_fragment_role_layout_t;

typedef struct loom_matrix_fragment_layout_t {
  // Stable owner-defined layout kind. Zero means unknown or absent.
  uint32_t kind;
  // Stable layout name used by diagnostics and tests.
  iree_string_view_t name;
  // Subgroup or wave size for which lane formulas are defined.
  uint16_t wave_size;
  // Logical tile shape covered by the layout.
  loom_matrix_fragment_tile_shape_t tile_shape;
  // Matrix A source role layout.
  loom_matrix_fragment_role_layout_t lhs;
  // Matrix B source role layout.
  loom_matrix_fragment_role_layout_t rhs;
  // Matrix C accumulator input role layout.
  loom_matrix_fragment_role_layout_t accumulator;
  // Matrix D result role layout.
  loom_matrix_fragment_role_layout_t result;
  // Generated compact placement facts for this native contraction layout.
  const loom_native_contraction_facts_t* native_contraction_facts;
} loom_matrix_fragment_layout_t;

// Applies a trusted generated coordinate projection term slice. |out_terms|
// is cleared before the terms are accumulated.
void loom_matrix_fragment_apply_coordinate_projection(
    const loom_matrix_fragment_coordinate_projection_term_t* terms,
    uint8_t term_count,
    const uint32_t
        source_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT],
    uint32_t out_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT]);

// Returns the coordinate dimension corresponding to one semantic axis.
loom_matrix_fragment_coordinate_dimension_t
loom_matrix_fragment_axis_coordinate_dimension(
    loom_matrix_fragment_axis_t axis);

// Returns the semantic axis corresponding to |dimension|, or
// LOOM_MATRIX_FRAGMENT_AXIS_COUNT for a physical coordinate dimension.
loom_matrix_fragment_axis_t loom_matrix_fragment_coordinate_dimension_axis(
    loom_matrix_fragment_coordinate_dimension_t dimension);

// Returns the role layout within |layout|, or NULL when the role is not
// modeled.
const loom_matrix_fragment_role_layout_t* loom_matrix_fragment_role_layout(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_MATRIX_FRAGMENT_LAYOUT_H_
