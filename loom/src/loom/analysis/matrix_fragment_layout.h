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

typedef enum loom_matrix_fragment_role_layout_flag_bits_e {
  // Lane xor 1 advances the column by one while preserving other coordinates.
  LOOM_MATRIX_FRAGMENT_ROLE_LAYOUT_FLAG_CONTIGUOUS_LANE_XOR1_COLUMNS = 1u << 0,
} loom_matrix_fragment_role_layout_flag_bits_t;

// Bitset of loom_matrix_fragment_role_layout_flag_bits_t values.
typedef uint16_t loom_matrix_fragment_role_layout_flags_t;

// Canonical semantic axes carried by matrix fragment layouts. The ordering is
// also the row-major delinearization order for per-lane payload elements.
typedef enum loom_matrix_fragment_axis_e {
  LOOM_MATRIX_FRAGMENT_AXIS_BLOCK = 0,
  LOOM_MATRIX_FRAGMENT_AXIS_ROW = 1,
  LOOM_MATRIX_FRAGMENT_AXIS_COLUMN = 2,
  LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION = 3,
  LOOM_MATRIX_FRAGMENT_AXIS_COUNT = 4,
} loom_matrix_fragment_axis_t;

// Factorization of one semantic axis across lane-local payload elements and
// subgroup lanes. The semantic coordinate is:
//
//   element + element_count *
//       (thread + thread_count * outer)
//
// Payload element indices are row-major across all element factors and then
// all outer factors. The lane contribution is
// `(lane / thread_stride) % thread_count`.
typedef struct loom_matrix_fragment_axis_layout_t {
  // Outer lane-local factor represented by discontiguous payload elements.
  uint16_t outer_count;
  // Cross-lane factor distributed across the subgroup.
  uint16_t thread_count;
  // Lane-id stride between adjacent cross-lane coordinates.
  uint16_t thread_stride;
  // Inner lane-local factor represented by contiguous payload elements.
  uint16_t element_count;
} loom_matrix_fragment_axis_layout_t;

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

typedef struct loom_matrix_fragment_role_layout_t {
  // Contract operand role described by this role layout.
  loom_contract_operand_role_t role;
  // Number of 32-bit payload registers held by each participating lane.
  uint16_t register_count;
  // Bit width of each logical scalar element.
  uint16_t element_bit_count;
  // Number of scalar elements in the role's source-level payload vector.
  uint16_t payload_element_count;
  // First payload element that carries a distinct logical coordinate.
  uint16_t coordinate_element_offset;
  // Payload element stride between distinct logical coordinates.
  uint16_t coordinate_element_stride;
  // Properties proven for the complete role layout during table generation.
  loom_matrix_fragment_role_layout_flags_t flags;
  // Coordinate axes produced by this role layout.
  loom_matrix_fragment_coordinate_flags_t coordinate_flags;
  // Physical-to-logical grouping for a compressed reduction axis.
  struct {
    // Number of elements physically stored in each reduction group.
    uint16_t storage_element_count;

    // Number of logical reduction elements represented by each group.
    uint16_t logical_element_count;
  } reduction_group;
  // Semantic factorization indexed by loom_matrix_fragment_axis_t.
  loom_matrix_fragment_axis_layout_t axes[LOOM_MATRIX_FRAGMENT_AXIS_COUNT];
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
} loom_matrix_fragment_layout_t;

typedef struct loom_matrix_fragment_coordinate_t {
  // Coordinate axes populated for this role.
  loom_matrix_fragment_coordinate_flags_t coordinate_flags;
  // Independent block/batch coordinate when BLOCK is set.
  uint16_t block;
  // M/result-row coordinate when ROW is set.
  uint16_t row;
  // N/result-column coordinate when COLUMN is set.
  uint16_t column;
  // K/reduction coordinate when REDUCTION is set.
  uint16_t reduction;
} loom_matrix_fragment_coordinate_t;

typedef struct loom_matrix_fragment_physical_element_t {
  // Subgroup lane that owns or replicates the logical coordinate.
  uint16_t lane;
  // Scalar element ordinal in the source-level payload vector.
  uint16_t payload_element_index;
} loom_matrix_fragment_physical_element_t;

// Returns the role layout within |layout|, or NULL when the role is not
// modeled.
const loom_matrix_fragment_role_layout_t* loom_matrix_fragment_role_layout(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role);

// Maps a lane-local payload element to a logical matrix coordinate. Returns
// false when |layout| is absent, the role is unmodeled, the lane/payload
// element is outside the layout domain or names padding, or runtime metadata
// is required to expand a compressed reduction coordinate.
bool loom_matrix_fragment_coordinate(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, uint16_t lane,
    uint16_t payload_element_index,
    loom_matrix_fragment_coordinate_t* out_coordinate);

// Maps a lane-local payload element using an already-selected role
// layout. This is useful for lowering loops that walk a role layout once and
// then enumerate all payload elements. Returns false when |layout| or
// |role_layout| is absent, or the lane/payload element is outside the layout
// domain.
bool loom_matrix_fragment_coordinate_from_role_layout(
    const loom_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout, uint16_t lane,
    uint16_t payload_element_index,
    loom_matrix_fragment_coordinate_t* out_coordinate);

// Returns true when the generated layout properties prove that every even/odd
// lane pair selected by lane xor 1 carries adjacent columns for each payload
// element in |role|. This is the reusable contract used by packed epilogues
// that let one lane store two horizontally adjacent logical elements.
bool loom_matrix_fragment_role_has_contiguous_lane_xor1_columns(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role);

// Counts physical lane/payload elements that carry |coordinate| for |role|.
// Some target fragment layouts intentionally replicate input operands across
// lanes; callers that need a canonical owner can request occurrence zero from
// loom_matrix_fragment_physical_element().
bool loom_matrix_fragment_physical_element_count(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role,
    loom_matrix_fragment_coordinate_t coordinate, uint16_t* out_count);

// Returns the |occurrence_index|-th physical lane/payload element that carries
// |coordinate| for |role|. occurrence_index is zero-based and follows ascending
// lane and payload-element order. Returns false when the role is unmodeled,
// the coordinate does not match that role's axes, or the occurrence is absent.
bool loom_matrix_fragment_physical_element(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role,
    loom_matrix_fragment_coordinate_t coordinate, uint16_t occurrence_index,
    loom_matrix_fragment_physical_element_t* out_element);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_MATRIX_FRAGMENT_LAYOUT_H_
