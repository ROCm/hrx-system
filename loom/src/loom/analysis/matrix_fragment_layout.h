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
  // Coordinate carries an M/result-row value.
  LOOM_MATRIX_FRAGMENT_COORDINATE_ROW = 1u << 0,
  // Coordinate carries an N/result-column value.
  LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN = 1u << 1,
  // Coordinate carries a K/reduction value.
  LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION = 1u << 2,
} loom_matrix_fragment_coordinate_flag_bits_t;

// Bitset of loom_matrix_fragment_coordinate_flag_bits_t values.
typedef uint32_t loom_matrix_fragment_coordinate_flags_t;

typedef enum loom_matrix_fragment_map_kind_e {
  // No lane/register coordinate formula is defined.
  LOOM_MATRIX_FRAGMENT_MAP_UNKNOWN = 0,
  // Row is lane mod M; reduction is packed by register element.
  LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION = 1,
  // Column is lane mod N; reduction is packed by register element.
  LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION = 2,
  // Row is register-interleaved by the lane group; column is lane mod N.
  LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN = 3,
  // Row is lane mod M; reduction is packed by lane group and register element.
  LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION = 4,
  // Column is lane mod N; reduction is packed by lane group and register
  // element.
  LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION = 5,
  // Row is register-local within a lane group; column is lane mod N.
  LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN = 6,
} loom_matrix_fragment_map_kind_t;

typedef struct loom_matrix_fragment_tile_shape_t {
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
  // Formula used to map lane/register elements into logical coordinates.
  loom_matrix_fragment_map_kind_t map_kind;
  // Number of 32-bit payload registers held by each participating lane.
  uint16_t register_count;
  // Number of logical scalar elements packed into each payload register.
  uint16_t elements_per_register;
  // Bit width of each logical scalar element.
  uint16_t element_bit_count;
  // Coordinate axes produced by this role layout.
  loom_matrix_fragment_coordinate_flags_t coordinate_flags;
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
  // Lane-local 32-bit payload register ordinal.
  uint16_t register_index;
  // Logical scalar element ordinal within register_index.
  uint16_t element_index;
} loom_matrix_fragment_physical_element_t;

// Returns the role layout within |layout|, or NULL when the role is not
// modeled.
const loom_matrix_fragment_role_layout_t* loom_matrix_fragment_role_layout(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role);

// Maps a lane-local payload register element to a logical matrix coordinate.
// Returns false when |layout| is absent, the role is unmodeled, or the
// lane/register element is outside the layout domain.
bool loom_matrix_fragment_coordinate(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, uint16_t lane, uint16_t register_index,
    uint16_t element_index, loom_matrix_fragment_coordinate_t* out_coordinate);

// Maps a lane-local payload register element using an already-selected role
// layout. This is useful for lowering loops that walk a role layout once and
// then enumerate all lane/register elements. Returns false when |layout| or
// |role_layout| is absent, or the lane/register element is outside the layout
// domain.
bool loom_matrix_fragment_coordinate_from_role_layout(
    const loom_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout, uint16_t lane,
    uint16_t register_index, uint16_t element_index,
    loom_matrix_fragment_coordinate_t* out_coordinate);

// Counts physical lane/register elements that carry |coordinate| for |role|.
// Some target fragment layouts intentionally replicate input operands across
// lanes; callers that need a canonical owner can request occurrence zero from
// loom_matrix_fragment_physical_element().
bool loom_matrix_fragment_physical_element_count(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role,
    loom_matrix_fragment_coordinate_t coordinate, uint16_t* out_count);

// Returns the |occurrence_index|-th physical lane/register element that carries
// |coordinate| for |role|. occurrence_index is zero-based and follows ascending
// lane, register, and element order. Returns false when the role is unmodeled,
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
