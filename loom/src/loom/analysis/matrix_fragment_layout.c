// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/matrix_fragment_layout.h"

static bool loom_matrix_fragment_coordinate_matches(
    loom_matrix_fragment_coordinate_t lhs,
    loom_matrix_fragment_coordinate_t rhs) {
  if (lhs.coordinate_flags != rhs.coordinate_flags) {
    return false;
  }
  if (iree_any_bit_set(lhs.coordinate_flags,
                       LOOM_MATRIX_FRAGMENT_COORDINATE_ROW) &&
      lhs.row != rhs.row) {
    return false;
  }
  if (iree_any_bit_set(lhs.coordinate_flags,
                       LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN) &&
      lhs.column != rhs.column) {
    return false;
  }
  if (iree_any_bit_set(lhs.coordinate_flags,
                       LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION) &&
      lhs.reduction != rhs.reduction) {
    return false;
  }
  return true;
}

const loom_matrix_fragment_role_layout_t* loom_matrix_fragment_role_layout(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role) {
  if (layout == NULL) {
    return NULL;
  }
  switch (role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
      return &layout->lhs;
    case LOOM_CONTRACT_OPERAND_ROLE_RHS:
      return &layout->rhs;
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
      return &layout->accumulator;
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT:
      return &layout->result;
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return NULL;
  }
}

bool loom_matrix_fragment_coordinate(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, uint16_t lane, uint16_t register_index,
    uint16_t element_index, loom_matrix_fragment_coordinate_t* out_coordinate) {
  IREE_ASSERT_ARGUMENT(out_coordinate);

  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, role);
  return loom_matrix_fragment_coordinate_from_role_layout(
      layout, role_layout, lane, register_index, element_index, out_coordinate);
}

bool loom_matrix_fragment_coordinate_from_role_layout(
    const loom_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout, uint16_t lane,
    uint16_t register_index, uint16_t element_index,
    loom_matrix_fragment_coordinate_t* out_coordinate) {
  IREE_ASSERT_ARGUMENT(out_coordinate);

  *out_coordinate = (loom_matrix_fragment_coordinate_t){0};
  if (layout == NULL || role_layout == NULL || layout->wave_size == 0 ||
      lane >= layout->wave_size ||
      register_index >= role_layout->register_count ||
      element_index >= role_layout->elements_per_register) {
    return false;
  }

  const loom_matrix_fragment_tile_shape_t tile_shape = layout->tile_shape;
  if (tile_shape.result_row_count == 0 || tile_shape.result_column_count == 0 ||
      tile_shape.reduction_count == 0) {
    return false;
  }

  out_coordinate->coordinate_flags = role_layout->coordinate_flags;
  switch (role_layout->map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION: {
      const uint32_t reduction =
          (uint32_t)register_index * role_layout->elements_per_register +
          element_index;
      if (reduction >= tile_shape.reduction_count) {
        return false;
      }
      out_coordinate->row = lane % tile_shape.result_row_count;
      out_coordinate->reduction = (uint16_t)reduction;
      return true;
    }
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION: {
      const uint32_t reduction =
          (uint32_t)register_index * role_layout->elements_per_register +
          element_index;
      if (reduction >= tile_shape.reduction_count) {
        return false;
      }
      out_coordinate->column = lane % tile_shape.result_column_count;
      out_coordinate->reduction = (uint16_t)reduction;
      return true;
    }
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN: {
      const uint32_t row =
          (uint32_t)register_index * 2u + lane / tile_shape.result_column_count;
      if (row >= tile_shape.result_row_count) {
        return false;
      }
      out_coordinate->row = (uint16_t)row;
      out_coordinate->column = lane % tile_shape.result_column_count;
      return true;
    }
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION: {
      const uint32_t lane_group = lane / tile_shape.result_row_count;
      const uint32_t reduction =
          lane_group * role_layout->register_count *
              role_layout->elements_per_register +
          (uint32_t)register_index * role_layout->elements_per_register +
          element_index;
      if (reduction >= tile_shape.reduction_count) {
        return false;
      }
      out_coordinate->row = lane % tile_shape.result_row_count;
      out_coordinate->reduction = (uint16_t)reduction;
      return true;
    }
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION: {
      const uint32_t lane_group = lane / tile_shape.result_column_count;
      const uint32_t reduction =
          lane_group * role_layout->register_count *
              role_layout->elements_per_register +
          (uint32_t)register_index * role_layout->elements_per_register +
          element_index;
      if (reduction >= tile_shape.reduction_count) {
        return false;
      }
      out_coordinate->column = lane % tile_shape.result_column_count;
      out_coordinate->reduction = (uint16_t)reduction;
      return true;
    }
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN: {
      const uint32_t row = (uint32_t)(lane / tile_shape.result_column_count) *
                               role_layout->register_count +
                           register_index;
      if (row >= tile_shape.result_row_count) {
        return false;
      }
      out_coordinate->row = (uint16_t)row;
      out_coordinate->column = lane % tile_shape.result_column_count;
      return true;
    }
    case LOOM_MATRIX_FRAGMENT_MAP_UNKNOWN:
    default:
      return false;
  }
}

bool loom_matrix_fragment_physical_element_count(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role,
    loom_matrix_fragment_coordinate_t coordinate, uint16_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  *out_count = 0;

  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, role);
  if (layout == NULL || role_layout == NULL ||
      coordinate.coordinate_flags != role_layout->coordinate_flags) {
    return false;
  }

  for (uint16_t lane = 0; lane < layout->wave_size; ++lane) {
    for (uint16_t register_index = 0;
         register_index < role_layout->register_count; ++register_index) {
      for (uint16_t element_index = 0;
           element_index < role_layout->elements_per_register;
           ++element_index) {
        loom_matrix_fragment_coordinate_t candidate = {0};
        if (!loom_matrix_fragment_coordinate_from_role_layout(
                layout, role_layout, lane, register_index, element_index,
                &candidate)) {
          continue;
        }
        if (!loom_matrix_fragment_coordinate_matches(candidate, coordinate)) {
          continue;
        }
        if (*out_count == UINT16_MAX) {
          return false;
        }
        ++(*out_count);
      }
    }
  }
  return *out_count != 0;
}

bool loom_matrix_fragment_physical_element(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role,
    loom_matrix_fragment_coordinate_t coordinate, uint16_t occurrence_index,
    loom_matrix_fragment_physical_element_t* out_element) {
  IREE_ASSERT_ARGUMENT(out_element);
  *out_element = (loom_matrix_fragment_physical_element_t){0};

  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, role);
  if (layout == NULL || role_layout == NULL ||
      coordinate.coordinate_flags != role_layout->coordinate_flags) {
    return false;
  }

  uint16_t current_occurrence = 0;
  for (uint16_t lane = 0; lane < layout->wave_size; ++lane) {
    for (uint16_t register_index = 0;
         register_index < role_layout->register_count; ++register_index) {
      for (uint16_t element_index = 0;
           element_index < role_layout->elements_per_register;
           ++element_index) {
        loom_matrix_fragment_coordinate_t candidate = {0};
        if (!loom_matrix_fragment_coordinate_from_role_layout(
                layout, role_layout, lane, register_index, element_index,
                &candidate)) {
          continue;
        }
        if (!loom_matrix_fragment_coordinate_matches(candidate, coordinate)) {
          continue;
        }
        if (current_occurrence == occurrence_index) {
          *out_element = (loom_matrix_fragment_physical_element_t){
              .lane = lane,
              .register_index = register_index,
              .element_index = element_index,
          };
          return true;
        }
        ++current_occurrence;
      }
    }
  }
  return false;
}
