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
                       LOOM_MATRIX_FRAGMENT_COORDINATE_BLOCK) &&
      lhs.block != rhs.block) {
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

static loom_matrix_fragment_coordinate_flags_t
loom_matrix_fragment_axis_coordinate_flag(loom_matrix_fragment_axis_t axis) {
  IREE_ASSERT_LT(axis, LOOM_MATRIX_FRAGMENT_AXIS_COUNT);
  return 1u << axis;
}

static uint16_t loom_matrix_fragment_axis_extent(
    const loom_matrix_fragment_tile_shape_t* tile_shape,
    loom_matrix_fragment_axis_t axis) {
  switch (axis) {
    case LOOM_MATRIX_FRAGMENT_AXIS_BLOCK:
      return tile_shape->block_count;
    case LOOM_MATRIX_FRAGMENT_AXIS_ROW:
      return tile_shape->result_row_count;
    case LOOM_MATRIX_FRAGMENT_AXIS_COLUMN:
      return tile_shape->result_column_count;
    case LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION:
      return tile_shape->reduction_count;
    case LOOM_MATRIX_FRAGMENT_AXIS_COUNT:
    default:
      IREE_ASSERT_UNREACHABLE("invalid matrix fragment semantic axis");
      return 0;
  }
}

static void loom_matrix_fragment_set_axis_coordinate(
    loom_matrix_fragment_coordinate_t* coordinate,
    loom_matrix_fragment_axis_t axis, uint16_t value) {
  switch (axis) {
    case LOOM_MATRIX_FRAGMENT_AXIS_BLOCK:
      coordinate->block = value;
      break;
    case LOOM_MATRIX_FRAGMENT_AXIS_ROW:
      coordinate->row = value;
      break;
    case LOOM_MATRIX_FRAGMENT_AXIS_COLUMN:
      coordinate->column = value;
      break;
    case LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION:
      coordinate->reduction = value;
      break;
    case LOOM_MATRIX_FRAGMENT_AXIS_COUNT:
    default:
      IREE_ASSERT_UNREACHABLE("invalid matrix fragment semantic axis");
      break;
  }
}

static bool loom_matrix_fragment_role_coordinate_element_count(
    const loom_matrix_fragment_role_layout_t* role_layout,
    uint32_t* out_element_count) {
  uint32_t element_count = 1;
  bool has_axis = false;
  for (iree_host_size_t i = 0; i < LOOM_MATRIX_FRAGMENT_AXIS_COUNT; ++i) {
    const loom_matrix_fragment_axis_t axis = (loom_matrix_fragment_axis_t)i;
    if (!iree_any_bit_set(role_layout->coordinate_flags,
                          loom_matrix_fragment_axis_coordinate_flag(axis))) {
      continue;
    }
    has_axis = true;
    const loom_matrix_fragment_axis_layout_t* axis_layout =
        &role_layout->axes[axis];
    const uint32_t axis_element_count =
        (uint32_t)axis_layout->outer_count * axis_layout->element_count;
    if (axis_element_count == 0 ||
        element_count > UINT16_MAX / axis_element_count) {
      return false;
    }
    element_count *= axis_element_count;
  }
  *out_element_count = has_axis ? element_count : 0;
  return has_axis;
}

bool loom_matrix_fragment_coordinate(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, uint16_t lane,
    uint16_t payload_element_index,
    loom_matrix_fragment_coordinate_t* out_coordinate) {
  IREE_ASSERT_ARGUMENT(out_coordinate);

  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, role);
  return loom_matrix_fragment_coordinate_from_role_layout(
      layout, role_layout, lane, payload_element_index, out_coordinate);
}

bool loom_matrix_fragment_coordinate_from_role_layout(
    const loom_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout, uint16_t lane,
    uint16_t payload_element_index,
    loom_matrix_fragment_coordinate_t* out_coordinate) {
  IREE_ASSERT_ARGUMENT(out_coordinate);
  *out_coordinate = (loom_matrix_fragment_coordinate_t){0};
  if (layout == NULL || role_layout == NULL || layout->wave_size == 0 ||
      lane >= layout->wave_size ||
      payload_element_index >= role_layout->payload_element_count ||
      role_layout->coordinate_element_stride == 0 ||
      payload_element_index < role_layout->coordinate_element_offset) {
    return false;
  }

  const uint32_t relative_payload_element =
      payload_element_index - role_layout->coordinate_element_offset;
  if ((relative_payload_element % role_layout->coordinate_element_stride) !=
      0) {
    return false;
  }
  const uint32_t coordinate_element_index =
      relative_payload_element / role_layout->coordinate_element_stride;
  uint32_t coordinate_element_count = 0;
  if (!loom_matrix_fragment_role_coordinate_element_count(
          role_layout, &coordinate_element_count) ||
      coordinate_element_index >= coordinate_element_count) {
    return false;
  }

  uint32_t inner_element_count = 1;
  uint32_t outer_element_count = 1;
  for (iree_host_size_t i = 0; i < LOOM_MATRIX_FRAGMENT_AXIS_COUNT; ++i) {
    const loom_matrix_fragment_axis_t axis = (loom_matrix_fragment_axis_t)i;
    if (!iree_any_bit_set(role_layout->coordinate_flags,
                          loom_matrix_fragment_axis_coordinate_flag(axis))) {
      continue;
    }
    inner_element_count *= role_layout->axes[axis].element_count;
    outer_element_count *= role_layout->axes[axis].outer_count;
  }
  IREE_ASSERT_EQ(coordinate_element_count,
                 inner_element_count * outer_element_count);
  const uint32_t inner_linear_index =
      coordinate_element_index % inner_element_count;
  const uint32_t outer_linear_index =
      coordinate_element_index / inner_element_count;

  out_coordinate->coordinate_flags = role_layout->coordinate_flags;
  uint32_t inner_stride = inner_element_count;
  uint32_t outer_stride = outer_element_count;
  for (iree_host_size_t i = 0; i < LOOM_MATRIX_FRAGMENT_AXIS_COUNT; ++i) {
    const loom_matrix_fragment_axis_t axis = (loom_matrix_fragment_axis_t)i;
    if (!iree_any_bit_set(role_layout->coordinate_flags,
                          loom_matrix_fragment_axis_coordinate_flag(axis))) {
      continue;
    }
    const loom_matrix_fragment_axis_layout_t* axis_layout =
        &role_layout->axes[axis];
    if (axis_layout->outer_count == 0 || axis_layout->thread_count == 0 ||
        axis_layout->thread_stride == 0 || axis_layout->element_count == 0) {
      return false;
    }
    inner_stride /= axis_layout->element_count;
    outer_stride /= axis_layout->outer_count;
    const uint32_t element_coordinate =
        (inner_linear_index / inner_stride) % axis_layout->element_count;
    const uint32_t thread_coordinate =
        (lane / axis_layout->thread_stride) % axis_layout->thread_count;
    const uint32_t outer_coordinate =
        (outer_linear_index / outer_stride) % axis_layout->outer_count;
    const uint32_t coordinate =
        element_coordinate +
        (uint32_t)axis_layout->element_count *
            (thread_coordinate +
             (uint32_t)axis_layout->thread_count * outer_coordinate);
    if (coordinate >=
        loom_matrix_fragment_axis_extent(&layout->tile_shape, axis)) {
      return false;
    }
    loom_matrix_fragment_set_axis_coordinate(out_coordinate, axis,
                                             (uint16_t)coordinate);
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

bool loom_matrix_fragment_role_has_contiguous_lane_xor1_columns(
    const loom_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role) {
  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, role);
  return role_layout != NULL &&
         iree_all_bits_set(
             role_layout->flags,
             LOOM_MATRIX_FRAGMENT_ROLE_LAYOUT_FLAG_CONTIGUOUS_LANE_XOR1_COLUMNS);
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
    for (uint16_t payload_element_index = 0;
         payload_element_index < role_layout->payload_element_count;
         ++payload_element_index) {
      loom_matrix_fragment_coordinate_t candidate = {0};
      if (!loom_matrix_fragment_coordinate_from_role_layout(
              layout, role_layout, lane, payload_element_index, &candidate)) {
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
    for (uint16_t payload_element_index = 0;
         payload_element_index < role_layout->payload_element_count;
         ++payload_element_index) {
      loom_matrix_fragment_coordinate_t candidate = {0};
      if (!loom_matrix_fragment_coordinate_from_role_layout(
              layout, role_layout, lane, payload_element_index, &candidate)) {
        continue;
      }
      if (!loom_matrix_fragment_coordinate_matches(candidate, coordinate)) {
        continue;
      }
      if (current_occurrence == occurrence_index) {
        *out_element = (loom_matrix_fragment_physical_element_t){
            .lane = lane,
            .payload_element_index = payload_element_index,
        };
        return true;
      }
      ++current_occurrence;
    }
  }
  return false;
}
