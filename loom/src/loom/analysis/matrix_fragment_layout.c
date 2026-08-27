// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/matrix_fragment_layout.h"

#include <string.h>

void loom_matrix_fragment_apply_coordinate_projection(
    const loom_matrix_fragment_coordinate_projection_term_t* terms,
    uint8_t term_count,
    const uint32_t
        source_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT],
    uint32_t out_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT]) {
  memset(
      out_terms, 0,
      LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT * sizeof(out_terms[0]));
  for (uint8_t i = 0; i < term_count; ++i) {
    const loom_matrix_fragment_coordinate_projection_term_t* term = &terms[i];
    uint32_t digit =
        source_terms[term->source_dimension] / (uint32_t)term->source_divisor;
    if (term->source_modulus != 0) {
      digit %= term->source_modulus;
    }
    out_terms[term->destination_dimension] +=
        digit * (uint32_t)term->destination_multiplier;
  }
}

loom_matrix_fragment_coordinate_dimension_t
loom_matrix_fragment_axis_coordinate_dimension(
    loom_matrix_fragment_axis_t axis) {
  switch (axis) {
    case LOOM_MATRIX_FRAGMENT_AXIS_BLOCK:
      return LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_BLOCK;
    case LOOM_MATRIX_FRAGMENT_AXIS_ROW:
      return LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW;
    case LOOM_MATRIX_FRAGMENT_AXIS_COLUMN:
      return LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN;
    case LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION:
      return LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_REDUCTION;
    case LOOM_MATRIX_FRAGMENT_AXIS_COUNT:
    default:
      IREE_ASSERT_UNREACHABLE("invalid matrix fragment semantic axis");
      IREE_BUILTIN_UNREACHABLE();
  }
}

loom_matrix_fragment_axis_t loom_matrix_fragment_coordinate_dimension_axis(
    loom_matrix_fragment_coordinate_dimension_t dimension) {
  switch (dimension) {
    case LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_BLOCK:
      return LOOM_MATRIX_FRAGMENT_AXIS_BLOCK;
    case LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW:
      return LOOM_MATRIX_FRAGMENT_AXIS_ROW;
    case LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN:
      return LOOM_MATRIX_FRAGMENT_AXIS_COLUMN;
    case LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_REDUCTION:
      return LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION;
    case LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT:
    case LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE:
    case LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT:
    default:
      return LOOM_MATRIX_FRAGMENT_AXIS_COUNT;
  }
}

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
      role_layout->coordinate_projection_plan == NULL ||
      role_layout->coordinate_element_stride == 0 ||
      payload_element_index < role_layout->coordinate_element_offset) {
    return false;
  }
  if (role_layout->reduction_group.storage_element_count != 0 ||
      role_layout->reduction_group.logical_element_count != 0) {
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
  if (coordinate_element_index >= role_layout->coordinate_element_count) {
    return false;
  }

  const loom_matrix_fragment_coordinate_projection_plan_t* plan =
      role_layout->coordinate_projection_plan;
  const uint32_t source_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT] =
      {
          [LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT] = lane,
          [LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE] =
              coordinate_element_index,
      };
  uint32_t terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT];
  loom_matrix_fragment_apply_coordinate_projection(
      plan->terms, plan->forward_term_count, source_terms, terms);
  out_coordinate->coordinate_flags = role_layout->coordinate_flags;
  out_coordinate->block =
      (uint16_t)terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_BLOCK];
  out_coordinate->row =
      (uint16_t)terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW];
  out_coordinate->column =
      (uint16_t)terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN];
  out_coordinate->reduction =
      (uint16_t)terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_REDUCTION];
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
