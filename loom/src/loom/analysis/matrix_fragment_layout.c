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
