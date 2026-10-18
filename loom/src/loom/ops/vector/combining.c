// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/combining.h"

#include "loom/ops/vector/ops.h"

typedef iree_status_t (*loom_vector_flagged_binary_builder_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

typedef iree_status_t (*loom_vector_binary_builder_t)(
    loom_builder_t* builder, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

bool loom_vector_combining_kind_op(loom_combining_kind_t kind,
                                   loom_op_kind_t* out_op_kind) {
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
      *out_op_kind = LOOM_OP_VECTOR_ADDI;
      return true;
    case LOOM_COMBINING_KIND_ADDF:
      *out_op_kind = LOOM_OP_VECTOR_ADDF;
      return true;
    case LOOM_COMBINING_KIND_MULI:
      *out_op_kind = LOOM_OP_VECTOR_MULI;
      return true;
    case LOOM_COMBINING_KIND_MULF:
      *out_op_kind = LOOM_OP_VECTOR_MULF;
      return true;
    case LOOM_COMBINING_KIND_MINSI:
      *out_op_kind = LOOM_OP_VECTOR_MINSI;
      return true;
    case LOOM_COMBINING_KIND_MAXSI:
      *out_op_kind = LOOM_OP_VECTOR_MAXSI;
      return true;
    case LOOM_COMBINING_KIND_MINUI:
      *out_op_kind = LOOM_OP_VECTOR_MINUI;
      return true;
    case LOOM_COMBINING_KIND_MAXUI:
      *out_op_kind = LOOM_OP_VECTOR_MAXUI;
      return true;
    case LOOM_COMBINING_KIND_ANDI:
      *out_op_kind = LOOM_OP_VECTOR_ANDI;
      return true;
    case LOOM_COMBINING_KIND_ORI:
      *out_op_kind = LOOM_OP_VECTOR_ORI;
      return true;
    case LOOM_COMBINING_KIND_XORI:
      *out_op_kind = LOOM_OP_VECTOR_XORI;
      return true;
    case LOOM_COMBINING_KIND_MINIMUMF:
      *out_op_kind = LOOM_OP_VECTOR_MINIMUMF;
      return true;
    case LOOM_COMBINING_KIND_MAXIMUMF:
      *out_op_kind = LOOM_OP_VECTOR_MAXIMUMF;
      return true;
    case LOOM_COMBINING_KIND_MINNUMF:
      *out_op_kind = LOOM_OP_VECTOR_MINNUMF;
      return true;
    case LOOM_COMBINING_KIND_MAXNUMF:
      *out_op_kind = LOOM_OP_VECTOR_MAXNUMF;
      return true;
    case LOOM_COMBINING_KIND_COUNT_:
      break;
  }
  *out_op_kind = LOOM_OP_KIND_UNKNOWN;
  return false;
}

iree_status_t loom_vector_combining_build(
    loom_builder_t* builder, loom_combining_kind_t kind, uint8_t fastmath_flags,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_result) {
  if (loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1) {
    const bool has_i1_equivalent =
        loom_combining_kind_i1_equivalent(kind, &kind);
    IREE_ASSERT(has_i1_equivalent);
  }
  loom_vector_flagged_binary_builder_t flagged_builder = NULL;
  loom_vector_binary_builder_t builder_without_flags = NULL;
  uint8_t instance_flags = 0;
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
      flagged_builder = loom_vector_addi_build;
      break;
    case LOOM_COMBINING_KIND_ADDF:
      flagged_builder = loom_vector_addf_build;
      instance_flags = fastmath_flags;
      break;
    case LOOM_COMBINING_KIND_MULI:
      flagged_builder = loom_vector_muli_build;
      break;
    case LOOM_COMBINING_KIND_MULF:
      flagged_builder = loom_vector_mulf_build;
      instance_flags = fastmath_flags;
      break;
    case LOOM_COMBINING_KIND_MINSI:
      builder_without_flags = loom_vector_minsi_build;
      break;
    case LOOM_COMBINING_KIND_MAXSI:
      builder_without_flags = loom_vector_maxsi_build;
      break;
    case LOOM_COMBINING_KIND_MINUI:
      builder_without_flags = loom_vector_minui_build;
      break;
    case LOOM_COMBINING_KIND_MAXUI:
      builder_without_flags = loom_vector_maxui_build;
      break;
    case LOOM_COMBINING_KIND_ANDI:
      builder_without_flags = loom_vector_andi_build;
      break;
    case LOOM_COMBINING_KIND_ORI:
      builder_without_flags = loom_vector_ori_build;
      break;
    case LOOM_COMBINING_KIND_XORI:
      builder_without_flags = loom_vector_xori_build;
      break;
    case LOOM_COMBINING_KIND_MINIMUMF:
      flagged_builder = loom_vector_minimumf_build;
      instance_flags = fastmath_flags;
      break;
    case LOOM_COMBINING_KIND_MAXIMUMF:
      flagged_builder = loom_vector_maximumf_build;
      instance_flags = fastmath_flags;
      break;
    case LOOM_COMBINING_KIND_MINNUMF:
      flagged_builder = loom_vector_minnumf_build;
      instance_flags = fastmath_flags;
      break;
    case LOOM_COMBINING_KIND_MAXNUMF:
      flagged_builder = loom_vector_maxnumf_build;
      instance_flags = fastmath_flags;
      break;
    case LOOM_COMBINING_KIND_COUNT_:
      IREE_ASSERT_UNREACHABLE(
          "verified vector reduction has an invalid combining kind");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_op_t* combine_op = NULL;
  if (flagged_builder != NULL) {
    IREE_RETURN_IF_ERROR(flagged_builder(builder, instance_flags, lhs, rhs,
                                         result_type, location, &combine_op));
  } else {
    IREE_RETURN_IF_ERROR(builder_without_flags(builder, lhs, rhs, result_type,
                                               location, &combine_op));
  }
  *out_result = loom_op_results(combine_op)[0];
  return iree_ok_status();
}
