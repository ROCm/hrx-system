// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/boundary/predicate_word.h"

#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"

bool loom_predicate_word_supports_type(loom_type_t type,
                                       uint32_t maximum_element_count) {
  if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_I1) {
    return false;
  }
  if (loom_type_is_scalar(type)) {
    return true;
  }
  uint64_t element_count = 0;
  return loom_type_is_vector(type) &&
         loom_type_static_element_count(type, &element_count) &&
         element_count != 0 && element_count <= maximum_element_count;
}

loom_type_t loom_predicate_word_type(loom_type_t predicate_type) {
  predicate_type.header = loom_type_make_header(
      loom_type_kind(predicate_type), LOOM_SCALAR_TYPE_I32,
      loom_type_rank(predicate_type), loom_type_flags(predicate_type));
  return predicate_type;
}

iree_status_t loom_predicate_word_materialize(loom_builder_t* builder,
                                              loom_value_id_t predicate,
                                              loom_type_t word_type,
                                              loom_location_id_t location,
                                              loom_value_id_t* out_word) {
  loom_op_t* true_op = NULL;
  loom_op_t* false_op = NULL;
  loom_op_t* select_op = NULL;
  if (loom_type_is_scalar(word_type)) {
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder, loom_attr_i64(1), word_type, location, &true_op));
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder, loom_attr_i64(0), word_type, location, &false_op));
    IREE_RETURN_IF_ERROR(loom_scf_select_build(
        builder, predicate, loom_scalar_constant_result(true_op),
        loom_scalar_constant_result(false_op), word_type, location,
        &select_op));
    *out_word = loom_scf_select_result(select_op);
  } else {
    IREE_RETURN_IF_ERROR(
        loom_vector_splati_build(builder, 1, word_type, location, &true_op));
    IREE_RETURN_IF_ERROR(
        loom_vector_splati_build(builder, 0, word_type, location, &false_op));
    IREE_RETURN_IF_ERROR(loom_vector_select_build(
        builder, predicate, loom_vector_splati_result(true_op),
        loom_vector_splati_result(false_op), word_type, location, &select_op));
    *out_word = loom_vector_select_result(select_op);
  }
  return iree_ok_status();
}

iree_status_t loom_predicate_word_reconstruct(loom_builder_t* builder,
                                              loom_value_id_t word,
                                              loom_type_t predicate_type,
                                              loom_location_id_t location,
                                              loom_value_id_t* out_predicate) {
  const loom_type_t word_type = loom_predicate_word_type(predicate_type);
  loom_op_t* zero_op = NULL;
  loom_op_t* compare_op = NULL;
  if (loom_type_is_scalar(predicate_type)) {
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder, loom_attr_i64(0), word_type, location, &zero_op));
    IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
        builder, LOOM_SCALAR_CMPI_PREDICATE_NE, word,
        loom_scalar_constant_result(zero_op), location, &compare_op));
    *out_predicate = loom_scalar_cmpi_result(compare_op);
  } else {
    IREE_RETURN_IF_ERROR(
        loom_vector_splati_build(builder, 0, word_type, location, &zero_op));
    IREE_RETURN_IF_ERROR(
        loom_vector_cmpi_build(builder, LOOM_VECTOR_CMPI_PREDICATE_NE, word,
                               loom_vector_splati_result(zero_op), word_type,
                               predicate_type, location, &compare_op));
    *out_predicate = loom_vector_cmpi_result(compare_op);
  }
  return iree_ok_status();
}
