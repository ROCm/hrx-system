// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/construction.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"
#include "loom/util/fact_vector_partition.h"

iree_status_t loom_vector_replace_single_result_with_value(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

iree_status_t loom_vector_replace_single_result_with_splat(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t scalar,
    loom_type_t result_type) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* splat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      &rewriter->builder, scalar, result_type, op->location, &splat_op));
  loom_value_id_t replacement = loom_vector_splat_result(splat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

iree_status_t loom_vector_replace_single_result_with_new_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* new_op,
    loom_value_id_t value_checkpoint) {
  loom_value_id_t replacement = loom_op_const_results(new_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static bool loom_vector_static_indices_are_proven_in_bounds(
    loom_type_t source_type, loom_attribute_t static_indices) {
  if (!loom_type_is_all_static(source_type)) {
    return false;
  }
  if (static_indices.count > loom_type_rank(source_type)) {
    return false;
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index < 0 || static_index == INT64_MIN) {
      return false;
    }
    if (static_index >= loom_type_dim_static_size_at(source_type, i)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_canonicalize_extract_from_splat(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t source_type, loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (!loom_vector_static_indices_are_proven_in_bounds(source_type,
                                                       static_indices)) {
    return iree_ok_status();
  }

  loom_value_id_t scalar = loom_vector_splat_scalar(source_def_op);
  if (loom_type_is_scalar(result_type)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, scalar));
  } else if (loom_type_is_vector(result_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_splat(
        op, rewriter, scalar, result_type));
  } else {
    return iree_ok_status();
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_extract_from_broadcast(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t source_type, loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  loom_value_id_t broadcast_source =
      loom_vector_broadcast_source(source_def_op);
  loom_type_t broadcast_source_type =
      loom_module_value_type(rewriter->module, broadcast_source);
  if (!loom_type_is_vector(source_type) ||
      !loom_type_is_vector(broadcast_source_type) ||
      !loom_type_equal(broadcast_source_type, result_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t broadcast_source_rank = loom_type_rank(broadcast_source_type);
  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (source_rank < broadcast_source_rank ||
      static_indices.count != source_rank - broadcast_source_rank ||
      loom_vector_extract_indices(op).count != 0 ||
      !loom_vector_static_indices_are_proven_in_bounds(source_type,
                                                       static_indices)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
      op, rewriter, broadcast_source));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_extract_from_elements(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t source_type, loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  const loom_attribute_t static_indices =
      loom_vector_extract_static_indices(op);
  if (!loom_vector_static_indices_are_proven_in_bounds(source_type,
                                                       static_indices)) {
    return iree_ok_status();
  }
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_def_op);
  iree_host_size_t offset = 0;
  iree_host_size_t count = elements.count;
  // A static leading prefix selects a contiguous row-major operand range.
  // In-bounds coordinates imply nonzero indexed extents; the verified
  // constructor's operand count bounds every intermediate offset and count.
  for (uint16_t axis = 0; axis < static_indices.count; ++axis) {
    count /= (iree_host_size_t)loom_type_dim_static_size_at(source_type, axis);
    offset += (iree_host_size_t)static_indices.i64_array[axis] * count;
  }
  if (loom_type_is_scalar(result_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
        op, rewriter, elements.values[offset]));
  } else {
    loom_builder_set_before(&rewriter->builder, op);
    const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
    loom_op_t* construction = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
        &rewriter->builder, count ? elements.values + offset : NULL, count,
        result_type, op->location, &construction));
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
        op, rewriter, construction, checkpoint));
  }
  *out_changed = true;
  return iree_ok_status();
}

// A scalar extraction consumes every source axis, so composing its indices
// with slice offsets does not need to preserve trailing slice extents.
static iree_status_t loom_vector_canonicalize_extract_from_slice(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t source_type, loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  if (!loom_type_is_scalar(result_type)) {
    return iree_ok_status();
  }

  const uint8_t rank = loom_type_rank(source_type);
  const loom_attribute_t extract_indices =
      loom_vector_extract_static_indices(op);
  const loom_attribute_t slice_offsets =
      loom_vector_slice_static_offsets(source_def_op);
  if (loom_vector_extract_indices(op).count != 0 ||
      loom_vector_slice_offsets(source_def_op).count != 0 ||
      extract_indices.kind != LOOM_ATTR_I64_ARRAY ||
      slice_offsets.kind != LOOM_ATTR_I64_ARRAY ||
      extract_indices.count != rank || slice_offsets.count != rank) {
    return iree_ok_status();
  }

  int64_t combined_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (uint8_t axis = 0; axis < rank; ++axis) {
    const int64_t extract_index = extract_indices.i64_array[axis];
    const int64_t slice_offset = slice_offsets.i64_array[axis];
    if (extract_index < 0 || extract_index == INT64_MIN || slice_offset < 0 ||
        slice_offset == INT64_MIN ||
        !iree_checked_add_i64(slice_offset, extract_index,
                              &combined_indices[axis])) {
      return iree_ok_status();
    }
  }

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &rewriter->builder, loom_vector_slice_source(source_def_op),
      /*indices=*/NULL, /*indices_count=*/0, combined_indices, rank,
      result_type, op->location, &replacement_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, replacement_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

typedef enum loom_vector_static_index_relation_e {
  LOOM_VECTOR_STATIC_INDEX_RELATION_UNKNOWN = 0,
  LOOM_VECTOR_STATIC_INDEX_RELATION_EXACT,
  LOOM_VECTOR_STATIC_INDEX_RELATION_LEFT_PREFIX,
  LOOM_VECTOR_STATIC_INDEX_RELATION_RIGHT_PREFIX,
  LOOM_VECTOR_STATIC_INDEX_RELATION_DISJOINT,
} loom_vector_static_index_relation_t;

static loom_vector_static_index_relation_t loom_vector_classify_static_indices(
    loom_attribute_t left_static_indices, loom_value_slice_t left_indices,
    loom_attribute_t right_static_indices, loom_value_slice_t right_indices) {
  if (left_indices.count != 0 || right_indices.count != 0 ||
      left_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      right_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      left_static_indices.count == 0 || right_static_indices.count == 0 ||
      !left_static_indices.i64_array || !right_static_indices.i64_array) {
    return LOOM_VECTOR_STATIC_INDEX_RELATION_UNKNOWN;
  }
  for (uint16_t i = 0; i < left_static_indices.count; ++i) {
    if (left_static_indices.i64_array[i] < 0 ||
        left_static_indices.i64_array[i] == INT64_MIN) {
      return LOOM_VECTOR_STATIC_INDEX_RELATION_UNKNOWN;
    }
  }
  for (uint16_t i = 0; i < right_static_indices.count; ++i) {
    if (right_static_indices.i64_array[i] < 0 ||
        right_static_indices.i64_array[i] == INT64_MIN) {
      return LOOM_VECTOR_STATIC_INDEX_RELATION_UNKNOWN;
    }
  }
  const uint16_t shared_rank =
      iree_min(left_static_indices.count, right_static_indices.count);
  for (uint16_t i = 0; i < shared_rank; ++i) {
    const int64_t left_index = left_static_indices.i64_array[i];
    const int64_t right_index = right_static_indices.i64_array[i];
    if (left_index != right_index) {
      return LOOM_VECTOR_STATIC_INDEX_RELATION_DISJOINT;
    }
  }
  if (left_static_indices.count == right_static_indices.count) {
    return LOOM_VECTOR_STATIC_INDEX_RELATION_EXACT;
  }
  if (left_static_indices.count < right_static_indices.count) {
    return LOOM_VECTOR_STATIC_INDEX_RELATION_LEFT_PREFIX;
  }
  return LOOM_VECTOR_STATIC_INDEX_RELATION_RIGHT_PREFIX;
}

static iree_status_t loom_vector_canonicalize_extract_from_insert(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* insert_op,
    loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  loom_attribute_t extract_static_indices =
      loom_vector_extract_static_indices(op);
  loom_value_slice_t extract_indices = loom_vector_extract_indices(op);
  loom_attribute_t insert_static_indices =
      loom_vector_insert_static_indices(insert_op);
  loom_value_slice_t insert_indices = loom_vector_insert_indices(insert_op);
  const loom_vector_static_index_relation_t relation =
      loom_vector_classify_static_indices(
          extract_static_indices, extract_indices, insert_static_indices,
          insert_indices);
  if (relation == LOOM_VECTOR_STATIC_INDEX_RELATION_EXACT) {
    loom_value_id_t inserted_value = loom_vector_insert_value(insert_op);
    if (!loom_type_equal(
            loom_module_value_type(rewriter->module, inserted_value),
            result_type)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
        op, rewriter, inserted_value));
    *out_changed = true;
    return iree_ok_status();
  }
  if (relation == LOOM_VECTOR_STATIC_INDEX_RELATION_RIGHT_PREFIX) {
    loom_builder_set_before(&rewriter->builder, op);
    loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
    loom_op_t* replacement_op = NULL;
    const uint16_t suffix_rank =
        extract_static_indices.count - insert_static_indices.count;
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        &rewriter->builder, loom_vector_insert_value(insert_op),
        /*indices=*/NULL, /*indices_count=*/0,
        extract_static_indices.i64_array + insert_static_indices.count,
        suffix_rank, result_type, op->location, &replacement_op));
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
        op, rewriter, replacement_op, value_checkpoint));
    *out_changed = true;
    return iree_ok_status();
  }
  if (relation != LOOM_VECTOR_STATIC_INDEX_RELATION_DISJOINT) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &rewriter->builder, loom_vector_insert_dest(insert_op),
      /*indices=*/NULL, /*indices_count=*/0, extract_static_indices.i64_array,
      extract_static_indices.count, result_type, op->location,
      &replacement_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, replacement_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_project_slice_structure(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;
  if (loom_vector_slice_offsets(op).count != 0) {
    return iree_ok_status();
  }
  const loom_value_id_t source = loom_vector_slice_source(op);
  const loom_type_t source_type =
      loom_module_value_type(rewriter->module, source);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_vector_slice_result(op));
  loom_op_t* source_def_op = loom_rewriter_value_defining_op(rewriter, source);
  if (!source_def_op || !loom_vector_concat_isa(source_def_op)) {
    return iree_ok_status();
  }

  if (!loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type)) {
    return iree_ok_status();
  }

  const uint8_t rank = loom_type_rank(source_type);
  if (rank == 0 || loom_type_rank(result_type) != rank) {
    return iree_ok_status();
  }
  const int64_t axis = loom_vector_concat_axis(source_def_op);
  if (axis < 0 || axis >= rank) {
    return iree_ok_status();
  }

  const loom_attribute_t static_offsets = loom_vector_slice_static_offsets(op);
  if (static_offsets.count != rank) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < rank; ++i) {
    if (static_offsets.i64_array[i] < 0) {
      return iree_ok_status();
    }
  }

  const int64_t slice_offset = static_offsets.i64_array[axis];
  const int64_t slice_extent =
      loom_type_dim_static_size_at(result_type, (uint8_t)axis);
  int64_t slice_end = 0;
  if (slice_extent <= 0 ||
      !iree_checked_add_i64(slice_offset, slice_extent, &slice_end)) {
    return iree_ok_status();
  }

  int64_t input_start = 0;
  const loom_value_slice_t inputs = loom_vector_concat_inputs(source_def_op);
  for (uint16_t i = 0; i < inputs.count; ++i) {
    const loom_value_id_t input = inputs.values[i];
    const loom_type_t input_type =
        loom_module_value_type(rewriter->module, input);
    if (!loom_type_is_vector(input_type) ||
        !loom_type_is_all_static(input_type) ||
        loom_type_rank(input_type) != rank) {
      return iree_ok_status();
    }

    const int64_t input_extent =
        loom_type_dim_static_size_at(input_type, (uint8_t)axis);
    int64_t input_end = 0;
    if (!iree_checked_add_i64(input_start, input_extent, &input_end)) {
      return iree_ok_status();
    }
    if (slice_offset < input_start || slice_end > input_end) {
      input_start = input_end;
      continue;
    }

    int64_t adjusted_offsets[LOOM_TYPE_MAX_RANK];
    bool all_offsets_are_zero = true;
    for (uint8_t j = 0; j < rank; ++j) {
      adjusted_offsets[j] = static_offsets.i64_array[j];
    }
    adjusted_offsets[axis] -= input_start;
    for (uint8_t j = 0; j < rank; ++j) {
      all_offsets_are_zero &= adjusted_offsets[j] == 0;
    }

    if (all_offsets_are_zero && loom_type_equal(input_type, result_type)) {
      IREE_RETURN_IF_ERROR(
          loom_vector_replace_single_result_with_value(op, rewriter, input));
      *out_changed = true;
      return iree_ok_status();
    }

    loom_builder_set_before(&rewriter->builder, op);
    const loom_value_id_t value_checkpoint =
        loom_rewriter_value_checkpoint(rewriter);
    loom_op_t* replacement_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        &rewriter->builder, input, /*offsets=*/NULL, /*offsets_count=*/0,
        adjusted_offsets, rank, result_type, op->location, &replacement_op));
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
        op, rewriter, replacement_op, value_checkpoint));
    *out_changed = true;
    return iree_ok_status();
  }

  return iree_ok_status();
}

static iree_status_t loom_vector_project_insert_structure(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;
  loom_value_id_t value = loom_vector_insert_value(op);
  loom_value_id_t dest = loom_vector_insert_dest(op);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_vector_insert_result(op));
  loom_attribute_t static_indices = loom_vector_insert_static_indices(op);
  loom_value_slice_t indices = loom_vector_insert_indices(op);

  loom_op_t* value_def_op = loom_rewriter_value_defining_op(rewriter, value);
  if (value_def_op && loom_vector_extract_isa(value_def_op) &&
      loom_type_equal(
          loom_module_value_type(rewriter->module,
                                 loom_vector_extract_source(value_def_op)),
          result_type) &&
      loom_vector_extract_source(value_def_op) == dest &&
      loom_vector_classify_static_indices(
          static_indices, indices,
          loom_vector_extract_static_indices(value_def_op),
          loom_vector_extract_indices(value_def_op)) ==
          LOOM_VECTOR_STATIC_INDEX_RELATION_EXACT) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, dest));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_op_t* dest_def_op = loom_rewriter_value_defining_op(rewriter, dest);
  if (!dest_def_op || !loom_vector_insert_isa(dest_def_op) ||
      loom_vector_classify_static_indices(
          static_indices, indices,
          loom_vector_insert_static_indices(dest_def_op),
          loom_vector_insert_indices(dest_def_op)) !=
          LOOM_VECTOR_STATIC_INDEX_RELATION_EXACT) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_insert_build(
      &rewriter->builder, value, loom_vector_insert_dest(dest_def_op),
      /*indices=*/NULL, /*indices_count=*/0, static_indices.i64_array,
      static_indices.count, result_type, op->location, &replacement_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, replacement_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

iree_status_t loom_vector_project_regular_partition(loom_op_t* op,
                                                    loom_rewriter_t* rewriter,
                                                    bool* out_changed) {
  *out_changed = false;
  const bool is_extract = loom_vector_extract_isa(op);
  if (!is_extract && !loom_vector_slice_isa(op)) {
    return iree_ok_status();
  }
  const loom_value_id_t source = loom_op_const_operands(op)[0];
  const loom_type_t source_type =
      loom_module_value_type(rewriter->module, source);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  if (loom_type_rank(source_type) != 1 ||
      (is_extract ? !loom_type_is_scalar(result_type)
                  : !loom_type_is_all_static(result_type))) {
    return iree_ok_status();
  }
  const loom_attribute_t offsets = is_extract
                                       ? loom_vector_extract_static_indices(op)
                                       : loom_vector_slice_static_offsets(op);
  const uint64_t length =
      is_extract ? 1 : loom_type_dim_static_size_at(result_type, 0);
  // An empty slice may start at the end of the source, outside any component.
  if (length == 0 || offsets.i64_array[0] == INT64_MIN) {
    return iree_ok_status();
  }
  const loom_value_fact_vector_partition_t partition =
      loom_value_fact_table_query_vector_partition(rewriter->fact_table,
                                                   source);
  if (partition.snapshot_lane_count == 0) {
    return iree_ok_status();
  }
  const uint64_t offset = offsets.i64_array[0];
  const loom_value_fact_vector_partition_component_t component =
      partition.snapshot_components[offset / partition.snapshot_lane_count];
  const int64_t component_offset = offset % partition.snapshot_lane_count;
  if (length > component.lane_count - component_offset) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacement = component.value;
  if (is_extract || component_offset != 0 ||
      !loom_type_equal(
          loom_module_value_type(rewriter->module, component.value),
          result_type)) {
    loom_op_t* projected = NULL;
    if (is_extract) {
      IREE_RETURN_IF_ERROR(loom_vector_extract_build(
          &rewriter->builder, component.value, NULL, 0, &component_offset, 1,
          result_type, op->location, &projected));
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          &rewriter->builder, component.value, NULL, 0, &component_offset, 1,
          result_type, op->location, &projected));
    }
    replacement = loom_op_const_results(projected)[0];
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_changed = true;
  return iree_ok_status();
}

iree_status_t loom_vector_fold_constant_lanes(loom_op_t* op,
                                              loom_rewriter_t* rewriter,
                                              bool* out_changed) {
  *out_changed = false;
  const loom_value_id_t result = loom_op_const_results(op)[0];
  const loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, result);
  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_value_facts_query_small_static_lanes(&rewriter->fact_table->context,
                                                 facts, &lanes)) {
    return iree_ok_status();
  }
  for (iree_host_size_t lane = 0; lane < lanes.count; ++lane) {
    if (!loom_value_facts_is_exact(lanes.lanes[lane])) {
      return iree_ok_status();
    }
  }

  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, result);
  const loom_type_t element_type =
      loom_type_scalar(loom_type_element_type(result_type));
  for (iree_host_size_t lane = 0; lane < lanes.count; ++lane) {
    if (!loom_rewriter_can_materialize_constant(rewriter, lanes.lanes[lane],
                                                element_type)) {
      return iree_ok_status();
    }
  }
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t elements[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT];
  for (iree_host_size_t lane = 0; lane < lanes.count; ++lane) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_build_constant(rewriter, lanes.lanes[lane], element_type,
                                     op->location, &elements[lane]));
  }
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &rewriter->builder, elements, lanes.count, result_type, op->location,
      &replacement_op));
  const loom_value_id_t replacement =
      loom_vector_from_elements_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_vector_value_def_op(const loom_rewriter_t* rewriter,
                                     loom_value_id_t value_id,
                                     loom_op_t** out_op) {
  const loom_value_t* value = loom_module_value(rewriter->module, value_id);
  *out_op = loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
  return *out_op != NULL;
}

#define LOOM_VECTOR_LANE_CHAIN_MAX_STEPS 8

typedef struct loom_vector_lane_chain_step_t {
  // Scalar unary op kind matched for this lane-chain step.
  loom_op_kind_t scalar_kind;
  // Per-instance flags copied from the scalar op.
  uint8_t instance_flags;
  // Scalar input type observed on the matched scalar op.
  loom_type_t input_type;
  // Scalar result type observed on the matched scalar op.
  loom_type_t result_type;
} loom_vector_lane_chain_step_t;

typedef struct loom_vector_lane_chain_t {
  // Source vector whose static lanes feed every reconstructed lane.
  loom_value_id_t source;
  // Type of the source vector.
  loom_type_t source_type;
  // Number of valid entries in steps.
  iree_host_size_t step_count;
  // Scalar unary steps from outermost producer toward the source extract.
  loom_vector_lane_chain_step_t steps[LOOM_VECTOR_LANE_CHAIN_MAX_STEPS];
} loom_vector_lane_chain_t;

static loom_type_t loom_vector_same_shape_with_element_type(
    loom_type_t shape_type, loom_scalar_type_t element_type) {
  loom_type_t type = shape_type;
  type.header = loom_type_make_header(LOOM_TYPE_VECTOR, element_type,
                                      loom_type_rank(shape_type),
                                      loom_type_flags(shape_type));
  return type;
}

static bool loom_vector_type_is_static_1d_vector(loom_type_t type,
                                                 int64_t* out_lane_count) {
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      loom_type_dim_is_dynamic_at(type, 0)) {
    return false;
  }
  *out_lane_count = loom_type_dim_static_size_at(type, 0);
  return *out_lane_count >= 0;
}

static bool loom_vector_lane_chain_steps_equal(
    const loom_vector_lane_chain_step_t* lhs,
    const loom_vector_lane_chain_step_t* rhs) {
  return lhs->scalar_kind == rhs->scalar_kind &&
         lhs->instance_flags == rhs->instance_flags &&
         loom_type_equal(lhs->input_type, rhs->input_type) &&
         loom_type_equal(lhs->result_type, rhs->result_type);
}

static bool loom_vector_scalar_lane_chain_kind_supported(
    loom_op_kind_t scalar_kind) {
  switch (scalar_kind) {
    case LOOM_OP_INDEX_CAST:
    case LOOM_OP_SCALAR_CEILF:
    case LOOM_OP_SCALAR_FLOORF:
    case LOOM_OP_SCALAR_ROUNDF:
    case LOOM_OP_SCALAR_ROUNDEVENF:
    case LOOM_OP_SCALAR_TRUNCF:
    case LOOM_OP_SCALAR_FPTOSI:
    case LOOM_OP_SCALAR_FPTOUI:
    case LOOM_OP_SCALAR_TRUNCI:
    case LOOM_OP_SCALAR_SITOFP:
    case LOOM_OP_SCALAR_UITOFP:
      return true;
    default:
      return false;
  }
}

static bool loom_vector_describe_scalar_lane_chain_step(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_vector_lane_chain_step_t* out_step, loom_value_id_t* out_input) {
  *out_step = (loom_vector_lane_chain_step_t){0};
  *out_input = LOOM_VALUE_ID_INVALID;
  if (!loom_vector_scalar_lane_chain_kind_supported(source_op->kind)) {
    return false;
  }
  out_step->scalar_kind = source_op->kind;
  out_step->instance_flags = source_op->instance_flags;
  *out_input = loom_op_const_operands(source_op)[0];
  out_step->input_type = loom_module_value_type(module, *out_input);
  out_step->result_type =
      loom_module_value_type(module, loom_op_const_results(source_op)[0]);
  return loom_type_is_scalar(out_step->input_type) &&
         loom_type_is_scalar(out_step->result_type);
}

static bool loom_vector_value_is_static_lane_extract(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id,
    iree_host_size_t expected_lane, loom_value_id_t* out_source,
    loom_type_t* out_source_type) {
  *out_source = LOOM_VALUE_ID_INVALID;
  *out_source_type = loom_type_none();

  loom_op_t* extract_op = NULL;
  if (!loom_vector_value_def_op(rewriter, value_id, &extract_op) ||
      !loom_vector_extract_isa(extract_op) ||
      loom_vector_extract_indices(extract_op).count != 0) {
    return false;
  }

  loom_attribute_t static_indices =
      loom_vector_extract_static_indices(extract_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] < 0 ||
      (iree_host_size_t)static_indices.i64_array[0] != expected_lane) {
    return false;
  }

  loom_value_id_t source = loom_vector_extract_source(extract_op);
  *out_source = source;
  *out_source_type = loom_module_value_type(rewriter->module, source);
  return true;
}

iree_status_t loom_vector_from_elements_fold_source(loom_op_t* op,
                                                    loom_rewriter_t* rewriter,
                                                    bool* out_changed) {
  *out_changed = false;
  const loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  const loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_from_elements_result(op));
  int64_t lane_count = 0;
  if (!loom_vector_type_is_static_1d_vector(result_type, &lane_count) ||
      lane_count == 0 || lane_count != (int64_t)elements.count) {
    return iree_ok_status();
  }

  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  loom_type_t source_type = loom_type_none();
  if (!loom_vector_value_is_static_lane_extract(rewriter, elements.values[0], 0,
                                                &source, &source_type) ||
      !loom_type_equal(source_type, result_type)) {
    return iree_ok_status();
  }
  for (iree_host_size_t lane = 1; lane < elements.count; ++lane) {
    loom_value_id_t lane_source = LOOM_VALUE_ID_INVALID;
    loom_type_t lane_source_type = loom_type_none();
    if (!loom_vector_value_is_static_lane_extract(
            rewriter, elements.values[lane], lane, &lane_source,
            &lane_source_type) ||
        lane_source != source) {
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &source, 1));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_vector_match_scalar_lane_chain(
    const loom_rewriter_t* rewriter, loom_value_id_t lane_value,
    iree_host_size_t lane_index, loom_vector_lane_chain_t* out_chain) {
  *out_chain = (loom_vector_lane_chain_t){0};
  out_chain->source = LOOM_VALUE_ID_INVALID;

  loom_value_id_t value = lane_value;
  while (out_chain->step_count < LOOM_VECTOR_LANE_CHAIN_MAX_STEPS) {
    loom_op_t* source_op = NULL;
    if (!loom_vector_value_def_op(rewriter, value, &source_op)) {
      break;
    }

    loom_vector_lane_chain_step_t step = {0};
    loom_value_id_t input = LOOM_VALUE_ID_INVALID;
    if (!loom_vector_describe_scalar_lane_chain_step(
            rewriter->module, source_op, &step, &input)) {
      break;
    }
    out_chain->steps[out_chain->step_count++] = step;
    value = input;
  }

  loom_op_t* overflow_op = NULL;
  if (out_chain->step_count == LOOM_VECTOR_LANE_CHAIN_MAX_STEPS &&
      loom_vector_value_def_op(rewriter, value, &overflow_op)) {
    loom_vector_lane_chain_step_t overflow_step = {0};
    loom_value_id_t overflow_input = LOOM_VALUE_ID_INVALID;
    if (loom_vector_describe_scalar_lane_chain_step(
            rewriter->module, overflow_op, &overflow_step, &overflow_input)) {
      return false;
    }
  }

  return loom_vector_value_is_static_lane_extract(
      rewriter, value, lane_index, &out_chain->source, &out_chain->source_type);
}

static bool loom_vector_lane_chain_matches(
    const loom_vector_lane_chain_t* expected,
    const loom_vector_lane_chain_t* candidate) {
  if (candidate->source != expected->source ||
      candidate->step_count != expected->step_count ||
      !loom_type_equal(candidate->source_type, expected->source_type)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < expected->step_count; ++i) {
    if (!loom_vector_lane_chain_steps_equal(&candidate->steps[i],
                                            &expected->steps[i])) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_build_lane_chain_step(
    loom_builder_t* builder, const loom_vector_lane_chain_step_t* step,
    loom_value_id_t input, loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  switch (step->scalar_kind) {
    case LOOM_OP_INDEX_CAST:
      return loom_vector_index_cast_build(builder, input, input_type,
                                          result_type, location, out_op);
    case LOOM_OP_SCALAR_CEILF:
      return loom_vector_ceilf_build(builder, step->instance_flags, input,
                                     result_type, location, out_op);
    case LOOM_OP_SCALAR_FLOORF:
      return loom_vector_floorf_build(builder, step->instance_flags, input,
                                      result_type, location, out_op);
    case LOOM_OP_SCALAR_ROUNDF:
      return loom_vector_roundf_build(builder, step->instance_flags, input,
                                      result_type, location, out_op);
    case LOOM_OP_SCALAR_ROUNDEVENF:
      return loom_vector_roundevenf_build(builder, step->instance_flags, input,
                                          result_type, location, out_op);
    case LOOM_OP_SCALAR_TRUNCF:
      return loom_vector_truncf_build(builder, step->instance_flags, input,
                                      result_type, location, out_op);
    case LOOM_OP_SCALAR_FPTOSI:
      return loom_vector_fptosi_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_FPTOUI:
      return loom_vector_fptoui_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_TRUNCI:
      return loom_vector_trunci_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_SITOFP:
      return loom_vector_sitofp_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_UITOFP:
      return loom_vector_uitofp_build(builder, input, input_type, result_type,
                                      location, out_op);
    default:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unsupported vector lane-chain step kind");
  return iree_ok_status();
}

iree_status_t loom_vector_from_elements_combine_lanes(loom_op_t* op,
                                                      loom_rewriter_t* rewriter,
                                                      bool* out_changed) {
  *out_changed = false;

  loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count == 0) {
    return iree_ok_status();
  }

  const loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_from_elements_result(op));
  int64_t result_lane_count = 0;
  if (!loom_vector_type_is_static_1d_vector(result_type, &result_lane_count) ||
      result_lane_count != (int64_t)elements.count) {
    return iree_ok_status();
  }

  loom_vector_lane_chain_t expected_chain = {0};
  if (!loom_vector_match_scalar_lane_chain(rewriter, elements.values[0], 0,
                                           &expected_chain) ||
      expected_chain.step_count == 0) {
    return iree_ok_status();
  }
  int64_t source_lane_count = 0;
  if (!loom_vector_type_is_static_1d_vector(expected_chain.source_type,
                                            &source_lane_count) ||
      source_lane_count != result_lane_count ||
      !loom_type_shape_equals(expected_chain.source_type, result_type)) {
    return iree_ok_status();
  }

  for (iree_host_size_t lane_index = 1; lane_index < elements.count;
       ++lane_index) {
    loom_vector_lane_chain_t candidate_chain = {0};
    if (!loom_vector_match_scalar_lane_chain(rewriter,
                                             elements.values[lane_index],
                                             lane_index, &candidate_chain) ||
        !loom_vector_lane_chain_matches(&expected_chain, &candidate_chain)) {
      return iree_ok_status();
    }
  }

  loom_type_t validated_input_type = expected_chain.source_type;
  for (iree_host_size_t step_ordinal = expected_chain.step_count;
       step_ordinal > 0; --step_ordinal) {
    const iree_host_size_t step_index = step_ordinal - 1;
    const loom_vector_lane_chain_step_t* step =
        &expected_chain.steps[step_index];
    if (loom_type_element_type(validated_input_type) !=
        loom_type_element_type(step->input_type)) {
      return iree_ok_status();
    }
    validated_input_type = loom_vector_same_shape_with_element_type(
        result_type, loom_type_element_type(step->result_type));
    if (step_index == 0) {
      validated_input_type = result_type;
    }
    if (loom_type_element_type(validated_input_type) !=
        loom_type_element_type(step->result_type)) {
      return iree_ok_status();
    }
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_value_id_t replacement = expected_chain.source;
  loom_type_t input_type = expected_chain.source_type;
  loom_op_t* replacement_op = NULL;
  for (iree_host_size_t step_ordinal = expected_chain.step_count;
       step_ordinal > 0; --step_ordinal) {
    const iree_host_size_t step_index = step_ordinal - 1;
    const loom_vector_lane_chain_step_t* step =
        &expected_chain.steps[step_index];
    loom_type_t step_result_type = loom_vector_same_shape_with_element_type(
        result_type, loom_type_element_type(step->result_type));
    if (step_index == 0) {
      step_result_type = result_type;
    }

    IREE_RETURN_IF_ERROR(loom_vector_build_lane_chain_step(
        &rewriter->builder, step, replacement, input_type, step_result_type,
        op->location, &replacement_op));
    replacement = loom_op_const_results(replacement_op)[0];
    input_type = step_result_type;
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_vector_match_static_scalar_insert(const loom_module_t* module,
                                                   const loom_op_t* op,
                                                   loom_type_t vector_type,
                                                   iree_host_size_t lane_count,
                                                   loom_value_id_t* out_value,
                                                   iree_host_size_t* out_lane) {
  if (!loom_vector_insert_isa(op) ||
      loom_vector_insert_indices(op).count != 0 ||
      !loom_type_equal(
          loom_module_value_type(module, loom_vector_insert_result(op)),
          vector_type) ||
      !loom_type_equal(
          loom_module_value_type(module, loom_vector_insert_dest(op)),
          vector_type)) {
    return false;
  }

  const loom_value_id_t value = loom_vector_insert_value(op);
  if (!loom_type_equal(loom_module_value_type(module, value),
                       loom_type_scalar(loom_type_element_type(vector_type)))) {
    return false;
  }

  const loom_attribute_t static_indices = loom_vector_insert_static_indices(op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] < 0 ||
      (uint64_t)static_indices.i64_array[0] >= lane_count) {
    return false;
  }

  if (out_value) {
    *out_value = value;
  }
  if (out_lane) {
    *out_lane = (iree_host_size_t)static_indices.i64_array[0];
  }
  return true;
}

// Construction-only prefixes are subsumed by their terminal insert. Deferring
// them prevents the greedy driver from materializing and then immediately
// reopening one vector.from_elements for every lane in a linear chain.
static bool loom_vector_insert_has_only_construction_users(
    const loom_rewriter_t* rewriter, const loom_op_t* op,
    loom_type_t vector_type, iree_host_size_t lane_count) {
  const loom_value_id_t result = loom_vector_insert_result(op);
  const loom_value_t* result_value =
      loom_module_value(rewriter->module, result);
  const loom_use_t* uses = loom_value_uses(result_value);
  for (uint32_t i = 0; i < result_value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (!loom_vector_match_static_scalar_insert(
            rewriter->module, user_op, vector_type, lane_count,
            /*out_value=*/NULL, /*out_lane=*/NULL) ||
        loom_vector_insert_dest(user_op) != result) {
      return false;
    }
  }
  return true;
}

// Rejects ordinary partial updates before allocating lane storage. A chain with
// at least one insert per lane may completely overwrite an arbitrary base; a
// shorter chain is a constructor only when its base exposes every missing lane.
static bool loom_vector_insert_chain_may_form_elements(
    const loom_rewriter_t* rewriter, const loom_op_t* root_op,
    loom_type_t vector_type, iree_host_size_t lane_count) {
  const loom_op_t* insert_op = root_op;
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  iree_host_size_t insert_count = 0;
  while (insert_op != NULL && loom_vector_match_static_scalar_insert(
                                  rewriter->module, insert_op, vector_type,
                                  lane_count, /*out_value=*/NULL,
                                  /*out_lane=*/NULL)) {
    ++insert_count;
    base = loom_vector_insert_dest(insert_op);
    const loom_op_t* base_op = loom_rewriter_value_defining_op(rewriter, base);
    insert_op = base_op && loom_vector_insert_isa(base_op) ? base_op : NULL;
  }
  if (insert_count >= lane_count) {
    return true;
  }

  const loom_op_t* base_op = loom_rewriter_value_defining_op(rewriter, base);
  if (base_op && loom_vector_splat_isa(base_op)) {
    return true;
  }
  return base_op && loom_vector_from_elements_isa(base_op) &&
         loom_vector_from_elements_elements(base_op).count == lane_count;
}

iree_status_t loom_vector_insert_combine_elements(loom_op_t* op,
                                                  loom_rewriter_t* rewriter,
                                                  bool* out_changed) {
  *out_changed = false;

  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_vector_insert_result(op));
  int64_t lane_count_i64 = 0;
  if (!loom_vector_type_is_static_1d_vector(result_type, &lane_count_i64) ||
      lane_count_i64 <= 0 || lane_count_i64 > UINT16_MAX) {
    return iree_ok_status();
  }
  const iree_host_size_t lane_count = (iree_host_size_t)lane_count_i64;

  if (!loom_vector_match_static_scalar_insert(
          rewriter->module, op, result_type, lane_count,
          /*out_value=*/NULL, /*out_lane=*/NULL) ||
      loom_vector_insert_has_only_construction_users(rewriter, op, result_type,
                                                     lane_count) ||
      !loom_vector_insert_chain_may_form_elements(rewriter, op, result_type,
                                                  lane_count)) {
    return iree_ok_status();
  }

  loom_value_id_t* elements = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_value_id_scratch(
      rewriter->fact_table, lane_count, &elements));
  for (iree_host_size_t lane = 0; lane < lane_count; ++lane) {
    elements[lane] = LOOM_VALUE_ID_INVALID;
  }

  iree_host_size_t assigned_count = 0;
  loom_op_t* insert_op = op;
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  while (insert_op != NULL) {
    loom_value_id_t inserted_value = LOOM_VALUE_ID_INVALID;
    iree_host_size_t inserted_lane = 0;
    if (!loom_vector_match_static_scalar_insert(
            rewriter->module, insert_op, result_type, lane_count,
            &inserted_value, &inserted_lane)) {
      return iree_ok_status();
    }
    if (elements[inserted_lane] == LOOM_VALUE_ID_INVALID) {
      elements[inserted_lane] = inserted_value;
      ++assigned_count;
    }

    base = loom_vector_insert_dest(insert_op);
    if (assigned_count == lane_count) {
      break;
    }
    loom_op_t* base_op = loom_rewriter_value_defining_op(rewriter, base);
    insert_op = base_op && loom_vector_insert_isa(base_op) ? base_op : NULL;
  }

  if (assigned_count != lane_count) {
    loom_op_t* base_op = loom_rewriter_value_defining_op(rewriter, base);
    const loom_value_id_t* base_elements = NULL;
    loom_value_id_t splat_element = LOOM_VALUE_ID_INVALID;
    if (base_op && loom_vector_splat_isa(base_op)) {
      splat_element = loom_vector_splat_scalar(base_op);
    } else if (base_op && loom_vector_from_elements_isa(base_op)) {
      const loom_value_slice_t element_slice =
          loom_vector_from_elements_elements(base_op);
      if (element_slice.count != lane_count) {
        return iree_ok_status();
      }
      base_elements = element_slice.values;
    } else {
      return iree_ok_status();
    }

    for (iree_host_size_t lane = 0; lane < lane_count; ++lane) {
      if (elements[lane] == LOOM_VALUE_ID_INVALID) {
        elements[lane] = base_elements ? base_elements[lane] : splat_element;
      }
    }
  }

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &rewriter->builder, elements, lane_count, result_type, op->location,
      &replacement_op));
  const loom_value_id_t replacement =
      loom_vector_from_elements_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_changed = true;
  return iree_ok_status();
}

// Coordinates are captured as numbers only. Operands that remain dynamic are
// deliberately read from the live root during application: another action may
// have replaced their producer since this action was prepared.
static loom_attribute_t loom_vector_coordinate_attributes(const loom_op_t* op) {
  if (loom_vector_extract_isa(op)) {
    return loom_vector_extract_static_indices(op);
  }
  if (loom_vector_insert_isa(op)) {
    return loom_vector_insert_static_indices(op);
  }
  return loom_vector_slice_static_offsets(op);
}

static loom_value_slice_t loom_vector_coordinate_operands(const loom_op_t* op) {
  if (loom_vector_extract_isa(op)) {
    return loom_vector_extract_indices(op);
  }
  if (loom_vector_insert_isa(op)) {
    return loom_vector_insert_indices(op);
  }
  return loom_vector_slice_offsets(op);
}

static bool loom_vector_resolve_coordinates(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_op_t* op, int64_t* coordinates) {
  const loom_value_slice_t dynamic = loom_vector_coordinate_operands(op);
  if (!facts || dynamic.count == 0) {
    return false;
  }
  const bool is_slice = loom_vector_slice_isa(op);
  const loom_value_id_t source = loom_vector_insert_isa(op)
                                     ? loom_vector_insert_dest(op)
                                     : loom_op_const_operands(op)[0];
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type =
      loom_module_value_type(module, loom_op_const_results(op)[0]);
  const loom_attribute_t offsets = loom_vector_coordinate_attributes(op);
  uint16_t dynamic_position = 0;
  bool changed = false;
  for (uint16_t axis = 0; axis < offsets.count; ++axis) {
    coordinates[axis] = offsets.i64_array[axis];
    if (coordinates[axis] != INT64_MIN) {
      continue;
    }
    const loom_value_id_t index = dynamic.values[dynamic_position++];
    int64_t coordinate = 0;
    if (!loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(facts, index), &coordinate) ||
        coordinate < 0) {
      continue;
    }
    if (!loom_type_dim_is_dynamic_at(source_type, axis)) {
      const int64_t bound = loom_type_dim_static_size_at(source_type, axis);
      if (is_slice) {
        // Empty slices permit the end coordinate; nonempty slices must fit.
        if (coordinate > bound) {
          continue;
        }
        if (!loom_type_dim_is_dynamic_at(result_type, axis)) {
          const int64_t extent =
              loom_type_dim_static_size_at(result_type, axis);
          if (extent > bound || coordinate > bound - extent) {
            continue;
          }
        }
      } else if (coordinate >= bound) {
        continue;
      }
    }
    coordinates[axis] = coordinate;
    changed = true;
  }
  return changed;
}

static iree_status_t loom_vector_apply_coordinates(loom_op_t* op,
                                                   loom_rewriter_t* rewriter,
                                                   const int64_t* coordinates) {
  const loom_attribute_t offsets = loom_vector_coordinate_attributes(op);
  const loom_value_slice_t dynamic = loom_vector_coordinate_operands(op);
  loom_value_id_t remaining[LOOM_TYPE_MAX_RANK];
  uint16_t dynamic_position = 0;
  uint16_t remaining_count = 0;
  for (uint16_t axis = 0; axis < offsets.count; ++axis) {
    if (offsets.i64_array[axis] != INT64_MIN) {
      continue;
    }
    const loom_value_id_t index = dynamic.values[dynamic_position++];
    if (coordinates[axis] == INT64_MIN) {
      remaining[remaining_count++] = index;
    }
  }
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement = NULL;
  if (loom_vector_extract_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        &rewriter->builder, loom_vector_extract_source(op), remaining,
        remaining_count, coordinates, offsets.count, result_type, op->location,
        &replacement));
  } else if (loom_vector_insert_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vector_insert_build(
        &rewriter->builder, loom_vector_insert_value(op),
        loom_vector_insert_dest(op), remaining, remaining_count, coordinates,
        offsets.count, result_type, op->location, &replacement));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        &rewriter->builder, loom_vector_slice_source(op), remaining,
        remaining_count, coordinates, offsets.count, result_type, op->location,
        &replacement));
  }
  return loom_vector_replace_single_result_with_new_op(op, rewriter,
                                                       replacement, checkpoint);
}

typedef struct loom_vector_coordinate_action_t {
  // Shared ownership and apply contract for the root's coordinate list.
  loom_rewrite_action_t action;
  // Static coordinates, or INT64_MIN for an operand that stays dynamic.
  int64_t coordinates[];
} loom_vector_coordinate_action_t;

static iree_status_t loom_vector_apply_coordinate_action(
    const loom_rewrite_action_t* action, loom_rewriter_t* rewriter) {
  const loom_vector_coordinate_action_t* coordinates =
      (const loom_vector_coordinate_action_t*)action;
  return loom_vector_apply_coordinates(action->root, rewriter,
                                       coordinates->coordinates);
}

iree_status_t loom_vector_prepare_coordinates(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_op_t* op, iree_arena_allocator_t* arena,
    loom_rewrite_action_t** out_action) {
  *out_action = NULL;
  int64_t coordinates[LOOM_TYPE_MAX_RANK];
  if (!loom_vector_resolve_coordinates(module, facts, op, coordinates)) {
    return iree_ok_status();
  }
  const uint16_t count = loom_vector_coordinate_attributes(op).count;
  loom_vector_coordinate_action_t* action = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, sizeof(*action) + count * sizeof(int64_t), (void**)&action));
  action->action.apply = loom_vector_apply_coordinate_action;
  memcpy(action->coordinates, coordinates, count * sizeof(int64_t));
  *out_action = &action->action;
  return iree_ok_status();
}

iree_status_t loom_vector_normalize_coordinates(loom_op_t* op,
                                                loom_rewriter_t* rewriter,
                                                bool* out_changed) {
  int64_t coordinates[LOOM_TYPE_MAX_RANK];
  *out_changed = loom_vector_resolve_coordinates(
      rewriter->module, rewriter->fact_table, op, coordinates);
  if (!*out_changed) {
    return iree_ok_status();
  }
  return loom_vector_apply_coordinates(op, rewriter, coordinates);
}

iree_status_t loom_vector_project_structure(loom_op_t* op,
                                            loom_rewriter_t* rewriter,
                                            bool* out_changed) {
  if (loom_vector_insert_isa(op)) {
    return loom_vector_project_insert_structure(op, rewriter, out_changed);
  }
  if (loom_vector_slice_isa(op)) {
    return loom_vector_project_slice_structure(op, rewriter, out_changed);
  }
  *out_changed = false;
  const loom_value_id_t source = loom_vector_extract_source(op);
  const loom_type_t source_type =
      loom_module_value_type(rewriter->module, source);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_vector_extract_result(op));
  if (loom_vector_extract_static_indices(op).count == 0 &&
      loom_type_equal(source_type, result_type)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, source));
    *out_changed = true;
    return iree_ok_status();
  }
  loom_op_t* producer = loom_rewriter_value_defining_op(rewriter, source);
  if (!producer) {
    return iree_ok_status();
  }
  if (loom_vector_broadcast_isa(producer)) {
    return loom_vector_canonicalize_extract_from_broadcast(
        op, rewriter, producer, source_type, result_type, out_changed);
  }
  if (loom_vector_splat_isa(producer)) {
    return loom_vector_canonicalize_extract_from_splat(
        op, rewriter, producer, source_type, result_type, out_changed);
  }
  if (loom_vector_from_elements_isa(producer)) {
    return loom_vector_canonicalize_extract_from_elements(
        op, rewriter, producer, source_type, result_type, out_changed);
  }
  if (loom_vector_slice_isa(producer)) {
    return loom_vector_canonicalize_extract_from_slice(
        op, rewriter, producer, source_type, result_type, out_changed);
  }
  if (loom_vector_insert_isa(producer)) {
    return loom_vector_canonicalize_extract_from_insert(
        op, rewriter, producer, result_type, out_changed);
  }
  return iree_ok_status();
}
