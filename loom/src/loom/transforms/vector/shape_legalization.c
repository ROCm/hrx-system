// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/shape_legalization.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/vector/ops.h"

// Bounds static structural expansion. Broadcast planning counts every new
// slice, extract, and splat together with its flatten, concat, and restore
// operations. Dynamic tail-vector insertion selects between strategies that
// emit about five operations per destination row or three per inserted lane.
#define LOOM_VECTOR_STATIC_SHAPE_OP_LIMIT 64u

// Bounds the lane walk used to plan static broadcast segments. Current target
// consumers admit at most 1024 logical lanes, and larger shapes must be
// packetized or rejected before structural expansion.
#define LOOM_VECTOR_STATIC_SHAPE_LANE_LIMIT 1024u

static loom_type_t loom_vector_static_shape_flat_type(loom_type_t type,
                                                      uint64_t element_count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(type),
                             loom_dim_pack_static((int64_t)element_count), 0);
}

static iree_status_t loom_vector_static_shape_flatten_value(
    loom_builder_t* builder, loom_value_id_t value, loom_type_t value_type,
    uint64_t element_count, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const loom_type_t flat_type =
      loom_vector_static_shape_flat_type(value_type, element_count);
  if (loom_type_equal(value_type, flat_type)) {
    *out_value = value;
    return iree_ok_status();
  }
  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
      builder, value, value_type, flat_type, location, &bitcast_op));
  *out_value = loom_vector_bitcast_result(bitcast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_static_shape_restore_result(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_value_id_t flat_value, loom_type_t flat_type, loom_type_t result_type,
    loom_value_id_t value_checkpoint) {
  loom_value_id_t replacement = flat_value;
  if (!loom_type_equal(flat_type, result_type)) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        &context->rewriter->builder, flat_value, flat_type, result_type,
        op->location, &bitcast_op));
    replacement = loom_vector_bitcast_result(bitcast_op);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(context->rewriter, op,
                                                  &replacement, 1);
}

// Linearizes mixed static and dynamic leading coordinates into one row-major
// ordinal. Static terms remain folded into |out_static_index| when every
// coordinate is static; otherwise |out_dynamic_index| carries the complete
// ordinal and |out_static_index| is zero.
static iree_status_t loom_vector_static_shape_linearize_indices(
    loom_builder_t* builder, loom_type_t shaped_type,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    loom_location_id_t location, loom_value_id_t* out_dynamic_index,
    int64_t* out_static_index) {
  *out_dynamic_index = LOOM_VALUE_ID_INVALID;
  *out_static_index = 0;

  loom_value_id_t dynamic_axis_indices[LOOM_TYPE_MAX_RANK];
  uint16_t dynamic_index_position = 0;
  for (uint16_t axis = 0; axis < static_indices.count; ++axis) {
    dynamic_axis_indices[axis] = LOOM_VALUE_ID_INVALID;
    if (static_indices.i64_array[axis] == INT64_MIN) {
      dynamic_axis_indices[axis] =
          dynamic_indices.values[dynamic_index_position++];
    }
  }

  int64_t stride = 1;
  int64_t static_offset = 0;
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  for (uint16_t reverse_axis = 0; reverse_axis < static_indices.count;
       ++reverse_axis) {
    const uint16_t axis = static_indices.count - reverse_axis - 1u;
    if (reverse_axis != 0) {
      stride *= loom_type_dim_static_size_at(shaped_type, axis + 1u);
    }

    const loom_value_id_t dynamic_index = dynamic_axis_indices[axis];
    if (dynamic_index == LOOM_VALUE_ID_INVALID) {
      static_offset += static_indices.i64_array[axis] * stride;
      continue;
    }

    if (stride == 1) {
      if (accumulator == LOOM_VALUE_ID_INVALID) {
        accumulator = dynamic_index;
      } else {
        loom_op_t* add_op = NULL;
        IREE_RETURN_IF_ERROR(loom_index_add_build(
            builder, accumulator, dynamic_index,
            loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &add_op));
        accumulator = loom_index_add_result(add_op);
      }
      continue;
    }

    loom_op_t* stride_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(stride),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &stride_op));
    const loom_value_id_t stride_value = loom_index_constant_result(stride_op);
    if (accumulator == LOOM_VALUE_ID_INVALID) {
      loom_op_t* multiply_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_mul_build(
          builder, dynamic_index, stride_value, location, &multiply_op));
      accumulator = loom_index_mul_result(multiply_op);
    } else {
      loom_op_t* multiply_add_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_madd_build(builder, dynamic_index,
                                                 stride_value, accumulator,
                                                 location, &multiply_add_op));
      accumulator = loom_index_madd_result(multiply_add_op);
    }
  }

  if (accumulator == LOOM_VALUE_ID_INVALID) {
    *out_static_index = static_offset;
    return iree_ok_status();
  }
  if (static_offset != 0) {
    loom_op_t* offset_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(static_offset),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &offset_op));
    loom_op_t* add_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_add_build(
        builder, accumulator, loom_index_constant_result(offset_op),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &add_op));
    accumulator = loom_index_add_result(add_op);
  }
  *out_dynamic_index = accumulator;
  return iree_ok_status();
}

// Maps a row-major result lane through trailing-axis broadcast semantics.
// Missing leading source axes and singleton source axes both select lane zero.
static uint64_t loom_vector_static_shape_broadcast_source_ordinal(
    loom_type_t source_type, loom_type_t result_type, uint64_t result_ordinal) {
  const uint8_t source_rank = loom_type_rank(source_type);
  const uint8_t result_rank = loom_type_rank(result_type);
  const uint8_t result_axis_offset = result_rank - source_rank;
  uint64_t result_suffix_extent = 1;
  uint64_t source_suffix_extent = 1;
  uint64_t source_ordinal = 0;
  for (uint8_t reverse_axis = 0; reverse_axis < result_rank; ++reverse_axis) {
    const uint8_t result_axis = result_rank - reverse_axis - 1;
    const uint64_t result_extent =
        (uint64_t)loom_type_dim_static_size_at(result_type, result_axis);
    const uint64_t result_coordinate =
        (result_ordinal / result_suffix_extent) % result_extent;
    if (result_axis >= result_axis_offset) {
      const uint8_t source_axis = result_axis - result_axis_offset;
      const uint64_t source_extent =
          (uint64_t)loom_type_dim_static_size_at(source_type, source_axis);
      const uint64_t source_coordinate =
          source_extent == 1 ? 0 : result_coordinate;
      source_ordinal += source_coordinate * source_suffix_extent;
      source_suffix_extent *= source_extent;
    }
    result_suffix_extent *= result_extent;
  }
  return source_ordinal;
}

typedef enum loom_vector_static_shape_broadcast_segment_kind_e {
  // Copies a contiguous source interval into the result.
  LOOM_VECTOR_STATIC_SHAPE_BROADCAST_SEGMENT_SEQUENCE = 0,
  // Replicates one source lane across the result interval.
  LOOM_VECTOR_STATIC_SHAPE_BROADCAST_SEGMENT_SPLAT = 1,
} loom_vector_static_shape_broadcast_segment_kind_t;

typedef struct loom_vector_static_shape_broadcast_segment_t {
  // Kind of source mapping represented by this segment.
  loom_vector_static_shape_broadcast_segment_kind_t kind;
  // First flat source lane selected by the segment.
  uint64_t source_ordinal;
  // Number of flat result lanes produced by the segment.
  uint64_t lane_count;
} loom_vector_static_shape_broadcast_segment_t;

static bool loom_vector_static_shape_broadcast_segment_equal(
    const loom_vector_static_shape_broadcast_segment_t* lhs,
    const loom_vector_static_shape_broadcast_segment_t* rhs) {
  return lhs->kind == rhs->kind && lhs->source_ordinal == rhs->source_ordinal &&
         lhs->lane_count == rhs->lane_count;
}

static iree_status_t loom_vector_static_shape_broadcast_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  const loom_value_id_t source = loom_vector_broadcast_source(op);
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_broadcast_result(op));
  uint64_t source_count = 0;
  uint64_t result_count = 0;
  if (!loom_type_static_element_count(source_type, &source_count) ||
      !loom_type_static_element_count(result_type, &result_count) ||
      source_count == 0 || result_count == 0 || source_count > INT64_MAX ||
      result_count > LOOM_VECTOR_STATIC_SHAPE_LANE_LIMIT ||
      result_count % source_count != 0) {
    return iree_ok_status();
  }

  if (loom_type_equal(source_type, result_type)) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        context->rewriter, op, &source, 1));
    *out_rewritten = true;
    return iree_ok_status();
  }

  // Partition the row-major result into contiguous source sequences and
  // repeated source lanes. This keeps leading-axis replication as shared
  // source references and lowers singleton axes through native splats instead
  // of arbitrary shuffles that targets may scalarize. Identical segments are
  // materialized once and reused by the final concat.
  loom_vector_static_shape_broadcast_segment_t
      segments[LOOM_VECTOR_STATIC_SHAPE_OP_LIMIT] = {0};
  iree_host_size_t segment_count = 0;
  iree_host_size_t operation_count = 0;
  uint64_t result_ordinal = 0;
  while (result_ordinal < result_count) {
    if (segment_count == IREE_ARRAYSIZE(segments)) {
      return iree_ok_status();
    }
    const uint64_t source_ordinal =
        loom_vector_static_shape_broadcast_source_ordinal(
            source_type, result_type, result_ordinal);
    uint64_t repeat_count = 1;
    while (result_ordinal + repeat_count < result_count &&
           loom_vector_static_shape_broadcast_source_ordinal(
               source_type, result_type, result_ordinal + repeat_count) ==
               source_ordinal) {
      ++repeat_count;
    }
    uint64_t sequence_count = 1;
    while (result_ordinal + sequence_count < result_count &&
           source_ordinal + sequence_count < source_count &&
           loom_vector_static_shape_broadcast_source_ordinal(
               source_type, result_type, result_ordinal + sequence_count) ==
               source_ordinal + sequence_count) {
      ++sequence_count;
    }

    loom_vector_static_shape_broadcast_segment_t* segment =
        &segments[segment_count++];
    segment->kind = repeat_count > sequence_count
                        ? LOOM_VECTOR_STATIC_SHAPE_BROADCAST_SEGMENT_SPLAT
                        : LOOM_VECTOR_STATIC_SHAPE_BROADCAST_SEGMENT_SEQUENCE;
    segment->source_ordinal = source_ordinal;
    segment->lane_count =
        segment->kind == LOOM_VECTOR_STATIC_SHAPE_BROADCAST_SEGMENT_SPLAT
            ? repeat_count
            : sequence_count;
    result_ordinal += segment->lane_count;

    bool reuses_segment = false;
    for (iree_host_size_t i = 0; i + 1 < segment_count; ++i) {
      reuses_segment = loom_vector_static_shape_broadcast_segment_equal(
          &segments[i], segment);
      if (reuses_segment) {
        break;
      }
    }
    if (!reuses_segment &&
        !(segment->kind ==
              LOOM_VECTOR_STATIC_SHAPE_BROADCAST_SEGMENT_SEQUENCE &&
          segment->source_ordinal == 0 &&
          segment->lane_count == source_count)) {
      operation_count +=
          segment->kind == LOOM_VECTOR_STATIC_SHAPE_BROADCAST_SEGMENT_SPLAT
              ? 2u
              : 1u;
    }
  }

  const loom_type_t flat_source_type =
      loom_vector_static_shape_flat_type(source_type, source_count);
  const loom_type_t flat_result_type =
      loom_vector_static_shape_flat_type(result_type, result_count);
  operation_count += !loom_type_equal(source_type, flat_source_type);
  operation_count += segment_count > 1;
  operation_count += !loom_type_equal(flat_result_type, result_type);
  if (operation_count > LOOM_VECTOR_STATIC_SHAPE_OP_LIMIT) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t flat_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
      &rewriter->builder, source, source_type, source_count, op->location,
      &flat_source));

  loom_value_id_t segment_values[LOOM_VECTOR_STATIC_SHAPE_OP_LIMIT] = {0};
  for (iree_host_size_t segment_index = 0; segment_index < segment_count;
       ++segment_index) {
    const loom_vector_static_shape_broadcast_segment_t* segment =
        &segments[segment_index];
    bool reused_segment = false;
    for (iree_host_size_t i = 0; i < segment_index; ++i) {
      if (loom_vector_static_shape_broadcast_segment_equal(&segments[i],
                                                           segment)) {
        segment_values[segment_index] = segment_values[i];
        reused_segment = true;
        break;
      }
    }
    if (reused_segment) {
      continue;
    }

    if (segment->kind == LOOM_VECTOR_STATIC_SHAPE_BROADCAST_SEGMENT_SEQUENCE &&
        segment->source_ordinal == 0 && segment->lane_count == source_count) {
      segment_values[segment_index] = flat_source;
      continue;
    }

    const loom_type_t segment_type =
        loom_vector_static_shape_flat_type(result_type, segment->lane_count);
    const int64_t static_ordinal = (int64_t)segment->source_ordinal;
    if (segment->kind == LOOM_VECTOR_STATIC_SHAPE_BROADCAST_SEGMENT_SPLAT) {
      loom_op_t* extract_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_extract_build(
          &rewriter->builder, flat_source, NULL, 0, &static_ordinal, 1,
          loom_type_scalar(loom_type_element_type(source_type)), op->location,
          &extract_op));
      loom_op_t* splat_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_splat_build(
          &rewriter->builder, loom_vector_extract_result(extract_op),
          segment_type, op->location, &splat_op));
      segment_values[segment_index] = loom_vector_splat_result(splat_op);
    } else {
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          &rewriter->builder, flat_source, NULL, 0, &static_ordinal, 1,
          segment_type, op->location, &slice_op));
      segment_values[segment_index] = loom_vector_slice_result(slice_op);
    }
  }

  loom_value_id_t flat_result = segment_values[0];
  if (segment_count > 1) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        &rewriter->builder, 0, segment_values, segment_count, flat_result_type,
        op->location, &concat_op));
    flat_result = loom_vector_concat_result(concat_op);
  }
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_restore_result(
      context, op, flat_result, flat_result_type, result_type,
      value_checkpoint));
  *out_rewritten = true;
  return iree_ok_status();
}

// Returns the row-major source ordinal for one trailing-axis result row.
// Slice rows are contiguous along the trailing axis; leading result
// coordinates select which source row supplies that interval.
static uint64_t loom_vector_static_shape_slice_row_source_ordinal(
    loom_type_t source_type, loom_type_t result_type,
    const int64_t* static_offsets, uint64_t result_row) {
  const uint8_t rank = loom_type_rank(source_type);
  const uint8_t trailing_axis = rank - 1;
  uint64_t source_ordinal = (uint64_t)static_offsets[trailing_axis];
  uint64_t source_suffix_extent =
      (uint64_t)loom_type_dim_static_size_at(source_type, trailing_axis);
  uint64_t result_suffix_extent = 1;
  for (uint8_t reverse_axis = 1; reverse_axis < rank; ++reverse_axis) {
    const uint8_t axis = rank - reverse_axis - 1;
    const uint64_t result_extent =
        (uint64_t)loom_type_dim_static_size_at(result_type, axis);
    const uint64_t result_coordinate =
        (result_row / result_suffix_extent) % result_extent;
    const uint64_t source_coordinate =
        (uint64_t)static_offsets[axis] + result_coordinate;
    source_ordinal += source_coordinate * source_suffix_extent;
    source_suffix_extent *=
        (uint64_t)loom_type_dim_static_size_at(source_type, axis);
    result_suffix_extent *= result_extent;
  }
  return source_ordinal;
}

static iree_status_t loom_vector_static_shape_slice_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  const loom_value_id_t source = loom_vector_slice_source(op);
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_slice_result(op));
  const uint8_t rank = loom_type_rank(source_type);
  const loom_attribute_t static_offsets = loom_vector_slice_static_offsets(op);
  uint64_t source_count = 0;
  uint64_t result_count = 0;
  if (rank <= 1 || loom_type_rank(result_type) != rank ||
      !loom_type_static_element_count(source_type, &source_count) ||
      !loom_type_static_element_count(result_type, &result_count) ||
      source_count == 0 || result_count == 0 || source_count > INT64_MAX ||
      result_count > INT64_MAX || static_offsets.kind != LOOM_ATTR_I64_ARRAY ||
      static_offsets.count != rank ||
      loom_vector_slice_offsets(op).count != 0) {
    return iree_ok_status();
  }
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (static_offsets.i64_array[axis] < 0) {
      return iree_ok_status();
    }
  }

  const uint64_t row_length =
      (uint64_t)loom_type_dim_static_size_at(result_type, rank - 1);
  const uint64_t row_count = result_count / row_length;
  uint64_t run_count = 1;
  uint64_t previous_end =
      loom_vector_static_shape_slice_row_source_ordinal(
          source_type, result_type, static_offsets.i64_array, 0) +
      row_length;
  for (uint64_t row = 1; row < row_count; ++row) {
    const uint64_t source_ordinal =
        loom_vector_static_shape_slice_row_source_ordinal(
            source_type, result_type, static_offsets.i64_array, row);
    if (source_ordinal != previous_end) {
      ++run_count;
    }
    previous_end = source_ordinal + row_length;
  }
  if (run_count > IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }

  loom_value_id_t single_run = LOOM_VALUE_ID_INVALID;
  loom_value_id_t* runs = &single_run;
  if (run_count > 1) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(context->arena, (iree_host_size_t)run_count,
                                  sizeof(*runs), (void**)&runs));
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t flat_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
      &rewriter->builder, source, source_type, source_count, op->location,
      &flat_source));

  const loom_type_t flat_result_type =
      loom_vector_static_shape_flat_type(result_type, result_count);
  const uint64_t first_ordinal =
      loom_vector_static_shape_slice_row_source_ordinal(
          source_type, result_type, static_offsets.i64_array, 0);
  if (run_count == 1 && first_ordinal == 0 && result_count == source_count) {
    runs[0] = flat_source;
  } else {
    uint64_t run_start = first_ordinal;
    uint64_t run_length = row_length;
    iree_host_size_t run_index = 0;
    for (uint64_t row = 1; row <= row_count; ++row) {
      uint64_t source_ordinal = 0;
      if (row < row_count) {
        source_ordinal = loom_vector_static_shape_slice_row_source_ordinal(
            source_type, result_type, static_offsets.i64_array, row);
        if (source_ordinal == run_start + run_length) {
          run_length += row_length;
          continue;
        }
      }

      const int64_t static_offset = (int64_t)run_start;
      const loom_type_t run_type =
          loom_vector_static_shape_flat_type(result_type, run_length);
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          &rewriter->builder, flat_source, /*offsets=*/NULL,
          /*offsets_count=*/0, &static_offset,
          /*static_offsets_count=*/1, run_type, op->location, &slice_op));
      runs[run_index++] = loom_vector_slice_result(slice_op);
      run_start = source_ordinal;
      run_length = row_length;
    }
  }

  loom_value_id_t flat_result = runs[0];
  if (run_count > 1) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        &rewriter->builder, 0, runs, (iree_host_size_t)run_count,
        flat_result_type, op->location, &concat_op));
    flat_result = loom_vector_concat_result(concat_op);
  }
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_restore_result(
      context, op, flat_result, flat_result_type, result_type,
      value_checkpoint));
  *out_rewritten = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_static_shape_build_dynamic_tail_lane_insert(
    loom_builder_t* builder, loom_value_id_t flat_value,
    loom_type_t flat_value_type, loom_value_id_t flat_dest,
    loom_type_t flat_dest_type, loom_value_id_t row_ordinal,
    uint64_t value_count, loom_location_id_t location,
    loom_value_id_t* out_flat_result) {
  loom_value_id_t insertion_index = row_ordinal;
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  if (value_count > 1) {
    loom_op_t* value_count_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64((int64_t)value_count),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &value_count_op));
    loom_op_t* base_index_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_mul_build(
        builder, row_ordinal, loom_index_constant_result(value_count_op),
        location, &base_index_op));
    insertion_index = loom_index_mul_result(base_index_op);

    loom_op_t* one_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(1), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        location, &one_op));
    one = loom_index_constant_result(one_op);
  }

  const loom_type_t scalar_type =
      loom_type_scalar(loom_type_element_type(flat_value_type));
  loom_value_id_t current_dest = flat_dest;
  const int64_t dynamic_sentinel = INT64_MIN;
  for (uint64_t lane = 0; lane < value_count; ++lane) {
    if (lane > 0) {
      loom_op_t* next_index_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          builder, insertion_index, one,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &next_index_op));
      insertion_index = loom_index_add_result(next_index_op);
    }

    const int64_t static_lane = (int64_t)lane;
    loom_op_t* extract_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(builder, flat_value, NULL, 0,
                                                   &static_lane, 1, scalar_type,
                                                   location, &extract_op));
    loom_op_t* insert_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_insert_build(
        builder, loom_vector_extract_result(extract_op), current_dest,
        &insertion_index, 1, &dynamic_sentinel, 1, flat_dest_type, location,
        &insert_op));
    current_dest = loom_vector_insert_result(insert_op);
  }

  *out_flat_result = current_dest;
  return iree_ok_status();
}

static iree_status_t loom_vector_static_shape_build_dynamic_tail_row_select(
    loom_target_legalization_context_t* context, loom_value_id_t flat_value,
    loom_type_t flat_value_type, loom_value_id_t flat_dest,
    loom_type_t flat_dest_type, loom_value_id_t row_ordinal,
    uint64_t value_count, uint64_t row_count, loom_location_id_t location,
    loom_value_id_t* out_flat_result) {
  if (row_count == 1) {
    *out_flat_result = flat_value;
    return iree_ok_status();
  }

  loom_type_t predicate_type = flat_value_type;
  predicate_type.header =
      loom_type_make_header(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1, /*rank=*/1,
                            loom_type_flags(flat_value_type));

  loom_value_id_t* rows = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena,
                                                 (iree_host_size_t)row_count,
                                                 sizeof(*rows), (void**)&rows));
  loom_builder_t* builder = &context->rewriter->builder;
  for (uint64_t row = 0; row < row_count; ++row) {
    const int64_t row_offset = (int64_t)(row * value_count);
    loom_op_t* row_slice_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_slice_build(builder, flat_dest, NULL, 0, &row_offset, 1,
                                flat_value_type, location, &row_slice_op));

    loom_op_t* expected_ordinal_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_constant_build(builder, loom_attr_i64((int64_t)row),
                                  loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                  location, &expected_ordinal_op));
    loom_op_t* row_condition_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_cmp_build(builder, LOOM_INDEX_CMP_PREDICATE_EQ, row_ordinal,
                             loom_index_constant_result(expected_ordinal_op),
                             location, &row_condition_op));
    loom_op_t* row_mask_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_splat_build(
        builder, loom_index_cmp_result(row_condition_op), predicate_type,
        location, &row_mask_op));
    loom_op_t* row_select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_select_build(
        builder, loom_vector_splat_result(row_mask_op), flat_value,
        loom_vector_slice_result(row_slice_op), flat_value_type, location,
        &row_select_op));
    rows[row] = loom_vector_select_result(row_select_op);
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_concat_build(builder, 0, rows, (iree_host_size_t)row_count,
                               flat_dest_type, location, &concat_op));
  *out_flat_result = loom_vector_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_static_shape_insert_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  const loom_value_id_t value = loom_vector_insert_value(op);
  const loom_value_id_t dest = loom_vector_insert_dest(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  const loom_type_t dest_type = loom_module_value_type(context->module, dest);
  const loom_attribute_t static_indices = loom_vector_insert_static_indices(op);
  const loom_value_slice_t dynamic_indices = loom_vector_insert_indices(op);
  uint64_t dest_count = 0;
  if (loom_type_rank(dest_type) <= 1 ||
      !loom_type_static_element_count(dest_type, &dest_count) ||
      dest_count == 0 || dest_count > INT64_MAX ||
      static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return iree_ok_status();
  }

  uint64_t value_count = 1;
  if (loom_type_is_vector(value_type) &&
      !loom_type_static_element_count(value_type, &value_count)) {
    return iree_ok_status();
  }
  if (value_count == 0 || value_count > dest_count) {
    return iree_ok_status();
  }
  const uint64_t row_count = dest_count / value_count;
  bool use_dynamic_row_select = false;
  if (dynamic_indices.count != 0 && loom_type_is_vector(value_type)) {
    const bool can_select_rows =
        row_count <= (LOOM_VECTOR_STATIC_SHAPE_OP_LIMIT - 4u) / 5u;
    const bool can_insert_lanes =
        value_count <= (LOOM_VECTOR_STATIC_SHAPE_OP_LIMIT - 4u) / 3u;
    if (!can_select_rows && !can_insert_lanes) {
      return iree_ok_status();
    }
    use_dynamic_row_select =
        can_select_rows &&
        (!can_insert_lanes || 5u * row_count <= 3u * value_count);
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t dynamic_prefix_ordinal = LOOM_VALUE_ID_INVALID;
  int64_t static_prefix_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_linearize_indices(
      &rewriter->builder, dest_type, static_indices, dynamic_indices,
      op->location, &dynamic_prefix_ordinal, &static_prefix_ordinal));
  loom_value_id_t flat_dest = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
      &rewriter->builder, dest, dest_type, dest_count, op->location,
      &flat_dest));
  const loom_type_t flat_dest_type =
      loom_vector_static_shape_flat_type(dest_type, dest_count);

  loom_value_id_t flat_result = LOOM_VALUE_ID_INVALID;
  if (!loom_type_is_vector(value_type)) {
    loom_op_t* insert_op = NULL;
    if (dynamic_prefix_ordinal == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_vector_insert_build(
          &rewriter->builder, value, flat_dest, NULL, 0, &static_prefix_ordinal,
          1, flat_dest_type, op->location, &insert_op));
    } else {
      const int64_t dynamic_sentinel = INT64_MIN;
      IREE_RETURN_IF_ERROR(loom_vector_insert_build(
          &rewriter->builder, value, flat_dest, &dynamic_prefix_ordinal, 1,
          &dynamic_sentinel, 1, flat_dest_type, op->location, &insert_op));
    }
    flat_result = loom_vector_insert_result(insert_op);
  } else if (dynamic_prefix_ordinal == LOOM_VALUE_ID_INVALID) {
    // A static tail insertion covers one contiguous row-major range. Retain
    // the untouched prefix and suffix as slices around the flattened value.
    const uint64_t insertion_ordinal =
        (uint64_t)static_prefix_ordinal * value_count;
    loom_value_id_t inputs[3] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
                                 LOOM_VALUE_ID_INVALID};
    iree_host_size_t input_count = 0;
    if (insertion_ordinal > 0) {
      const int64_t static_offset = 0;
      const loom_type_t prefix_type =
          loom_vector_static_shape_flat_type(dest_type, insertion_ordinal);
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          &rewriter->builder, flat_dest, NULL, 0, &static_offset, 1,
          prefix_type, op->location, &slice_op));
      inputs[input_count++] = loom_vector_slice_result(slice_op);
    }
    IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
        &rewriter->builder, value, value_type, value_count, op->location,
        &inputs[input_count++]));
    const uint64_t suffix_ordinal = insertion_ordinal + value_count;
    if (suffix_ordinal < dest_count) {
      const int64_t static_offset = (int64_t)suffix_ordinal;
      const loom_type_t suffix_type = loom_vector_static_shape_flat_type(
          dest_type, dest_count - suffix_ordinal);
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          &rewriter->builder, flat_dest, NULL, 0, &static_offset, 1,
          suffix_type, op->location, &slice_op));
      inputs[input_count++] = loom_vector_slice_result(slice_op);
    }
    flat_result = inputs[0];
    if (input_count > 1) {
      loom_op_t* concat_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_vector_concat_build(&rewriter->builder, 0, inputs, input_count,
                                   flat_dest_type, op->location, &concat_op));
      flat_result = loom_vector_concat_result(concat_op);
    }
  } else {
    // A dynamic tail insertion uses the cheaper of whole-row vector selection
    // and inserting the flattened value's scalar lanes. Both forms are
    // element-type independent and have statically bounded expansion.
    loom_value_id_t flat_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
        &rewriter->builder, value, value_type, value_count, op->location,
        &flat_value));
    const loom_type_t flat_value_type =
        loom_vector_static_shape_flat_type(value_type, value_count);
    if (use_dynamic_row_select) {
      IREE_RETURN_IF_ERROR(
          loom_vector_static_shape_build_dynamic_tail_row_select(
              context, flat_value, flat_value_type, flat_dest, flat_dest_type,
              dynamic_prefix_ordinal, value_count, row_count, op->location,
              &flat_result));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_vector_static_shape_build_dynamic_tail_lane_insert(
              &rewriter->builder, flat_value, flat_value_type, flat_dest,
              flat_dest_type, dynamic_prefix_ordinal, value_count, op->location,
              &flat_result));
    }
  }

  IREE_RETURN_IF_ERROR(loom_vector_static_shape_restore_result(
      context, op, flat_result, flat_dest_type, dest_type, value_checkpoint));
  *out_rewritten = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_static_shape_table_lookup_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  const loom_value_id_t indices = loom_vector_table_lookup_indices(op);
  const loom_type_t index_type =
      loom_module_value_type(context->module, indices);
  const loom_type_t result_type = loom_module_value_type(
      context->module, loom_vector_table_lookup_result(op));
  uint64_t element_count = 0;
  if (loom_type_rank(result_type) <= 1 ||
      !loom_type_static_element_count(result_type, &element_count) ||
      element_count == 0 || element_count > INT64_MAX) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  // Lookup maps each index lane independently. Flattening the verified
  // index/result pair therefore preserves their row-major correspondence.
  loom_value_id_t flat_indices = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
      &rewriter->builder, indices, index_type, element_count, op->location,
      &flat_indices));
  const loom_type_t flat_result_type =
      loom_vector_static_shape_flat_type(result_type, element_count);
  loom_op_t* flat_lookup_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_table_lookup_build(
      &rewriter->builder, loom_vector_table_lookup_table(op), flat_indices,
      flat_result_type, op->location, &flat_lookup_op));
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_restore_result(
      context, op, loom_vector_table_lookup_result(flat_lookup_op),
      flat_result_type, result_type, value_checkpoint));
  *out_rewritten = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_static_shape_select_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_select_result(op));
  uint64_t element_count = 0;
  if (loom_type_rank(result_type) <= 1 ||
      !loom_type_static_element_count(result_type, &element_count) ||
      element_count == 0 || element_count > INT64_MAX) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t source_values[] = {
      loom_vector_select_condition(op),
      loom_vector_select_true_value(op),
      loom_vector_select_false_value(op),
  };
  loom_value_id_t flat_values[IREE_ARRAYSIZE(source_values)] = {0};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(source_values); ++i) {
    const loom_type_t source_type =
        loom_module_value_type(context->module, source_values[i]);
    IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
        &rewriter->builder, source_values[i], source_type, element_count,
        op->location, &flat_values[i]));
  }

  const loom_type_t flat_result_type =
      loom_vector_static_shape_flat_type(result_type, element_count);
  loom_op_t* flat_select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_select_build(
      &rewriter->builder, flat_values[0], flat_values[1], flat_values[2],
      flat_result_type, op->location, &flat_select_op));
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_restore_result(
      context, op, loom_vector_select_result(flat_select_op), flat_result_type,
      result_type, value_checkpoint));
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_vector_static_shape_rewrite_op(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  switch (op->kind) {
    case LOOM_OP_VECTOR_BROADCAST:
      return loom_vector_static_shape_broadcast_rewrite(context, op,
                                                        out_rewritten);
    case LOOM_OP_VECTOR_INSERT:
      return loom_vector_static_shape_insert_rewrite(context, op,
                                                     out_rewritten);
    case LOOM_OP_VECTOR_SLICE:
      return loom_vector_static_shape_slice_rewrite(context, op, out_rewritten);
    case LOOM_OP_VECTOR_TABLE_LOOKUP:
      return loom_vector_static_shape_table_lookup_rewrite(context, op,
                                                           out_rewritten);
    case LOOM_OP_VECTOR_SELECT:
      return loom_vector_static_shape_select_rewrite(context, op,
                                                     out_rewritten);
    default:
      return iree_ok_status();
  }
}
