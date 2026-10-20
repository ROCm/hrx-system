// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/shape_legalization.h"

#include <stdint.h>

#include "loom/analysis/motion.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/vector/construction.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

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

static bool loom_vector_static_shape_memory_is_contiguous(
    loom_type_t view_type, loom_type_t vector_type) {
  if (!loom_type_is_view(view_type) || !loom_type_is_vector(vector_type) ||
      loom_type_rank(vector_type) <= 1 ||
      loom_type_rank(vector_type) > loom_type_rank(view_type) ||
      !loom_type_is_all_static(view_type) ||
      !loom_type_is_all_static(vector_type) || view_type.encoding_id != 0 ||
      vector_type.encoding_id != 0 ||
      !loom_type_element_type_equals(view_type, vector_type)) {
    return false;
  }
  const uint8_t vector_rank = loom_type_rank(vector_type);
  const uint8_t first_vector_axis = loom_type_rank(view_type) - vector_rank;
  for (uint8_t vector_axis = 1; vector_axis < vector_rank; ++vector_axis) {
    if (loom_type_dim_static_size_at(vector_type, vector_axis) !=
        loom_type_dim_static_size_at(view_type,
                                     first_vector_axis + vector_axis)) {
      return false;
    }
  }
  return true;
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

// Re-expresses a multidimensional extraction over a static vector shape as a
// contiguous access into the flattened row-major carrier. The explicit
// leading indices select one complete tail value, so their prefix ordinal
// scales by the tail element count to produce the flat lane offset.
static iree_status_t loom_vector_static_shape_extract_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  const loom_value_id_t source = loom_vector_extract_source(op);
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_extract_result(op));
  const loom_attribute_t static_indices =
      loom_vector_extract_static_indices(op);
  const loom_value_slice_t dynamic_indices = loom_vector_extract_indices(op);
  uint64_t source_count = 0;
  uint64_t result_count = 1;
  if (loom_type_rank(source_type) <= 1 || static_indices.count == 0 ||
      !loom_type_static_element_count(source_type, &source_count) ||
      source_count == 0 || source_count > LOOM_DIM_MAX_STATIC_SIZE ||
      (loom_type_is_vector(result_type) &&
       (!loom_type_static_element_count(result_type, &result_count) ||
        result_count == 0 || result_count > source_count))) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t dynamic_prefix_ordinal = LOOM_VALUE_ID_INVALID;
  int64_t static_prefix_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_linearize_indices(
      &rewriter->builder, source_type, static_indices, dynamic_indices,
      op->location, &dynamic_prefix_ordinal, &static_prefix_ordinal));

  loom_value_id_t dynamic_offset = dynamic_prefix_ordinal;
  if (dynamic_offset != LOOM_VALUE_ID_INVALID && result_count != 1) {
    loom_op_t* scale_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        &rewriter->builder, loom_attr_i64((int64_t)result_count),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), op->location, &scale_op));
    loom_op_t* multiply_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_mul_build(
        &rewriter->builder, dynamic_offset,
        loom_index_constant_result(scale_op), op->location, &multiply_op));
    dynamic_offset = loom_index_mul_result(multiply_op);
  }
  int64_t static_offset = static_prefix_ordinal * (int64_t)result_count;

  loom_value_id_t flat_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
      &rewriter->builder, source, source_type, source_count, op->location,
      &flat_source));

  loom_value_id_t flat_result = LOOM_VALUE_ID_INVALID;
  loom_type_t flat_result_type = result_type;
  const loom_value_id_t* dynamic_offsets = NULL;
  iree_host_size_t dynamic_offset_count = 0;
  if (dynamic_offset != LOOM_VALUE_ID_INVALID) {
    dynamic_offsets = &dynamic_offset;
    dynamic_offset_count = 1;
    static_offset = INT64_MIN;
  }
  if (loom_type_is_vector(result_type)) {
    flat_result_type =
        loom_vector_static_shape_flat_type(result_type, result_count);
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        &rewriter->builder, flat_source, dynamic_offsets, dynamic_offset_count,
        &static_offset, /*static_offsets_count=*/1, flat_result_type,
        op->location, &slice_op));
    flat_result = loom_vector_slice_result(slice_op);
  } else {
    loom_op_t* extract_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        &rewriter->builder, flat_source, dynamic_offsets, dynamic_offset_count,
        &static_offset, /*static_indices_count=*/1, result_type, op->location,
        &extract_op));
    flat_result = loom_vector_extract_result(extract_op);
  }

  IREE_RETURN_IF_ERROR(loom_vector_static_shape_restore_result(
      context, op, flat_result, flat_result_type, result_type,
      value_checkpoint));
  *out_rewritten = true;
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

// Keeps an exclusive load and its rectangle together before structural
// expansion introduces independent row consumers. The selected footprint may
// move only across the shared motion analysis's ordinary-read intervals.
static iree_status_t loom_vector_static_shape_slice_load_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_type_t source_type, loom_type_t result_type, bool* out_rewritten) {
  const loom_value_id_t source = loom_vector_slice_source(op);
  const loom_value_t* source_value = loom_module_value(context->module, source);
  if (loom_value_is_block_arg(source_value)) {
    return iree_ok_status();
  }
  loom_op_t* source_op = loom_value_def_op(source_value);
  if (!loom_vector_load_isa(source_op) || source_type.encoding_id != 0 ||
      !loom_value_has_single_use(source_value) ||
      loom_value_has_attribute_uses(source_value) ||
      loom_module_value_has_type_uses(context->module, source) ||
      !loom_motion_op_is_ordinary_load(context->module, source_op)) {
    return iree_ok_status();
  }
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, source_op,
                                               &cache_policy)) {
    return iree_ok_status();
  }
  if (context->read_motion_barriers == NULL) {
    IREE_RETURN_IF_ERROR(loom_motion_read_barrier_table_allocate(
        context->module, context->arena, &context->read_motion_barriers));
  }
  bool can_cross = false;
  IREE_RETURN_IF_ERROR(loom_motion_read_barrier_table_can_cross(
      context->read_motion_barriers, source_op, op, &can_cross));
  if (!can_cross) {
    return iree_ok_status();
  }

  const loom_value_id_t view = loom_vector_load_view(source_op);
  const loom_type_t view_type = loom_module_value_type(context->module, view);
  const uint8_t view_rank = loom_type_rank(view_type);
  const uint8_t first_vector_axis = view_rank - loom_type_rank(source_type);
  const loom_attribute_t load_origins =
      loom_vector_load_static_indices(source_op);
  const loom_value_slice_t load_indices = loom_vector_load_indices(source_op);
  const loom_attribute_t slice_origins = loom_vector_slice_static_offsets(op);
  const loom_value_slice_t slice_indices = loom_vector_slice_offsets(op);
  int64_t static_indices[LOOM_TYPE_MAX_RANK];
  loom_value_id_t dynamic_indices[LOOM_TYPE_MAX_RANK];
  uint16_t load_index_position = 0;
  uint16_t slice_index_position = 0;
  uint16_t dynamic_index_count = 0;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    int64_t static_index = load_origins.i64_array[axis];
    loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
    if (static_index == INT64_MIN) {
      dynamic_index = load_indices.values[load_index_position++];
      static_index = 0;
    }
    if (axis >= first_vector_axis) {
      const int64_t slice_origin =
          slice_origins.i64_array[axis - first_vector_axis];
      if (slice_origin == INT64_MIN) {
        const loom_value_id_t slice_index =
            slice_indices.values[slice_index_position++];
        if (dynamic_index == LOOM_VALUE_ID_INVALID) {
          dynamic_index = slice_index;
        } else {
          loom_op_t* add_op = NULL;
          IREE_RETURN_IF_ERROR(loom_index_add_build(builder, dynamic_index,
                                                    slice_index, index_type,
                                                    op->location, &add_op));
          dynamic_index = loom_index_add_result(add_op);
        }
      } else {
        static_index += slice_origin;
      }
    }
    static_indices[axis] = static_index;
    if (dynamic_index != LOOM_VALUE_ID_INVALID) {
      if (static_index != 0) {
        loom_op_t* constant_op = NULL;
        IREE_RETURN_IF_ERROR(
            loom_index_constant_build(builder, loom_attr_i64(static_index),
                                      index_type, op->location, &constant_op));
        loom_op_t* add_op = NULL;
        IREE_RETURN_IF_ERROR(loom_index_add_build(
            builder, dynamic_index, loom_index_constant_result(constant_op),
            index_type, op->location, &add_op));
        dynamic_index = loom_index_add_result(add_op);
      }
      static_indices[axis] = INT64_MIN;
      dynamic_indices[dynamic_index_count++] = dynamic_index;
    }
  }

  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      builder, cache_policy.build_flags,
      loom_vector_load_memory_flags(source_op), view, dynamic_indices,
      dynamic_index_count, static_indices, view_rank, cache_policy.cache_scope,
      cache_policy.cache_temporal, result_type, op->location, &load_op));
  const loom_value_id_t replacement = loom_vector_load_result(load_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, source_op));
  *out_rewritten = true;
  return iree_ok_status();
}

// Maps a result-relative ordinal to its displacement in the source rectangle.
// The common origin is linearized once, independently of these static strides.
static uint64_t loom_vector_static_shape_slice_source_offset(
    loom_type_t source_type, loom_type_t result_type, uint64_t result_ordinal) {
  const uint8_t rank = loom_type_rank(source_type);
  uint64_t source_offset = 0;
  uint64_t source_stride = 1;
  for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
    const uint8_t axis = rank - reverse_axis - 1;
    const uint64_t result_extent =
        (uint64_t)loom_type_dim_static_size_at(result_type, axis);
    source_offset += (result_ordinal % result_extent) * source_stride;
    result_ordinal /= result_extent;
    source_stride *= (uint64_t)loom_type_dim_static_size_at(source_type, axis);
  }
  return source_offset;
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
  const loom_value_slice_t dynamic_offsets = loom_vector_slice_offsets(op);
  uint64_t source_count = 0;
  uint64_t result_count = 0;
  if (rank <= 1 ||
      !loom_type_static_element_count(source_type, &source_count) ||
      !loom_type_static_element_count(result_type, &result_count) ||
      source_count == 0 || result_count == 0 || source_count > INT64_MAX ||
      result_count > INT64_MAX) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_static_shape_slice_load_rewrite(
      context, op, source_type, result_type, out_rewritten));
  if (*out_rewritten) {
    return iree_ok_status();
  }

  // Complete trailing dimensions form one contiguous run. Only the remaining
  // leading coordinates introduce gaps; run geometry does not depend on the
  // runtime origin. This avoids enumerating every row to discover adjacency.
  uint8_t run_axis = rank - 1;
  uint64_t run_length =
      (uint64_t)loom_type_dim_static_size_at(result_type, run_axis);
  while (run_axis > 0 &&
         loom_type_dim_static_size_at(result_type, run_axis) ==
             loom_type_dim_static_size_at(source_type, run_axis)) {
    --run_axis;
    run_length *= (uint64_t)loom_type_dim_static_size_at(result_type, run_axis);
  }
  const uint64_t run_count = result_count / run_length;
  // Two shape bitcasts and one concat surround the slices. A runtime origin
  // adds at most two operations per axis and a constant/add for later runs.
  const uint32_t setup_operation_count =
      3u + (dynamic_offsets.count != 0 ? 2u * rank - 2u : 0u);
  const uint32_t run_operation_count = dynamic_offsets.count != 0 ? 3u : 1u;
  if (run_count > (LOOM_VECTOR_STATIC_SHAPE_OP_LIMIT - setup_operation_count) /
                      run_operation_count) {
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
  loom_value_id_t dynamic_origin = LOOM_VALUE_ID_INVALID;
  int64_t static_origin = 0;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_linearize_indices(
      &rewriter->builder, source_type, static_offsets, dynamic_offsets,
      op->location, &dynamic_origin, &static_origin));
  loom_value_id_t flat_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
      &rewriter->builder, source, source_type, source_count, op->location,
      &flat_source));

  const loom_type_t flat_result_type =
      loom_vector_static_shape_flat_type(result_type, result_count);
  if (dynamic_origin == LOOM_VALUE_ID_INVALID && static_origin == 0 &&
      result_count == source_count) {
    runs[0] = flat_source;
  } else {
    const loom_type_t run_type =
        loom_vector_static_shape_flat_type(result_type, run_length);
    const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    for (uint64_t run = 0; run < run_count; ++run) {
      const int64_t relative_offset =
          (int64_t)loom_vector_static_shape_slice_source_offset(
              source_type, result_type, run * run_length);
      loom_value_id_t dynamic_offset = dynamic_origin;
      int64_t static_offset = static_origin + relative_offset;
      uint16_t dynamic_offset_count = 0;
      if (dynamic_origin != LOOM_VALUE_ID_INVALID) {
        if (relative_offset != 0) {
          loom_op_t* displacement_op = NULL;
          IREE_RETURN_IF_ERROR(loom_index_constant_build(
              &rewriter->builder, loom_attr_i64(relative_offset), index_type,
              op->location, &displacement_op));
          loom_op_t* add_op = NULL;
          IREE_RETURN_IF_ERROR(
              loom_index_add_build(&rewriter->builder, dynamic_origin,
                                   loom_index_constant_result(displacement_op),
                                   index_type, op->location, &add_op));
          dynamic_offset = loom_index_add_result(add_op);
        }
        static_offset = INT64_MIN;
        dynamic_offset_count = 1;
      }
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          &rewriter->builder, flat_source, &dynamic_offset,
          dynamic_offset_count, &static_offset, 1, run_type, op->location,
          &slice_op));
      runs[run] = loom_vector_slice_result(slice_op);
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

// Concatenation along the leading axis appends complete row-major payloads.
// Flattening each input therefore preserves the authored element order without
// introducing lane movement. Other axes interleave input rows and require a
// distinct shuffle legalization.
static iree_status_t loom_vector_static_shape_concat_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  const loom_value_slice_t inputs = loom_vector_concat_inputs(op);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_concat_result(op));
  uint64_t result_count = 0;
  if (loom_vector_concat_axis(op) != 0 || loom_type_rank(result_type) <= 1 ||
      inputs.count == 0 ||
      inputs.count + 2u > LOOM_VECTOR_STATIC_SHAPE_OP_LIMIT ||
      !loom_type_static_element_count(result_type, &result_count) ||
      result_count == 0 || result_count > LOOM_DIM_MAX_STATIC_SIZE) {
    return iree_ok_status();
  }

  uint64_t input_total_count = 0;
  for (iree_host_size_t i = 0; i < inputs.count; ++i) {
    const loom_type_t input_type =
        loom_module_value_type(context->module, inputs.values[i]);
    uint64_t input_count = 0;
    if (!loom_type_static_element_count(input_type, &input_count) ||
        input_count == 0 || input_count > result_count ||
        input_total_count > result_count - input_count) {
      return iree_ok_status();
    }
    input_total_count += input_count;
  }
  if (input_total_count != result_count) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t* flat_inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, inputs.count,
                                                 sizeof(*flat_inputs),
                                                 (void**)&flat_inputs));
  for (iree_host_size_t i = 0; i < inputs.count; ++i) {
    const loom_type_t input_type =
        loom_module_value_type(context->module, inputs.values[i]);
    uint64_t input_count = 0;
    (void)loom_type_static_element_count(input_type, &input_count);
    IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
        &rewriter->builder, inputs.values[i], input_type, input_count,
        op->location, &flat_inputs[i]));
  }

  const loom_type_t flat_result_type =
      loom_vector_static_shape_flat_type(result_type, result_count);
  loom_op_t* flat_concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_concat_build(
      &rewriter->builder, /*axis=*/0, flat_inputs, inputs.count,
      flat_result_type, op->location, &flat_concat_op));
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_restore_result(
      context, op, loom_vector_concat_result(flat_concat_op), flat_result_type,
      result_type, value_checkpoint));
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_vector_static_shape_rewrite_op(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  IREE_RETURN_IF_ERROR(loom_vector_project_regular_partition(
      op, context->rewriter, out_rewritten));
  if (*out_rewritten) {
    return iree_ok_status();
  }
  switch (op->kind) {
    case LOOM_OP_VECTOR_BROADCAST:
      return loom_vector_static_shape_broadcast_rewrite(context, op,
                                                        out_rewritten);
    case LOOM_OP_VECTOR_CONCAT:
      return loom_vector_static_shape_concat_rewrite(context, op,
                                                     out_rewritten);
    case LOOM_OP_VECTOR_EXTRACT:
      return loom_vector_static_shape_extract_rewrite(context, op,
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

static iree_status_t loom_vector_static_shape_contiguous_memory_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  const bool is_load = loom_vector_load_isa(op);
  const bool is_store = loom_vector_store_isa(op);
  if (!is_load && !is_store) {
    return iree_ok_status();
  }

  const loom_value_id_t view =
      is_load ? loom_vector_load_view(op) : loom_vector_store_view(op);
  const loom_value_id_t vector_value =
      is_load ? loom_vector_load_result(op) : loom_vector_store_value(op);
  const loom_type_t view_type = loom_module_value_type(context->module, view);
  const loom_type_t vector_type =
      loom_module_value_type(context->module, vector_value);
  uint64_t view_element_count = 0;
  uint64_t vector_element_count = 0;
  if (!loom_vector_static_shape_memory_is_contiguous(view_type, vector_type) ||
      !loom_type_static_element_count(view_type, &view_element_count) ||
      !loom_type_static_element_count(vector_type, &vector_element_count) ||
      view_element_count == 0 || vector_element_count == 0 ||
      view_element_count > LOOM_DIM_MAX_STATIC_SIZE ||
      vector_element_count > view_element_count) {
    return iree_ok_status();
  }

  const loom_attribute_t static_indices =
      is_load ? loom_vector_load_static_indices(op)
              : loom_vector_store_static_indices(op);
  const loom_value_slice_t dynamic_indices =
      is_load ? loom_vector_load_indices(op) : loom_vector_store_indices(op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != loom_type_rank(view_type)) {
    return iree_ok_status();
  }
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, op,
                                               &cache_policy)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  int64_t static_index = 0;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_linearize_indices(
      builder, view_type, static_indices, dynamic_indices, op->location,
      &dynamic_index, &static_index));

  loom_type_t flat_view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, loom_type_element_type(view_type),
      loom_dim_pack_static((int64_t)view_element_count), /*encoding_id=*/0);
  flat_view_type = loom_type_view_with_alignment(
      flat_view_type, loom_type_view_alignment(view_type));
  loom_op_t* flat_view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_bitcast_build(builder, view, flat_view_type,
                                               op->location, &flat_view_op));
  const loom_type_t flat_vector_type =
      loom_vector_static_shape_flat_type(vector_type, vector_element_count);
  const loom_value_id_t* flat_indices = NULL;
  iree_host_size_t flat_index_count = 0;
  if (dynamic_index != LOOM_VALUE_ID_INVALID) {
    flat_indices = &dynamic_index;
    flat_index_count = 1;
    static_index = INT64_MIN;
  }

  if (is_load) {
    loom_op_t* flat_load_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_load_build(
        builder, cache_policy.build_flags, loom_vector_load_memory_flags(op),
        loom_view_bitcast_result(flat_view_op), flat_indices, flat_index_count,
        &static_index, /*static_indices_count=*/1, cache_policy.cache_scope,
        cache_policy.cache_temporal, flat_vector_type, op->location,
        &flat_load_op));
    IREE_RETURN_IF_ERROR(loom_vector_static_shape_restore_result(
        context, op, loom_vector_load_result(flat_load_op), flat_vector_type,
        vector_type, value_checkpoint));
  } else {
    loom_value_id_t flat_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
        builder, vector_value, vector_type, vector_element_count, op->location,
        &flat_value));
    loom_op_t* flat_store_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_store_build(
        builder, cache_policy.build_flags, loom_vector_store_memory_flags(op),
        flat_value, loom_view_bitcast_result(flat_view_op), flat_indices,
        flat_index_count, &static_index, /*static_indices_count=*/1,
        cache_policy.cache_scope, cache_policy.cache_temporal, op->location,
        &flat_store_op));
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  }
  *out_rewritten = true;
  return iree_ok_status();
}

static iree_host_size_t loom_vector_static_shape_memory_dynamic_index_position(
    loom_attribute_t static_indices, uint8_t axis) {
  iree_host_size_t position = 0;
  for (uint8_t i = 0; i < axis; ++i) {
    if (static_indices.i64_array[i] == INT64_MIN) {
      ++position;
    }
  }
  return position;
}

static bool loom_vector_static_shape_row_memory_shape(
    loom_type_t view_type, loom_type_t vector_type,
    loom_attribute_t static_indices, bool is_store, uint32_t* out_row_count,
    uint32_t* out_row_lane_count, uint64_t* out_element_count) {
  *out_row_count = 0;
  *out_row_lane_count = 0;
  *out_element_count = 0;
  if (!loom_type_is_view(view_type) || !loom_type_is_vector(vector_type) ||
      loom_type_rank(vector_type) <= 1 ||
      loom_type_rank(vector_type) > loom_type_rank(view_type) ||
      !loom_type_is_all_static(vector_type) || view_type.encoding_id != 0 ||
      vector_type.encoding_id != 0 ||
      !loom_type_element_type_equals(view_type, vector_type) ||
      static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != loom_type_rank(view_type)) {
    return false;
  }

  uint64_t element_count = 0;
  if (!loom_type_static_element_count(vector_type, &element_count) ||
      element_count == 0 || element_count > LOOM_DIM_MAX_STATIC_SIZE) {
    return false;
  }
  const uint8_t vector_rank = loom_type_rank(vector_type);
  const int64_t row_lane_count =
      loom_type_dim_static_size_at(vector_type, vector_rank - 1u);
  if (row_lane_count < 1 || row_lane_count > UINT32_MAX ||
      element_count % (uint64_t)row_lane_count != 0) {
    return false;
  }
  const uint64_t row_count = element_count / (uint64_t)row_lane_count;
  if (row_count == 0 || row_count > UINT32_MAX) {
    return false;
  }

  const uint8_t first_vector_axis = loom_type_rank(view_type) - vector_rank;
  uint64_t index_operation_count = 0;
  for (uint8_t vector_axis = 0; vector_axis + 1u < vector_rank; ++vector_axis) {
    const uint8_t view_axis = first_vector_axis + vector_axis;
    const int64_t origin = static_indices.i64_array[view_axis];
    if (origin == INT64_MIN) {
      const uint64_t dimension_size =
          (uint64_t)loom_type_dim_static_size_at(vector_type, vector_axis);
      const uint64_t nonzero_coordinate_count =
          row_count - row_count / dimension_size;
      index_operation_count += nonzero_coordinate_count * 2u;
      continue;
    }
    const int64_t maximum_row_offset =
        loom_type_dim_static_size_at(vector_type, vector_axis) - 1;
    int64_t unused_adjusted_origin = 0;
    if (!iree_checked_add_i64(origin, maximum_row_offset,
                              &unused_adjusted_origin)) {
      return false;
    }
  }
  const uint64_t structural_operation_count =
      is_store ? row_count * 2u + 1u : row_count + 2u;
  if (structural_operation_count + index_operation_count >
      LOOM_VECTOR_STATIC_SHAPE_OP_LIMIT) {
    return false;
  }

  *out_row_count = (uint32_t)row_count;
  *out_row_lane_count = (uint32_t)row_lane_count;
  *out_element_count = element_count;
  return true;
}

static iree_status_t loom_vector_static_shape_build_row_memory_origin(
    loom_target_legalization_context_t* context, loom_builder_t* builder,
    loom_type_t view_type, loom_type_t vector_type,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    uint32_t row_ordinal, loom_location_id_t location,
    const int64_t** out_static_indices,
    const loom_value_id_t** out_dynamic_indices) {
  int64_t* row_static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, static_indices.count, sizeof(*row_static_indices),
      (void**)&row_static_indices));
  for (uint8_t i = 0; i < static_indices.count; ++i) {
    row_static_indices[i] = static_indices.i64_array[i];
  }

  loom_value_id_t* row_dynamic_indices = NULL;
  if (dynamic_indices.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, dynamic_indices.count, sizeof(*row_dynamic_indices),
        (void**)&row_dynamic_indices));
    for (iree_host_size_t i = 0; i < dynamic_indices.count; ++i) {
      row_dynamic_indices[i] = dynamic_indices.values[i];
    }
  }

  const uint8_t vector_rank = loom_type_rank(vector_type);
  const uint8_t first_vector_axis = loom_type_rank(view_type) - vector_rank;
  uint32_t remaining_ordinal = row_ordinal;
  for (uint8_t reverse_axis = vector_rank - 1u; reverse_axis > 0;
       --reverse_axis) {
    const uint8_t vector_axis = reverse_axis - 1u;
    const uint32_t dimension_size =
        (uint32_t)loom_type_dim_static_size_at(vector_type, vector_axis);
    const uint32_t coordinate = remaining_ordinal % dimension_size;
    remaining_ordinal /= dimension_size;
    if (coordinate == 0) {
      continue;
    }

    const uint8_t view_axis = first_vector_axis + vector_axis;
    if (row_static_indices[view_axis] != INT64_MIN) {
      row_static_indices[view_axis] += (int64_t)coordinate;
      continue;
    }

    const iree_host_size_t dynamic_position =
        loom_vector_static_shape_memory_dynamic_index_position(static_indices,
                                                               view_axis);
    const loom_value_id_t origin = row_dynamic_indices[dynamic_position];
    const loom_type_t origin_type =
        loom_module_value_type(context->module, origin);
    loom_op_t* coordinate_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_constant_build(builder, loom_attr_i64(coordinate),
                                  origin_type, location, &coordinate_op));
    loom_op_t* adjusted_origin_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_add_build(
        builder, origin, loom_index_constant_result(coordinate_op), origin_type,
        location, &adjusted_origin_op));
    row_dynamic_indices[dynamic_position] =
        loom_index_add_result(adjusted_origin_op);
  }

  *out_static_indices = row_static_indices;
  *out_dynamic_indices = row_dynamic_indices;
  return iree_ok_status();
}

static iree_status_t loom_vector_static_shape_row_memory_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  const bool is_load = loom_vector_load_isa(op);
  const bool is_store = loom_vector_store_isa(op);
  if (!is_load && !is_store) {
    return iree_ok_status();
  }

  const loom_value_id_t view =
      is_load ? loom_vector_load_view(op) : loom_vector_store_view(op);
  const loom_value_id_t vector_value =
      is_load ? loom_vector_load_result(op) : loom_vector_store_value(op);
  const loom_type_t view_type = loom_module_value_type(context->module, view);
  const loom_type_t vector_type =
      loom_module_value_type(context->module, vector_value);
  const loom_attribute_t static_indices =
      is_load ? loom_vector_load_static_indices(op)
              : loom_vector_store_static_indices(op);
  const loom_value_slice_t dynamic_indices =
      is_load ? loom_vector_load_indices(op) : loom_vector_store_indices(op);
  uint32_t row_count = 0;
  uint32_t row_lane_count = 0;
  uint64_t element_count = 0;
  if (!loom_vector_static_shape_row_memory_shape(
          view_type, vector_type, static_indices, is_store, &row_count,
          &row_lane_count, &element_count)) {
    return iree_ok_status();
  }
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, op,
                                               &cache_policy)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t row_type =
      loom_vector_static_shape_flat_type(vector_type, (uint64_t)row_lane_count);
  const loom_type_t flat_vector_type =
      loom_vector_static_shape_flat_type(vector_type, element_count);
  loom_value_id_t flat_store_value = LOOM_VALUE_ID_INVALID;
  if (is_store) {
    IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
        builder, vector_value, vector_type, element_count, op->location,
        &flat_store_value));
  }
  loom_value_id_t* rows = NULL;
  if (is_load) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, row_count, sizeof(*rows), (void**)&rows));
  }

  for (uint32_t row = 0; row < row_count; ++row) {
    const int64_t* row_static_indices = NULL;
    const loom_value_id_t* row_dynamic_indices = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_static_shape_build_row_memory_origin(
        context, builder, view_type, vector_type, static_indices,
        dynamic_indices, row, op->location, &row_static_indices,
        &row_dynamic_indices));
    if (is_load) {
      loom_op_t* row_load_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_load_build(
          builder, cache_policy.build_flags, loom_vector_load_memory_flags(op),
          view, row_dynamic_indices, dynamic_indices.count, row_static_indices,
          static_indices.count, cache_policy.cache_scope,
          cache_policy.cache_temporal, row_type, op->location, &row_load_op));
      rows[row] = loom_vector_load_result(row_load_op);
      continue;
    }

    loom_value_id_t row_value = flat_store_value;
    if (row_count != 1) {
      const int64_t row_lane_offset = (int64_t)row * row_lane_count;
      loom_op_t* row_slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          builder, flat_store_value, /*offsets=*/NULL, /*offsets_count=*/0,
          &row_lane_offset, /*static_offsets_count=*/1, row_type, op->location,
          &row_slice_op));
      row_value = loom_vector_slice_result(row_slice_op);
    }
    loom_op_t* row_store_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_store_build(
        builder, cache_policy.build_flags, loom_vector_store_memory_flags(op),
        row_value, view, row_dynamic_indices, dynamic_indices.count,
        row_static_indices, static_indices.count, cache_policy.cache_scope,
        cache_policy.cache_temporal, op->location, &row_store_op));
  }

  if (is_load) {
    loom_value_id_t flat_result = rows[0];
    if (row_count != 1) {
      loom_op_t* concat_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_concat_build(builder, /*axis=*/0, rows,
                                                    row_count, flat_vector_type,
                                                    op->location, &concat_op));
      flat_result = loom_vector_concat_result(concat_op);
    }
    IREE_RETURN_IF_ERROR(loom_vector_static_shape_restore_result(
        context, op, flat_result, flat_vector_type, vector_type,
        value_checkpoint));
  } else {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  }
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_vector_static_shape_memory_rewrite_op(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_contiguous_memory_rewrite(
      context, op, out_rewritten));
  if (*out_rewritten) {
    return iree_ok_status();
  }
  return loom_vector_static_shape_row_memory_rewrite(context, op,
                                                     out_rewritten);
}
