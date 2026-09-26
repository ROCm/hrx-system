// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/shape_legalization.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"

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
  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
      &context->rewriter->builder, flat_value, flat_type, result_type,
      op->location, &bitcast_op));
  loom_value_id_t replacement = loom_vector_bitcast_result(bitcast_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(context->rewriter, op,
                                                  &replacement, 1);
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
  if (loom_type_rank(result_type) <= 1 ||
      !loom_type_static_element_count(source_type, &source_count) ||
      !loom_type_static_element_count(result_type, &result_count) ||
      source_count == 0 || result_count == 0 || source_count > INT64_MAX ||
      result_count > INT64_MAX || result_count % source_count != 0) {
    return iree_ok_status();
  }

  const uint64_t packet_count = result_count / source_count;
  if (packet_count > IREE_HOST_SIZE_MAX || source_count > IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }
  loom_value_id_t single_packet = LOOM_VALUE_ID_INVALID;
  loom_value_id_t* packets = &single_packet;
  if (packet_count > 1) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, (iree_host_size_t)packet_count, sizeof(*packets),
        (void**)&packets));
  }
  int64_t single_source_lane = 0;
  int64_t* source_lanes = &single_source_lane;
  if (source_count > 1) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, (iree_host_size_t)source_count, sizeof(*source_lanes),
        (void**)&source_lanes));
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t flat_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
      &rewriter->builder, source, source_type, source_count, op->location,
      &flat_source));
  const loom_type_t flat_source_type =
      loom_vector_static_shape_flat_type(source_type, source_count);
  // A shuffle preserves its source width, so partition the result into
  // source-width packets before concatenating the complete flat carrier.
  for (uint64_t packet = 0; packet < packet_count; ++packet) {
    bool identity = true;
    for (uint64_t lane = 0; lane < source_count; ++lane) {
      const uint64_t result_ordinal = packet * source_count + lane;
      const uint64_t source_ordinal =
          loom_vector_static_shape_broadcast_source_ordinal(
              source_type, result_type, result_ordinal);
      source_lanes[lane] = (int64_t)source_ordinal;
      identity &= source_ordinal == lane;
    }
    packets[packet] = flat_source;
    if (!identity) {
      loom_op_t* shuffle_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_shuffle_build(
          &rewriter->builder, source_lanes, (iree_host_size_t)source_count,
          flat_source, flat_source_type, op->location, &shuffle_op));
      packets[packet] = loom_vector_shuffle_result(shuffle_op);
    }
  }

  const loom_type_t flat_result_type =
      loom_vector_static_shape_flat_type(result_type, result_count);
  loom_value_id_t flat_result = packets[0];
  if (packet_count > 1) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        &rewriter->builder, 0, packets, (iree_host_size_t)packet_count,
        flat_result_type, op->location, &concat_op));
    flat_result = loom_vector_concat_result(concat_op);
  }
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_restore_result(
      context, op, flat_result, flat_result_type, result_type,
      value_checkpoint));
  *out_rewritten = true;
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
  uint64_t dest_count = 0;
  if (loom_type_rank(dest_type) <= 1 ||
      !loom_type_static_element_count(dest_type, &dest_count) ||
      dest_count == 0 || dest_count > INT64_MAX ||
      static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      loom_vector_insert_indices(op).count != 0) {
    return iree_ok_status();
  }

  // A tail insertion covers one contiguous row-major range. Linearize its
  // leading coordinates and retain the untouched prefix and suffix as slices.
  uint64_t prefix_ordinal = 0;
  for (uint16_t axis = 0; axis < static_indices.count; ++axis) {
    const int64_t index = static_indices.i64_array[axis];
    if (index < 0) {
      return iree_ok_status();
    }
    const uint64_t extent =
        (uint64_t)loom_type_dim_static_size_at(dest_type, (uint8_t)axis);
    prefix_ordinal = prefix_ordinal * extent + (uint64_t)index;
  }
  uint64_t value_count = 1;
  if (loom_type_is_vector(value_type) &&
      !loom_type_static_element_count(value_type, &value_count)) {
    return iree_ok_status();
  }
  const uint64_t insertion_ordinal = prefix_ordinal * value_count;
  if (value_count == 0 || insertion_ordinal > dest_count ||
      value_count > dest_count - insertion_ordinal) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t flat_dest = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_static_shape_flatten_value(
      &rewriter->builder, dest, dest_type, dest_count, op->location,
      &flat_dest));
  const loom_type_t flat_dest_type =
      loom_vector_static_shape_flat_type(dest_type, dest_count);

  loom_value_id_t flat_result = LOOM_VALUE_ID_INVALID;
  if (!loom_type_is_vector(value_type)) {
    const int64_t static_index = (int64_t)insertion_ordinal;
    loom_op_t* insert_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_insert_build(
        &rewriter->builder, value, flat_dest, NULL, 0, &static_index, 1,
        flat_dest_type, op->location, &insert_op));
    flat_result = loom_vector_insert_result(insert_op);
  } else {
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
    case LOOM_OP_VECTOR_TABLE_LOOKUP:
      return loom_vector_static_shape_table_lookup_rewrite(context, op,
                                                           out_rewritten);
    default:
      return iree_ok_status();
  }
}
