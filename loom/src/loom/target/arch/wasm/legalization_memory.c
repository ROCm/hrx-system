// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/legalization_memory.h"

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

typedef struct loom_wasm_partial_memory_chunk_t {
  // Byte count covered by this scalar memory access.
  uint8_t byte_count;
  // Logical vector lane offset of this chunk.
  uint8_t lane_offset;
  // Logical vector lane count covered by this chunk.
  uint8_t lane_count;
} loom_wasm_partial_memory_chunk_t;

typedef struct loom_wasm_partial_memory_plan_t {
  // Original rank-one view type.
  loom_type_t view_type;
  // Original rank-one vector payload type.
  loom_type_t vector_type;
  // Byte count of one logical vector element.
  uint8_t element_byte_count;
  // Static logical lane count of the vector payload.
  uint8_t lane_count;
  // Power-of-two scalar accesses covering the payload in address order.
  loom_wasm_partial_memory_chunk_t chunks[4];
  // Number of populated chunks.
  uint8_t chunk_count;
} loom_wasm_partial_memory_plan_t;

static loom_scalar_type_t loom_wasm_partial_memory_integer_type(
    uint8_t byte_count) {
  switch (byte_count) {
    case 1:
      return LOOM_SCALAR_TYPE_I8;
    case 2:
      return LOOM_SCALAR_TYPE_I16;
    case 4:
      return LOOM_SCALAR_TYPE_I32;
    case 8:
      return LOOM_SCALAR_TYPE_I64;
    default:
      return LOOM_SCALAR_TYPE_NONE;
  }
}

static bool loom_wasm_partial_memory_plan(
    loom_type_t view_type, loom_type_t vector_type,
    loom_wasm_partial_memory_plan_t* out) {
  *out = (loom_wasm_partial_memory_plan_t){0};
  const loom_scalar_type_t element_type = loom_type_element_type(vector_type);
  if (!loom_type_is_view(view_type) || loom_type_rank(view_type) != 1 ||
      view_type.encoding_id != 0 || !loom_type_is_vector(vector_type) ||
      loom_type_rank(vector_type) != 1 ||
      !loom_type_is_all_static(vector_type) || vector_type.encoding_id != 0 ||
      !loom_type_element_type_equals(view_type, vector_type) ||
      !loom_scalar_type_set_contains(
          LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD | LOOM_SCALAR_TYPE_SET_FLOAT,
          element_type)) {
    return false;
  }

  const int32_t element_bit_count = loom_scalar_type_bitwidth(element_type);
  uint64_t lane_count = 0;
  if (element_bit_count <= 0 || element_bit_count > 64 ||
      element_bit_count % 8 != 0 ||
      !loom_type_static_element_count(vector_type, &lane_count) ||
      lane_count == 0 || lane_count > UINT8_MAX) {
    return false;
  }
  const uint32_t element_byte_count = (uint32_t)element_bit_count / 8u;
  uint64_t payload_byte_count = 0;
  if (!iree_checked_mul_u64(lane_count, element_byte_count,
                            &payload_byte_count) ||
      payload_byte_count == 0 || payload_byte_count >= 16u) {
    return false;
  }

  *out = (loom_wasm_partial_memory_plan_t){
      .view_type = view_type,
      .vector_type = vector_type,
      .element_byte_count = (uint8_t)element_byte_count,
      .lane_count = (uint8_t)lane_count,
  };
  uint32_t remaining_byte_count = (uint32_t)payload_byte_count;
  uint32_t lane_offset = 0;
  while (remaining_byte_count != 0) {
    uint32_t chunk_byte_count = 8u;
    while (chunk_byte_count > remaining_byte_count) {
      chunk_byte_count >>= 1;
    }
    if (chunk_byte_count < element_byte_count ||
        chunk_byte_count % element_byte_count != 0 ||
        out->chunk_count >= IREE_ARRAYSIZE(out->chunks)) {
      return false;
    }
    const uint32_t chunk_lane_count = chunk_byte_count / element_byte_count;
    out->chunks[out->chunk_count++] = (loom_wasm_partial_memory_chunk_t){
        .byte_count = (uint8_t)chunk_byte_count,
        .lane_offset = (uint8_t)lane_offset,
        .lane_count = (uint8_t)chunk_lane_count,
    };
    lane_offset += chunk_lane_count;
    remaining_byte_count -= chunk_byte_count;
  }
  return true;
}

static loom_type_t loom_wasm_partial_memory_view_type(
    loom_scalar_type_t element_type, uint8_t element_count, uint8_t alignment) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VIEW, element_type,
                                         loom_dim_pack_static(element_count),
                                         /*encoding_id=*/0);
  return loom_type_view_with_alignment(type, alignment);
}

static loom_type_t loom_wasm_partial_memory_vector_type(
    loom_scalar_type_t element_type, uint8_t element_count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type,
                             loom_dim_pack_static(element_count),
                             /*encoding_id=*/0);
}

static iree_status_t loom_wasm_partial_memory_build_chunk_view(
    loom_builder_t* builder, loom_value_id_t window,
    const loom_wasm_partial_memory_plan_t* plan,
    loom_wasm_partial_memory_chunk_t chunk, loom_location_id_t location,
    loom_value_id_t* out_view) {
  const uint8_t alignment = loom_type_view_alignment(plan->view_type);
  loom_value_id_t chunk_view = window;
  if (chunk.lane_offset != 0 || chunk.lane_count != plan->lane_count) {
    const int64_t static_offset = chunk.lane_offset;
    const loom_type_t chunk_view_type = loom_wasm_partial_memory_view_type(
        loom_type_element_type(plan->vector_type), chunk.lane_count, alignment);
    loom_op_t* chunk_view_op = NULL;
    IREE_RETURN_IF_ERROR(loom_view_subview_build(
        builder, window, /*offsets=*/NULL, 0, &static_offset,
        /*static_offsets_count=*/1, &chunk_view_type, /*result_count=*/1,
        location, &chunk_view_op));
    chunk_view = loom_view_subview_result(chunk_view_op);
  }

  const loom_scalar_type_t integer_type =
      loom_wasm_partial_memory_integer_type(chunk.byte_count);
  const loom_type_t scalar_view_type = loom_wasm_partial_memory_view_type(
      integer_type, /*element_count=*/1, alignment);
  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_bitcast_build(
      builder, chunk_view, scalar_view_type, location, &bitcast_op));
  *out_view = loom_view_bitcast_result(bitcast_op);
  return iree_ok_status();
}

static iree_status_t loom_wasm_partial_memory_build_window(
    loom_builder_t* builder, const loom_op_t* op,
    const loom_wasm_partial_memory_plan_t* plan, loom_value_id_t view,
    loom_value_slice_t dynamic_indices, loom_attribute_t static_indices,
    loom_value_id_t* out_window) {
  const loom_type_t window_type = loom_wasm_partial_memory_view_type(
      loom_type_element_type(plan->vector_type), plan->lane_count,
      loom_type_view_alignment(plan->view_type));
  loom_op_t* window_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_subview_build(
      builder, view, dynamic_indices.values, dynamic_indices.count,
      static_indices.i64_array, static_indices.count, &window_type,
      /*result_count=*/1, op->location, &window_op));
  *out_window = loom_view_subview_result(window_op);
  return iree_ok_status();
}

static iree_status_t loom_wasm_partial_memory_rewrite_load(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_wasm_partial_memory_plan_t* plan,
    loom_vector_memory_cache_policy_t cache_policy) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_value_id_t window = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_partial_memory_build_window(
      builder, op, plan, loom_vector_load_view(op),
      loom_vector_load_indices(op), loom_vector_load_static_indices(op),
      &window));

  loom_value_id_t fragments[IREE_ARRAYSIZE(plan->chunks)];
  for (uint8_t i = 0; i < plan->chunk_count; ++i) {
    const loom_wasm_partial_memory_chunk_t chunk = plan->chunks[i];
    const loom_scalar_type_t integer_type =
        loom_wasm_partial_memory_integer_type(chunk.byte_count);
    loom_value_id_t chunk_view = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_partial_memory_build_chunk_view(
        builder, window, plan, chunk, op->location, &chunk_view));

    const int64_t static_index = 0;
    loom_op_t* load_op = NULL;
    const loom_memory_access_flags_t chunk_memory_flags =
        loom_vector_load_memory_flags(op) &
        ~LOOM_MEMORY_ACCESS_FLAG_TYPE_DISJOINT;
    IREE_RETURN_IF_ERROR(loom_view_load_build(
        builder, cache_policy.build_flags, chunk_memory_flags, chunk_view,
        /*indices=*/NULL, 0, &static_index,
        /*static_indices_count=*/1, cache_policy.cache_scope,
        cache_policy.cache_temporal, loom_type_scalar(integer_type),
        op->location, &load_op));

    const loom_type_t integer_vector_type =
        loom_wasm_partial_memory_vector_type(integer_type, 1);
    loom_op_t* splat_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_splat_build(builder, loom_view_load_result(load_op),
                                integer_vector_type, op->location, &splat_op));
    fragments[i] = loom_vector_splat_result(splat_op);

    const loom_type_t fragment_type = loom_wasm_partial_memory_vector_type(
        loom_type_element_type(plan->vector_type), chunk.lane_count);
    if (!loom_type_equal(integer_vector_type, fragment_type)) {
      loom_op_t* bitcast_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_vector_bitcast_build(builder, fragments[i], integer_vector_type,
                                    fragment_type, op->location, &bitcast_op));
      fragments[i] = loom_vector_bitcast_result(bitcast_op);
    }
  }

  loom_value_id_t replacement = fragments[0];
  if (plan->chunk_count != 1 ||
      !loom_type_equal(loom_module_value_type(context->module, replacement),
                       plan->vector_type)) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        builder, /*axis=*/0, fragments, plan->chunk_count, plan->vector_type,
        op->location, &concat_op));
    replacement = loom_vector_concat_result(concat_op);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_wasm_partial_memory_rewrite_store(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_wasm_partial_memory_plan_t* plan,
    loom_vector_memory_cache_policy_t cache_policy) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);

  loom_value_id_t window = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_partial_memory_build_window(
      builder, op, plan, loom_vector_store_view(op),
      loom_vector_store_indices(op), loom_vector_store_static_indices(op),
      &window));

  const loom_value_id_t value = loom_vector_store_value(op);
  for (uint8_t i = 0; i < plan->chunk_count; ++i) {
    const loom_wasm_partial_memory_chunk_t chunk = plan->chunks[i];
    const loom_scalar_type_t integer_type =
        loom_wasm_partial_memory_integer_type(chunk.byte_count);
    const loom_type_t fragment_type = loom_wasm_partial_memory_vector_type(
        loom_type_element_type(plan->vector_type), chunk.lane_count);
    loom_value_id_t fragment = value;
    if (chunk.lane_offset != 0 || chunk.lane_count != plan->lane_count) {
      const int64_t static_offset = chunk.lane_offset;
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          builder, value, /*offsets=*/NULL, 0, &static_offset,
          /*static_offsets_count=*/1, fragment_type, op->location, &slice_op));
      fragment = loom_vector_slice_result(slice_op);
    }

    const loom_type_t integer_vector_type =
        loom_wasm_partial_memory_vector_type(integer_type, 1);
    if (!loom_type_equal(fragment_type, integer_vector_type)) {
      loom_op_t* bitcast_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
          builder, fragment, fragment_type, integer_vector_type, op->location,
          &bitcast_op));
      fragment = loom_vector_bitcast_result(bitcast_op);
    }

    const int64_t static_index = 0;
    loom_op_t* extract_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        builder, fragment, /*indices=*/NULL, 0, &static_index,
        /*static_indices_count=*/1, loom_type_scalar(integer_type),
        op->location, &extract_op));

    loom_value_id_t chunk_view = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_partial_memory_build_chunk_view(
        builder, window, plan, chunk, op->location, &chunk_view));
    loom_op_t* store_op = NULL;
    const loom_memory_access_flags_t chunk_memory_flags =
        loom_vector_store_memory_flags(op) &
        ~LOOM_MEMORY_ACCESS_FLAG_TYPE_DISJOINT;
    IREE_RETURN_IF_ERROR(loom_view_store_build(
        builder, cache_policy.build_flags, chunk_memory_flags,
        loom_vector_extract_result(extract_op), chunk_view, /*indices=*/NULL, 0,
        &static_index, /*static_indices_count=*/1, cache_policy.cache_scope,
        cache_policy.cache_temporal, op->location, &store_op));
  }
  return loom_rewriter_erase(rewriter, op);
}

iree_status_t loom_wasm_partial_vector_memory_rewrite_op(
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
  loom_wasm_partial_memory_plan_t plan;
  if (!loom_wasm_partial_memory_plan(
          loom_module_value_type(context->module, view),
          loom_module_value_type(context->module, vector_value), &plan)) {
    return iree_ok_status();
  }
  const loom_attribute_t static_indices =
      is_load ? loom_vector_load_static_indices(op)
              : loom_vector_store_static_indices(op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1) {
    return iree_ok_status();
  }
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, op,
                                               &cache_policy)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(is_load ? loom_wasm_partial_memory_rewrite_load(
                                     context, op, &plan, cache_policy)
                               : loom_wasm_partial_memory_rewrite_store(
                                     context, op, &plan, cache_policy));
  *out_rewritten = true;
  return iree_ok_status();
}
