// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/legalization.h"

#include <string.h>

#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/transforms/vector/to_scalar.h"

static bool loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id == LOOM_AMDGPU_TARGET_STABLE_ID;
}

static bool loom_amdgpu_subgroup_mask_type_covers_wavefront(
    loom_type_t mask_type, uint32_t wavefront_size) {
  if (!loom_type_is_scalar(mask_type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(mask_type);
  if (!loom_scalar_type_is_integer(scalar_type)) {
    return false;
  }
  return (uint32_t)loom_scalar_type_bitwidth(scalar_type) >= wavefront_size;
}

static uint32_t loom_amdgpu_legalizer_wavefront_size(
    const loom_target_legalization_context_t* context) {
  if (context->bundle == NULL || context->bundle->snapshot == NULL) {
    return 0;
  }
  return context->bundle->snapshot->subgroup_size;
}

static loom_target_legalizer_action_t loom_amdgpu_defer_or_reject_final(
    const loom_target_legalization_context_t* context) {
  return context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL
             ? LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL
             : LOOM_TARGET_LEGALIZER_ACTION_DEFER;
}

static iree_status_t loom_amdgpu_retain_native_vector_op(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  (void)op;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
  };
  return iree_ok_status();
}

static bool loom_amdgpu_match_value_type_is_supported(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static iree_status_t loom_amdgpu_build_i32_constant(
    loom_builder_t* builder, int64_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      location, &op));
  *out_value = loom_scalar_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_build_zero_mask(loom_builder_t* builder,
                                                 loom_type_t mask_type,
                                                 loom_location_id_t location,
                                                 loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, loom_attr_i64(0),
                                                  mask_type, location, &op));
  *out_value = loom_scalar_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_build_match_any_lane_step(
    loom_builder_t* builder, loom_value_id_t value, loom_type_t value_type,
    loom_value_id_t lane_id_i32, uint32_t source_lane_index,
    loom_value_id_t current_mask, loom_type_t mask_type,
    loom_location_id_t location, loom_value_id_t* out_next_mask) {
  *out_next_mask = LOOM_VALUE_ID_INVALID;
  const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_i32_constant(
      builder, source_lane_index, location, &source_lane));

  loom_op_t* is_source_lane_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, lane_id_i32, source_lane,
      i32_type, i1_type, location, &is_source_lane_op));
  const loom_value_id_t is_source_lane =
      loom_scalar_cmpi_result(is_source_lane_op);

  loom_op_t* source_active_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_vote_any_build(
      builder, is_source_lane, i1_type, location, &source_active_op));
  const loom_value_id_t source_active =
      loom_kernel_subgroup_vote_any_result(source_active_op);

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                        source_active, &mask_type, 1, /*tied_results=*/NULL,
                        /*tied_result_count=*/0, location, &if_op));

  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(builder, if_op, loom_scf_if_then_region(if_op));
  loom_op_t* broadcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_broadcast_build(
      builder, value, source_lane, value_type, location, &broadcast_op));
  const loom_value_id_t source_value =
      loom_kernel_subgroup_broadcast_result(broadcast_op);

  loom_op_t* equal_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, value, source_value, value_type,
      i1_type, location, &equal_op));
  const loom_value_id_t equal = loom_scalar_cmpi_result(equal_op);

  loom_op_t* equivalence_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_vote_ballot_build(
      builder, equal, mask_type, location, &equivalence_mask_op));
  const loom_value_id_t equivalence_mask =
      loom_kernel_subgroup_vote_ballot_mask(equivalence_mask_op);

  loom_op_t* selected_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, is_source_lane, equivalence_mask, current_mask, mask_type,
      location, &selected_mask_op));
  const loom_value_id_t selected_mask =
      loom_scf_select_result(selected_mask_op);
  loom_op_t* then_yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &selected_mask, 1,
                                            location, &then_yield_op));
  loom_builder_restore(builder, saved_ip);

  saved_ip =
      loom_builder_enter_region(builder, if_op, loom_scf_if_else_region(if_op));
  loom_op_t* else_yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &current_mask, 1, location,
                                            &else_yield_op));
  loom_builder_restore(builder, saved_ip);

  *out_next_mask = loom_scf_if_results(if_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_kernel_subgroup_match_any(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_value_id_t value = loom_kernel_subgroup_match_any_value(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  const loom_value_id_t mask = loom_kernel_subgroup_match_any_mask(op);
  const loom_type_t mask_type = loom_module_value_type(context->module, mask);
  const uint32_t wavefront_size = loom_amdgpu_legalizer_wavefront_size(context);
  if (!loom_amdgpu_match_value_type_is_supported(value_type) ||
      !loom_amdgpu_wavefront_size_is_valid(wavefront_size) ||
      !loom_amdgpu_subgroup_mask_type_covers_wavefront(mask_type,
                                                       wavefront_size)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = loom_amdgpu_defer_or_reject_final(context),
    };
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* lane_id_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_lane_id_build(
      &rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      op->location, &lane_id_op));
  const loom_value_id_t lane_id =
      loom_kernel_subgroup_lane_id_result(lane_id_op);

  loom_op_t* lane_id_i32_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      &rewriter->builder, lane_id, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), op->location, &lane_id_i32_op));
  const loom_value_id_t lane_id_i32 = loom_index_cast_result(lane_id_i32_op);

  loom_value_id_t current_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_zero_mask(
      &rewriter->builder, mask_type, op->location, &current_mask));
  for (uint32_t i = 0; i < wavefront_size; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_match_any_lane_step(
        &rewriter->builder, value, value_type, lane_id_i32, i, current_mask,
        mask_type, op->location, &current_mask));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &current_mask, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &current_mask, 1));

  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_kernel_subgroup_match_all(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_value_id_t value = loom_kernel_subgroup_match_all_value(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  if (!loom_amdgpu_match_value_type_is_supported(value_type)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = loom_amdgpu_defer_or_reject_final(context),
    };
    return iree_ok_status();
  }

  const loom_value_id_t mask = loom_kernel_subgroup_match_all_mask(op);
  const loom_type_t mask_type = loom_module_value_type(context->module, mask);
  const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* first_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_broadcast_first_build(
      &rewriter->builder, value, value_type, op->location, &first_op));
  const loom_value_id_t first_value =
      loom_kernel_subgroup_broadcast_first_result(first_op);

  loom_op_t* equal_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      &rewriter->builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, value, first_value,
      value_type, i1_type, op->location, &equal_op));
  const loom_value_id_t equal = loom_scalar_cmpi_result(equal_op);

  loom_op_t* all_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_vote_all_build(
      &rewriter->builder, equal, i1_type, op->location, &all_op));
  const loom_value_id_t all_equal =
      loom_kernel_subgroup_vote_all_result(all_op);

  loom_op_t* active_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_active_mask_build(
      &rewriter->builder, mask_type, op->location, &active_mask_op));
  const loom_value_id_t active_mask =
      loom_kernel_subgroup_active_mask_mask(active_mask_op);

  loom_op_t* zero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      &rewriter->builder, loom_attr_i64(0), mask_type, op->location, &zero_op));
  const loom_value_id_t zero_mask = loom_scalar_constant_result(zero_op);

  loom_op_t* selected_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(&rewriter->builder, all_equal,
                                             active_mask, zero_mask, mask_type,
                                             op->location, &selected_mask_op));
  const loom_value_id_t replacements[] = {
      loom_scf_select_result(selected_mask_op),
      all_equal,
  };
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements),
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements)));

  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

typedef struct loom_amdgpu_vector_memory_chunk_shape_t {
  // Number of logical source lanes in the oversized vector.
  uint32_t lane_count;
  // Maximum logical source lanes per chunk.
  uint32_t chunk_lane_count;
  // Number of chunks needed to cover lane_count.
  uint32_t chunk_count;
} loom_amdgpu_vector_memory_chunk_shape_t;

static bool loom_amdgpu_vector_memory_chunk_shape(
    loom_type_t vector_type,
    loom_amdgpu_vector_memory_chunk_shape_t* out_shape) {
  *out_shape = (loom_amdgpu_vector_memory_chunk_shape_t){0};
  if (!loom_type_is_vector(vector_type) || loom_type_rank(vector_type) != 1 ||
      !loom_type_is_all_static(vector_type)) {
    return false;
  }
  const int64_t lane_count_i64 = loom_type_dim_static_size_at(vector_type, 0);
  if (lane_count_i64 < 1 || lane_count_i64 > UINT32_MAX) {
    return false;
  }

  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(vector_type));
  if (element_bit_count != 8 && element_bit_count != 16 &&
      element_bit_count != 32) {
    return false;
  }
  const uint32_t chunk_lane_count =
      (LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES * 32u) / (uint32_t)element_bit_count;

  const uint32_t lane_count = (uint32_t)lane_count_i64;
  if (lane_count <= chunk_lane_count) {
    return false;
  }
  const uint32_t chunk_count =
      (lane_count + chunk_lane_count - 1u) / chunk_lane_count;
  *out_shape = (loom_amdgpu_vector_memory_chunk_shape_t){
      .lane_count = lane_count,
      .chunk_lane_count = chunk_lane_count,
      .chunk_count = chunk_count,
  };
  return true;
}

static loom_type_t loom_amdgpu_vector_memory_chunk_type(loom_type_t vector_type,
                                                        uint32_t lane_count) {
  return loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(vector_type),
      loom_dim_pack_static(lane_count), vector_type.encoding_id);
}

static uint32_t loom_amdgpu_vector_memory_chunk_lane_offset(
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    uint32_t chunk_index) {
  return chunk_index * shape->chunk_lane_count;
}

static uint32_t loom_amdgpu_vector_memory_chunk_lane_count(
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    uint32_t chunk_index) {
  const uint32_t chunk_lane_offset =
      loom_amdgpu_vector_memory_chunk_lane_offset(shape, chunk_index);
  return iree_min(shape->lane_count - chunk_lane_offset,
                  shape->chunk_lane_count);
}

static loom_type_t loom_amdgpu_vector_memory_chunk_type_at(
    loom_type_t vector_type,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    uint32_t chunk_index) {
  return loom_amdgpu_vector_memory_chunk_type(
      vector_type,
      loom_amdgpu_vector_memory_chunk_lane_count(shape, chunk_index));
}

static bool loom_amdgpu_vector_memory_chunk_shape_equal(
    const loom_amdgpu_vector_memory_chunk_shape_t* lhs,
    const loom_amdgpu_vector_memory_chunk_shape_t* rhs) {
  return lhs->lane_count == rhs->lane_count &&
         lhs->chunk_lane_count == rhs->chunk_lane_count &&
         lhs->chunk_count == rhs->chunk_count;
}

typedef struct loom_amdgpu_vector_packetized_value_t {
  // Original oversized source value represented by chunks.
  loom_value_id_t source;
  // Original oversized source type.
  loom_type_t source_type;
  // Shared chunking shape for source_type.
  loom_amdgpu_vector_memory_chunk_shape_t shape;
  // Legal smaller vector values covering source in logical lane order.
  loom_value_id_t* chunks;
} loom_amdgpu_vector_packetized_value_t;

typedef struct loom_amdgpu_vector_packetization_t {
  // Target legalization context that owns the rewrite.
  loom_target_legalization_context_t* context;
  // Arena-backed packetized values materialized for this root rewrite.
  loom_amdgpu_vector_packetized_value_t* values;
  // Number of populated values.
  uint32_t value_count;
  // Number of allocated value entries.
  uint32_t value_capacity;
} loom_amdgpu_vector_packetization_t;

static bool loom_amdgpu_vector_memory_find_dynamic_axis_index(
    loom_attribute_t static_indices, iree_host_size_t axis,
    iree_host_size_t* out_dynamic_index) {
  iree_host_size_t dynamic_index = 0;
  for (iree_host_size_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] != INT64_MIN) {
      continue;
    }
    if (i == axis) {
      *out_dynamic_index = dynamic_index;
      return true;
    }
    ++dynamic_index;
  }
  return false;
}

static bool loom_amdgpu_vector_memory_can_build_chunk_origins(
    const loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape) {
  if (shape->chunk_count <= 1) {
    return true;
  }
  if (footprint->static_indices.count == 0) {
    return false;
  }

  const uint32_t max_lane_offset =
      (shape->chunk_count - 1u) * shape->chunk_lane_count;
  const iree_host_size_t last_axis = footprint->static_indices.count - 1u;
  const int64_t last_static_index =
      footprint->static_indices.i64_array[last_axis];
  if (last_static_index != INT64_MIN) {
    int64_t last_chunk_static_index = 0;
    return iree_checked_add_i64(last_static_index, (int64_t)max_lane_offset,
                                &last_chunk_static_index);
  }

  iree_host_size_t dynamic_index = 0;
  if (!loom_amdgpu_vector_memory_find_dynamic_axis_index(
          footprint->static_indices, last_axis, &dynamic_index) ||
      dynamic_index >= footprint->dynamic_indices.count) {
    return false;
  }
  const loom_value_id_t dynamic_value =
      footprint->dynamic_indices.values[dynamic_index];
  const loom_type_t dynamic_type =
      loom_module_value_type(context->module, dynamic_value);
  return loom_type_is_scalar(dynamic_type) &&
         loom_type_element_type(dynamic_type) == LOOM_SCALAR_TYPE_INDEX;
}

static iree_status_t loom_amdgpu_vector_memory_build_chunk_origin(
    loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint, const loom_op_t* source_op,
    uint32_t chunk_lane_offset, const loom_value_id_t** out_dynamic_indices,
    iree_host_size_t* out_dynamic_index_count,
    const int64_t** out_static_indices,
    iree_host_size_t* out_static_index_count, bool* out_built) {
  *out_dynamic_indices = footprint->dynamic_indices.values;
  *out_dynamic_index_count = footprint->dynamic_indices.count;
  *out_static_indices = footprint->static_indices.i64_array;
  *out_static_index_count = footprint->static_indices.count;
  *out_built = true;
  if (chunk_lane_offset == 0) {
    return iree_ok_status();
  }
  if (footprint->static_indices.count == 0) {
    *out_built = false;
    return iree_ok_status();
  }

  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->rewriter->builder.arena, footprint->static_indices.count,
      sizeof(*static_indices), (void**)&static_indices));
  memcpy(static_indices, footprint->static_indices.i64_array,
         footprint->static_indices.count * sizeof(*static_indices));

  const iree_host_size_t last_axis = footprint->static_indices.count - 1u;
  if (static_indices[last_axis] != INT64_MIN) {
    if (!iree_checked_add_i64(static_indices[last_axis],
                              (int64_t)chunk_lane_offset,
                              &static_indices[last_axis])) {
      *out_built = false;
      return iree_ok_status();
    }
    *out_static_indices = static_indices;
    return iree_ok_status();
  }

  iree_host_size_t dynamic_index = 0;
  if (!loom_amdgpu_vector_memory_find_dynamic_axis_index(
          footprint->static_indices, last_axis, &dynamic_index) ||
      dynamic_index >= footprint->dynamic_indices.count) {
    *out_built = false;
    return iree_ok_status();
  }
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->rewriter->builder.arena, footprint->dynamic_indices.count,
      sizeof(*dynamic_indices), (void**)&dynamic_indices));
  memcpy(dynamic_indices, footprint->dynamic_indices.values,
         footprint->dynamic_indices.count * sizeof(*dynamic_indices));

  const loom_value_id_t dynamic_value = dynamic_indices[dynamic_index];
  const loom_type_t dynamic_type =
      loom_module_value_type(context->module, dynamic_value);
  if (!loom_type_is_scalar(dynamic_type) ||
      loom_type_element_type(dynamic_type) != LOOM_SCALAR_TYPE_INDEX) {
    *out_built = false;
    return iree_ok_status();
  }
  loom_op_t* offset_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &context->rewriter->builder, loom_attr_i64(chunk_lane_offset),
      dynamic_type, source_op->location, &offset_op));
  loom_op_t* add_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_add_build(&context->rewriter->builder, dynamic_value,
                           loom_index_constant_result(offset_op), dynamic_type,
                           source_op->location, &add_op));
  dynamic_indices[dynamic_index] = loom_index_add_result(add_op);

  *out_dynamic_indices = dynamic_indices;
  *out_static_indices = static_indices;
  return iree_ok_status();
}

static const loom_fact_context_t* loom_amdgpu_vector_packet_fact_context(
    const loom_target_legalization_context_t* context) {
  return context->fact_table ? &context->fact_table->context : NULL;
}

static bool loom_amdgpu_vector_packet_type_shape_matches(
    loom_type_t type, const loom_amdgpu_vector_memory_chunk_shape_t* shape) {
  loom_amdgpu_vector_memory_chunk_shape_t type_shape = {0};
  return loom_amdgpu_vector_memory_chunk_shape(type, &type_shape) &&
         loom_amdgpu_vector_memory_chunk_shape_equal(shape, &type_shape);
}

static bool loom_amdgpu_vector_packet_can_materialize_value(
    const loom_target_legalization_context_t* context, loom_value_id_t source,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape);

static bool loom_amdgpu_vector_packet_can_materialize_memory_load(
    const loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape) {
  loom_vector_memory_footprint_t footprint = {0};
  if (!loom_vector_memory_footprint_describe(
          loom_amdgpu_vector_packet_fact_context(context), context->module, op,
          &footprint) ||
      footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE ||
      !loom_type_equal(footprint.vector_type,
                       loom_module_value_type(context->module,
                                              loom_vector_load_result(op)))) {
    return false;
  }
  loom_vector_memory_cache_policy_t cache_policy = {0};
  return loom_vector_memory_cache_policy_from_op(context->module, op,
                                                 &cache_policy) &&
         loom_amdgpu_vector_memory_can_build_chunk_origins(context, &footprint,
                                                           shape);
}

static bool loom_amdgpu_vector_packet_can_materialize_op(
    const loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape) {
  if (!iree_all_bits_set(op->traits, LOOM_TRAIT_DECOMPOSABLE)) {
    return false;
  }
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  if (!loom_amdgpu_vector_packet_type_shape_matches(result_type, shape)) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_type_t operand_type =
        loom_module_value_type(context->module, operands[i]);
    if (!loom_type_equal(operand_type, result_type) ||
        !loom_amdgpu_vector_packet_can_materialize_value(context, operands[i],
                                                         shape)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_vector_packet_can_materialize_value(
    const loom_target_legalization_context_t* context, loom_value_id_t source,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape) {
  const loom_value_t* value = loom_module_value(context->module, source);
  if (value == NULL || loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  if (!loom_amdgpu_vector_packet_type_shape_matches(source_type, shape)) {
    return false;
  }
  const loom_op_t* op = loom_value_def_op(value);
  if (op == NULL) {
    return false;
  }
  if (loom_vector_load_isa(op)) {
    return loom_amdgpu_vector_packet_can_materialize_memory_load(context, op,
                                                                 shape);
  }
  if (loom_vector_constant_isa(op) || loom_vector_poison_isa(op) ||
      loom_vector_splat_isa(op)) {
    return true;
  }
  if (loom_vector_from_elements_isa(op)) {
    const loom_value_slice_t elements = loom_vector_from_elements_elements(op);
    return elements.count == shape->lane_count;
  }
  return loom_amdgpu_vector_packet_can_materialize_op(context, op, shape);
}

static loom_amdgpu_vector_packetized_value_t* loom_amdgpu_vector_packet_find(
    loom_amdgpu_vector_packetization_t* packetization, loom_value_id_t source) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    if (packetization->values[i].source == source) {
      return &packetization->values[i];
    }
  }
  return NULL;
}

static iree_status_t loom_amdgpu_vector_packet_reserve(
    loom_amdgpu_vector_packetization_t* packetization, uint32_t capacity) {
  if (capacity <= packetization->value_capacity) {
    return iree_ok_status();
  }
  uint32_t new_capacity =
      packetization->value_capacity == 0 ? 8u : packetization->value_capacity;
  while (new_capacity < capacity) {
    if (new_capacity > UINT32_MAX / 2u) {
      new_capacity = capacity;
      break;
    }
    new_capacity *= 2u;
  }
  loom_amdgpu_vector_packetized_value_t* new_values = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(packetization->context->arena, new_capacity,
                                sizeof(*new_values), (void**)&new_values));
  if (packetization->value_count != 0) {
    memcpy(new_values, packetization->values,
           packetization->value_count * sizeof(*new_values));
  }
  packetization->values = new_values;
  packetization->value_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_insert(
    loom_amdgpu_vector_packetization_t* packetization, loom_value_id_t source,
    loom_type_t source_type,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_amdgpu_vector_packetized_value_t** out_value) {
  *out_value = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_reserve(
      packetization, packetization->value_count + 1u));
  loom_amdgpu_vector_packetized_value_t* value =
      &packetization->values[packetization->value_count++];
  *value = (loom_amdgpu_vector_packetized_value_t){
      .source = source,
      .source_type = source_type,
      .shape = *shape,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      packetization->context->arena, shape->chunk_count, sizeof(*value->chunks),
      (void**)&value->chunks));
  for (uint32_t i = 0; i < shape->chunk_count; ++i) {
    value->chunks[i] = LOOM_VALUE_ID_INVALID;
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_value(
    loom_amdgpu_vector_packetization_t* packetization, loom_value_id_t source,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_amdgpu_vector_packetized_value_t** out_value);

static iree_status_t loom_amdgpu_vector_packet_materialize_memory_load(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  loom_target_legalization_context_t* context = packetization->context;
  loom_vector_memory_footprint_t footprint = {0};
  const bool footprint_described = loom_vector_memory_footprint_describe(
      loom_amdgpu_vector_packet_fact_context(context), context->module, op,
      &footprint);
  IREE_ASSERT_TRUE(footprint_described);
  (void)footprint_described;
  loom_vector_memory_cache_policy_t cache_policy = {0};
  const bool cache_policy_described = loom_vector_memory_cache_policy_from_op(
      context->module, op, &cache_policy);
  IREE_ASSERT_TRUE(cache_policy_described);
  (void)cache_policy_described;

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, op);
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const uint32_t chunk_lane_offset =
        loom_amdgpu_vector_memory_chunk_lane_offset(shape, chunk_index);
    const loom_type_t chunk_type = loom_amdgpu_vector_memory_chunk_type_at(
        packetized_value->source_type, shape, chunk_index);

    const loom_value_id_t* dynamic_indices = NULL;
    iree_host_size_t dynamic_index_count = 0;
    const int64_t* static_indices = NULL;
    iree_host_size_t static_index_count = 0;
    bool origin_built = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_memory_build_chunk_origin(
        context, &footprint, op, chunk_lane_offset, &dynamic_indices,
        &dynamic_index_count, &static_indices, &static_index_count,
        &origin_built));
    IREE_ASSERT_TRUE(origin_built);
    loom_op_t* chunk_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_load_build(
        builder, cache_policy.build_flags, footprint.view, dynamic_indices,
        dynamic_index_count, static_indices, static_index_count,
        cache_policy.cache_scope, cache_policy.cache_temporal, chunk_type,
        op->location, &chunk_op));
    packetized_value->chunks[chunk_index] = loom_vector_load_result(chunk_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_simple_op(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  loom_target_legalization_context_t* context = packetization->context;
  loom_builder_t* builder = &context->rewriter->builder;
  const loom_value_id_t* source_operands = loom_op_const_operands(op);
  loom_amdgpu_vector_packetized_value_t** operand_packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, op->operand_count, sizeof(*operand_packets),
      (void**)&operand_packets));
  for (uint16_t operand_index = 0; operand_index < op->operand_count;
       ++operand_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_value(
        packetization, source_operands[operand_index], shape,
        &operand_packets[operand_index]));
  }

  loom_builder_set_before(builder, op);
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const loom_type_t chunk_type = loom_amdgpu_vector_memory_chunk_type_at(
        packetized_value->source_type, shape, chunk_index);

    loom_op_t* chunk_op = NULL;
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        builder, op->kind, op->operand_count, op->result_count,
        /*region_count=*/0, /*tied_result_count=*/0, op->attribute_count,
        op->location, &chunk_op));
    chunk_op->instance_flags = op->instance_flags;
    for (uint16_t operand_index = 0; operand_index < op->operand_count;
         ++operand_index) {
      loom_op_operands(chunk_op)[operand_index] =
          operand_packets[operand_index]->chunks[chunk_index];
    }
    if (op->attribute_count != 0) {
      memcpy(loom_op_attrs(chunk_op), loom_op_const_attrs(op),
             op->attribute_count * sizeof(loom_attribute_t));
    }
    loom_value_id_t result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_builder_define_value(builder, chunk_type, &result));
    loom_op_results(chunk_op)[0] = result;
    IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, chunk_op));
    packetized_value->chunks[chunk_index] = result;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_constant(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_attribute_t value = loom_vector_constant_value(op);
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const loom_type_t chunk_type = loom_amdgpu_vector_memory_chunk_type_at(
        packetized_value->source_type, shape, chunk_index);
    loom_op_t* chunk_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(builder, value, chunk_type,
                                                    op->location, &chunk_op));
    packetized_value->chunks[chunk_index] =
        loom_vector_constant_result(chunk_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_poison(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  loom_builder_set_before(builder, op);
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const loom_type_t chunk_type = loom_amdgpu_vector_memory_chunk_type_at(
        packetized_value->source_type, shape, chunk_index);
    loom_op_t* chunk_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_poison_build(builder, chunk_type, op->location, &chunk_op));
    packetized_value->chunks[chunk_index] = loom_vector_poison_result(chunk_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_splat(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t scalar = loom_vector_splat_scalar(op);
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const loom_type_t chunk_type = loom_amdgpu_vector_memory_chunk_type_at(
        packetized_value->source_type, shape, chunk_index);
    loom_op_t* chunk_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_splat_build(builder, scalar, chunk_type,
                                                 op->location, &chunk_op));
    packetized_value->chunks[chunk_index] = loom_vector_splat_result(chunk_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_from_elements(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const uint32_t chunk_lane_offset =
        loom_amdgpu_vector_memory_chunk_lane_offset(shape, chunk_index);
    const uint32_t chunk_lane_count =
        loom_amdgpu_vector_memory_chunk_lane_count(shape, chunk_index);
    const loom_type_t chunk_type = loom_amdgpu_vector_memory_chunk_type_at(
        packetized_value->source_type, shape, chunk_index);
    loom_op_t* chunk_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
        builder, &elements.values[chunk_lane_offset], chunk_lane_count,
        chunk_type, op->location, &chunk_op));
    packetized_value->chunks[chunk_index] =
        loom_vector_from_elements_result(chunk_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_value(
    loom_amdgpu_vector_packetization_t* packetization, loom_value_id_t source,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_amdgpu_vector_packetized_value_t** out_value) {
  *out_value = NULL;
  loom_amdgpu_vector_packetized_value_t* cached =
      loom_amdgpu_vector_packet_find(packetization, source);
  if (cached != NULL) {
    *out_value = cached;
    return iree_ok_status();
  }
  loom_target_legalization_context_t* context = packetization->context;
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  loom_amdgpu_vector_packetized_value_t* packetized_value = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_insert(
      packetization, source, source_type, shape, &packetized_value));

  const loom_value_t* value = loom_module_value(context->module, source);
  loom_op_t* op = loom_value_def_op(value);
  if (loom_vector_load_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_memory_load(
        packetization, op, shape, packetized_value));
  } else if (loom_vector_constant_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_constant(
        packetization, op, shape, packetized_value));
  } else if (loom_vector_poison_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_poison(
        packetization, op, shape, packetized_value));
  } else if (loom_vector_splat_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_splat(
        packetization, op, shape, packetized_value));
  } else if (loom_vector_from_elements_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_from_elements(
        packetization, op, shape, packetized_value));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_simple_op(
        packetization, op, shape, packetized_value));
  }

  *out_value = packetized_value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_store(
    loom_amdgpu_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    const loom_amdgpu_vector_packetized_value_t* packetized_value,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op) {
  loom_target_legalization_context_t* context = packetization->context;
  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, store_op);
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const uint32_t chunk_lane_offset =
        loom_amdgpu_vector_memory_chunk_lane_offset(shape, chunk_index);
    const loom_value_id_t* dynamic_indices = NULL;
    iree_host_size_t dynamic_index_count = 0;
    const int64_t* static_indices = NULL;
    iree_host_size_t static_index_count = 0;
    bool origin_built = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_memory_build_chunk_origin(
        context, store_footprint, store_op, chunk_lane_offset, &dynamic_indices,
        &dynamic_index_count, &static_indices, &static_index_count,
        &origin_built));
    IREE_ASSERT_TRUE(origin_built);
    loom_op_t* chunk_store_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_store_build(
        builder, store_cache_policy.build_flags,
        packetized_value->chunks[chunk_index], store_footprint->view,
        dynamic_indices, dynamic_index_count, static_indices,
        static_index_count, store_cache_policy.cache_scope,
        store_cache_policy.cache_temporal, store_op->location,
        &chunk_store_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_erase_dead_sources(
    loom_amdgpu_vector_packetization_t* packetization) {
  loom_rewriter_t* rewriter = packetization->context->rewriter;
  for (uint32_t i = packetization->value_count; i > 0; --i) {
    const loom_amdgpu_vector_packetized_value_t* packetized_value =
        &packetization->values[i - 1];
    const loom_value_t* value = loom_module_value(
        packetization->context->module, packetized_value->source);
    if (value == NULL || loom_value_is_block_arg(value)) continue;
    loom_op_t* op = loom_value_def_op(value);
    if (op == NULL) continue;
    bool erased = false;
    IREE_RETURN_IF_ERROR(loom_rewriter_erase_if_dead(rewriter, op, &erased));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_epilogue_plan_needs_physical_loop(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  if (plan->operation_kind != LOOM_AMDGPU_MEMORY_OPERATION_STORE ||
      plan->role != LOOM_CONTRACT_OPERAND_ROLE_RESULT ||
      plan->register_count <= 1 || plan->packet_count == 0) {
    return false;
  }
  switch (plan->payload_form) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32:
      break;
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT:
    default:
      return false;
  }
  for (uint16_t i = 0; i < plan->packet_count; ++i) {
    if (plan->packets[i].result_register_count != 1) {
      return false;
    }
  }
  return true;
}

enum {
  LOOM_AMDGPU_FRAGMENT_EPILOGUE_LOOP_MIN_REGISTER_ITERATIONS = 8,
};

static bool loom_amdgpu_fragment_epilogue_group_wants_physical_loop(
    const loom_amdgpu_fragment_memory_plan_t* plan,
    iree_host_size_t group_count) {
  return plan->register_count * group_count >=
         LOOM_AMDGPU_FRAGMENT_EPILOGUE_LOOP_MIN_REGISTER_ITERATIONS;
}

typedef struct loom_amdgpu_fragment_store_rectangle_t {
  // Destination view value for the fragment store.
  loom_value_id_t view;
  // Symbolic inclusive begin coordinates for the rank-2 logical footprint.
  loom_symbolic_expr_t begin[2];
  // Symbolic exclusive end coordinates for the rank-2 logical footprint.
  loom_symbolic_expr_t end[2];
} loom_amdgpu_fragment_store_rectangle_t;

static iree_status_t loom_amdgpu_fragment_store_origin_expression(
    loom_symbolic_expr_context_t* expression_context,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    uint8_t axis, uint16_t* dynamic_index_ordinal,
    loom_symbolic_expr_t* out_expression, bool* out_selected) {
  *out_selected = false;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      axis >= static_indices.count) {
    return iree_ok_status();
  }
  const int64_t static_index = static_indices.i64_array[axis];
  if (static_index == INT64_MIN) {
    if (*dynamic_index_ordinal >= dynamic_indices.count) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
        expression_context, dynamic_indices.values[*dynamic_index_ordinal],
        out_expression));
    ++*dynamic_index_ordinal;
  } else {
    loom_symbolic_expr_constant(static_index, out_expression);
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_store_rectangle_from_op(
    loom_symbolic_expr_context_t* expression_context, const loom_op_t* op,
    loom_amdgpu_fragment_store_rectangle_t* out_rectangle, bool* out_selected) {
  *out_selected = false;
  if (!loom_vector_fragment_store_isa(op)) {
    return iree_ok_status();
  }
  loom_attribute_t static_indices =
      loom_vector_fragment_store_static_indices(op);
  loom_value_slice_t dynamic_indices = loom_vector_fragment_store_indices(op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != IREE_ARRAYSIZE(out_rectangle->begin)) {
    return iree_ok_status();
  }

  loom_value_id_t extents[2] = {
      loom_vector_fragment_store_rows(op),
      loom_vector_fragment_store_columns(op),
  };
  *out_rectangle = (loom_amdgpu_fragment_store_rectangle_t){
      .view = loom_vector_fragment_store_view(op),
  };
  uint16_t dynamic_index_ordinal = 0;
  for (uint8_t axis = 0; axis < IREE_ARRAYSIZE(out_rectangle->begin); ++axis) {
    bool selected = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_origin_expression(
        expression_context, static_indices, dynamic_indices, axis,
        &dynamic_index_ordinal, &out_rectangle->begin[axis], &selected));
    if (!selected) {
      return iree_ok_status();
    }
    loom_symbolic_expr_t extent = {0};
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(expression_context,
                                                       extents[axis], &extent));
    IREE_RETURN_IF_ERROR(
        loom_symbolic_expr_add(expression_context, &out_rectangle->begin[axis],
                               &extent, &out_rectangle->end[axis]));
  }
  if (dynamic_index_ordinal != dynamic_indices.count) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_store_rectangles_are_disjoint(
    loom_symbolic_expr_context_t* expression_context,
    const loom_amdgpu_fragment_store_rectangle_t* left,
    const loom_amdgpu_fragment_store_rectangle_t* right, bool* out_disjoint) {
  *out_disjoint = false;
  if (left->view != right->view) {
    return iree_ok_status();
  }
  for (uint8_t axis = 0; axis < IREE_ARRAYSIZE(left->begin); ++axis) {
    loom_symbolic_proof_result_t result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
        expression_context, &left->end[axis], &right->begin[axis], &result));
    if (result == LOOM_SYMBOLIC_PROOF_TRUE) {
      *out_disjoint = true;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
        expression_context, &right->end[axis], &left->begin[axis], &result));
    if (result == LOOM_SYMBOLIC_PROOF_TRUE) {
      *out_disjoint = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_store_group_reserve(
    loom_target_legalization_context_t* context, loom_op_t*** group_ops,
    loom_amdgpu_fragment_store_rectangle_t** rectangles,
    iree_host_size_t current_count, iree_host_size_t* capacity,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= *capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = *capacity * 2;
  if (new_capacity < minimum_capacity) {
    new_capacity = minimum_capacity;
  }
  loom_op_t** new_group_ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, new_capacity,
                                                 sizeof(*new_group_ops),
                                                 (void**)&new_group_ops));
  loom_amdgpu_fragment_store_rectangle_t* new_rectangles = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, new_capacity,
                                                 sizeof(*new_rectangles),
                                                 (void**)&new_rectangles));
  memcpy(new_group_ops, *group_ops, current_count * sizeof(*new_group_ops));
  memcpy(new_rectangles, *rectangles, current_count * sizeof(*new_rectangles));
  *group_ops = new_group_ops;
  *rectangles = new_rectangles;
  *capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_store_plan_can_join_group(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_amdgpu_matrix_fragment_layout_kind_t layout_kind,
    uint16_t register_count, loom_amdgpu_fragment_memory_plan_t* out_plan,
    bool* out_selected) {
  *out_selected = false;
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_analyze_vector_fragment_memory_plan(
      context->module, context->fact_table, context->bundle,
      context->descriptor_set, context->target_ref, context->function, op,
      LOOM_AMDGPU_MEMORY_OPERATION_STORE, out_plan, &selected));
  if (!selected ||
      !loom_amdgpu_fragment_epilogue_plan_needs_physical_loop(out_plan) ||
      out_plan->view_rank != 2 || out_plan->layout_kind != layout_kind ||
      out_plan->register_count != register_count) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_collect_fragment_store_epilogue_group(
    loom_target_legalization_context_t* context, loom_op_t* current_op,
    const loom_amdgpu_fragment_memory_plan_t* first_plan,
    loom_op_t*** out_group_ops, iree_host_size_t* out_group_count) {
  *out_group_ops = NULL;
  *out_group_count = 0;

  loom_op_t* local_ops[8] = {0};
  loom_amdgpu_fragment_store_rectangle_t local_rectangles[8] = {0};
  loom_op_t** group_ops = local_ops;
  loom_amdgpu_fragment_store_rectangle_t* rectangles = local_rectangles;
  iree_host_size_t group_count = 1;
  iree_host_size_t capacity = IREE_ARRAYSIZE(local_ops);
  group_ops[0] = current_op;

  bool selected = false;
  loom_amdgpu_fragment_memory_plan_t neighbor_plan = {0};
  if (current_op->prev_op != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_plan_can_join_group(
        context, current_op->prev_op, first_plan->layout_kind,
        first_plan->register_count, &neighbor_plan, &selected));
  }
  if (!selected && current_op->next_op != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_plan_can_join_group(
        context, current_op->next_op, first_plan->layout_kind,
        first_plan->register_count, &neighbor_plan, &selected));
  }
  if (!selected) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, group_count,
                                                   sizeof(*group_ops),
                                                   (void**)out_group_ops));
    (*out_group_ops)[0] = current_op;
    *out_group_count = group_count;
    return iree_ok_status();
  }

  loom_symbolic_expr_context_t expression_context = {0};
  loom_symbolic_expr_context_initialize(context->module, context->fact_table,
                                        context->arena, &expression_context);
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_rectangle_from_op(
      &expression_context, current_op, &rectangles[0], &selected));
  if (!selected) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, group_count,
                                                   sizeof(*group_ops),
                                                   (void**)out_group_ops));
    (*out_group_ops)[0] = current_op;
    *out_group_count = group_count;
    return iree_ok_status();
  }

  for (loom_op_t* candidate = current_op->prev_op; candidate != NULL;
       candidate = candidate->prev_op) {
    loom_amdgpu_fragment_memory_plan_t candidate_plan = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_plan_can_join_group(
        context, candidate, first_plan->layout_kind, first_plan->register_count,
        &candidate_plan, &selected));
    if (!selected) {
      break;
    }

    loom_amdgpu_fragment_store_rectangle_t candidate_rectangle = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_rectangle_from_op(
        &expression_context, candidate, &candidate_rectangle, &selected));
    if (!selected) {
      break;
    }

    for (iree_host_size_t i = 0; i < group_count; ++i) {
      bool disjoint = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_rectangles_are_disjoint(
          &expression_context, &rectangles[i], &candidate_rectangle,
          &disjoint));
      if (!disjoint) {
        selected = false;
        break;
      }
    }
    if (!selected) {
      break;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_group_reserve(
        context, &group_ops, &rectangles, group_count, &capacity,
        group_count + 1));
    memmove(&group_ops[1], group_ops, group_count * sizeof(*group_ops));
    memmove(&rectangles[1], rectangles, group_count * sizeof(*rectangles));
    group_ops[0] = candidate;
    rectangles[0] = candidate_rectangle;
    ++group_count;
  }

  for (loom_op_t* candidate = current_op->next_op; candidate != NULL;
       candidate = candidate->next_op) {
    loom_amdgpu_fragment_memory_plan_t candidate_plan = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_plan_can_join_group(
        context, candidate, first_plan->layout_kind, first_plan->register_count,
        &candidate_plan, &selected));
    if (!selected) {
      break;
    }

    loom_amdgpu_fragment_store_rectangle_t candidate_rectangle = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_rectangle_from_op(
        &expression_context, candidate, &candidate_rectangle, &selected));
    if (!selected) {
      break;
    }

    for (iree_host_size_t i = 0; i < group_count; ++i) {
      bool disjoint = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_rectangles_are_disjoint(
          &expression_context, &rectangles[i], &candidate_rectangle,
          &disjoint));
      if (!disjoint) {
        selected = false;
        break;
      }
    }
    if (!selected) {
      break;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_store_group_reserve(
        context, &group_ops, &rectangles, group_count, &capacity,
        group_count + 1));
    group_ops[group_count] = candidate;
    rectangles[group_count] = candidate_rectangle;
    ++group_count;
  }

  if (group_ops == local_ops) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, group_count,
                                                   sizeof(*group_ops),
                                                   (void**)out_group_ops));
    memcpy(*out_group_ops, group_ops, group_count * sizeof(*group_ops));
  } else {
    *out_group_ops = group_ops;
  }
  *out_group_count = group_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_result_fragment_store_epilogue_loop(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  loom_amdgpu_fragment_memory_plan_t plan = {0};
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_analyze_vector_fragment_memory_plan(
      context->module, context->fact_table, context->bundle,
      context->descriptor_set, context->target_ref, context->function, op,
      LOOM_AMDGPU_MEMORY_OPERATION_STORE, &plan, &selected));
  if (!selected ||
      !loom_amdgpu_fragment_epilogue_plan_needs_physical_loop(&plan) ||
      plan.view_rank != 2) {
    return iree_ok_status();
  }

  const loom_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan.layout_kind);
  loom_op_t** group_ops = NULL;
  iree_host_size_t group_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_collect_fragment_store_epilogue_group(
      context, op, &plan, &group_ops, &group_count));
  if (!loom_amdgpu_fragment_epilogue_group_wants_physical_loop(&plan,
                                                               group_count)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_fragment_store_to_scalar_physical_result_loop_rewrite_ops(
          context->pass, context->rewriter, group_ops, group_count, layout,
          plan.register_count, &rewritten));
  if (!rewritten) {
    return iree_ok_status();
  }
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_oversized_vector_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  loom_vector_memory_footprint_t store_footprint = {0};
  if (!loom_vector_memory_footprint_describe(
          loom_amdgpu_vector_packet_fact_context(context), context->module, op,
          &store_footprint) ||
      store_footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE) {
    return iree_ok_status();
  }
  loom_amdgpu_vector_memory_chunk_shape_t shape = {0};
  if (!loom_amdgpu_vector_memory_chunk_shape(store_footprint.vector_type,
                                             &shape)) {
    return iree_ok_status();
  }

  loom_vector_memory_cache_policy_t store_cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, op,
                                               &store_cache_policy) ||
      !loom_amdgpu_vector_memory_can_build_chunk_origins(
          context, &store_footprint, &shape) ||
      !loom_amdgpu_vector_packet_can_materialize_value(
          context, store_footprint.value, &shape)) {
    return iree_ok_status();
  }

  loom_amdgpu_vector_packetization_t packetization = {
      .context = context,
  };
  loom_amdgpu_vector_packetized_value_t* packetized_value = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_value(
      &packetization, store_footprint.value, &shape, &packetized_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_store(
      &packetization, &store_footprint, &shape, packetized_value,
      store_cache_policy, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_packet_erase_dead_sources(&packetization));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_legalize_oversized_vector_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_legalizer_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_value_id_t input = loom_vector_reduce_input(op);
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  loom_amdgpu_vector_memory_chunk_shape_t shape = {0};
  if (!loom_amdgpu_vector_memory_chunk_shape(input_type, &shape) ||
      !loom_amdgpu_vector_packet_can_materialize_value(context, input,
                                                       &shape)) {
    return iree_ok_status();
  }

  loom_amdgpu_vector_packetization_t packetization = {
      .context = context,
  };
  loom_amdgpu_vector_packetized_value_t* packetized_input = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_value(
      &packetization, input, &shape, &packetized_input));

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_combining_kind_t kind = loom_vector_reduce_kind(op);
  const uint8_t fastmath_flags = loom_vector_reduce_fastmath(op);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_reduce_result(op));
  loom_value_id_t accumulator = loom_vector_reduce_init(op);
  for (uint32_t chunk_index = 0; chunk_index < shape.chunk_count;
       ++chunk_index) {
    loom_op_t* chunk_reduce_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
        builder, kind, fastmath_flags, packetized_input->chunks[chunk_index],
        accumulator, result_type, op->location, &chunk_reduce_op));
    accumulator = loom_vector_reduce_result(chunk_reduce_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &accumulator, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &accumulator, 1));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_packet_erase_dead_sources(&packetization));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static const loom_target_legalizer_entry_t kAmdgpuLegalizerEntries[] = {
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_VECTOR_FRAGMENT_STORE,
        .legalize = loom_amdgpu_legalize_result_fragment_store_epilogue_loop,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE,
        .legalize = loom_amdgpu_legalize_oversized_vector_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE,
        .legalize = loom_amdgpu_legalize_oversized_vector_reduce,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_EXTRACTU,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_EXTRACTS,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_INSERT,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITPACK,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKU,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKS,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOTF,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT2F,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4I,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT8I4,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4F8,
        .legalize = loom_amdgpu_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_MATCH_ANY,
        .legalize = loom_amdgpu_legalize_kernel_subgroup_match_any,
    },
    {
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL,
        .legalize = loom_amdgpu_legalize_kernel_subgroup_match_all,
    },
};

const loom_target_legalizer_provider_t
    loom_amdgpu_target_legalizer_provider_storage = {
        .name = IREE_SVL("amdgpu"),
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
        .entries = kAmdgpuLegalizerEntries,
        .entry_count = IREE_ARRAYSIZE(kAmdgpuLegalizerEntries),
};

const loom_target_legalizer_provider_t* loom_amdgpu_target_legalizer_provider(
    void) {
  return &loom_amdgpu_target_legalizer_provider_storage;
}
