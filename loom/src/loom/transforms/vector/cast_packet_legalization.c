// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/cast_packet_legalization.h"

#include <stdint.h>

#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/vector/packet_legalization.h"

// Caps the slices, packet operations, and final concat created for one cast.
#define LOOM_VECTOR_CAST_PACKET_OP_LIMIT 64u

static bool loom_vector_packet_static_lane_count(loom_type_t vector_type,
                                                 uint32_t* out_lane_count) {
  *out_lane_count = 0;
  if (!loom_type_is_vector(vector_type) || loom_type_rank(vector_type) != 1 ||
      !loom_type_is_all_static(vector_type)) {
    return false;
  }
  const int64_t lane_count_i64 = loom_type_dim_static_size_at(vector_type, 0);
  if (lane_count_i64 < 1 || lane_count_i64 > UINT32_MAX) {
    return false;
  }
  *out_lane_count = (uint32_t)lane_count_i64;
  return true;
}

static loom_type_t loom_vector_packet_type(loom_type_t vector_type,
                                           uint32_t lane_count) {
  return loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(vector_type),
      loom_dim_pack_static(lane_count), vector_type.encoding_id);
}

static iree_status_t loom_vector_packet_bitcast_candidate_is_legal(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_type_t source_type, loom_type_t result_type,
    uint32_t source_lane_count, uint32_t packet_lane_count, bool* out_legal) {
  *out_legal = false;
  const loom_target_contract_vector_lane_projection_t projection = {
      .source_lane_count = source_lane_count,
      .projected_lane_count = packet_lane_count,
  };
  loom_target_contract_query_result_t query_result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(
      loom_target_legalization_query_contract_with_vector_lane_projection(
          context, op, projection, &query_result));
  if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
    *out_legal = true;
    return iree_ok_status();
  }
  if (query_result.outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }

  const loom_target_vector_carrier_supported_callback_t callback =
      context->source_vector_carrier_supported;
  if (callback.fn == NULL) {
    return iree_ok_status();
  }
  const loom_type_t packet_source_type =
      loom_vector_packet_type(source_type, packet_lane_count);
  const loom_type_t packet_result_type =
      loom_vector_packet_type(result_type, packet_lane_count);
  *out_legal =
      callback.fn(callback.user_data, context->target_facts, context->module,
                  NULL, LOOM_VALUE_ID_INVALID, packet_source_type) &&
      callback.fn(callback.user_data, context->target_facts, context->module,
                  NULL, LOOM_VALUE_ID_INVALID, packet_result_type);
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_bitcast_select_lane_count(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t source_type,
    loom_type_t result_type, uint32_t source_lane_count,
    uint32_t* out_packet_lane_count) {
  *out_packet_lane_count = 0;
  for (uint8_t i = 0; i < policy->native_lane_count_count; ++i) {
    const uint32_t packet_lane_count = policy->native_lane_counts[i];
    if (packet_lane_count == 0 || packet_lane_count >= source_lane_count ||
        packet_lane_count <= *out_packet_lane_count) {
      continue;
    }
    const uint32_t packet_count =
        (source_lane_count - 1u) / packet_lane_count + 1u;
    if ((uint64_t)packet_count * 2u + 1u > LOOM_VECTOR_CAST_PACKET_OP_LIMIT) {
      continue;
    }
    bool full_packet_is_legal = false;
    IREE_RETURN_IF_ERROR(loom_vector_packet_bitcast_candidate_is_legal(
        context, op, source_type, result_type, source_lane_count,
        packet_lane_count, &full_packet_is_legal));
    if (!full_packet_is_legal) {
      continue;
    }
    const uint32_t tail_lane_count = source_lane_count % packet_lane_count;
    if (tail_lane_count != 0) {
      bool tail_packet_is_legal = false;
      IREE_RETURN_IF_ERROR(loom_vector_packet_bitcast_candidate_is_legal(
          context, op, source_type, result_type, source_lane_count,
          tail_lane_count, &tail_packet_is_legal));
      if (!tail_packet_is_legal) {
        continue;
      }
    }
    *out_packet_lane_count = packet_lane_count;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_packet_legalize_bitcast(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_bitcast_isa(op) || policy == NULL) {
    return iree_ok_status();
  }

  const loom_value_id_t source = loom_vector_bitcast_input(op);
  const loom_value_id_t result = loom_vector_bitcast_result(op);
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const loom_type_t result_type =
      loom_module_value_type(context->module, result);
  uint32_t source_lane_count = 0;
  uint32_t result_lane_count = 0;
  if (!loom_vector_packet_static_lane_count(source_type, &source_lane_count) ||
      !loom_vector_packet_static_lane_count(result_type, &result_lane_count) ||
      source_lane_count <= 1 || source_lane_count != result_lane_count) {
    return iree_ok_status();
  }
  const int32_t source_element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  const int32_t result_element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (source_element_bit_count <= 0 ||
      source_element_bit_count != result_element_bit_count) {
    return iree_ok_status();
  }

  uint32_t packet_lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_packet_bitcast_select_lane_count(
      context, op, policy, source_type, result_type, source_lane_count,
      &packet_lane_count));
  if (packet_lane_count == 0) {
    return iree_ok_status();
  }

  const uint32_t packet_count =
      (source_lane_count - 1u) / packet_lane_count + 1u;
  loom_value_id_t* packet_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, packet_count,
                                                 sizeof(*packet_results),
                                                 (void**)&packet_results));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  uint32_t lane_offset = 0;
  for (uint32_t i = 0; i < packet_count; ++i) {
    const uint32_t lane_count =
        iree_min(source_lane_count - lane_offset, packet_lane_count);
    const int64_t static_lane_offset = lane_offset;
    const loom_type_t packet_source_type =
        loom_vector_packet_type(source_type, lane_count);
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        &rewriter->builder, source, /*offsets=*/NULL, /*offsets_count=*/0,
        &static_lane_offset, /*static_offsets_count=*/1, packet_source_type,
        op->location, &slice_op));
    const loom_type_t packet_result_type =
        loom_vector_packet_type(result_type, lane_count);
    loom_op_t* packet_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        &rewriter->builder, loom_vector_slice_result(slice_op),
        packet_source_type, packet_result_type, op->location, &packet_op));
    packet_results[i] = loom_vector_bitcast_result(packet_op);
    lane_offset += lane_count;
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_concat_build(
      &rewriter->builder, /*axis=*/0, packet_results, packet_count, result_type,
      op->location, &concat_op));
  const loom_value_id_t replacement = loom_vector_concat_result(concat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

static bool loom_vector_packet_predicate_extension_carriers_supported(
    const loom_target_legalization_context_t* context, loom_type_t source_type,
    loom_type_t result_type, uint32_t lane_count) {
  return loom_target_legalization_has_source_vector_carrier(
             context, loom_vector_packet_type(source_type, lane_count),
             LOOM_VALUE_ID_INVALID) &&
         loom_target_legalization_has_source_vector_carrier(
             context, loom_vector_packet_type(result_type, lane_count),
             LOOM_VALUE_ID_INVALID);
}

static bool loom_vector_predicate_extension_packet_matches(
    void* user_data, const loom_module_t* module, const loom_op_t* op) {
  (void)user_data;
  if ((op->kind != LOOM_OP_VECTOR_EXTSI && op->kind != LOOM_OP_VECTOR_EXTUI) ||
      op->operand_count != 1 || op->result_count != 1) {
    return false;
  }
  const loom_type_t source_type =
      loom_module_value_type(module, loom_op_const_operands(op)[0]);
  const loom_type_t result_type =
      loom_module_value_type(module, loom_op_const_results(op)[0]);
  uint32_t source_lane_count = 0;
  uint32_t result_lane_count = 0;
  return loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_I1 &&
         loom_vector_packet_static_lane_count(source_type,
                                              &source_lane_count) &&
         loom_vector_packet_static_lane_count(result_type,
                                              &result_lane_count) &&
         source_lane_count == result_lane_count;
}

static bool loom_vector_predicate_extension_packet_supports(
    void* user_data, const loom_target_legalization_context_t* context,
    const loom_op_t* op, uint32_t lane_count) {
  (void)user_data;
  const loom_type_t source_type =
      loom_module_value_type(context->module, loom_op_const_operands(op)[0]);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_const_results(op)[0]);
  uint32_t source_lane_count = 0;
  return lane_count != 0 &&
         loom_vector_packet_static_lane_count(source_type,
                                              &source_lane_count) &&
         lane_count <= source_lane_count &&
         loom_vector_packet_predicate_extension_carriers_supported(
             context, source_type, result_type, lane_count);
}

static iree_status_t loom_vector_predicate_extension_packet_materialize(
    void* user_data, loom_target_legalization_context_t* context,
    const loom_op_t* op, uint32_t lane_count, loom_value_id_t source,
    loom_value_id_t* out_result) {
  (void)user_data;
  const loom_type_t result_type = loom_vector_packet_type(
      loom_module_value_type(context->module, loom_op_const_results(op)[0]),
      lane_count);
  loom_builder_t* builder = &context->rewriter->builder;
  loom_op_t* true_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splati_build(
      builder, op->kind == LOOM_OP_VECTOR_EXTSI ? -1 : 1, result_type,
      op->location, &true_op));
  loom_op_t* false_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splati_build(builder, 0, result_type,
                                                op->location, &false_op));
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_select_build(
      builder, source, loom_vector_splati_result(true_op),
      loom_vector_splati_result(false_op), result_type, op->location,
      &select_op));
  *out_result = loom_vector_select_result(select_op);
  return iree_ok_status();
}

loom_vector_packet_consumer_materializer_t
loom_vector_predicate_extension_packet_materializer(void) {
  return (loom_vector_packet_consumer_materializer_t){
      .operation_count = 3,
      .matches = loom_vector_predicate_extension_packet_matches,
      .supports_packet = loom_vector_predicate_extension_packet_supports,
      .materialize = loom_vector_predicate_extension_packet_materialize,
  };
}

iree_status_t loom_vector_packet_legalize_predicate_extension(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten) {
  *out_rewritten = false;
  if ((op->kind != LOOM_OP_VECTOR_EXTSI && op->kind != LOOM_OP_VECTOR_EXTUI) ||
      policy == NULL) {
    return iree_ok_status();
  }

  const loom_value_id_t source = loom_op_operands(op)[0];
  const loom_value_id_t result = loom_op_results(op)[0];
  const loom_value_t* source_value = loom_module_value(context->module, source);
  loom_op_t* source_op = loom_value_def_op(source_value);
  if (loom_vector_slice_isa(source_op) &&
      loom_vector_slice_offsets(source_op).count != 0) {
    const loom_vector_packet_consumer_materializer_t materializer =
        loom_vector_predicate_extension_packet_materializer();
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_dynamic_slice(
        context, source_op, policy, &materializer, out_rewritten));
    if (*out_rewritten) {
      return iree_ok_status();
    }
  }
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const loom_type_t result_type =
      loom_module_value_type(context->module, result);
  uint32_t source_lane_count = 0;
  uint32_t result_lane_count = 0;
  if (loom_type_element_type(source_type) != LOOM_SCALAR_TYPE_I1 ||
      !loom_vector_packet_static_lane_count(source_type, &source_lane_count) ||
      !loom_vector_packet_static_lane_count(result_type, &result_lane_count) ||
      source_lane_count <= 1 || source_lane_count != result_lane_count) {
    return iree_ok_status();
  }

  uint32_t packet_lane_count = 0;
  for (uint8_t i = 0; i < policy->native_lane_count_count; ++i) {
    const uint32_t candidate_lane_count = policy->native_lane_counts[i];
    if (candidate_lane_count == 0 ||
        candidate_lane_count >= source_lane_count ||
        candidate_lane_count <= packet_lane_count ||
        !loom_vector_packet_predicate_extension_carriers_supported(
            context, source_type, result_type, candidate_lane_count)) {
      continue;
    }
    const uint32_t tail_lane_count = source_lane_count % candidate_lane_count;
    if (tail_lane_count != 0 &&
        !loom_vector_packet_predicate_extension_carriers_supported(
            context, source_type, result_type, tail_lane_count)) {
      continue;
    }
    const uint32_t packet_count =
        (source_lane_count - 1u) / candidate_lane_count + 1u;
    if ((uint64_t)packet_count * 4u + 1u > LOOM_VECTOR_CAST_PACKET_OP_LIMIT) {
      continue;
    }
    packet_lane_count = candidate_lane_count;
  }
  if (packet_lane_count == 0) {
    return iree_ok_status();
  }

  const uint32_t packet_count =
      (source_lane_count - 1u) / packet_lane_count + 1u;
  loom_value_id_t* packet_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, packet_count,
                                                 sizeof(*packet_results),
                                                 (void**)&packet_results));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  uint32_t lane_offset = 0;
  for (uint32_t i = 0; i < packet_count; ++i) {
    const uint32_t lane_count =
        iree_min(source_lane_count - lane_offset, packet_lane_count);
    const int64_t static_lane_offset = lane_offset;
    const loom_type_t packet_source_type =
        loom_vector_packet_type(source_type, lane_count);
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        &rewriter->builder, source, /*offsets=*/NULL, /*offsets_count=*/0,
        &static_lane_offset, /*static_offsets_count=*/1, packet_source_type,
        op->location, &slice_op));
    IREE_RETURN_IF_ERROR(loom_vector_predicate_extension_packet_materialize(
        /*user_data=*/NULL, context, op, lane_count,
        loom_vector_slice_result(slice_op), &packet_results[i]));
    lane_offset += lane_count;
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_concat_build(
      &rewriter->builder, /*axis=*/0, packet_results, packet_count, result_type,
      op->location, &concat_op));
  const loom_value_id_t replacement = loom_vector_concat_result(concat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

typedef struct loom_vector_packet_bitstream_shape_t {
  // Number of ordinary logical lanes represented by the authored operation.
  uint32_t logical_lane_count;
  // Number of packed storage lanes represented by the authored operation.
  uint32_t storage_lane_count;
  // Number of ordinary logical lanes in each full packet.
  uint32_t packet_logical_lane_count;
  // Number of packed storage lanes in each full packet.
  uint32_t packet_storage_lane_count;
  // Number of packets covering the authored operation.
  uint32_t packet_count;
} loom_vector_packet_bitstream_shape_t;

static bool loom_vector_packet_bitstream_projection(
    const loom_vector_packet_bitstream_shape_t* shape,
    uint32_t projected_logical_lane_count,
    uint32_t projected_storage_lane_count,
    loom_target_contract_vector_lane_projection_t* out_projection) {
  *out_projection = (loom_target_contract_vector_lane_projection_t){
      .source_lane_count = shape->logical_lane_count,
      .projected_lane_count = projected_logical_lane_count,
  };
  if (shape->storage_lane_count == shape->logical_lane_count) {
    return projected_storage_lane_count == projected_logical_lane_count;
  }
  out_projection->secondary_source_lane_count = shape->storage_lane_count;
  out_projection->secondary_projected_lane_count = projected_storage_lane_count;
  return true;
}

static iree_status_t loom_vector_packet_bitstream_candidate_is_legal(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_vector_packet_bitstream_shape_t* shape,
    uint32_t logical_lane_count, uint32_t storage_lane_count, bool* out_legal) {
  *out_legal = false;
  loom_target_contract_vector_lane_projection_t projection = {0};
  if (!loom_vector_packet_bitstream_projection(
          shape, logical_lane_count, storage_lane_count, &projection)) {
    return iree_ok_status();
  }
  loom_target_contract_query_result_t query_result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(
      loom_target_legalization_query_contract_with_vector_lane_projection(
          context, op, projection, &query_result));
  *out_legal = query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_bitstream_select_shape(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t logical_type,
    loom_type_t storage_type, uint32_t width,
    loom_vector_packet_bitstream_shape_t* out_shape) {
  *out_shape = (loom_vector_packet_bitstream_shape_t){0};
  uint32_t logical_lane_count = 0;
  uint32_t storage_lane_count = 0;
  if (!loom_vector_packet_static_lane_count(logical_type,
                                            &logical_lane_count) ||
      !loom_vector_packet_static_lane_count(storage_type,
                                            &storage_lane_count)) {
    return iree_ok_status();
  }
  const int32_t storage_element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_type));
  if (width == 0 || storage_element_bit_count <= 0 ||
      (uint64_t)logical_lane_count * width !=
          (uint64_t)storage_lane_count * (uint32_t)storage_element_bit_count) {
    return iree_ok_status();
  }

  loom_vector_packet_bitstream_shape_t shape = {
      .logical_lane_count = logical_lane_count,
      .storage_lane_count = storage_lane_count,
  };
  for (uint8_t i = 0; i < policy->native_lane_count_count; ++i) {
    const uint32_t candidate_logical_lane_count = policy->native_lane_counts[i];
    if (candidate_logical_lane_count == 0 ||
        candidate_logical_lane_count >= logical_lane_count ||
        candidate_logical_lane_count <= shape.packet_logical_lane_count) {
      continue;
    }
    const uint64_t candidate_storage_bit_count =
        (uint64_t)candidate_logical_lane_count * width;
    if (candidate_storage_bit_count % (uint32_t)storage_element_bit_count !=
        0) {
      continue;
    }
    const uint64_t candidate_storage_lane_count =
        candidate_storage_bit_count / (uint32_t)storage_element_bit_count;
    if (candidate_storage_lane_count == 0 ||
        candidate_storage_lane_count > UINT32_MAX) {
      continue;
    }
    const uint32_t packet_count =
        (logical_lane_count - 1u) / candidate_logical_lane_count + 1u;
    if ((uint64_t)packet_count * 2u + 1u > LOOM_VECTOR_CAST_PACKET_OP_LIMIT) {
      continue;
    }

    bool full_packet_is_legal = false;
    IREE_RETURN_IF_ERROR(loom_vector_packet_bitstream_candidate_is_legal(
        context, op, &shape, candidate_logical_lane_count,
        (uint32_t)candidate_storage_lane_count, &full_packet_is_legal));
    if (!full_packet_is_legal) {
      continue;
    }
    const uint32_t tail_logical_lane_count =
        logical_lane_count % candidate_logical_lane_count;
    if (tail_logical_lane_count != 0) {
      const uint64_t tail_storage_bit_count =
          (uint64_t)tail_logical_lane_count * width;
      if (tail_storage_bit_count % (uint32_t)storage_element_bit_count != 0) {
        continue;
      }
      const uint32_t tail_storage_lane_count =
          (uint32_t)(tail_storage_bit_count /
                     (uint32_t)storage_element_bit_count);
      bool tail_packet_is_legal = false;
      IREE_RETURN_IF_ERROR(loom_vector_packet_bitstream_candidate_is_legal(
          context, op, &shape, tail_logical_lane_count, tail_storage_lane_count,
          &tail_packet_is_legal));
      if (!tail_packet_is_legal) {
        continue;
      }
    }

    shape.packet_logical_lane_count = candidate_logical_lane_count;
    shape.packet_storage_lane_count = (uint32_t)candidate_storage_lane_count;
    shape.packet_count = packet_count;
  }
  *out_shape = shape;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_legalize_bitstream(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, bool* out_rewritten) {
  *out_rewritten = false;
  const bool is_bitpack = loom_vector_bitpack_isa(op);
  const bool is_unsigned_bitunpack = loom_vector_bitunpacku_isa(op);
  const bool is_signed_bitunpack = loom_vector_bitunpacks_isa(op);
  if ((!is_bitpack && !is_unsigned_bitunpack && !is_signed_bitunpack) ||
      policy == NULL) {
    return iree_ok_status();
  }

  const loom_value_id_t source =
      is_bitpack ? loom_vector_bitpack_source(op)
                 : (is_unsigned_bitunpack ? loom_vector_bitunpacku_source(op)
                                          : loom_vector_bitunpacks_source(op));
  const loom_value_id_t result =
      is_bitpack ? loom_vector_bitpack_result(op)
                 : (is_unsigned_bitunpack ? loom_vector_bitunpacku_result(op)
                                          : loom_vector_bitunpacks_result(op));
  const int64_t width_i64 =
      is_bitpack ? loom_vector_bitpack_width(op)
                 : (is_unsigned_bitunpack ? loom_vector_bitunpacku_width(op)
                                          : loom_vector_bitunpacks_width(op));
  if (width_i64 <= 0 || width_i64 > UINT32_MAX) {
    return iree_ok_status();
  }
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const loom_type_t result_type =
      loom_module_value_type(context->module, result);
  const loom_type_t logical_type = is_bitpack ? source_type : result_type;
  const loom_type_t storage_type = is_bitpack ? result_type : source_type;
  loom_vector_packet_bitstream_shape_t shape = {0};
  IREE_RETURN_IF_ERROR(loom_vector_packet_bitstream_select_shape(
      context, op, policy, logical_type, storage_type, (uint32_t)width_i64,
      &shape));
  if (shape.packet_count == 0) {
    return iree_ok_status();
  }

  loom_value_id_t* packet_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, shape.packet_count, sizeof(*packet_results),
      (void**)&packet_results));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  uint32_t logical_lane_offset = 0;
  uint32_t storage_lane_offset = 0;
  for (uint32_t i = 0; i < shape.packet_count; ++i) {
    const uint32_t logical_lane_count =
        iree_min(shape.logical_lane_count - logical_lane_offset,
                 shape.packet_logical_lane_count);
    const uint32_t storage_lane_count =
        iree_min(shape.storage_lane_count - storage_lane_offset,
                 shape.packet_storage_lane_count);
    const uint32_t source_lane_offset =
        is_bitpack ? logical_lane_offset : storage_lane_offset;
    const uint32_t source_lane_count =
        is_bitpack ? logical_lane_count : storage_lane_count;
    const int64_t static_source_lane_offset = source_lane_offset;
    const loom_type_t packet_source_type =
        loom_vector_packet_type(source_type, source_lane_count);
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        &rewriter->builder, source, /*offsets=*/NULL, /*offsets_count=*/0,
        &static_source_lane_offset, /*static_offsets_count=*/1,
        packet_source_type, op->location, &slice_op));
    const loom_value_id_t packet_source = loom_vector_slice_result(slice_op);
    const uint32_t result_lane_count =
        is_bitpack ? storage_lane_count : logical_lane_count;
    const loom_type_t packet_result_type =
        loom_vector_packet_type(result_type, result_lane_count);
    loom_op_t* packet_op = NULL;
    if (is_bitpack) {
      IREE_RETURN_IF_ERROR(loom_vector_bitpack_build(
          &rewriter->builder, width_i64, packet_source, packet_result_type,
          op->location, &packet_op));
      packet_results[i] = loom_vector_bitpack_result(packet_op);
    } else if (is_unsigned_bitunpack) {
      IREE_RETURN_IF_ERROR(loom_vector_bitunpacku_build(
          &rewriter->builder, width_i64, packet_source, packet_result_type,
          op->location, &packet_op));
      packet_results[i] = loom_vector_bitunpacku_result(packet_op);
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_bitunpacks_build(
          &rewriter->builder, width_i64, packet_source, packet_result_type,
          op->location, &packet_op));
      packet_results[i] = loom_vector_bitunpacks_result(packet_op);
    }
    logical_lane_offset += logical_lane_count;
    storage_lane_offset += storage_lane_count;
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_concat_build(
      &rewriter->builder, /*axis=*/0, packet_results, shape.packet_count,
      result_type, op->location, &concat_op));
  const loom_value_id_t replacement = loom_vector_concat_result(concat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}
