// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/legalization.h"

#include <string.h>

#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/gather.h"
#include "loom/target/arch/amd/xdna/aie2p/legalization_compare.h"
#include "loom/target/arch/amd/xdna/aie2p/legalization_table.h"
#include "loom/transforms/scalar/target_legalization.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/transforms/vector/shape_legalization.h"
#include "loom/transforms/vector/table_legalization.h"
#include "loom/transforms/vector/target_legalization.h"
#include "loom/transforms/vector/to_scalar.h"

static bool loom_aie2p_legalizer_descriptor_set_is_core(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set == loom_aie2p_core_descriptor_set();
}

// Multidimensional memory survives eager legalization for shared view
// normalization. Final legalization owns fallback for layouts left unchanged.
static bool loom_aie2p_defer_multidimensional_memory_reference(
    const loom_target_legalization_context_t* context,
    loom_type_t payload_type) {
  return context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL &&
         loom_type_rank(payload_type) > 1;
}

static bool loom_aie2p_match_scalar_multiply_add(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  loom_scalar_multiply_add_match_t match = {0};
  return loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set) &&
         loom_scalar_match_multiply_add(context->module, op, &match);
}

static iree_status_t loom_aie2p_legalize_scalar_multiply_add(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  loom_scalar_multiply_add_match_t match = {0};
  if (!loom_scalar_match_multiply_add(context->module, op, &match)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_scalar_fuse_multiply_add_match(context->rewriter, &match));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

// AIE vector registers are linear carriers: logical multidimensional shapes
// do not survive source-to-low type conversion. Normalize unsupported lane
// access and construction shapes while their row-major semantics are visible.
static iree_status_t loom_aie2p_legalize_linear_vector_shape(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  if (loom_vector_extract_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vector_extract_flatten_static_shape_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  } else if (loom_vector_from_elements_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vector_from_elements_linearize_rewrite_op(
        context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_to_scalar(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }
  if (!loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_static_vector_shape_or_scalarize(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }
  if (!loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_static_shape_rewrite_op(context, op, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

// Static broadcasts have a bounded structural legalization. Shapes above that
// bound remain unsupported instead of falling through to an even larger
// lane-by-lane scalar expansion.
static iree_status_t loom_aie2p_legalize_static_vector_broadcast(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set) ||
      !loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_static_shape_rewrite_op(context, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  } else {
    out_result->action =
        context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL
            ? LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL
            : LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_insert(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }
  if (!loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_static_shape_rewrite_op(context, op, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_insert(
        context, op, context->vector_packet_policy, &rewritten));
  }
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static bool loom_aie2p_match_i8_4x4_transpose_shuffle(
    const loom_module_t* module, const loom_op_t* op) {
  static const int64_t kTransposeSourceLanes[] = {
      0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
  };
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_shuffle_source(op));
  if (loom_type_element_type(source_type) != LOOM_SCALAR_TYPE_I8 ||
      loom_type_rank(source_type) != 1 ||
      loom_type_dim_is_dynamic_at(source_type, 0) ||
      loom_type_dim_static_size_at(source_type, 0) != 16) {
    return false;
  }
  const loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(op);
  return source_lanes.count == IREE_ARRAYSIZE(kTransposeSourceLanes) &&
         memcmp(source_lanes.i64_array, kTransposeSourceLanes,
                sizeof(kTransposeSourceLanes)) == 0;
}

static iree_status_t loom_aie2p_legalize_vector_shuffle(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }
  if (!loom_aie2p_match_i8_4x4_transpose_shuffle(context->module, op)) {
    return loom_aie2p_legalize_vector_to_scalar(entry, context, op, out_result);
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t source = loom_vector_shuffle_source(op);
  const loom_type_t flat_type = loom_module_value_type(context->module, source);
  const loom_type_t matrix_type = loom_type_shaped_2d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(4),
      loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_op_t* matrix_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(&rewriter->builder, source,
                                                 flat_type, matrix_type,
                                                 op->location, &matrix_op));
  const int64_t permutation[] = {1, 0};
  loom_op_t* transpose_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_transpose_build(
      &rewriter->builder, permutation, IREE_ARRAYSIZE(permutation),
      loom_vector_bitcast_result(matrix_op), matrix_type, op->location,
      &transpose_op));
  loom_op_t* flat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
      &rewriter->builder, loom_vector_transpose_result(transpose_op),
      matrix_type, flat_type, op->location, &flat_op));
  const loom_value_id_t replacement = loom_vector_bitcast_result(flat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_select(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_select_result(op));
  bool rewritten = false;
  if (loom_type_rank(result_type) > 1) {
    IREE_RETURN_IF_ERROR(
        loom_vector_static_shape_rewrite_op(context, op, &rewritten));
  }
  if (!rewritten &&
      loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_concat(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_value_slice_t inputs = loom_vector_concat_inputs(op);
  if (inputs.count <= 2) {
    return iree_ok_status();
  }

  // AIE2P contracts describe binary carrier transitions. Normalize static
  // variadic inputs through the shared adjacent-pair canonicalizer before
  // selecting those contracts.
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_concat_result(op));
  const int64_t axis = loom_vector_concat_axis(op);
  if (axis < 0 || axis >= loom_type_rank(result_type)) {
    return iree_ok_status();
  }
  bool has_nonempty_input = false;
  for (iree_host_size_t i = 0; i < inputs.count; ++i) {
    const loom_type_t input_type =
        loom_module_value_type(context->module, inputs.values[i]);
    if (loom_type_dim_is_dynamic_at(input_type, (uint8_t)axis)) {
      return iree_ok_status();
    }
    has_nonempty_input |=
        loom_type_dim_static_size_at(input_type, (uint8_t)axis) != 0;
  }
  if (has_nonempty_input) {
    IREE_RETURN_IF_ERROR(
        loom_vector_concat_canonicalize(op, context->rewriter));
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_table_lookup(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_table_lookup_rewrite(
      context, op, context->vector_packet_policy, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(
        loom_vector_static_shape_rewrite_op(context, op, &rewritten));
  }
  if (!rewritten &&
      loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_table_quantize(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }
  const loom_type_t input_type = loom_module_value_type(
      context->module, loom_vector_table_quantize_input(op));
  const loom_type_t result_type = loom_module_value_type(
      context->module, loom_vector_table_quantize_result(op));
  const loom_scalar_type_t input_element_type =
      loom_type_element_type(input_type);
  const loom_scalar_type_t result_element_type =
      loom_type_element_type(result_type);
  const uint32_t result_bit_count =
      loom_scalar_type_bitwidth(result_element_type);
  const uint32_t packet_bit_count = 512;
  if ((input_element_type != LOOM_SCALAR_TYPE_F16 &&
       input_element_type != LOOM_SCALAR_TYPE_BF16 &&
       input_element_type != LOOM_SCALAR_TYPE_F32) ||
      result_bit_count > 64) {
    return iree_ok_status();
  }
  const loom_vector_table_quantize_policy_t policy = {
      .packet_bit_count = packet_bit_count,
      .comparison_element_type = input_element_type,
      .ordinal_element_type =
          result_bit_count > 32 ? LOOM_SCALAR_TYPE_I32 : result_element_type,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_table_quantize_rewrite(context, op, &policy, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_load(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_load(
      context, op, context->vector_packet_policy, &rewritten));
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_load_result(op));
  if (!rewritten &&
      !loom_aie2p_defer_multidimensional_memory_reference(context,
                                                          result_type) &&
      loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_gather(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  loom_low_source_memory_access_plan_t access = {0};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  loom_aie2p_immutable_gather_match_t match = {0};
  if (loom_low_source_memory_access_plan_build(context->view_regions, op,
                                               &access, &diagnostic) &&
      loom_aie2p_match_immutable_gather(context->module, context->fact_table,
                                        op, &access, &match)) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_type_t value_type =
      loom_module_value_type(context->module, loom_vector_store_value(op));
  const bool has_native_store = context->contract_query_result->outcome ==
                                LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  if (has_native_store) {
    // Keep an accepted store for target lowering if its producer is opaque to
    // packetization instead of continuing into semantic reference rewrites.
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
    // Accepted memory rules have static rank-one payloads. The ordinary
    // four-W carrier still permits packetization of decomposable producers;
    // native-width and accumulator stores retain their selected realization.
    const uint64_t payload_bit_count =
        (uint64_t)loom_type_dim_static_size_at(value_type, 0) *
        loom_scalar_type_bitwidth(loom_type_element_type(value_type));
    if (payload_bit_count != 1024) {
      return iree_ok_status();
    }
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_store(
      context, op, context->vector_packet_policy, &rewritten));
  if (!rewritten && !has_native_store &&
      !loom_aie2p_defer_multidimensional_memory_reference(context,
                                                          value_type) &&
      loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    IREE_RETURN_IF_ERROR(loom_vector_store_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }

  loom_vector_packet_reduce_result_t packet_result =
      LOOM_VECTOR_PACKET_REDUCE_RESULT_NONE;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_reduce(
      context, op, context->vector_packet_policy, &packet_result));
  bool rewritten = packet_result == LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
  const bool may_scalarize =
      loom_target_legalization_op_has_source_vector_carriers(context, op);
  if (packet_result == LOOM_VECTOR_PACKET_REDUCE_RESULT_CAPTURE_INPUT &&
      may_scalarize) {
    IREE_RETURN_IF_ERROR(loom_vector_reduce_captured_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  } else if (!rewritten && may_scalarize) {
    IREE_RETURN_IF_ERROR(loom_vector_reduce_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_reduce_axes(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    return iree_ok_status();
  }
  if (!loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_axes_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kAie2pLegalizerRules[] = {
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_SCALAR_MULI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I32,
        .match = loom_aie2p_match_scalar_multiply_add,
        .legalize = loom_aie2p_legalize_scalar_multiply_add,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BROADCAST,
        .legalize = loom_aie2p_legalize_static_vector_broadcast,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SLICE,
        .legalize = loom_aie2p_legalize_static_vector_shape_or_scalarize,
    },
    {
        .root_kind = LOOM_OP_VECTOR_INSERT,
        .legalize = loom_aie2p_legalize_vector_insert,
    },
    {
        .root_kind = LOOM_OP_VECTOR_INTERLEAVE,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRANSPOSE,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHUFFLE,
        .legalize = loom_aie2p_legalize_vector_shuffle,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTF,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FPTRUNC,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SITOFP,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_UITOFP,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FPTOSI,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FPTOUI,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SELECT,
        .legalize = loom_aie2p_legalize_vector_select,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CONCAT,
        .legalize = loom_aie2p_legalize_vector_concat,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINNUMF,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXNUMF,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINIMUMF,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXIMUMF,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CLAMPF,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_LOOKUP,
        .legalize = loom_aie2p_legalize_table_lookup,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_QUANTIZE,
        .legalize = loom_aie2p_legalize_table_quantize,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FROM_ELEMENTS,
        .legalize = loom_aie2p_legalize_linear_vector_shape,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTRACT,
        .legalize = loom_aie2p_legalize_linear_vector_shape,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CMPF,
        .legalize = loom_aie2p_legalize_vector_cmpf,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DEINTERLEAVE,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_LOAD,
        .legalize = loom_aie2p_legalize_vector_load,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_VECTOR_GATHER,
        .legalize = loom_aie2p_legalize_vector_gather,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
        .root_kind = LOOM_OP_VECTOR_STORE,
        .legalize = loom_aie2p_legalize_vector_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE,
        .legalize = loom_aie2p_legalize_vector_reduce,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE_AXES,
        .legalize = loom_aie2p_legalize_vector_reduce_axes,
    },
};

const loom_target_legalizer_provider_t
    loom_aie2p_target_legalizer_provider_storage = {
        .name = IREE_SVL("aie2p"),
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
        .rules = kAie2pLegalizerRules,
        .rule_count = IREE_ARRAYSIZE(kAie2pLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_aie2p_target_legalizer_provider(
    void) {
  return &loom_aie2p_target_legalizer_provider_storage;
}
