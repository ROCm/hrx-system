// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/target_legalization.h"

#include <math.h>
#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/transpose.h"
#include "loom/transforms/vector/reduction_legalization.h"
#include "loom/transforms/vector/to_scalar.h"
#include "loom/util/numeric_format.h"

static loom_vector_mma_to_scalar_options_t loom_vector_mma_options(
    const loom_target_legalization_context_t* context) {
  const loom_target_contract_query_result_t* query_result =
      context->contract_query_result;
  loom_vector_to_scalar_flags_t flags = LOOM_VECTOR_TO_SCALAR_FLAG_NONE;
  if (loom_kernel_def_isa(context->function.op)) {
    flags |= LOOM_VECTOR_TO_SCALAR_FLAG_ALLOW_SUBGROUP_COMMUNICATION;
  }
  return (loom_vector_mma_to_scalar_options_t){
      .matrix_fragment_layout =
          query_result ? query_result->selected_matrix_fragment_layout : NULL,
      .flags = flags,
  };
}

static bool loom_vector_mma_has_fragment_store_user(const loom_module_t* module,
                                                    const loom_op_t* op) {
  const loom_value_id_t result_id = loom_vector_mma_result(op);
  const loom_value_t* result = loom_module_value(module, result_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(result, use) {
    const loom_op_t* user = loom_use_user_op(*use);
    if (loom_vector_memory_op_footprint_kind(module, user) !=
        LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT) {
      continue;
    }
    loom_memory_access_t access = loom_memory_access_cast(module, user);
    if (loom_memory_access_operation_kind(access) ==
            LOOM_MEMORY_ACCESS_OPERATION_STORE &&
        loom_memory_access_value(access) == result_id) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_vector_legalize_reduce_axes(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->source_function_has_unsupported_vector_carrier ||
      !loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_axes_to_vector_rewrite_op(
      context->rewriter, op, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_reduce_axes_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_descriptor(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->source_function_has_unsupported_vector_carrier ||
      !loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_non_dense_memory(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_atomic(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_atomic_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_transpose_to_shuffle_rewrite_op(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_transpose_isa(op)) {
    return iree_ok_status();
  }

  const loom_value_id_t source = loom_vector_transpose_source(op);
  const loom_type_t source_type =
      loom_module_value_type(rewriter->module, source);
  const loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_transpose_result(op));
  uint64_t element_count = 0;
  if (!loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type) ||
      !loom_type_static_element_count(source_type, &element_count) ||
      element_count > UINT16_MAX) {
    return iree_ok_status();
  }

  int64_t* source_lanes = NULL;
  if (element_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, (iree_host_size_t)element_count, sizeof(*source_lanes),
        (void**)&source_lanes));
  }

  const loom_attribute_t permutation = loom_vector_transpose_permutation(op);
  bool is_identity = true;
  for (uint64_t result_lane = 0; result_lane < element_count; ++result_lane) {
    const uint64_t source_lane = loom_vector_transpose_source_lane(
        source_type, result_type, permutation.i64_array, result_lane);
    source_lanes[result_lane] = (int64_t)source_lane;
    is_identity &= source_lane == result_lane;
  }

  const loom_type_t flat_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(source_type),
      loom_dim_pack_static((int64_t)element_count), /*encoding_id=*/0);
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t flat_source = source;
  if (!loom_type_equal(source_type, flat_type)) {
    loom_op_t* flatten_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(&rewriter->builder, source,
                                                   source_type, flat_type,
                                                   op->location, &flatten_op));
    flat_source = loom_vector_bitcast_result(flatten_op);
  }

  loom_value_id_t replacement = flat_source;
  if (!is_identity) {
    loom_op_t* shuffle_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_shuffle_build(
        &rewriter->builder, source_lanes, (iree_host_size_t)element_count,
        flat_source, flat_type, op->location, &shuffle_op));
    replacement = loom_vector_shuffle_result(shuffle_op);
  }
  if (!loom_type_equal(flat_type, result_type)) {
    loom_op_t* restore_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_bitcast_build(&rewriter->builder, replacement, flat_type,
                                  result_type, op->location, &restore_op));
    replacement = loom_vector_bitcast_result(restore_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_transpose(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->source_function_has_unsupported_vector_carrier ||
      !loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_transpose_to_shuffle_rewrite_op(
      context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_from_elements_linearize_rewrite_op(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_from_elements_isa(op)) {
    return iree_ok_status();
  }
  const loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count == 0) {
    return iree_ok_status();
  }
  const loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_from_elements_result(op));
  if (!loom_type_is_vector(result_type) ||
      !loom_type_is_all_static(result_type)) {
    return iree_ok_status();
  }

  const bool restore_shape = loom_type_rank(result_type) != 1;
  const loom_type_t construction_type =
      restore_shape ? loom_type_shaped_1d(
                          LOOM_TYPE_VECTOR, loom_type_element_type(result_type),
                          loom_dim_pack_static((int64_t)elements.count),
                          /*encoding_id=*/0)
                    : result_type;

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* splat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_splat_build(&rewriter->builder, elements.values[0],
                              construction_type, op->location, &splat_op));
  loom_value_id_t replacement = loom_vector_splat_result(splat_op);

  for (iree_host_size_t element_index = 1; element_index < elements.count;
       ++element_index) {
    const int64_t static_index = (int64_t)element_index;
    loom_op_t* insert_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_insert_build(
        &rewriter->builder, elements.values[element_index], replacement, NULL,
        0, &static_index, 1, construction_type, op->location, &insert_op));
    replacement = loom_vector_insert_result(insert_op);
  }

  if (restore_shape) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        &rewriter->builder, replacement, construction_type, result_type,
        op->location, &bitcast_op));
    replacement = loom_vector_bitcast_result(bitcast_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

// Expands a variadic rank-one vector constructor into the fixed-arity
// structural ops that targets commonly select for linear register vectors.
// Splatting the first lane provides a defined seed without requiring a
// target-level poison or zero materialization.
static iree_status_t loom_vector_legalize_from_elements(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_type_t result_type = loom_module_value_type(
      context->module, loom_vector_from_elements_result(op));
  if (loom_type_rank(result_type) != 1) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_linearize_rewrite_op(
      context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_dotf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_dotf_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_transform(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_transform_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_decode(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_decode_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
    return iree_ok_status();
  }
  if (context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL,
        .source_rejection_bits =
            loom_vector_decode_to_scalar_reference_rejection_bits(
                context->pass, context->rewriter, op),
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_mma(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }
  if (loom_vector_mma_has_fragment_store_user(context->module, op)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  const loom_vector_mma_to_scalar_options_t options =
      loom_vector_mma_options(context);
  IREE_RETURN_IF_ERROR(loom_vector_mma_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, options, &rewritten));
  *out_result = (loom_target_legalizer_result_t){
      .action = rewritten
                    ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                    : LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL,
      .source_rejection_bits =
          rewritten ? 0
                    : loom_vector_mma_to_scalar_reference_rejection_bits(
                          context->pass, context->rewriter, op, options),
      .source_rejection_detail =
          rewritten ? 0
                    : loom_vector_mma_to_scalar_reference_rejection_detail(
                          context->pass, context->rewriter, op, options),
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_store_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_fragment_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_fragment_store_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  *out_result = (loom_target_legalizer_result_t){
      .action = rewritten
                    ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                    : LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL,
      .source_rejection_bits =
          rewritten
              ? 0
              : loom_vector_fragment_store_to_scalar_reference_rejection_bits(
                    context->pass, context->rewriter, op),
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_extract(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_extract_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_predicate_extension(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  loom_value_id_t input = loom_op_operands(op)[0];
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  loom_op_t* true_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &rewriter->builder,
      loom_attr_i64(op->kind == LOOM_OP_VECTOR_EXTSI ? -1 : 1), result_type,
      op->location, &true_op));
  loom_op_t* false_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(&rewriter->builder,
                                                  loom_attr_i64(0), result_type,
                                                  op->location, &false_op));
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_select_build(
      &rewriter->builder, input, loom_vector_constant_result(true_op),
      loom_vector_constant_result(false_op), result_type, op->location,
      &select_op));
  loom_value_id_t replacement = loom_vector_select_result(select_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

// Number-preferring extrema already implement signed-zero ordering. An
// unordered comparison adds the IEEE NaN policy without changing numeric
// operands or exposing a NaN payload guarantee.
static iree_status_t loom_vector_legalize_ieee_extrema(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t lhs = loom_op_operands(op)[0];
  const loom_value_id_t rhs = loom_op_operands(op)[1];
  const loom_type_t type = loom_module_value_type(context->module, lhs);
  loom_type_t predicate_type = type;
  predicate_type.header =
      loom_type_make_header(loom_type_kind(type), LOOM_SCALAR_TYPE_I1,
                            loom_type_rank(type), loom_type_flags(type));

  loom_op_t* number = NULL;
  if (loom_vector_minimumf_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_vector_minnumf_build(
        builder, op->instance_flags, lhs, rhs, type, op->location, &number));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_maxnumf_build(
        builder, op->instance_flags, lhs, rhs, type, op->location, &number));
  }
  loom_op_t* unordered = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(
      builder, op->instance_flags, LOOM_VECTOR_CMPF_PREDICATE_UNO, lhs, rhs,
      type, predicate_type, op->location, &unordered));
  loom_op_t* nan = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(builder, loom_attr_f64(NAN),
                                                  type, op->location, &nan));
  loom_op_t* select = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_select_build(
      builder, loom_vector_cmpf_result(unordered),
      loom_vector_constant_result(nan), loom_op_results(number)[0], type,
      op->location, &select));
  const loom_value_id_t replacement = loom_vector_select_result(select);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_float_classification(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_value_id_t input = loom_op_operands(op)[0];
  const loom_type_t float_type = loom_module_value_type(context->module, input);
  loom_numeric_float_encoding_t encoding = {0};
  if (!loom_numeric_float_encoding(loom_type_element_type(float_type),
                                   &encoding)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (op->kind == LOOM_OP_VECTOR_ISINFF &&
      encoding.special_layout == LOOM_NUMERIC_FLOAT_SPECIAL_LAYOUT_FINITE_NAN) {
    loom_op_t* false_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64(0), result_type, op->location, &false_op));
    replacement = loom_vector_constant_result(false_op);
  } else {
    loom_type_t integer_type = float_type;
    integer_type.header = loom_type_make_header(
        loom_type_kind(float_type), encoding.integer_type,
        loom_type_rank(float_type), loom_type_flags(float_type));
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        builder, input, float_type, integer_type, op->location, &bitcast_op));
    loom_op_t* magnitude_mask_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64((int64_t)encoding.magnitude_mask), integer_type,
        op->location, &magnitude_mask_op));
    loom_op_t* magnitude_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_andi_build(builder, loom_vector_bitcast_result(bitcast_op),
                               loom_vector_constant_result(magnitude_mask_op),
                               integer_type, op->location, &magnitude_op));
    loom_op_t* special_magnitude_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64((int64_t)encoding.special_magnitude),
        integer_type, op->location, &special_magnitude_op));

    loom_vector_cmpi_predicate_t predicate = LOOM_VECTOR_CMPI_PREDICATE_EQ;
    if (op->kind == LOOM_OP_VECTOR_ISNANF) {
      predicate =
          encoding.special_layout == LOOM_NUMERIC_FLOAT_SPECIAL_LAYOUT_IEEE
              ? LOOM_VECTOR_CMPI_PREDICATE_UGT
              : LOOM_VECTOR_CMPI_PREDICATE_EQ;
    } else if (op->kind == LOOM_OP_VECTOR_ISFINITEF) {
      predicate =
          encoding.special_layout == LOOM_NUMERIC_FLOAT_SPECIAL_LAYOUT_IEEE
              ? LOOM_VECTOR_CMPI_PREDICATE_ULT
              : LOOM_VECTOR_CMPI_PREDICATE_NE;
    }
    loom_op_t* comparison_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
        builder, predicate, loom_vector_andi_result(magnitude_op),
        loom_vector_constant_result(special_magnitude_op), integer_type,
        result_type, op->location, &comparison_op));
    replacement = loom_vector_cmpi_result(comparison_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kVectorLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_VECTOR_LOAD_MASK,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE_MASK,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_LOAD_EXPAND,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE_COMPRESS,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINIMUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_ieee_extrema,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXIMUMF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_F32,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_ieee_extrema,
    },
    // Combining operations share one complete reference family. Native target
    // contracts retain precedence; explicit rejection selects the ordinary
    // scalar lane recipe. Keep this list aligned with loom_combining_kind_t.
    {
        .root_kind = LOOM_OP_VECTOR_ADDI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ADDF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MULI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MULF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINSI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXSI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINUI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXUI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ANDI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ORI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_XORI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINIMUMF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXIMUMF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MINNUMF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MAXNUMF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ISNANF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_vector_legalize_float_classification,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ISINFF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_vector_legalize_float_classification,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ISFINITEF,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_vector_legalize_float_classification,
    },
    {
        .root_kind = LOOM_OP_VECTOR_GATHER,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_GATHER_MASK,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SCATTER,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SCATTER_MASK,
        .legalize = loom_vector_legalize_non_dense_memory,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_REDUCE,
        .legalize = loom_vector_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK,
        .legalize = loom_vector_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_RMW,
        .legalize = loom_vector_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_RMW_MASK,
        .legalize = loom_vector_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_CMPXCHG,
        .legalize = loom_vector_legalize_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1,
        .legalize = loom_vector_legalize_predicate_extension,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_I1,
        .legalize = loom_vector_legalize_predicate_extension,
    },
// Every cast has a generated or explicit lane program. Keep this family tied
// to the dialect definition so adding a cast cannot silently omit its target
// reference fallback. Target-native contracts and the predicate-extension
// specializations above retain precedence over these rejected-contract rows.
#define LOOM_VECTOR_CAST_LEGALIZER_ROW(op_kind)                             \
  {                                                                         \
      .root_kind = (op_kind),                                               \
      .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION, \
      .legalize = loom_vector_legalize_descriptor,                          \
  },
#include "loom/ops/vector/cast_legalizer_rows.inl"
#undef LOOM_VECTOR_CAST_LEGALIZER_ROW
    {
        .root_kind = LOOM_OP_SCF_SELECT,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FROM_ELEMENTS,
        .legalize = loom_vector_legalize_from_elements,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MASK_RANGE,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRANSPOSE,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_transpose,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHUFFLE,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CMPI,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CMPF,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE,
        .legalize = loom_vector_legalize_reduce,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE_AXES,
        .legalize = loom_vector_legalize_reduce_axes,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_EXTRACTU,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_EXTRACTS,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_INSERT,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITPACK,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKU,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKS,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_QUANTIZE,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_LOOKUP,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DIVSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DIVUI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REMSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REMUI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CEILDIVSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CEILDIVUI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FLOORDIVSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHLI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHRSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHRUI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CTLZI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CTTZI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_CTPOPI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOTF,
        .legalize = loom_vector_legalize_dotf,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRANSFORM,
        .legalize = loom_vector_legalize_transform,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT2F,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4I,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT8I4,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4F8,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DECODE,
        .legalize = loom_vector_legalize_decode,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MMA,
        .legalize = loom_vector_legalize_mma,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE,
        .legalize = loom_vector_legalize_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FRAGMENT_STORE,
        .legalize = loom_vector_legalize_fragment_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTRACT,
        .legalize = loom_vector_legalize_extract,
    },
};

static const loom_target_legalizer_provider_t kVectorLegalizerProvider = {
    .name = IREE_SVL("vector"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .rules = kVectorLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kVectorLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_vector_target_legalizer_provider(
    void) {
  return &kVectorLegalizerProvider;
}
