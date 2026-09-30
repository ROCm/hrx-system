// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/legalization.h"

#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/legalization_compare.h"
#include "loom/target/arch/amd/xdna/aie2p/legalization_table.h"
#include "loom/transforms/scalar/target_legalization.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/transforms/vector/shape_legalization.h"
#include "loom/transforms/vector/table_legalization.h"
#include "loom/transforms/vector/target_legalization.h"
#include "loom/transforms/vector/to_scalar.h"

static const uint16_t kAie2pVectorPacketBitCounts[] = {128u, 256u, 512u};

static const loom_vector_packet_policy_t kAie2pVectorPacketPolicy = {
    .native_bit_counts = kAie2pVectorPacketBitCounts,
    .native_bit_count_count = IREE_ARRAYSIZE(kAie2pVectorPacketBitCounts),
    .maximum_unpacketized_bit_count = 0,
};

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

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_legalize_vector_splat(
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
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_splat(
      context, op, &kAie2pVectorPacketPolicy, &rewritten));
  if (rewritten) {
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
      context, op, &kAie2pVectorPacketPolicy, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(
        loom_vector_static_shape_rewrite_op(context, op, &rewritten));
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
      context, op, &kAie2pVectorPacketPolicy, &rewritten));
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_load_result(op));
  if (!rewritten && !loom_aie2p_defer_multidimensional_memory_reference(
                        context, result_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
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
      context, op, &kAie2pVectorPacketPolicy, &rewritten));
  if (!rewritten && !has_native_store &&
      !loom_aie2p_defer_multidimensional_memory_reference(context,
                                                          value_type)) {
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
      context, op, &kAie2pVectorPacketPolicy, &packet_result));
  bool rewritten = packet_result == LOOM_VECTOR_PACKET_REDUCE_RESULT_REWRITTEN;
  if (packet_result == LOOM_VECTOR_PACKET_REDUCE_RESULT_CAPTURE_INPUT) {
    IREE_RETURN_IF_ERROR(loom_vector_reduce_captured_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
  } else if (!rewritten) {
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
        .root_kind = LOOM_OP_VECTOR_SPLAT,
        .legalize = loom_aie2p_legalize_vector_splat,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BROADCAST,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SLICE,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
    },
    {
        .root_kind = LOOM_OP_VECTOR_INSERT,
        .legalize = loom_aie2p_legalize_vector_to_scalar,
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
        .legalize = loom_aie2p_legalize_vector_to_scalar,
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
