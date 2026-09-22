// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/legalization.h"

#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/legalization_compare.h"
#include "loom/target/arch/amd/xdna/aie2p/legalization_table.h"
#include "loom/transforms/vector/packet_legalization.h"
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

static iree_status_t loom_aie2p_legalize_table_lookup(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  if (!loom_aie2p_legalizer_descriptor_set_is_core(context->descriptor_set)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
    };
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_table_lookup_rewrite(context, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
    return iree_ok_status();
  }
  return loom_aie2p_legalize_vector_to_scalar(entry, context, op, out_result);
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
  if (!rewritten) {
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

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_store(
      context, op, &kAie2pVectorPacketPolicy, &rewritten));
  if (!rewritten) {
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

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_reduce(
      context, op, &kAie2pVectorPacketPolicy, &rewritten));
  if (!rewritten) {
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
