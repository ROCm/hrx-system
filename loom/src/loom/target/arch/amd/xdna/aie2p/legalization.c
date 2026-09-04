// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/legalization.h"

#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/transforms/vector/packet_legalization.h"
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
  IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
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

static const loom_target_legalizer_rule_t kAie2pLegalizerRules[] = {
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
