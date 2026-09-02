// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/legalization.h"

#include "loom/ops/vector/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/transforms/vector/to_scalar.h"

static bool loom_spirv_legalizer_descriptor_set_is_spirv(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id ==
             loom_spirv_logical_core_descriptor_set()->target_stable_id;
}

static iree_status_t loom_spirv_legalize_vector_atomic(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_spirv_legalizer_descriptor_set_is_spirv(context->descriptor_set)) {
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_atomic_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

static const loom_target_legalizer_entry_t kSpirvLegalizerEntries[] = {
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_REDUCE,
        .legalize = loom_spirv_legalize_vector_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK,
        .legalize = loom_spirv_legalize_vector_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_RMW,
        .legalize = loom_spirv_legalize_vector_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_RMW_MASK,
        .legalize = loom_spirv_legalize_vector_atomic,
    },
    {
        .root_kind = LOOM_OP_VECTOR_ATOMIC_CMPXCHG,
        .legalize = loom_spirv_legalize_vector_atomic,
    },
};

const loom_target_legalizer_provider_t
    loom_spirv_target_legalizer_provider_storage = {
        .name = IREE_SVL("spirv"),
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
        .entries = kSpirvLegalizerEntries,
        .entry_count = IREE_ARRAYSIZE(kSpirvLegalizerEntries),
};

const loom_target_legalizer_provider_t* loom_spirv_target_legalizer_provider(
    void) {
  return &loom_spirv_target_legalizer_provider_storage;
}
