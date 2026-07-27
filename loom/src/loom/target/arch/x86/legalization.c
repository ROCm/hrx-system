// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/legalization.h"

#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/records/target_records.h"

static bool loom_x86_legalizer_descriptor_set_is_x86(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id == LOOM_X86_TARGET_STABLE_ID;
}

static iree_status_t loom_x86_retain_native_vector_op(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  (void)op;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_x86_legalizer_descriptor_set_is_x86(context->descriptor_set)) {
    return iree_ok_status();
  }
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
  };
  return iree_ok_status();
}

static const loom_target_legalizer_entry_t kX86LegalizerEntries[] = {
    {
        .root_kind = LOOM_OP_VECTOR_DOT2F,
        .legalize = loom_x86_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4I,
        .legalize = loom_x86_retain_native_vector_op,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT8I4,
        .legalize = loom_x86_retain_native_vector_op,
    },
};

const loom_target_legalizer_provider_t
    loom_x86_target_legalizer_provider_storage = {
        .name = IREE_SVL("x86"),
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
        .entries = kX86LegalizerEntries,
        .entry_count = IREE_ARRAYSIZE(kX86LegalizerEntries),
};

const loom_target_legalizer_provider_t* loom_x86_target_legalizer_provider(
    void) {
  return &loom_x86_target_legalizer_provider_storage;
}
