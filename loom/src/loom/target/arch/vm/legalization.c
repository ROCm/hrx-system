// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/legalization.h"

#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/transforms/vector/to_scalar.h"

static iree_status_t loom_vm_legalize_vector_to_scalar(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->descriptor_set != loom_vm_core_descriptor_set()) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

#define LOOM_VM_VECTOR_TO_SCALAR_ENTRY(op_kind)      \
  {                                                  \
      .root_kind = (op_kind),                        \
      .legalize = loom_vm_legalize_vector_to_scalar, \
  }

static const loom_target_legalizer_entry_t kVmLegalizerEntries[] = {
    LOOM_VM_VECTOR_TO_SCALAR_ENTRY(LOOM_OP_SCF_SELECT),
    LOOM_VM_VECTOR_TO_SCALAR_ENTRY(LOOM_OP_VECTOR_TRANSPOSE),
    LOOM_VM_VECTOR_TO_SCALAR_ENTRY(LOOM_OP_VECTOR_SHUFFLE),
    LOOM_VM_VECTOR_TO_SCALAR_ENTRY(LOOM_OP_VECTOR_SELECT),
    LOOM_VM_VECTOR_TO_SCALAR_ENTRY(LOOM_OP_VECTOR_BITCAST),
#define LOOM_VECTOR_SCALARIZATION_ROW(vector_op, scalar_op, flags, \
                                      seed_operand_index)          \
  LOOM_VM_VECTOR_TO_SCALAR_ENTRY(vector_op),
#include "loom/ops/vector/scalarization_rows.inl"
#undef LOOM_VECTOR_SCALARIZATION_ROW
};

#undef LOOM_VM_VECTOR_TO_SCALAR_ENTRY

const loom_target_legalizer_provider_t loom_vm_legalizer_provider = {
    .name = IREE_SVL("vm"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
    .entries = kVmLegalizerEntries,
    .entry_count = IREE_ARRAYSIZE(kVmLegalizerEntries),
};
