// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/legalization.h"

#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/transforms/scalar/target_legalization.h"
#include "loom/transforms/vector/to_scalar.h"

static iree_status_t loom_vm_legalize_narrow_integer_binary(
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
  return loom_scalar_target_legalize_narrow_integer_binary_reference(
      context, op, out_result);
}

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

static iree_status_t loom_vm_legalize_dynamic_vector_insert(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->descriptor_set != loom_vm_core_descriptor_set() ||
      loom_vector_insert_indices(op).count == 0) {
    return iree_ok_status();
  }
  return loom_vm_legalize_vector_to_scalar(entry, context, op, out_result);
}

#define LOOM_VM_VECTOR_TO_SCALAR_ENTRY(op_kind)      \
  {                                                  \
      .root_kind = (op_kind),                        \
      .legalize = loom_vm_legalize_vector_to_scalar, \
  }

#define LOOM_VM_NARROW_INTEGER_ENTRY(op_kind)             \
  {                                                       \
      .root_kind = (op_kind),                             \
      .legalize = loom_vm_legalize_narrow_integer_binary, \
  }

static const loom_target_legalizer_entry_t kVmLegalizerEntries[] = {
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_ADDI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_SUBI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_MULI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_MINSI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_MAXSI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_MINUI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_MAXUI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_ANDI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_ORI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_XORI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_SHLI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_SHRSI),
    LOOM_VM_NARROW_INTEGER_ENTRY(LOOM_OP_SCALAR_SHRUI),
    LOOM_VM_VECTOR_TO_SCALAR_ENTRY(LOOM_OP_SCF_SELECT),
    {
        .root_kind = LOOM_OP_VECTOR_INSERT,
        .legalize = loom_vm_legalize_dynamic_vector_insert,
    },
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

#undef LOOM_VM_NARROW_INTEGER_ENTRY
#undef LOOM_VM_VECTOR_TO_SCALAR_ENTRY

const loom_target_legalizer_provider_t loom_vm_legalizer_provider = {
    .name = IREE_SVL("vm"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
    .entries = kVmLegalizerEntries,
    .entry_count = IREE_ARRAYSIZE(kVmLegalizerEntries),
};
