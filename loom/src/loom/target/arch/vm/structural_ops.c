// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/structural_ops.h"

#include "loom/ops/low/ops.h"

loom_vm_structural_op_kind_t loom_vm_structural_op_classify(
    const loom_op_t* op) {
  if (loom_low_return_isa(op)) {
    return LOOM_VM_STRUCTURAL_OP_KIND_RETURN;
  }
  if (loom_low_copy_isa(op)) {
    return LOOM_VM_STRUCTURAL_OP_KIND_RETAIN_TRANSFER;
  }
  if (loom_low_move_isa(op) || loom_low_slice_isa(op) ||
      loom_low_concat_isa(op)) {
    return LOOM_VM_STRUCTURAL_OP_KIND_MOVE_TRANSFER;
  }
  if (loom_low_storage_reserve_isa(op) || loom_low_storage_view_isa(op)) {
    return LOOM_VM_STRUCTURAL_OP_KIND_STORAGE;
  }
  if (loom_low_spill_isa(op) || loom_low_reload_isa(op)) {
    return LOOM_VM_STRUCTURAL_OP_KIND_LOCAL_TRANSFER;
  }
  if (loom_low_func_call_isa(op) || loom_low_func_call_indirect_isa(op)) {
    return LOOM_VM_STRUCTURAL_OP_KIND_CALL;
  }
  if (loom_low_func_null_isa(op) || loom_low_func_compare_null_isa(op) ||
      loom_low_func_address_isa(op) || loom_low_func_import_resolved_isa(op)) {
    return LOOM_VM_STRUCTURAL_OP_KIND_FUNCTION_VALUE;
  }
  if (loom_low_br_isa(op)) {
    return LOOM_VM_STRUCTURAL_OP_KIND_BRANCH;
  }
  if (loom_low_cond_br_isa(op)) {
    return LOOM_VM_STRUCTURAL_OP_KIND_CONDITIONAL_BRANCH;
  }
  if (loom_low_switch_isa(op)) {
    return LOOM_VM_STRUCTURAL_OP_KIND_SWITCH;
  }
  return LOOM_VM_STRUCTURAL_OP_KIND_NONE;
}
