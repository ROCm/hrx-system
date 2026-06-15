// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/llvmir/registry.h"

#include "loom/ops/llvmir/ops.h"

iree_status_t loom_llvmir_ops_register_dialect(loom_context_t* context) {
  iree_host_size_t vtable_count = 0;
  const loom_op_vtable_t* const* vtables =
      loom_llvmir_dialect_vtables(&vtable_count);
  iree_host_size_t semantic_count = 0;
  const loom_op_semantics_t* semantics =
      loom_llvmir_dialect_op_semantics(&semantic_count);
  loom_dialect_vtables_t* dialect =
      &context->op_vtables.dialects[LOOM_DIALECT_LLVMIR];
  if (dialect->entries != NULL) {
    if (dialect->entries != vtables || dialect->op_count != vtable_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "LLVMIR dialect is already registered with different vtables");
    }
    if (dialect->semantics != NULL) {
      if (dialect->semantics != semantics || semantic_count != vtable_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "LLVMIR dialect is already registered with different semantics");
      }
      return iree_ok_status();
    }
    return loom_context_register_dialect_semantics(
        context, LOOM_DIALECT_LLVMIR, semantics, (uint16_t)semantic_count);
  }
  IREE_RETURN_IF_ERROR(loom_context_register_dialect(
      context, LOOM_DIALECT_LLVMIR, vtables, (uint16_t)vtable_count));
  return loom_context_register_dialect_semantics(
      context, LOOM_DIALECT_LLVMIR, semantics, (uint16_t)semantic_count);
}
