// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/ireevm/ops/registry.h"

#include "loom/target/arch/ireevm/ops/ops.h"

iree_status_t loom_ireevm_ops_register_dialect(loom_context_t* context) {
  iree_host_size_t vtable_count = 0;
  const loom_op_vtable_t* const* vtables =
      loom_ireevm_dialect_vtables(&vtable_count);
  iree_host_size_t semantic_count = 0;
  const loom_op_semantics_t* semantics =
      loom_ireevm_dialect_op_semantics(&semantic_count);
  IREE_RETURN_IF_ERROR(loom_context_register_dialect(
      context, LOOM_DIALECT_IREEVM, vtables, (uint16_t)vtable_count));
  return loom_context_register_dialect_semantics(
      context, LOOM_DIALECT_IREEVM, semantics, (uint16_t)semantic_count);
}
