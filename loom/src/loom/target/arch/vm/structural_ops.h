// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VM encodings for descriptor-less target-low operations.

#ifndef LOOM_TARGET_ARCH_VM_STRUCTURAL_OPS_H_
#define LOOM_TARGET_ARCH_VM_STRUCTURAL_OPS_H_

#include "iree/base/api.h"

typedef struct loom_op_t loom_op_t;

#ifdef __cplusplus
extern "C" {
#endif

// Encoding family selected for one descriptor-less Low operation.
typedef uint8_t loom_vm_structural_op_kind_t;
enum loom_vm_structural_op_kind_e {
  // The operation has no structural Core VM encoding.
  LOOM_VM_STRUCTURAL_OP_KIND_NONE = 0,
  // Function return.
  LOOM_VM_STRUCTURAL_OP_KIND_RETURN = 1,
  // Ownership-preserving register transfer.
  LOOM_VM_STRUCTURAL_OP_KIND_RETAIN_TRANSFER = 2,
  // Ownership-moving register transfer.
  LOOM_VM_STRUCTURAL_OP_KIND_MOVE_TRANSFER = 3,
  // Compile-time storage reservation or view.
  LOOM_VM_STRUCTURAL_OP_KIND_STORAGE = 4,
  // Spill or reload between registers and frame-local storage.
  LOOM_VM_STRUCTURAL_OP_KIND_LOCAL_TRANSFER = 5,
  // Direct or indirect function call.
  LOOM_VM_STRUCTURAL_OP_KIND_CALL = 6,
  // Function reference construction, comparison, or query.
  LOOM_VM_STRUCTURAL_OP_KIND_FUNCTION_VALUE = 7,
  // Unconditional branch or yield.
  LOOM_VM_STRUCTURAL_OP_KIND_BRANCH = 8,
  // Conditional branch.
  LOOM_VM_STRUCTURAL_OP_KIND_CONDITIONAL_BRANCH = 9,
  // Multi-way switch.
  LOOM_VM_STRUCTURAL_OP_KIND_SWITCH = 10,
};

// Classifies |op| into its Core VM structural encoding family. Descriptor-
// backed low.op and low.const operations return NONE and are encoded from their
// descriptor rows instead.
loom_vm_structural_op_kind_t loom_vm_structural_op_classify(
    const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_STRUCTURAL_OPS_H_
