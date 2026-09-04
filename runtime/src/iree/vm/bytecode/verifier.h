// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_VERIFIER_H_
#define IREE_VM_BYTECODE_VERIFIER_H_

#include "iree/vm/bytecode/layout.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Verifies provider-independent image and declaration structure and produces
// the exact mapped construction plan. This does not validate instruction
// records. |out_plan| is untouched on failure.
iree_status_t iree_vm_bytecode_verify_module_structure(
    iree_const_byte_span_t contents, iree_vm_bytecode_module_plan_t* out_plan);

// Verifies provider-independent declaration semantics in an already safely
// mapped module plan. This does not validate instruction records.
iree_status_t iree_vm_bytecode_verify_module_layout(
    const iree_vm_bytecode_module_plan_t* plan);

// Verifies every instruction record in a structurally verified module plan.
// |block_offsets| provides transient storage for
// |plan->layout.functions.maximum_block_count| uint32_t entries and may be null
// when the count is zero.
iree_status_t iree_vm_bytecode_verify_module_instructions(
    const iree_vm_bytecode_module_plan_t* plan, uint32_t* block_offsets);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_VERIFIER_H_
