// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_VERIFICATION_H_
#define IREE_VM_BYTECODE_VERIFICATION_H_

#include "iree/vm/bytecode/module_storage.h"

// Verifies provider-independent image and declaration structure and derives the
// exact mapped construction plan. Instruction execution semantics are not
// proven by this pass.
iree_status_t iree_vm_bytecode_module_verify_structure(
    iree_const_byte_span_t contents, iree_vm_bytecode_module_plan_t* out_plan);

// Proves that a structurally verified plan is inside the implemented execution
// closure. Functions with more than 32 blocks use one temporary allocation
// from |scratch_allocator|; |plan| is not modified.
iree_status_t iree_vm_bytecode_module_verify_executable(
    const iree_vm_bytecode_module_plan_t* plan,
    iree_allocator_t scratch_allocator);

// Proves that every declared extension page is known to inspection tooling.
// This performs no allocation and does not modify |plan|.
iree_status_t iree_vm_bytecode_module_verify_inspectable(
    const iree_vm_bytecode_module_plan_t* plan);

#endif  // IREE_VM_BYTECODE_VERIFICATION_H_
