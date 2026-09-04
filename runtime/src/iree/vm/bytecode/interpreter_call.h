// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_CALL_H_
#define IREE_VM_BYTECODE_INTERPRETER_CALL_H_

#include "iree/vm/bytecode/interpreter_frame.h"
#include "iree/vm/bytecode/wire/core.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Publishes one verified direct child call after validating dynamic arguments.
// Failure occurs before ref mutation or provider entry. Success parks |state|
// until the iterative invocation driver completes the requested child.
iree_status_t iree_vm_bytecode_call_direct(
    const iree_vm_bytecode_image_t* image,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    const iree_vm_bytecode_control_call_t* record,
    iree_vm_execution_outcome_t* out_outcome);

// Publishes one verified indirect child call after validating the dynamic
// target and arguments with the encoded callable contract.
iree_status_t iree_vm_bytecode_call_indirect(
    const iree_vm_bytecode_image_t* image,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    const iree_vm_bytecode_control_call_indirect_t* record,
    iree_vm_execution_outcome_t* out_outcome);

// Retires caller-owned ref state after a requested child completes.
void iree_vm_bytecode_call_cleanup_completed(
    iree_vm_bytecode_execution_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_INTERPRETER_CALL_H_
