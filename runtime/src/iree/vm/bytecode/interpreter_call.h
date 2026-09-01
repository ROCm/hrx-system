// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_CALL_H_
#define IREE_VM_BYTECODE_INTERPRETER_CALL_H_

#include "iree/vm/bytecode/interpreter_frame.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/selectors.h"

// Starts one verified direct call using the caller's canonical packet prefix.
// On suspension, |state| retains the caller-owned argument regions that must be
// released after the child completes. On failure, the invocation core owns the
// terminal frame unwind.
iree_status_t iree_vm_bytecode_call_start_direct(
    iree_vm_bytecode_module_t* module,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    const iree_vm_isa_control_call_record_t* record,
    iree_vm_execution_outcome_t* out_outcome);

// Starts one verified indirect call after validating its dynamic target.
// Invalid targets fail before argument staging or provider entry. Suspension
// and failure ownership otherwise match direct calls.
iree_status_t iree_vm_bytecode_call_start_indirect(
    iree_vm_bytecode_module_t* module,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    const iree_vm_isa_control_call_indirect_record_t* record,
    iree_vm_execution_outcome_t* out_outcome);

// Releases caller-owned packet arguments after a suspended child completes.
void iree_vm_bytecode_call_cleanup_completed(
    iree_vm_bytecode_execution_state_t* state);

#endif  // IREE_VM_BYTECODE_INTERPRETER_CALL_H_
