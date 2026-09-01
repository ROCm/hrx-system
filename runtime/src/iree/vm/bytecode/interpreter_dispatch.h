// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_DISPATCH_H_
#define IREE_VM_BYTECODE_INTERPRETER_DISPATCH_H_

#include "iree/vm/bytecode/interpreter_frame.h"

// Dispatches verified Core 0.0 records until the function completes, suspends,
// or fails. The caller owns frame selection and terminal frame cleanup.
iree_status_t iree_vm_bytecode_dispatch(
    iree_vm_bytecode_module_t* module,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_module_execution_t* execution,
    iree_vm_bytecode_execution_state_t* state,
    iree_vm_execution_outcome_t* out_outcome);

#endif  // IREE_VM_BYTECODE_INTERPRETER_DISPATCH_H_
