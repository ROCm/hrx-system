// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_H_
#define IREE_VM_BYTECODE_INTERPRETER_H_

#include "iree/vm/bytecode/module_storage.h"

// Starts and synchronously executes one verified Core 0.0 bytecode function.
iree_status_t iree_vm_bytecode_function_start(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome);

// Resumes one suspended verified Core 0.0 bytecode function frame.
iree_status_t iree_vm_bytecode_function_resume(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome);

#endif  // IREE_VM_BYTECODE_INTERPRETER_H_
