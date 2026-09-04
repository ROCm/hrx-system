// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_H_
#define IREE_VM_BYTECODE_INTERPRETER_H_

#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Starts one verified bytecode function from its first instruction record.
iree_status_t iree_vm_bytecode_interpreter_start(
    iree_vm_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome);

// Resumes one verified bytecode function from its durable frame state.
iree_status_t iree_vm_bytecode_interpreter_resume(
    iree_vm_module_t* module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_INTERPRETER_H_
