// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VM function contract materialization.

#ifndef LOOM_TARGET_ARCH_VM_CONTRACTS_MATERIALIZATION_H_
#define LOOM_TARGET_ARCH_VM_CONTRACTS_MATERIALIZATION_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns static pass metadata for vm-materialize-function-contracts.
const loom_pass_info_t* loom_vm_materialize_function_contracts_pass_info(void);

// Materializes function predicates as executable entry checks.
iree_status_t loom_vm_materialize_function_contracts_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_CONTRACTS_MATERIALIZATION_H_
