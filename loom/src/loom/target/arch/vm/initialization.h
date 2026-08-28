// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical VM module initialization pass.

#ifndef LOOM_TARGET_ARCH_VM_INITIALIZATION_H_
#define LOOM_TARGET_ARCH_VM_INITIALIZATION_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns static pass metadata for vm-materialize-initializer.
const loom_pass_info_t* loom_vm_materialize_initializer_pass_info(void);

// Canonicalizes or synthesizes the executable VM initializer function.
iree_status_t loom_vm_materialize_initializer_run(loom_pass_t* pass,
                                                  loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_INITIALIZATION_H_
