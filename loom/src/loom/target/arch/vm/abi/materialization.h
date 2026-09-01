// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VM call ABI materialization.
//
// The VM uses fixed physical register prefixes for direct arguments and
// results. This pass gives every such boundary a fresh SSA identity before
// allocation so ordinary values remain freely allocatable between call
// boundaries.

#ifndef LOOM_TARGET_ARCH_VM_ABI_MATERIALIZATION_H_
#define LOOM_TARGET_ARCH_VM_ABI_MATERIALIZATION_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns static pass metadata for vm-materialize-call-abi.
const loom_pass_info_t* loom_vm_materialize_call_abi_pass_info(void);

// Materializes fixed-prefix staging values for all VM functions in |module|.
iree_status_t loom_vm_materialize_call_abi_run(loom_pass_t* pass,
                                               loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_ABI_MATERIALIZATION_H_
