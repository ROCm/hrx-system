// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Logical VM call ABI signature metadata.

#ifndef LOOM_TARGET_ARCH_VM_ABI_LAYOUT_H_
#define LOOM_TARGET_ARCH_VM_ABI_LAYOUT_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds a canonical abi_layout dictionary containing the logical mapped Low
// function signature. The signature is independent of the physical function
// boundary and remains unchanged as boundary values are materialized.
iree_status_t loom_vm_call_abi_layout_make_attr(
    loom_module_t* module, const loom_type_t* argument_types,
    iree_host_size_t argument_count, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_attribute_t* out_attr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_ABI_LAYOUT_H_
