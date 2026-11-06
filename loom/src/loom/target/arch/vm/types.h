// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_VM_TYPES_H_
#define LOOM_TARGET_ARCH_VM_TYPES_H_

#include "loom/ir/module.h"
#include "loom/ir/type_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves an inline managed handle, a declared opaque type, or the VM mapping
// for built-in buffer. Returns false when the type has no VM reference ABI.
// Key strings borrow the immutable source module or registered type metadata.
bool loom_vm_type_reference_key(const loom_module_t* module, loom_type_t type,
                                loom_type_reference_key_t* out_key);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_TYPES_H_
