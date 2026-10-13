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

// Returns the declared external identity for an opaque managed type, the VM
// mapping for built-in buffer, or NULL when the type has no VM reference ABI.
// The returned key borrows immutable compiler type metadata.
const loom_type_reference_key_t* loom_vm_type_reference_key(
    const loom_module_t* module, loom_type_t type);

// Returns the exact VM scalar signature kind, or zero for a non-scalar or an
// unsupported scalar. Predicates use the ordinary canonical i32 ABI.
uint16_t loom_vm_type_scalar_kind(loom_type_t type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_TYPES_H_
