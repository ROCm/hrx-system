// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INSPECTION_H_
#define IREE_VM_BYTECODE_INSPECTION_H_

#include "iree/vm/bytecode/storage.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Verifies and creates one provider-independent inspection module.
//
// Ref types are represented by module-owned reflection-only descriptors, so no
// environment or runtime provider is required. The resulting module supports
// the ordinary generic reflection queries but is not linkable and cannot enter
// a program, process, or invocation. Inspection verifies image and declaration
// structure without claiming that serialized instruction semantics are
// executable. |module_name| is cloned into the resulting module. Success
// transfers |storage.deallocator| and keeps the immutable image live through
// the module. Failure leaves storage ownership with the caller. |out_module| is
// set null before any other work and remains null on failure.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create_for_inspection(
    iree_string_view_t module_name, iree_vm_bytecode_module_storage_t storage,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_INSPECTION_H_
