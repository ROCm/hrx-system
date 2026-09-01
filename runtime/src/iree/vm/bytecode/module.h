// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_H_
#define IREE_VM_BYTECODE_MODULE_H_

#include "iree/base/api.h"
#include "iree/vm/bytecode/storage.h"
#include "iree/vm/environment.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Verifies that |contents| is a structurally valid executable bytecode image.
//
// This validates instruction semantics without resolving the ref types named by
// the image. |contents| must be nonempty and eight-byte aligned and is borrowed
// only for the duration of the call. Functions with large control-flow graphs
// may use |scratch_allocator| for temporary verification storage.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_verify(
    iree_const_byte_span_t contents, iree_allocator_t scratch_allocator);

// Verifies and creates one executable bytecode module.
//
// |environment| supplies the canonical ref types named by the image and is
// borrowed only during this call. |module_name| is cloned into the resulting
// module. Success transfers |storage.deallocator| and keeps the immutable image
// live through the module and every escaped image-backed rodata buffer. Failure
// leaves storage ownership with the caller. |out_module| is set null before any
// other work and remains null on failure.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create(
    iree_vm_environment_t* environment, iree_string_view_t module_name,
    iree_vm_bytecode_module_storage_t storage, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_MODULE_H_
