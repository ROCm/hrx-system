// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_H_
#define IREE_VM_BYTECODE_MODULE_H_

#include "iree/base/api.h"
#include "iree/vm/environment.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable bytecode image storage paired with its optional owner.
typedef struct iree_vm_bytecode_module_storage_t {
  // Exact immutable image bytes with an eight-byte-aligned base address.
  iree_const_byte_span_t contents;
  // Optional deallocator for |contents.data|; null denotes borrowed storage.
  iree_allocator_t deallocator;
} iree_vm_bytecode_module_storage_t;

// Verifies that |contents| is a complete Core bytecode image.
//
// The image must be nonempty and eight-byte aligned and is borrowed only for
// the duration of the call. Functions with large control-flow graphs may use
// |scratch_allocator| for temporary block-index storage.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_verify(
    iree_const_byte_span_t contents, iree_allocator_t scratch_allocator);

// Verifies and creates one bytecode module.
//
// |environment| is borrowed while resolving every named ref type and is not
// retained. |module_name| is cloned. Success transfers |storage.deallocator|
// and keeps the immutable image live through the module and every escaped
// image-backed rodata buffer. Failure leaves storage ownership with the caller.
// |out_module| is set null before any other work and remains null on failure.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create(
    iree_vm_environment_t* environment, iree_string_view_t module_name,
    iree_vm_bytecode_module_storage_t storage, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

// Creates one bytecode module from trusted in-process compiler output.
//
// This performs the same bounds-safe image mapping, runtime compatibility,
// type resolution, allocation, and ownership transfer as
// iree_vm_bytecode_module_create but omits semantic and instruction
// verification. The caller places the producer in the host trusted computing
// base and must guarantee that |storage.contents| is a canonical module image.
// Files, caches, network data, and other externally supplied images must use
// iree_vm_bytecode_module_create instead. The ownership and output contracts
// are otherwise identical.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create_trusted(
    iree_vm_environment_t* environment, iree_string_view_t module_name,
    iree_vm_bytecode_module_storage_t storage, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_MODULE_H_
