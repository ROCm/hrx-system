// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_STORAGE_H_
#define IREE_VM_BYTECODE_STORAGE_H_

#include "iree/base/api.h"

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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_STORAGE_H_
