// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_IMAGE_H_
#define IREE_VM_BYTECODE_IMAGE_H_

#include "iree/base/internal/atomics.h"
#include "iree/vm/buffer_provider.h"
#include "iree/vm/bytecode/layout.h"
#include "iree/vm/bytecode/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Private immutable image slab implementing one generic bytecode module. The
// generic module and image/slab owner counts are intentionally independent:
// escaped rodata roots may keep this allocation live after final module
// release.
typedef struct iree_vm_bytecode_image_t {
  // Owners held by the module lifetime and each embedded rodata root.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this complete aligned slab.
  iree_allocator_t host_allocator;
  // Transferred immutable image storage.
  iree_vm_bytecode_module_storage_t storage;
  // Generic immutable module descriptor.
  iree_vm_module_descriptor_t descriptor;
  // Complete verified mapped image layout.
  iree_vm_bytecode_module_layout_t layout;
  // Exact opaque per-process storage layout.
  iree_vm_bytecode_process_layout_t process_layout;
  // Canonical ref handles in module-local ordinal order.
  iree_vm_ref_type_t* resolved_ref_types;
  // Canonical core vm.buffer descriptor used by interpreter operations.
  iree_vm_ref_type_t buffer_type;
  // Embedded module-owned read-only rodata roots.
  iree_vm_buffer_t* rodata_roots;
  // Coallocated generic module provider retaining this image.
  iree_vm_module_t base_module;
} iree_vm_bytecode_image_t;

static_assert(offsetof(iree_vm_bytecode_image_t, ref_count) == 0,
              "bytecode image ref counts must be at offset zero");

// Returns the private image containing a generic bytecode module.
static inline iree_vm_bytecode_image_t* iree_vm_bytecode_image_from_module(
    iree_vm_module_t* module) {
  return (iree_vm_bytecode_image_t*)((uint8_t*)module -
                                     offsetof(iree_vm_bytecode_image_t,
                                              base_module));
}

// Returns the private image containing a const generic bytecode module.
static inline const iree_vm_bytecode_image_t*
iree_vm_bytecode_image_from_module_const(const iree_vm_module_t* module) {
  return (const iree_vm_bytecode_image_t*)((const uint8_t*)module -
                                           offsetof(iree_vm_bytecode_image_t,
                                                    base_module));
}

// Retains one private immutable image owner. A null image is ignored.
void iree_vm_bytecode_image_retain(iree_vm_bytecode_image_t* image);

// Releases one private immutable image owner. Final release deallocates the
// transferred image storage and complete aligned slab.
void iree_vm_bytecode_image_release(iree_vm_bytecode_image_t* image);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_IMAGE_H_
