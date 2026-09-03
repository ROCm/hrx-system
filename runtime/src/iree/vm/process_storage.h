// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_PROCESS_STORAGE_H_
#define IREE_VM_PROCESS_STORAGE_H_

#include "iree/vm/process.h"
#include "iree/vm/program_storage.h"

// Private process representation shared with invocation execution. No
// declaration in this file is part of the public ABI.

// One allocation containing the process header and all opaque module slices.
struct iree_vm_process_t {
  // Intrusive owner count published only after successful state sealing.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this process slab.
  iree_allocator_t host_allocator;
  // Retained immutable linked program.
  iree_vm_program_t* program;
};

// Returns the max-aligned first byte of the opaque process-state tail.
static inline uint8_t* iree_vm_process_storage_base(
    iree_vm_process_t* process) {
  return (uint8_t*)process + iree_sizeof_struct(*process);
}

// Returns one linked module's stable opaque state span. Zero-state modules
// receive a canonical empty span even though lifecycle callbacks still run.
static inline iree_byte_span_t iree_vm_process_module_state(
    iree_vm_process_t* process, const iree_vm_linked_module_t* linked_module) {
  const iree_host_size_t storage_size =
      linked_module->module->descriptor->process_storage_size;
  return storage_size == 0
             ? iree_byte_span_empty()
             : iree_make_byte_span(iree_vm_process_storage_base(process) +
                                       linked_module->process_storage_offset,
                                   storage_size);
}

#endif  // IREE_VM_PROCESS_STORAGE_H_
