// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_BUFFER_H_
#define IREE_VM_BYTECODE_INTERPRETER_BUFFER_H_

#include <stdint.h>

#include "iree/vm/buffer_provider.h"

// Returns whether the complete one-slab buffer allocation, including the
// aligned-allocation prefix and padding, is representable on this host. This
// makes the subsequent construction call incapable of reporting a size
// overflow after the bytecode operation has classified it as exhaustion.
static inline bool iree_vm_bytecode_buffer_allocation_is_representable(
    uint64_t length) {
  if (length > IREE_HOST_SIZE_MAX) return false;
  iree_host_size_t allocation_size = sizeof(iree_vm_buffer_t);
  return iree_host_size_checked_add(allocation_size, (iree_host_size_t)length,
                                    &allocation_size) &&
         iree_host_size_checked_add(sizeof(uintptr_t), allocation_size,
                                    &allocation_size) &&
         iree_host_size_checked_add(allocation_size, iree_max_align_t,
                                    &allocation_size);
}

// Borrows one required ref after checking the exact module-resolved vm.buffer
// descriptor. The object and descriptor are immutable for the ref lifetime.
static inline iree_status_t iree_vm_bytecode_buffer_check_deref(
    iree_vm_ref_t ref, iree_vm_ref_type_t buffer_type,
    iree_vm_buffer_t** out_buffer) {
  if (!ref.object) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "required vm.buffer ref is null");
  }
  if (iree_vm_ref_type(ref) != buffer_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ref type is not vm.buffer");
  }
  *out_buffer = (iree_vm_buffer_t*)ref.object;
  return iree_ok_status();
}

// Resolves and maps one checked scaled lane range. Arithmetic remains u64
// until host representability is proven. The mapped span is populated only
// after arithmetic, bounds, liveness, and rights checks all succeed.
static inline iree_status_t iree_vm_bytecode_buffer_map_lanes(
    iree_vm_buffer_t* buffer, iree_vm_buffer_access_flags_t required_access,
    uint64_t base, uint64_t index, uint8_t scale, uint8_t access_length,
    iree_byte_span_t* out_span) {
  uint64_t offset = 0;
  if (!iree_checked_mul_u64(index, scale, &offset) ||
      !iree_checked_add_u64(base, offset, &offset) ||
      offset > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer address arithmetic overflow");
  }
  return iree_vm_buffer_map_range(buffer, required_access,
                                  (iree_host_size_t)offset, access_length,
                                  out_span);
}

#endif  // IREE_VM_BYTECODE_INTERPRETER_BUFFER_H_
