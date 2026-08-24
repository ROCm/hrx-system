// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BUFFER_PROVIDER_H_
#define IREE_VM_BUFFER_PROVIDER_H_

#include "iree/vm/buffer.h"

// Private representation bits sharing the public access word.
enum iree_vm_buffer_flag_bits_e {
  // The buffer is a proper view retaining a flattened root.
  IREE_VM_BUFFER_FLAG_VIEW = 1u << 2,
  // The root wraps storage released by a callback.
  IREE_VM_BUFFER_FLAG_EXTERNAL = 1u << 3,
  // The complete buffer object is embedded in callback-owned storage.
  IREE_VM_BUFFER_FLAG_EMBEDDED = 1u << 4,
};
typedef uint32_t iree_vm_buffer_flags_t;

// Public access bits stored in the shared flags word.
enum {
  IREE_VM_BUFFER_ACCESS_MASK =
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
};

// Uniform buffer object. The fields through |root| form the access-hot prefix;
// the allocator and release callback are touched only during construction and
// final destruction.
struct iree_vm_buffer_t {
  // Required offset-zero VM ref-count prefix.
  iree_vm_ref_object_t ref_object;
  // Public access bits and private representation bits.
  iree_vm_buffer_flags_t flags;
  // Immutable logical byte length.
  iree_host_size_t length;
  // Direct root or cached-view byte start.
  uint8_t* data;
  // Retained flattened root, or null when this object is a root.
  iree_vm_buffer_t* root;
  // Allocator owning this complete buffer object.
  iree_allocator_t host_allocator;
  // External root storage release callback, or null for heaps and views.
  iree_vm_buffer_release_callback_t release_callback;
};

#define IREE_VM_BUFFER_HOT_SIZE \
  (8 + sizeof(iree_host_size_t) + 2 * sizeof(void*))

static_assert(offsetof(iree_vm_buffer_t, ref_object) == 0,
              "VM buffers require an offset-zero ref object");
static_assert(offsetof(iree_vm_buffer_t, flags) == 4,
              "VM buffer flags must follow the ref object");
static_assert(offsetof(iree_vm_buffer_t, length) == 8,
              "VM buffer length must begin at byte eight");
static_assert(offsetof(iree_vm_buffer_t, host_allocator) ==
                  IREE_VM_BUFFER_HOT_SIZE,
              "VM buffer hot prefix layout changed");
static_assert(sizeof(void*) != 8 || IREE_VM_BUFFER_HOT_SIZE == 32,
              "64-bit VM buffer hot prefix must remain 32 bytes");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_buffer_t) == 64,
              "64-bit VM buffer object must remain 64 bytes");

#undef IREE_VM_BUFFER_HOT_SIZE

// Returns the access bits carried directly by |buffer|. Views may still lose
// effective access when their flattened root closes.
static inline iree_vm_buffer_access_flags_t iree_vm_buffer_local_access(
    const iree_vm_buffer_t* buffer) {
  return buffer->flags & IREE_VM_BUFFER_ACCESS_MASK;
}

// Returns access currently granted by |buffer| and its flattened root.
static inline iree_vm_buffer_access_flags_t iree_vm_buffer_effective_access(
    const iree_vm_buffer_t* buffer) {
  const iree_vm_buffer_access_flags_t access =
      iree_vm_buffer_local_access(buffer);
  if (access == IREE_VM_BUFFER_ACCESS_FLAG_NONE) return access;
  if ((buffer->flags & IREE_VM_BUFFER_FLAG_VIEW) &&
      iree_vm_buffer_local_access(buffer->root) ==
          IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
    return IREE_VM_BUFFER_ACCESS_FLAG_NONE;
  }
  return access;
}

// Returns one checked borrowed range. All liveness, rights, and range checks
// complete before |out_span| is populated or a host pointer is formed.
static inline iree_status_t iree_vm_buffer_map_range(
    const iree_vm_buffer_t* buffer,
    iree_vm_buffer_access_flags_t required_access, iree_host_size_t offset,
    iree_host_size_t length, iree_byte_span_t* out_span) {
  const iree_vm_buffer_access_flags_t local_access =
      iree_vm_buffer_local_access(buffer);
  if ((local_access & required_access) != required_access) {
    if (local_access == IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "buffer is closed");
    }
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "buffer does not permit the requested access");
  }
  if (offset > buffer->length || length > buffer->length - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer range offset %" PRIhsz " length %" PRIhsz
                            " exceeds buffer length %" PRIhsz,
                            offset, length, buffer->length);
  }
  if ((buffer->flags & IREE_VM_BUFFER_FLAG_VIEW) &&
      iree_vm_buffer_local_access(buffer->root) ==
          IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer root is closed");
  }

  *out_span = length == 0 ? iree_byte_span_empty()
                          : iree_make_byte_span(buffer->data + offset, length);
  return iree_ok_status();
}

// Initializes one allocation-free read-only root embedded in storage owned by
// |release_callback|. The callback is required and may invalidate
// |out_buffer|, so final release performs no access after calling it.
static inline void iree_vm_buffer_initialize_embedded_read_only(
    iree_const_byte_span_t storage,
    iree_vm_buffer_release_callback_t release_callback,
    iree_vm_buffer_t* out_buffer) {
  iree_vm_ref_object_initialize(&out_buffer->ref_object);
  out_buffer->flags = IREE_VM_BUFFER_ACCESS_FLAG_READ |
                      IREE_VM_BUFFER_FLAG_EXTERNAL |
                      IREE_VM_BUFFER_FLAG_EMBEDDED;
  out_buffer->length = storage.data_length;
  out_buffer->data = (uint8_t*)storage.data;
  out_buffer->root = NULL;
  out_buffer->host_allocator = iree_allocator_null();
  out_buffer->release_callback = release_callback;
}

// Returns the process-static provider table for the core "vm" family.
const iree_vm_ref_type_table_t* iree_vm_buffer_provider_table(void);

#endif  // IREE_VM_BUFFER_PROVIDER_H_
