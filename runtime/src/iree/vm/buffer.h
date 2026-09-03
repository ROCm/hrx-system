// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BUFFER_H_
#define IREE_VM_BUFFER_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/variant.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_buffer_t iree_vm_buffer_t;

//===----------------------------------------------------------------------===//
// Core Reference Types
//===----------------------------------------------------------------------===//

// Append-only core "vm" ref-type ordinals.
enum iree_vm_ref_type_ordinal_e {
  // CPU-addressable byte buffer type.
  IREE_VM_REF_TYPE_BUFFER = 0,
  // Number of core ref types known by this API version.
  IREE_VM_REF_TYPE_COUNT = 1,
};
typedef uint32_t iree_vm_ref_type_ordinal_t;

// Consumer-owned resolved prefix of the core "vm" ref-type family. Fields are
// append-only and remain in exact ordinal order. Values borrow provider-owned
// descriptors whose host-managed provider scope must remain live.
typedef struct iree_vm_ref_types_t {
  // Canonical core vm.buffer descriptor at ordinal zero.
  iree_vm_ref_type_t buffer;
} iree_vm_ref_types_t;

static_assert(offsetof(iree_vm_ref_types_t, buffer) ==
                  IREE_VM_REF_TYPE_BUFFER * sizeof(iree_vm_ref_type_t),
              "core buffer type must remain at ordinal zero");
static_assert(sizeof(iree_vm_ref_types_t) ==
                  IREE_VM_REF_TYPE_COUNT * sizeof(iree_vm_ref_type_t),
              "core ref-type fields must remain dense");

// Resolves the core "vm" prefix known by this consumer. |table| must have been
// returned by successful environment registration or lookup. Newer providers
// may append fields. Failure leaves |out_types| untouched.
static inline iree_status_t iree_vm_ref_types_resolve(
    const iree_vm_ref_type_table_t* table, iree_vm_ref_types_t* out_types) {
  if (!table || !out_types) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table and out_types are required");
  }
  if (!iree_string_view_equal(table->namespace_name, IREE_SV("vm")) ||
      table->types.count < IREE_VM_REF_TYPE_COUNT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "core ref-type prefix is unavailable");
  }
  iree_vm_ref_types_t types = {
      iree_vm_ref_type_storage_at(table->types, IREE_VM_REF_TYPE_BUFFER),
  };
  if (!iree_string_view_equal(types.buffer->type_name, IREE_SV("buffer"))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "core ref-type ordinal mismatch");
  }
  *out_types = types;
  return iree_ok_status();
}

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_vm_buffer, iree_vm_ref_types_t, buffer,
                             iree_vm_buffer_t)

//===----------------------------------------------------------------------===//
// Core Byte Buffers
//===----------------------------------------------------------------------===//

// Allowed byte access on an open VM buffer.
enum iree_vm_buffer_access_flag_bits_e {
  // No byte access. Invalid when constructing a buffer.
  IREE_VM_BUFFER_ACCESS_FLAG_NONE = 0u,
  // Bytes may be read.
  IREE_VM_BUFFER_ACCESS_FLAG_READ = 1u << 0,
  // Bytes may be written.
  IREE_VM_BUFFER_ACCESS_FLAG_WRITE = 1u << 1,
};
typedef uint32_t iree_vm_buffer_access_flags_t;

// Releases the exact external storage transferred to a successful wrap. The
// callback cannot fail and may run on the thread releasing the final owner.
typedef void(IREE_API_PTR* iree_vm_buffer_release_fn_t)(
    void* user_data, iree_byte_span_t storage);

// Optional no-fail external-storage release operation.
typedef struct iree_vm_buffer_release_callback_t {
  // Release function, or null when the storage owner outlives the buffer.
  iree_vm_buffer_release_fn_t function;
  // Unowned callback data transferred with |function| on successful wrap.
  void* user_data;
} iree_vm_buffer_release_callback_t;

// Returns a callback requiring no external-storage cleanup.
static inline iree_vm_buffer_release_callback_t
iree_vm_buffer_release_callback_null(void) {
  iree_vm_buffer_release_callback_t callback = {NULL, NULL};
  return callback;
}

// Creates one zero-initialized READ|WRITE buffer. A zero |minimum_alignment|
// selects the runtime's natural maximum alignment. A nonzero alignment must be
// a representable power of two. The result is a distinct nonnull object even
// when |length| is zero. |out_buffer| is set null before construction and
// remains null on failure.
IREE_API_EXPORT iree_status_t iree_vm_buffer_create(
    iree_host_size_t length, iree_host_size_t minimum_alignment,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_buffer);

// Clones host bytes into a new buffer with exactly |access|. |access| must
// contain READ, WRITE, or both. Nonempty source storage requires a nonnull data
// pointer. |out_buffer| is set null before construction and remains null on
// failure.
IREE_API_EXPORT iree_status_t iree_vm_buffer_clone(
    iree_vm_buffer_access_flags_t access, iree_const_byte_span_t source_bytes,
    iree_host_size_t minimum_alignment, iree_allocator_t host_allocator,
    iree_vm_buffer_t** out_buffer);

// Wraps externally managed CPU-addressable bytes without copying. Success
// transfers |release_callback| and the obligation to release exactly |storage|.
// Failure leaves the callback obligation with the caller. A zero-length buffer
// may have a null storage pointer. |out_buffer| is set null before construction
// and remains null on failure.
IREE_API_EXPORT iree_status_t iree_vm_buffer_wrap(
    iree_vm_buffer_access_flags_t access, iree_byte_span_t storage,
    iree_vm_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_buffer);

// Materializes an exact range with equal or narrower access. A complete
// same-access range retains and returns |source_buffer|. Every proper view
// caches its direct byte start and retains the flattened root. |out_buffer| is
// set null before construction and remains null on failure.
IREE_API_EXPORT iree_status_t iree_vm_buffer_subspan(
    iree_vm_buffer_t* source_buffer, iree_host_size_t source_offset,
    iree_host_size_t length, iree_vm_buffer_access_flags_t access,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_buffer);

// Adds one ordinary owner. A null buffer is ignored.
IREE_API_EXPORT void iree_vm_buffer_retain(iree_vm_buffer_t* buffer);

// Releases one ordinary owner. A null buffer is ignored.
IREE_API_EXPORT void iree_vm_buffer_release(iree_vm_buffer_t* buffer);

// Returns currently effective access, or NONE after the root is closed. This
// is an instantaneous query rather than a byte-access lease.
IREE_API_EXPORT iree_vm_buffer_access_flags_t
iree_vm_buffer_access(const iree_vm_buffer_t* buffer);

// Returns the immutable logical byte length, including after root closure.
IREE_API_EXPORT iree_host_size_t
iree_vm_buffer_length(const iree_vm_buffer_t* buffer);

// Returns an exact borrowed readable range. The operation requires READ
// access, performs subtraction-form bounds checking, and leaves |out_span|
// untouched on failure. It creates no native mapping or unmap obligation.
IREE_API_EXPORT iree_status_t iree_vm_buffer_map_read(
    const iree_vm_buffer_t* buffer, iree_host_size_t offset,
    iree_host_size_t length, iree_const_byte_span_t* out_span);

// Returns an exact borrowed writable range. The operation requires WRITE
// access, performs subtraction-form bounds checking, and leaves |out_span|
// untouched on failure. It creates no native mapping or unmap obligation.
IREE_API_EXPORT iree_status_t
iree_vm_buffer_map_write(iree_vm_buffer_t* buffer, iree_host_size_t offset,
                         iree_host_size_t length, iree_byte_span_t* out_span);

// Returns the full writable byte pointer at this instant, or null when WRITE
// access is absent or the root is closed. A valid open empty buffer may also
// return null. The returned pointer is not a revocable lease; its later
// lifetime and synchronization are the trusted C caller's concern.
IREE_API_EXPORT void* iree_vm_buffer_data(iree_vm_buffer_t* buffer);

// Returns the full readable byte pointer at this instant, or null when READ
// access is absent or the root is closed. A valid open empty buffer may also
// return null. The returned pointer is not a revocable lease; its later
// lifetime and synchronization are the trusted C caller's concern.
IREE_API_EXPORT const void* iree_vm_buffer_const_data(
    const iree_vm_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BUFFER_H_
