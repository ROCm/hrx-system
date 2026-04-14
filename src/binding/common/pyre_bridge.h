// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Bridge header for streaming code that calls pyre APIs while remaining
// canonical IREE code (iree_status_t returns, IREE_RETURN_IF_ERROR, etc).
//
// Streaming is always built from the same source tree as libpyre and
// shares internal type representations. Type punning between pyre and
// IREE types is valid (verified by _Static_assert in pyre_internal.h).

#ifndef PYRE_STREAMING_BRIDGE_H_
#define PYRE_STREAMING_BRIDGE_H_

#include "pyre_runtime.h"

#include "iree/base/api.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Status bridging: pyre_status_t <-> iree_status_t
//
// pyre_status_t and iree_status_t have incompatible internal layouts
// (pyre uses pyre_status_s*, IREE uses iree_status_storage_t*).
// Conversion must extract the code, free the source, and create a new
// status in the target format.
//
// Both use NULL = success, and both follow gRPC status code numbering
// (PYRE_STATUS_* values == IREE_STATUS_* values for the same semantics).
//===----------------------------------------------------------------------===//

static inline iree_status_t pyre_to_iree_status(pyre_status_t s) {
  if (pyre_status_is_ok(s)) return iree_ok_status();
  iree_status_code_t code = (iree_status_code_t)pyre_status_code(s);
  pyre_status_ignore(s);
  return iree_make_status(code);
}

static inline pyre_status_t iree_to_pyre_status(iree_status_t s) {
  if (iree_status_is_ok(s)) return pyre_ok_status();
  pyre_status_code_t code = (pyre_status_code_t)iree_status_code(s);
  iree_status_ignore(s);
  return pyre_make_status(code, NULL);
}

//===----------------------------------------------------------------------===//
// Allocator bridging: pyre_host_allocator_t <-> iree_allocator_t
//
// Both are two-word structs {self, ctl} with identical layout.
// Verified by _Static_assert in pyre_internal.h.
//===----------------------------------------------------------------------===//

static inline iree_allocator_t pyre_to_iree_allocator(
    pyre_host_allocator_t a) {
  iree_allocator_t v;
  memcpy(&v, &a, sizeof(v));
  return v;
}

static inline pyre_host_allocator_t iree_to_pyre_allocator(
    iree_allocator_t a) {
  pyre_host_allocator_t v;
  memcpy(&v, &a, sizeof(v));
  return v;
}

//===----------------------------------------------------------------------===//
// PYRE_CALL: wrap pyre API calls for use with IREE error macros
//
// Usage:
//   IREE_RETURN_IF_ERROR(PYRE_CALL(pyre_gpu_initialize(0)));
//   IREE_RETURN_IF_ERROR(PYRE_CALL(pyre_semaphore_create(dev, 0, &sem)));
//===----------------------------------------------------------------------===//

#define PYRE_CALL(expr) pyre_to_iree_status(expr)

//===----------------------------------------------------------------------===//
// Internal accessors
//
// Streaming shares internal type representations with libpyre (always
// built from the same source tree). These accessors extract IREE HAL
// handles from pyre types for direct HAL usage. NOT part of the public
// pyre API — only for code that mates with libpyre.
//===----------------------------------------------------------------------===//

#include "pyre_internal.h"

// Get the HAL device from a pyre device (for HAL calls not wrapped by pyre).
static inline iree_hal_device_t* pyre_device_hal(pyre_device_t dev) {
  return dev ? dev->hal_device : NULL;
}

// Get the system allocator as iree_allocator_t (shares mimalloc heap).
static inline iree_allocator_t pyre_system_iree_allocator(void) {
  return pyre_to_iree_allocator(pyre_host_allocator_system());
}

// Create a pyre_buffer_s wrapping a HAL buffer for buffer interop.
// The pyre_buf retains the HAL buffer and the device; caller owns the
// returned pyre_buffer_t with ref_count=1.  |hal_buffer| may be NULL
// for host-only allocations.
static inline iree_status_t pyre_buffer_create_from_hal(
    iree_hal_buffer_t* hal_buffer, pyre_device_t device,
    pyre_memory_type_t mem_type, size_t size, void* mapped_ptr,
    pyre_buffer_t* out_buffer) {
  pyre_buffer_s* buf = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(), sizeof(*buf), (void**)&buf));
  memset(buf, 0, sizeof(*buf));
  iree_atomic_ref_count_init(&buf->ref_count);
  buf->hal_buffer = hal_buffer;
  if (hal_buffer) iree_hal_buffer_retain(hal_buffer);
  buf->device = device;
  if (device) pyre_device_retain(device);
  buf->mem_type = mem_type;
  buf->size = size;
  buf->mapped_ptr = mapped_ptr;
  *out_buffer = buf;
  return iree_ok_status();
}

#endif  // PYRE_STREAMING_BRIDGE_H_
