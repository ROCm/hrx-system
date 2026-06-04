// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Bridge header for streaming code that calls hrx APIs while remaining
// canonical IREE code (iree_status_t returns, IREE_RETURN_IF_ERROR, etc).
//
// Streaming is always built from the same source tree as libhrx and
// shares internal type representations. Type punning between hrx and
// IREE types is valid (verified by _Static_assert in hrx_internal.h).

#ifndef HRX_STREAMING_BRIDGE_H_
#define HRX_STREAMING_BRIDGE_H_

#include <string.h>

#include "hrx_runtime.h"
#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// Status bridging: hrx_status_t <-> iree_status_t
//
// NULL is OK for both types so the success path is free. For non-OK
// statuses we cannot simply reinterpret-cast: hrx_status_t is a pointer
// to a heap-allocated hrx_status_s, while iree_status_t is a tagged
// pointer whose low bits hold the status code and whose high bits hold
// optional iree_status_storage_t. Feeding a hrx pointer to
// iree_status_free/_ignore would mask off the low bits and dereference
// garbage (which is exactly the crash we used to hit in the kernel-
// launch pointer scan).
//
// hrx_status_code_t values are the same as iree_status_code_t so we can
// safely pass the code along; we also copy the message so iree callers
// can format the error cleanly. The incoming hrx status is consumed
// (freed) since ownership is transferred.
//===----------------------------------------------------------------------===//

static inline iree_status_t hrx_to_iree_status(hrx_status_t s) {
  if (hrx_status_is_ok(s)) return iree_ok_status();
  iree_status_code_t code = (iree_status_code_t)hrx_status_code(s);
  char* message_buf = NULL;
  size_t message_len = 0;
  hrx_status_t to_str_status =
      hrx_status_to_string(s, &message_buf, &message_len);
  iree_status_t iree_s;
  if (hrx_status_is_ok(to_str_status) && message_buf) {
    iree_s = iree_make_status(code, "%.*s", (int)message_len, message_buf);
  } else {
    iree_s = iree_status_from_code(code);
  }
  if (message_buf) free(message_buf);
  if (!hrx_status_is_ok(to_str_status)) hrx_status_ignore(to_str_status);
  hrx_status_ignore(s);
  return iree_s;
}

static inline hrx_status_t iree_to_hrx_status(iree_status_t s) {
  return (hrx_status_t)(uintptr_t)s;
}

//===----------------------------------------------------------------------===//
// Allocator bridging: hrx_host_allocator_t <-> iree_allocator_t
//
// Both are two-word structs {self, ctl} with identical layout.
// Verified by _Static_assert in hrx_internal.h.
//===----------------------------------------------------------------------===//

static inline iree_allocator_t hrx_to_iree_allocator(hrx_host_allocator_t a) {
  iree_allocator_t v;
  memcpy(&v, &a, sizeof(v));
  return v;
}

static inline hrx_host_allocator_t iree_to_hrx_allocator(iree_allocator_t a) {
  hrx_host_allocator_t v;
  memcpy(&v, &a, sizeof(v));
  return v;
}

//===----------------------------------------------------------------------===//
// HRX_CALL: wrap hrx API calls for use with IREE error macros
//
// Usage:
//   IREE_RETURN_IF_ERROR(HRX_CALL(hrx_gpu_initialize(0)));
//   IREE_RETURN_IF_ERROR(HRX_CALL(hrx_semaphore_create(dev, 0, &sem)));
//===----------------------------------------------------------------------===//

#define HRX_CALL(expr) hrx_to_iree_status(expr)

//===----------------------------------------------------------------------===//
// Internal accessors
//
// Streaming shares internal type representations with libhrx (always
// built from the same source tree). These accessors extract IREE HAL
// handles from hrx types for direct HAL usage. NOT part of the public
// hrx API — only for code that mates with libhrx.
//===----------------------------------------------------------------------===//

#include "hrx_internal.h"

// Get the HAL device from a hrx device (for HAL calls not wrapped by hrx).
static inline iree_hal_device_t* hrx_device_hal(hrx_device_t dev) {
  return dev ? dev->hal_device : NULL;
}

// Get the system allocator as iree_allocator_t (shares mimalloc heap).
static inline iree_allocator_t hrx_system_iree_allocator(void) {
  return hrx_to_iree_allocator(hrx_host_allocator_system());
}

// Create a hrx_buffer_s wrapping a HAL buffer for buffer interop.
// The hrx_buf retains the HAL buffer and the device; caller owns the
// returned hrx_buffer_t with ref_count=1. |hal_buffer| may be NULL
// for host-only allocations.
static inline iree_status_t hrx_buffer_create_from_hal(
    iree_hal_buffer_t* hal_buffer, hrx_device_t device,
    hrx_memory_type_t mem_type, size_t size, void* mapped_ptr,
    hrx_buffer_t* out_buffer) {
  hrx_buffer_s* buf = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(iree_allocator_system(),
                                             sizeof(*buf), (void**)&buf));
  memset(buf, 0, sizeof(*buf));
  iree_atomic_ref_count_init(&buf->ref_count);
  buf->hal_buffer = hal_buffer;
  if (hal_buffer) iree_hal_buffer_retain(hal_buffer);
  buf->device = device;
  if (device) hrx_device_retain(device);
  buf->mem_type = mem_type;
  buf->size = size;
  buf->mapped_ptr = mapped_ptr;
  *out_buffer = buf;
  return iree_ok_status();
}

#endif  // HRX_STREAMING_BRIDGE_H_
