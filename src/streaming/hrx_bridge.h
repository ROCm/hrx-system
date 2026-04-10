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

#include "hrx_runtime.h"

#include "iree/base/api.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Status bridging: hrx_status_t <-> iree_status_t
//
// Both are opaque pointers with NULL = OK. Cast is valid because:
// 1. Both use NULL to signal success
// 2. Both are pointer-sized
// 3. Streaming lives inside the hrx DSO boundary
//===----------------------------------------------------------------------===//

static inline iree_status_t hrx_to_iree_status(hrx_status_t s) {
  return (iree_status_t)(uintptr_t)s;
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

static inline iree_allocator_t hrx_to_iree_allocator(
    hrx_host_allocator_t a) {
  iree_allocator_t v;
  memcpy(&v, &a, sizeof(v));
  return v;
}

static inline hrx_host_allocator_t iree_to_hrx_allocator(
    iree_allocator_t a) {
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

#endif  // HRX_STREAMING_BRIDGE_H_
