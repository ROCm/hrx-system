// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Bridge header for streaming code that calls pyre APIs while remaining
// canonical IREE code (iree_status_t returns, IREE_RETURN_IF_ERROR, etc).
//
// Streaming is always compiled to mate with libpyre.so, so type punning
// between pyre and IREE types is valid (verified by _Static_assert in
// pyre_internal.h).

#ifndef PYRE_STREAMING_BRIDGE_H_
#define PYRE_STREAMING_BRIDGE_H_

#include "pyre_runtime.h"

#include "iree/base/api.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Status bridging: pyre_status_t <-> iree_status_t
//
// Both are opaque pointers with NULL = OK. Cast is valid because:
// 1. Both use NULL to signal success
// 2. Both are pointer-sized
// 3. Streaming lives inside the pyre DSO boundary
//===----------------------------------------------------------------------===//

static inline iree_status_t pyre_to_iree_status(pyre_status_t s) {
  return (iree_status_t)(uintptr_t)s;
}

static inline pyre_status_t iree_to_pyre_status(iree_status_t s) {
  return (pyre_status_t)(uintptr_t)s;
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

#endif  // PYRE_STREAMING_BRIDGE_H_
