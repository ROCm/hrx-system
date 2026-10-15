// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_RING_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_RING_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed native ring and indices. Positions and capacity are in BYTES.
// Exactly one producer owns this state. External synchronization is required
// to transfer ownership. No storage is allocated or freed by this interface.
typedef struct iree_hal_amdgpu_sdma_ring_t {
  uint32_t* base;
  uint32_t capacity;
  iree_atomic_int64_t* read_position;
  iree_atomic_int64_t* write_position;
  volatile uint64_t* doorbell;
  uint64_t committed_position;
  uint32_t reserved_bytes;
} iree_hal_amdgpu_sdma_ring_t;
// Validates all pointers and native positions before publishing |out_ring|.
iree_status_t iree_hal_amdgpu_sdma_ring_initialize(
    uint32_t* base, uint32_t capacity, uint64_t* read_position,
    uint64_t* write_position, volatile uint64_t* doorbell,
    iree_hal_amdgpu_sdma_ring_t* out_ring);
// Nonblocking. UNAVAILABLE means retry after native capacity returns. May
// publish NOP wrap padding even when the requested span cannot yet be reserved.
// No requested commands are published until commit. Output unchanged on error.
// At most one outstanding reservation; caller must commit or cancel it.
iree_status_t iree_hal_amdgpu_sdma_ring_try_reserve(
    iree_hal_amdgpu_sdma_ring_t* ring, uint32_t byte_count,
    uint32_t** out_dwords);
void iree_hal_amdgpu_sdma_ring_cancel(iree_hal_amdgpu_sdma_ring_t* ring);
// Publishes all packet writes, then the native byte index and doorbell.
// Caller must populate the complete reserved span with valid packets first.
void iree_hal_amdgpu_sdma_ring_commit(iree_hal_amdgpu_sdma_ring_t* ring);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_RING_H_
