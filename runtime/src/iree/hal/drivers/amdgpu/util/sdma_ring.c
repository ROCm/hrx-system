// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/sdma_ring.h"

#include <string.h>
static void iree_hal_amdgpu_sdma_ring_publish(iree_hal_amdgpu_sdma_ring_t* ring,
                                              uint64_t end) {
  iree_atomic_store(ring->write_position, (int64_t)end,
                    iree_memory_order_release);
  iree_atomic_store((iree_atomic_int64_t*)ring->doorbell, (int64_t)end,
                    iree_memory_order_release);
  ring->committed_position = end;
}

iree_status_t iree_hal_amdgpu_sdma_ring_initialize(
    uint32_t* base, uint32_t capacity, uint64_t* read_position,
    uint64_t* write_position, volatile uint64_t* doorbell,
    iree_hal_amdgpu_sdma_ring_t* out_ring) {
  IREE_ASSERT_ARGUMENT(out_ring);
  if (!base || ((uintptr_t)base & 3) || capacity < 256 ||
      (capacity & (capacity - 1)) || !read_position || !write_position ||
      !doorbell || ((uintptr_t)read_position & 7) ||
      ((uintptr_t)write_position & 7) || ((uintptr_t)doorbell & 7)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid SDMA ring storage");
  }
  uint64_t w = (uint64_t)iree_atomic_load((iree_atomic_int64_t*)write_position,
                                          iree_memory_order_acquire);
  uint64_t r = (uint64_t)iree_atomic_load((iree_atomic_int64_t*)read_position,
                                          iree_memory_order_acquire);
  if ((w & 3) || (r & 3) || r > w || w - r >= capacity) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "invalid SDMA ring positions");
  }
  *out_ring = (iree_hal_amdgpu_sdma_ring_t){
      .base = base,
      .capacity = capacity,
      .read_position = (iree_atomic_int64_t*)read_position,
      .write_position = (iree_atomic_int64_t*)write_position,
      .doorbell = doorbell,
      .committed_position = w,
      .reserved_bytes = 0};
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_sdma_ring_try_reserve(
    iree_hal_amdgpu_sdma_ring_t* ring, uint32_t byte_count,
    uint32_t** out_dwords) {
  IREE_ASSERT_ARGUMENT(ring);
  IREE_ASSERT_ARGUMENT(out_dwords);
  if (!byte_count || (byte_count & 3) || byte_count >= ring->capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "SDMA span does not fit ring");
  }
  if (ring->reserved_bytes) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "outstanding SDMA reservation");
  }
  uint64_t w = ring->committed_position;
  uint64_t r = (uint64_t)iree_atomic_load(ring->read_position,
                                          iree_memory_order_acquire);
  if ((r & 3) || r > w || w - r >= ring->capacity) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "invalid SDMA read position");
  }
  uint32_t offset = (uint32_t)(w & (ring->capacity - 1));
  uint32_t padding =
      byte_count > ring->capacity - offset ? ring->capacity - offset : 0;
  if (w > UINT64_MAX - padding - byte_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "SDMA position overflow");
  }
  if (padding) {
    if (w - r + padding >= ring->capacity)
      return iree_status_from_code(IREE_STATUS_UNAVAILABLE);
    memset((uint8_t*)ring->base + offset, 0, padding);
    w += padding;
    iree_hal_amdgpu_sdma_ring_publish(ring, w);
    offset = 0;
  }
  if (w - r + byte_count >= ring->capacity)
    return iree_status_from_code(IREE_STATUS_UNAVAILABLE);
  ring->reserved_bytes = byte_count;
  *out_dwords = (uint32_t*)((uint8_t*)ring->base + offset);
  return iree_ok_status();
}

void iree_hal_amdgpu_sdma_ring_cancel(iree_hal_amdgpu_sdma_ring_t* ring) {
  ring->reserved_bytes = 0;
}

void iree_hal_amdgpu_sdma_ring_commit(iree_hal_amdgpu_sdma_ring_t* ring) {
  IREE_ASSERT(ring->reserved_bytes != 0);
  iree_hal_amdgpu_sdma_ring_publish(
      ring, ring->committed_position + ring->reserved_bytes);
  ring->reserved_bytes = 0;
}
