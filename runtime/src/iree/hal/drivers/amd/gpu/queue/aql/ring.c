// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/ring.h"

#include "iree/hal/memory_scope.h"

void iree_hal_amd_gpu_aql_ring_initialize(
    iree_hal_amd_gpu_aql_packet_t* packets, uint64_t device_address,
    uint32_t capacity, const iree_atomic_int64_t* read_index,
    iree_atomic_int64_t* write_index, volatile uint64_t* doorbell,
    amdf_cache_transition_executor_t host_publish_executor,
    iree_hal_amd_gpu_aql_ring_t* out_ring) {
  *out_ring = (iree_hal_amd_gpu_aql_ring_t){
      .packets = packets,
      .read_index = read_index,
      .write_index = write_index,
      .doorbell = doorbell,
      .device_address = device_address,
      .capacity = capacity,
      .host_publish_executor = host_publish_executor,
  };
}

bool iree_hal_amd_gpu_aql_ring_try_reserve(iree_hal_amd_gpu_aql_ring_t* ring,
                                           uint32_t packet_count,
                                           uint64_t* out_position) {
  if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
    ring->consumed =
        (uint64_t)iree_atomic_load(ring->read_index, iree_memory_order_acquire);
    if (ring->reserved - ring->consumed + packet_count > ring->capacity) {
      return false;
    }
  }
  *out_position = ring->reserved;
  ring->reserved += packet_count;
  iree_atomic_store(ring->write_index, ring->reserved,
                    iree_memory_order_release);
  return true;
}

void iree_hal_amd_gpu_aql_ring_write_body(
    const iree_hal_amd_gpu_aql_ring_t* ring, uint64_t position,
    const iree_hal_amd_gpu_aql_packet_t* packet) {
  uint32_t* words = ring->packets[position & (ring->capacity - 1)].words;
  memcpy(words + 1, packet->words + 1, sizeof(*packet) - sizeof(uint32_t));
}

amdf_status_t iree_hal_amd_gpu_aql_ring_publish_header(
    const iree_hal_amd_gpu_aql_ring_t* ring,
    const iree_hal_amd_gpu_user_queue_t* storage, uint64_t position,
    uint32_t header) {
  uint32_t* words = ring->packets[position & (ring->capacity - 1)].words;
  if (ring->host_publish_executor != AMDF_CACHE_TRANSITION_EXECUTOR_NONE) {
    const amdf_status_t status = iree_hal_amd_gpu_user_queue_flush(storage);
    if (IREE_UNLIKELY(!amdf_status_is_ok(status))) {
      return status;
    }
  }
  iree_atomic_store((iree_atomic_int32_t*)words, header,
                    iree_memory_order_release);
  return AMDF_STATUS_OK;
}

void iree_hal_amd_gpu_aql_ring_notify(const iree_hal_amd_gpu_aql_ring_t* ring) {
  const iree_hal_memory_transition_recipe_info_t publication_fence = {
      .host = {.fence_after = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE},
  };
  iree_hal_memory_transition_execute_host_direct(&publication_fence, NULL, 1);
  *ring->doorbell = ring->reserved - 1;
}
