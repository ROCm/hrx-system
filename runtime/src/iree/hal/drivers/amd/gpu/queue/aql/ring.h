// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_RING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_RING_H_

#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/packet.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One cache line owned by the queue's current publisher. Native construction
// supplies the mapped operands and a power-of-two packet capacity. Ring words
// are write-only to the host; only the native read index is observed. That
// index releases packet slots and never releases executable/argument storage.
typedef struct iree_hal_amd_gpu_aql_ring_t {
  // Borrowed host mapping, with native INVALID headers in all initial slots.
  iree_hal_amd_gpu_aql_packet_t* packets;
  // Borrowed coherent firmware consumption index, in packets.
  const iree_atomic_int64_t* read_index;
  // Borrowed coherent reservation index, written by this single publisher.
  iree_atomic_int64_t* write_index;
  // Borrowed write-only 64-bit MMIO notification aperture.
  volatile uint64_t* doorbell;
  // Same ring's GPU address for device fixups of reserved INVALID slots.
  uint64_t device_address;
  // Native power-of-two capacity in 64-byte packets.
  uint32_t capacity;
  // Capacity minus one, captured for slot and GPU-address arithmetic.
  uint32_t capacity_mask;
  // Exclusive end of all accepted reservations, in monotonic packet indices.
  uint64_t reserved;
  // Last acquired native consumption, refreshed only when capacity is short.
  uint64_t consumed;
} iree_hal_amd_gpu_aql_ring_t;
static_assert(sizeof(void*) != 8 || sizeof(iree_hal_amd_gpu_aql_ring_t) == 64,
              "AQL publication state fits one cache line");

// Binds an empty, already admitted native mapping without reading its ring.
// The caller owns native storage, all operand lifetimes, and shutdown.
void iree_hal_amd_gpu_aql_ring_initialize(
    iree_hal_amd_gpu_aql_packet_t* packets, uint64_t device_address,
    uint32_t capacity, const iree_atomic_int64_t* read_index,
    iree_atomic_int64_t* write_index, volatile uint64_t* doorbell,
    iree_hal_amd_gpu_aql_ring_t* out_ring);

// Reserves an admitted nonempty bounded chunk of at most capacity packets.
// Success publishes its native write index before any header can become valid.
// The current publisher owns encoding and notification through this chunk.
// No fallible step follows reservation; GPU-produced later headers may remain
// INVALID when the host notifies. False changes no reservation, native storage,
// or output; the caller defers on completion progress instead of spin-waiting.
bool iree_hal_amd_gpu_aql_ring_try_reserve(iree_hal_amd_gpu_aql_ring_t* ring,
                                           uint32_t packet_count,
                                           uint64_t* out_position);

// Copies only the body to an exclusively reserved slot, leaving its header
// INVALID. The designated host or device publisher later writes the complete
// header/setup DWORD. The source is disjoint immutable host storage.
void iree_hal_amd_gpu_aql_ring_write_body(
    const iree_hal_amd_gpu_aql_ring_t* ring, uint64_t position,
    const iree_hal_amd_gpu_aql_packet_t* packet);

// Release-publishes one host-owned header after its body is complete. Device
// fixup slots are never passed here. The enclosing chunk's notification follows
// all host header stores and explicitly permits later GPU-owned INVALID slots.
void iree_hal_amd_gpu_aql_ring_publish_header(
    const iree_hal_amd_gpu_aql_ring_t* ring, uint64_t position,
    uint32_t header);

// Fences host device stores and notifies the last reserved packet index.
// Called once per accepted nonempty chunk, without reading the MMIO aperture.
void iree_hal_amd_gpu_aql_ring_notify(const iree_hal_amd_gpu_aql_ring_t* ring);

// Exact GPU address of a reserved slot, without reading mapped native storage.
static inline uint64_t iree_hal_amd_gpu_aql_ring_packet_address(
    const iree_hal_amd_gpu_aql_ring_t* ring, uint64_t position) {
  return ring->device_address + (position & ring->capacity_mask) *
                                    sizeof(iree_hal_amd_gpu_aql_packet_t);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_RING_H_
