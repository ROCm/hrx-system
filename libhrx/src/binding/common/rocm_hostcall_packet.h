// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_STREAMING_ROCM_HOSTCALL_PACKET_H_
#define IREE_HAL_STREAMING_ROCM_HOSTCALL_PACKET_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

typedef struct iree_hal_streaming_hostcall_packet_header_t {
  // Tagged pointer to the next packet in the intrusive stack.
  uint64_t next;
  // Bitmask of active lanes whose payload slots are valid.
  uint64_t activemask;
  // Device-library service identifier requested by the wave.
  uint32_t service;
  // Bit 0 is the ready flag. The device waits until the host clears it.
  iree_atomic_uint32_t control;
} iree_hal_streaming_hostcall_packet_header_t;

typedef struct iree_hal_streaming_hostcall_payload_t {
  // One eight-qword argument/return slot for each possible wave lane.
  uint64_t slots[64][8];
} iree_hal_streaming_hostcall_payload_t;

typedef struct iree_hal_streaming_hostcall_buffer_header_t {
  // Device address of the packet header array.
  uint64_t headers;
  // Device address of the packet payload array.
  uint64_t payloads;
  // HSA signal handle used by the device to notify the listener.
  uint64_t doorbell;
  // Tagged stack of packets available to device waves.
  uint64_t free_stack;
  // Tagged stack of packets awaiting host processing.
  iree_atomic_uint64_t ready_stack;
  // Mask used by device code to extract packet indexes from tagged pointers.
  uint64_t index_mask;
} iree_hal_streaming_hostcall_buffer_header_t;

// Bounded iterator over a device-authored packet stack. The mask and traversal
// budget are derived from the immutable host allocation size so neither a
// corrupted shared mask nor a cyclic next link can escape the packet table.
typedef struct iree_hal_streaming_hostcall_packet_iterator_t {
  // Host view of the complete packet header table.
  const iree_hal_streaming_hostcall_packet_header_t* packet_headers;
  // Next tagged packet pointer supplied by the shared packet protocol.
  uint64_t next;
  // Host-derived mask selecting a valid packet table index.
  uint64_t packet_index_mask;
  // Maximum number of packet links that may still be traversed.
  uint32_t remaining_count;
} iree_hal_streaming_hostcall_packet_iterator_t;

static inline void iree_hal_streaming_hostcall_packet_iterator_initialize(
    const iree_hal_streaming_hostcall_packet_header_t* packet_headers,
    uint32_t packet_count, uint64_t ready_stack,
    iree_hal_streaming_hostcall_packet_iterator_t* out_iterator) {
  *out_iterator = (iree_hal_streaming_hostcall_packet_iterator_t){
      /*.packet_headers=*/packet_headers,
      /*.next=*/ready_stack,
      /*.packet_index_mask=*/packet_count ? packet_count - 1 : 0,
      /*.remaining_count=*/packet_count,
  };
}

static inline bool iree_hal_streaming_hostcall_packet_iterator_advance(
    iree_hal_streaming_hostcall_packet_iterator_t* iterator,
    uint32_t* out_packet_index) {
  if (iterator->next == 0 || iterator->remaining_count == 0) return false;
  const uint32_t packet_index =
      (uint32_t)(iterator->next & iterator->packet_index_mask);
  iterator->next = iterator->packet_headers[packet_index].next;
  --iterator->remaining_count;
  *out_packet_index = packet_index;
  return true;
}

#endif  // IREE_HAL_STREAMING_ROCM_HOSTCALL_PACKET_H_
