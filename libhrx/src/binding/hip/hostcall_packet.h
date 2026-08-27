// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_HOSTCALL_PACKET_H_
#define HRX_BINDING_HIP_HOSTCALL_PACKET_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Number of qwords carried for one active wave lane.
#define IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT 8

// Maximum number of wave lanes represented by one packet.
#define IREE_HIP_HOSTCALL_PACKET_LANE_COUNT 64

// Bit identifying a packet that the device will wait for the host to complete.
#define IREE_HIP_HOSTCALL_PACKET_CONTROL_READY UINT32_C(1)

// Device-library service identifiers carried by packet headers.
typedef enum iree_hip_hostcall_service_e {
  IREE_HIP_HOSTCALL_SERVICE_PRINTF = 2,
} iree_hip_hostcall_service_t;

// Host view of one ROCm device-library hostcall packet header.
//
// This is the stack/tag hostcall ABI selected by ROCm device-library builds
// without USE_NEW_HOSTCALL_IMPL.
typedef struct iree_hip_hostcall_packet_header_t {
  // Tagged pointer to the next packet in the intrusive stack.
  uint64_t next;
  // Bitmask of active lanes whose payload slots are valid.
  uint64_t activemask;
  // Device-library service identifier requested by the wave.
  uint32_t service;
  // Bit 0 is the ready flag. The device waits until the host release-clears it.
  //
  // Setting this bit does not publish the packet to the host. The device owns
  // the packet until it release-pushes the packet onto |ready_stack|.
  iree_atomic_uint32_t control;
} iree_hip_hostcall_packet_header_t;

// Host view of one ROCm device-library hostcall packet payload.
typedef struct iree_hip_hostcall_packet_payload_t {
  // One eight-qword argument/return slot for each possible wave lane.
  uint64_t slots[IREE_HIP_HOSTCALL_PACKET_LANE_COUNT]
                [IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT];
} iree_hip_hostcall_packet_payload_t;

// ROCm device-library hostcall buffer descriptor shared with device code.
typedef struct iree_hip_hostcall_buffer_header_t {
  // Device address of the packet header array.
  uint64_t headers;
  // Device address of the packet payload array.
  uint64_t payloads;
  // HSA signal handle used by the device to notify the listener.
  uint64_t doorbell;
  // Tagged stack of packets available to device waves.
  uint64_t free_stack;
  // Tagged stack of packets awaiting host processing.
  //
  // A device release-push publishes packet contents. A host acquire-exchange
  // transfers ownership of the detached chain while device producers continue
  // pushing a new chain. Reading READY bits cannot substitute for detaching
  // this stack: READY is the device wait condition, not the ownership edge.
  iree_atomic_uint64_t ready_stack;
  // Mask used by device code to extract packet indexes from tagged pointers.
  uint64_t index_mask;
} iree_hip_hostcall_buffer_header_t;

static_assert(sizeof(iree_hip_hostcall_packet_header_t) == 24,
              "ROCm hostcall packet header ABI mismatch");
static_assert(offsetof(iree_hip_hostcall_packet_header_t, control) == 20,
              "ROCm hostcall packet control offset mismatch");
static_assert(sizeof(iree_hip_hostcall_packet_payload_t) == 4096,
              "ROCm hostcall packet payload ABI mismatch");
static_assert(sizeof(iree_hip_hostcall_buffer_header_t) == 48,
              "ROCm hostcall buffer header ABI mismatch");
static_assert(offsetof(iree_hip_hostcall_buffer_header_t, ready_stack) == 32,
              "ROCm hostcall ready stack offset mismatch");

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_HOSTCALL_PACKET_H_
