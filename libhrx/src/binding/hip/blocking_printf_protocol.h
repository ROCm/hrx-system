// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_BLOCKING_PRINTF_PROTOCOL_H_
#define HRX_BINDING_HIP_BLOCKING_PRINTF_PROTOCOL_H_

#include <stdio.h>

#include "binding/hip/hostcall_message.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Placement of one legacy hostcall packet pool in shared memory.
typedef struct iree_hip_blocking_printf_protocol_layout_t {
  // Packet count covering every potentially resident wave.
  uint32_t packet_count;
  // Power-of-two-minus-one mask used to decode tagged packet indexes.
  uint64_t index_mask;
  // Byte offset of the packet header table.
  iree_host_size_t packet_headers_offset;
  // Byte offset of the packet payload table.
  iree_host_size_t packet_payloads_offset;
  // Exact shared allocation size in bytes.
  iree_host_size_t allocation_size;
} iree_hip_blocking_printf_protocol_layout_t;

// Host-owned view of one initialized legacy hostcall packet pool.
//
// The storage remains owned by the hosting service. Exactly one listener may
// process this view at a time.
typedef struct iree_hip_blocking_printf_protocol_t {
  // Shared device-library buffer descriptor.
  iree_hip_hostcall_buffer_header_t* buffer_header;
  // Shared packet header table.
  iree_hip_hostcall_packet_header_t* packet_headers;
  // Shared packet payload table.
  iree_hip_hostcall_packet_payload_t* packet_payloads;
  // Host-private packet count used to bound device-authored stack traversal.
  uint32_t packet_count;
  // Host-private mask used to decode tagged packet indexes.
  uint64_t index_mask;
} iree_hip_blocking_printf_protocol_t;

// Calculates the packet pool required for one physical GPU queue.
//
// The legacy device library may block every resident wave on a distinct packet.
// The pool contains exactly one packet per potentially resident wave and is
// always at least two packets. Its tagged index mask covers the next power of
// two without allocating unused packet payloads.
iree_status_t iree_hip_blocking_printf_protocol_calculate_layout(
    uint32_t compute_unit_count, uint32_t maximum_waves_per_compute_unit,
    iree_hip_blocking_printf_protocol_layout_t* out_layout);

// Initializes shared storage before its device address is published.
//
// |host_pointer| addresses exactly |layout->allocation_size| writable bytes.
// |device_address| is the device-visible address of that same allocation.
void iree_hip_blocking_printf_protocol_initialize(
    void* host_pointer, uint64_t device_address, uint64_t doorbell_token,
    const iree_hip_blocking_printf_protocol_layout_t* layout,
    iree_hip_blocking_printf_protocol_t* out_protocol);

// Acquires and completes every packet currently published on the ready stack.
//
// Only service ID 2 is accepted. Every acquired packet is release-completed
// even when one request fails, ensuring resident device waves can make
// progress. A failure makes the remaining packets in the detached stack
// receive ABI error results without executing additional output.
iree_status_t iree_hip_blocking_printf_protocol_process_ready(
    iree_hip_blocking_printf_protocol_t* protocol,
    iree_hip_hostcall_message_table_t* message_table, FILE* output_stream,
    FILE* error_stream);

// Acquires ready packets and completes every active lane with an ABI error.
//
// This is the terminal listener path after a protocol or output failure.
iree_status_t iree_hip_blocking_printf_protocol_fail_ready(
    iree_hip_blocking_printf_protocol_t* protocol);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_BLOCKING_PRINTF_PROTOCOL_H_
