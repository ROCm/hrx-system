// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_BLOCKING_PRINTF_PROTOCOL_H_
#define HRX_BINDING_HIP_BLOCKING_PRINTF_PROTOCOL_H_

#include "binding/hip/hostcall_message.h"
#include "iree/hal/device.h"

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

// Output stream selected by a legacy blocking printf message.
typedef uint32_t iree_hip_blocking_printf_stream_t;
enum iree_hip_blocking_printf_stream_bits_t {
  IREE_HIP_BLOCKING_PRINTF_STREAM_STDOUT = 0u,
  IREE_HIP_BLOCKING_PRINTF_STREAM_STDERR = 1u,
};

// Receives one complete formatted printf message.
//
// |text| is borrowed from the service and valid only for the duration of the
// callback. The callback is synchronous: the device wave remains blocked until
// it returns.
typedef void(IREE_API_PTR* iree_hip_blocking_printf_output_fn_t)(
    void* user_data, iree_hip_blocking_printf_stream_t stream,
    iree_string_view_t text);

// Complete-message output sink owned by the hosting HIP layer.
typedef struct iree_hip_blocking_printf_output_sink_t {
  // Callback receiving one complete formatted message.
  iree_hip_blocking_printf_output_fn_t fn;
  // Opaque callback data.
  void* user_data;
} iree_hip_blocking_printf_output_sink_t;

// Thread-confined service state for one physical-device packet pool.
typedef struct iree_hip_blocking_printf_service_t {
  // Borrowed packet pool serviced by this context.
  iree_hip_blocking_printf_protocol_t* protocol;
  // Fragmented-message state required by the legacy device ABI.
  iree_hip_hostcall_message_table_t message_table;
  // Reusable contiguous copy of one completed encoded device message.
  iree_string_builder_t encoded_message_builder;
  // Reusable unpublished complete-message storage.
  iree_string_builder_t text_builder;
  // Reusable contiguous conversion-specifier storage.
  iree_string_builder_t format_scratch_builder;
  // Synchronous complete-message output sink.
  iree_hip_blocking_printf_output_sink_t output_sink;
  // Device error callback consuming structural failures.
  iree_hal_hostcall_error_callback_t error_callback;
  // True after a structural failure makes further interpretation unsafe.
  bool has_failed;
} iree_hip_blocking_printf_service_t;

// Calculates the packet pool required for one physical GPU.
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

// Initializes a thread-confined service over |protocol|.
//
// The sinks and allocator are retained until
// iree_hip_blocking_printf_service_deinitialize. Exactly one thread may call
// process_ready for the service.
void iree_hip_blocking_printf_service_initialize(
    iree_hip_blocking_printf_protocol_t* protocol,
    iree_hip_blocking_printf_output_sink_t output_sink,
    iree_hal_hostcall_error_callback_t error_callback,
    iree_allocator_t host_allocator,
    iree_hip_blocking_printf_service_t* out_service);

// Deinitializes host-only service state after its listener has joined.
void iree_hip_blocking_printf_service_deinitialize(
    iree_hip_blocking_printf_service_t* service);

// Acquires and completes every packet currently published on the ready stack.
//
// Format and argument errors produce a local printf -1 result without failing
// the service. A structural packet or fragment error publishes one terminal
// failure and puts the service into fail-completion mode. Every safely
// reachable acquired packet is release-completed so resident waves can make
// progress.
void iree_hip_blocking_printf_service_process_ready(
    iree_hip_blocking_printf_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_BLOCKING_PRINTF_PROTOCOL_H_
