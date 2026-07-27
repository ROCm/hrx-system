// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/blocking_printf_protocol.h"

#include <inttypes.h>
#include <string.h>

#include "binding/hip/printf_format.h"

iree_status_t iree_hip_blocking_printf_protocol_calculate_layout(
    uint32_t compute_unit_count, uint32_t maximum_waves_per_compute_unit,
    iree_hip_blocking_printf_protocol_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));

  if (IREE_UNLIKELY(compute_unit_count == 0 ||
                    maximum_waves_per_compute_unit == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "blocking printf requires non-zero compute-unit and resident-wave "
        "counts");
  }

  const uint64_t minimum_packet_count =
      (uint64_t)compute_unit_count * maximum_waves_per_compute_unit;
  if (IREE_UNLIKELY(minimum_packet_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "blocking printf packet count exceeds the legacy "
                            "hostcall ABI range");
  }

  const uint32_t packet_count =
      (uint32_t)iree_max(UINT64_C(2), minimum_packet_count);
  uint64_t index_capacity = 2;
  while (index_capacity < packet_count) index_capacity *= 2;
  const uint64_t index_mask = index_capacity - 1;

  iree_host_size_t packet_headers_offset = 0;
  iree_host_size_t packet_headers_size = 0;
  iree_host_size_t packet_headers_end = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_align(
                        sizeof(iree_hip_hostcall_buffer_header_t),
                        iree_alignof(iree_hip_hostcall_packet_header_t),
                        &packet_headers_offset) ||
                    !iree_host_size_checked_mul(
                        packet_count, sizeof(iree_hip_hostcall_packet_header_t),
                        &packet_headers_size) ||
                    !iree_host_size_checked_add(packet_headers_offset,
                                                packet_headers_size,
                                                &packet_headers_end))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "blocking printf packet header table size overflow");
  }

  iree_host_size_t packet_payloads_offset = 0;
  iree_host_size_t packet_payloads_size = 0;
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_align(
                        packet_headers_end,
                        iree_alignof(iree_hip_hostcall_packet_payload_t),
                        &packet_payloads_offset) ||
                    !iree_host_size_checked_mul(
                        packet_count,
                        sizeof(iree_hip_hostcall_packet_payload_t),
                        &packet_payloads_size) ||
                    !iree_host_size_checked_add(packet_payloads_offset,
                                                packet_payloads_size,
                                                &allocation_size))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "blocking printf packet payload table size overflow");
  }

  *out_layout = (iree_hip_blocking_printf_protocol_layout_t){
      .packet_count = packet_count,
      .index_mask = index_mask,
      .packet_headers_offset = packet_headers_offset,
      .packet_payloads_offset = packet_payloads_offset,
      .allocation_size = allocation_size,
  };
  return iree_ok_status();
}

void iree_hip_blocking_printf_protocol_initialize(
    void* host_pointer, uint64_t device_address, uint64_t doorbell_token,
    const iree_hip_blocking_printf_protocol_layout_t* layout,
    iree_hip_blocking_printf_protocol_t* out_protocol) {
  IREE_ASSERT_ARGUMENT(host_pointer);
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT_ARGUMENT(out_protocol);

  memset(host_pointer, 0, layout->allocation_size);
  uint8_t* host_base = (uint8_t*)host_pointer;
  iree_hip_hostcall_buffer_header_t* buffer_header =
      (iree_hip_hostcall_buffer_header_t*)host_base;
  iree_hip_hostcall_packet_header_t* packet_headers =
      (iree_hip_hostcall_packet_header_t*)(host_base +
                                           layout->packet_headers_offset);
  iree_hip_hostcall_packet_payload_t* packet_payloads =
      (iree_hip_hostcall_packet_payload_t*)(host_base +
                                            layout->packet_payloads_offset);

  buffer_header->headers = device_address + layout->packet_headers_offset;
  buffer_header->payloads = device_address + layout->packet_payloads_offset;
  buffer_header->doorbell = doorbell_token;
  buffer_header->index_mask = layout->index_mask;
  iree_atomic_store(&buffer_header->ready_stack, 0, iree_memory_order_relaxed);

  // Zero is the empty-stack sentinel, so packet zero begins with a non-zero
  // tag. Remaining packets can use their index directly for the first tag.
  uint64_t next = layout->index_mask + 1;
  for (uint32_t i = 0; i < layout->packet_count; ++i) {
    packet_headers[i].next = i == 0 ? 0 : next;
    if (i > 0) next = i;
  }
  buffer_header->free_stack = next;

  *out_protocol = (iree_hip_blocking_printf_protocol_t){
      .buffer_header = buffer_header,
      .packet_headers = packet_headers,
      .packet_payloads = packet_payloads,
      .packet_count = layout->packet_count,
      .index_mask = layout->index_mask,
  };
}

static void iree_hip_blocking_printf_protocol_complete_error(
    uint64_t payload[IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT]) {
  payload[0] = UINT64_MAX;
  payload[1] = 0;
}

static void iree_hip_blocking_printf_protocol_release_packet(
    iree_hip_hostcall_packet_header_t* packet_header) {
  const uint32_t control =
      iree_atomic_load(&packet_header->control, iree_memory_order_relaxed);
  iree_atomic_store(&packet_header->control,
                    control & ~IREE_HIP_HOSTCALL_PACKET_CONTROL_READY,
                    iree_memory_order_release);
}

static void iree_hip_blocking_printf_protocol_fail_packet(
    iree_hip_hostcall_packet_header_t* packet_header,
    iree_hip_hostcall_packet_payload_t* packet_payload) {
  uint64_t activemask = packet_header->activemask;
  while (activemask != 0) {
    const uint32_t lane = (uint32_t)__builtin_ctzll(activemask);
    activemask &= activemask - 1;
    iree_hip_blocking_printf_protocol_complete_error(
        packet_payload->slots[lane]);
  }
  iree_hip_blocking_printf_protocol_release_packet(packet_header);
}

static iree_status_t iree_hip_blocking_printf_protocol_format_message(
    const iree_hip_hostcall_message_result_t* message, FILE* output_stream,
    FILE* error_stream, int* out_count) {
  *out_count = -1;
  if (IREE_UNLIKELY(message->count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "blocking printf message has no control qword");
  }

  const uint64_t control = message->data[0];
  if (IREE_UNLIKELY(control & ~UINT64_C(1))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "blocking printf control qword has reserved bits set");
  }
  FILE* stream = (control & UINT64_C(1)) ? error_stream : output_stream;

  const iree_host_size_t message_length =
      (message->count - 1) * sizeof(message->data[0]);
  const uint8_t* message_bytes = (const uint8_t*)(message->data + 1);
  const char* format_end = memchr(message_bytes, '\0', message_length);
  if (IREE_UNLIKELY(!format_end)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "blocking printf format is unterminated");
  }

  const iree_host_size_t format_length =
      (iree_host_size_t)(format_end - (const char*)message_bytes);
  const iree_host_size_t argument_offset =
      iree_host_align(format_length + 1, sizeof(uint64_t));

  return iree_hip_printf_format(
      stream, iree_make_string_view((const char*)message_bytes, format_length),
      message_bytes + argument_offset, message_length - argument_offset,
      out_count);
}

static iree_status_t iree_hip_blocking_printf_protocol_process_lane(
    uint64_t payload[IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT],
    iree_hip_hostcall_message_table_t* message_table, FILE* output_stream,
    FILE* error_stream) {
  iree_hip_hostcall_message_result_t message;
  iree_status_t status = iree_hip_hostcall_message_consume_fragment(
      message_table, payload, &message);
  if (!iree_status_is_ok(status)) {
    iree_hip_blocking_printf_protocol_complete_error(payload);
    return status;
  }

  if (message.type == IREE_HIP_HOSTCALL_MESSAGE_RESULT_CONTINUE) {
    payload[0] = message.continuation_descriptor;
    payload[1] = 0;
    return iree_ok_status();
  }

  int result = -1;
  status = iree_hip_blocking_printf_protocol_format_message(
      &message, output_stream, error_stream, &result);
  iree_hip_hostcall_message_release(message_table, message.message_id);
  if (iree_status_is_ok(status)) {
    payload[0] = (uint64_t)(int64_t)result;
    payload[1] = 0;
  } else {
    iree_hip_blocking_printf_protocol_complete_error(payload);
  }
  return status;
}

static iree_status_t iree_hip_blocking_printf_protocol_process_packet(
    iree_hip_hostcall_packet_header_t* packet_header,
    iree_hip_hostcall_packet_payload_t* packet_payload,
    iree_hip_hostcall_message_table_t* message_table, FILE* output_stream,
    FILE* error_stream) {
  if (IREE_UNLIKELY(packet_header->service !=
                    IREE_HIP_HOSTCALL_SERVICE_PRINTF)) {
    const uint32_t service = packet_header->service;
    iree_hip_blocking_printf_protocol_fail_packet(packet_header,
                                                  packet_payload);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "blocking hostcall service %" PRIu32 " is unsupported", service);
  }

  iree_status_t status = iree_ok_status();
  uint64_t activemask = packet_header->activemask;
  while (activemask != 0) {
    const uint32_t lane = (uint32_t)__builtin_ctzll(activemask);
    activemask &= activemask - 1;
    uint64_t* payload = packet_payload->slots[lane];
    if (iree_status_is_ok(status)) {
      status = iree_hip_blocking_printf_protocol_process_lane(
          payload, message_table, output_stream, error_stream);
    } else {
      iree_hip_blocking_printf_protocol_complete_error(payload);
    }
  }
  iree_hip_blocking_printf_protocol_release_packet(packet_header);
  return status;
}

typedef enum iree_hip_blocking_printf_drain_mode_e {
  IREE_HIP_BLOCKING_PRINTF_DRAIN_MODE_PROCESS = 0,
  IREE_HIP_BLOCKING_PRINTF_DRAIN_MODE_FAIL = 1,
} iree_hip_blocking_printf_drain_mode_t;

static iree_status_t iree_hip_blocking_printf_protocol_drain_ready(
    iree_hip_blocking_printf_protocol_t* protocol,
    iree_hip_blocking_printf_drain_mode_t mode,
    iree_hip_hostcall_message_table_t* message_table, FILE* output_stream,
    FILE* error_stream) {
  uint64_t next = iree_atomic_exchange(&protocol->buffer_header->ready_stack, 0,
                                       iree_memory_order_acquire);
  iree_status_t status = iree_ok_status();
  uint32_t packet_ordinal = 0;
  while (next != 0 && packet_ordinal < protocol->packet_count) {
    const uint32_t packet_index = (uint32_t)(next & protocol->index_mask);
    if (IREE_UNLIKELY(packet_index >= protocol->packet_count)) {
      status = iree_status_join(
          status,
          iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "blocking printf ready stack references packet index %" PRIu32
              " outside the %" PRIu32 "-packet pool",
              packet_index, protocol->packet_count));
      break;
    }
    iree_hip_hostcall_packet_header_t* packet_header =
        &protocol->packet_headers[packet_index];
    iree_hip_hostcall_packet_payload_t* packet_payload =
        &protocol->packet_payloads[packet_index];
    const uint32_t control =
        iree_atomic_load(&packet_header->control, iree_memory_order_relaxed);
    if (IREE_UNLIKELY(!(control & IREE_HIP_HOSTCALL_PACKET_CONTROL_READY))) {
      status = iree_status_join(
          status,
          iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "blocking printf ready stack repeats or references an unready "
              "packet"));
      break;
    }

    next = packet_header->next;
    if (mode == IREE_HIP_BLOCKING_PRINTF_DRAIN_MODE_PROCESS &&
        iree_status_is_ok(status)) {
      status = iree_status_join(
          status, iree_hip_blocking_printf_protocol_process_packet(
                      packet_header, packet_payload, message_table,
                      output_stream, error_stream));
    } else {
      iree_hip_blocking_printf_protocol_fail_packet(packet_header,
                                                    packet_payload);
    }
    ++packet_ordinal;
  }

  if (IREE_UNLIKELY(next != 0 && packet_ordinal == protocol->packet_count)) {
    status = iree_status_join(
        status, iree_make_status(
                    IREE_STATUS_INVALID_ARGUMENT,
                    "blocking printf ready stack exceeds the packet pool"));
  }
  return status;
}

iree_status_t iree_hip_blocking_printf_protocol_process_ready(
    iree_hip_blocking_printf_protocol_t* protocol,
    iree_hip_hostcall_message_table_t* message_table, FILE* output_stream,
    FILE* error_stream) {
  IREE_ASSERT_ARGUMENT(protocol);
  IREE_ASSERT_ARGUMENT(message_table);
  IREE_ASSERT_ARGUMENT(output_stream);
  IREE_ASSERT_ARGUMENT(error_stream);
  return iree_hip_blocking_printf_protocol_drain_ready(
      protocol, IREE_HIP_BLOCKING_PRINTF_DRAIN_MODE_PROCESS, message_table,
      output_stream, error_stream);
}

iree_status_t iree_hip_blocking_printf_protocol_fail_ready(
    iree_hip_blocking_printf_protocol_t* protocol) {
  IREE_ASSERT_ARGUMENT(protocol);
  return iree_hip_blocking_printf_protocol_drain_ready(
      protocol, IREE_HIP_BLOCKING_PRINTF_DRAIN_MODE_FAIL,
      /*message_table=*/NULL, /*output_stream=*/NULL, /*error_stream=*/NULL);
}
