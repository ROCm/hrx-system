// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/bootstrap.h"

#include <string.h>

iree_status_t iree_net_bootstrap_topology_layout_calculate(
    iree_net_bootstrap_type_t message_type, uint32_t axis_count,
    iree_host_size_t application_data_length,
    iree_net_bootstrap_topology_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));

  iree_host_size_t fixed_length = 0;
  switch (message_type) {
    case IREE_NET_BOOTSTRAP_TYPE_HELLO:
      fixed_length = sizeof(iree_net_bootstrap_hello_t);
      break;
    case IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK:
      fixed_length = sizeof(iree_net_bootstrap_hello_ack_t);
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "bootstrap message type %u does not carry a topology",
          (unsigned)message_type);
  }

  iree_host_size_t axis_entries_length = 0;
  if (!iree_host_size_checked_mul(axis_count,
                                  sizeof(iree_net_bootstrap_axis_entry_t),
                                  &axis_entries_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bootstrap axis entry size overflow");
  }

  iree_host_size_t application_data_offset = 0;
  if (!iree_host_size_checked_add(fixed_length, axis_entries_length,
                                  &application_data_offset)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bootstrap payload header size overflow");
  }

  iree_host_size_t padded_application_data_length = 0;
  if (!iree_host_size_checked_align(application_data_length, 8,
                                    &padded_application_data_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bootstrap application data size overflow");
  }

  iree_host_size_t payload_length = 0;
  if (!iree_host_size_checked_add(application_data_offset,
                                  padded_application_data_length,
                                  &payload_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bootstrap payload size overflow");
  }

  out_layout->axis_entries_offset = fixed_length;
  out_layout->application_data_offset = application_data_offset;
  out_layout->payload_length = payload_length;
  return iree_ok_status();
}

iree_net_bootstrap_axis_entry_t iree_net_bootstrap_axis_list_get(
    const iree_net_bootstrap_axis_list_t* axis_list, uint32_t index) {
  IREE_ASSERT_ARGUMENT(axis_list);
  IREE_ASSERT(index < axis_list->count);
  iree_net_bootstrap_axis_entry_t entry;
  memcpy(&entry,
         axis_list->encoded_entries.data +
             index * sizeof(iree_net_bootstrap_axis_entry_t),
         sizeof(entry));
  return entry;
}

static iree_status_t iree_net_bootstrap_validate_capabilities(
    const char* message_name, iree_net_bootstrap_capabilities_t capabilities) {
  iree_net_bootstrap_capabilities_t unrecognized_capabilities =
      capabilities & ~IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED;
  if (unrecognized_capabilities != IREE_NET_BOOTSTRAP_CAPABILITY_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s capabilities contain unrecognized bits 0x%08x",
                            message_name, unrecognized_capabilities);
  }
  return iree_ok_status();
}

static iree_status_t iree_net_bootstrap_parse_topology(
    const char* message_name, iree_net_bootstrap_type_t message_type,
    uint16_t axis_count, uint64_t application_data_length,
    iree_const_byte_span_t payload, iree_net_bootstrap_axis_list_t* out_axes,
    iree_const_byte_span_t* out_application_data) {
  if (application_data_length > (uint64_t)IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%s application data length exceeds host size: %" PRIu64, message_name,
        application_data_length);
  }

  iree_net_bootstrap_topology_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_net_bootstrap_topology_layout_calculate(
      message_type, axis_count, (iree_host_size_t)application_data_length,
      &layout));
  if (payload.data_length != layout.payload_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s payload length %" PRIhsz
                            " does not match canonical length %" PRIhsz,
                            message_name, payload.data_length,
                            layout.payload_length);
  }

  iree_host_size_t application_data_end = 0;
  if (!iree_host_size_checked_add(layout.application_data_offset,
                                  (iree_host_size_t)application_data_length,
                                  &application_data_end)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s application data extent overflow",
                            message_name);
  }
  for (iree_host_size_t i = application_data_end; i < layout.payload_length;
       ++i) {
    if (payload.data[i] != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s alignment padding byte %" PRIhsz
                              " is nonzero",
                              message_name, i - application_data_end);
    }
  }

  out_axes->encoded_entries = iree_make_const_byte_span(
      payload.data + layout.axis_entries_offset,
      layout.application_data_offset - layout.axis_entries_offset);
  out_axes->count = axis_count;
  *out_application_data =
      iree_make_const_byte_span(payload.data + layout.application_data_offset,
                                (iree_host_size_t)application_data_length);
  return iree_ok_status();
}

static iree_status_t iree_net_bootstrap_parse_hello(
    iree_const_byte_span_t payload,
    iree_net_bootstrap_hello_view_t* out_hello) {
  if (payload.data_length < sizeof(iree_net_bootstrap_hello_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HELLO payload has a truncated fixed prefix");
  }

  iree_net_bootstrap_hello_t hello;
  memcpy(&hello, payload.data, sizeof(hello));
  if (hello.protocol_version != IREE_NET_BOOTSTRAP_PROTOCOL_VERSION) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HELLO protocol version %u does not match current version %u",
        hello.protocol_version, IREE_NET_BOOTSTRAP_PROTOCOL_VERSION);
  }
  IREE_RETURN_IF_ERROR(
      iree_net_bootstrap_validate_capabilities("HELLO", hello.capabilities));
  if (hello.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HELLO reserved field is nonzero");
  }

  iree_net_bootstrap_axis_list_t axes;
  iree_const_byte_span_t application_data;
  IREE_RETURN_IF_ERROR(iree_net_bootstrap_parse_topology(
      "HELLO", IREE_NET_BOOTSTRAP_TYPE_HELLO, hello.axis_count,
      hello.application_data_length, payload, &axes, &application_data));

  out_hello->fixed = hello;
  out_hello->axes = axes;
  out_hello->application_data = application_data;
  return iree_ok_status();
}

static iree_status_t iree_net_bootstrap_parse_hello_ack(
    iree_const_byte_span_t payload,
    iree_net_bootstrap_hello_ack_view_t* out_hello_ack) {
  if (payload.data_length < sizeof(iree_net_bootstrap_hello_ack_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HELLO_ACK payload has a truncated fixed prefix");
  }

  iree_net_bootstrap_hello_ack_t hello_ack;
  memcpy(&hello_ack, payload.data, sizeof(hello_ack));
  if (hello_ack.session_id == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HELLO_ACK session ID must be nonzero");
  }
  IREE_RETURN_IF_ERROR(iree_net_bootstrap_validate_capabilities(
      "HELLO_ACK", hello_ack.negotiated_capabilities));

  iree_net_bootstrap_axis_list_t axes;
  iree_const_byte_span_t application_data;
  IREE_RETURN_IF_ERROR(iree_net_bootstrap_parse_topology(
      "HELLO_ACK", IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK, hello_ack.axis_count,
      hello_ack.application_data_length, payload, &axes, &application_data));

  out_hello_ack->fixed = hello_ack;
  out_hello_ack->axes = axes;
  out_hello_ack->application_data = application_data;
  return iree_ok_status();
}

static iree_status_t iree_net_bootstrap_parse_reject(
    iree_const_byte_span_t payload,
    iree_net_bootstrap_reject_view_t* out_reject) {
  if (payload.data_length < sizeof(iree_net_bootstrap_reject_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "REJECT payload has a truncated fixed prefix");
  }

  iree_net_bootstrap_reject_t reject;
  memcpy(&reject, payload.data, sizeof(reject));
  if (reject.reason_code == IREE_STATUS_OK ||
      reject.reason_code > IREE_STATUS_CODE_MASK) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "REJECT reason code %u is invalid",
                            reject.reason_code);
  }
  if (reject.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "REJECT reserved field is nonzero");
  }

  out_reject->fixed = reject;
  out_reject->reason =
      iree_make_string_view((const char*)payload.data + sizeof(reject),
                            payload.data_length - sizeof(reject));
  return iree_ok_status();
}

iree_status_t iree_net_bootstrap_message_parse(
    iree_const_byte_span_t payload,
    iree_net_bootstrap_message_view_t* out_message) {
  IREE_ASSERT_ARGUMENT(out_message);
  memset(out_message, 0, sizeof(*out_message));

  if (payload.data_length < sizeof(iree_net_bootstrap_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bootstrap message has a truncated header");
  }

  iree_net_bootstrap_header_t header;
  memcpy(&header, payload.data, sizeof(header));
  if (header.reserved0[0] != 0 || header.reserved0[1] != 0 ||
      header.reserved0[2] != 0 || header.reserved1 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bootstrap header reserved fields are nonzero");
  }

  iree_net_bootstrap_message_view_t message;
  memset(&message, 0, sizeof(message));
  iree_status_t status = iree_ok_status();
  switch ((iree_net_bootstrap_type_t)header.type) {
    case IREE_NET_BOOTSTRAP_TYPE_HELLO:
      status = iree_net_bootstrap_parse_hello(payload, &message.value.hello);
      break;
    case IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK:
      status =
          iree_net_bootstrap_parse_hello_ack(payload, &message.value.hello_ack);
      break;
    case IREE_NET_BOOTSTRAP_TYPE_REJECT:
      status = iree_net_bootstrap_parse_reject(payload, &message.value.reject);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "bootstrap message type %u is unrecognized",
                                header.type);
      break;
  }

  if (iree_status_is_ok(status)) {
    message.type = (iree_net_bootstrap_type_t)header.type;
    *out_message = message;
  }
  return status;
}
