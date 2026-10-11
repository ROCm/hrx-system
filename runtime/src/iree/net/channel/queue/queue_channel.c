// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/queue/queue_channel.h"

#if !defined(IREE_ENDIANNESS_LITTLE) || !IREE_ENDIANNESS_LITTLE
#error "IREE queue channels require little-endian hosts"
#endif  // !IREE_ENDIANNESS_LITTLE

enum {
  IREE_NET_QUEUE_MESSAGE_VERSION_OFFSET = 0,
  IREE_NET_QUEUE_MESSAGE_TYPE_OFFSET = 1,
  IREE_NET_QUEUE_MESSAGE_WAIT_COUNT_OFFSET = 2,
  IREE_NET_QUEUE_MESSAGE_SIGNAL_COUNT_OFFSET = 3,
  IREE_NET_QUEUE_MESSAGE_QUEUE_ID_OFFSET = 4,
};

struct iree_net_queue_channel_t {
  // Borrowed endpoint used for receive callback dispatch and message sends.
  iree_net_message_endpoint_t endpoint;
  // Application callbacks installed during protocol handoff.
  iree_net_queue_channel_callbacks_t callbacks;
  // Host allocator used to free this channel.
  iree_allocator_t host_allocator;
};

typedef struct iree_net_queue_send_prefix_t {
  // Queue message type encoded in the header.
  iree_net_queue_message_type_t type;
  // Queue ID encoded in the header.
  uint32_t queue_id;
  // Send parameters borrowed for the synchronous endpoint call.
  const iree_net_queue_channel_send_params_t* params;
} iree_net_queue_send_prefix_t;

static void iree_net_queue_message_encode_header(
    iree_net_queue_message_type_t type, uint8_t wait_frontier_count,
    uint8_t signal_frontier_count, uint32_t queue_id,
    uint8_t out_header[IREE_NET_QUEUE_MESSAGE_HEADER_SIZE]) {
  out_header[IREE_NET_QUEUE_MESSAGE_VERSION_OFFSET] =
      IREE_NET_QUEUE_MESSAGE_VERSION;
  out_header[IREE_NET_QUEUE_MESSAGE_TYPE_OFFSET] = (uint8_t)type;
  out_header[IREE_NET_QUEUE_MESSAGE_WAIT_COUNT_OFFSET] = wait_frontier_count;
  out_header[IREE_NET_QUEUE_MESSAGE_SIGNAL_COUNT_OFFSET] =
      signal_frontier_count;
  iree_unaligned_store_le_u32(
      out_header + IREE_NET_QUEUE_MESSAGE_QUEUE_ID_OFFSET, queue_id);
}

static iree_net_queue_frontier_view_t iree_net_queue_frontier_view_make(
    const uint8_t* encoded_entries, uint8_t count) {
  return (iree_net_queue_frontier_view_t){
      .encoded_entries = iree_make_const_byte_span(
          encoded_entries,
          (iree_host_size_t)count * IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE),
      .count = count,
  };
}

static iree_net_queue_frontier_builder_t iree_net_queue_frontier_builder_make(
    uint8_t* encoded_entries, uint8_t count) {
  return (iree_net_queue_frontier_builder_t){
      .encoded_entries = iree_make_byte_span(
          encoded_entries,
          (iree_host_size_t)count * IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE),
      .count = count,
  };
}

static iree_status_t iree_net_queue_frontier_validate(
    const iree_net_queue_frontier_view_t* frontier, const char* frontier_name) {
  iree_async_axis_t previous_axis = 0;
  for (iree_host_size_t i = 0; i < frontier->count; ++i) {
    const iree_async_frontier_entry_t entry =
        iree_net_queue_frontier_view_get(frontier, i);
    if (entry.epoch == 0) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "%s frontier entry %" PRIhsz " has a zero epoch",
                              frontier_name, i);
    }
    if (i > 0 && entry.axis <= previous_axis) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "%s frontier entry %" PRIhsz " axis 0x%016" PRIx64
          " is not strictly greater than previous axis 0x%016" PRIx64,
          frontier_name, i, entry.axis, previous_axis);
    }
    previous_axis = entry.axis;
  }
  return iree_ok_status();
}

static iree_status_t iree_net_queue_channel_on_message(
    void* user_data, iree_const_byte_span_t message,
    iree_async_buffer_lease_t* lease) {
  iree_net_queue_channel_t* channel = (iree_net_queue_channel_t*)user_data;
  if (message.data_length < IREE_NET_QUEUE_MESSAGE_HEADER_SIZE) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "queue message has %" PRIhsz " bytes; expected at least %u",
        message.data_length, IREE_NET_QUEUE_MESSAGE_HEADER_SIZE);
  }

  const uint8_t* header = message.data;
  const uint8_t version = header[IREE_NET_QUEUE_MESSAGE_VERSION_OFFSET];
  if (version != IREE_NET_QUEUE_MESSAGE_VERSION) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "unsupported queue message version %u", version);
  }
  const uint8_t type = header[IREE_NET_QUEUE_MESSAGE_TYPE_OFFSET];
  const uint8_t wait_frontier_count =
      header[IREE_NET_QUEUE_MESSAGE_WAIT_COUNT_OFFSET];
  const uint8_t signal_frontier_count =
      header[IREE_NET_QUEUE_MESSAGE_SIGNAL_COUNT_OFFSET];
  const uint32_t queue_id = iree_unaligned_load_le_u32(
      header + IREE_NET_QUEUE_MESSAGE_QUEUE_ID_OFFSET);

  const iree_host_size_t wait_frontier_size =
      (iree_host_size_t)wait_frontier_count *
      IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  const iree_host_size_t signal_frontier_size =
      (iree_host_size_t)signal_frontier_count *
      IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  const iree_host_size_t frontier_size =
      wait_frontier_size + signal_frontier_size;
  const iree_host_size_t available_frontier_size =
      message.data_length - IREE_NET_QUEUE_MESSAGE_HEADER_SIZE;
  if (frontier_size > available_frontier_size) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "queue message frontiers require %" PRIhsz
                            " bytes but only %" PRIhsz " remain",
                            frontier_size, available_frontier_size);
  }

  const uint8_t* wait_frontier_data =
      message.data + IREE_NET_QUEUE_MESSAGE_HEADER_SIZE;
  const uint8_t* signal_frontier_data = wait_frontier_data + wait_frontier_size;
  const uint8_t* payload_data = signal_frontier_data + signal_frontier_size;
  const iree_net_queue_frontier_view_t wait_frontier =
      iree_net_queue_frontier_view_make(wait_frontier_data,
                                        wait_frontier_count);
  const iree_net_queue_frontier_view_t signal_frontier =
      iree_net_queue_frontier_view_make(signal_frontier_data,
                                        signal_frontier_count);
  const iree_const_byte_span_t payload = iree_make_const_byte_span(
      payload_data, available_frontier_size - frontier_size);

  IREE_RETURN_IF_ERROR(
      iree_net_queue_frontier_validate(&wait_frontier, "wait"));
  IREE_RETURN_IF_ERROR(
      iree_net_queue_frontier_validate(&signal_frontier, "signal"));

  switch (type) {
    case IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND:
      if (payload.data_length == 0) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "COMMAND message payload must be nonempty");
      }
      return channel->callbacks.on_command(channel->callbacks.user_data,
                                           queue_id, &wait_frontier,
                                           &signal_frontier, payload, lease);
    case IREE_NET_QUEUE_MESSAGE_TYPE_ADVANCE:
      if (queue_id != IREE_NET_QUEUE_ID_NONE) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "ADVANCE message queue ID must be NONE");
      }
      if (wait_frontier.count != 0) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "ADVANCE message wait frontier must be empty");
      }
      if (signal_frontier.count == 0) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "ADVANCE message signal frontier must be nonempty");
      }
      return channel->callbacks.on_advance(channel->callbacks.user_data,
                                           &signal_frontier, payload, lease);
    default:
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "unknown queue message type 0x%02X", type);
  }
}

static void iree_net_queue_channel_on_error(void* user_data,
                                            iree_status_t status) {
  iree_net_queue_channel_t* channel = (iree_net_queue_channel_t*)user_data;
  channel->callbacks.on_error(channel->callbacks.user_data, status);
}

static void iree_net_queue_channel_on_capacity(void* user_data) {
  iree_net_queue_channel_t* channel = user_data;
  channel->callbacks.on_capacity(channel->callbacks.user_data);
}

iree_status_t iree_net_queue_channel_allocate(
    iree_net_message_endpoint_t endpoint,
    iree_net_queue_channel_callbacks_t callbacks,
    iree_allocator_t host_allocator, iree_net_queue_channel_t** out_channel) {
  if (!out_channel) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_channel is required");
  }
  *out_channel = NULL;
  if (!endpoint.self || !endpoint.vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "message endpoint is required");
  }
  if (!callbacks.on_command || !callbacks.on_advance || !callbacks.on_error) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "COMMAND, ADVANCE, and error callbacks are required");
  }

  iree_net_queue_channel_t* channel = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*channel),
                                             (void**)&channel));
  channel->endpoint = endpoint;
  channel->callbacks = callbacks;
  channel->host_allocator = host_allocator;
  *out_channel = channel;
  return iree_ok_status();
}

void iree_net_queue_channel_free(iree_net_queue_channel_t* channel) {
  if (!channel) {
    return;
  }
  iree_allocator_free(channel->host_allocator, channel);
}

void iree_net_queue_channel_attach(iree_net_queue_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  iree_net_message_endpoint_set_callbacks(
      channel->endpoint,
      (iree_net_message_endpoint_callbacks_t){
          .on_message = iree_net_queue_channel_on_message,
          .on_error = iree_net_queue_channel_on_error,
          .user_data = channel,
          .on_capacity = channel->callbacks.on_capacity
                             ? iree_net_queue_channel_on_capacity
                             : NULL,
      });
}

iree_net_carrier_send_budget_t iree_net_queue_channel_query_send_budget(
    iree_net_queue_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  return iree_net_message_endpoint_query_send_budget(channel->endpoint);
}

static iree_status_t iree_net_queue_channel_measure_send(
    iree_net_queue_message_type_t type,
    const iree_net_queue_channel_send_params_t* params,
    iree_host_size_t* out_generated_prefix_length) {
  *out_generated_prefix_length = 0;
  if (!params) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue send parameters are required");
  }
  if (!params->completion_callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send completion callback is required");
  }
  if (params->payload.count > 0 && !params->payload.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue payload span list has null storage");
  }

  const bool builder_required = params->wait_frontier_count > 0 ||
                                params->signal_frontier_count > 0 ||
                                params->generated_payload_length > 0;
  if (builder_required != (params->build != NULL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue generated fields and builder callback disagree");
  }
  if (!params->build && params->build_user_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue send without a builder has builder data");
  }
  if (type == IREE_NET_QUEUE_MESSAGE_TYPE_ADVANCE &&
      params->wait_frontier_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ADVANCE wait frontier must be empty");
  }
  if (type == IREE_NET_QUEUE_MESSAGE_TYPE_ADVANCE &&
      params->signal_frontier_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ADVANCE signal frontier must be nonempty");
  }

  const iree_host_size_t wait_frontier_size =
      (iree_host_size_t)params->wait_frontier_count *
      IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  const iree_host_size_t signal_frontier_size =
      (iree_host_size_t)params->signal_frontier_count *
      IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  const iree_host_size_t generated_payload_offset =
      IREE_NET_QUEUE_MESSAGE_HEADER_SIZE + wait_frontier_size +
      signal_frontier_size;
  iree_host_size_t generated_prefix_length = 0;
  if (!iree_host_size_checked_add(generated_payload_offset,
                                  params->generated_payload_length,
                                  &generated_prefix_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "queue generated prefix length overflow");
  }

  bool has_payload = params->generated_payload_length > 0;
  iree_host_size_t message_length = generated_prefix_length;
  for (iree_host_size_t i = 0; i < params->payload.count; ++i) {
    const iree_host_size_t span_length = params->payload.values[i].length;
    has_payload |= span_length > 0;
    if (!iree_host_size_checked_add(message_length, span_length,
                                    &message_length)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "queue message length overflow");
    }
  }
  if (message_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "queue message length %" PRIhsz
                            " exceeds the 32-bit wire extent",
                            message_length);
  }
  if (type == IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND && !has_payload) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "COMMAND payload must be nonempty");
  }

  *out_generated_prefix_length = generated_prefix_length;
  return iree_ok_status();
}

static iree_status_t iree_net_queue_channel_write_prefix(
    void* user_data, iree_byte_span_t target) {
  const iree_net_queue_send_prefix_t* prefix =
      (const iree_net_queue_send_prefix_t*)user_data;
  const iree_net_queue_channel_send_params_t* params = prefix->params;
  iree_net_queue_message_encode_header(
      prefix->type, params->wait_frontier_count, params->signal_frontier_count,
      prefix->queue_id, target.data);

  uint8_t* wait_frontier_data =
      target.data + IREE_NET_QUEUE_MESSAGE_HEADER_SIZE;
  uint8_t* signal_frontier_data =
      wait_frontier_data + (iree_host_size_t)params->wait_frontier_count *
                               IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  uint8_t* generated_payload_data =
      signal_frontier_data + (iree_host_size_t)params->signal_frontier_count *
                                 IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  iree_net_queue_message_builder_t builder = {
      .wait_frontier = iree_net_queue_frontier_builder_make(
          wait_frontier_data, params->wait_frontier_count),
      .signal_frontier = iree_net_queue_frontier_builder_make(
          signal_frontier_data, params->signal_frontier_count),
      .generated_payload = iree_make_byte_span(
          generated_payload_data, params->generated_payload_length),
  };
  if (params->build) {
    return params->build(params->build_user_data, &builder);
  }
  return iree_ok_status();
}

static iree_status_t iree_net_queue_channel_send(
    iree_net_queue_channel_t* channel, iree_net_queue_message_type_t type,
    uint32_t queue_id, const iree_net_queue_channel_send_params_t* params) {
  iree_host_size_t generated_prefix_length = 0;
  IREE_RETURN_IF_ERROR(iree_net_queue_channel_measure_send(
      type, params, &generated_prefix_length));

  iree_net_queue_send_prefix_t prefix = {
      .type = type,
      .queue_id = queue_id,
      .params = params,
  };
  const iree_net_message_endpoint_send_params_t endpoint_params = {
      .generated_prefix =
          {
              .length = generated_prefix_length,
              .write = iree_net_queue_channel_write_prefix,
              .user_data = &prefix,
          },
      .data = params->payload,
      .completion_callback = params->completion_callback,
  };
  return iree_net_message_endpoint_send(channel->endpoint, &endpoint_params);
}

iree_status_t iree_net_queue_channel_send_command(
    iree_net_queue_channel_t* channel, uint32_t queue_id,
    const iree_net_queue_channel_send_params_t* params) {
  IREE_ASSERT_ARGUMENT(channel);
  return iree_net_queue_channel_send(
      channel, IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND, queue_id, params);
}

iree_status_t iree_net_queue_channel_send_advance(
    iree_net_queue_channel_t* channel,
    const iree_net_queue_channel_send_params_t* params) {
  IREE_ASSERT_ARGUMENT(channel);
  return iree_net_queue_channel_send(channel,
                                     IREE_NET_QUEUE_MESSAGE_TYPE_ADVANCE,
                                     IREE_NET_QUEUE_ID_NONE, params);
}
