// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/bulk/bulk_channel.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/atomics.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/channel/util/send_gate.h"

// Flags that require payload transforms not implemented by the bulk channel.
#define IREE_NET_BULK_CHANNEL_UNSUPPORTED_TRANSFORM_FLAGS \
  IREE_NET_BULK_FRAME_FLAG_COMPRESSED

// Flags defined by the current bulk frame format.
#define IREE_NET_BULK_CHANNEL_KNOWN_FLAGS \
  (IREE_NET_BULK_FRAME_FLAG_COMPRESSED | IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK)

//===----------------------------------------------------------------------===//
// Fixed-capacity frame sender context pool
//===----------------------------------------------------------------------===//

typedef struct iree_net_bulk_send_context_t {
  // Intrusive free-list pointer for available send contexts.
  iree_atomic_slist_intrusive_ptr_t slist_next;

  // Frame-sender context storage returned through the allocator interface.
  iree_net_frame_send_context_t context;
} iree_net_bulk_send_context_t;

IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_net_bulk_send_context,
                                iree_net_bulk_send_context_t,
                                offsetof(iree_net_bulk_send_context_t,
                                         slist_next));

typedef struct iree_net_bulk_send_context_pool_t {
  // Available send context wrappers.
  iree_net_bulk_send_context_slist_t available_slist;

  // Total number of send context wrappers.
  iree_host_size_t capacity;

  // Number of wrappers currently available.
  iree_atomic_int32_t available_count;
} iree_net_bulk_send_context_pool_t;

static void iree_net_bulk_send_context_pool_initialize(
    iree_host_size_t capacity, iree_net_bulk_send_context_t* contexts,
    iree_net_bulk_send_context_pool_t* out_pool) {
  iree_net_bulk_send_context_slist_initialize(&out_pool->available_slist);
  out_pool->capacity = capacity;
  iree_atomic_store(&out_pool->available_count, (int32_t)capacity,
                    iree_memory_order_relaxed);
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    iree_net_bulk_send_context_slist_push(&out_pool->available_slist,
                                          &contexts[i]);
  }
}

static void iree_net_bulk_send_context_pool_deinitialize(
    iree_net_bulk_send_context_pool_t* pool) {
  int32_t available_count =
      iree_atomic_load(&pool->available_count, iree_memory_order_acquire);
  IREE_ASSERT((iree_host_size_t)available_count == pool->capacity,
              "bulk channel destroyed with %" PRIi32 " of %" PRIhsz
              " send contexts available",
              available_count, pool->capacity);
  iree_net_bulk_send_context_slist_discard(&pool->available_slist);
  iree_net_bulk_send_context_slist_deinitialize(&pool->available_slist);
}

static iree_status_t iree_net_bulk_send_context_pool_acquire(
    void* self, iree_net_frame_send_context_t** out_context) {
  *out_context = NULL;
  iree_net_bulk_send_context_pool_t* pool =
      (iree_net_bulk_send_context_pool_t*)self;
  iree_net_bulk_send_context_t* context =
      iree_net_bulk_send_context_slist_pop(&pool->available_slist);
  if (!context) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  iree_atomic_fetch_sub(&pool->available_count, 1, iree_memory_order_release);
  *out_context = &context->context;
  return iree_ok_status();
}

static void iree_net_bulk_send_context_pool_release(
    void* self, iree_net_frame_send_context_t* frame_context) {
  if (!frame_context) return;
  iree_net_bulk_send_context_pool_t* pool =
      (iree_net_bulk_send_context_pool_t*)self;
  iree_net_bulk_send_context_t* context =
      (iree_net_bulk_send_context_t*)((uint8_t*)frame_context -
                                      offsetof(iree_net_bulk_send_context_t,
                                               context));
  iree_net_bulk_send_context_slist_push(&pool->available_slist, context);
  iree_atomic_fetch_add(&pool->available_count, 1, iree_memory_order_release);
}

//===----------------------------------------------------------------------===//
// iree_net_bulk_channel_t
//===----------------------------------------------------------------------===//

struct iree_net_bulk_channel_t {
  // Reference count for channel lifetime management.
  iree_atomic_ref_count_t ref_count;

  // Host allocator used for the channel allocation.
  iree_allocator_t host_allocator;

  // Borrowed view into the transport. Valid until detach zeroes it.
  iree_net_message_endpoint_t endpoint;

  // Gates endpoint access and signals exact send quiescence during detach.
  iree_net_channel_send_gate_t send_gate;

  // Owned fallback header pool for unusually large scatter-gather sends.
  iree_async_buffer_pool_t* header_pool;

  // Fixed-capacity pool backing frame sender contexts.
  iree_net_bulk_send_context_pool_t send_context_pool;

  // Embedded frame sender for the send path.
  iree_net_frame_sender_t sender;

  // Maximum DATA chunk credits the peer may grant ahead of sent DATA.
  uint64_t remote_chunk_credit_capacity;

  // Cumulative local DATA chunk receive credit grant sent to the peer.
  iree_atomic_uint64_t local_chunk_credit_limit;

  // Cumulative peer DATA chunk receive credit grant observed locally.
  iree_atomic_uint64_t remote_chunk_credit_limit;

  // DATA chunks sent locally and charged against peer credit.
  iree_atomic_uint64_t remote_chunk_credit_consumed;

  // Application callbacks.
  iree_net_bulk_channel_callbacks_t callbacks;

  // Lifecycle state.
  iree_atomic_int32_t state;
};

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

static iree_net_bulk_channel_state_t iree_net_bulk_channel_load_state(
    const iree_net_bulk_channel_t* channel) {
  return (iree_net_bulk_channel_state_t)iree_atomic_load(
      &((iree_net_bulk_channel_t*)channel)->state, iree_memory_order_acquire);
}

static void iree_net_bulk_channel_set_state(
    iree_net_bulk_channel_t* channel, iree_net_bulk_channel_state_t new_state) {
  iree_atomic_store(&channel->state, (int32_t)new_state,
                    iree_memory_order_release);
}

static iree_status_t iree_net_bulk_channel_validate_supported_flags(
    iree_net_bulk_frame_type_t type, iree_net_bulk_frame_flags_t flags) {
  if (iree_any_bit_set(flags, ~IREE_NET_BULK_CHANNEL_KNOWN_FLAGS)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown bulk frame flags: type=0x%02X "
                            "flags=0x%02X",
                            (unsigned)type, (unsigned)flags);
  }
  if (iree_any_bit_set(flags,
                       IREE_NET_BULK_CHANNEL_UNSUPPORTED_TRANSFORM_FLAGS)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "bulk frame transform flags are not supported: "
                            "type=0x%02X flags=0x%02X",
                            (unsigned)type, (unsigned)flags);
  }
  if (type != IREE_NET_BULK_FRAME_TYPE_DATA &&
      iree_any_bit_set(flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bulk FINAL_CHUNK flag is only valid on DATA frames: type=0x%02X",
        (unsigned)type);
  }
  return iree_ok_status();
}

static iree_status_t iree_net_bulk_channel_require_state(
    iree_net_bulk_channel_t* channel, iree_net_bulk_frame_type_t type) {
  iree_net_bulk_channel_state_t state =
      iree_net_bulk_channel_load_state(channel);
  if (state != IREE_NET_BULK_CHANNEL_STATE_OPERATIONAL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot send bulk frame type 0x%02X: "
                            "channel state is %d",
                            (unsigned)type, (int)state);
  }
  return iree_ok_status();
}

static iree_status_t iree_net_bulk_channel_payload_length(
    iree_async_span_list_t payload, uint32_t* out_payload_length) {
  if (payload.count > 0 && !payload.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk payload span list is missing values");
  }
  iree_host_size_t payload_length = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < payload.count;
       ++i) {
    if (!iree_host_size_checked_add(payload_length, payload.values[i].length,
                                    &payload_length)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "bulk payload size overflow");
    }
  }
  if (iree_status_is_ok(status) && payload_length > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "bulk payload too large: %" PRIhsz
                              " bytes (max %" PRIu32 ")",
                              payload_length, UINT32_MAX);
  }
  if (iree_status_is_ok(status)) {
    *out_payload_length = (uint32_t)payload_length;
  }
  return status;
}

static iree_status_t iree_net_bulk_channel_grant_local_chunk_credits(
    iree_net_bulk_channel_t* channel, uint32_t credit_delta,
    uint64_t* out_credit_limit) {
  if (credit_delta == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk credit delta must be non-zero");
  }

  uint64_t credit_limit = iree_atomic_load(&channel->local_chunk_credit_limit,
                                           iree_memory_order_acquire);
  while (true) {
    if (credit_limit > UINT64_MAX - (uint64_t)credit_delta) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "bulk credit grant overflow");
    }
    uint64_t updated_credit_limit = credit_limit + (uint64_t)credit_delta;
    if (iree_atomic_compare_exchange_weak(&channel->local_chunk_credit_limit,
                                          &credit_limit, updated_credit_limit,
                                          iree_memory_order_acq_rel,
                                          iree_memory_order_acquire)) {
      *out_credit_limit = updated_credit_limit;
      return iree_ok_status();
    }
  }
}

static void iree_net_bulk_channel_revoke_local_chunk_credits(
    iree_net_bulk_channel_t* channel, uint32_t credit_delta,
    uint64_t credit_limit) {
  uint64_t expected_credit_limit = credit_limit;
  iree_atomic_compare_exchange_strong(
      &channel->local_chunk_credit_limit, &expected_credit_limit,
      credit_limit - (uint64_t)credit_delta, iree_memory_order_acq_rel,
      iree_memory_order_acquire);
}

static iree_status_t iree_net_bulk_channel_set_remote_chunk_credit_limit(
    iree_net_bulk_channel_t* channel, uint64_t credit_limit,
    uint32_t* out_credit_delta, uint32_t* out_available_credit_count) {
  uint64_t current_limit = iree_atomic_load(&channel->remote_chunk_credit_limit,
                                            iree_memory_order_acquire);
  while (true) {
    uint64_t consumed = iree_atomic_load(&channel->remote_chunk_credit_consumed,
                                         iree_memory_order_acquire);
    if (consumed > current_limit) {
      current_limit = iree_atomic_load(&channel->remote_chunk_credit_limit,
                                       iree_memory_order_acquire);
      continue;
    }

    if (credit_limit <= current_limit) {
      uint64_t available = current_limit - consumed;
      if (available > channel->remote_chunk_credit_capacity) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "bulk credit exceeds configured capacity: available=%" PRIu64
            " capacity=%" PRIu64,
            available, channel->remote_chunk_credit_capacity);
      }
      *out_credit_delta = 0;
      *out_available_credit_count = (uint32_t)available;
      return iree_ok_status();
    }

    uint64_t available = credit_limit - consumed;
    if (available > channel->remote_chunk_credit_capacity) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "bulk credit exceeds configured capacity: available=%" PRIu64
          " capacity=%" PRIu64,
          available, channel->remote_chunk_credit_capacity);
    }
    if (iree_atomic_compare_exchange_weak(
            &channel->remote_chunk_credit_limit, &current_limit, credit_limit,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      uint64_t credit_delta = credit_limit - current_limit;
      *out_credit_delta =
          credit_delta > UINT32_MAX ? UINT32_MAX : (uint32_t)credit_delta;
      *out_available_credit_count = (uint32_t)available;
      return iree_ok_status();
    }
  }
}

static iree_status_t iree_net_bulk_channel_consume_remote_chunk_credit(
    iree_net_bulk_channel_t* channel) {
  uint64_t consumed = iree_atomic_load(&channel->remote_chunk_credit_consumed,
                                       iree_memory_order_acquire);
  while (true) {
    uint64_t credit_limit = iree_atomic_load(
        &channel->remote_chunk_credit_limit, iree_memory_order_acquire);
    if (consumed >= credit_limit) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    uint64_t updated = consumed + 1;
    if (iree_atomic_compare_exchange_weak(
            &channel->remote_chunk_credit_consumed, &consumed, updated,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      return iree_ok_status();
    }
  }
}

static void iree_net_bulk_channel_refund_remote_chunk_credit(
    iree_net_bulk_channel_t* channel) {
  uint64_t previous = iree_atomic_fetch_sub(
      &channel->remote_chunk_credit_consumed, 1, iree_memory_order_acq_rel);
  IREE_ASSERT_GT(previous, 0u);
}

// Submit callback for frame_sender: routes sends through the message endpoint.
static iree_status_t iree_net_bulk_channel_submit_send(
    void* user_data, iree_async_span_list_t data, uint64_t send_user_data) {
  iree_net_bulk_channel_t* channel = (iree_net_bulk_channel_t*)user_data;

  if (!iree_net_channel_send_gate_try_enter(&channel->send_gate)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "bulk channel endpoint detached");
  }

  // Keep the channel, embedded frame sender, and fixed send-context pool alive
  // until the carrier posts the corresponding send completion.
  iree_net_bulk_channel_retain(channel);
  iree_net_message_endpoint_send_params_t params = {
      .data = data,
      .user_data = send_user_data,
  };
  iree_status_t status =
      iree_net_message_endpoint_send(channel->endpoint, &params);
  if (!iree_status_is_ok(status)) {
    iree_net_bulk_channel_release(channel);
  }

  iree_net_channel_send_gate_leave(&channel->send_gate);
  return status;
}

static void iree_net_bulk_channel_on_sender_complete(
    void* callback_user_data, uint64_t operation_user_data,
    iree_status_t status) {
  iree_net_bulk_channel_t* channel =
      (iree_net_bulk_channel_t*)callback_user_data;
  if (channel->callbacks.on_send_complete) {
    channel->callbacks.on_send_complete(channel->callbacks.user_data,
                                        operation_user_data, status);
  } else {
    iree_status_ignore(status);
  }
  iree_net_bulk_channel_release(channel);
}

static void iree_net_bulk_channel_on_endpoint_send_ready(void* user_data) {
  iree_net_bulk_channel_t* channel = (iree_net_bulk_channel_t*)user_data;
  if (iree_net_bulk_channel_load_state(channel) ==
          IREE_NET_BULK_CHANNEL_STATE_OPERATIONAL &&
      channel->callbacks.on_send_ready) {
    channel->callbacks.on_send_ready(channel->callbacks.user_data);
  }
}

//===----------------------------------------------------------------------===//
// Receive path
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_bulk_channel_on_message(
    void* user_data, iree_const_byte_span_t message,
    iree_async_buffer_lease_t* lease) {
  iree_net_bulk_channel_t* channel = (iree_net_bulk_channel_t*)user_data;

  iree_net_bulk_channel_state_t state =
      iree_net_bulk_channel_load_state(channel);
  if (state == IREE_NET_BULK_CHANNEL_STATE_ERROR) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "bulk channel is in error state");
  }

  if (message.data_length < IREE_NET_BULK_FRAME_HEADER_SIZE) {
    return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
  }

  iree_net_bulk_frame_header_t header;
  memcpy(&header, message.data, sizeof(header));
  IREE_RETURN_IF_ERROR(iree_net_bulk_frame_header_validate(header));

  iree_host_size_t payload_length =
      message.data_length - IREE_NET_BULK_FRAME_HEADER_SIZE;
  if (header.chunk_length != payload_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk frame chunk_length mismatch: header says %u, "
                            "message has %" PRIhsz " bytes after header",
                            header.chunk_length, payload_length);
  }

  iree_net_bulk_frame_type_t type = iree_net_bulk_frame_header_type(header);
  iree_net_bulk_frame_flags_t flags = iree_net_bulk_frame_header_flags(header);
  IREE_RETURN_IF_ERROR(
      iree_net_bulk_channel_validate_supported_flags(type, flags));

  iree_const_byte_span_t payload = iree_make_const_byte_span(
      message.data + IREE_NET_BULK_FRAME_HEADER_SIZE, payload_length);

  switch (type) {
    case IREE_NET_BULK_FRAME_TYPE_START:
      if (header.chunk_length != 0 || header.chunk_offset != 0 ||
          header.sequence != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "START frames cannot carry chunk payload metadata");
      }
      return channel->callbacks.on_start(channel->callbacks.user_data,
                                         header.transfer_id, header.total_size,
                                         flags);
    case IREE_NET_BULK_FRAME_TYPE_DATA:
      if (header.total_size != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "DATA frames cannot carry total_size");
      }
      return channel->callbacks.on_data(channel->callbacks.user_data,
                                        header.transfer_id, header.chunk_offset,
                                        header.sequence, flags, payload, lease);
    case IREE_NET_BULK_FRAME_TYPE_COMPLETE:
      if (header.chunk_length != 0 || header.chunk_offset != 0 ||
          header.total_size != 0 || header.sequence != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "COMPLETE frames cannot carry chunk payload metadata");
      }
      return channel->callbacks.on_complete(channel->callbacks.user_data,
                                            header.transfer_id);
    case IREE_NET_BULK_FRAME_TYPE_ABORT:
      if (header.total_size != 0 || header.chunk_offset != 0 ||
          header.sequence != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "ABORT frames cannot carry chunk payload metadata");
      }
      return channel->callbacks.on_abort(channel->callbacks.user_data,
                                         header.transfer_id, payload, lease);
    case IREE_NET_BULK_FRAME_TYPE_CREDIT: {
      if (flags != IREE_NET_BULK_FRAME_FLAG_NONE || header.transfer_id != 0 ||
          header.chunk_length != 0 || header.chunk_offset != 0 ||
          header.sequence != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "CREDIT frames cannot carry transfer or chunk metadata");
      }
      if (header.total_size == 0) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "invalid bulk credit limit: %" PRIu64,
                                header.total_size);
      }
      uint32_t credit_delta = 0;
      uint32_t available_credit_count = 0;
      IREE_RETURN_IF_ERROR(iree_net_bulk_channel_set_remote_chunk_credit_limit(
          channel, header.total_size, &credit_delta, &available_credit_count));
      if (channel->callbacks.on_credit && credit_delta > 0) {
        channel->callbacks.on_credit(channel->callbacks.user_data, credit_delta,
                                     available_credit_count);
      }
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown bulk frame type: 0x%02X",
                              (unsigned)type);
  }
}

static void iree_net_bulk_channel_on_endpoint_error(void* user_data,
                                                    iree_status_t status) {
  iree_net_bulk_channel_t* channel = (iree_net_bulk_channel_t*)user_data;

  iree_net_bulk_channel_set_state(channel, IREE_NET_BULK_CHANNEL_STATE_ERROR);

  if (channel->callbacks.on_transport_error) {
    channel->callbacks.on_transport_error(channel->callbacks.user_data, status);
  } else {
    iree_status_ignore(status);
  }
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_net_bulk_channel_destroy(iree_net_bulk_channel_t* channel);

iree_status_t iree_net_bulk_channel_create(
    iree_net_message_endpoint_t endpoint,
    const iree_net_bulk_channel_options_t* options,
    iree_async_buffer_pool_t* header_pool,
    iree_net_bulk_channel_callbacks_t callbacks,
    iree_allocator_t host_allocator, iree_net_bulk_channel_t** out_channel) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(header_pool);
  IREE_ASSERT_ARGUMENT(out_channel);
  *out_channel = NULL;

  iree_status_t status = iree_ok_status();
  if (!callbacks.on_start || !callbacks.on_data || !callbacks.on_complete ||
      !callbacks.on_abort) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bulk channel requires start, data, complete, and abort callbacks");
  }

  iree_net_bulk_channel_options_t resolved_options =
      iree_net_bulk_channel_options_default();
  if (options) {
    resolved_options = *options;
    if (resolved_options.send_context_capacity == 0) {
      resolved_options.send_context_capacity =
          IREE_NET_BULK_CHANNEL_DEFAULT_SEND_CONTEXT_CAPACITY;
    }
  }
  if (iree_status_is_ok(status) &&
      resolved_options.send_context_capacity > INT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "bulk send context capacity too large: %" PRIhsz,
                              resolved_options.send_context_capacity);
  }
  if (iree_status_is_ok(status) &&
      resolved_options.remote_chunk_credit_capacity == 0) {
    resolved_options.remote_chunk_credit_capacity =
        IREE_NET_BULK_CHANNEL_DEFAULT_REMOTE_CHUNK_CREDIT_CAPACITY;
  }
  if (iree_status_is_ok(status) &&
      resolved_options.remote_chunk_credit_capacity > INT32_MAX) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "bulk remote chunk credit capacity too large: %" PRIu32,
        resolved_options.remote_chunk_credit_capacity);
  }
  if (iree_status_is_ok(status) && resolved_options.max_send_spans == 0) {
    resolved_options.max_send_spans = IREE_NET_FRAME_SENDER_MAX_SPANS;
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t send_contexts_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_net_bulk_channel_t), &total_size,
        IREE_STRUCT_FIELD(resolved_options.send_context_capacity,
                          iree_net_bulk_send_context_t, &send_contexts_offset));
  }

  iree_net_bulk_channel_t* channel = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&channel);
  }

  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&channel->ref_count);
    channel->host_allocator = host_allocator;
    channel->endpoint = endpoint;
    iree_net_channel_send_gate_initialize(&channel->send_gate);
    channel->header_pool = header_pool;
    channel->callbacks = callbacks;
    channel->remote_chunk_credit_capacity =
        (uint64_t)resolved_options.remote_chunk_credit_capacity;
    iree_atomic_store(&channel->local_chunk_credit_limit, 0,
                      iree_memory_order_relaxed);
    iree_atomic_store(&channel->remote_chunk_credit_limit, 0,
                      iree_memory_order_relaxed);
    iree_atomic_store(&channel->remote_chunk_credit_consumed, 0,
                      iree_memory_order_relaxed);
    iree_atomic_store(&channel->state,
                      (int32_t)IREE_NET_BULK_CHANNEL_STATE_CREATED,
                      iree_memory_order_release);

    iree_net_bulk_send_context_t* send_contexts =
        (iree_net_bulk_send_context_t*)((uint8_t*)channel +
                                        send_contexts_offset);
    iree_net_bulk_send_context_pool_initialize(
        resolved_options.send_context_capacity, send_contexts,
        &channel->send_context_pool);

    iree_net_frame_send_complete_callback_t send_complete = {
        .fn = iree_net_bulk_channel_on_sender_complete,
        .user_data = channel,
    };
    iree_net_frame_sender_context_pool_t send_context_pool = {
        .acquire = iree_net_bulk_send_context_pool_acquire,
        .release = iree_net_bulk_send_context_pool_release,
        .user_data = &channel->send_context_pool,
    };
    status = iree_net_frame_sender_initialize_with_context_pool(
        &channel->sender, iree_net_bulk_channel_submit_send, channel,
        resolved_options.max_send_spans, header_pool, send_complete,
        send_context_pool, host_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_channel = channel;
  } else {
    if (channel) {
      iree_net_channel_send_gate_deinitialize(&channel->send_gate);
      iree_net_bulk_send_context_pool_deinitialize(&channel->send_context_pool);
      iree_allocator_free(host_allocator, channel);
    }
    iree_async_buffer_pool_release(header_pool);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_net_bulk_channel_retain(iree_net_bulk_channel_t* channel) {
  if (IREE_LIKELY(channel)) {
    iree_atomic_ref_count_inc(&channel->ref_count);
  }
}

void iree_net_bulk_channel_release(iree_net_bulk_channel_t* channel) {
  if (IREE_LIKELY(channel) &&
      iree_atomic_ref_count_dec(&channel->ref_count) == 1) {
    iree_net_bulk_channel_destroy(channel);
  }
}

void iree_net_bulk_channel_detach(iree_net_bulk_channel_t* channel) {
  if (!channel) return;

  if (iree_net_channel_send_gate_begin_close(&channel->send_gate)) {
    iree_net_channel_send_gate_await_quiescence(&channel->send_gate);
    if (channel->endpoint.self) {
      iree_net_message_endpoint_callbacks_t empty_callbacks;
      memset(&empty_callbacks, 0, sizeof(empty_callbacks));
      iree_net_message_endpoint_set_callbacks(channel->endpoint,
                                              empty_callbacks);
      memset(&channel->endpoint, 0, sizeof(channel->endpoint));
    }
    iree_net_bulk_channel_set_state(channel, IREE_NET_BULK_CHANNEL_STATE_ERROR);
    iree_net_channel_send_gate_finish_close(&channel->send_gate);
  } else {
    iree_net_channel_send_gate_await_closed(&channel->send_gate);
  }
}

static void iree_net_bulk_channel_destroy(iree_net_bulk_channel_t* channel) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (channel->endpoint.self) {
    iree_net_message_endpoint_callbacks_t empty_callbacks;
    memset(&empty_callbacks, 0, sizeof(empty_callbacks));
    iree_net_message_endpoint_set_callbacks(channel->endpoint, empty_callbacks);
  }

  iree_net_frame_sender_deinitialize(&channel->sender);
  iree_net_channel_send_gate_deinitialize(&channel->send_gate);
  iree_net_bulk_send_context_pool_deinitialize(&channel->send_context_pool);
  iree_async_buffer_pool_release(channel->header_pool);

  iree_allocator_t host_allocator = channel->host_allocator;
  iree_allocator_free(host_allocator, channel);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_net_bulk_channel_activate(iree_net_bulk_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_bulk_channel_state_t state =
      iree_net_bulk_channel_load_state(channel);
  if (state != IREE_NET_BULK_CHANNEL_STATE_CREATED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "channel not in CREATED state (state=%d)",
                            (int)state);
  }

  iree_net_message_endpoint_callbacks_t endpoint_callbacks = {
      .on_message = iree_net_bulk_channel_on_message,
      .on_error = iree_net_bulk_channel_on_endpoint_error,
      .user_data = channel,
      .on_send_ready = iree_net_bulk_channel_on_endpoint_send_ready,
  };
  iree_net_message_endpoint_set_callbacks(channel->endpoint,
                                          endpoint_callbacks);

  iree_status_t status = iree_net_message_endpoint_activate(channel->endpoint);
  if (iree_status_is_ok(status)) {
    iree_net_bulk_channel_set_state(channel,
                                    IREE_NET_BULK_CHANNEL_STATE_OPERATIONAL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Query
//===----------------------------------------------------------------------===//

iree_net_bulk_channel_state_t iree_net_bulk_channel_state(
    const iree_net_bulk_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  return iree_net_bulk_channel_load_state(channel);
}

bool iree_net_bulk_channel_has_pending_sends(
    const iree_net_bulk_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  return iree_net_frame_sender_has_pending(&channel->sender) ||
         iree_net_channel_send_gate_pending_count(&channel->send_gate) > 0;
}

iree_net_carrier_send_budget_t iree_net_bulk_channel_query_send_budget(
    iree_net_bulk_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  iree_net_bulk_channel_state_t state =
      iree_net_bulk_channel_load_state(channel);
  if (state != IREE_NET_BULK_CHANNEL_STATE_OPERATIONAL ||
      !channel->endpoint.self) {
    iree_net_carrier_send_budget_t empty_budget = {0};
    return empty_budget;
  }
  return iree_net_message_endpoint_query_send_budget(channel->endpoint);
}

uint32_t iree_net_bulk_channel_remote_chunk_credit_count(
    const iree_net_bulk_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  uint64_t credit_limit = iree_atomic_load(
      &((iree_net_bulk_channel_t*)channel)->remote_chunk_credit_limit,
      iree_memory_order_acquire);
  uint64_t consumed = iree_atomic_load(
      &((iree_net_bulk_channel_t*)channel)->remote_chunk_credit_consumed,
      iree_memory_order_acquire);
  if (credit_limit <= consumed) return 0;
  uint64_t available = credit_limit - consumed;
  return available > UINT32_MAX ? UINT32_MAX : (uint32_t)available;
}

//===----------------------------------------------------------------------===//
// Send path
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_bulk_channel_send_frame(
    iree_net_bulk_channel_t* channel, iree_net_bulk_frame_type_t type,
    iree_net_bulk_frame_flags_t flags, uint64_t transfer_id,
    uint64_t total_size, uint64_t chunk_offset, uint32_t chunk_length,
    uint32_t sequence, iree_async_span_list_t payload,
    uint64_t operation_user_data, bool consumes_remote_chunk_credit) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_net_bulk_channel_require_state(channel, type);
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_channel_validate_supported_flags(type, flags);
  }
  bool consumed_remote_chunk_credit = false;
  if (iree_status_is_ok(status) && consumes_remote_chunk_credit) {
    status = iree_net_bulk_channel_consume_remote_chunk_credit(channel);
    consumed_remote_chunk_credit = iree_status_is_ok(status);
  }

  iree_net_bulk_frame_header_t header;
  if (iree_status_is_ok(status)) {
    iree_net_bulk_frame_header_initialize(type, flags, transfer_id, total_size,
                                          chunk_offset, chunk_length, sequence,
                                          &header);
    status = iree_net_frame_sender_send(
        &channel->sender, iree_make_const_byte_span(&header, sizeof(header)),
        payload, operation_user_data);
  }
  if (!iree_status_is_ok(status) && consumed_remote_chunk_credit) {
    iree_net_bulk_channel_refund_remote_chunk_credit(channel);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_net_bulk_channel_send_start(
    iree_net_bulk_channel_t* channel, uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags, uint64_t operation_user_data) {
  return iree_net_bulk_channel_send_frame(
      channel, IREE_NET_BULK_FRAME_TYPE_START, flags, transfer_id, total_size,
      /*chunk_offset=*/0, /*chunk_length=*/0, /*sequence=*/0,
      iree_async_span_list_empty(), operation_user_data,
      /*consumes_remote_chunk_credit=*/false);
}

iree_status_t iree_net_bulk_channel_send_data(
    iree_net_bulk_channel_t* channel, uint64_t transfer_id,
    uint64_t chunk_offset, uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_async_span_list_t chunk_payload, uint64_t operation_user_data) {
  uint32_t chunk_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_net_bulk_channel_payload_length(chunk_payload, &chunk_length));
  return iree_net_bulk_channel_send_frame(
      channel, IREE_NET_BULK_FRAME_TYPE_DATA, flags, transfer_id,
      /*total_size=*/0, chunk_offset, chunk_length, sequence, chunk_payload,
      operation_user_data, /*consumes_remote_chunk_credit=*/true);
}

iree_status_t iree_net_bulk_channel_send_complete(
    iree_net_bulk_channel_t* channel, uint64_t transfer_id,
    uint64_t operation_user_data) {
  return iree_net_bulk_channel_send_frame(
      channel, IREE_NET_BULK_FRAME_TYPE_COMPLETE, IREE_NET_BULK_FRAME_FLAG_NONE,
      transfer_id, /*total_size=*/0,
      /*chunk_offset=*/0, /*chunk_length=*/0, /*sequence=*/0,
      iree_async_span_list_empty(), operation_user_data,
      /*consumes_remote_chunk_credit=*/false);
}

iree_status_t iree_net_bulk_channel_send_credit(
    iree_net_bulk_channel_t* channel, uint32_t credit_delta,
    uint64_t operation_user_data) {
  uint64_t credit_limit = 0;
  iree_status_t status = iree_net_bulk_channel_grant_local_chunk_credits(
      channel, credit_delta, &credit_limit);
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_channel_send_frame(
        channel, IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
        /*transfer_id=*/0, /*total_size=*/credit_limit, /*chunk_offset=*/0,
        /*chunk_length=*/0, /*sequence=*/0, iree_async_span_list_empty(),
        operation_user_data, /*consumes_remote_chunk_credit=*/false);
  }
  if (!iree_status_is_ok(status) && credit_limit > 0) {
    iree_net_bulk_channel_revoke_local_chunk_credits(channel, credit_delta,
                                                     credit_limit);
  }
  return status;
}

iree_status_t iree_net_bulk_channel_refresh_credit(
    iree_net_bulk_channel_t* channel, uint64_t operation_user_data) {
  uint64_t credit_limit = iree_atomic_load(&channel->local_chunk_credit_limit,
                                           iree_memory_order_acquire);
  if (credit_limit == 0) return iree_ok_status();
  return iree_net_bulk_channel_send_frame(
      channel, IREE_NET_BULK_FRAME_TYPE_CREDIT, IREE_NET_BULK_FRAME_FLAG_NONE,
      /*transfer_id=*/0, /*total_size=*/credit_limit, /*chunk_offset=*/0,
      /*chunk_length=*/0, /*sequence=*/0, iree_async_span_list_empty(),
      operation_user_data, /*consumes_remote_chunk_credit=*/false);
}

iree_status_t iree_net_bulk_channel_send_abort(
    iree_net_bulk_channel_t* channel, uint64_t transfer_id,
    iree_async_span_list_t abort_payload, uint64_t operation_user_data) {
  uint32_t abort_payload_length = 0;
  IREE_RETURN_IF_ERROR(iree_net_bulk_channel_payload_length(
      abort_payload, &abort_payload_length));
  return iree_net_bulk_channel_send_frame(
      channel, IREE_NET_BULK_FRAME_TYPE_ABORT, IREE_NET_BULK_FRAME_FLAG_NONE,
      transfer_id, /*total_size=*/0, /*chunk_offset=*/0, abort_payload_length,
      /*sequence=*/0, abort_payload, operation_user_data,
      /*consumes_remote_chunk_credit=*/false);
}
