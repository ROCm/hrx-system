// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk_profile_receiver.h"

#include <string.h>

#include "iree/hal/remote/protocol/profile.h"
#include "iree/hal/remote/util/bulk_transfer_tracker.h"
#include "iree/net/channel/bulk/transfer_table.h"

static iree_hal_remote_client_profile_transfer_t*
iree_hal_remote_client_profile_transfer_from_sequence_node(
    iree_net_sequence_node_t* sequence_node) {
  return iree_containerof(
      sequence_node, iree_hal_remote_client_profile_transfer_t, sequence_node);
}

void iree_hal_remote_client_bulk_profile_receiver_free_transfer(
    iree_hal_remote_client_profile_transfer_t* transfer) {
  if (!transfer) return;
  iree_allocator_t host_allocator = transfer->host_allocator;
  iree_hal_remote_bulk_transfer_tracker_deinitialize(
      &transfer->receive_tracker);
  iree_allocator_free(host_allocator, transfer->contents.data);
  iree_allocator_free(host_allocator, transfer);
}

static void iree_hal_remote_client_profile_transfer_list_free(
    iree_net_sequence_node_t* transfer_list) {
  while (transfer_list) {
    iree_net_sequence_node_t* next = transfer_list->next;
    iree_hal_remote_client_profile_transfer_t* profile_transfer =
        iree_hal_remote_client_profile_transfer_from_sequence_node(
            transfer_list);
    iree_hal_remote_client_bulk_profile_receiver_free_transfer(
        profile_transfer);
    transfer_list = next;
  }
}

static iree_hal_remote_client_profile_transfer_t*
iree_hal_remote_client_bulk_profile_receiver_take_transfer(
    iree_net_bulk_transfer_table_t* table, iree_net_bulk_transfer_t* transfer) {
  const uint64_t transfer_id = iree_net_bulk_transfer_id(transfer);
  iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_client_bulk_transfer_storage(transfer);
  iree_hal_remote_client_profile_transfer_t* profile_transfer =
      bulk_transfer->profile_receive;
  memset(bulk_transfer, 0, sizeof(*bulk_transfer));
  bool was_removed = iree_net_bulk_transfer_table_remove(table, transfer_id);
  IREE_ASSERT(was_removed);
  (void)was_removed;
  return profile_transfer;
}

static void iree_hal_remote_client_bulk_profile_receiver_release_transfer(
    iree_net_bulk_transfer_table_t* table, iree_net_bulk_transfer_t* transfer) {
  iree_hal_remote_client_bulk_profile_receiver_free_transfer(
      iree_hal_remote_client_bulk_profile_receiver_take_transfer(table,
                                                                 transfer));
}

static void iree_hal_remote_client_bulk_profile_receiver_release_pending_locked(
    iree_hal_remote_client_device_t* device) {
  iree_net_sequence_node_t* pending_list = NULL;
  iree_net_sequence_window_take_pending(
      &device->bulk_session.profile_sequence_window, &pending_list);
  iree_hal_remote_client_profile_transfer_list_free(pending_list);
}

static void iree_hal_remote_client_bulk_collect_profile_transfer(
    void* user_data, iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_client_bulk_transfer_id_list_t* id_list =
      (iree_hal_remote_client_bulk_transfer_id_list_t*)user_data;
  iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_client_bulk_transfer_storage(table_transfer);
  if (bulk_transfer->kind !=
      IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE) {
    return;
  }
  if (id_list->transfer_count >= IREE_ARRAYSIZE(id_list->transfer_ids)) return;
  id_list->transfer_ids[id_list->transfer_count++] =
      iree_net_bulk_transfer_id(table_transfer);
}

void iree_hal_remote_client_bulk_profile_receiver_release_all_locked(
    iree_hal_remote_client_device_t* device) {
  iree_hal_remote_client_bulk_profile_receiver_release_pending_locked(device);
  if (!device->bulk_session.transfers) return;

  iree_hal_remote_client_bulk_transfer_id_list_t id_list;
  memset(&id_list, 0, sizeof(id_list));
  iree_net_bulk_transfer_table_visit(
      device->bulk_session.transfers,
      iree_hal_remote_client_bulk_collect_profile_transfer, &id_list);
  for (iree_host_size_t i = 0; i < id_list.transfer_count; ++i) {
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(device->bulk_session.transfers,
                                            id_list.transfer_ids[i]);
    if (table_transfer) {
      iree_hal_remote_client_bulk_profile_receiver_release_transfer(
          device->bulk_session.transfers, table_transfer);
    }
  }
}

iree_status_t iree_hal_remote_client_bulk_begin_profile_session(
    iree_hal_remote_client_device_t* device, iree_hal_profile_sink_t* sink) {
  IREE_ASSERT_ARGUMENT(device);

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  status = iree_hal_remote_client_bulk_session_check_active_locked(
      &device->bulk_session);
  if (iree_status_is_ok(status) && device->bulk_session.profile_sink) {
    status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "remote profiling session already active");
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_bulk_profile_receiver_release_all_locked(device);
    iree_net_sequence_window_deinitialize(
        &device->bulk_session.profile_sequence_window);
    status = iree_net_sequence_window_initialize(
        /*initial_observed_sequence=*/0, /*initial_capacity=*/64,
        device->host_allocator, &device->bulk_session.profile_sequence_window);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_profile_sink_retain(sink);
    device->bulk_session.profile_sink = sink;
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  return status;
}

bool iree_hal_remote_client_bulk_has_profile_session(
    iree_hal_remote_client_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);

  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  const bool has_profile_session = device->bulk_session.profile_sink != NULL;
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  return has_profile_session;
}

void iree_hal_remote_client_bulk_end_profile_session(
    iree_hal_remote_client_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);

  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_hal_profile_sink_t* sink = device->bulk_session.profile_sink;
  device->bulk_session.profile_sink = NULL;
  iree_hal_remote_client_bulk_profile_receiver_release_all_locked(device);
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  iree_hal_profile_sink_release(sink);
}

static iree_status_t iree_hal_remote_client_profile_transfer_parse(
    iree_hal_remote_client_profile_transfer_t* transfer,
    iree_host_size_t* out_content_type_offset,
    iree_host_size_t* out_name_offset, iree_host_size_t* out_payload_offset) {
  *out_content_type_offset = 0;
  *out_name_offset = 0;
  *out_payload_offset = 0;

  if (transfer->contents.data_length <
      sizeof(iree_hal_remote_profile_transfer_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote profile transfer_id=%" PRIu64
                            " too small for header: %" PRIhsz " bytes",
                            transfer->transfer_id,
                            transfer->contents.data_length);
  }

  memcpy(&transfer->header, transfer->contents.data, sizeof(transfer->header));
  const iree_hal_remote_profile_transfer_header_t* header = &transfer->header;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(header->reserved); ++i) {
    if (header->reserved[i] != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote profile transfer_id=%" PRIu64
                              " reserved[%zu] is nonzero",
                              transfer->transfer_id, i);
    }
  }
  if (header->sequence == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote profile transfer_id=%" PRIu64
                            " has zero sequence",
                            transfer->transfer_id);
  }
  switch ((iree_hal_remote_profile_callback_type_t)header->callback_type) {
    case IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_BEGIN_SESSION:
    case IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_WRITE_CHUNK:
    case IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_END_SESSION:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote profile transfer_id=%" PRIu64
                              " has invalid callback type 0x%02x",
                              transfer->transfer_id, header->callback_type);
  }
  if (header->callback_type !=
          IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_END_SESSION &&
      header->session_status_code != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote profile transfer_id=%" PRIu64
                            " has session_status_code on a non-END callback",
                            transfer->transfer_id);
  }
  if (header->payload_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "remote profile transfer_id=%" PRIu64 " payload length %" PRIu64
        " exceeds host size max %" PRIhsz,
        transfer->transfer_id, header->payload_length, IREE_HOST_SIZE_MAX);
  }

  iree_host_size_t required_length = 0;
  iree_status_t status = iree_hal_remote_profile_transfer_layout(
      header->content_type_length, header->name_length,
      (iree_host_size_t)header->payload_length, &required_length,
      out_content_type_offset, out_name_offset, out_payload_offset);
  if (iree_status_is_ok(status) &&
      required_length != transfer->contents.data_length) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote profile transfer_id=%" PRIu64
                              " length mismatch: header describes %" PRIhsz
                              " bytes but transfer has %" PRIhsz,
                              transfer->transfer_id, required_length,
                              transfer->contents.data_length);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_dispatch_profile_transfer(
    iree_hal_profile_sink_t* sink,
    iree_hal_remote_client_profile_transfer_t* transfer) {
  iree_host_size_t content_type_offset = 0;
  iree_host_size_t name_offset = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_profile_transfer_parse(
      transfer, &content_type_offset, &name_offset, &payload_offset));

  const iree_hal_remote_profile_transfer_header_t* header = &transfer->header;
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = iree_make_string_view(
      (const char*)transfer->contents.data + content_type_offset,
      header->content_type_length);
  metadata.name = iree_make_string_view(
      (const char*)transfer->contents.data + name_offset, header->name_length);
  metadata.session_id = header->session_id;
  metadata.stream_id = header->stream_id;
  metadata.event_id = header->event_id;
  metadata.executable_id = header->executable_id;
  metadata.command_buffer_id = header->command_buffer_id;
  metadata.physical_device_ordinal = header->physical_device_ordinal;
  metadata.queue_ordinal = header->queue_ordinal;
  metadata.flags = header->chunk_flags;
  metadata.dropped_record_count = header->dropped_record_count;

  iree_const_byte_span_t payload =
      iree_make_const_byte_span(transfer->contents.data + payload_offset,
                                (iree_host_size_t)header->payload_length);
  const iree_const_byte_span_t* payloads =
      payload.data_length > 0 ? &payload : NULL;
  iree_host_size_t payload_count = payload.data_length > 0 ? 1 : 0;

  if (!sink) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "remote profile transfer received without an "
                            "active profile sink");
  }
  switch ((iree_hal_remote_profile_callback_type_t)header->callback_type) {
    case IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_BEGIN_SESSION:
      return iree_hal_profile_sink_begin_session(sink, &metadata);
    case IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_WRITE_CHUNK:
      return iree_hal_profile_sink_write(sink, &metadata, payload_count,
                                         payloads);
    case IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_END_SESSION:
      return iree_hal_profile_sink_end_session(
          sink, &metadata, (iree_status_code_t)header->session_status_code);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote profile callback type changed during "
                              "dispatch");
  }
}

iree_status_t iree_hal_remote_client_bulk_profile_receiver_begin_locked(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id,
    uint64_t total_size) {
  if (!device->bulk_session.profile_sink) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "remote profile START received without an active "
                            "profile sink");
  }
  if (total_size < sizeof(iree_hal_remote_profile_transfer_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote profile transfer_id=%" PRIu64
                            " too small for profile header: %" PRIu64 " bytes",
                            transfer_id, total_size);
  }
  if (total_size > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "remote profile transfer_id=%" PRIu64
                            " size %" PRIu64 " exceeds host size max %" PRIhsz,
                            transfer_id, total_size, IREE_HOST_SIZE_MAX);
  }

  iree_hal_remote_client_profile_transfer_t* profile_transfer = NULL;
  iree_status_t status =
      iree_allocator_malloc(device->host_allocator, sizeof(*profile_transfer),
                            (void**)&profile_transfer);
  if (iree_status_is_ok(status)) {
    memset(profile_transfer, 0, sizeof(*profile_transfer));
    profile_transfer->host_allocator = device->host_allocator;
    profile_transfer->transfer_id = transfer_id;
    status = iree_allocator_malloc(device->host_allocator,
                                   (iree_host_size_t)total_size,
                                   (void**)&profile_transfer->contents.data);
  }
  if (iree_status_is_ok(status)) {
    profile_transfer->contents.data_length = (iree_host_size_t)total_size;
    status = iree_hal_remote_bulk_transfer_tracker_initialize(
        total_size, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH,
        device->host_allocator, &profile_transfer->receive_tracker);
  }

  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_transfer_table_insert(
        device->bulk_session.transfers, transfer_id, total_size,
        /*user_value=*/0, &table_transfer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(table_transfer);
    memset(bulk_transfer, 0, sizeof(*bulk_transfer));
    bulk_transfer->kind =
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE;
    bulk_transfer->profile_receive = profile_transfer;
    profile_transfer = NULL;
  }
  iree_hal_remote_client_bulk_profile_receiver_free_transfer(profile_transfer);
  return status;
}

static iree_status_t iree_hal_remote_client_dispatch_profile_transfers(
    iree_net_bulk_channel_t* bulk_channel, iree_hal_profile_sink_t* sink,
    iree_net_sequence_node_t* dispatch_list) {
  iree_status_t transport_status = iree_ok_status();
  while (dispatch_list) {
    iree_net_sequence_node_t* dispatch_node = dispatch_list;
    dispatch_list = dispatch_node->next;
    iree_hal_remote_client_profile_transfer_t* profile_transfer =
        iree_hal_remote_client_profile_transfer_from_sequence_node(
            dispatch_node);
    profile_transfer->sequence_node.next = NULL;

    iree_status_t sink_status =
        iree_hal_remote_client_dispatch_profile_transfer(sink,
                                                         profile_transfer);
    if (iree_status_is_ok(sink_status)) {
      if (bulk_channel) {
        transport_status = iree_net_bulk_channel_send_complete(
            bulk_channel, profile_transfer->transfer_id,
            /*operation_user_data=*/0);
      } else {
        transport_status = iree_make_status(
            IREE_STATUS_UNAVAILABLE, "remote bulk channel is not available");
      }
    } else {
      iree_status_free(sink_status);
      if (bulk_channel) {
        transport_status = iree_net_bulk_channel_send_abort(
            bulk_channel, profile_transfer->transfer_id,
            iree_async_span_list_empty(), /*operation_user_data=*/0);
      } else {
        transport_status = iree_make_status(
            IREE_STATUS_UNAVAILABLE, "remote bulk channel is not available");
      }
    }
    iree_hal_remote_client_bulk_profile_receiver_free_transfer(
        profile_transfer);
    if (!iree_status_is_ok(transport_status)) break;
  }
  iree_hal_remote_client_profile_transfer_list_free(dispatch_list);
  return transport_status;
}

iree_status_t iree_hal_remote_client_bulk_profile_receiver_on_data(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t transfer_id, uint64_t chunk_offset,
    iree_net_bulk_frame_flags_t flags, iree_const_byte_span_t chunk_data,
    bool* out_handled) {
  IREE_ASSERT_ARGUMENT(device);
  if (out_handled) *out_handled = false;

  iree_status_t status = iree_ok_status();
  bool send_credit = false;
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  if (!transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote client bulk DATA unknown transfer_id=%" PRIu64, transfer_id);
    if (out_handled) *out_handled = true;
  } else {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE) {
      iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
      return iree_ok_status();
    }
    if (out_handled) *out_handled = true;

    iree_hal_remote_client_profile_transfer_t* profile_transfer =
        bulk_transfer->profile_receive;
    const uint64_t chunk_length = (uint64_t)chunk_data.data_length;
    const bool chunk_range_overflow = chunk_offset > UINT64_MAX - chunk_length;
    const uint64_t chunk_end =
        chunk_range_overflow ? UINT64_MAX : chunk_offset + chunk_length;
    const bool final_chunk =
        iree_all_bits_set(flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
    const bool expected_final_chunk =
        !chunk_range_overflow &&
        chunk_end == iree_net_bulk_transfer_total_size(transfer);
    if (!profile_transfer) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "remote client profile DATA missing transfer state for "
          "transfer_id=%" PRIu64,
          transfer_id);
    } else if (chunk_range_overflow ||
               chunk_end > iree_net_bulk_transfer_total_size(transfer)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "remote client profile DATA range [%" PRIu64 ", %" PRIu64
          ") exceeds transfer length %" PRIu64,
          chunk_offset, chunk_end, iree_net_bulk_transfer_total_size(transfer));
    } else if (final_chunk != expected_final_chunk) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "remote client profile DATA final flag mismatch "
                           "for transfer_id=%" PRIu64,
                           transfer_id);
    } else {
      status = iree_hal_remote_bulk_transfer_tracker_record_chunk(
          &profile_transfer->receive_tracker, chunk_offset,
          chunk_data.data_length);
      if (iree_status_is_ok(status)) {
        memcpy(profile_transfer->contents.data + (iree_host_size_t)chunk_offset,
               chunk_data.data, chunk_data.data_length);
        send_credit = true;
      }
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);

  if (iree_status_is_ok(status) && send_credit) {
    if (channel) {
      status = iree_net_bulk_channel_send_credit(channel, /*credit_delta=*/1,
                                                 /*operation_user_data=*/0);
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "remote bulk channel is not available");
    }
  }
  return status;
}

iree_status_t iree_hal_remote_client_bulk_profile_receiver_on_complete(
    iree_hal_remote_client_device_t* device, iree_net_bulk_channel_t* channel,
    uint64_t transfer_id, bool* out_handled) {
  IREE_ASSERT_ARGUMENT(device);
  if (out_handled) *out_handled = false;

  iree_status_t status = iree_ok_status();
  iree_net_sequence_node_t* profile_dispatch_list = NULL;
  iree_hal_profile_sink_t* profile_dispatch_sink = NULL;
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  if (!transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote client bulk COMPLETE unknown transfer_id=%" PRIu64,
        transfer_id);
    if (out_handled) *out_handled = true;
  } else {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE) {
      iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
      return iree_ok_status();
    }
    if (out_handled) *out_handled = true;

    iree_hal_remote_client_profile_transfer_t* profile_transfer =
        bulk_transfer->profile_receive;
    bool profile_transfer_transferred = false;
    if (!profile_transfer) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "remote client profile COMPLETE missing transfer state for "
          "transfer_id=%" PRIu64,
          transfer_id);
    } else if (!iree_hal_remote_bulk_transfer_tracker_is_complete(
                   &profile_transfer->receive_tracker)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "remote client profile COMPLETE before all DATA for "
                           "transfer_id=%" PRIu64,
                           transfer_id);
    } else {
      iree_host_size_t content_type_offset = 0;
      iree_host_size_t name_offset = 0;
      iree_host_size_t payload_offset = 0;
      status = iree_hal_remote_client_profile_transfer_parse(
          profile_transfer, &content_type_offset, &name_offset,
          &payload_offset);
    }
    if (iree_status_is_ok(status) &&
        iree_net_sequence_window_has_observed(
            &device->bulk_session.profile_sequence_window,
            profile_transfer->header.sequence)) {
      status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "remote client profile sequence %" PRIu64
                                " was already dispatched",
                                profile_transfer->header.sequence);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_sequence_window_reserve(
          &device->bulk_session.profile_sequence_window,
          profile_transfer->header.sequence);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_sequence_window_observe(
          &device->bulk_session.profile_sequence_window,
          profile_transfer->header.sequence, &profile_dispatch_list);
      if (iree_status_is_ok(status)) {
        profile_transfer =
            iree_hal_remote_client_bulk_profile_receiver_take_transfer(
                device->bulk_session.transfers, transfer);
        transfer = NULL;
        if (profile_transfer->header.sequence <=
            iree_net_sequence_window_observed(
                &device->bulk_session.profile_sequence_window)) {
          profile_transfer->sequence_node.next = profile_dispatch_list;
          profile_transfer->sequence_node.sequence =
              profile_transfer->header.sequence;
          profile_dispatch_list = &profile_transfer->sequence_node;
          profile_transfer_transferred = true;
        } else {
          status = iree_net_sequence_window_defer_until(
              &device->bulk_session.profile_sequence_window,
              profile_transfer->header.sequence,
              &profile_transfer->sequence_node, &profile_dispatch_list);
          if (iree_status_is_ok(status)) profile_transfer_transferred = true;
        }
      }
    }
    if (!iree_status_is_ok(status)) {
      if (!profile_transfer_transferred) {
        if (transfer) {
          iree_hal_remote_client_bulk_profile_receiver_release_transfer(
              device->bulk_session.transfers, transfer);
        } else {
          iree_hal_remote_client_bulk_profile_receiver_free_transfer(
              profile_transfer);
        }
      }
    }
  }
  if (iree_status_is_ok(status) && profile_dispatch_list) {
    profile_dispatch_sink = device->bulk_session.profile_sink;
    iree_hal_profile_sink_retain(profile_dispatch_sink);
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);

  if (iree_status_is_ok(status) && profile_dispatch_list) {
    status = iree_hal_remote_client_dispatch_profile_transfers(
        channel, profile_dispatch_sink, profile_dispatch_list);
  } else if (profile_dispatch_list) {
    iree_status_free(iree_hal_remote_client_dispatch_profile_transfers(
        /*bulk_channel=*/NULL, profile_dispatch_sink, profile_dispatch_list));
  }
  iree_hal_profile_sink_release(profile_dispatch_sink);
  return status;
}

iree_status_t iree_hal_remote_client_bulk_profile_receiver_on_abort(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id,
    bool* out_handled) {
  IREE_ASSERT_ARGUMENT(device);
  if (out_handled) *out_handled = false;

  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  if (transfer) {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind ==
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE) {
      if (out_handled) *out_handled = true;
      iree_hal_remote_client_bulk_profile_receiver_release_transfer(
          device->bulk_session.transfers, transfer);
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  return iree_ok_status();
}
