// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/bulk.h"

#include "iree/hal/remote/client/bulk_download_receiver.h"
#include "iree/hal/remote/client/bulk_upload_sender.h"
#include "iree/hal/remote/protocol/profile.h"
#include "iree/hal/remote/util/bulk_transfer_tracker.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/channel/bulk/bulk_channel.h"
#include "iree/net/channel/bulk/chunk_pool.h"
#include "iree/net/channel/bulk/transfer_table.h"

// Header buffers used by the bulk channel frame sender. Bulk DATA payloads are
// not copied into this pool; only the 40-byte frame headers are retained until
// send completion.
#define IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_COUNT 128
#define IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_SIZE 128

static void iree_hal_remote_client_device_on_bulk_transport_error(
    void* user_data, iree_status_t status);

static void iree_hal_remote_client_bulk_release_transfer(
    iree_net_bulk_transfer_table_t* table, iree_net_bulk_transfer_t* transfer);
static iree_hal_remote_client_profile_transfer_t*
iree_hal_remote_client_profile_transfer_from_sequence_node(
    iree_net_sequence_node_t* sequence_node);

static void iree_hal_remote_client_profile_transfer_free(
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
    iree_hal_remote_client_profile_transfer_free(profile_transfer);
    transfer_list = next;
  }
}

static void iree_hal_remote_client_bulk_transfer_deinitialize(
    iree_hal_remote_client_bulk_transfer_t* transfer) {
  switch (transfer->kind) {
    case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ:
      iree_hal_remote_client_bulk_upload_sender_deinitialize_transfer(
          &transfer->file_read);
      break;
    case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE:
      iree_hal_remote_client_bulk_download_receiver_deinitialize_transfer(
          &transfer->file_write);
      break;
    case IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE:
      iree_hal_remote_client_profile_transfer_free(transfer->profile_receive);
      break;
    default:
      break;
  }
  memset(transfer, 0, sizeof(*transfer));
}

static void
iree_hal_remote_client_bulk_release_pending_profile_transfers_locked(
    iree_hal_remote_client_device_t* device) {
  iree_net_sequence_node_t* pending_list = NULL;
  iree_net_sequence_window_take_pending(
      &device->bulk_session.profile_sequence_window, &pending_list);
  while (pending_list) {
    iree_net_sequence_node_t* next = pending_list->next;
    iree_hal_remote_client_profile_transfer_t* profile_transfer =
        iree_hal_remote_client_profile_transfer_from_sequence_node(
            pending_list);
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(device->bulk_session.transfers,
                                            profile_transfer->transfer_id);
    if (table_transfer) {
      iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
          iree_hal_remote_client_bulk_transfer_storage(table_transfer);
      if (bulk_transfer->kind ==
              IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE &&
          bulk_transfer->profile_receive == profile_transfer) {
        bulk_transfer->profile_receive = NULL;
        iree_hal_remote_client_bulk_release_transfer(
            device->bulk_session.transfers, table_transfer);
      }
    }
    iree_hal_remote_client_profile_transfer_free(profile_transfer);
    pending_list = next;
  }
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

static void iree_hal_remote_client_bulk_release_all_profile_transfers_locked(
    iree_hal_remote_client_device_t* device) {
  iree_hal_remote_client_bulk_release_pending_profile_transfers_locked(device);
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
      iree_hal_remote_client_bulk_release_transfer(
          device->bulk_session.transfers, table_transfer);
    }
  }
}

static void iree_hal_remote_client_bulk_release_transfer(
    iree_net_bulk_transfer_table_t* table, iree_net_bulk_transfer_t* transfer) {
  uint64_t transfer_id = iree_net_bulk_transfer_id(transfer);
  iree_hal_remote_client_bulk_transfer_deinitialize(
      iree_hal_remote_client_bulk_transfer_storage(transfer));
  iree_net_bulk_transfer_table_remove(table, transfer_id);
}

static void iree_hal_remote_client_bulk_deinitialize_transfer(
    void* user_data, iree_net_bulk_transfer_t* transfer) {
  (void)user_data;
  iree_hal_remote_client_bulk_transfer_deinitialize(
      iree_hal_remote_client_bulk_transfer_storage(transfer));
}

iree_status_t iree_hal_remote_client_device_initialize_bulk_transfers(
    iree_hal_remote_client_device_t* device) {
  iree_hal_remote_client_bulk_session_transfer_options_t options =
      iree_hal_remote_client_bulk_session_transfer_options_default();
  options.transfer_user_storage_size =
      sizeof(iree_hal_remote_client_bulk_transfer_t);
  options.transfer_user_storage_alignment =
      iree_alignof(iree_hal_remote_client_bulk_transfer_t);
  options.chunk_user_storage_size = iree_max(
      iree_hal_remote_client_bulk_upload_sender_chunk_storage_size(),
      iree_hal_remote_client_bulk_download_receiver_chunk_storage_size());
  options.chunk_user_storage_alignment = iree_max(
      iree_hal_remote_client_bulk_upload_sender_chunk_storage_alignment(),
      iree_hal_remote_client_bulk_download_receiver_chunk_storage_alignment());
  return iree_hal_remote_client_bulk_session_initialize_transfers(
      &device->bulk_session, &options, device->host_allocator);
}

static void iree_hal_remote_client_bulk_deinitialize_transfers_locked(
    void* user_data, iree_net_bulk_transfer_table_t* transfers) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_hal_remote_client_bulk_release_all_profile_transfers_locked(device);
  if (transfers) {
    iree_net_bulk_transfer_table_visit(
        transfers, iree_hal_remote_client_bulk_deinitialize_transfer, NULL);
    iree_net_bulk_transfer_table_clear(transfers);
  }
}

void iree_hal_remote_client_device_deinitialize_bulk_transfers(
    iree_hal_remote_client_device_t* device) {
  iree_hal_remote_client_bulk_session_deinitialize_transfers(
      &device->bulk_session,
      iree_hal_remote_client_bulk_deinitialize_transfers_locked, device);
}

void iree_hal_remote_client_bulk_cancel_transfer(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  if (!transfer_id) return;
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  if (transfer) {
    iree_hal_remote_client_bulk_release_transfer(device->bulk_session.transfers,
                                                 transfer);
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
}

iree_status_t iree_hal_remote_client_bulk_upload_file_read(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  iree_status_t status =
      iree_hal_remote_client_bulk_upload_sender_upload(device, transfer_id);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_on_bulk_transport_error(
        device, iree_status_clone(status));
  }
  return status;
}

iree_status_t iree_hal_remote_client_bulk_upload_buffer_unmap(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  return iree_hal_remote_client_bulk_upload_file_read(device, transfer_id);
}

void iree_hal_remote_client_bulk_end_buffer_unmap(
    iree_hal_remote_client_device_t* device, uint64_t transfer_id) {
  iree_hal_remote_client_bulk_cancel_transfer(device, transfer_id);
}

iree_status_t iree_hal_remote_client_bulk_begin_profile_session(
    iree_hal_remote_client_device_t* device, iree_hal_profile_sink_t* sink) {
  IREE_ASSERT_ARGUMENT(device);

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  if (device->bulk_session.profile_sink) {
    status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "remote profiling session already active");
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_bulk_release_all_profile_transfers_locked(device);
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
  iree_hal_remote_client_bulk_release_all_profile_transfers_locked(device);
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  iree_hal_profile_sink_release(sink);
}

typedef struct iree_hal_remote_client_bulk_endpoint_context_t {
  // Owning device receiving the bulk channel.
  iree_hal_remote_client_device_t* device;

  // Queue channel created before bulk endpoint provisioning. Ownership is
  // transferred to the device only after the bulk channel is ready.
  iree_net_queue_channel_t* queue_channel;
} iree_hal_remote_client_bulk_endpoint_context_t;

static iree_hal_remote_client_profile_transfer_t*
iree_hal_remote_client_profile_transfer_from_sequence_node(
    iree_net_sequence_node_t* sequence_node) {
  return iree_containerof(
      sequence_node, iree_hal_remote_client_profile_transfer_t, sequence_node);
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
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_profile_transfer_header_t), &required_length,
      IREE_STRUCT_FIELD_ALIGNED(header->content_type_length, char, 8,
                                out_content_type_offset),
      IREE_STRUCT_FIELD_ALIGNED(header->name_length, char, 8, out_name_offset),
      IREE_STRUCT_FIELD((iree_host_size_t)header->payload_length, uint8_t,
                        out_payload_offset));
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

static iree_status_t iree_hal_remote_client_begin_profile_receive_locked(
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
  iree_hal_remote_client_profile_transfer_free(profile_transfer);
  return status;
}

static iree_status_t iree_hal_remote_client_collect_ready_profile_transfers(
    iree_hal_remote_client_device_t* device,
    iree_net_sequence_node_t* ready_list,
    iree_net_sequence_node_t** out_dispatch_list) {
  *out_dispatch_list = NULL;
  iree_net_sequence_node_t** dispatch_tail = out_dispatch_list;
  iree_status_t status = iree_ok_status();
  while (ready_list && iree_status_is_ok(status)) {
    iree_net_sequence_node_t* next = ready_list->next;
    iree_hal_remote_client_profile_transfer_t* profile_transfer =
        iree_hal_remote_client_profile_transfer_from_sequence_node(ready_list);
    bool profile_transfer_transferred = false;
    iree_net_bulk_transfer_t* table_transfer =
        iree_net_bulk_transfer_table_lookup(device->bulk_session.transfers,
                                            profile_transfer->transfer_id);
    if (!table_transfer) {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "ready remote profile transfer_id=%" PRIu64
                                " is missing",
                                profile_transfer->transfer_id);
    } else {
      iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
          iree_hal_remote_client_bulk_transfer_storage(table_transfer);
      if (bulk_transfer->kind !=
              IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE ||
          bulk_transfer->profile_receive != profile_transfer) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "ready remote profile transfer_id=%" PRIu64
                                  " does not match profile receive state",
                                  profile_transfer->transfer_id);
      } else {
        bulk_transfer->profile_receive = NULL;
        iree_hal_remote_client_bulk_release_transfer(
            device->bulk_session.transfers, table_transfer);
        profile_transfer->sequence_node.next = NULL;
        *dispatch_tail = &profile_transfer->sequence_node;
        dispatch_tail = &profile_transfer->sequence_node.next;
        profile_transfer_transferred = true;
      }
    }
    if (!iree_status_is_ok(status) && !profile_transfer_transferred) {
      profile_transfer->sequence_node.next = NULL;
      iree_hal_remote_client_profile_transfer_free(profile_transfer);
    }
    ready_list = next;
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_profile_transfer_list_free(ready_list);
    iree_hal_remote_client_profile_transfer_list_free(*out_dispatch_list);
    *out_dispatch_list = NULL;
  }
  return status;
}

static iree_status_t iree_hal_remote_client_dispatch_profile_transfers(
    iree_hal_remote_client_device_t* device,
    iree_net_bulk_channel_t* bulk_channel, iree_hal_profile_sink_t* sink,
    iree_net_sequence_node_t* dispatch_list) {
  (void)device;
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
      iree_status_ignore(sink_status);
      if (bulk_channel) {
        transport_status = iree_net_bulk_channel_send_abort(
            bulk_channel, profile_transfer->transfer_id,
            iree_async_span_list_empty(), /*operation_user_data=*/0);
      } else {
        transport_status = iree_make_status(
            IREE_STATUS_UNAVAILABLE, "remote bulk channel is not available");
      }
    }
    iree_hal_remote_client_profile_transfer_free(profile_transfer);
    if (!iree_status_is_ok(transport_status)) break;
  }
  iree_hal_remote_client_profile_transfer_list_free(dispatch_list);
  return transport_status;
}

static iree_status_t iree_hal_remote_client_device_on_bulk_start(
    void* user_data, uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  if (flags != IREE_NET_BULK_FRAME_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported remote client bulk START flags: 0x%02x", flags);
  }

  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  iree_status_t status = iree_ok_status();
  if (!transfer) {
    status = iree_hal_remote_client_begin_profile_receive_locked(
        device, transfer_id, total_size);
  } else if (iree_net_bulk_transfer_total_size(transfer) != total_size) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "remote client bulk START size mismatch for transfer_id=%" PRIu64,
        transfer_id);
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind ==
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE) {
      iree_hal_remote_client_bulk_download_receiver_fail_start_locked(
          &bulk_transfer->file_write, iree_status_clone(status));
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  if (iree_status_is_ok(status)) {
    iree_net_bulk_channel_t* bulk_channel =
        iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
    if (bulk_channel) {
      status = iree_net_bulk_channel_refresh_credit(bulk_channel,
                                                    /*operation_user_data=*/0);
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_client_device_on_bulk_data(
    void* user_data, uint64_t transfer_id, uint64_t chunk_offset,
    uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  if (flags & ~IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported remote client bulk DATA flags: 0x%02x",
                            flags);
  }

  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  bool handled = false;
  iree_status_t status = iree_hal_remote_client_bulk_download_receiver_on_data(
      device, bulk_channel, transfer_id, chunk_offset, sequence, flags,
      chunk_data, lease, &handled);
  if (!iree_status_is_ok(status) || handled) return status;

  bool send_credit = false;
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_net_bulk_transfer_t* transfer = iree_net_bulk_transfer_table_lookup(
      device->bulk_session.transfers, transfer_id);
  if (!transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote client bulk DATA unknown transfer_id=%" PRIu64, transfer_id);
  } else {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind ==
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE) {
      iree_hal_remote_client_profile_transfer_t* profile_transfer =
          bulk_transfer->profile_receive;
      const uint64_t chunk_length = (uint64_t)chunk_data.data_length;
      const bool chunk_range_overflow =
          chunk_offset > UINT64_MAX - chunk_length;
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
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "remote client profile DATA range [%" PRIu64
                             ", %" PRIu64 ") exceeds transfer length %" PRIu64,
                             chunk_offset, chunk_end,
                             iree_net_bulk_transfer_total_size(transfer));
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
          memcpy(
              profile_transfer->contents.data + (iree_host_size_t)chunk_offset,
              chunk_data.data, chunk_data.data_length);
          send_credit = true;
        }
      }
    } else {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "remote client bulk DATA received for "
                                "unsupported transfer kind %u "
                                "transfer_id=%" PRIu64,
                                bulk_transfer->kind, transfer_id);
    }
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);

  if (iree_status_is_ok(status) && send_credit) {
    if (bulk_channel) {
      status = iree_net_bulk_channel_send_credit(
          bulk_channel, /*credit_delta=*/1, /*operation_user_data=*/0);
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "remote bulk channel is not available");
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_client_device_on_bulk_complete(
    void* user_data, uint64_t transfer_id) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);

  bool handled = false;
  iree_status_t status =
      iree_hal_remote_client_bulk_download_receiver_on_complete(
          device, bulk_channel, transfer_id, &handled);
  if (!iree_status_is_ok(status) || handled) return status;

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
  } else {
    iree_hal_remote_client_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_client_bulk_transfer_storage(transfer);
    if (bulk_transfer->kind ==
        IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE) {
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
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
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
        iree_net_sequence_node_t* ready_list = NULL;
        status = iree_net_sequence_window_observe(
            &device->bulk_session.profile_sequence_window,
            profile_transfer->header.sequence, &ready_list);
        if (iree_status_is_ok(status)) {
          if (profile_transfer->header.sequence <=
              iree_net_sequence_window_observed(
                  &device->bulk_session.profile_sequence_window)) {
            profile_transfer->sequence_node.next = ready_list;
            profile_transfer->sequence_node.sequence =
                profile_transfer->header.sequence;
            ready_list = &profile_transfer->sequence_node;
            profile_transfer_transferred = true;
          } else {
            status = iree_net_sequence_window_defer_until(
                &device->bulk_session.profile_sequence_window,
                profile_transfer->header.sequence,
                &profile_transfer->sequence_node, &ready_list);
            if (iree_status_is_ok(status)) profile_transfer_transferred = true;
          }
        }
        if (iree_status_is_ok(status)) {
          status = iree_hal_remote_client_collect_ready_profile_transfers(
              device, ready_list, &profile_dispatch_list);
        }
      }
      if (!iree_status_is_ok(status)) {
        if (!profile_transfer_transferred) {
          iree_hal_remote_client_bulk_release_transfer(
              device->bulk_session.transfers, transfer);
          transfer = NULL;
        }
      }
    } else {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "remote client bulk COMPLETE received for "
                                "unsupported transfer kind %u "
                                "transfer_id=%" PRIu64,
                                bulk_transfer->kind, transfer_id);
    }
  }
  if (iree_status_is_ok(status) && profile_dispatch_list) {
    profile_dispatch_sink = device->bulk_session.profile_sink;
    iree_hal_profile_sink_retain(profile_dispatch_sink);
  }
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);

  if (iree_status_is_ok(status) && profile_dispatch_list) {
    status = iree_hal_remote_client_dispatch_profile_transfers(
        device, bulk_channel, profile_dispatch_sink, profile_dispatch_list);
  } else if (profile_dispatch_list) {
    iree_status_ignore(iree_hal_remote_client_dispatch_profile_transfers(
        device, /*bulk_channel=*/NULL, profile_dispatch_sink,
        profile_dispatch_list));
  }
  iree_hal_profile_sink_release(profile_dispatch_sink);
  return status;
}

static iree_status_t iree_hal_remote_client_device_on_bulk_abort(
    void* user_data, uint64_t transfer_id, iree_const_byte_span_t abort_data,
    iree_async_buffer_lease_t* lease) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  (void)abort_data;
  (void)lease;

  bool handled = false;
  iree_status_t status = iree_hal_remote_client_bulk_download_receiver_on_abort(
      device, transfer_id, &handled);
  if (!iree_status_is_ok(status) || handled) return status;

  iree_hal_remote_client_bulk_cancel_transfer(device, transfer_id);
  return iree_ok_status();
}

static void iree_hal_remote_client_device_on_bulk_transport_error(
    void* user_data, iree_status_t status) {
  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_hal_remote_client_device_notify_bulk_transport_error(device, status);
}

static void iree_hal_remote_client_device_on_bulk_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status) {
  if (operation_user_data == 0) {
    iree_status_ignore(status);
    return;
  }

  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  iree_status_t transport_status =
      iree_hal_remote_client_bulk_upload_sender_send_complete(
          device, bulk_channel, operation_user_data, status);
  if (!iree_status_is_ok(transport_status)) {
    iree_hal_remote_client_device_on_bulk_transport_error(user_data,
                                                          transport_status);
  }
}

static void iree_hal_remote_client_device_on_bulk_credit(
    void* user_data, uint32_t credit_delta, uint32_t available_credit_count) {
  (void)credit_delta;
  if (available_credit_count == 0) return;

  iree_hal_remote_client_device_t* device =
      (iree_hal_remote_client_device_t*)user_data;
  iree_net_bulk_channel_t* bulk_channel =
      iree_hal_remote_client_bulk_session_load_channel(&device->bulk_session);
  iree_slim_mutex_lock(&device->bulk_session.transfer_mutex);
  iree_status_t status =
      iree_hal_remote_client_bulk_upload_sender_try_send_all_locked(
          device, bulk_channel);
  iree_slim_mutex_unlock(&device->bulk_session.transfer_mutex);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_on_bulk_transport_error(user_data, status);
  }
}

static void iree_hal_remote_client_device_on_bulk_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_hal_remote_client_bulk_endpoint_context_t* context =
      (iree_hal_remote_client_bulk_endpoint_context_t*)user_data;
  iree_hal_remote_client_device_t* device = context->device;
  iree_net_queue_channel_t* queue_channel = context->queue_channel;
  iree_allocator_t host_allocator = device->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, context);
  context = NULL;

  iree_async_buffer_pool_t* header_pool = NULL;
  iree_net_bulk_channel_t* bulk_channel = NULL;
  if (iree_hal_remote_client_device_load_state(device) !=
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING) {
    iree_status_ignore(status);
    iree_net_queue_channel_release(queue_channel);
  } else {
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_create_queue_header_pool(
          IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_COUNT,
          IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_SIZE, host_allocator,
          &header_pool);
    }
    if (iree_status_is_ok(status)) {
      iree_net_bulk_channel_callbacks_t callbacks = {
          .on_start = iree_hal_remote_client_device_on_bulk_start,
          .on_data = iree_hal_remote_client_device_on_bulk_data,
          .on_complete = iree_hal_remote_client_device_on_bulk_complete,
          .on_abort = iree_hal_remote_client_device_on_bulk_abort,
          .on_transport_error =
              iree_hal_remote_client_device_on_bulk_transport_error,
          .on_send_complete =
              iree_hal_remote_client_device_on_bulk_send_complete,
          .on_credit = iree_hal_remote_client_device_on_bulk_credit,
          .user_data = device,
      };
      iree_async_buffer_pool_t* channel_header_pool = header_pool;
      header_pool = NULL;  // Ownership transfers to create, including failure.
      status = iree_net_bulk_channel_create(endpoint, /*options=*/NULL,
                                            channel_header_pool, callbacks,
                                            host_allocator, &bulk_channel);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_channel_activate(bulk_channel);
    }
    if (iree_status_is_ok(status)) {
      const uint32_t initial_chunk_credit =
          (uint32_t)iree_net_bulk_chunk_pool_capacity(
              device->bulk_session.receive_chunks);
      status = iree_net_bulk_channel_send_credit(
          bulk_channel, initial_chunk_credit, /*operation_user_data=*/0);
    }

    if (iree_status_is_ok(status)) {
      iree_net_queue_channel_t* old_queue_channel =
          iree_hal_remote_client_device_publish_queue_channel(device,
                                                              queue_channel);
      queue_channel = NULL;
      iree_net_queue_channel_detach(old_queue_channel);
      iree_net_queue_channel_release(old_queue_channel);

      iree_net_bulk_channel_t* old_bulk_channel =
          iree_hal_remote_client_device_publish_bulk_channel(device,
                                                             bulk_channel);
      bulk_channel = NULL;
      iree_net_bulk_channel_detach(old_bulk_channel);
      iree_net_bulk_channel_release(old_bulk_channel);

      iree_hal_remote_client_device_store_state(
          device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);
      iree_hal_remote_client_device_complete_connect(device, iree_ok_status());
    } else {
      iree_net_bulk_channel_release(bulk_channel);
      iree_async_buffer_pool_free(header_pool);
      iree_net_queue_channel_release(queue_channel);
      iree_hal_remote_client_device_store_state(
          device, IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR);
      iree_hal_remote_client_device_complete_connect(device, status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_remote_client_device_open_bulk_endpoint(
    iree_hal_remote_client_device_t* device,
    iree_net_queue_channel_t* queue_channel) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(queue_channel);

  iree_hal_remote_client_bulk_endpoint_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      device->host_allocator, sizeof(*context), (void**)&context));
  context->device = device;
  context->queue_channel = queue_channel;

  iree_net_endpoint_ready_callback_t endpoint_callback = {
      .fn = iree_hal_remote_client_device_on_bulk_endpoint_ready,
      .user_data = context,
  };
  iree_status_t status =
      iree_net_session_open_endpoint(device->session, endpoint_callback);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(device->host_allocator, context);
  }
  return status;
}
