// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Client bulk transfer table storage.
//
// The client bulk session owns a fixed-capacity transfer table. Each table
// entry carries this owner-managed storage so upload, download, and profile
// paths can share ID lookup/removal without allocating per transfer.

#ifndef IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_H_
#define IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/client/file.h"
#include "iree/hal/remote/protocol/profile.h"
#include "iree/hal/remote/util/bulk_transfer_tracker.h"
#include "iree/net/channel/bulk/transfer_table.h"
#include "iree/net/channel/util/sequence_window.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_bulk_chunk_t iree_net_bulk_chunk_t;

// DATA payloads share the frame with a bulk header; keep chunks comfortably
// below the default queue frame size instead of filling the whole frame.
#define IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH (32 * 1024)

typedef enum iree_hal_remote_client_bulk_transfer_kind_e {
  IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_EMPTY = 0u,
  IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_READ = 1u,
  IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_FILE_WRITE = 2u,
  IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_KIND_PROFILE_RECEIVE = 3u,
} iree_hal_remote_client_bulk_transfer_kind_e;
typedef uint8_t iree_hal_remote_client_bulk_transfer_kind_t;

typedef uint8_t iree_hal_remote_client_file_read_transfer_flags_t;
enum iree_hal_remote_client_file_read_transfer_flag_bits_e {
  IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_CHUNK_IN_FLIGHT = 1u << 0,
  IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_START_SENT = 1u << 1,
  IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_COMPLETE_SENT = 1u << 2,
  IREE_HAL_REMOTE_CLIENT_FILE_READ_TRANSFER_FLAG_TERMINAL = 1u << 3,
};

typedef struct iree_hal_remote_client_file_read_transfer_t {
  // Source HAL file retained for the transfer lifetime.
  iree_hal_file_t* source_file;

  // Resolved source file capabilities captured at queue_read submission.
  iree_hal_remote_client_file_view_t source_file_view;

  // Source file byte offset for the first transfer byte.
  uint64_t source_offset;

  // Total number of bytes sent by this transfer.
  uint64_t total_length;

  // Number of host bytes submitted as DATA frames.
  uint64_t send_offset;

  // Async file read chunk ready to send or awaiting send retry.
  iree_net_bulk_chunk_t* pending_chunk;

  // Next DATA sequence number for this transfer.
  uint32_t next_sequence;

  // Number of bulk START/DATA/COMPLETE frames still awaiting send completion.
  uint32_t pending_send_count;

  // State bits from iree_hal_remote_client_file_read_transfer_flag_bits_e.
  iree_hal_remote_client_file_read_transfer_flags_t flags;
} iree_hal_remote_client_file_read_transfer_t;

typedef uint8_t iree_hal_remote_client_file_write_transfer_flags_t;
enum iree_hal_remote_client_file_write_transfer_flag_bits_e {
  IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_SERVER_COMPLETE = 1u << 0,
  IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_WRITE_FAILED = 1u << 1,
  IREE_HAL_REMOTE_CLIENT_FILE_WRITE_TRANSFER_FLAG_TERMINAL = 1u << 2,
};

typedef void (*iree_hal_remote_client_bulk_completion_fn_t)(
    void* user_data, iree_status_t status);
typedef struct iree_hal_remote_client_bulk_completion_callback_t {
  // Completion callback. Consumes |status|.
  iree_hal_remote_client_bulk_completion_fn_t fn;

  // User data passed to |fn|.
  void* user_data;
} iree_hal_remote_client_bulk_completion_callback_t;

typedef struct iree_hal_remote_client_file_write_transfer_t {
  // Target HAL file retained for the transfer lifetime.
  iree_hal_file_t* target_file;

  // Resolved target file capabilities captured at queue_write submission.
  iree_hal_remote_client_file_view_t target_file_view;

  // Target file byte offset for the first transfer byte.
  uint64_t target_offset;

  // Total number of bytes expected from the server.
  uint64_t total_length;

  // Tracks fixed-grid DATA chunks received from the server.
  iree_hal_remote_bulk_transfer_tracker_t receive_tracker;

  // Number of async file writes still in flight.
  uint32_t pending_write_count;

  // State bits from iree_hal_remote_client_file_write_transfer_flag_bits_e.
  iree_hal_remote_client_file_write_transfer_flags_t flags;

  // Optional completion callback for control-path bulk reads.
  iree_hal_remote_client_bulk_completion_callback_t completion_callback;
} iree_hal_remote_client_file_write_transfer_t;

typedef struct iree_hal_remote_client_profile_transfer_t {
  // Intrusive sequence node used by profile_sequence_window.
  iree_net_sequence_node_t sequence_node;

  // Host allocator used for this transfer and reassembled contents.
  iree_allocator_t host_allocator;

  // Bulk transfer ID used for protocol acknowledgement and active-table lookup.
  uint64_t transfer_id;

  // Reassembled transfer payload bytes.
  iree_byte_span_t contents;

  // Tracks fixed-grid DATA chunks received from the server.
  iree_hal_remote_bulk_transfer_tracker_t receive_tracker;

  // Parsed transfer header copied out of |contents|.
  iree_hal_remote_profile_transfer_header_t header;
} iree_hal_remote_client_profile_transfer_t;

typedef struct iree_hal_remote_client_bulk_transfer_t {
  // Active transfer kind stored in the union below.
  iree_hal_remote_client_bulk_transfer_kind_t kind;

  union {
    // Client-local queue_read upload state.
    iree_hal_remote_client_file_read_transfer_t file_read;

    // Client-local queue_write download state.
    iree_hal_remote_client_file_write_transfer_t file_write;

    // Server-originated profile callback receive state.
    iree_hal_remote_client_profile_transfer_t* profile_receive;
  };
} iree_hal_remote_client_bulk_transfer_t;

typedef struct iree_hal_remote_client_bulk_transfer_id_list_t {
  // Active transfer IDs collected for retry or cancellation.
  uint64_t transfer_ids[IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY];

  // Number of populated entries in |transfer_ids|.
  iree_host_size_t transfer_count;
} iree_hal_remote_client_bulk_transfer_id_list_t;

static inline iree_hal_remote_client_bulk_transfer_t*
iree_hal_remote_client_bulk_transfer_storage(
    iree_net_bulk_transfer_t* transfer) {
  return (iree_hal_remote_client_bulk_transfer_t*)
      iree_net_bulk_transfer_user_storage(transfer)
          .data;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BULK_TRANSFER_H_
