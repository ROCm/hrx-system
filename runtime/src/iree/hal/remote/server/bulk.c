// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk.h"

#include "iree/async/semaphore.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/protocol/profile.h"
#include "iree/hal/remote/server/profile.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/hal/remote/util/bulk_channel_writer.h"
#include "iree/hal/remote/util/bulk_transfer_scheduler.h"
#include "iree/hal/remote/util/bulk_transfer_tracker.h"
#include "iree/io/file_handle.h"
#include "iree/net/channel/bulk/chunk_pool.h"
#include "iree/net/channel/bulk/receive_window.h"

// DATA payloads share the frame with a bulk header; keep chunks comfortably
// below the default queue frame size instead of filling the whole frame.
#define IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH (32 * 1024)
#define IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY \
  IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY
#define IREE_HAL_REMOTE_BULK_STAGING_SLOT_COUNT \
  IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY

// Host allocation wrapped as a HAL memory file for local queue_write staging.
typedef struct iree_hal_remote_server_bulk_host_allocation_t {
  // Host allocator used to free this allocation.
  iree_allocator_t host_allocator;

  // Byte-addressable file contents.
  uint8_t data[];
} iree_hal_remote_server_bulk_host_allocation_t;

typedef struct iree_hal_remote_server_client_file_read_ready_t
    iree_hal_remote_server_client_file_read_ready_t;
typedef struct iree_hal_remote_server_bulk_staging_pool_t
    iree_hal_remote_server_bulk_staging_pool_t;
typedef struct iree_hal_remote_server_bulk_staging_slot_t
    iree_hal_remote_server_bulk_staging_slot_t;

typedef uint8_t iree_hal_remote_server_bulk_staging_slot_flags_t;
enum iree_hal_remote_server_bulk_staging_slot_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_BULK_STAGING_SLOT_FLAG_IN_USE = 1u << 0,
};

typedef uint8_t iree_hal_remote_server_client_file_read_transfer_flags_t;
enum iree_hal_remote_server_client_file_read_transfer_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_START_RECEIVED = 1u
                                                                         << 0,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_PEER_COMPLETE = 1u << 1,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_COMMAND_READY = 1u << 2,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_READY_PENDING = 1u << 3,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_READY_COMPLETE = 1u
                                                                         << 4,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_SIGNAL_CONSUMED = 1u
                                                                          << 5,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_CONTROL_RESPONSE = 1u
                                                                           << 6,
};

typedef struct iree_hal_remote_server_client_file_read_transfer_t {
  // Server retained while transfer state may outlive the session lock.
  iree_hal_remote_server_t* server;

  // Session slot that owns this transfer scheduler entry.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| while callbacks are active.
  uint64_t session_id;

  // Bulk transfer ID used for retained receive chunk cleanup.
  uint64_t transfer_id;

  // Local HAL device borrowed from |server|.
  iree_hal_device_t* local_device;

  // Target buffer retained until all local queue reads finish.
  iree_hal_buffer_t* target_buffer;

  // Target buffer byte offset for the first uploaded byte.
  iree_device_size_t target_offset;

  // Signal semaphore list cloned from the remote command.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Control response envelope used by BUFFER_UNMAP bulk uploads.
  iree_hal_remote_control_envelope_t response_envelope;

  // Internal semaphore signaled after the original command waits resolve.
  iree_hal_semaphore_t* ready_semaphore;

  // Reusable timepoint context for the readiness barrier completion.
  iree_hal_remote_server_client_file_read_ready_t* ready_context;

  // Tracks fixed-grid DATA chunks received from the client.
  iree_hal_remote_bulk_transfer_tracker_t receive_tracker;

  // Number of active local barrier/read callbacks referencing this state.
  uint32_t pending_operation_count;

  // State bits from
  // iree_hal_remote_server_client_file_read_transfer_flag_bits_e.
  iree_hal_remote_server_client_file_read_transfer_flags_t flags;
} iree_hal_remote_server_client_file_read_transfer_t;

typedef struct iree_hal_remote_server_client_file_write_ready_t
    iree_hal_remote_server_client_file_write_ready_t;

typedef uint16_t iree_hal_remote_server_client_file_write_transfer_flags_t;
enum iree_hal_remote_server_client_file_write_transfer_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_START_SENT = 1u << 0,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SEND_PENDING = 1u << 1,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_WRITE_PENDING =
      1u << 2,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SUBMIT_PENDING =
      1u << 3,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_DATA_READY =
      1u << 4,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SEND_PENDING =
      1u << 5,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_COMPLETE_SENT = 1u
                                                                         << 6,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_PEER_COMPLETE = 1u
                                                                         << 7,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SIGNAL_CONSUMED = 1u
                                                                           << 8,
  IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_INITIAL_WAIT_CONSUMED =
      1u << 9,
};

typedef uint8_t iree_hal_remote_server_profile_transfer_flags_t;
enum iree_hal_remote_server_profile_transfer_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_START_SENT = 1u << 0,
  IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING = 1u << 1,
  IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_COMPLETE_SENT = 1u << 2,
  IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_PEER_COMPLETE = 1u << 3,
};

typedef struct iree_hal_remote_server_client_file_write_transfer_t {
  // Server retained while callbacks may reference the session array.
  iree_hal_remote_server_t* server;

  // Session slot that owns this transfer scheduler entry.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| while callbacks are active.
  uint64_t session_id;

  // Local HAL device borrowed from |server| for staging queue writes.
  iree_hal_device_t* local_device;

  // Source buffer retained until all chunks have been staged.
  iree_hal_buffer_t* source_buffer;

  // Source buffer byte offset for the first streamed byte.
  iree_device_size_t source_offset;

  // Queue write flags provided by the remote command.
  iree_hal_write_flags_t write_flags;

  // Session staging pool borrowed while |staging_slot| is acquired.
  iree_hal_remote_server_bulk_staging_pool_t* staging_pool;

  // Acquired staging slot returned to |staging_pool| at transfer teardown.
  iree_hal_remote_server_bulk_staging_slot_t* staging_slot;

  // Server-side memory file borrowed from |staging_slot|.
  iree_hal_file_t* staging_file;

  // Host allocation contents borrowed from |staging_slot|.
  iree_byte_span_t staging_contents;

  // Local semaphore borrowed from |staging_slot|.
  iree_hal_semaphore_t* staging_semaphore;

  // Reusable timepoint context for staging queue write completion.
  iree_hal_remote_server_client_file_write_ready_t* ready_context;

  // Initial local wait semaphore list cloned from the remote command.
  iree_hal_semaphore_list_t initial_wait_semaphore_list;

  // Final local signal semaphore list cloned from the remote command.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Transfer-relative byte offset currently resident in |staging_contents|.
  uint64_t staging_offset;

  // Number of bytes currently resident in |staging_contents|.
  iree_device_size_t staging_length;

  // Transfer-relative byte offset for the next staging queue write.
  uint64_t next_staging_offset;

  // Last payload value signaled on |staging_semaphore|.
  uint64_t last_staging_signal_value;

  // Next DATA sequence number for this transfer.
  uint32_t next_sequence;

  // Number of active callbacks or unlocked submissions referencing this state.
  uint32_t pending_operation_count;

  // State bits from
  // iree_hal_remote_server_client_file_write_transfer_flag_bits_e.
  iree_hal_remote_server_client_file_write_transfer_flags_t flags;
} iree_hal_remote_server_client_file_write_transfer_t;

typedef struct iree_hal_remote_server_profile_transfer_t {
  // Server retained while callbacks may reference the session array.
  iree_hal_remote_server_t* server;

  // Session slot that owns this transfer scheduler entry.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| while callbacks are active.
  uint64_t session_id;

  // Retained profile callback payload bytes.
  iree_byte_span_t payload;

  // Profile callback sequence number carried by |payload|.
  uint64_t sequence;

  // Transfer-relative byte offset for the next DATA frame.
  uint64_t send_offset;

  // Next DATA sequence number for this transfer.
  uint32_t next_sequence;

  // State bits from iree_hal_remote_server_profile_transfer_flag_bits_e.
  iree_hal_remote_server_profile_transfer_flags_t flags;
} iree_hal_remote_server_profile_transfer_t;

struct iree_hal_remote_server_profile_pending_transfer_t {
  // Next queued profile transfer in FIFO order.
  iree_hal_remote_server_profile_pending_transfer_t* next;

  // Session ID expected in |session_slot| when the transfer is activated.
  uint64_t session_id;

  // Profile callback sequence number carried by |payload|.
  uint64_t sequence;

  // Retained profile callback payload bytes.
  iree_byte_span_t payload;
};

typedef enum iree_hal_remote_server_bulk_transfer_kind_e {
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_EMPTY = 0u,
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ = 1u,
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE = 2u,
  IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND = 3u,
} iree_hal_remote_server_bulk_transfer_kind_e;
typedef uint8_t iree_hal_remote_server_bulk_transfer_kind_t;

typedef struct iree_hal_remote_server_bulk_transfer_t {
  // Active transfer kind stored in the union below.
  iree_hal_remote_server_bulk_transfer_kind_t kind;

  union {
    // Client-file queue_read upload state.
    iree_hal_remote_server_client_file_read_transfer_t client_file_read;

    // Client-file queue_write download state.
    iree_hal_remote_server_client_file_write_transfer_t client_file_write;

    // Server-originated profile callback transfer state.
    iree_hal_remote_server_profile_transfer_t profile_send;
  };
} iree_hal_remote_server_bulk_transfer_t;

struct iree_hal_remote_server_client_file_write_ready_t {
  // Reference count covering the transfer state and active timepoint callback.
  iree_atomic_ref_count_t ref_count;

  // Timepoint registered on the local queue_write signal semaphore.
  iree_async_semaphore_timepoint_t timepoint;

  // Server retained while the timepoint may fire.
  iree_hal_remote_server_t* server;

  // Session slot that owned the transfer when submitted.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| when the timepoint fires.
  uint64_t session_id;

  // Bulk transfer ID to resume after local queue_write completion.
  uint64_t transfer_id;

  // Local queue_write signal semaphore retained for the timepoint.
  iree_hal_semaphore_t* local_semaphore;

  // Host allocator used to free this context.
  iree_allocator_t host_allocator;
};

struct iree_hal_remote_server_client_file_read_ready_t {
  // Reference count covering the transfer state and active timepoint callback.
  iree_atomic_ref_count_t ref_count;

  // Timepoint registered on the local readiness barrier signal semaphore.
  iree_async_semaphore_timepoint_t timepoint;

  // Server retained while the timepoint may fire.
  iree_hal_remote_server_t* server;

  // Session slot that owned the transfer when submitted.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| when the timepoint fires.
  uint64_t session_id;

  // Bulk transfer ID to resume after barrier completion.
  uint64_t transfer_id;

  // Internal local semaphore retained for the timepoint.
  iree_hal_semaphore_t* local_semaphore;

  // Host allocator used to free this context.
  iree_allocator_t host_allocator;
};

static iree_hal_remote_server_bulk_transfer_t*
iree_hal_remote_server_bulk_transfer_storage(
    iree_net_bulk_transfer_t* transfer) {
  return (iree_hal_remote_server_bulk_transfer_t*)
      iree_net_bulk_transfer_user_storage(transfer)
          .data;
}

static iree_hal_remote_server_client_file_read_transfer_t*
iree_hal_remote_server_client_file_read_storage(
    iree_net_bulk_transfer_t* transfer) {
  return &iree_hal_remote_server_bulk_transfer_storage(transfer)
              ->client_file_read;
}

static iree_hal_remote_server_client_file_write_transfer_t*
iree_hal_remote_server_client_file_write_storage(
    iree_net_bulk_transfer_t* transfer) {
  return &iree_hal_remote_server_bulk_transfer_storage(transfer)
              ->client_file_write;
}

static void iree_hal_remote_server_bulk_host_allocation_release(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  (void)user_data;
  IREE_ASSERT(handle_primitive.type ==
              IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION);
  iree_byte_span_t host_allocation = handle_primitive.value.host_allocation;
  if (!host_allocation.data) return;
  iree_hal_remote_server_bulk_host_allocation_t* allocation =
      (iree_hal_remote_server_bulk_host_allocation_t*)(host_allocation.data -
                                                       sizeof(*allocation));
  iree_allocator_t host_allocator = allocation->host_allocator;
  iree_allocator_free(host_allocator, allocation);
}

static void iree_hal_remote_server_bulk_host_contents_free(
    iree_byte_span_t host_contents) {
  if (!host_contents.data) return;
  iree_hal_remote_server_bulk_host_allocation_t* allocation =
      (iree_hal_remote_server_bulk_host_allocation_t*)(host_contents.data -
                                                       sizeof(*allocation));
  iree_allocator_t host_allocator = allocation->host_allocator;
  iree_allocator_free(host_allocator, allocation);
}

static iree_status_t iree_hal_remote_server_bulk_host_contents_allocate(
    iree_host_size_t allocation_size, iree_allocator_t host_allocator,
    iree_byte_span_t* out_host_contents) {
  *out_host_contents = iree_byte_span_empty();

  iree_host_size_t total_size = 0;
  iree_host_size_t data_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_server_bulk_host_allocation_t), &total_size,
      IREE_STRUCT_FIELD(allocation_size, uint8_t, &data_offset));

  iree_hal_remote_server_bulk_host_allocation_t* allocation = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&allocation);
  }

  if (iree_status_is_ok(status)) {
    allocation->host_allocator = host_allocator;
    iree_byte_span_t host_contents = iree_make_byte_span(
        (uint8_t*)allocation + data_offset, allocation_size);
    *out_host_contents = host_contents;
  }
  return status;
}

static iree_status_t iree_hal_remote_server_bulk_host_allocation_create(
    iree_host_size_t allocation_size, iree_allocator_t host_allocator,
    iree_byte_span_t* out_host_contents,
    iree_io_file_handle_t** out_file_handle) {
  *out_file_handle = NULL;
  iree_status_t status = iree_hal_remote_server_bulk_host_contents_allocate(
      allocation_size, host_allocator, out_host_contents);

  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_hal_remote_server_bulk_host_allocation_release,
      .user_data = NULL,
  };
  if (iree_status_is_ok(status)) {
    status = iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        *out_host_contents, release_callback, host_allocator, out_file_handle);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_host_contents_free(*out_host_contents);
    *out_host_contents = iree_byte_span_empty();
  }
  return status;
}

struct iree_hal_remote_server_bulk_staging_slot_t {
  // Owning staging pool retained while callbacks may reference this slot.
  iree_hal_remote_server_bulk_staging_pool_t* pool;

  // Local device this slot's HAL file and semaphore are bound to.
  iree_hal_device_t* local_device;

  // Imported HAL memory file wrapping |file_handle| for |local_device|.
  iree_hal_file_t* file;

  // Local semaphore sequencing queue writes into |file|.
  iree_hal_semaphore_t* semaphore;

  // Host allocation file handle for |contents|.
  iree_io_file_handle_t* file_handle;

  // Host bytes exposed through |file_handle|.
  iree_byte_span_t contents;

  // Last payload value signaled on |semaphore|.
  uint64_t last_signal_value;

  // Payload value for the active callback timepoint.
  uint64_t callback_signal_value;

  // Timepoint registered on |semaphore| for a local queue read chunk.
  iree_async_semaphore_timepoint_t callback_timepoint;

  // Server retained while |callback_timepoint| may fire.
  iree_hal_remote_server_t* callback_server;

  // Session slot that owned the callback transfer when submitted.
  iree_hal_remote_server_session_t* callback_session_slot;

  // Session ID expected in |callback_session_slot| when the callback fires.
  uint64_t callback_session_id;

  // Bulk transfer ID to resume after local queue read completion.
  uint64_t callback_transfer_id;

  // State bits from iree_hal_remote_server_bulk_staging_slot_flag_bits_e.
  iree_hal_remote_server_bulk_staging_slot_flags_t flags;
};

struct iree_hal_remote_server_bulk_staging_pool_t {
  // Reference count for pool lifetime management.
  iree_atomic_ref_count_t ref_count;

  // Host allocator used for pool and slot storage.
  iree_allocator_t host_allocator;

  // Number of entries in |slots|.
  iree_host_size_t slot_count;

  // Byte length of each slot allocation.
  iree_host_size_t slot_length;

  // FAM: staging slots.
  iree_hal_remote_server_bulk_staging_slot_t slots[];
};

static void iree_hal_remote_server_bulk_staging_pool_retain(
    iree_hal_remote_server_bulk_staging_pool_t* pool) {
  if (!pool) return;
  iree_atomic_ref_count_inc(&pool->ref_count);
}

static void iree_hal_remote_server_bulk_staging_slot_deinitialize(
    iree_hal_remote_server_bulk_staging_slot_t* slot) {
  iree_hal_remote_server_release(slot->callback_server);
  iree_hal_semaphore_release(slot->semaphore);
  iree_hal_file_release(slot->file);
  iree_io_file_handle_release(slot->file_handle);
  iree_hal_device_release(slot->local_device);
  memset(slot, 0, sizeof(*slot));
}

static void iree_hal_remote_server_bulk_staging_pool_destroy(
    iree_hal_remote_server_bulk_staging_pool_t* pool) {
  if (!pool) return;
  iree_allocator_t host_allocator = pool->host_allocator;
  for (iree_host_size_t i = 0; i < pool->slot_count; ++i) {
    iree_hal_remote_server_bulk_staging_slot_deinitialize(&pool->slots[i]);
  }
  iree_allocator_free(host_allocator, pool);
}

static void iree_hal_remote_server_bulk_staging_pool_release(
    iree_hal_remote_server_bulk_staging_pool_t* pool) {
  if (pool && iree_atomic_ref_count_dec(&pool->ref_count) == 1) {
    iree_hal_remote_server_bulk_staging_pool_destroy(pool);
  }
}

static iree_status_t iree_hal_remote_server_bulk_staging_pool_create(
    iree_host_size_t slot_count, iree_host_size_t slot_length,
    iree_allocator_t host_allocator,
    iree_hal_remote_server_bulk_staging_pool_t** out_pool) {
  *out_pool = NULL;

  iree_host_size_t total_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_server_bulk_staging_pool_t), &total_size,
      IREE_STRUCT_FIELD_FAM(slot_count,
                            iree_hal_remote_server_bulk_staging_slot_t));

  iree_hal_remote_server_bulk_staging_pool_t* pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size, (void**)&pool);
  }
  if (iree_status_is_ok(status)) {
    memset(pool, 0, total_size);
    iree_atomic_ref_count_init(&pool->ref_count);
    pool->host_allocator = host_allocator;
    pool->slot_count = slot_count;
    pool->slot_length = slot_length;
  }

  for (iree_host_size_t i = 0; i < slot_count && iree_status_is_ok(status);
       ++i) {
    pool->slots[i].pool = pool;
    status = iree_hal_remote_server_bulk_host_allocation_create(
        slot_length, host_allocator, &pool->slots[i].contents,
        &pool->slots[i].file_handle);
  }

  if (iree_status_is_ok(status)) {
    *out_pool = pool;
  } else {
    iree_hal_remote_server_bulk_staging_pool_destroy(pool);
  }
  return status;
}

static void iree_hal_remote_server_bulk_staging_slot_unbind(
    iree_hal_remote_server_bulk_staging_slot_t* slot) {
  iree_hal_semaphore_release(slot->semaphore);
  slot->semaphore = NULL;
  iree_hal_file_release(slot->file);
  slot->file = NULL;
  iree_hal_device_release(slot->local_device);
  slot->local_device = NULL;
  slot->last_signal_value = 0;
}

static iree_status_t iree_hal_remote_server_bulk_staging_slot_bind(
    iree_hal_remote_server_bulk_staging_slot_t* slot,
    iree_hal_device_t* local_device) {
  if (slot->local_device == local_device && slot->file && slot->semaphore) {
    return iree_ok_status();
  }

  iree_hal_remote_server_bulk_staging_slot_unbind(slot);
  iree_hal_device_retain(local_device);
  slot->local_device = local_device;

  iree_status_t status = iree_hal_file_import(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      slot->file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &slot->file);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &slot->semaphore);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_staging_slot_unbind(slot);
  }
  return status;
}

static iree_status_t iree_hal_remote_server_bulk_staging_pool_try_acquire(
    iree_hal_remote_server_bulk_staging_pool_t* pool,
    iree_hal_device_t* local_device,
    iree_hal_remote_server_bulk_staging_slot_t** out_slot) {
  *out_slot = NULL;
  for (iree_host_size_t i = 0; i < pool->slot_count; ++i) {
    iree_hal_remote_server_bulk_staging_slot_t* slot = &pool->slots[i];
    if (iree_any_bit_set(
            slot->flags,
            IREE_HAL_REMOTE_SERVER_BULK_STAGING_SLOT_FLAG_IN_USE)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_remote_server_bulk_staging_slot_bind(slot, local_device));
    slot->flags |= IREE_HAL_REMOTE_SERVER_BULK_STAGING_SLOT_FLAG_IN_USE;
    *out_slot = slot;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_bulk_staging_pool_acquire(
    iree_hal_remote_server_bulk_staging_pool_t* pool,
    iree_hal_device_t* local_device,
    iree_hal_remote_server_bulk_staging_slot_t** out_slot) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_bulk_staging_pool_try_acquire(
      pool, local_device, out_slot));
  if (*out_slot) return iree_ok_status();
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "no remote bulk staging slots available");
}

static void iree_hal_remote_server_bulk_staging_pool_release_slot(
    iree_hal_remote_server_bulk_staging_pool_t* pool,
    iree_hal_remote_server_bulk_staging_slot_t* slot,
    uint64_t last_signal_value) {
  (void)pool;
  if (!slot) return;
  slot->last_signal_value = last_signal_value;
  slot->callback_signal_value = 0;
  memset(&slot->callback_timepoint, 0, sizeof(slot->callback_timepoint));
  iree_hal_remote_server_release(slot->callback_server);
  slot->callback_server = NULL;
  slot->callback_session_slot = NULL;
  slot->callback_session_id = 0;
  slot->callback_transfer_id = 0;
  slot->flags &= ~IREE_HAL_REMOTE_SERVER_BULK_STAGING_SLOT_FLAG_IN_USE;
}

static void iree_hal_remote_server_client_file_read_ready_context_release(
    iree_hal_remote_server_client_file_read_ready_t* context);

static void iree_hal_remote_server_client_file_read_ready_context_retain(
    iree_hal_remote_server_client_file_read_ready_t* context);

static void iree_hal_remote_server_client_file_read_ready_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status);

static void iree_hal_remote_server_client_file_read_chunk_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status);

static void iree_hal_remote_server_session_try_complete_bulk_drain(
    iree_hal_remote_server_session_t* session_slot);

typedef struct iree_hal_remote_server_bulk_chunk_list_t {
  // Retained DATA chunks collected for release or processing.
  iree_net_bulk_chunk_t* chunks[IREE_HAL_REMOTE_BULK_INITIAL_CHUNK_CREDIT];

  // Number of populated entries in |chunks|.
  iree_host_size_t chunk_count;

  // Transfer ID selected for collection.
  uint64_t transfer_id;
} iree_hal_remote_server_bulk_chunk_list_t;

static void iree_hal_remote_server_collect_chunks_for_transfer(
    void* user_data, iree_net_bulk_chunk_t* chunk) {
  iree_hal_remote_server_bulk_chunk_list_t* chunk_list =
      (iree_hal_remote_server_bulk_chunk_list_t*)user_data;
  if (iree_net_bulk_chunk_transfer_id(chunk) != chunk_list->transfer_id) return;
  if (chunk_list->chunk_count >= IREE_ARRAYSIZE(chunk_list->chunks)) return;
  chunk_list->chunks[chunk_list->chunk_count++] = chunk;
}

static void iree_hal_remote_server_release_chunks_for_transfer(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id) {
  if (!session_slot->bulk_receive_window) return;
  iree_hal_remote_server_bulk_chunk_list_t chunk_list;
  memset(&chunk_list, 0, sizeof(chunk_list));
  chunk_list.transfer_id = transfer_id;
  iree_net_bulk_receive_window_visit_chunks(
      session_slot->bulk_receive_window,
      iree_hal_remote_server_collect_chunks_for_transfer, &chunk_list);
  for (iree_host_size_t i = 0; i < chunk_list.chunk_count; ++i) {
    iree_net_bulk_receive_window_release_chunk(
        session_slot->bulk_receive_window, chunk_list.chunks[i]);
  }
}

static iree_status_t iree_hal_remote_server_bulk_receive_window_send_credit(
    void* user_data, uint32_t credit_delta) {
  iree_hal_remote_server_session_t* session_slot =
      (iree_hal_remote_server_session_t*)user_data;
  if (!session_slot->bulk_channel) {
    return iree_status_from_code(IREE_STATUS_ABORTED);
  }
  iree_status_t failure_status = iree_ok_status();
  iree_hal_remote_bulk_channel_send_result_t send_result =
      iree_hal_remote_bulk_channel_send_credit(
          session_slot->bulk_channel, credit_delta, /*operation_user_data=*/0,
          &failure_status);
  if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_ACCEPTED) {
    return iree_ok_status();
  }
  if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  return failure_status;
}

static iree_status_t iree_hal_remote_server_bulk_flush_receive_window_locked(
    iree_hal_remote_server_session_t* session_slot) {
  if (!session_slot->bulk_receive_window) return iree_ok_status();
  iree_status_t status = iree_net_bulk_receive_window_flush_credit(
      session_slot->bulk_receive_window);
  if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
    iree_status_free(status);
    return iree_ok_status();
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_flush_receive_window(
    iree_hal_remote_server_session_t* session_slot) {
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_status_t status =
      iree_hal_remote_server_bulk_flush_receive_window_locked(session_slot);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  return status;
}

static void iree_hal_remote_server_bulk_shutdown_session_on_error(
    iree_hal_remote_server_session_t* session_slot, iree_status_t status,
    iree_string_view_t reason) {
  if (iree_status_is_ok(status)) return;
  iree_net_session_t* session = NULL;
  iree_slim_mutex_lock(&session_slot->server->session_mutex);
  session = session_slot->session;
  if (session) iree_net_session_retain(session);
  iree_slim_mutex_unlock(&session_slot->server->session_mutex);
  if (session) {
    status = iree_status_join(
        status, iree_net_session_shutdown(session, /*reason_code=*/0, reason));
    iree_net_session_release(session);
  }
  iree_status_free(status);
}

static void iree_hal_remote_server_client_file_read_free_signal_list(
    iree_hal_remote_server_client_file_read_transfer_t* transfer) {
  iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                               transfer->server->host_allocator);
  transfer->signal_semaphore_list = iree_hal_semaphore_list_empty();
}

static void iree_hal_remote_server_client_file_read_signal_failure(
    iree_hal_remote_server_client_file_read_transfer_t* transfer,
    iree_status_t status) {
  if (!iree_all_bits_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_COMMAND_READY) ||
      iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    iree_status_ignore(status);
    return;
  }
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_CONTROL_RESPONSE)) {
    iree_status_t send_status =
        iree_hal_remote_server_session_send_error_response(
            transfer->session_slot, &transfer->response_envelope, status);
    iree_status_ignore(send_status);
  } else {
    iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
    iree_hal_remote_server_client_file_read_free_signal_list(transfer);
  }
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_SIGNAL_CONSUMED;
}

static void iree_hal_remote_server_client_file_read_deinitialize(
    iree_hal_remote_server_client_file_read_transfer_t* transfer) {
  if (iree_all_bits_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_COMMAND_READY) &&
      !iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    iree_hal_remote_server_client_file_read_signal_failure(
        transfer, iree_make_status(IREE_STATUS_CANCELLED,
                                   "remote bulk transfer cancelled"));
  }
  if (transfer->session_slot) {
    iree_hal_remote_server_release_chunks_for_transfer(transfer->session_slot,
                                                       transfer->transfer_id);
  }
  iree_hal_remote_server_client_file_read_ready_context_release(
      transfer->ready_context);
  iree_hal_semaphore_release(transfer->ready_semaphore);
  iree_hal_buffer_release(transfer->target_buffer);
  iree_hal_remote_bulk_transfer_tracker_deinitialize(
      &transfer->receive_tracker);
  iree_hal_remote_server_release(transfer->server);
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_server_client_file_write_signal_failure(
    iree_hal_remote_server_client_file_write_transfer_t* transfer,
    iree_status_t status) {
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    iree_status_ignore(status);
    return;
  }
  iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
  iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                               transfer->server->host_allocator);
  memset(&transfer->signal_semaphore_list, 0,
         sizeof(transfer->signal_semaphore_list));
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SIGNAL_CONSUMED;
}

static void iree_hal_remote_server_client_file_write_ready_context_release(
    iree_hal_remote_server_client_file_write_ready_t* context);

static void iree_hal_remote_server_client_file_write_ready_context_retain(
    iree_hal_remote_server_client_file_write_ready_t* context);

static void iree_hal_remote_server_client_file_write_ready_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status);

static void iree_hal_remote_server_client_file_write_free_initial_wait_list(
    iree_hal_remote_server_client_file_write_transfer_t* transfer) {
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_INITIAL_WAIT_CONSUMED)) {
    return;
  }
  iree_hal_semaphore_list_free(transfer->initial_wait_semaphore_list,
                               transfer->server->host_allocator);
  transfer->initial_wait_semaphore_list = iree_hal_semaphore_list_empty();
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_INITIAL_WAIT_CONSUMED;
}

static void iree_hal_remote_server_client_file_write_deinitialize(
    iree_hal_remote_server_client_file_write_transfer_t* transfer) {
  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    iree_hal_remote_server_client_file_write_signal_failure(
        transfer, iree_make_status(IREE_STATUS_CANCELLED,
                                   "remote bulk transfer cancelled"));
  }
  iree_hal_remote_server_client_file_write_free_initial_wait_list(transfer);
  iree_hal_remote_server_bulk_staging_pool_release_slot(
      transfer->staging_pool, transfer->staging_slot,
      transfer->last_staging_signal_value);
  iree_hal_remote_server_client_file_write_ready_context_release(
      transfer->ready_context);
  iree_hal_buffer_release(transfer->source_buffer);
  iree_hal_remote_server_release(transfer->server);
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_server_profile_transfer_deinitialize(
    iree_hal_remote_server_profile_transfer_t* transfer) {
  if (transfer->payload.data) {
    iree_allocator_free(transfer->server->host_allocator,
                        transfer->payload.data);
  }
  iree_hal_remote_server_release(transfer->server);
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_server_profile_pending_transfer_free(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_profile_pending_transfer_t* pending_transfer) {
  if (!pending_transfer) return;
  iree_allocator_t host_allocator = server->host_allocator;
  iree_allocator_free(host_allocator, pending_transfer->payload.data);
  iree_allocator_free(host_allocator, pending_transfer);
}

static void iree_hal_remote_server_profile_pending_transfer_free_list(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_profile_pending_transfer_t* pending_transfer) {
  while (pending_transfer) {
    iree_hal_remote_server_profile_pending_transfer_t* next =
        pending_transfer->next;
    iree_hal_remote_server_profile_pending_transfer_free(server,
                                                         pending_transfer);
    pending_transfer = next;
  }
}

static void iree_hal_remote_server_bulk_transfer_deinitialize(
    iree_hal_remote_server_bulk_transfer_t* transfer) {
  switch (transfer->kind) {
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ:
      iree_hal_remote_server_client_file_read_deinitialize(
          &transfer->client_file_read);
      break;
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE:
      iree_hal_remote_server_client_file_write_deinitialize(
          &transfer->client_file_write);
      break;
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND:
      iree_hal_remote_server_profile_transfer_deinitialize(
          &transfer->profile_send);
      break;
    default:
      break;
  }
  memset(transfer, 0, sizeof(*transfer));
}

static void iree_hal_remote_server_bulk_transfer_deinitialize_callback(
    void* user_data, iree_net_bulk_transfer_t* transfer) {
  (void)user_data;
  iree_hal_remote_server_bulk_transfer_deinitialize(
      iree_hal_remote_server_bulk_transfer_storage(transfer));
}

static void iree_hal_remote_server_bulk_release_transfer(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_net_bulk_transfer_t* transfer) {
  iree_hal_remote_bulk_transfer_scheduler_release(scheduler, transfer);
}

static void iree_hal_remote_server_client_file_write_release_transfer(
    iree_hal_remote_bulk_transfer_scheduler_t* scheduler,
    iree_net_bulk_transfer_t* transfer) {
  iree_hal_remote_server_bulk_release_transfer(scheduler, transfer);
}

static void iree_hal_remote_server_client_file_read_try_process_chunks_locked(
    iree_hal_remote_server_session_t* session_slot);

static void iree_hal_remote_server_client_file_read_try_finish_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_client_file_read_transfer_t* transfer =
      iree_hal_remote_server_client_file_read_storage(table_transfer);
  if (iree_all_bits_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_COMMAND_READY |
              IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_PEER_COMPLETE |
              IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_READY_COMPLETE) &&
      !iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_SIGNAL_CONSUMED) &&
      transfer->pending_operation_count == 0 &&
      iree_hal_remote_bulk_transfer_tracker_is_complete(
          &transfer->receive_tracker)) {
    if (iree_any_bit_set(
            transfer->flags,
            IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_CONTROL_RESPONSE)) {
      iree_status_t status = iree_hal_remote_server_session_send_response(
          transfer->session_slot, &transfer->response_envelope, IREE_STATUS_OK,
          NULL, 0);
      iree_status_ignore(status);
    } else {
      iree_status_t status = iree_hal_semaphore_list_signal(
          transfer->signal_semaphore_list, /*frontier=*/NULL);
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
      }
      iree_hal_remote_server_client_file_read_free_signal_list(transfer);
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_SIGNAL_CONSUMED;
  }
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_SIGNAL_CONSUMED) &&
      transfer->pending_operation_count == 0) {
    iree_hal_remote_server_bulk_release_transfer(
        session_slot->bulk_transfer_scheduler, table_transfer);
  }
}

static void iree_hal_remote_server_client_file_read_fail_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, iree_status_t status) {
  iree_hal_remote_server_client_file_read_transfer_t* transfer =
      iree_hal_remote_server_client_file_read_storage(table_transfer);
  iree_hal_remote_server_client_file_read_signal_failure(transfer, status);
  iree_hal_remote_server_release_chunks_for_transfer(session_slot,
                                                     transfer->transfer_id);
  if (transfer->pending_operation_count == 0) {
    iree_hal_remote_server_bulk_release_transfer(
        session_slot->bulk_transfer_scheduler, table_transfer);
  }
}

typedef struct iree_hal_remote_server_ready_chunk_query_t {
  // Session slot containing the transfer scheduler used for chunk readiness.
  iree_hal_remote_server_session_t* session_slot;

  // First chunk whose transfer is ready to process.
  iree_net_bulk_chunk_t* chunk;
} iree_hal_remote_server_ready_chunk_query_t;

static void iree_hal_remote_server_collect_ready_client_file_read_chunk(
    void* user_data, iree_net_bulk_chunk_t* chunk) {
  iree_hal_remote_server_ready_chunk_query_t* query =
      (iree_hal_remote_server_ready_chunk_query_t*)user_data;
  if (query->chunk) return;

  iree_hal_remote_server_session_t* session_slot = query->session_slot;
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler,
          iree_net_bulk_chunk_transfer_id(chunk));
  if (!table_transfer) return;

  iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_server_bulk_transfer_storage(table_transfer);
  if (bulk_transfer->kind !=
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ) {
    return;
  }
  iree_hal_remote_server_client_file_read_transfer_t* transfer =
      &bulk_transfer->client_file_read;
  if (!iree_all_bits_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_COMMAND_READY) ||
      iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    return;
  }

  query->chunk = chunk;
}

static iree_net_bulk_chunk_t*
iree_hal_remote_server_take_ready_client_file_read_chunk_locked(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_server_ready_chunk_query_t query;
  memset(&query, 0, sizeof(query));
  query.session_slot = session_slot;
  iree_net_bulk_receive_window_visit_chunks(
      session_slot->bulk_receive_window,
      iree_hal_remote_server_collect_ready_client_file_read_chunk, &query);
  return query.chunk;
}

static bool iree_hal_remote_server_client_file_read_submit_chunk_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_chunk_t* chunk) {
  const uint64_t transfer_id = iree_net_bulk_chunk_transfer_id(chunk);
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
  if (!table_transfer) return false;
  iree_hal_remote_server_client_file_read_transfer_t* transfer =
      iree_hal_remote_server_client_file_read_storage(table_transfer);

  iree_hal_remote_server_bulk_staging_slot_t* staging_slot = NULL;
  iree_status_t status = iree_hal_remote_server_bulk_staging_pool_try_acquire(
      session_slot->bulk_staging_pool, transfer->local_device, &staging_slot);
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_client_file_read_fail_locked(session_slot,
                                                        table_transfer, status);
    return false;
  }
  if (!staging_slot) return false;

  const uint64_t chunk_offset = iree_net_bulk_chunk_offset(chunk);
  const iree_const_byte_span_t chunk_payload =
      iree_net_bulk_chunk_payload(chunk);
  memcpy(staging_slot->contents.data, chunk_payload.data,
         chunk_payload.data_length);
  iree_net_bulk_receive_window_release_chunk(session_slot->bulk_receive_window,
                                             chunk);

  iree_hal_remote_server_t* server = transfer->server;
  iree_hal_remote_server_retain(server);
  iree_hal_device_t* local_device = transfer->local_device;
  iree_hal_device_retain(local_device);
  iree_hal_buffer_t* target_buffer = transfer->target_buffer;
  iree_hal_buffer_retain(target_buffer);
  iree_hal_file_t* staging_file = staging_slot->file;
  iree_hal_file_retain(staging_file);
  iree_hal_semaphore_t* staging_semaphore = staging_slot->semaphore;
  iree_hal_semaphore_retain(staging_semaphore);
  iree_hal_remote_server_bulk_staging_pool_t* staging_pool = staging_slot->pool;
  iree_hal_remote_server_bulk_staging_pool_retain(staging_pool);

  uint64_t ready_wait_value = 1;
  iree_hal_semaphore_t* ready_semaphore = transfer->ready_semaphore;
  iree_hal_semaphore_list_t wait_list = {
      .count = 1,
      .semaphores = &ready_semaphore,
      .payload_values = &ready_wait_value,
  };
  uint64_t staging_signal_value = staging_slot->last_signal_value + 1;
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &staging_semaphore,
      .payload_values = &staging_signal_value,
  };

  staging_slot->callback_signal_value = staging_signal_value;
  staging_slot->callback_timepoint.callback =
      iree_hal_remote_server_client_file_read_chunk_callback;
  staging_slot->callback_timepoint.user_data = staging_slot;
  staging_slot->callback_server = server;
  staging_slot->callback_session_slot = session_slot;
  staging_slot->callback_session_id = transfer->session_id;
  staging_slot->callback_transfer_id = transfer_id;
  ++transfer->pending_operation_count;

  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  status = iree_hal_device_queue_read(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      staging_file, /*source_offset=*/0, target_buffer,
      transfer->target_offset + (iree_device_size_t)chunk_offset,
      (iree_device_size_t)chunk_payload.data_length, IREE_HAL_READ_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)staging_semaphore, staging_signal_value,
        &staging_slot->callback_timepoint);
  }
  iree_hal_semaphore_release(staging_semaphore);
  iree_hal_file_release(staging_file);
  iree_hal_buffer_release(target_buffer);
  iree_hal_device_release(local_device);
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);

  if (iree_status_is_ok(status)) return true;

  iree_hal_remote_server_bulk_staging_pool_release_slot(
      staging_pool, staging_slot, staging_slot->last_signal_value);
  iree_hal_remote_server_bulk_staging_pool_release(staging_pool);
  table_transfer = NULL;
  if (session_slot->bulk_transfer_scheduler) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        session_slot->bulk_transfer_scheduler, transfer_id);
  }
  if (table_transfer) {
    transfer = iree_hal_remote_server_client_file_read_storage(table_transfer);
    if (transfer->pending_operation_count > 0) {
      --transfer->pending_operation_count;
    }
    iree_hal_remote_server_client_file_read_fail_locked(session_slot,
                                                        table_transfer, status);
  } else {
    iree_status_free(status);
  }
  return false;
}

static void iree_hal_remote_server_client_file_read_try_process_chunks_locked(
    iree_hal_remote_server_session_t* session_slot) {
  if (!session_slot->bulk_transfer_scheduler ||
      !session_slot->bulk_receive_window) {
    return;
  }
  while (true) {
    iree_net_bulk_chunk_t* chunk =
        iree_hal_remote_server_take_ready_client_file_read_chunk_locked(
            session_slot);
    if (!chunk) return;
    if (!iree_hal_remote_server_client_file_read_submit_chunk_locked(
            session_slot, chunk)) {
      return;
    }
  }
}

static void iree_hal_remote_server_client_file_write_try_finish_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      iree_hal_remote_server_client_file_write_storage(table_transfer);
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_PEER_COMPLETE) &&
      !iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    iree_status_t status = iree_hal_semaphore_list_signal(
        transfer->signal_semaphore_list, /*frontier=*/NULL);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_list_fail(transfer->signal_semaphore_list, status);
    }
    iree_hal_semaphore_list_free(transfer->signal_semaphore_list,
                                 transfer->server->host_allocator);
    memset(&transfer->signal_semaphore_list, 0,
           sizeof(transfer->signal_semaphore_list));
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SIGNAL_CONSUMED;
  }
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SIGNAL_CONSUMED) &&
      transfer->pending_operation_count == 0) {
    iree_hal_remote_server_client_file_write_release_transfer(
        session_slot->bulk_transfer_scheduler, table_transfer);
  }
}

static void iree_hal_remote_server_client_file_write_fail_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer, iree_status_t status) {
  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      iree_hal_remote_server_client_file_write_storage(table_transfer);
  iree_hal_remote_server_client_file_write_signal_failure(transfer, status);
  if (transfer->pending_operation_count == 0) {
    iree_hal_remote_server_client_file_write_release_transfer(
        session_slot->bulk_transfer_scheduler, table_transfer);
  }
}

static void iree_hal_remote_server_client_file_write_try_send_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel,
    iree_net_bulk_transfer_t* table_transfer);

static void
iree_hal_remote_server_client_file_write_submit_next_staging_write_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      iree_hal_remote_server_client_file_write_storage(table_transfer);
  const iree_hal_remote_server_client_file_write_transfer_flags_t staging_busy_flags =
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SUBMIT_PENDING |
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_WRITE_PENDING |
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_DATA_READY |
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SEND_PENDING;
  if (iree_any_bit_set(transfer->flags, staging_busy_flags)) {
    return;
  }

  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  if (transfer->next_staging_offset >= total_length) {
    return;
  }

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  const uint64_t remaining_length =
      total_length - transfer->next_staging_offset;
  const iree_device_size_t staging_length = (iree_device_size_t)iree_min(
      remaining_length, (uint64_t)transfer->staging_contents.data_length);
  const iree_device_size_t source_offset =
      transfer->source_offset + transfer->next_staging_offset;
  uint64_t staging_signal_value = transfer->last_staging_signal_value + 1;

  iree_allocator_t host_allocator = transfer->server->host_allocator;
  iree_hal_device_t* local_device = transfer->local_device;
  iree_hal_device_retain(local_device);
  iree_hal_buffer_t* source_buffer = transfer->source_buffer;
  iree_hal_buffer_retain(source_buffer);
  iree_hal_file_t* staging_file = transfer->staging_file;
  iree_hal_file_retain(staging_file);
  iree_hal_semaphore_t* staging_semaphore = transfer->staging_semaphore;
  iree_hal_semaphore_retain(staging_semaphore);
  iree_hal_write_flags_t write_flags = transfer->write_flags;
  iree_hal_remote_server_client_file_write_ready_t* ready_context =
      transfer->ready_context;
  iree_hal_remote_server_client_file_write_ready_context_retain(ready_context);

  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  uint64_t staging_wait_value = 0;
  const bool uses_initial_wait_list = transfer->next_staging_offset == 0;
  if (uses_initial_wait_list) {
    wait_list = transfer->initial_wait_semaphore_list;
    transfer->initial_wait_semaphore_list = iree_hal_semaphore_list_empty();
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_INITIAL_WAIT_CONSUMED;
  } else {
    staging_wait_value = transfer->last_staging_signal_value;
    wait_list = (iree_hal_semaphore_list_t){
        .count = 1,
        .semaphores = &staging_semaphore,
        .payload_values = &staging_wait_value,
    };
  }
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &staging_semaphore,
      .payload_values = &staging_signal_value,
  };

  transfer->staging_offset = transfer->next_staging_offset;
  transfer->staging_length = staging_length;
  transfer->next_staging_offset += staging_length;
  transfer->last_staging_signal_value = staging_signal_value;
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SUBMIT_PENDING |
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_WRITE_PENDING;
  ++transfer->pending_operation_count;

  ready_context->timepoint.callback =
      iree_hal_remote_server_client_file_write_ready_callback;
  ready_context->timepoint.user_data = ready_context;
  iree_hal_remote_server_client_file_write_ready_context_retain(ready_context);

  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  iree_status_t status = iree_hal_device_queue_write(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      source_buffer, source_offset, staging_file,
      /*target_offset=*/0, staging_length, write_flags);
  if (uses_initial_wait_list) {
    iree_hal_semaphore_list_free(wait_list, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)staging_semaphore, staging_signal_value,
        &ready_context->timepoint);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_client_file_write_ready_context_release(
        ready_context);
  }
  iree_hal_remote_server_client_file_write_ready_context_release(ready_context);
  iree_hal_semaphore_release(staging_semaphore);
  iree_hal_file_release(staging_file);
  iree_hal_buffer_release(source_buffer);
  iree_hal_device_release(local_device);
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);

  if (!session_slot->bulk_transfer_scheduler) {
    iree_status_ignore(status);
    return;
  }
  table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
      session_slot->bulk_transfer_scheduler, transfer_id);
  if (!table_transfer) {
    iree_status_ignore(status);
    return;
  }
  transfer = iree_hal_remote_server_client_file_write_storage(table_transfer);
  if (!iree_status_is_ok(status) && transfer->pending_operation_count > 0) {
    --transfer->pending_operation_count;
  }
  transfer->flags &=
      ~IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SUBMIT_PENDING;
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_client_file_write_try_send_locked(
        session_slot, session_slot->bulk_channel, table_transfer);
    if (!session_slot->bulk_transfer_scheduler) {
      iree_status_ignore(status);
      return;
    }
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        session_slot->bulk_transfer_scheduler, transfer_id);
    if (!table_transfer) {
      iree_status_ignore(status);
      return;
    }
    transfer = iree_hal_remote_server_client_file_write_storage(table_transfer);
    if (iree_any_bit_set(
            transfer->flags,
            IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SIGNAL_CONSUMED) &&
        transfer->pending_operation_count == 0) {
      iree_hal_remote_server_client_file_write_release_transfer(
          session_slot->bulk_transfer_scheduler, table_transfer);
    }
  } else {
    iree_hal_remote_server_client_file_write_fail_locked(
        session_slot, table_transfer, status);
  }
}

static void iree_hal_remote_server_client_file_write_try_send_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      iree_hal_remote_server_client_file_write_storage(table_transfer);
  const iree_hal_remote_server_client_file_write_transfer_flags_t
      terminal_or_send_pending_flags =
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SIGNAL_CONSUMED |
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SEND_PENDING;
  if (iree_any_bit_set(transfer->flags, terminal_or_send_pending_flags)) {
    return;
  }
  if (!bulk_channel) {
    iree_hal_remote_server_client_file_write_fail_locked(
        session_slot, table_transfer,
        iree_make_status(IREE_STATUS_UNAVAILABLE,
                         "remote bulk channel is not available"));
    return;
  }

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_START_SENT)) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_start(
            bulk_channel, transfer_id, total_length,
            IREE_NET_BULK_FRAME_FLAG_NONE, transfer_id, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return;
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_client_file_write_fail_locked(
          session_slot, table_transfer, failure_status);
      return;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_START_SENT |
        IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SEND_PENDING;
    ++transfer->pending_operation_count;
    return;
  }

  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_DATA_READY)) {
    if (iree_net_bulk_channel_remote_chunk_credit_count(bulk_channel) == 0) {
      return;
    }
    const uint64_t chunk_end =
        transfer->staging_offset + transfer->staging_length;
    iree_net_bulk_frame_flags_t flags =
        chunk_end == total_length ? IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK
                                  : IREE_NET_BULK_FRAME_FLAG_NONE;
    iree_async_span_t chunk_span = iree_async_span_from_ptr(
        transfer->staging_contents.data, transfer->staging_length);
    iree_async_span_list_t chunk_payload =
        iree_async_span_list_make(&chunk_span, 1);
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_data(
            bulk_channel, transfer_id, transfer->staging_offset,
            transfer->next_sequence, flags, chunk_payload, transfer_id,
            &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return;
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_client_file_write_fail_locked(
          session_slot, table_transfer, failure_status);
      return;
    }
    transfer->flags &=
        ~IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_DATA_READY;
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SEND_PENDING |
        IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SEND_PENDING;
    ++transfer->next_sequence;
    ++transfer->pending_operation_count;
    return;
  }

  const iree_hal_remote_server_client_file_write_transfer_flags_t staging_pending_flags =
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SUBMIT_PENDING |
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_WRITE_PENDING |
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SEND_PENDING;
  if (iree_any_bit_set(transfer->flags, staging_pending_flags)) {
    return;
  }

  if (transfer->next_staging_offset < total_length) {
    iree_hal_remote_server_client_file_write_submit_next_staging_write_locked(
        session_slot, table_transfer);
    return;
  }

  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_COMPLETE_SENT)) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_complete(
            bulk_channel, transfer_id, transfer_id, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return;
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_client_file_write_fail_locked(
          session_slot, table_transfer, failure_status);
      return;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_COMPLETE_SENT |
        IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SEND_PENDING;
    ++transfer->pending_operation_count;
    return;
  }

  iree_hal_remote_server_client_file_write_try_finish_locked(session_slot,
                                                             table_transfer);
}

static void iree_hal_remote_server_profile_transfer_try_finish_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_profile_transfer_t* transfer =
      &iree_hal_remote_server_bulk_transfer_storage(table_transfer)
           ->profile_send;
  if (iree_all_bits_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_COMPLETE_SENT |
              IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_PEER_COMPLETE) &&
      !iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING)) {
    iree_hal_remote_server_bulk_release_transfer(
        session_slot->bulk_transfer_scheduler, table_transfer);
  }
}

static void iree_hal_remote_server_profile_transfer_fail_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_bulk_release_transfer(
      session_slot->bulk_transfer_scheduler, table_transfer);
}

static iree_status_t iree_hal_remote_server_profile_transfer_try_send_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_profile_transfer_t* transfer =
      &iree_hal_remote_server_bulk_transfer_storage(table_transfer)
           ->profile_send;
  const iree_hal_remote_server_profile_transfer_flags_t
      terminal_or_send_pending_flags =
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_PEER_COMPLETE |
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING;
  if (iree_any_bit_set(transfer->flags, terminal_or_send_pending_flags)) {
    return iree_ok_status();
  }
  if (!bulk_channel) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_UNAVAILABLE, "remote bulk channel is not available");
    iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                        table_transfer);
    return status;
  }

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  const uint64_t total_length =
      iree_net_bulk_transfer_total_size(table_transfer);
  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_START_SENT)) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_start(
            bulk_channel, transfer_id, total_length,
            IREE_NET_BULK_FRAME_FLAG_NONE, transfer_id, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return iree_ok_status();
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                          table_transfer);
      return failure_status;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_START_SENT |
        IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING;
    return iree_ok_status();
  }

  if (transfer->send_offset < total_length) {
    if (iree_net_bulk_channel_remote_chunk_credit_count(bulk_channel) == 0) {
      return iree_ok_status();
    }
    const uint64_t remaining_length = total_length - transfer->send_offset;
    const iree_host_size_t chunk_length = (iree_host_size_t)iree_min(
        remaining_length, (uint64_t)IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH);
    const uint64_t chunk_end = transfer->send_offset + chunk_length;
    iree_net_bulk_frame_flags_t flags =
        chunk_end == total_length ? IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK
                                  : IREE_NET_BULK_FRAME_FLAG_NONE;
    iree_async_span_t chunk_span = iree_async_span_from_ptr(
        transfer->payload.data + (iree_host_size_t)transfer->send_offset,
        chunk_length);
    iree_async_span_list_t chunk_payload =
        iree_async_span_list_make(&chunk_span, 1);
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_data(
            bulk_channel, transfer_id, transfer->send_offset,
            transfer->next_sequence, flags, chunk_payload, transfer_id,
            &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return iree_ok_status();
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                          table_transfer);
      return failure_status;
    }
    transfer->send_offset = chunk_end;
    ++transfer->next_sequence;
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING;
    return iree_ok_status();
  }

  if (!iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_COMPLETE_SENT)) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_send_complete(
            bulk_channel, transfer_id, transfer_id, &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED) {
      return iree_ok_status();
    }
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                          table_transfer);
      return failure_status;
    }
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_COMPLETE_SENT |
        IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING;
    return iree_ok_status();
  }

  iree_hal_remote_server_profile_transfer_try_finish_locked(session_slot,
                                                            table_transfer);
  return iree_ok_status();
}

static iree_status_t
iree_hal_remote_server_profile_pending_transfer_enqueue_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    uint64_t sequence, iree_byte_span_t* payload) {
  iree_hal_remote_server_t* server = session_slot->server;
  iree_hal_remote_server_profile_pending_transfer_t* pending_transfer = NULL;
  iree_status_t status =
      iree_allocator_malloc(server->host_allocator, sizeof(*pending_transfer),
                            (void**)&pending_transfer);
  if (iree_status_is_ok(status)) {
    memset(pending_transfer, 0, sizeof(*pending_transfer));
    pending_transfer->session_id = session_id;
    pending_transfer->sequence = sequence;
    pending_transfer->payload = *payload;
    *payload = iree_byte_span_empty();
    if (session_slot->profile_pending_transfer_tail) {
      session_slot->profile_pending_transfer_tail->next = pending_transfer;
    } else {
      session_slot->profile_pending_transfer_head = pending_transfer;
    }
    session_slot->profile_pending_transfer_tail = pending_transfer;
  }
  return status;
}

static iree_status_t iree_hal_remote_server_profile_transfer_activate_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t session_id,
    uint64_t profile_sequence, iree_byte_span_t* payload,
    bool* out_table_full) {
  *out_table_full = false;
  iree_status_t status = iree_ok_status();
  if (!session_slot->bulk_transfer_scheduler) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  } else if (iree_hal_remote_bulk_transfer_scheduler_count(
                 session_slot->bulk_transfer_scheduler) >=
             iree_hal_remote_bulk_transfer_scheduler_capacity(
                 session_slot->bulk_transfer_scheduler)) {
    *out_table_full = true;
  }

  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (iree_status_is_ok(status) && !*out_table_full) {
    status = iree_hal_remote_bulk_transfer_scheduler_allocate_local(
        session_slot->bulk_transfer_scheduler, payload->data_length,
        /*user_value=*/0, &table_transfer);
  }
  if (iree_status_is_ok(status) && table_transfer) {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    memset(bulk_transfer, 0, sizeof(*bulk_transfer));
    bulk_transfer->kind =
        IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND;
    iree_hal_remote_server_profile_transfer_t* transfer =
        &bulk_transfer->profile_send;
    transfer->server = session_slot->server;
    iree_hal_remote_server_retain(transfer->server);
    transfer->session_slot = session_slot;
    transfer->session_id = session_id;
    transfer->payload = *payload;
    transfer->sequence = profile_sequence;
    *payload = iree_byte_span_empty();
    status = iree_hal_remote_server_profile_transfer_try_send_locked(
        session_slot, bulk_channel, table_transfer);
  }
  return status;
}

static iree_status_t
iree_hal_remote_server_profile_pending_transfer_try_drain_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t* out_failed_sequence) {
  *out_failed_sequence = 0;
  iree_status_t status = iree_ok_status();
  while (session_slot->bulk_transfer_scheduler && iree_status_is_ok(status) &&
         session_slot->profile_pending_transfer_head) {
    iree_hal_remote_server_profile_pending_transfer_t* pending_transfer =
        session_slot->profile_pending_transfer_head;
    iree_byte_span_t payload = pending_transfer->payload;
    bool table_full = false;
    status = iree_hal_remote_server_profile_transfer_activate_locked(
        session_slot, bulk_channel, pending_transfer->session_id,
        pending_transfer->sequence, &payload, &table_full);
    if (table_full) break;

    session_slot->profile_pending_transfer_head = pending_transfer->next;
    if (!session_slot->profile_pending_transfer_head) {
      session_slot->profile_pending_transfer_tail = NULL;
    }
    pending_transfer->next = NULL;
    pending_transfer->payload = payload;
    if (!iree_status_is_ok(status)) {
      *out_failed_sequence = pending_transfer->sequence;
    }
    iree_hal_remote_server_profile_pending_transfer_free(session_slot->server,
                                                         pending_transfer);
  }
  return status;
}

static iree_status_t iree_hal_remote_server_profile_pending_transfer_drain(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t returned_sequence) {
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    uint64_t failed_sequence = 0;
    iree_status_t drain_status = iree_ok_status();
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    drain_status =
        iree_hal_remote_server_profile_pending_transfer_try_drain_locked(
            session_slot, bulk_channel, &failed_sequence);
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    if (iree_status_is_ok(drain_status)) break;
    if (failed_sequence == returned_sequence) {
      status = drain_status;
    } else if (failed_sequence != 0) {
      iree_status_t observe_status =
          iree_hal_remote_server_profile_observe_transfer(
              session_slot, failed_sequence, drain_status);
      iree_status_ignore(observe_status);
    } else {
      status = drain_status;
    }
  }
  return status;
}

static bool iree_hal_remote_server_select_ready_client_file_write(
    void* user_data, iree_net_bulk_transfer_t* table_transfer) {
  (void)user_data;
  iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_server_bulk_transfer_storage(table_transfer);
  if (bulk_transfer->kind !=
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE) {
    return false;
  }
  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      &bulk_transfer->client_file_write;
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SIGNAL_CONSUMED)) {
    return false;
  }
  return true;
}

static bool iree_hal_remote_server_select_ready_profile_transfer(
    void* user_data, iree_net_bulk_transfer_t* table_transfer) {
  (void)user_data;
  iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_server_bulk_transfer_storage(table_transfer);
  if (bulk_transfer->kind !=
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND) {
    return false;
  }
  iree_hal_remote_server_profile_transfer_t* transfer =
      &bulk_transfer->profile_send;
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_PEER_COMPLETE)) {
    return false;
  }
  return true;
}

static void iree_hal_remote_server_client_file_write_try_send_all_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel) {
  if (!session_slot->bulk_transfer_scheduler) return;
  uint64_t transfer_ids[IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY];
  iree_host_size_t transfer_count = 0;
  bool all_ids_collected =
      iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
          session_slot->bulk_transfer_scheduler,
          iree_hal_remote_server_select_ready_client_file_write, NULL,
          transfer_ids, IREE_ARRAYSIZE(transfer_ids), &transfer_count);
  IREE_ASSERT(all_ids_collected);
  (void)all_ids_collected;
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    if (!session_slot->bulk_transfer_scheduler) return;
    iree_net_bulk_transfer_t* table_transfer =
        iree_hal_remote_bulk_transfer_scheduler_lookup(
            session_slot->bulk_transfer_scheduler, transfer_ids[i]);
    if (table_transfer) {
      iree_hal_remote_server_client_file_write_try_send_locked(
          session_slot, bulk_channel, table_transfer);
    }
  }
}

static iree_status_t
iree_hal_remote_server_profile_transfer_try_send_all_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_channel_t* bulk_channel, uint64_t* out_failed_sequence) {
  *out_failed_sequence = 0;
  if (!session_slot->bulk_transfer_scheduler) return iree_ok_status();
  uint64_t transfer_ids[IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY];
  iree_host_size_t transfer_count = 0;
  bool all_ids_collected =
      iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
          session_slot->bulk_transfer_scheduler,
          iree_hal_remote_server_select_ready_profile_transfer, NULL,
          transfer_ids, IREE_ARRAYSIZE(transfer_ids), &transfer_count);
  IREE_ASSERT(all_ids_collected);
  (void)all_ids_collected;
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    if (!session_slot->bulk_transfer_scheduler) return iree_ok_status();
    iree_net_bulk_transfer_t* table_transfer =
        iree_hal_remote_bulk_transfer_scheduler_lookup(
            session_slot->bulk_transfer_scheduler, transfer_ids[i]);
    if (table_transfer) {
      iree_hal_remote_server_profile_transfer_t* transfer =
          &iree_hal_remote_server_bulk_transfer_storage(table_transfer)
               ->profile_send;
      const uint64_t profile_sequence = transfer->sequence;
      iree_status_t status =
          iree_hal_remote_server_profile_transfer_try_send_locked(
              session_slot, bulk_channel, table_transfer);
      if (!iree_status_is_ok(status)) {
        *out_failed_sequence = profile_sequence;
        return status;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_remote_server_client_file_read_get_or_insert_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t total_length, iree_net_bulk_transfer_t** out_table_transfer) {
  *out_table_transfer = NULL;
  if (total_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "CLIENT_FILE_READ length %" PRIu64
                            " exceeds host size max %" PRIhsz,
                            total_length, IREE_HOST_SIZE_MAX);
  }

  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
  if (table_transfer) {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bulk transfer_id=%" PRIu64
                              " is already used by another transfer kind",
                              transfer_id);
    }
    if (iree_net_bulk_transfer_total_size(table_transfer) != total_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "CLIENT_FILE_READ size mismatch for transfer_id=%" PRIu64,
          transfer_id);
    }
    *out_table_transfer = table_transfer;
    return iree_ok_status();
  }

  iree_status_t status = iree_hal_remote_bulk_transfer_scheduler_insert_peer(
      session_slot->bulk_transfer_scheduler, transfer_id, total_length,
      /*user_value=*/0, &table_transfer);
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    memset(bulk_transfer, 0, sizeof(*bulk_transfer));
    bulk_transfer->kind =
        IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ;
    iree_hal_remote_server_client_file_read_transfer_t* transfer =
        &bulk_transfer->client_file_read;
    transfer->server = session_slot->server;
    iree_hal_remote_server_retain(transfer->server);
    transfer->session_slot = session_slot;
    transfer->session_id = session_slot->session_id;
    transfer->transfer_id = transfer_id;
    status = iree_hal_remote_bulk_transfer_tracker_initialize(
        total_length, IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH,
        session_slot->server->host_allocator, &transfer->receive_tracker);
  }
  if (!iree_status_is_ok(status) && table_transfer) {
    iree_hal_remote_server_bulk_release_transfer(
        session_slot->bulk_transfer_scheduler, table_transfer);
    table_transfer = NULL;
  }

  *out_table_transfer = table_transfer;
  return status;
}

static void iree_hal_remote_server_client_file_read_ready_context_retain(
    iree_hal_remote_server_client_file_read_ready_t* context) {
  if (!context) return;
  iree_atomic_ref_count_inc(&context->ref_count);
}

static void iree_hal_remote_server_client_file_read_ready_context_release(
    iree_hal_remote_server_client_file_read_ready_t* context) {
  if (!context) return;
  if (iree_atomic_ref_count_dec(&context->ref_count) != 1) return;
  iree_allocator_t host_allocator = context->host_allocator;
  iree_hal_semaphore_release(context->local_semaphore);
  iree_hal_remote_server_release(context->server);
  iree_allocator_free(host_allocator, context);
}

static void iree_hal_remote_server_client_file_read_ready_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_remote_server_client_file_read_ready_t* context =
      (iree_hal_remote_server_client_file_read_ready_t*)user_data;
  iree_hal_remote_server_t* server = context->server;
  iree_hal_remote_server_session_t* session_slot = context->session_slot;

  iree_net_bulk_channel_t* bulk_channel = NULL;
  bool session_active = false;
  iree_slim_mutex_lock(&server->session_mutex);
  session_active = session_slot->session_id == context->session_id &&
                   session_slot->session != NULL &&
                   session_slot->bulk_channel != NULL;
  if (session_active) {
    bulk_channel = session_slot->bulk_channel;
    iree_net_bulk_channel_retain(bulk_channel);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  bool drain_profile_pending = false;
  if (!session_active) {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    iree_net_bulk_transfer_t* table_transfer = NULL;
    if (session_slot->bulk_transfer_scheduler) {
      table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, context->transfer_id);
    }
    if (table_transfer) {
      iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
          iree_hal_remote_server_bulk_transfer_storage(table_transfer);
      if (bulk_transfer->kind ==
          IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ) {
        iree_hal_remote_server_client_file_read_transfer_t* transfer =
            &bulk_transfer->client_file_read;
        if (transfer->pending_operation_count > 0) {
          --transfer->pending_operation_count;
        }
        transfer->flags &=
            ~IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_READY_PENDING;
        iree_hal_remote_server_client_file_read_try_finish_locked(
            session_slot, table_transfer);
      }
    }
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_ignore(status);
  } else {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    iree_net_bulk_transfer_t* table_transfer = NULL;
    if (session_slot->bulk_transfer_scheduler) {
      table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, context->transfer_id);
    }
    if (!table_transfer) {
      iree_status_ignore(status);
    } else {
      iree_hal_remote_server_client_file_read_transfer_t* transfer =
          iree_hal_remote_server_client_file_read_storage(table_transfer);
      if (transfer->pending_operation_count > 0) {
        --transfer->pending_operation_count;
      }
      transfer->flags &=
          ~IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_READY_PENDING;
      if (iree_status_is_ok(status)) {
        transfer->flags |=
            IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_READY_COMPLETE;
        iree_hal_remote_server_client_file_read_try_process_chunks_locked(
            session_slot);
        table_transfer = NULL;
        if (session_slot->bulk_transfer_scheduler) {
          table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
              session_slot->bulk_transfer_scheduler, context->transfer_id);
        }
        if (table_transfer) {
          iree_hal_remote_server_client_file_read_try_finish_locked(
              session_slot, table_transfer);
        }
      } else {
        iree_hal_remote_server_client_file_read_fail_locked(
            session_slot, table_transfer, status);
      }
    }
    drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }

  if (bulk_channel) {
    iree_status_t flush_status =
        iree_hal_remote_server_bulk_flush_receive_window(session_slot);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, flush_status,
        iree_make_cstring_view("bulk receive credit flush failed"));
  }
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot, bulk_channel, /*returned_sequence=*/0);
    iree_status_ignore(drain_status);
  }
  iree_net_bulk_channel_release(bulk_channel);
  iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
  iree_hal_remote_server_client_file_read_ready_context_release(context);
}

static void iree_hal_remote_server_client_file_read_chunk_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_remote_server_bulk_staging_slot_t* staging_slot =
      (iree_hal_remote_server_bulk_staging_slot_t*)user_data;
  iree_hal_remote_server_t* server = staging_slot->callback_server;
  iree_hal_remote_server_retain(server);
  iree_hal_remote_server_session_t* session_slot =
      staging_slot->callback_session_slot;
  const uint64_t session_id = staging_slot->callback_session_id;
  const uint64_t transfer_id = staging_slot->callback_transfer_id;
  iree_hal_remote_server_bulk_staging_pool_t* staging_pool = staging_slot->pool;
  const uint64_t signal_value = staging_slot->callback_signal_value;

  iree_net_bulk_channel_t* bulk_channel = NULL;
  bool session_active = false;
  iree_slim_mutex_lock(&server->session_mutex);
  session_active = session_slot->session_id == session_id &&
                   session_slot->session != NULL &&
                   session_slot->bulk_channel != NULL;
  if (session_active) {
    bulk_channel = session_slot->bulk_channel;
    iree_net_bulk_channel_retain(bulk_channel);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  bool drain_profile_pending = false;
  if (!session_active) {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    iree_net_bulk_transfer_t* table_transfer = NULL;
    if (session_slot->bulk_transfer_scheduler) {
      table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
    }
    iree_hal_remote_server_bulk_staging_pool_release_slot(
        staging_pool, staging_slot, signal_value);
    if (table_transfer) {
      iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
          iree_hal_remote_server_bulk_transfer_storage(table_transfer);
      if (bulk_transfer->kind ==
          IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ) {
        iree_hal_remote_server_client_file_read_transfer_t* transfer =
            &bulk_transfer->client_file_read;
        if (transfer->pending_operation_count > 0) {
          --transfer->pending_operation_count;
        }
        iree_hal_remote_server_client_file_read_try_finish_locked(
            session_slot, table_transfer);
      }
    }
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_ignore(status);
  } else {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    iree_net_bulk_transfer_t* table_transfer = NULL;
    if (session_slot->bulk_transfer_scheduler) {
      table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
    }
    iree_hal_remote_server_bulk_staging_pool_release_slot(
        staging_pool, staging_slot, signal_value);
    if (!table_transfer) {
      iree_status_ignore(status);
    } else {
      iree_hal_remote_server_client_file_read_transfer_t* transfer =
          iree_hal_remote_server_client_file_read_storage(table_transfer);
      if (transfer->pending_operation_count > 0) {
        --transfer->pending_operation_count;
      }
      if (iree_status_is_ok(status)) {
        iree_hal_remote_server_client_file_read_try_process_chunks_locked(
            session_slot);
        table_transfer = NULL;
        if (session_slot->bulk_transfer_scheduler) {
          table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
              session_slot->bulk_transfer_scheduler, transfer_id);
        }
        if (table_transfer) {
          iree_hal_remote_server_client_file_read_try_finish_locked(
              session_slot, table_transfer);
        }
      } else {
        iree_hal_remote_server_client_file_read_fail_locked(
            session_slot, table_transfer, status);
      }
    }
    drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }

  if (bulk_channel) {
    iree_status_t flush_status =
        iree_hal_remote_server_bulk_flush_receive_window(session_slot);
    iree_hal_remote_server_bulk_shutdown_session_on_error(
        session_slot, flush_status,
        iree_make_cstring_view("bulk receive credit flush failed"));
  }
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot, bulk_channel, /*returned_sequence=*/0);
    iree_status_ignore(drain_status);
  }
  iree_net_bulk_channel_release(bulk_channel);
  iree_hal_remote_server_bulk_staging_pool_release(staging_pool);
  iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
  iree_hal_remote_server_release(server);
}

static void iree_hal_remote_server_client_file_write_ready_context_retain(
    iree_hal_remote_server_client_file_write_ready_t* context) {
  if (!context) return;
  iree_atomic_ref_count_inc(&context->ref_count);
}

static void iree_hal_remote_server_client_file_write_ready_context_release(
    iree_hal_remote_server_client_file_write_ready_t* context) {
  if (!context) return;
  if (iree_atomic_ref_count_dec(&context->ref_count) != 1) return;
  iree_allocator_t host_allocator = context->host_allocator;
  iree_hal_semaphore_release(context->local_semaphore);
  iree_hal_remote_server_release(context->server);
  iree_allocator_free(host_allocator, context);
}

static void iree_hal_remote_server_client_file_write_ready_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_remote_server_client_file_write_ready_t* context =
      (iree_hal_remote_server_client_file_write_ready_t*)user_data;
  iree_hal_remote_server_t* server = context->server;
  iree_hal_remote_server_session_t* session_slot = context->session_slot;

  iree_net_bulk_channel_t* bulk_channel = NULL;
  bool session_active = false;
  iree_slim_mutex_lock(&server->session_mutex);
  session_active = session_slot->session_id == context->session_id &&
                   session_slot->session != NULL &&
                   session_slot->bulk_channel != NULL;
  if (session_active) {
    bulk_channel = session_slot->bulk_channel;
    iree_net_bulk_channel_retain(bulk_channel);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  bool drain_profile_pending = false;
  if (!session_active) {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    iree_net_bulk_transfer_t* table_transfer = NULL;
    if (session_slot->bulk_transfer_scheduler) {
      table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, context->transfer_id);
    }
    if (table_transfer) {
      iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
          iree_hal_remote_server_bulk_transfer_storage(table_transfer);
      if (bulk_transfer->kind ==
          IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE) {
        iree_hal_remote_server_client_file_write_transfer_t* transfer =
            &bulk_transfer->client_file_write;
        if (transfer->pending_operation_count > 0) {
          --transfer->pending_operation_count;
        }
        transfer->flags &=
            ~IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_WRITE_PENDING;
        iree_hal_remote_server_client_file_write_try_finish_locked(
            session_slot, table_transfer);
      }
    }
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_ignore(status);
  } else {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    iree_net_bulk_transfer_t* table_transfer = NULL;
    if (session_slot->bulk_transfer_scheduler) {
      table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, context->transfer_id);
    }
    if (!table_transfer) {
      iree_status_ignore(status);
    } else if (iree_status_is_ok(status)) {
      iree_hal_remote_server_client_file_write_transfer_t* transfer =
          iree_hal_remote_server_client_file_write_storage(table_transfer);
      if (transfer->pending_operation_count > 0) {
        --transfer->pending_operation_count;
      }
      transfer->flags &=
          ~IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_WRITE_PENDING;
      transfer->flags |=
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_DATA_READY;
      iree_hal_remote_server_client_file_write_try_send_locked(
          session_slot, bulk_channel, table_transfer);
      table_transfer = NULL;
      if (session_slot->bulk_transfer_scheduler) {
        table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
            session_slot->bulk_transfer_scheduler, context->transfer_id);
      }
      if (table_transfer) {
        iree_hal_remote_server_client_file_write_try_finish_locked(
            session_slot, table_transfer);
      }
    } else {
      iree_hal_remote_server_client_file_write_transfer_t* transfer =
          iree_hal_remote_server_client_file_write_storage(table_transfer);
      if (transfer->pending_operation_count > 0) {
        --transfer->pending_operation_count;
      }
      iree_hal_remote_server_client_file_write_fail_locked(
          session_slot, table_transfer, status);
    }
    drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    if (drain_profile_pending) {
      iree_status_t drain_status =
          iree_hal_remote_server_profile_pending_transfer_drain(
              session_slot, bulk_channel, /*returned_sequence=*/0);
      iree_status_ignore(drain_status);
    }
    iree_net_bulk_channel_release(bulk_channel);
  }

  iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
  iree_hal_remote_server_client_file_write_ready_context_release(context);
}

static void iree_hal_remote_server_bulk_fail_transfer_for_drain_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer) {
  iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_server_bulk_transfer_storage(table_transfer);
  switch (bulk_transfer->kind) {
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ:
      iree_hal_remote_server_client_file_read_fail_locked(
          session_slot, table_transfer,
          iree_make_status(IREE_STATUS_CANCELLED,
                           "remote bulk transfer cancelled"));
      break;
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE:
      iree_hal_remote_server_client_file_write_fail_locked(
          session_slot, table_transfer,
          iree_make_status(IREE_STATUS_CANCELLED,
                           "remote bulk transfer cancelled"));
      break;
    case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND:
      iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                          table_transfer);
      break;
    default:
      iree_hal_remote_server_bulk_release_transfer(
          session_slot->bulk_transfer_scheduler, table_transfer);
      break;
  }
}

static void iree_hal_remote_server_client_file_read_submit_ready_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_bulk_transfer_t* table_transfer,
    iree_hal_semaphore_list_t wait_list) {
  iree_hal_remote_server_client_file_read_transfer_t* transfer =
      iree_hal_remote_server_client_file_read_storage(table_transfer);
  const iree_hal_remote_server_client_file_read_transfer_flags_t
      ready_done_or_pending_flags =
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_READY_PENDING |
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_READY_COMPLETE |
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_SIGNAL_CONSUMED;
  if (iree_any_bit_set(transfer->flags, ready_done_or_pending_flags)) {
    return;
  }

  iree_hal_device_t* local_device = transfer->local_device;
  iree_hal_device_retain(local_device);
  iree_hal_semaphore_t* ready_semaphore = transfer->ready_semaphore;
  iree_hal_semaphore_retain(ready_semaphore);
  iree_hal_remote_server_client_file_read_ready_t* ready_context =
      transfer->ready_context;
  iree_hal_remote_server_client_file_read_ready_context_retain(ready_context);

  uint64_t ready_signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &ready_semaphore,
      .payload_values = &ready_signal_value,
  };

  ready_context->timepoint.callback =
      iree_hal_remote_server_client_file_read_ready_callback;
  ready_context->timepoint.user_data = ready_context;
  iree_hal_remote_server_client_file_read_ready_context_retain(ready_context);
  transfer->flags |=
      IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_READY_PENDING;
  ++transfer->pending_operation_count;

  const uint64_t transfer_id = iree_net_bulk_transfer_id(table_transfer);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  iree_status_t status = iree_hal_device_queue_barrier(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)ready_semaphore, ready_signal_value,
        &ready_context->timepoint);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_client_file_read_ready_context_release(
        ready_context);
  }
  iree_hal_remote_server_client_file_read_ready_context_release(ready_context);
  iree_hal_semaphore_release(ready_semaphore);
  iree_hal_device_release(local_device);
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);

  if (iree_status_is_ok(status)) return;
  table_transfer = NULL;
  if (session_slot->bulk_transfer_scheduler) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        session_slot->bulk_transfer_scheduler, transfer_id);
  }
  if (table_transfer) {
    transfer = iree_hal_remote_server_client_file_read_storage(table_transfer);
    if (transfer->pending_operation_count > 0) {
      --transfer->pending_operation_count;
    }
    transfer->flags &=
        ~IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_READY_PENDING;
    iree_hal_remote_server_client_file_read_fail_locked(session_slot,
                                                        table_transfer, status);
  } else {
    iree_status_ignore(status);
  }
}

iree_status_t iree_hal_remote_server_session_initialize_bulk_transfers(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator) {
  iree_hal_remote_bulk_transfer_scheduler_options_t options =
      iree_hal_remote_bulk_transfer_scheduler_options_default();
  options.capacity = IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY;
  options.user_storage_size = sizeof(iree_hal_remote_server_bulk_transfer_t);
  options.user_storage_alignment =
      iree_alignof(iree_hal_remote_server_bulk_transfer_t);
  options.initial_transfer_id = 2;
  options.transfer_id_stride = 2;
  iree_hal_remote_bulk_transfer_scheduler_callbacks_t callbacks = {
      .deinitialize =
          iree_hal_remote_server_bulk_transfer_deinitialize_callback,
      .user_data = NULL,
  };
  iree_status_t status = iree_hal_remote_bulk_transfer_scheduler_allocate(
      &options, callbacks, host_allocator,
      &session_slot->bulk_transfer_scheduler);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_staging_pool_create(
        IREE_HAL_REMOTE_BULK_STAGING_SLOT_COUNT,
        IREE_HAL_REMOTE_BULK_DATA_CHUNK_LENGTH, host_allocator,
        &session_slot->bulk_staging_pool);
  }
  iree_net_bulk_receive_window_options_t receive_window_options =
      iree_net_bulk_receive_window_options_default();
  receive_window_options.chunk_pool.capacity =
      IREE_HAL_REMOTE_BULK_INITIAL_CHUNK_CREDIT;
  iree_net_bulk_receive_window_callbacks_t receive_window_callbacks = {
      .send_credit = iree_hal_remote_server_bulk_receive_window_send_credit,
      .user_data = session_slot,
  };
  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_receive_window_allocate(
        &receive_window_options, receive_window_callbacks, host_allocator,
        &session_slot->bulk_receive_window);
  }
  if (!iree_status_is_ok(status)) {
    iree_net_bulk_receive_window_free(session_slot->bulk_receive_window);
    session_slot->bulk_receive_window = NULL;
    iree_hal_remote_server_bulk_staging_pool_release(
        session_slot->bulk_staging_pool);
    session_slot->bulk_staging_pool = NULL;
    iree_hal_remote_bulk_transfer_scheduler_free(
        session_slot->bulk_transfer_scheduler);
    session_slot->bulk_transfer_scheduler = NULL;
  }
  return status;
}

static bool iree_hal_remote_server_session_take_drained_bulk_state_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_bulk_transfer_scheduler_t** out_scheduler,
    iree_hal_remote_server_bulk_staging_pool_t** out_staging_pool,
    iree_net_bulk_receive_window_t** out_receive_window) {
  *out_scheduler = NULL;
  *out_staging_pool = NULL;
  *out_receive_window = NULL;

  if (!iree_any_bit_set(
          session_slot->flags,
          IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING)) {
    return false;
  }
  if (session_slot->bulk_transfer_scheduler &&
      iree_hal_remote_bulk_transfer_scheduler_count(
          session_slot->bulk_transfer_scheduler) != 0) {
    return false;
  }
  if (session_slot->profile_pending_transfer_head ||
      session_slot->profile_pending_transfer_tail) {
    return false;
  }
  if (session_slot->bulk_channel &&
      iree_net_bulk_channel_has_pending_sends(session_slot->bulk_channel)) {
    return false;
  }

  *out_scheduler = session_slot->bulk_transfer_scheduler;
  session_slot->bulk_transfer_scheduler = NULL;
  *out_staging_pool = session_slot->bulk_staging_pool;
  session_slot->bulk_staging_pool = NULL;
  *out_receive_window = session_slot->bulk_receive_window;
  session_slot->bulk_receive_window = NULL;
  return true;
}

static void iree_hal_remote_server_session_try_complete_bulk_drain(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler = NULL;
  iree_hal_remote_server_bulk_staging_pool_t* staging_pool = NULL;
  iree_net_bulk_receive_window_t* receive_window = NULL;

  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  bool drained = iree_hal_remote_server_session_take_drained_bulk_state_locked(
      session_slot, &scheduler, &staging_pool, &receive_window);
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);

  if (drained) {
    iree_hal_remote_bulk_transfer_scheduler_free(scheduler);
    iree_hal_remote_server_bulk_staging_pool_release(staging_pool);
    iree_net_bulk_receive_window_free(receive_window);
    iree_hal_remote_server_session_complete_bulk_drain(session_slot);
  }
}

void iree_hal_remote_server_session_deinitialize_bulk_transfers(
    iree_hal_remote_server_session_t* session_slot) {
  if (!session_slot->bulk_transfer_scheduler &&
      !session_slot->bulk_staging_pool && !session_slot->bulk_receive_window &&
      !session_slot->profile_pending_transfer_head &&
      !session_slot->profile_pending_transfer_tail) {
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }

  const bool drain_pending =
      iree_any_bit_set(session_slot->flags,
                       IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING);
  bool bulk_drained = false;
  iree_hal_remote_server_profile_pending_transfer_t* pending_profile_transfers =
      NULL;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  pending_profile_transfers = session_slot->profile_pending_transfer_head;
  session_slot->profile_pending_transfer_head = NULL;
  session_slot->profile_pending_transfer_tail = NULL;
  if (!drain_pending && session_slot->bulk_transfer_scheduler) {
    iree_hal_remote_bulk_transfer_scheduler_clear(
        session_slot->bulk_transfer_scheduler);
  } else if (drain_pending && session_slot->bulk_transfer_scheduler) {
    uint64_t transfer_ids[IREE_HAL_REMOTE_BULK_ACTIVE_TRANSFER_CAPACITY];
    iree_host_size_t transfer_count = 0;
    bool all_ids_collected =
        iree_hal_remote_bulk_transfer_scheduler_collect_transfer_ids(
            session_slot->bulk_transfer_scheduler, /*select=*/NULL, NULL,
            transfer_ids, IREE_ARRAYSIZE(transfer_ids), &transfer_count);
    IREE_ASSERT(all_ids_collected);
    (void)all_ids_collected;
    for (iree_host_size_t i = 0; i < transfer_count; ++i) {
      if (!session_slot->bulk_transfer_scheduler) break;
      iree_net_bulk_transfer_t* table_transfer =
          iree_hal_remote_bulk_transfer_scheduler_lookup(
              session_slot->bulk_transfer_scheduler, transfer_ids[i]);
      if (table_transfer) {
        iree_hal_remote_server_bulk_fail_transfer_for_drain_locked(
            session_slot, table_transfer);
      }
    }
  }
  iree_hal_remote_bulk_transfer_scheduler_t* scheduler =
      session_slot->bulk_transfer_scheduler;
  iree_hal_remote_server_bulk_staging_pool_t* staging_pool =
      session_slot->bulk_staging_pool;
  iree_net_bulk_receive_window_t* receive_window =
      session_slot->bulk_receive_window;
  if (!drain_pending) {
    session_slot->bulk_transfer_scheduler = NULL;
    session_slot->bulk_staging_pool = NULL;
    session_slot->bulk_receive_window = NULL;
  } else {
    bulk_drained =
        iree_hal_remote_server_session_take_drained_bulk_state_locked(
            session_slot, &scheduler, &staging_pool, &receive_window);
    if (!bulk_drained) {
      scheduler = NULL;
      staging_pool = NULL;
      receive_window = NULL;
    }
  }
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  iree_hal_remote_server_profile_pending_transfer_free_list(
      session_slot->server, pending_profile_transfers);
  iree_hal_remote_bulk_transfer_scheduler_free(scheduler);
  iree_hal_remote_server_bulk_staging_pool_release(staging_pool);
  iree_net_bulk_receive_window_free(receive_window);
  if (drain_pending && bulk_drained) {
    iree_hal_remote_server_session_complete_bulk_drain(session_slot);
  }
}

static iree_status_t iree_hal_remote_server_bulk_submit_client_file_read_impl(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, uint64_t transfer_id,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags,
    const iree_hal_remote_control_envelope_t* response_envelope) {
  iree_status_t status =
      iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  if (iree_status_is_ok(status) && flags != IREE_HAL_READ_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported CLIENT_FILE_READ flags: 0x%" PRIx64,
                              flags);
  }

  iree_hal_remote_server_t* server = session_slot->server;
  iree_allocator_t host_allocator = server->host_allocator;
  iree_hal_semaphore_t* ready_semaphore = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &ready_semaphore);
  }

  iree_hal_remote_server_client_file_read_ready_t* ready_context = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*ready_context),
                                   (void**)&ready_context);
  }
  if (iree_status_is_ok(status)) {
    memset(ready_context, 0, sizeof(*ready_context));
    iree_atomic_ref_count_init(&ready_context->ref_count);
    ready_context->server = server;
    iree_hal_remote_server_retain(ready_context->server);
    ready_context->session_slot = session_slot;
    ready_context->session_id = session_slot->session_id;
    ready_context->transfer_id = transfer_id;
    ready_context->local_semaphore = ready_semaphore;
    iree_hal_semaphore_retain(ready_context->local_semaphore);
    ready_context->host_allocator = host_allocator;
    ready_context->timepoint.callback =
        iree_hal_remote_server_client_file_read_ready_callback;
    ready_context->timepoint.user_data = ready_context;
  }

  iree_net_bulk_transfer_t* table_transfer = NULL;
  bool transfer_inserted_or_found = false;
  bool drain_profile_pending = false;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    if (!session_slot->bulk_transfer_scheduler ||
        !session_slot->bulk_receive_window) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_client_file_read_get_or_insert_locked(
          session_slot, transfer_id, (uint64_t)length, &table_transfer);
      transfer_inserted_or_found = iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_remote_server_client_file_read_transfer_t* transfer =
          iree_hal_remote_server_client_file_read_storage(table_transfer);
      if (iree_any_bit_set(
              transfer->flags,
              IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_COMMAND_READY)) {
        status = iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "CLIENT_FILE_READ command already attached to transfer_id=%" PRIu64,
            transfer_id);
      }
      if (iree_status_is_ok(status)) {
        if (response_envelope) {
          transfer->signal_semaphore_list = iree_hal_semaphore_list_empty();
        } else {
          status = iree_hal_semaphore_list_clone(
              &signal_list, host_allocator, &transfer->signal_semaphore_list);
        }
      }
      if (iree_status_is_ok(status)) {
        transfer->local_device = local_device;
        transfer->target_buffer = target_buffer;
        iree_hal_buffer_retain(transfer->target_buffer);
        transfer->target_offset = target_offset;
        transfer->ready_semaphore = ready_semaphore;
        ready_semaphore = NULL;
        transfer->ready_context = ready_context;
        ready_context = NULL;
        if (response_envelope) {
          transfer->response_envelope = *response_envelope;
          transfer->flags |=
              IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_CONTROL_RESPONSE;
        }
        transfer->flags |=
            IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_COMMAND_READY;
        iree_hal_remote_server_client_file_read_submit_ready_locked(
            session_slot, table_transfer, wait_list);
        table_transfer = NULL;
        if (session_slot->bulk_transfer_scheduler) {
          table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
              session_slot->bulk_transfer_scheduler, transfer_id);
        }
      }
      if (iree_status_is_ok(status) && table_transfer) {
        iree_hal_remote_server_client_file_read_try_process_chunks_locked(
            session_slot);
        table_transfer = NULL;
        if (session_slot->bulk_transfer_scheduler) {
          table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
              session_slot->bulk_transfer_scheduler, transfer_id);
        }
      }
      if (iree_status_is_ok(status) && table_transfer) {
        iree_hal_remote_server_client_file_read_try_finish_locked(
            session_slot, table_transfer);
      }
    }
    if (!iree_status_is_ok(status) && transfer_inserted_or_found &&
        table_transfer) {
      iree_hal_remote_server_bulk_release_transfer(
          session_slot->bulk_transfer_scheduler, table_transfer);
    }
    drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_flush_receive_window(session_slot);
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot, session_slot->bulk_channel,
                    /*returned_sequence=*/0));
  }
  iree_hal_remote_server_client_file_read_ready_context_release(ready_context);
  iree_hal_semaphore_release(ready_semaphore);
  return status;
}

iree_status_t iree_hal_remote_server_bulk_submit_client_file_read(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, uint64_t transfer_id,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_hal_remote_server_bulk_submit_client_file_read_impl(
      session_slot, local_device, wait_list, signal_list, transfer_id,
      target_buffer, target_offset, length, flags, /*response_envelope=*/NULL);
}

iree_status_t iree_hal_remote_server_bulk_submit_buffer_unmap(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device,
    const iree_hal_remote_control_envelope_t* response_envelope,
    uint64_t transfer_id, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  return iree_hal_remote_server_bulk_submit_client_file_read_impl(
      session_slot, local_device, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), transfer_id, target_buffer,
      target_offset, length, IREE_HAL_READ_FLAG_NONE, response_envelope);
}

iree_status_t iree_hal_remote_server_bulk_submit_client_file_write(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, uint64_t transfer_id,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_status_t status =
      iree_hal_buffer_validate_range(source_buffer, source_offset, length);

  iree_hal_remote_server_t* server = session_slot->server;
  iree_allocator_t host_allocator = server->host_allocator;
  iree_hal_remote_server_client_file_write_ready_t* ready_context = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*ready_context),
                                   (void**)&ready_context);
  }
  if (iree_status_is_ok(status)) {
    memset(ready_context, 0, sizeof(*ready_context));
    iree_atomic_ref_count_init(&ready_context->ref_count);
    ready_context->server = server;
    iree_hal_remote_server_retain(ready_context->server);
    ready_context->session_slot = session_slot;
    ready_context->transfer_id = transfer_id;
    ready_context->host_allocator = host_allocator;
    ready_context->timepoint.callback =
        iree_hal_remote_server_client_file_write_ready_callback;
    ready_context->timepoint.user_data = ready_context;
  }

  uint64_t session_id = 0;
  iree_net_bulk_transfer_t* table_transfer = NULL;
  bool transfer_inserted = false;
  bool drain_profile_pending = false;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&server->session_mutex);
    session_id = session_slot->session_id;
    bool session_active = session_slot->session != NULL;
    iree_slim_mutex_unlock(&server->session_mutex);

    if (!session_active) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    if (!session_slot->bulk_transfer_scheduler) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    } else {
      status = iree_hal_remote_bulk_transfer_scheduler_insert_peer(
          session_slot->bulk_transfer_scheduler, transfer_id, length,
          /*user_value=*/0, &table_transfer);
      transfer_inserted = iree_status_is_ok(status);
      if (iree_status_is_ok(status)) {
        iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
            iree_hal_remote_server_bulk_transfer_storage(table_transfer);
        memset(bulk_transfer, 0, sizeof(*bulk_transfer));
        bulk_transfer->kind =
            IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE;
        iree_hal_remote_server_client_file_write_transfer_t* transfer =
            iree_hal_remote_server_client_file_write_storage(table_transfer);
        transfer->server = server;
        iree_hal_remote_server_retain(transfer->server);
        transfer->session_slot = session_slot;
        transfer->session_id = session_id;
        transfer->local_device = local_device;
        transfer->source_buffer = source_buffer;
        iree_hal_buffer_retain(transfer->source_buffer);
        transfer->source_offset = source_offset;
        transfer->write_flags = flags;
        transfer->ready_context = ready_context;
        ready_context = NULL;
        transfer->initial_wait_semaphore_list = iree_hal_semaphore_list_empty();
        transfer->signal_semaphore_list = iree_hal_semaphore_list_empty();
        status = iree_hal_remote_server_bulk_staging_pool_acquire(
            session_slot->bulk_staging_pool, local_device,
            &transfer->staging_slot);
        if (iree_status_is_ok(status)) {
          transfer->staging_pool = session_slot->bulk_staging_pool;
          transfer->staging_file = transfer->staging_slot->file;
          transfer->staging_contents = transfer->staging_slot->contents;
          transfer->staging_semaphore = transfer->staging_slot->semaphore;
          transfer->last_staging_signal_value =
              transfer->staging_slot->last_signal_value;
          transfer->ready_context->local_semaphore =
              transfer->staging_semaphore;
          iree_hal_semaphore_retain(transfer->ready_context->local_semaphore);
        }
      }
      if (iree_status_is_ok(status)) {
        iree_hal_remote_server_client_file_write_transfer_t* transfer =
            iree_hal_remote_server_client_file_write_storage(table_transfer);
        status = iree_hal_semaphore_list_clone(
            &wait_list, host_allocator, &transfer->initial_wait_semaphore_list);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_remote_server_client_file_write_transfer_t* transfer =
            iree_hal_remote_server_client_file_write_storage(table_transfer);
        status = iree_hal_semaphore_list_clone(
            &signal_list, host_allocator, &transfer->signal_semaphore_list);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_remote_server_client_file_write_transfer_t* transfer =
            iree_hal_remote_server_client_file_write_storage(table_transfer);
        transfer->ready_context->session_id = session_id;
        iree_hal_remote_server_client_file_write_try_send_locked(
            session_slot, session_slot->bulk_channel, table_transfer);
      }
      if (!iree_status_is_ok(status) && transfer_inserted) {
        iree_hal_remote_server_client_file_write_release_transfer(
            session_slot->bulk_transfer_scheduler, table_transfer);
        transfer_inserted = false;
        table_transfer = NULL;
      }
    }
    drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }

  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot, session_slot->bulk_channel,
                    /*returned_sequence=*/0));
  }
  iree_hal_remote_server_client_file_write_ready_context_release(ready_context);
  return status;
}

iree_status_t iree_hal_remote_server_bulk_submit_profile_transfer(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    iree_hal_profile_sink_t* profile_sink, iree_byte_span_t payload) {
  iree_hal_remote_server_t* server = session_slot->server;
  iree_allocator_t host_allocator = server->host_allocator;

  iree_status_t status = iree_ok_status();
  uint64_t profile_sequence = 0;
  if (payload.data_length < sizeof(iree_hal_remote_profile_transfer_header_t)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "profile payload too small for header: %" PRIhsz
                              " bytes",
                              payload.data_length);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_remote_profile_transfer_header_t* header =
        (const iree_hal_remote_profile_transfer_header_t*)payload.data;
    profile_sequence = header->sequence;
    if (profile_sequence == 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "profile payload must have a nonzero callback sequence");
    }
  }

  iree_net_bulk_channel_t* bulk_channel = NULL;
  if (iree_status_is_ok(status)) {
    bool session_active = false;
    iree_slim_mutex_lock(&server->session_mutex);
    session_active = session_slot->session_id == session_id &&
                     session_slot->session != NULL &&
                     session_slot->bulk_channel != NULL &&
                     session_slot->profile_sink == profile_sink;
    if (session_active) {
      bulk_channel = session_slot->bulk_channel;
      iree_net_bulk_channel_retain(bulk_channel);
    }
    iree_slim_mutex_unlock(&server->session_mutex);
    if (!session_active) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }
  }

  bool drain_profile_pending = false;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
    if (!session_slot->bulk_transfer_scheduler) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }
    if (iree_status_is_ok(status)) {
      if (session_slot->profile_pending_transfer_head) {
        status = iree_hal_remote_server_profile_pending_transfer_enqueue_locked(
            session_slot, session_id, profile_sequence, &payload);
        drain_profile_pending = iree_status_is_ok(status);
      } else {
        bool table_full = false;
        status = iree_hal_remote_server_profile_transfer_activate_locked(
            session_slot, bulk_channel, session_id, profile_sequence, &payload,
            &table_full);
        if (iree_status_is_ok(status) && table_full) {
          status =
              iree_hal_remote_server_profile_pending_transfer_enqueue_locked(
                  session_slot, session_id, profile_sequence, &payload);
          drain_profile_pending = iree_status_is_ok(status);
        }
      }
    }
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  }
  if (iree_status_is_ok(status) && drain_profile_pending) {
    status = iree_hal_remote_server_profile_pending_transfer_drain(
        session_slot, bulk_channel, profile_sequence);
  }

  iree_net_bulk_channel_release(bulk_channel);
  iree_allocator_free(host_allocator, payload.data);
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_start(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t total_size, iree_net_bulk_frame_flags_t flags) {
  if (flags != IREE_NET_BULK_FRAME_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported remote server bulk START flags: 0x%02x", flags);
  }

  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_status_t status = iree_ok_status();
  if (!session_slot->bulk_transfer_scheduler) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  }
  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_client_file_read_get_or_insert_locked(
        session_slot, transfer_id, total_size, &table_transfer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_client_file_read_transfer_t* transfer =
        iree_hal_remote_server_client_file_read_storage(table_transfer);
    transfer->flags |=
        IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_START_RECEIVED;
  }
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  if (iree_status_is_ok(status) && session_slot->bulk_channel) {
    iree_status_t failure_status = iree_ok_status();
    iree_hal_remote_bulk_channel_send_result_t send_result =
        iree_hal_remote_bulk_channel_refresh_credit(session_slot->bulk_channel,
                                                    /*operation_user_data=*/0,
                                                    &failure_status);
    if (send_result == IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED) {
      status = failure_status;
    }
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_data(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id,
    uint64_t chunk_offset, uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease) {
  if (flags & ~IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported remote server bulk DATA flags: 0x%02x",
                            flags);
  }

  iree_status_t status = iree_ok_status();
  bool drain_profile_pending = false;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (session_slot->bulk_transfer_scheduler) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        session_slot->bulk_transfer_scheduler, transfer_id);
  }
  if (!table_transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote server bulk DATA unknown transfer_id=%" PRIu64, transfer_id);
  } else {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    if (bulk_transfer->kind !=
        IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "remote server bulk DATA received for "
                                "non-upload transfer_id=%" PRIu64,
                                transfer_id);
    } else {
      iree_hal_remote_server_client_file_read_transfer_t* transfer =
          &bulk_transfer->client_file_read;
      const uint64_t chunk_length = (uint64_t)chunk_data.data_length;
      const bool chunk_range_overflow =
          chunk_offset > UINT64_MAX - chunk_length;
      const uint64_t chunk_end =
          chunk_range_overflow ? UINT64_MAX : chunk_offset + chunk_length;
      const uint64_t total_length =
          iree_net_bulk_transfer_total_size(table_transfer);
      const bool final_chunk =
          iree_all_bits_set(flags, IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK);
      const bool expected_final_chunk =
          !chunk_range_overflow && chunk_end == total_length;
      if (!iree_any_bit_set(
              transfer->flags,
              IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_START_RECEIVED)) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "remote server bulk DATA before START for transfer_id=%" PRIu64,
            transfer_id);
      } else if (chunk_range_overflow || chunk_end > total_length) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "remote server bulk DATA range [%" PRIu64
                             ", %" PRIu64 ") exceeds transfer length %" PRIu64,
                             chunk_offset, chunk_end, total_length);
      } else if (final_chunk != expected_final_chunk) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "remote server bulk DATA final flag mismatch for "
                             "transfer_id=%" PRIu64,
                             transfer_id);
      } else {
        status = iree_hal_remote_bulk_transfer_tracker_record_chunk(
            &transfer->receive_tracker, chunk_offset, chunk_data.data_length);
      }
      if (iree_status_is_ok(status)) {
        iree_net_bulk_chunk_t* chunk = NULL;
        status = iree_net_bulk_receive_window_acquire_chunk(
            session_slot->bulk_receive_window, transfer_id, chunk_offset,
            sequence, flags, chunk_data, lease, /*user_value=*/0, &chunk);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_remote_server_client_file_read_try_process_chunks_locked(
            session_slot);
      }
    }
  }
  drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_flush_receive_window(session_slot);
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot, session_slot->bulk_channel,
                    /*returned_sequence=*/0));
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_complete(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id) {
  iree_status_t status = iree_ok_status();
  bool drain_profile_pending = false;
  uint64_t profile_sequence = 0;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  if (!session_slot->bulk_transfer_scheduler) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  }
  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (iree_status_is_ok(status)) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        session_slot->bulk_transfer_scheduler, transfer_id);
  }
  if (!table_transfer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote server bulk COMPLETE unknown transfer_id=%" PRIu64,
        transfer_id);
  } else {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    switch (bulk_transfer->kind) {
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ: {
        iree_hal_remote_server_client_file_read_transfer_t* transfer =
            &bulk_transfer->client_file_read;
        transfer->flags |=
            IREE_HAL_REMOTE_SERVER_CLIENT_FILE_READ_TRANSFER_FLAG_PEER_COMPLETE;
        if (!iree_hal_remote_bulk_transfer_tracker_is_complete(
                &transfer->receive_tracker)) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "CLIENT_FILE_READ COMPLETE before all DATA "
                                    "for transfer_id=%" PRIu64,
                                    transfer_id);
        } else {
          iree_hal_remote_server_client_file_read_try_finish_locked(
              session_slot, table_transfer);
        }
        break;
      }
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE: {
        iree_hal_remote_server_client_file_write_transfer_t* transfer =
            &bulk_transfer->client_file_write;
        transfer->flags |=
            IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_PEER_COMPLETE;
        iree_hal_remote_server_client_file_write_try_finish_locked(
            session_slot, table_transfer);
        break;
      }
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND: {
        iree_hal_remote_server_profile_transfer_t* transfer =
            &bulk_transfer->profile_send;
        profile_sequence = transfer->sequence;
        transfer->flags |=
            IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_PEER_COMPLETE;
        iree_hal_remote_server_profile_transfer_try_finish_locked(
            session_slot, table_transfer);
        break;
      }
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "remote server bulk COMPLETE received for "
                                  "empty transfer_id=%" PRIu64,
                                  transfer_id);
        break;
    }
  }
  drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  if (iree_status_is_ok(status) && profile_sequence != 0) {
    status = iree_hal_remote_server_profile_observe_transfer(
        session_slot, profile_sequence, iree_ok_status());
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot, session_slot->bulk_channel,
                    /*returned_sequence=*/0));
  }
  return status;
}

iree_status_t iree_hal_remote_server_bulk_on_abort(
    iree_hal_remote_server_session_t* session_slot, uint64_t transfer_id) {
  bool drain_profile_pending = false;
  uint64_t profile_sequence = 0;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_net_bulk_transfer_t* table_transfer = NULL;
  if (session_slot->bulk_transfer_scheduler) {
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        session_slot->bulk_transfer_scheduler, transfer_id);
  }
  if (table_transfer) {
    iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
        iree_hal_remote_server_bulk_transfer_storage(table_transfer);
    switch (bulk_transfer->kind) {
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_READ:
        iree_hal_remote_server_client_file_read_fail_locked(
            session_slot, table_transfer,
            iree_make_status(IREE_STATUS_ABORTED,
                             "remote client aborted bulk transfer"));
        break;
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE:
        iree_hal_remote_server_client_file_write_fail_locked(
            session_slot, table_transfer,
            iree_make_status(IREE_STATUS_ABORTED,
                             "remote client aborted bulk transfer"));
        break;
      case IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND:
        profile_sequence = bulk_transfer->profile_send.sequence;
        iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                            table_transfer);
        break;
      default:
        break;
    }
  }
  drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  iree_status_t status = iree_ok_status();
  if (profile_sequence != 0) {
    status = iree_hal_remote_server_profile_observe_transfer(
        session_slot, profile_sequence,
        iree_make_status(IREE_STATUS_ABORTED,
                         "remote client aborted profile transfer"));
  }
  if (drain_profile_pending) {
    status = iree_status_join(
        status, iree_hal_remote_server_profile_pending_transfer_drain(
                    session_slot, session_slot->bulk_channel,
                    /*returned_sequence=*/0));
  }
  return status;
}

void iree_hal_remote_server_bulk_on_send_complete(
    iree_hal_remote_server_session_t* session_slot,
    uint64_t operation_user_data, iree_status_t status) {
  if (operation_user_data == 0) {
    iree_status_ignore(status);
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }
  const uint64_t transfer_id = operation_user_data;

  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  if (!session_slot->bulk_transfer_scheduler) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_ignore(status);
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }
  iree_net_bulk_transfer_t* table_transfer =
      iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
  if (!table_transfer) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_ignore(status);
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }

  iree_hal_remote_server_bulk_transfer_t* bulk_transfer =
      iree_hal_remote_server_bulk_transfer_storage(table_transfer);
  if (bulk_transfer->kind ==
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_PROFILE_SEND) {
    bool drain_profile_pending = false;
    iree_hal_remote_server_profile_transfer_t* transfer =
        &bulk_transfer->profile_send;
    const uint64_t profile_sequence = transfer->sequence;
    transfer->flags &=
        ~IREE_HAL_REMOTE_SERVER_PROFILE_TRANSFER_FLAG_SEND_PENDING;
    if (iree_status_is_ok(status)) {
      iree_status_t send_status =
          iree_hal_remote_server_profile_transfer_try_send_locked(
              session_slot, session_slot->bulk_channel, table_transfer);
      table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
          session_slot->bulk_transfer_scheduler, transfer_id);
      if (table_transfer) {
        iree_hal_remote_server_profile_transfer_try_finish_locked(
            session_slot, table_transfer);
      }
      drain_profile_pending =
          session_slot->profile_pending_transfer_head != NULL;
      iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
      iree_status_ignore(status);
      if (!iree_status_is_ok(send_status)) {
        iree_status_t observe_status =
            iree_hal_remote_server_profile_observe_transfer(
                session_slot, profile_sequence, send_status);
        iree_status_ignore(observe_status);
      }
      if (drain_profile_pending) {
        iree_status_t drain_status =
            iree_hal_remote_server_profile_pending_transfer_drain(
                session_slot, session_slot->bulk_channel,
                /*returned_sequence=*/0);
        iree_status_ignore(drain_status);
      }
      iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
      return;
    }

    iree_hal_remote_server_profile_transfer_fail_locked(session_slot,
                                                        table_transfer);
    drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_t observe_status =
        iree_hal_remote_server_profile_observe_transfer(
            session_slot, profile_sequence, iree_status_clone(status));
    iree_status_ignore(observe_status);
    iree_status_ignore(status);
    if (drain_profile_pending) {
      iree_status_t drain_status =
          iree_hal_remote_server_profile_pending_transfer_drain(
              session_slot, session_slot->bulk_channel,
              /*returned_sequence=*/0);
      iree_status_ignore(drain_status);
    }
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }

  if (bulk_transfer->kind !=
      IREE_HAL_REMOTE_SERVER_BULK_TRANSFER_KIND_CLIENT_FILE_WRITE) {
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_ignore(status);
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }

  iree_hal_remote_server_client_file_write_transfer_t* transfer =
      &bulk_transfer->client_file_write;
  if (transfer->pending_operation_count > 0) {
    --transfer->pending_operation_count;
  }
  transfer->flags &=
      ~IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_SEND_PENDING;
  if (iree_any_bit_set(
          transfer->flags,
          IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SEND_PENDING)) {
    transfer->flags &=
        ~IREE_HAL_REMOTE_SERVER_CLIENT_FILE_WRITE_TRANSFER_FLAG_STAGING_SEND_PENDING;
  }
  if (iree_status_is_ok(status)) {
    bool drain_profile_pending = false;
    iree_hal_remote_server_client_file_write_try_send_locked(
        session_slot, session_slot->bulk_channel, table_transfer);
    table_transfer = iree_hal_remote_bulk_transfer_scheduler_lookup(
        session_slot->bulk_transfer_scheduler, transfer_id);
    if (table_transfer) {
      iree_hal_remote_server_client_file_write_try_finish_locked(
          session_slot, table_transfer);
    }
    drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
    iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
    iree_status_ignore(status);
    if (drain_profile_pending) {
      iree_status_t drain_status =
          iree_hal_remote_server_profile_pending_transfer_drain(
              session_slot, session_slot->bulk_channel,
              /*returned_sequence=*/0);
      iree_status_ignore(drain_status);
    }
    iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
    return;
  }

  bool drain_profile_pending = false;
  iree_hal_remote_server_client_file_write_fail_locked(
      session_slot, table_transfer, iree_status_clone(status));
  drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  iree_status_ignore(status);
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot, session_slot->bulk_channel,
            /*returned_sequence=*/0);
    iree_status_ignore(drain_status);
  }
  iree_hal_remote_server_session_try_complete_bulk_drain(session_slot);
}

void iree_hal_remote_server_bulk_on_credit(
    iree_hal_remote_server_session_t* session_slot) {
  uint64_t failed_profile_sequence = 0;
  bool drain_profile_pending = false;
  iree_slim_mutex_lock(&session_slot->bulk_transfer_mutex);
  iree_hal_remote_server_client_file_write_try_send_all_locked(
      session_slot, session_slot->bulk_channel);
  iree_status_t status =
      iree_hal_remote_server_profile_transfer_try_send_all_locked(
          session_slot, session_slot->bulk_channel, &failed_profile_sequence);
  drain_profile_pending = session_slot->profile_pending_transfer_head != NULL;
  iree_slim_mutex_unlock(&session_slot->bulk_transfer_mutex);
  if (!iree_status_is_ok(status) && failed_profile_sequence != 0) {
    iree_status_t observe_status =
        iree_hal_remote_server_profile_observe_transfer(
            session_slot, failed_profile_sequence, status);
    iree_status_ignore(observe_status);
  } else {
    iree_status_ignore(status);
  }
  if (drain_profile_pending) {
    iree_status_t drain_status =
        iree_hal_remote_server_profile_pending_transfer_drain(
            session_slot, session_slot->bulk_channel,
            /*returned_sequence=*/0);
    iree_status_ignore(drain_status);
  }
}
