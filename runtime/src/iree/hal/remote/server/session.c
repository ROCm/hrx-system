// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/session.h"

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier.h"
#include "iree/async/semaphore.h"
#include "iree/hal/remote/protocol/commands.h"
#include "iree/hal/remote/protocol/common.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/protocol/queue.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/net/channel/bulk/bulk_channel.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/status_wire.h"

// Maximum stack-allocated buffer for serializing iree_status_t to wire format.
// Sufficient for typical error messages with source location + 1-2 annotations.
// Statuses exceeding this are sent code-only (message lost, code preserved).
#define IREE_HAL_REMOTE_MAX_STATUS_WIRE_SIZE 232

//===----------------------------------------------------------------------===//
// Session slot helpers
//===----------------------------------------------------------------------===//

// Context for a pending command completion on the server. Heap-allocated before
// local submission and released after sending the ordered ADVANCE frame or when
// the session shuts down. Used for all queue operations.
typedef struct iree_hal_remote_server_command_completion_t {
  // Timepoint registered on the local signal semaphore.
  iree_async_semaphore_timepoint_t timepoint;
  // Intrusive node held by completed_signal_window until ADVANCE is ordered.
  iree_net_sequence_node_t sequence_node;
  // Owning server retained while the completion can fire asynchronously.
  iree_hal_remote_server_t* server;
  // Session slot that owned the command when it was submitted.
  iree_hal_remote_server_session_t* session_slot;
  // Session ID expected in |session_slot| when the completion fires.
  uint64_t session_id;
  // Queue channel retained for the eventual ADVANCE frame.
  iree_net_queue_channel_t* queue_channel;
  // Local signal semaphore retained for completion tracking.
  iree_hal_semaphore_t* local_semaphore;
  // Host allocator used to free this context.
  iree_allocator_t host_allocator;
  // Single-entry frontier signaled by this command.
  iree_async_single_frontier_t signal_frontier;
  // Number of resolution entries piggybacked on ADVANCE.
  uint16_t resolution_count;
  // Padding reserved to keep following payload storage aligned.
  uint16_t resolution_padding[3];
  // BUFFER_ALLOCA provisional-to-resolved mapping, valid when count is one.
  iree_hal_remote_resolution_entry_t resolution;
} iree_hal_remote_server_command_completion_t;

static void iree_hal_remote_server_release_command_completion(
    iree_hal_remote_server_command_completion_t* completion) {
  if (!completion) return;
  iree_allocator_t host_allocator = completion->host_allocator;
  iree_net_queue_channel_release(completion->queue_channel);
  iree_hal_semaphore_release(completion->local_semaphore);
  iree_hal_remote_server_release(completion->server);
  iree_allocator_free(host_allocator, completion);
}

static void iree_hal_remote_server_free_command_completion_nodes(
    iree_net_sequence_node_t* pending_list) {
  while (pending_list) {
    iree_net_sequence_node_t* next = pending_list->next;
    iree_hal_remote_server_command_completion_t* completion = iree_containerof(
        pending_list, iree_hal_remote_server_command_completion_t,
        sequence_node);
    iree_hal_remote_server_release_command_completion(completion);
    pending_list = next;
  }
}

typedef struct iree_hal_remote_server_resource_release_node_t {
  // Intrusive node held by observed_submission_window until release is safe.
  iree_net_sequence_node_t sequence_node;
  // Host allocator used to free this node.
  iree_allocator_t host_allocator;
  // Number of resource IDs in the trailing array.
  uint32_t resource_count;
  // Reserved padding for stable trailing-array alignment.
  uint32_t reserved;
  // Resource IDs to release once the required submission epoch is observed.
  iree_hal_remote_resource_id_t resource_ids[];
} iree_hal_remote_server_resource_release_node_t;

static void iree_hal_remote_server_free_resource_release_nodes(
    iree_net_sequence_node_t* pending_list) {
  while (pending_list) {
    iree_net_sequence_node_t* next = pending_list->next;
    iree_hal_remote_server_resource_release_node_t* release_node =
        iree_containerof(pending_list,
                         iree_hal_remote_server_resource_release_node_t,
                         sequence_node);
    iree_allocator_t host_allocator = release_node->host_allocator;
    iree_allocator_free(host_allocator, release_node);
    pending_list = next;
  }
}

static void iree_hal_remote_server_session_take_window_nodes(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_sequence_node_t** out_pending_releases,
    iree_net_sequence_node_t** out_pending_completions) {
  *out_pending_releases = NULL;
  *out_pending_completions = NULL;
  iree_net_sequence_window_take_pending(
      &session_slot->observed_submission_window, out_pending_releases);
  iree_net_sequence_window_deinitialize(
      &session_slot->observed_submission_window);

  iree_net_sequence_window_take_pending(&session_slot->completed_signal_window,
                                        out_pending_completions);
  iree_net_sequence_window_deinitialize(&session_slot->completed_signal_window);
}

void iree_hal_remote_server_session_deinitialize_windows(
    iree_hal_remote_server_session_t* session_slot) {
  iree_net_sequence_node_t* pending_releases = NULL;
  iree_net_sequence_node_t* pending_completions = NULL;
  iree_hal_remote_server_session_take_window_nodes(
      session_slot, &pending_releases, &pending_completions);
  iree_hal_remote_server_free_resource_release_nodes(pending_releases);
  iree_hal_remote_server_free_command_completion_nodes(pending_completions);
}

// Finds the slot holding the given session. Returns -1 if not found.
static int32_t iree_hal_remote_server_find_session_slot(
    iree_hal_remote_server_t* server, iree_net_session_t* session) {
  for (uint32_t i = 0; i < server->options.max_connections; ++i) {
    if (server->sessions[i].session == session) return (int32_t)i;
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// Session removal
//===----------------------------------------------------------------------===//

void iree_hal_remote_server_remove_session(iree_hal_remote_server_t* server,
                                           iree_net_session_t* session) {
  iree_net_queue_channel_t* queue_channel = NULL;
  iree_net_bulk_channel_t* bulk_channel = NULL;
  iree_hal_remote_server_stopped_callback_t stopped_callback;
  memset(&stopped_callback, 0, sizeof(stopped_callback));

  // Snapshot the resource table and epoch mapping for cleanup outside the lock.
  iree_hal_remote_resource_table_t resource_table;
  memset(&resource_table, 0, sizeof(resource_table));
  iree_hal_remote_server_epoch_slot_state_t* epoch_map_states = NULL;
  iree_async_axis_t* epoch_map_axes = NULL;
  uint64_t* epoch_map_epochs = NULL;
  iree_hal_semaphore_t** epoch_map_semaphores = NULL;
  iree_host_size_t epoch_map_capacity = 0;
  iree_hal_remote_resource_id_t* prov_map_provisionals = NULL;
  iree_hal_remote_resource_id_t* prov_map_resolved = NULL;
  iree_net_sequence_node_t* pending_releases = NULL;
  iree_net_sequence_node_t* pending_completions = NULL;

  iree_slim_mutex_lock(&server->session_mutex);
  int32_t slot = iree_hal_remote_server_find_session_slot(server, session);
  if (slot >= 0) {
    // Snapshot references to clean up outside the lock.
    queue_channel = server->sessions[slot].queue_channel;
    server->sessions[slot].queue_channel = NULL;
    bulk_channel = server->sessions[slot].bulk_channel;
    server->sessions[slot].bulk_channel = NULL;

    resource_table = server->sessions[slot].resource_table;
    memset(&server->sessions[slot].resource_table, 0,
           sizeof(server->sessions[slot].resource_table));

    epoch_map_states = server->sessions[slot].epoch_semaphore_map.states;
    epoch_map_axes = server->sessions[slot].epoch_semaphore_map.axes;
    epoch_map_epochs = server->sessions[slot].epoch_semaphore_map.epochs;
    epoch_map_semaphores =
        server->sessions[slot].epoch_semaphore_map.semaphores;
    epoch_map_capacity = server->sessions[slot].epoch_semaphore_map.capacity;
    memset(&server->sessions[slot].epoch_semaphore_map, 0,
           sizeof(server->sessions[slot].epoch_semaphore_map));

    prov_map_provisionals =
        server->sessions[slot].provisional_map.provisional_ids;
    prov_map_resolved = server->sessions[slot].provisional_map.resolved_ids;
    memset(&server->sessions[slot].provisional_map, 0,
           sizeof(server->sessions[slot].provisional_map));

    iree_hal_remote_server_session_take_window_nodes(
        &server->sessions[slot], &pending_releases, &pending_completions);

    server->sessions[slot].session = NULL;
    server->sessions[slot].session_id = 0;
    --server->active_session_count;

    // Check if shutdown is now complete.
    if (server->state == IREE_HAL_REMOTE_SERVER_STATE_STOPPING &&
        server->active_session_count == 0 && !server->listener) {
      server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
      stopped_callback = server->stopped_callback;
    }
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  if (slot < 0) return;  // Already removed (e.g., double callback).

  // Release all resources in the table.
  iree_hal_remote_resource_table_deinitialize(&resource_table,
                                              server->host_allocator);

  // Release all local semaphores in the epoch mapping.
  for (iree_host_size_t i = 0; i < epoch_map_capacity; ++i) {
    iree_hal_semaphore_release(epoch_map_semaphores[i]);
  }
  iree_allocator_free(server->host_allocator, epoch_map_states);
  iree_allocator_free(server->host_allocator, epoch_map_axes);
  iree_allocator_free(server->host_allocator, epoch_map_epochs);
  iree_allocator_free(server->host_allocator, epoch_map_semaphores);

  // Free provisional mapping arrays.
  iree_allocator_free(server->host_allocator, prov_map_provisionals);
  iree_allocator_free(server->host_allocator, prov_map_resolved);

  // Release owner-managed nodes that were pending in sequence windows.
  iree_hal_remote_server_free_resource_release_nodes(pending_releases);
  iree_hal_remote_server_free_command_completion_nodes(pending_completions);

  // Detach channels from their endpoints before releasing the session. Command
  // completions may hold retained references to the queue channel that outlive
  // the session. Detach clears endpoint callbacks while endpoints are alive and
  // zeroes endpoint references so eventual channel destroy does not UAF.
  iree_net_bulk_channel_detach(bulk_channel);
  iree_net_bulk_channel_release(bulk_channel);
  iree_net_queue_channel_detach(queue_channel);
  iree_net_queue_channel_release(queue_channel);
  iree_net_session_release(session);

  if (stopped_callback.fn) {
    stopped_callback.fn(stopped_callback.user_data);
  }
}

//===----------------------------------------------------------------------===//
// Epoch→semaphore mapping
//===----------------------------------------------------------------------===//

static uint64_t iree_hal_remote_server_mix_u64(uint64_t value) {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return value;
}

static iree_host_size_t iree_hal_remote_server_epoch_semaphore_slot(
    iree_async_axis_t axis, uint64_t epoch, iree_host_size_t capacity) {
  uint64_t hash = iree_hal_remote_server_mix_u64(axis);
  hash ^= iree_hal_remote_server_mix_u64(epoch);
  return (iree_host_size_t)(hash & (capacity - 1));
}

static iree_host_size_t iree_hal_remote_server_epoch_map_capacity(
    iree_host_size_t minimum_capacity) {
  iree_host_size_t capacity = 64;
  while (capacity < minimum_capacity) {
    if (capacity > IREE_HOST_SIZE_MAX / 2) return 0;
    capacity *= 2;
  }
  return capacity;
}

static iree_status_t iree_hal_remote_server_resize_epoch_semaphore_map(
    iree_hal_remote_server_session_t* session_slot,
    iree_host_size_t minimum_capacity, iree_allocator_t host_allocator) {
  iree_host_size_t new_capacity =
      iree_hal_remote_server_epoch_map_capacity(minimum_capacity);
  if (new_capacity == 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "epoch semaphore map capacity overflow");
  }

  iree_hal_remote_server_epoch_slot_state_t* new_states = NULL;
  iree_async_axis_t* new_axes = NULL;
  uint64_t* new_epochs = NULL;
  iree_hal_semaphore_t** new_semaphores = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, new_capacity, sizeof(*new_states), (void**)&new_states);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, new_capacity,
                                         sizeof(*new_axes), (void**)&new_axes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, new_capacity, sizeof(*new_epochs), (void**)&new_epochs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, new_capacity,
                                         sizeof(*new_semaphores),
                                         (void**)&new_semaphores);
  }

  if (iree_status_is_ok(status)) {
    memset(new_states, 0, new_capacity * sizeof(*new_states));
    memset(new_axes, 0, new_capacity * sizeof(*new_axes));
    memset(new_epochs, 0, new_capacity * sizeof(*new_epochs));
    memset(new_semaphores, 0, new_capacity * sizeof(*new_semaphores));

    for (iree_host_size_t i = 0; i < session_slot->epoch_semaphore_map.capacity;
         ++i) {
      if (session_slot->epoch_semaphore_map.states &&
          session_slot->epoch_semaphore_map.states[i] !=
              IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_OCCUPIED) {
        continue;
      }
      iree_hal_semaphore_t* semaphore =
          session_slot->epoch_semaphore_map.semaphores[i];
      if (!semaphore) continue;
      iree_async_axis_t axis = session_slot->epoch_semaphore_map.axes[i];
      uint64_t epoch = session_slot->epoch_semaphore_map.epochs[i];
      iree_host_size_t slot = iree_hal_remote_server_epoch_semaphore_slot(
          axis, epoch, new_capacity);
      while (new_states[slot] == IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_OCCUPIED) {
        slot = (slot + 1) & (new_capacity - 1);
      }
      new_states[slot] = IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_OCCUPIED;
      new_axes[slot] = axis;
      new_epochs[slot] = epoch;
      new_semaphores[slot] = semaphore;
    }

    iree_allocator_free(host_allocator,
                        session_slot->epoch_semaphore_map.states);
    iree_allocator_free(host_allocator, session_slot->epoch_semaphore_map.axes);
    iree_allocator_free(host_allocator,
                        session_slot->epoch_semaphore_map.epochs);
    iree_allocator_free(host_allocator,
                        session_slot->epoch_semaphore_map.semaphores);
    session_slot->epoch_semaphore_map.states = new_states;
    session_slot->epoch_semaphore_map.axes = new_axes;
    session_slot->epoch_semaphore_map.epochs = new_epochs;
    session_slot->epoch_semaphore_map.semaphores = new_semaphores;
    session_slot->epoch_semaphore_map.capacity = new_capacity;
    session_slot->epoch_semaphore_map.used_count =
        session_slot->epoch_semaphore_map.count;
  } else {
    iree_allocator_free(host_allocator, new_states);
    iree_allocator_free(host_allocator, new_axes);
    iree_allocator_free(host_allocator, new_epochs);
    iree_allocator_free(host_allocator, new_semaphores);
  }

  return status;
}

// Stores a mapping from signal frontier entry to local semaphore.
// Retains the semaphore.
static iree_status_t iree_hal_remote_server_store_epoch_semaphore(
    iree_hal_remote_server_session_t* session_slot, iree_async_axis_t axis,
    uint64_t epoch, iree_hal_semaphore_t* semaphore,
    iree_allocator_t host_allocator) {
  iree_host_size_t minimum_used_capacity =
      session_slot->epoch_semaphore_map.used_count + 1;
  if (minimum_used_capacity * 4 >=
      session_slot->epoch_semaphore_map.capacity * 3) {
    IREE_RETURN_IF_ERROR(iree_hal_remote_server_resize_epoch_semaphore_map(
        session_slot, (session_slot->epoch_semaphore_map.count + 1) * 2,
        host_allocator));
  }

  iree_host_size_t slot = iree_hal_remote_server_epoch_semaphore_slot(
      axis, epoch, session_slot->epoch_semaphore_map.capacity);
  iree_host_size_t first_tombstone = IREE_HOST_SIZE_MAX;
  while (session_slot->epoch_semaphore_map.states[slot] !=
         IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_EMPTY) {
    if (session_slot->epoch_semaphore_map.states[slot] ==
        IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_TOMBSTONE) {
      if (first_tombstone == IREE_HOST_SIZE_MAX) first_tombstone = slot;
    } else if (session_slot->epoch_semaphore_map.axes[slot] == axis &&
               session_slot->epoch_semaphore_map.epochs[slot] == epoch) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "duplicate signal frontier entry axis=0x%016" PRIx64
          " epoch=%" PRIu64,
          axis, epoch);
    }
    slot = (slot + 1) & (session_slot->epoch_semaphore_map.capacity - 1);
  }
  if (first_tombstone != IREE_HOST_SIZE_MAX) {
    slot = first_tombstone;
  } else {
    ++session_slot->epoch_semaphore_map.used_count;
  }

  session_slot->epoch_semaphore_map.states[slot] =
      IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_OCCUPIED;
  session_slot->epoch_semaphore_map.axes[slot] = axis;
  session_slot->epoch_semaphore_map.epochs[slot] = epoch;
  session_slot->epoch_semaphore_map.semaphores[slot] = semaphore;
  ++session_slot->epoch_semaphore_map.count;
  iree_hal_semaphore_retain(semaphore);
  return iree_ok_status();
}

// Looks up the local semaphore for a given frontier entry. Returns NULL if not
// found. The returned pointer is borrowed — the epoch mapping retains it.
static iree_hal_semaphore_t* iree_hal_remote_server_lookup_epoch_semaphore(
    iree_hal_remote_server_session_t* session_slot, iree_async_axis_t axis,
    uint64_t epoch) {
  if (session_slot->epoch_semaphore_map.capacity == 0) return NULL;
  iree_host_size_t slot = iree_hal_remote_server_epoch_semaphore_slot(
      axis, epoch, session_slot->epoch_semaphore_map.capacity);
  while (session_slot->epoch_semaphore_map.states[slot] !=
         IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_EMPTY) {
    if (session_slot->epoch_semaphore_map.states[slot] ==
            IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_OCCUPIED &&
        session_slot->epoch_semaphore_map.axes[slot] == axis &&
        session_slot->epoch_semaphore_map.epochs[slot] == epoch) {
      return session_slot->epoch_semaphore_map.semaphores[slot];
    }
    slot = (slot + 1) & (session_slot->epoch_semaphore_map.capacity - 1);
  }
  return NULL;
}

// Removes the local semaphore for a given frontier entry and transfers the map
// retain to the caller. Returns NULL when no entry was found.
static iree_hal_semaphore_t* iree_hal_remote_server_remove_epoch_semaphore(
    iree_hal_remote_server_session_t* session_slot, iree_async_axis_t axis,
    uint64_t epoch) {
  if (session_slot->epoch_semaphore_map.capacity == 0) return NULL;
  iree_host_size_t slot = iree_hal_remote_server_epoch_semaphore_slot(
      axis, epoch, session_slot->epoch_semaphore_map.capacity);
  while (session_slot->epoch_semaphore_map.states[slot] !=
         IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_EMPTY) {
    if (session_slot->epoch_semaphore_map.states[slot] ==
            IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_OCCUPIED &&
        session_slot->epoch_semaphore_map.axes[slot] == axis &&
        session_slot->epoch_semaphore_map.epochs[slot] == epoch) {
      iree_hal_semaphore_t* semaphore =
          session_slot->epoch_semaphore_map.semaphores[slot];
      session_slot->epoch_semaphore_map.states[slot] =
          IREE_HAL_REMOTE_SERVER_EPOCH_SLOT_TOMBSTONE;
      session_slot->epoch_semaphore_map.axes[slot] = 0;
      session_slot->epoch_semaphore_map.epochs[slot] = 0;
      session_slot->epoch_semaphore_map.semaphores[slot] = NULL;
      --session_slot->epoch_semaphore_map.count;
      return semaphore;
    }
    slot = (slot + 1) & (session_slot->epoch_semaphore_map.capacity - 1);
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Command completion
//===----------------------------------------------------------------------===//

// Sends an error ADVANCE frame to the client. The ADVANCE carries the full
// serialized iree_status_t so the client can reconstruct the error with source
// locations, messages, and annotations. The status is consumed by this
// function.
//
// On the client side, the non-zero status_code in the advance payload triggers
// frontier_tracker_fail_axis() which fails the semaphore and surfaces the error
// at iree_hal_semaphore_wait().
static void iree_hal_remote_server_send_error_advance(
    iree_net_queue_channel_t* queue_channel,
    const iree_async_frontier_t* signal_frontier, iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Build the advance payload header with the error status code.
  iree_hal_remote_advance_payload_t advance_header;
  memset(&advance_header, 0, sizeof(advance_header));
  advance_header.status_code = (uint16_t)iree_status_code(status);

  // Serialize the full status for the client.
  iree_host_size_t wire_size = 0;
  iree_net_status_wire_size(status, &wire_size);
  uint8_t wire_buffer[IREE_HAL_REMOTE_MAX_STATUS_WIRE_SIZE];
  if (wire_size <= sizeof(wire_buffer)) {
    iree_status_t serialize_status = iree_net_status_wire_serialize(
        status, iree_make_byte_span(wire_buffer, wire_size));
    if (iree_status_is_ok(serialize_status)) {
      advance_header.status_wire_length = (uint32_t)wire_size;
      iree_async_span_t spans[2] = {
          iree_async_span_from_ptr(&advance_header, sizeof(advance_header)),
          iree_async_span_from_ptr(wire_buffer, wire_size),
      };
      iree_async_span_list_t payload = {spans, 2};
      iree_status_ignore(iree_net_queue_channel_send_advance(
          queue_channel, signal_frontier, payload, 0));
    } else {
      // Serialization failed; send code-only error ADVANCE.
      iree_status_ignore(serialize_status);
      iree_async_span_t spans[1] = {
          iree_async_span_from_ptr(&advance_header, sizeof(advance_header)),
      };
      iree_async_span_list_t payload = {spans, 1};
      iree_status_ignore(iree_net_queue_channel_send_advance(
          queue_channel, signal_frontier, payload, 0));
    }
  } else {
    // Status too large for stack storage; send code-only.
    iree_async_span_t spans[1] = {
        iree_async_span_from_ptr(&advance_header, sizeof(advance_header)),
    };
    iree_async_span_list_t payload = {spans, 1};
    iree_status_ignore(iree_net_queue_channel_send_advance(
        queue_channel, signal_frontier, payload, 0));
  }

  iree_status_free(status);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_remote_server_send_success_advance(
    iree_hal_remote_server_command_completion_t* completion) {
  iree_async_frontier_t* signal_frontier =
      iree_async_single_frontier_as_frontier(&completion->signal_frontier);

  if (completion->resolution_count > 0) {
    // BUFFER_ALLOCA: include resolution entries in the ADVANCE payload.
    iree_hal_remote_advance_payload_t advance_header;
    memset(&advance_header, 0, sizeof(advance_header));
    advance_header.resolution_count = completion->resolution_count;
    iree_async_span_t spans[2] = {
        iree_async_span_from_ptr(&advance_header, sizeof(advance_header)),
        iree_async_span_from_ptr(&completion->resolution,
                                 sizeof(completion->resolution)),
    };
    iree_async_span_list_t payload = {spans, 2};
    iree_status_t send_status = iree_net_queue_channel_send_advance(
        completion->queue_channel, signal_frontier, payload,
        /*operation_user_data=*/0);
    iree_status_ignore(send_status);
  } else {
    // Non-alloca: empty ADVANCE payload.
    iree_async_span_list_t empty_payload = {NULL, 0};
    iree_status_t send_status = iree_net_queue_channel_send_advance(
        completion->queue_channel, signal_frontier, empty_payload,
        /*operation_user_data=*/0);
    iree_status_ignore(send_status);
  }
}

static void iree_hal_remote_server_process_ready_command_completions(
    iree_net_sequence_node_t* ready_list) {
  while (ready_list) {
    iree_net_sequence_node_t* next = ready_list->next;
    iree_hal_remote_server_command_completion_t* completion = iree_containerof(
        ready_list, iree_hal_remote_server_command_completion_t, sequence_node);
    iree_hal_remote_server_send_success_advance(completion);
    iree_hal_remote_server_release_command_completion(completion);
    ready_list = next;
  }
}

// Fired by the local device's semaphore when the queue operation completes.
// Successful completions retire their local semaphore immediately but only send
// ADVANCE frames in contiguous epoch order. Failed completions fail the client
// frontier immediately because the axis is terminal after an execution error.
static void iree_hal_remote_server_on_command_complete(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_remote_server_command_completion_t* completion =
      (iree_hal_remote_server_command_completion_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_frontier_t* signal_frontier =
      iree_async_single_frontier_as_frontier(&completion->signal_frontier);
  iree_async_axis_t signal_axis = signal_frontier->entries[0].axis;
  uint64_t signal_epoch = signal_frontier->entries[0].epoch;
  iree_net_sequence_node_t* ready_completions = NULL;
  iree_hal_semaphore_t* removed_semaphore = NULL;
  bool completion_transferred = false;
  bool session_active = false;

  iree_hal_remote_server_t* server = completion->server;
  iree_hal_remote_server_session_t* session_slot = completion->session_slot;
  iree_slim_mutex_lock(&server->session_mutex);
  session_active = session_slot->session_id == completion->session_id &&
                   session_slot->session != NULL;
  if (session_active) {
    removed_semaphore = iree_hal_remote_server_remove_epoch_semaphore(
        session_slot, signal_axis, signal_epoch);
    if (iree_status_is_ok(status)) {
      if (!removed_semaphore) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "missing local semaphore for completed signal epoch %" PRIu64,
            signal_epoch);
      } else if (iree_net_sequence_window_has_observed(
                     &session_slot->completed_signal_window, signal_epoch)) {
        status = iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "duplicate command completion for signal epoch %" PRIu64,
            signal_epoch);
      } else {
        status = iree_net_sequence_window_observe(
            &session_slot->completed_signal_window, signal_epoch,
            &ready_completions);
      }
      if (iree_status_is_ok(status)) {
        if (signal_epoch <= iree_net_sequence_window_observed(
                                &session_slot->completed_signal_window)) {
          completion->sequence_node.next = ready_completions;
          completion->sequence_node.sequence = signal_epoch;
          ready_completions = &completion->sequence_node;
          completion_transferred = true;
        } else {
          status = iree_net_sequence_window_defer_until(
              &session_slot->completed_signal_window, signal_epoch,
              &completion->sequence_node, &ready_completions);
          if (iree_status_is_ok(status)) {
            completion_transferred = true;
          }
        }
      }
    }
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  iree_hal_semaphore_release(removed_semaphore);

  if (session_active) {
    if (iree_status_is_ok(status)) {
      iree_hal_remote_server_process_ready_command_completions(
          ready_completions);
    } else {
      // Execution failed locally, or completion ordering bookkeeping detected
      // a protocol violation. Fail the client axis immediately.
      iree_hal_remote_server_send_error_advance(completion->queue_channel,
                                                signal_frontier, status);
    }
  } else {
    iree_status_ignore(status);
  }

  if (!completion_transferred) {
    iree_hal_remote_server_release_command_completion(completion);
  }

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Provisional→resolved resource ID mapping
//===----------------------------------------------------------------------===//

// Removes a provisional mapping by provisional_id. Called when the resolved
// resource is released to prevent unbounded map growth.
static void iree_hal_remote_server_remove_provisional(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id) {
  for (iree_host_size_t i = 0; i < session_slot->provisional_map.count; ++i) {
    if (session_slot->provisional_map.provisional_ids[i] == provisional_id) {
      // Swap-remove with the last entry.
      iree_host_size_t last = session_slot->provisional_map.count - 1;
      if (i != last) {
        session_slot->provisional_map.provisional_ids[i] =
            session_slot->provisional_map.provisional_ids[last];
        session_slot->provisional_map.resolved_ids[i] =
            session_slot->provisional_map.resolved_ids[last];
      }
      --session_slot->provisional_map.count;
      return;
    }
  }
}

// Stores a provisional→resolved resource ID mapping. Used by BUFFER_ALLOCA
// so that subsequent commands referencing the provisional ID can be resolved.
static iree_status_t iree_hal_remote_server_store_provisional(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id,
    iree_hal_remote_resource_id_t resolved_id,
    iree_allocator_t host_allocator) {
  // Cap the map at the resource table capacity. There can never be more live
  // provisionals than resource table slots. Use the actual table capacity
  // since it's set at initialization time.
  if (session_slot->provisional_map.count >=
      session_slot->resource_table.capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "provisional map full");
  }
  iree_host_size_t minimum_capacity = session_slot->provisional_map.count + 1;
  if (minimum_capacity > session_slot->provisional_map.capacity) {
    iree_host_size_t prov_capacity = session_slot->provisional_map.capacity;
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        host_allocator, minimum_capacity, sizeof(iree_hal_remote_resource_id_t),
        &prov_capacity,
        (void**)&session_slot->provisional_map.provisional_ids));
    iree_host_size_t res_capacity = session_slot->provisional_map.capacity;
    iree_status_t status = iree_allocator_grow_array(
        host_allocator, minimum_capacity, sizeof(iree_hal_remote_resource_id_t),
        &res_capacity, (void**)&session_slot->provisional_map.resolved_ids);
    if (!iree_status_is_ok(status)) {
      session_slot->provisional_map.capacity = prov_capacity;
      return status;
    }
    session_slot->provisional_map.capacity =
        iree_min(prov_capacity, res_capacity);
  }
  iree_host_size_t index = session_slot->provisional_map.count++;
  session_slot->provisional_map.provisional_ids[index] = provisional_id;
  session_slot->provisional_map.resolved_ids[index] = resolved_id;
  return iree_ok_status();
}

// Resolves a resource ID that may be provisional. If the ID has the
// PROVISIONAL flag set, looks up the mapping and returns the resolved ID.
// If not provisional, returns the ID unchanged.
static iree_hal_remote_resource_id_t iree_hal_remote_server_resolve_resource_id(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t resource_id) {
  if (!IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(resource_id)) {
    return resource_id;
  }
  for (iree_host_size_t i = 0; i < session_slot->provisional_map.count; ++i) {
    if (session_slot->provisional_map.provisional_ids[i] == resource_id) {
      return session_slot->provisional_map.resolved_ids[i];
    }
  }
  return resource_id;  // Not found — return as-is (will fail in table lookup).
}

static iree_status_t iree_hal_remote_server_resolve_command_buffer_ref(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t buffer_id, uint32_t buffer_slot,
    uint64_t offset, uint64_t length, const char* command_name,
    iree_hal_buffer_ref_t* out_ref) {
  if (buffer_id == 0) {
    *out_ref = iree_hal_make_indirect_buffer_ref(buffer_slot, offset, length);
    return iree_ok_status();
  }

  iree_hal_remote_resource_id_t resolved_id =
      iree_hal_remote_server_resolve_resource_id(session_slot, buffer_id);
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &session_slot->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
          resolved_id);
  if (!buffer) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "%s buffer 0x%016" PRIx64 " not found",
                            command_name, resolved_id);
  }
  *out_ref = iree_hal_make_buffer_ref(buffer, offset, length);
  return iree_ok_status();
}

// Resolves a wire wait_frontier to a local wait semaphore list. For each
// (axis, epoch) entry in the wait frontier, completed epochs are skipped and
// live epochs are resolved through the epoch mapping. All resolved semaphores
// use payload_value=1 (each local semaphore signals to 1 on completion).
//
// |out_semaphores| and |out_values| are caller-provided arrays of |capacity|.
// Returns the number of resolved entries in |out_count|. Resolved semaphores
// are retained and must be released by the caller.
static iree_status_t iree_hal_remote_server_resolve_wait_frontier(
    iree_hal_remote_server_session_t* session_slot,
    const iree_async_frontier_t* wait_frontier,
    iree_hal_semaphore_t** out_semaphores, uint64_t* out_values,
    iree_host_size_t capacity, iree_host_size_t* out_count) {
  *out_count = 0;
  if (!wait_frontier) return iree_ok_status();
  for (uint8_t i = 0; i < wait_frontier->entry_count; ++i) {
    if (iree_net_sequence_window_has_observed(
            &session_slot->completed_signal_window,
            wait_frontier->entries[i].epoch)) {
      continue;
    }

    iree_hal_semaphore_t* local_semaphore =
        iree_hal_remote_server_lookup_epoch_semaphore(
            session_slot, wait_frontier->entries[i].axis,
            wait_frontier->entries[i].epoch);
    if (!local_semaphore) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no local semaphore for wait frontier "
                              "axis=0x%016" PRIx64 " epoch=%" PRIu64,
                              wait_frontier->entries[i].axis,
                              wait_frontier->entries[i].epoch);
    }
    if (*out_count >= capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "wait frontier exceeds local semaphore capacity");
    }
    out_semaphores[*out_count] = local_semaphore;
    iree_hal_semaphore_retain(local_semaphore);
    out_values[*out_count] = 1;
    ++*out_count;
  }
  return iree_ok_status();
}

// Common setup for a queue operation on the server: creates a local signal
// semaphore, stores the epoch mapping, submits the operation to the local
// device, and registers a timepoint to send ADVANCE on completion.
//
// |submit_fn| is called to perform the actual device queue operation, receiving
// the local wait and signal semaphore lists. The caller provides the specific
// device queue call (fill, copy, update, barrier) via this callback.
typedef iree_status_t (*iree_hal_remote_server_submit_fn_t)(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list, iree_hal_semaphore_list_t signal_list);

static iree_status_t iree_hal_remote_server_process_ready_resource_releases(
    iree_hal_remote_server_session_t* entry,
    iree_net_sequence_node_t* ready_list);

// Context passed to op submit callbacks. Carries the command payload and
// session slot reference. Alloca callbacks also populate resolution data
// that gets piggybacked on the ADVANCE frame.
typedef struct iree_hal_remote_server_op_context_t {
  // Session slot whose resource tables are used by the operation.
  iree_hal_remote_server_session_t* session_slot;
  // Queue command payload being decoded.
  iree_const_byte_span_t command_data;
  // Populated by BUFFER_ALLOCA callback to piggyback resolution on ADVANCE.
  uint16_t resolution_count;
  // BUFFER_ALLOCA provisional-to-resolved mapping copied into completion.
  iree_hal_remote_resolution_entry_t resolution;
} iree_hal_remote_server_op_context_t;

// The submit_fn callback may populate resolution data on the op_context
// (e.g., BUFFER_ALLOCA stores provisional→resolved mapping). After the
// callback returns, submit_command checks the op_context for resolution
// data and copies it into the completion context for piggybacking on the
// ADVANCE frame. Non-alloca ops leave resolution_count=0.
static iree_status_t iree_hal_remote_server_submit_command(
    iree_hal_remote_server_session_t* session_slot,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_hal_remote_server_submit_fn_t submit_fn, void* submit_user_data) {
  iree_hal_remote_server_t* server = session_slot->server;
  iree_hal_device_t* local_device = server->devices[0];
  IREE_ASSERT(signal_frontier && signal_frontier->entry_count == 1);
  iree_async_axis_t signal_axis = signal_frontier->entries[0].axis;
  uint64_t signal_epoch = signal_frontier->entries[0].epoch;

  iree_net_queue_channel_t* queue_channel = NULL;
  uint64_t session_id = 0;
  iree_hal_semaphore_t* wait_semaphores[8] = {0};
  uint64_t wait_values[8] = {0};
  iree_host_size_t wait_count = 0;
  bool submission_window_reserved = false;

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&server->session_mutex);
  if (!session_slot->session || !session_slot->queue_channel) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  } else {
    session_id = session_slot->session_id;
    queue_channel = session_slot->queue_channel;
    iree_net_queue_channel_retain(queue_channel);
    status = iree_net_sequence_window_reserve(
        &session_slot->observed_submission_window, signal_epoch);
    submission_window_reserved = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      status = iree_net_sequence_window_reserve(
          &session_slot->completed_signal_window, signal_epoch);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_resolve_wait_frontier(
          session_slot, wait_frontier, wait_semaphores, wait_values,
          IREE_ARRAYSIZE(wait_semaphores), &wait_count);
    }
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  // Create local signal semaphore (initial_value=0).
  iree_hal_semaphore_t* local_semaphore = NULL;
  if (iree_status_is_ok(status) && submission_window_reserved) {
    status = iree_hal_semaphore_create(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &local_semaphore);
  }

  // Allocate the completion context before publishing the epoch or submitting
  // the local command so that later failures can still be reported cleanly.
  iree_hal_remote_server_command_completion_t* completion = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(server->host_allocator, sizeof(*completion),
                                   (void**)&completion);
  }

  if (iree_status_is_ok(status)) {
    memset(completion, 0, sizeof(*completion));
    completion->host_allocator = server->host_allocator;
    completion->server = server;
    iree_hal_remote_server_retain(server);
    completion->session_slot = session_slot;
    completion->session_id = session_id;
    completion->queue_channel = queue_channel;
    iree_net_queue_channel_retain(completion->queue_channel);
    completion->local_semaphore = local_semaphore;
    iree_hal_semaphore_retain(local_semaphore);
    iree_async_single_frontier_initialize(&completion->signal_frontier,
                                          signal_axis, signal_epoch);
    completion->timepoint.callback = iree_hal_remote_server_on_command_complete;
    completion->timepoint.user_data = completion;
  }

  // Store epoch→semaphore mapping for future wait frontier resolution.
  bool epoch_mapping_stored = false;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&server->session_mutex);
    if (session_slot->session_id == session_id && session_slot->session) {
      status = iree_hal_remote_server_store_epoch_semaphore(
          session_slot, signal_axis, signal_epoch, local_semaphore,
          server->host_allocator);
    } else {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    }
    iree_slim_mutex_unlock(&server->session_mutex);
    epoch_mapping_stored = iree_status_is_ok(status);
  }

  bool completion_registered = false;
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t wait_list = {
        .count = wait_count,
        .semaphores = wait_semaphores,
        .payload_values = wait_values,
    };
    uint64_t signal_value = 1;
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &local_semaphore,
        .payload_values = &signal_value,
    };
    status = submit_fn(submit_user_data, local_device, wait_list, signal_list);
  }

  if (iree_status_is_ok(status)) {
    // Copy resolution entry from the op context if the submit callback
    // populated one (e.g., BUFFER_ALLOCA stores provisional→resolved mapping).
    iree_hal_remote_server_op_context_t* op_context =
        (iree_hal_remote_server_op_context_t*)submit_user_data;
    if (op_context && op_context->resolution_count > 0) {
      completion->resolution_count = op_context->resolution_count;
      completion->resolution = op_context->resolution;
    }

    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)local_semaphore, /*minimum_value=*/1,
        &completion->timepoint);
    completion_registered = iree_status_is_ok(status);
  }

  if (!iree_status_is_ok(status)) {
    // Command setup or submission failed. Send error ADVANCE so the client
    // doesn't deadlock, then clean up. We return OK to the caller (the queue
    // channel frame handler) because the error has been delivered to the client
    // via the ADVANCE frame — propagating it upward would trigger a transport
    // error on a channel that is otherwise healthy.
    iree_status_t failure_status = status;
    status = iree_ok_status();

    if (epoch_mapping_stored) {
      iree_hal_semaphore_t* removed_semaphore = NULL;
      iree_slim_mutex_lock(&server->session_mutex);
      if (session_slot->session_id == session_id && session_slot->session) {
        removed_semaphore = iree_hal_remote_server_remove_epoch_semaphore(
            session_slot, signal_axis, signal_epoch);
      }
      iree_slim_mutex_unlock(&server->session_mutex);
      iree_hal_semaphore_release(removed_semaphore);
    }

    if (local_semaphore) {
      iree_hal_semaphore_fail(local_semaphore,
                              iree_status_clone(failure_status));
    }
    if (queue_channel) {
      iree_hal_remote_server_send_error_advance(queue_channel, signal_frontier,
                                                failure_status);
    } else {
      iree_status_ignore(failure_status);
    }
  }

  if (iree_status_is_ok(status) && submission_window_reserved) {
    iree_net_sequence_node_t* ready_releases = NULL;
    bool session_active = false;
    iree_slim_mutex_lock(&server->session_mutex);
    session_active =
        session_slot->session_id == session_id && session_slot->session != NULL;
    if (session_active) {
      status = iree_net_sequence_window_observe(
          &session_slot->observed_submission_window, signal_epoch,
          &ready_releases);
    }
    iree_slim_mutex_unlock(&server->session_mutex);
    if (iree_status_is_ok(status) && session_active) {
      status = iree_hal_remote_server_process_ready_resource_releases(
          session_slot, ready_releases);
    }
  }

  if (completion && !completion_registered) {
    iree_hal_remote_server_release_command_completion(completion);
  }
  iree_hal_semaphore_list_t retained_wait_list = {
      .count = wait_count,
      .semaphores = wait_semaphores,
      .payload_values = wait_values,
  };
  iree_hal_semaphore_list_release(retained_wait_list);
  iree_hal_semaphore_release(local_semaphore);
  iree_net_queue_channel_release(queue_channel);
  return status;
}

//===----------------------------------------------------------------------===//
// Queue operation handlers
//===----------------------------------------------------------------------===//

// Submit callback for queue_barrier (empty COMMAND).
static iree_status_t iree_hal_remote_server_submit_barrier(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  (void)user_data;
  return iree_hal_device_queue_barrier(local_device,
                                       IREE_HAL_QUEUE_AFFINITY_ANY, wait_list,
                                       signal_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_remote_server_submit_buffer_fill(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_buffer_fill_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_FILL command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_buffer_fill_op_t));
  }
  const iree_hal_remote_buffer_fill_op_t* op =
      (const iree_hal_remote_buffer_fill_op_t*)context->command_data.data;

  iree_hal_remote_resource_id_t target_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->target_buffer_id);
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, target_id);
  if (!buffer) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "BUFFER_FILL target buffer 0x%016" PRIx64 " not found", target_id);
  }

  return iree_hal_device_queue_fill(local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                    wait_list, signal_list, buffer,
                                    op->target_offset, op->length, &op->pattern,
                                    (iree_host_size_t)op->pattern_length,
                                    (iree_hal_fill_flags_t)op->fill_flags);
}

static iree_status_t iree_hal_remote_server_submit_buffer_copy(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_buffer_copy_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_COPY command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_buffer_copy_op_t));
  }
  const iree_hal_remote_buffer_copy_op_t* op =
      (const iree_hal_remote_buffer_copy_op_t*)context->command_data.data;

  iree_hal_remote_resource_id_t source_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->source_buffer_id);
  iree_hal_buffer_t* source_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, source_id);
  if (!source_buffer) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "BUFFER_COPY source buffer 0x%016" PRIx64 " not found", source_id);
  }

  iree_hal_remote_resource_id_t target_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->target_buffer_id);
  iree_hal_buffer_t* target_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, target_id);
  if (!target_buffer) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "BUFFER_COPY target buffer 0x%016" PRIx64 " not found", target_id);
  }

  return iree_hal_device_queue_copy(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      source_buffer, op->source_offset, target_buffer, op->target_offset,
      op->length, (iree_hal_copy_flags_t)op->copy_flags);
}

static iree_status_t iree_hal_remote_server_submit_buffer_update(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_buffer_update_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_UPDATE command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_buffer_update_op_t));
  }
  const iree_hal_remote_buffer_update_op_t* op =
      (const iree_hal_remote_buffer_update_op_t*)context->command_data.data;

  // Inline source data follows the op struct. Use overflow-checked addition
  // because op->length is a wire-controlled uint64_t.
  iree_host_size_t inline_data_offset =
      sizeof(iree_hal_remote_buffer_update_op_t);
  iree_host_size_t update_required = 0;
  if (!iree_host_size_checked_add(
          inline_data_offset, (iree_host_size_t)op->length, &update_required) ||
      context->command_data.data_length < update_required) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "BUFFER_UPDATE inline data truncated or length overflow");
  }
  const void* inline_data = context->command_data.data + inline_data_offset;

  iree_hal_remote_resource_id_t target_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->target_buffer_id);
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, target_id);
  if (!buffer) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "BUFFER_UPDATE target buffer 0x%016" PRIx64 " not found", target_id);
  }

  return iree_hal_device_queue_update(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      inline_data, /*source_offset=*/0, buffer, op->target_offset, op->length,
      (iree_hal_update_flags_t)op->update_flags);
}

static iree_status_t iree_hal_remote_server_submit_buffer_alloca(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_buffer_alloca_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_ALLOCA command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_buffer_alloca_op_t));
  }
  const iree_hal_remote_buffer_alloca_op_t* op =
      (const iree_hal_remote_buffer_alloca_op_t*)context->command_data.data;
  if (op->pool != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "remote BUFFER_ALLOCA pool handles are not supported");
  }

  // Translate wire buffer params to HAL buffer params.
  iree_hal_buffer_params_t params = {0};
  params.usage = (iree_hal_buffer_usage_t)op->params.usage;
  params.access = (iree_hal_memory_access_t)op->params.access;
  params.type = (iree_hal_memory_type_t)op->params.type;
  params.queue_affinity = (iree_hal_queue_affinity_t)op->params.queue_affinity;
  params.min_alignment = (iree_device_size_t)op->params.min_alignment;
  const iree_hal_queue_affinity_t queue_affinity =
      iree_hal_queue_affinity_is_empty(params.queue_affinity)
          ? IREE_HAL_QUEUE_AFFINITY_ANY
          : params.queue_affinity;

  // Allocate on the local device. queue_alloca returns a buffer handle
  // immediately (synchronous allocation, async queue ordering).
  iree_hal_buffer_t* local_buffer = NULL;
  iree_status_t status = iree_hal_device_queue_alloca(
      local_device, queue_affinity, wait_list, signal_list, /*pool=*/NULL,
      params, (iree_device_size_t)op->allocation_size,
      (iree_hal_alloca_flags_t)op->alloca_flags, &local_buffer);

  // Assign the buffer to the resource table to get a canonical resolved ID.
  iree_hal_remote_resource_id_t resolved_id = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_resource_table_assign(
        &context->session_slot->resource_table,
        IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, local_buffer, &resolved_id);
  }

  // Store the provisional→resolved mapping so subsequent commands referencing
  // the provisional ID can be resolved.
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_store_provisional(
        context->session_slot, op->provisional_buffer_id, resolved_id,
        context->session_slot->server->host_allocator);
  }

  // Store the resolution entry in the op context so submit_command can
  // populate the completion's resolution field.
  if (iree_status_is_ok(status)) {
    context->resolution_count = 1;
    context->resolution.provisional_id = op->provisional_buffer_id;
    context->resolution.resolved_id = resolved_id;
  }

  iree_hal_buffer_release(local_buffer);
  return status;
}

static iree_status_t iree_hal_remote_server_submit_buffer_dealloca(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_buffer_dealloca_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_DEALLOCA command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_buffer_dealloca_op_t));
  }
  const iree_hal_remote_buffer_dealloca_op_t* op =
      (const iree_hal_remote_buffer_dealloca_op_t*)context->command_data.data;

  // Resolve the buffer ID (may be provisional).
  iree_hal_remote_resource_id_t resolved_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->buffer_id);
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, resolved_id);
  if (!buffer) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "BUFFER_DEALLOCA buffer 0x%016" PRIx64 " not found",
                            resolved_id);
  }

  return iree_hal_device_queue_dealloca(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list, buffer,
      (iree_hal_dealloca_flags_t)op->dealloca_flags);
}

static iree_status_t iree_hal_remote_server_resolve_dispatch_config(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_dispatch_config_t* wire_config,
    iree_hal_dispatch_flags_t flags, bool allow_indirect_buffer_ref,
    const char* label, iree_hal_dispatch_config_t* out_config) {
  memset(out_config, 0, sizeof(*out_config));
  memcpy(out_config->workgroup_size, wire_config->workgroup_size,
         sizeof(out_config->workgroup_size));
  memcpy(out_config->workgroup_count, wire_config->workgroup_count,
         sizeof(out_config->workgroup_count));
  out_config->dynamic_workgroup_local_memory =
      wire_config->dynamic_workgroup_local_memory;

  if (wire_config->workgroup_count_buffer_id != 0) {
    iree_hal_remote_resource_id_t buffer_id =
        iree_hal_remote_server_resolve_resource_id(
            session_slot, wire_config->workgroup_count_buffer_id);
    iree_hal_buffer_t* buffer =
        (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
            &session_slot->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
            buffer_id);
    if (!buffer) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "%s workgroup count buffer 0x%016" PRIx64
                              " not found",
                              label, buffer_id);
    }
    out_config->workgroup_count_ref =
        iree_hal_make_buffer_ref(buffer, wire_config->workgroup_count_offset,
                                 wire_config->workgroup_count_length);
  } else if (allow_indirect_buffer_ref &&
             iree_hal_dispatch_uses_indirect_parameters(flags)) {
    out_config->workgroup_count_ref = iree_hal_make_indirect_buffer_ref(
        wire_config->workgroup_count_buffer_slot,
        wire_config->workgroup_count_offset,
        wire_config->workgroup_count_length);
  } else {
    out_config->workgroup_count_ref.offset =
        wire_config->workgroup_count_offset;
    out_config->workgroup_count_ref.length =
        wire_config->workgroup_count_length;
    out_config->workgroup_count_ref.buffer_slot =
        wire_config->workgroup_count_buffer_slot;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_submit_dispatch(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_dispatch_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DISPATCH command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_dispatch_op_t));
  }
  const iree_hal_remote_dispatch_op_t* op =
      (const iree_hal_remote_dispatch_op_t*)context->command_data.data;

  // Resolve executable.
  iree_hal_remote_resource_id_t executable_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->executable_id);
  iree_hal_executable_t* executable =
      (iree_hal_executable_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE, executable_id);
  if (!executable) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "DISPATCH executable 0x%016" PRIx64 " not found",
                            executable_id);
  }

  // Parse variable-length constants and bindings with overflow checks.
  iree_host_size_t constants_size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)op->constant_count,
                                  sizeof(uint32_t), &constants_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "DISPATCH constants size overflow");
  }
  iree_host_size_t constants_padded = iree_host_align(constants_size, 8);
  iree_host_size_t bindings_offset = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_remote_dispatch_op_t),
                                  constants_padded, &bindings_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "DISPATCH bindings offset overflow");
  }
  iree_host_size_t bindings_size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)op->binding_count,
                                  sizeof(iree_hal_remote_binding_t),
                                  &bindings_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "DISPATCH bindings size overflow");
  }
  iree_host_size_t required_length = 0;
  if (!iree_host_size_checked_add(bindings_offset, bindings_size,
                                  &required_length) ||
      context->command_data.data_length < required_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DISPATCH command truncated: bindings");
  }

  iree_const_byte_span_t constants = iree_make_const_byte_span(
      context->command_data.data + sizeof(iree_hal_remote_dispatch_op_t),
      constants_size);

  const iree_hal_remote_binding_t* wire_bindings =
      (const iree_hal_remote_binding_t*)(context->command_data.data +
                                         bindings_offset);

  // Resolve buffer bindings to local buffers.
  // Stack-allocate for typical dispatch sizes.
  iree_hal_buffer_ref_t local_bindings[32];
  if (op->binding_count > IREE_ARRAYSIZE(local_bindings)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "DISPATCH binding count %u exceeds stack limit %zu",
                            op->binding_count, IREE_ARRAYSIZE(local_bindings));
  }
  memset(local_bindings, 0,
         (iree_host_size_t)op->binding_count * sizeof(iree_hal_buffer_ref_t));
  for (uint16_t i = 0; i < op->binding_count; ++i) {
    if (wire_bindings[i].buffer_id != 0) {
      iree_hal_remote_resource_id_t buffer_id =
          iree_hal_remote_server_resolve_resource_id(
              context->session_slot, wire_bindings[i].buffer_id);
      iree_hal_buffer_t* buffer =
          (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
              &context->session_slot->resource_table,
              IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, buffer_id);
      if (!buffer) {
        return iree_make_status(IREE_STATUS_NOT_FOUND,
                                "DISPATCH binding[%u] buffer 0x%016" PRIx64
                                " not found",
                                i, buffer_id);
      }
      local_bindings[i].buffer = buffer;
    }
    local_bindings[i].buffer_slot = wire_bindings[i].buffer_slot;
    local_bindings[i].offset = (iree_device_size_t)wire_bindings[i].offset;
    local_bindings[i].length = (iree_device_size_t)wire_bindings[i].length;
  }

  iree_hal_dispatch_config_t local_config;
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_resolve_dispatch_config(
      context->session_slot, &op->config,
      (iree_hal_dispatch_flags_t)op->dispatch_flags,
      /*allow_indirect_buffer_ref=*/false, "DISPATCH", &local_config));

  iree_hal_buffer_ref_list_t binding_list = {
      .count = op->binding_count,
      .values = local_bindings,
  };

  return iree_hal_device_queue_dispatch(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      executable, iree_hal_executable_function_from_index(op->export_ordinal),
      local_config, constants, binding_list,
      (iree_hal_dispatch_flags_t)op->dispatch_flags);
}

//===----------------------------------------------------------------------===//
// Control channel dispatch
//===----------------------------------------------------------------------===//

typedef struct iree_hal_remote_control_send_allocation_t {
  iree_allocator_t host_allocator;
  // Trailing response frame bytes.
} iree_hal_remote_control_send_allocation_t;

// Sends a control channel response. Builds envelope + response_prefix + body
// in retained storage and sends via the session. The request envelope is used
// to echo the request_id and message_type.
static iree_status_t iree_hal_remote_server_send_response(
    iree_allocator_t host_allocator, iree_net_session_t* session,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_code_t status_code, const void* body,
    iree_host_size_t body_length) {
  iree_host_size_t response_length =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_control_response_prefix_t) + body_length;

  iree_host_size_t allocation_size = 0;
  iree_status_t status =
      iree_host_size_checked_add(
          sizeof(iree_hal_remote_control_send_allocation_t), response_length,
          &allocation_size)
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "control response allocation size overflow");
  iree_hal_remote_control_send_allocation_t* allocation = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, allocation_size,
                                   (void**)&allocation);
  }
  if (!iree_status_is_ok(status)) return status;
  allocation->host_allocator = host_allocator;
  uint8_t* response_storage = (uint8_t*)(allocation + 1);
  memset(response_storage, 0, response_length);

  // Envelope.
  iree_hal_remote_control_envelope_t* envelope =
      (iree_hal_remote_control_envelope_t*)response_storage;
  envelope->message_type = request_envelope->message_type;
  envelope->message_flags = IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE;
  envelope->request_id = request_envelope->request_id;

  // Response prefix.
  iree_hal_remote_control_response_prefix_t* prefix =
      (iree_hal_remote_control_response_prefix_t*)(response_storage +
                                                   sizeof(
                                                       iree_hal_remote_control_envelope_t));
  prefix->status_code = (uint32_t)status_code;

  // Body.
  if (body && body_length > 0) {
    memcpy(response_storage + sizeof(iree_hal_remote_control_envelope_t) +
               sizeof(iree_hal_remote_control_response_prefix_t),
           body, body_length);
  }

  iree_async_span_t span =
      iree_async_span_from_ptr(response_storage, response_length);
  iree_async_span_list_t payload = {&span, 1};
  status = iree_net_session_send_control_data(session, /*flags=*/0, payload,
                                              (uint64_t)(uintptr_t)allocation);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, allocation);
  }
  return status;
}

// Sends an error response with the full serialized status as body.
// Consumes |status| (the caller must not use it after this call).
// The response prefix carries the status code for fast-path checking; the body
// carries the full wire-format status (source location, message, annotations).
static iree_status_t iree_hal_remote_server_send_error_response(
    iree_allocator_t host_allocator, iree_net_session_t* session,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_t status) {
  iree_status_code_t code = iree_status_code(status);

  // Compute serialized size and serialize if it fits in the stack buffer.
  iree_host_size_t wire_size = 0;
  iree_net_status_wire_size(status, &wire_size);

  iree_status_t send_status;
  if (wire_size <= IREE_HAL_REMOTE_MAX_STATUS_WIRE_SIZE) {
    uint8_t wire_buffer[IREE_HAL_REMOTE_MAX_STATUS_WIRE_SIZE];
    iree_status_t serialize_status = iree_net_status_wire_serialize(
        status, iree_make_byte_span(wire_buffer, wire_size));
    if (iree_status_is_ok(serialize_status)) {
      send_status = iree_hal_remote_server_send_response(
          host_allocator, session, request_envelope, code, wire_buffer,
          wire_size);
    } else {
      // Serialization failed; send code-only response.
      iree_status_ignore(serialize_status);
      send_status = iree_hal_remote_server_send_response(
          host_allocator, session, request_envelope, code, NULL, 0);
    }
  } else {
    // Status too large for stack buffer; send code-only response.
    send_status = iree_hal_remote_server_send_response(
        host_allocator, session, request_envelope, code, NULL, 0);
  }

  iree_status_ignore(status);
  return send_status;
}

// Handles BUFFER_ALLOC: allocates a buffer on the local device and assigns
// a resource slot in the session's table.
static iree_status_t iree_hal_remote_server_handle_buffer_alloc(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_alloc_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_ALLOC body too small: %" PRIhsz " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_alloc_request_t* request =
      (const iree_hal_remote_buffer_alloc_request_t*)body;

  // Convert wire params to HAL params.
  iree_hal_buffer_params_t params = {
      .usage = (iree_hal_buffer_usage_t)request->params.usage,
      .access = (iree_hal_memory_access_t)request->params.access,
      .type = (iree_hal_memory_type_t)request->params.type,
      .queue_affinity =
          (iree_hal_queue_affinity_t)request->params.queue_affinity,
      .min_alignment = (iree_device_size_t)request->params.min_alignment,
  };

  iree_device_size_t allocation_size =
      (iree_device_size_t)request->allocation_size;

  // Allocate on the local device.
  iree_hal_remote_server_t* server = entry->server;
  iree_hal_device_t* local_device = server->devices[0];
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(local_device);

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      allocator, params, allocation_size, &buffer);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  // Assign a slot in the session's resource table.
  iree_hal_remote_resource_id_t resolved_id = 0;
  status = iree_hal_remote_resource_table_assign(
      &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, buffer,
      &resolved_id);
  // The table retains the buffer; release our allocation reference.
  iree_hal_buffer_release(buffer);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  // Send success response with the resolved resource_id.
  iree_hal_remote_buffer_alloc_response_t response = {
      .resolved_id = resolved_id,
  };
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

// Handles BUFFER_QUERY_HEAPS: queries the local device's memory heap topology
// and sends the descriptions back to the client.
static iree_status_t iree_hal_remote_server_handle_buffer_query_heaps(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  iree_hal_remote_server_t* server = entry->server;
  iree_hal_device_t* local_device = server->devices[0];
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(local_device);

  // Query heap count first. The HAL contract returns OUT_OF_RANGE when
  // capacity < count (the standard pre-sizing pattern). We use capacity=0
  // to trigger this and read the count from out_count.
  iree_host_size_t heap_count = 0;
  iree_status_t status =
      iree_hal_allocator_query_memory_heaps(allocator, 0, NULL, &heap_count);
  if (iree_status_code(status) == IREE_STATUS_OUT_OF_RANGE) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  // Query full heap descriptions (stack-allocated for reasonable counts).
  iree_hal_allocator_memory_heap_t heaps_storage[16];
  if (heap_count > IREE_ARRAYSIZE(heaps_storage)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "too many heaps: %" PRIhsz, heap_count));
  }
  status = iree_hal_allocator_query_memory_heaps(allocator, heap_count,
                                                 heaps_storage, &heap_count);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  // Build response: header + wire heap descriptions.
  uint8_t response_body[sizeof(iree_hal_remote_buffer_query_heaps_response_t) +
                        16 * sizeof(iree_hal_remote_memory_heap_t)];
  memset(response_body, 0, sizeof(response_body));

  iree_hal_remote_buffer_query_heaps_response_t* response_header =
      (iree_hal_remote_buffer_query_heaps_response_t*)response_body;
  response_header->heap_count = (uint16_t)heap_count;

  iree_hal_remote_memory_heap_t* wire_heaps =
      (iree_hal_remote_memory_heap_t*)(response_body +
                                       sizeof(
                                           iree_hal_remote_buffer_query_heaps_response_t));
  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    wire_heaps[i].type = (uint32_t)heaps_storage[i].type;
    wire_heaps[i].allowed_usage = (uint32_t)heaps_storage[i].allowed_usage;
    wire_heaps[i].max_allocation_size =
        (uint64_t)heaps_storage[i].max_allocation_size;
    wire_heaps[i].min_alignment = (uint64_t)heaps_storage[i].min_alignment;
  }

  iree_host_size_t response_body_length =
      sizeof(iree_hal_remote_buffer_query_heaps_response_t) +
      heap_count * sizeof(iree_hal_remote_memory_heap_t);
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      response_body, response_body_length);
}

// Sends a control channel response with a variable-length data payload.
// The response is copied into one retained allocation because control sends are
// asynchronous and callers usually hold mapped or staging memory only for this
// function's dynamic extent. The session send-complete callback frees the
// allocation.
static iree_status_t iree_hal_remote_server_send_response_with_data(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_code_t status_code, const void* body_header,
    iree_host_size_t body_header_length, const void* data,
    iree_host_size_t data_length) {
  iree_host_size_t header_length =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_control_response_prefix_t) + body_header_length;
  iree_host_size_t response_length = 0;
  iree_status_t status =
      iree_host_size_checked_add(header_length, data_length, &response_length)
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "response payload size overflow");

  iree_host_size_t allocation_size = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_add(
          sizeof(iree_hal_remote_control_send_allocation_t), response_length,
          &allocation_size)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "response allocation size overflow");
  }

  iree_hal_remote_control_send_allocation_t* allocation = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(entry->server->host_allocator,
                                   allocation_size, (void**)&allocation);
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  allocation->host_allocator = entry->server->host_allocator;
  uint8_t* response_data = (uint8_t*)(allocation + 1);
  memset(response_data, 0, response_length);

  iree_hal_remote_control_envelope_t* envelope =
      (iree_hal_remote_control_envelope_t*)response_data;
  envelope->message_type = request_envelope->message_type;
  envelope->message_flags = IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE;
  envelope->request_id = request_envelope->request_id;

  iree_hal_remote_control_response_prefix_t* prefix =
      (iree_hal_remote_control_response_prefix_t*)(response_data +
                                                   sizeof(*envelope));
  prefix->status_code = (uint32_t)status_code;

  if (body_header && body_header_length > 0) {
    memcpy(response_data + sizeof(*envelope) + sizeof(*prefix), body_header,
           body_header_length);
  }
  if (data && data_length > 0) {
    memcpy(response_data + header_length, data, data_length);
  }

  iree_async_span_t span =
      iree_async_span_from_ptr(response_data, response_length);
  iree_async_span_list_t payload = {&span, 1};
  status = iree_net_session_send_control_data(
      entry->session, /*flags=*/0, payload, (uint64_t)(uintptr_t)allocation);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(entry->server->host_allocator, allocation);
  }
  return status;
}

static bool iree_hal_remote_server_buffer_supports_scoped_mapping(
    iree_hal_buffer_t* buffer) {
  return iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                           IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
         iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                           IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED);
}

// Handles BUFFER_MAP: maps a buffer on the local device, reads the requested
// region, and returns the data inline in the response. No persistent
// server-side mapping state is created — the map/unmap is scoped to this
// handler.
static iree_status_t iree_hal_remote_server_handle_buffer_map(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_map_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_MAP body too small: %" PRIhsz " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_map_request_t* request =
      (const iree_hal_remote_buffer_map_request_t*)body;
  iree_hal_remote_resource_id_t buffer_id =
      iree_hal_remote_server_resolve_resource_id(entry, request->buffer_id);

  // Look up the buffer in the resource table.
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
          buffer_id);
  if (!buffer) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "BUFFER_MAP: buffer_id 0x%016" PRIx64 " not found",
                         buffer_id));
  }
  iree_hal_memory_access_t memory_access =
      (iree_hal_memory_access_t)request->memory_access;
  iree_device_size_t offset = (iree_device_size_t)request->offset;
  iree_device_size_t length = (iree_device_size_t)request->length;

  // Only perform the local map+read if READ access was requested.
  if (iree_all_bits_set(memory_access, IREE_HAL_MEMORY_ACCESS_READ)) {
    const void* response_data = NULL;
    iree_host_size_t response_data_length = (iree_host_size_t)length;
    iree_hal_buffer_mapping_t mapping;
    memset(&mapping, 0, sizeof(mapping));
    bool did_map = false;

    iree_status_t status = iree_ok_status();
    if (length > IREE_HOST_SIZE_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "BUFFER_MAP length %" PRIdsz
                                " exceeds host size max %" PRIhsz,
                                length, IREE_HOST_SIZE_MAX);
    } else if (length == 0) {
      response_data = NULL;
      response_data_length = 0;
    } else if (iree_hal_remote_server_buffer_supports_scoped_mapping(buffer)) {
      status = iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                         IREE_HAL_MEMORY_ACCESS_READ, offset,
                                         length, &mapping);
      if (iree_status_is_ok(status)) {
        did_map = true;
        response_data = mapping.contents.data;
        response_data_length = mapping.contents.data_length;
      }
    } else {
      status =
          iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                           "BUFFER_MAP of non-host-visible buffer 0x%016" PRIx64
                           " requires async remote staging",
                           buffer_id);
    }

    if (!iree_status_is_ok(status)) {
      if (did_map) {
        status =
            iree_status_join(status, iree_hal_buffer_unmap_range(&mapping));
      }
      return iree_hal_remote_server_send_error_response(
          entry->server->host_allocator, entry->session, envelope, status);
    }

    iree_hal_remote_buffer_map_response_t response = {
        .mapped_offset = offset,
        .mapped_length = response_data_length,
    };
    iree_status_t send_status = iree_hal_remote_server_send_response_with_data(
        entry, envelope, IREE_STATUS_OK, &response, sizeof(response),
        response_data, response_data_length);

    if (did_map) {
      iree_status_ignore(iree_hal_buffer_unmap_range(&mapping));
    }
    return send_status;
  }

  if (length > 0 &&
      !iree_hal_remote_server_buffer_supports_scoped_mapping(buffer)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "BUFFER_MAP write of non-host-visible buffer "
                         "0x%016" PRIx64 " requires async remote staging",
                         buffer_id));
  }

  // WRITE-only or DISCARD: no data to send, just acknowledge.
  iree_hal_remote_buffer_map_response_t response = {
      .mapped_offset = offset,
      .mapped_length = 0,
  };
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

// Handles BUFFER_UNMAP: writes inline data from the client into a buffer on
// the local device. Maps the buffer, copies the data, and responds with status.
static iree_status_t iree_hal_remote_server_handle_buffer_unmap(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_unmap_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_UNMAP body too small: %" PRIhsz " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_unmap_request_t* request =
      (const iree_hal_remote_buffer_unmap_request_t*)body;
  iree_hal_remote_resource_id_t buffer_id =
      iree_hal_remote_server_resolve_resource_id(entry, request->buffer_id);
  iree_device_size_t offset = (iree_device_size_t)request->offset;
  iree_device_size_t length = (iree_device_size_t)request->length;

  // Validate inline data is present.
  const uint8_t* data = body + sizeof(iree_hal_remote_buffer_unmap_request_t);
  iree_host_size_t data_length =
      body_length - sizeof(iree_hal_remote_buffer_unmap_request_t);
  if (data_length < length) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_UNMAP data truncated: %" PRIhsz
                         " bytes, expected %" PRIdsz,
                         data_length, length));
  }

  // Look up the buffer in the resource table.
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
          buffer_id);
  if (!buffer) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "BUFFER_UNMAP: buffer_id 0x%016" PRIx64 " not found",
                         buffer_id));
  }

  iree_status_t status = iree_ok_status();
  if (length > IREE_HOST_SIZE_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "BUFFER_UNMAP length %" PRIdsz
                              " exceeds host size max %" PRIhsz,
                              length, IREE_HOST_SIZE_MAX);
  } else if (length == 0) {
    // No-op.
  } else if (iree_hal_remote_server_buffer_supports_scoped_mapping(buffer)) {
    iree_hal_buffer_mapping_t mapping;
    memset(&mapping, 0, sizeof(mapping));
    status = iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                       IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                       offset, length, &mapping);
    if (iree_status_is_ok(status)) {
      memcpy(mapping.contents.data, data, (iree_host_size_t)length);
      status = iree_status_join(status, iree_hal_buffer_unmap_range(&mapping));
    }
  } else {
    status =
        iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "BUFFER_UNMAP of non-host-visible buffer 0x%016" PRIx64
                         " requires async remote staging",
                         buffer_id);
  }
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  // Status-only response (no body).
  return iree_hal_remote_server_send_response(entry->server->host_allocator,
                                              entry->session, envelope,
                                              IREE_STATUS_OK, NULL, 0);
}

// Handles EXECUTABLE_UPLOAD: loads a compiled executable on the server.
static iree_status_t iree_hal_remote_server_handle_executable_upload(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_executable_upload_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "EXECUTABLE_UPLOAD body too small: %" PRIhsz " bytes",
                         body_length));
  }

  const iree_hal_remote_executable_upload_request_t* request =
      (const iree_hal_remote_executable_upload_request_t*)body;

  // Validate INLINE_DATA flag (bulk transfer not yet supported).
  if (!(request->upload_flags & IREE_HAL_REMOTE_UPLOAD_FLAG_INLINE_DATA)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "EXECUTABLE_UPLOAD: only INLINE_DATA is supported"));
  }

  // Validate data_length fits in iree_host_size_t (prevents silent truncation
  // on 32-bit platforms where uint64_t data_length exceeds SIZE_MAX).
  if (request->data_length > IREE_HOST_SIZE_MAX) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "EXECUTABLE_UPLOAD data_length exceeds host capacity"));
  }

  // Extract variable-length sections: format string, constants, data.
  // Layout after the fixed header:
  //   format[format_length]  (padded to 8)
  //   constants[constant_count]  (padded to 8)
  //   data[data_length]
  iree_host_size_t constants_size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)request->constant_count,
                                  sizeof(uint32_t), &constants_size)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "EXECUTABLE_UPLOAD constants size overflow"));
  }
  iree_host_size_t constants_padded = iree_host_align(constants_size, 8);
  const uint8_t* inline_data = NULL;
  iree_host_size_t format_length = (iree_host_size_t)request->format_length;
  iree_host_size_t format_padded = iree_host_align(format_length, 8);
  iree_host_size_t format_offset =
      sizeof(iree_hal_remote_executable_upload_request_t);
  iree_host_size_t format_end = 0;
  if (!iree_host_size_checked_add(format_offset, format_padded, &format_end) ||
      format_end > body_length) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "EXECUTABLE_UPLOAD format string truncated"));
  }
  const char* format_chars = (const char*)(body + format_offset);

  // Recalculate constants and data offsets to account for the format string.
  const uint32_t* constants = (const uint32_t*)(body + format_end);
  iree_host_size_t constants_data_offset = 0;
  if (!iree_host_size_checked_add(format_end, constants_padded,
                                  &constants_data_offset)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "EXECUTABLE_UPLOAD data offset overflow"));
  }
  iree_host_size_t total_required = 0;
  if (!iree_host_size_checked_add(constants_data_offset,
                                  (iree_host_size_t)request->data_length,
                                  &total_required) ||
      body_length < total_required) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "EXECUTABLE_UPLOAD inline data truncated"));
  }
  inline_data = body + constants_data_offset;

  // Build executable params.
  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.executable_format = iree_make_string_view(format_chars, format_length);
  params.executable_data = iree_make_const_byte_span(
      inline_data, (iree_host_size_t)request->data_length);
  params.constant_count = (iree_host_size_t)request->constant_count;
  params.constants = constants;

  // Load via the server's shared executable cache.
  iree_hal_remote_server_t* server = entry->server;
  iree_hal_executable_t* executable = NULL;
  iree_status_t status = iree_hal_executable_cache_prepare_executable(
      server->executable_caches[0], &params, &executable);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  // Query function count from the loaded executable.
  iree_host_size_t export_count =
      iree_hal_executable_function_count(executable);

  // Assign to the session's resource table.
  iree_hal_remote_resource_id_t resolved_id = 0;
  status = iree_hal_remote_resource_table_assign(
      &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE,
      executable, &resolved_id);
  iree_hal_executable_release(executable);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  // Send success response.
  iree_hal_remote_executable_upload_response_t response;
  memset(&response, 0, sizeof(response));
  response.resolved_id = resolved_id;
  response.export_count = (uint32_t)export_count;
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

//===----------------------------------------------------------------------===//
// Command stream replay
//===----------------------------------------------------------------------===//

// Replays a single DISPATCH command from the serialized stream.
static iree_status_t iree_hal_remote_server_replay_dispatch_cmd(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer, const uint8_t* cmd_data,
    const iree_hal_remote_cmd_header_t* header) {
  if (header->length < sizeof(iree_hal_remote_dispatch_cmd_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DISPATCH command truncated");
  }
  const iree_hal_remote_dispatch_cmd_t* cmd =
      (const iree_hal_remote_dispatch_cmd_t*)cmd_data;

  // Resolve executable.
  iree_hal_remote_resource_id_t executable_id =
      iree_hal_remote_server_resolve_resource_id(session_slot,
                                                 cmd->executable_id);
  iree_hal_executable_t* executable =
      (iree_hal_executable_t*)iree_hal_remote_resource_table_lookup(
          &session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE, executable_id);
  if (!executable) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "DISPATCH (cmd stream) executable 0x%016" PRIx64
                            " not found",
                            executable_id);
  }

  // Parse constants.
  iree_host_size_t constants_size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)cmd->constant_count,
                                  sizeof(uint32_t), &constants_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "DISPATCH constants size overflow");
  }
  iree_host_size_t constants_padded = iree_host_align(constants_size, 8);
  iree_host_size_t bindings_size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)cmd->binding_count,
                                  sizeof(iree_hal_remote_binding_t),
                                  &bindings_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "DISPATCH bindings size overflow");
  }
  // Validate that constants + bindings fit within the command's length.
  iree_host_size_t dispatch_payload_size = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_remote_dispatch_cmd_t),
                                  constants_padded, &dispatch_payload_size) ||
      !iree_host_size_checked_add(dispatch_payload_size, bindings_size,
                                  &dispatch_payload_size) ||
      header->length < dispatch_payload_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "DISPATCH constants/bindings exceed command length");
  }
  const uint8_t* constants_data = (const uint8_t*)(cmd + 1);
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constants_data, constants_size);

  // Parse and resolve bindings.
  const iree_hal_remote_binding_t* wire_bindings =
      (const iree_hal_remote_binding_t*)(constants_data + constants_padded);
  iree_hal_buffer_ref_t local_bindings[32];
  if (cmd->binding_count > IREE_ARRAYSIZE(local_bindings)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "DISPATCH binding count %u exceeds limit %zu",
                            cmd->binding_count, IREE_ARRAYSIZE(local_bindings));
  }
  memset(local_bindings, 0, cmd->binding_count * sizeof(iree_hal_buffer_ref_t));
  for (uint16_t i = 0; i < cmd->binding_count; ++i) {
    if (wire_bindings[i].buffer_id != 0) {
      iree_hal_remote_resource_id_t buffer_id =
          iree_hal_remote_server_resolve_resource_id(
              session_slot, wire_bindings[i].buffer_id);
      iree_hal_buffer_t* buffer =
          (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
              &session_slot->resource_table,
              IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, buffer_id);
      if (!buffer) {
        return iree_make_status(IREE_STATUS_NOT_FOUND,
                                "DISPATCH (cmd stream) binding[%u] buffer "
                                "0x%016" PRIx64 " not found",
                                i, buffer_id);
      }
      local_bindings[i] = iree_hal_make_buffer_ref(
          buffer, wire_bindings[i].offset, wire_bindings[i].length);
    } else {
      local_bindings[i] = iree_hal_make_indirect_buffer_ref(
          wire_bindings[i].buffer_slot, wire_bindings[i].offset,
          wire_bindings[i].length);
    }
  }

  iree_hal_dispatch_config_t config;
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_resolve_dispatch_config(
      session_slot, &cmd->config,
      (iree_hal_dispatch_flags_t)cmd->dispatch_flags,
      /*allow_indirect_buffer_ref=*/true, "DISPATCH (cmd stream)", &config));

  iree_hal_buffer_ref_list_t bindings = {
      .count = cmd->binding_count,
      .values = local_bindings,
  };
  return iree_hal_command_buffer_dispatch(
      local_command_buffer, executable,
      iree_hal_executable_function_from_index(cmd->export_ordinal), config,
      constants, bindings, (iree_hal_dispatch_flags_t)cmd->dispatch_flags);
}

// Replays a serialized command stream into a local command buffer. Iterates
// the stream (iree_hal_remote_cmd_header_t sequence), resolves resource IDs
// via the session's resource table, and translates each command to the
// corresponding HAL command buffer API call.
static iree_status_t iree_hal_remote_server_replay_command_stream(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer, const uint8_t* stream_data,
    iree_host_size_t stream_length) {
  iree_host_size_t offset = 0;
  while (offset < stream_length) {
    if (offset + sizeof(iree_hal_remote_cmd_header_t) > stream_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command stream truncated at offset %" PRIhsz,
                              offset);
    }
    const iree_hal_remote_cmd_header_t* header =
        (const iree_hal_remote_cmd_header_t*)(stream_data + offset);
    if (header->length < sizeof(*header) ||
        offset + header->length > stream_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command at offset %" PRIhsz
                              " has invalid length %u",
                              offset, header->length);
    }

    iree_status_t status = iree_ok_status();
    switch (header->type) {
      case IREE_HAL_REMOTE_CMD_EXECUTION_BARRIER: {
        if (header->length < sizeof(iree_hal_remote_execution_barrier_cmd_t)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "EXECUTION_BARRIER command truncated");
        }
        const iree_hal_remote_execution_barrier_cmd_t* cmd =
            (const iree_hal_remote_execution_barrier_cmd_t*)(stream_data +
                                                             offset);
        // For the remote replay path, we pass the barrier through without
        // fully translating individual memory/buffer barriers. The server's
        // local device handles the semantics.
        status = iree_hal_command_buffer_execution_barrier(
            local_command_buffer,
            (iree_hal_execution_stage_t)cmd->source_stage_mask,
            (iree_hal_execution_stage_t)cmd->target_stage_mask,
            IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
            /*memory_barrier_count=*/0, /*memory_barriers=*/NULL,
            /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL);
        break;
      }

      case IREE_HAL_REMOTE_CMD_BUFFER_FILL: {
        if (header->length < sizeof(iree_hal_remote_buffer_fill_cmd_t)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "BUFFER_FILL command truncated");
        }
        const iree_hal_remote_buffer_fill_cmd_t* cmd =
            (const iree_hal_remote_buffer_fill_cmd_t*)(stream_data + offset);
        iree_hal_buffer_ref_t target_ref;
        status = iree_hal_remote_server_resolve_command_buffer_ref(
            session_slot, cmd->target_buffer_id, cmd->target_buffer_slot,
            cmd->target_offset, cmd->target_length, "BUFFER_FILL", &target_ref);
        if (!iree_status_is_ok(status)) break;
        status = iree_hal_command_buffer_fill_buffer(
            local_command_buffer, target_ref, &cmd->pattern,
            (iree_host_size_t)cmd->pattern_length,
            (iree_hal_fill_flags_t)cmd->fill_flags);
        break;
      }

      case IREE_HAL_REMOTE_CMD_BUFFER_UPDATE: {
        if (header->length < sizeof(iree_hal_remote_buffer_update_cmd_t)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "BUFFER_UPDATE command truncated");
        }
        const iree_hal_remote_buffer_update_cmd_t* cmd =
            (const iree_hal_remote_buffer_update_cmd_t*)(stream_data + offset);
        // Validate that the inline payload fits within the command's length.
        iree_host_size_t update_required = 0;
        if (!iree_host_size_checked_add(
                sizeof(iree_hal_remote_buffer_update_cmd_t),
                iree_host_align((iree_host_size_t)cmd->target_length, 8),
                &update_required) ||
            header->length < update_required) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "BUFFER_UPDATE payload exceeds command length");
        }
        const void* source_data = (const void*)(cmd + 1);
        iree_hal_buffer_ref_t target_ref;
        status = iree_hal_remote_server_resolve_command_buffer_ref(
            session_slot, cmd->target_buffer_id, cmd->target_buffer_slot,
            cmd->target_offset, cmd->target_length, "BUFFER_UPDATE",
            &target_ref);
        if (!iree_status_is_ok(status)) break;
        status = iree_hal_command_buffer_update_buffer(
            local_command_buffer, source_data, /*source_offset=*/0, target_ref,
            (iree_hal_update_flags_t)cmd->update_flags);
        break;
      }

      case IREE_HAL_REMOTE_CMD_BUFFER_COPY: {
        if (header->length < sizeof(iree_hal_remote_buffer_copy_cmd_t)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "BUFFER_COPY command truncated");
        }
        const iree_hal_remote_buffer_copy_cmd_t* cmd =
            (const iree_hal_remote_buffer_copy_cmd_t*)(stream_data + offset);
        iree_hal_buffer_ref_t source_ref;
        status = iree_hal_remote_server_resolve_command_buffer_ref(
            session_slot, cmd->source_buffer_id, cmd->source_buffer_slot,
            cmd->source_offset, cmd->length, "BUFFER_COPY source", &source_ref);
        if (!iree_status_is_ok(status)) break;
        iree_hal_buffer_ref_t target_ref;
        status = iree_hal_remote_server_resolve_command_buffer_ref(
            session_slot, cmd->target_buffer_id, cmd->target_buffer_slot,
            cmd->target_offset, cmd->length, "BUFFER_COPY target", &target_ref);
        if (!iree_status_is_ok(status)) break;
        status = iree_hal_command_buffer_copy_buffer(
            local_command_buffer, source_ref, target_ref,
            (iree_hal_copy_flags_t)cmd->copy_flags);
        break;
      }

      case IREE_HAL_REMOTE_CMD_DISPATCH:
        status = iree_hal_remote_server_replay_dispatch_cmd(
            session_slot, local_command_buffer, stream_data + offset, header);
        break;

      case IREE_HAL_REMOTE_CMD_DEBUG_GROUP_BEGIN:
      case IREE_HAL_REMOTE_CMD_DEBUG_GROUP_END:
        // Debug groups are informational — skip during replay.
        break;

      case IREE_HAL_REMOTE_CMD_BUFFER_ADVISE:
        // Advise is a hint — skip during replay.
        break;

      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "command type 0x%04x not implemented",
                                header->type);
    }
    IREE_RETURN_IF_ERROR(status);

    offset += header->length;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// COMMAND_BUFFER_UPLOAD handler
//===----------------------------------------------------------------------===//

// Handles COMMAND_BUFFER_UPLOAD: stores a reusable command buffer recording.
static iree_status_t iree_hal_remote_server_handle_command_buffer_upload(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_command_buffer_upload_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "COMMAND_BUFFER_UPLOAD body too small: %" PRIhsz
                         " bytes",
                         body_length));
  }

  const iree_hal_remote_command_buffer_upload_request_t* request =
      (const iree_hal_remote_command_buffer_upload_request_t*)body;

  if (!(request->upload_flags & IREE_HAL_REMOTE_UPLOAD_FLAG_INLINE_DATA)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "COMMAND_BUFFER_UPLOAD: only INLINE_DATA supported"));
  }

  // Validate data_length fits in iree_host_size_t (prevents silent truncation
  // on 32-bit platforms where uint64_t data_length exceeds SIZE_MAX).
  if (request->data_length > IREE_HOST_SIZE_MAX) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "COMMAND_BUFFER_UPLOAD data_length exceeds host capacity"));
  }

  // Validate inline data length.
  iree_host_size_t data_offset =
      sizeof(iree_hal_remote_command_buffer_upload_request_t);
  iree_host_size_t required_length = 0;
  if (!iree_host_size_checked_add(data_offset,
                                  (iree_host_size_t)request->data_length,
                                  &required_length) ||
      body_length < required_length) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "COMMAND_BUFFER_UPLOAD inline data truncated"));
  }

  // Replay the command stream into a real local command buffer. The
  // resource table stores the native iree_hal_command_buffer_t* directly,
  // so subsequent executes skip replay entirely.
  iree_hal_device_t* local_device = entry->server->devices[0];
  const uint8_t* stream_data = body + data_offset;
  iree_host_size_t stream_length = (iree_host_size_t)request->data_length;

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      local_device, (iree_hal_command_buffer_mode_t)request->mode,
      (iree_hal_command_category_t)request->categories,
      IREE_HAL_QUEUE_AFFINITY_ANY, (iree_host_size_t)request->binding_capacity,
      &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_replay_command_stream(
        entry, command_buffer, stream_data, stream_length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }

  // Assign the native command buffer to the resource table.
  iree_hal_remote_resource_id_t resolved_id = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_resource_table_assign(
        &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_COMMAND_BUFFER,
        command_buffer, &resolved_id);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(command_buffer);
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  // The resource table retains the command buffer; release our creation ref.
  iree_hal_command_buffer_release(command_buffer);

  // Send response with resolved ID.
  iree_hal_remote_command_buffer_upload_response_t response;
  memset(&response, 0, sizeof(response));
  response.resolved_id = resolved_id;
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

//===----------------------------------------------------------------------===//
// COMMAND_BUFFER_EXECUTE submit callback
//===----------------------------------------------------------------------===//

// Submit callback for COMMAND_BUFFER_EXECUTE queue ops.
static iree_status_t iree_hal_remote_server_submit_command_buffer_execute(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  iree_hal_remote_server_session_t* session_slot = context->session_slot;
  const uint8_t* command_data = context->command_data.data;
  iree_host_size_t command_length = context->command_data.data_length;

  // Parse the execute op header.
  if (command_length < sizeof(iree_hal_remote_command_buffer_execute_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "COMMAND_BUFFER_EXECUTE op truncated");
  }
  const iree_hal_remote_command_buffer_execute_op_t* op =
      (const iree_hal_remote_command_buffer_execute_op_t*)command_data;

  // Parse binding table.
  iree_host_size_t bindings_size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)op->binding_count,
                                  sizeof(iree_hal_remote_binding_t),
                                  &bindings_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "COMMAND_BUFFER_EXECUTE bindings size overflow");
  }
  iree_host_size_t bindings_offset =
      sizeof(iree_hal_remote_command_buffer_execute_op_t);
  iree_host_size_t stream_offset = 0;
  if (!iree_host_size_checked_add(bindings_offset, bindings_size,
                                  &stream_offset) ||
      stream_offset > command_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "COMMAND_BUFFER_EXECUTE binding table truncated");
  }

  // Resolve wire binding table to local buffer bindings.
  const iree_hal_remote_binding_t* wire_bindings =
      (const iree_hal_remote_binding_t*)(command_data + bindings_offset);
  iree_hal_buffer_binding_t local_bindings[32];
  if (op->binding_count > IREE_ARRAYSIZE(local_bindings)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "COMMAND_BUFFER_EXECUTE binding count %u exceeds limit %zu",
        op->binding_count, IREE_ARRAYSIZE(local_bindings));
  }
  memset(local_bindings, 0,
         op->binding_count * sizeof(iree_hal_buffer_binding_t));
  for (uint16_t i = 0; i < op->binding_count; ++i) {
    if (wire_bindings[i].buffer_id != 0) {
      iree_hal_remote_resource_id_t buffer_id =
          iree_hal_remote_server_resolve_resource_id(
              session_slot, wire_bindings[i].buffer_id);
      iree_hal_buffer_t* buffer =
          (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
              &session_slot->resource_table,
              IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, buffer_id);
      if (!buffer) {
        return iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "COMMAND_BUFFER_EXECUTE binding[%u] buffer 0x%016" PRIx64
            " not found",
            i, buffer_id);
      }
      local_bindings[i].buffer = buffer;
      local_bindings[i].offset = wire_bindings[i].offset;
      local_bindings[i].length = wire_bindings[i].length;
    }
  }
  iree_hal_buffer_binding_table_t binding_table = {
      .count = op->binding_count,
      .bindings = local_bindings,
  };

  bool inline_stream = iree_all_bits_set(
      op->header.flags, IREE_HAL_REMOTE_EXECUTE_FLAG_INLINE_COMMAND_STREAM);

  if (!inline_stream) {
    // Reusable: the UPLOAD handler already replayed the stream into a native
    // local command buffer. Just look it up and submit directly — no replay.
    iree_hal_remote_resource_id_t command_buffer_id =
        iree_hal_remote_server_resolve_resource_id(session_slot,
                                                   op->command_buffer_id);
    iree_hal_command_buffer_t* local_command_buffer =
        (iree_hal_command_buffer_t*)iree_hal_remote_resource_table_lookup(
            &session_slot->resource_table,
            IREE_HAL_REMOTE_RESOURCE_TYPE_COMMAND_BUFFER, command_buffer_id);
    if (!local_command_buffer) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "COMMAND_BUFFER_EXECUTE command buffer "
                              "0x%016" PRIx64 " not found",
                              command_buffer_id);
    }
    return iree_hal_device_queue_execute(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
        local_command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  }

  // Inline one-shot: replay the stream into a local one-shot command buffer.
  const uint8_t* stream_data = command_data + stream_offset;
  iree_host_size_t stream_length = command_length - stream_offset;

  iree_hal_command_buffer_t* local_command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      local_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
      op->binding_count, &local_command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(local_command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_replay_command_stream(
        session_slot, local_command_buffer, stream_data, stream_length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(local_command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
        local_command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  }

  iree_hal_command_buffer_release(local_command_buffer);
  return status;
}

static iree_status_t iree_hal_remote_server_release_resource_ids(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_resource_id_t* resource_ids,
    uint32_t resource_count) {
  for (uint32_t i = 0; i < resource_count; ++i) {
    iree_hal_remote_resource_id_t resolved_id =
        iree_hal_remote_server_resolve_resource_id(entry, resource_ids[i]);
    iree_hal_remote_resource_table_release(&entry->resource_table, resolved_id);
    if (resolved_id != resource_ids[i]) {
      iree_hal_remote_server_remove_provisional(entry, resource_ids[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_release_resource_node(
    iree_hal_remote_server_session_t* entry,
    iree_hal_remote_server_resource_release_node_t* release_node) {
  iree_status_t status = iree_hal_remote_server_release_resource_ids(
      entry, release_node->resource_ids, release_node->resource_count);
  iree_allocator_t host_allocator = release_node->host_allocator;
  iree_allocator_free(host_allocator, release_node);
  return status;
}

static iree_status_t iree_hal_remote_server_process_ready_resource_releases(
    iree_hal_remote_server_session_t* entry,
    iree_net_sequence_node_t* ready_list) {
  iree_status_t status = iree_ok_status();
  while (ready_list && iree_status_is_ok(status)) {
    iree_net_sequence_node_t* next = ready_list->next;
    iree_hal_remote_server_resource_release_node_t* release_node =
        iree_containerof(ready_list,
                         iree_hal_remote_server_resource_release_node_t,
                         sequence_node);
    status = iree_hal_remote_server_release_resource_node(entry, release_node);
    ready_list = next;
  }
  iree_hal_remote_server_free_resource_release_nodes(ready_list);
  return status;
}

static iree_status_t iree_hal_remote_server_schedule_resource_release(
    iree_hal_remote_server_session_t* entry, uint64_t required_observed_epoch,
    const iree_hal_remote_resource_id_t* resource_ids,
    uint32_t resource_count) {
  if (resource_count == 0) return iree_ok_status();
  iree_hal_remote_server_t* server = entry->server;
  bool release_now = false;

  iree_slim_mutex_lock(&server->session_mutex);
  if (!entry->session) {
    release_now = false;
  } else if (required_observed_epoch <=
             iree_net_sequence_window_observed(
                 &entry->observed_submission_window)) {
    release_now = true;
  }
  iree_slim_mutex_unlock(&server->session_mutex);
  if (release_now) {
    return iree_hal_remote_server_release_resource_ids(entry, resource_ids,
                                                       resource_count);
  }

  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_server_resource_release_node_t), &total_size,
      IREE_STRUCT_FIELD_FAM(resource_count, iree_hal_remote_resource_id_t)));

  iree_allocator_t host_allocator = server->host_allocator;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&server->session_mutex);
  if (!entry->session) {
    status = iree_status_from_code(IREE_STATUS_ABORTED);
  } else {
    status = iree_net_sequence_window_reserve(
        &entry->observed_submission_window, required_observed_epoch);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  iree_hal_remote_server_resource_release_node_t* release_node = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size,
                                   (void**)&release_node);
  }
  if (iree_status_is_ok(status)) {
    memset(release_node, 0, sizeof(*release_node));
    release_node->host_allocator = host_allocator;
    release_node->resource_count = resource_count;
    memcpy(release_node->resource_ids, resource_ids,
           (iree_host_size_t)resource_count *
               sizeof(iree_hal_remote_resource_id_t));
  }

  iree_net_sequence_node_t* ready_list = NULL;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&server->session_mutex);
    if (!entry->session) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    } else if (required_observed_epoch <=
               iree_net_sequence_window_observed(
                   &entry->observed_submission_window)) {
      release_now = true;
    } else {
      status = iree_net_sequence_window_defer_until(
          &entry->observed_submission_window, required_observed_epoch,
          &release_node->sequence_node, &ready_list);
      if (iree_status_is_ok(status)) release_node = NULL;
    }
    iree_slim_mutex_unlock(&server->session_mutex);
  }
  if (iree_status_is_ok(status)) {
    if (release_now) {
      status =
          iree_hal_remote_server_release_resource_node(entry, release_node);
      release_node = NULL;
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_process_ready_resource_releases(
          entry, ready_list);
    }
  }
  if (release_node) {
    iree_allocator_free(host_allocator, release_node);
  }
  return status;
}

// Handles legacy control-channel RESOURCE_RELEASE_BATCH. Current clients send
// releases over the queue channel with an observed-submission epoch. Control
// frames do not carry that ordering information, so accepting one here would
// risk racing ahead of queue commands still resolving the resources.
static iree_status_t iree_hal_remote_server_handle_resource_release_batch(
    iree_hal_remote_server_session_t* entry, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_resource_release_batch_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RESOURCE_RELEASE_BATCH body too small: %" PRIhsz
                            " bytes",
                            body_length);
  }

  const iree_hal_remote_resource_release_batch_t* batch =
      (const iree_hal_remote_resource_release_batch_t*)body;
  uint32_t resource_count = batch->resource_count;

  iree_host_size_t expected_size = 0;
  if (!iree_host_size_checked_mul_add(
          sizeof(iree_hal_remote_resource_release_batch_t), resource_count,
          sizeof(iree_hal_remote_resource_id_t), &expected_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RESOURCE_RELEASE_BATCH size overflow");
  }
  if (body_length < expected_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RESOURCE_RELEASE_BATCH truncated: %" PRIhsz
                            " bytes, expected "
                            "%" PRIhsz " for %u resources",
                            body_length, expected_size, resource_count);
  }

  (void)entry;
  (void)resource_count;
  return iree_ok_status();
}

// Handles queue-channel RESOURCE_RELEASE_BATCH. The frame carries the greatest
// submission epoch that may still reference the resources. Once all COMMAND
// packets up to that epoch have been processed, the wrapped local HAL has
// retained anything it still needs and the resource table entries can be
// released without waiting for execution completion.
static iree_status_t iree_hal_remote_server_handle_queue_resource_release_batch(
    iree_hal_remote_server_session_t* entry,
    iree_const_byte_span_t command_data) {
  if (command_data.data_length <
      sizeof(iree_hal_remote_resource_release_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RESOURCE_RELEASE_BATCH op too small: %" PRIhsz
                            " bytes",
                            command_data.data_length);
  }

  const iree_hal_remote_resource_release_op_t* op =
      (const iree_hal_remote_resource_release_op_t*)command_data.data;
  uint32_t resource_count = op->resource_count;

  iree_host_size_t expected_size = 0;
  if (!iree_host_size_checked_mul_add(
          sizeof(iree_hal_remote_resource_release_op_t), resource_count,
          sizeof(iree_hal_remote_resource_id_t), &expected_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RESOURCE_RELEASE_BATCH op size overflow");
  }
  if (command_data.data_length < expected_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RESOURCE_RELEASE_BATCH op truncated: %" PRIhsz
                            " bytes, expected %" PRIhsz " for %u resources",
                            command_data.data_length, expected_size,
                            resource_count);
  }

  const iree_hal_remote_resource_id_t* resource_ids =
      (const iree_hal_remote_resource_id_t*)(command_data.data +
                                             sizeof(
                                                 iree_hal_remote_resource_release_op_t));
  return iree_hal_remote_server_schedule_resource_release(
      entry, op->required_observed_epoch, resource_ids, resource_count);
}

//===----------------------------------------------------------------------===//
// Queue channel callbacks
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_server_validate_queue_frontier(
    iree_hal_remote_server_session_t* session_slot, const char* frontier_name,
    const iree_async_frontier_t* frontier) {
  if (!frontier) return iree_ok_status();
  iree_async_axis_t queue_axis = session_slot->server->local_topology.axes[0];
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    if (frontier->entries[i].axis != queue_axis) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s frontier entry %u has axis=0x%016" PRIx64
                              "; expected queue axis=0x%016" PRIx64,
                              frontier_name, i, frontier->entries[i].axis,
                              queue_axis);
    }
  }
  return iree_ok_status();
}

// Server receives COMMAND frames from clients and dispatches to the local
// device. On completion (via semaphore timepoint), sends ADVANCE back.
static iree_status_t iree_hal_remote_server_on_command(
    void* user_data, uint32_t stream_id,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t command_data, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_server_session_t* session_slot =
      (iree_hal_remote_server_session_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status;
  const iree_hal_remote_queue_op_header_t* op_header = NULL;
  if (command_data.data_length >= sizeof(iree_hal_remote_queue_op_header_t)) {
    op_header = (const iree_hal_remote_queue_op_header_t*)command_data.data;
    if (op_header->type == IREE_HAL_REMOTE_QUEUE_OP_RESOURCE_RELEASE_BATCH) {
      status = iree_hal_remote_server_handle_queue_resource_release_batch(
          session_slot, command_data);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  if (!signal_frontier || signal_frontier->entry_count != 1) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "COMMAND frame must have exactly one signal frontier entry");
  }

  status = iree_hal_remote_server_validate_queue_frontier(session_slot, "wait",
                                                          wait_frontier);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_validate_queue_frontier(
        session_slot, "signal", signal_frontier);
  }
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  if (command_data.data_length == 0) {
    // Barrier operation (empty payload).
    status = iree_hal_remote_server_submit_command(
        session_slot, wait_frontier, signal_frontier,
        iree_hal_remote_server_submit_barrier, NULL);
  } else if (op_header) {
    iree_hal_remote_server_op_context_t op_context = {
        .session_slot = session_slot,
        .command_data = command_data,
        .resolution_count = 0,
    };
    switch (op_header->type) {
      case IREE_HAL_REMOTE_QUEUE_OP_BUFFER_ALLOCA:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_buffer_alloca, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_BUFFER_DEALLOCA:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_buffer_dealloca, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_BUFFER_FILL:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_buffer_fill, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_BUFFER_COPY:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_buffer_copy, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_BUFFER_UPDATE:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_buffer_update, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_DISPATCH:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_dispatch, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_COMMAND_BUFFER_EXECUTE:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_command_buffer_execute, &op_context);
        break;
      default:
        status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "queue op type 0x%04x not implemented",
                                  op_header->type);
        break;
    }
  } else {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "COMMAND payload too short for op header: "
                              "%" PRIhsz " bytes",
                              command_data.data_length);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Server does not receive ADVANCE frames (only clients do).
static iree_status_t iree_hal_remote_server_on_advance(
    void* user_data, const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t advance_data, iree_async_buffer_lease_t* lease) {
  (void)user_data;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "server does not accept ADVANCE frames");
}

static void iree_hal_remote_server_on_queue_transport_error(
    void* user_data, iree_status_t status) {
  (void)user_data;
  // TODO(benvanik): propagate queue channel transport error to session.
  iree_status_ignore(status);
}

// Header buffers used by the bulk channel frame sender. Bulk DATA payloads are
// not copied into this pool; only the 40-byte frame headers are retained until
// send completion.
#define IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_COUNT 128
#define IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_SIZE 128

static iree_status_t iree_hal_remote_server_on_bulk_start(
    void* user_data, uint64_t transfer_id, uint64_t total_size,
    iree_net_bulk_frame_flags_t flags) {
  (void)user_data;
  (void)transfer_id;
  (void)total_size;
  (void)flags;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote server bulk START is not implemented yet");
}

static iree_status_t iree_hal_remote_server_on_bulk_data(
    void* user_data, uint64_t transfer_id, uint64_t chunk_offset,
    uint32_t sequence, iree_net_bulk_frame_flags_t flags,
    iree_const_byte_span_t chunk_data, iree_async_buffer_lease_t* lease) {
  (void)user_data;
  (void)transfer_id;
  (void)chunk_offset;
  (void)sequence;
  (void)flags;
  (void)chunk_data;
  (void)lease;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote server bulk DATA is not implemented yet");
}

static iree_status_t iree_hal_remote_server_on_bulk_complete(
    void* user_data, uint64_t transfer_id) {
  (void)user_data;
  (void)transfer_id;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote server bulk COMPLETE is not implemented yet");
}

static iree_status_t iree_hal_remote_server_on_bulk_abort(
    void* user_data, uint64_t transfer_id, iree_const_byte_span_t abort_data,
    iree_async_buffer_lease_t* lease) {
  (void)user_data;
  (void)transfer_id;
  (void)abort_data;
  (void)lease;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote server bulk ABORT is not implemented yet");
}

static void iree_hal_remote_server_on_bulk_transport_error(
    void* user_data, iree_status_t status) {
  iree_hal_remote_server_session_t* session_slot =
      (iree_hal_remote_server_session_t*)user_data;
  iree_status_t shutdown_status = iree_net_session_shutdown(
      session_slot->session, /*reason_code=*/0,
      iree_make_cstring_view("bulk channel transport error"));
  iree_status_ignore(shutdown_status);
  iree_status_ignore(status);
}

static void iree_hal_remote_server_on_bulk_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status) {
  (void)operation_user_data;
  if (iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;
  }
  iree_hal_remote_server_on_bulk_transport_error(user_data, status);
}

static void iree_hal_remote_server_on_bulk_credit(
    void* user_data, uint32_t credit_delta, uint32_t available_credit_count) {
  (void)user_data;
  (void)credit_delta;
  (void)available_credit_count;
}

// Context passed to the endpoint_ready callback to identify which session
// slot should receive the application channel.
typedef struct iree_hal_remote_server_endpoint_context_t {
  iree_hal_remote_server_t* server;
  iree_net_session_t* session;  // retained
  iree_allocator_t host_allocator;
} iree_hal_remote_server_endpoint_context_t;

static void iree_hal_remote_server_on_queue_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_hal_remote_server_endpoint_context_t* context =
      (iree_hal_remote_server_endpoint_context_t*)user_data;
  iree_hal_remote_server_t* server = context->server;
  iree_net_session_t* session = context->session;
  iree_allocator_t host_allocator = context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Free the context first (we've captured what we need).
  iree_allocator_free(host_allocator, context);
  context = NULL;

  if (!iree_status_is_ok(status)) {
    iree_status_t shutdown_status = iree_net_session_shutdown(
        session, /*reason_code=*/0,
        iree_make_cstring_view("queue endpoint open failed"));
    iree_status_ignore(shutdown_status);
    iree_status_ignore(status);
    iree_net_session_release(session);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Find the session slot (under lock — stop() may be iterating).
  iree_slim_mutex_lock(&server->session_mutex);
  int32_t slot = iree_hal_remote_server_find_session_slot(server, session);
  iree_slim_mutex_unlock(&server->session_mutex);
  if (slot < 0) {
    // Session was removed while endpoint was opening.
    iree_net_session_release(session);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Create header pool and queue channel outside the lock (allocation, no
  // shared state).
  iree_async_buffer_pool_t* header_pool = NULL;
  status = iree_hal_remote_create_queue_header_pool(
      IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_COUNT,
      IREE_HAL_REMOTE_QUEUE_HEADER_POOL_BUFFER_SIZE, host_allocator,
      &header_pool);

  iree_net_queue_channel_t* queue_channel = NULL;
  if (iree_status_is_ok(status)) {
    iree_net_queue_channel_callbacks_t callbacks = {
        .on_command = iree_hal_remote_server_on_command,
        .on_advance = iree_hal_remote_server_on_advance,
        .on_transport_error = iree_hal_remote_server_on_queue_transport_error,
        .user_data = &server->sessions[slot],
    };

    iree_async_buffer_pool_t* channel_header_pool = header_pool;
    header_pool = NULL;  // Ownership transfers to create, including failure.
    status = iree_net_queue_channel_create(
        endpoint, IREE_NET_FRAME_SENDER_MAX_SPANS, channel_header_pool,
        callbacks, host_allocator, &queue_channel);
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_queue_channel_activate(queue_channel);
  }

  if (iree_status_is_ok(status)) {
    // Re-verify the session is still in its slot before storing the channel.
    // Between the slot lookup above and now, stop() → remove_session() could
    // have cleared this slot.
    iree_slim_mutex_lock(&server->session_mutex);
    if (server->sessions[slot].session == session) {
      server->sessions[slot].queue_channel = queue_channel;
      queue_channel = NULL;  // Ownership transferred.
    }
    iree_slim_mutex_unlock(&server->session_mutex);

    // If the session was removed while we were setting up the channel then
    // ownership remains local. Otherwise this is NULL after transfer.
    iree_net_queue_channel_release(queue_channel);
  } else {
    // Channel create failed or wasn't reached. The channel owns the pool after
    // create is called; otherwise header_pool remains local.
    iree_net_queue_channel_release(queue_channel);
    iree_async_buffer_pool_free(header_pool);
    // Shut down the session so it transitions to a terminal state and frees
    // its slot. A session without a queue channel cannot process commands.
    iree_status_t shutdown_status = iree_net_session_shutdown(
        session, /*reason_code=*/0,
        iree_make_cstring_view("queue channel setup failed"));
    iree_status_ignore(shutdown_status);
    iree_status_ignore(status);
  }

  iree_net_session_release(session);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_remote_server_on_bulk_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_hal_remote_server_endpoint_context_t* context =
      (iree_hal_remote_server_endpoint_context_t*)user_data;
  iree_hal_remote_server_t* server = context->server;
  iree_net_session_t* session = context->session;
  iree_allocator_t host_allocator = context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, context);
  context = NULL;

  if (!iree_status_is_ok(status)) {
    iree_status_t shutdown_status = iree_net_session_shutdown(
        session, /*reason_code=*/0,
        iree_make_cstring_view("bulk endpoint open failed"));
    iree_status_ignore(shutdown_status);
    iree_status_ignore(status);
    iree_net_session_release(session);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  iree_slim_mutex_lock(&server->session_mutex);
  int32_t slot = iree_hal_remote_server_find_session_slot(server, session);
  iree_slim_mutex_unlock(&server->session_mutex);
  if (slot < 0) {
    iree_net_session_release(session);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  iree_async_buffer_pool_t* header_pool = NULL;
  status = iree_hal_remote_create_queue_header_pool(
      IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_COUNT,
      IREE_HAL_REMOTE_BULK_HEADER_POOL_BUFFER_SIZE, host_allocator,
      &header_pool);

  iree_net_bulk_channel_t* bulk_channel = NULL;
  if (iree_status_is_ok(status)) {
    iree_net_bulk_channel_callbacks_t callbacks = {
        .on_start = iree_hal_remote_server_on_bulk_start,
        .on_data = iree_hal_remote_server_on_bulk_data,
        .on_complete = iree_hal_remote_server_on_bulk_complete,
        .on_abort = iree_hal_remote_server_on_bulk_abort,
        .on_transport_error = iree_hal_remote_server_on_bulk_transport_error,
        .on_send_complete = iree_hal_remote_server_on_bulk_send_complete,
        .on_credit = iree_hal_remote_server_on_bulk_credit,
        .user_data = &server->sessions[slot],
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
    iree_slim_mutex_lock(&server->session_mutex);
    if (server->sessions[slot].session == session) {
      server->sessions[slot].bulk_channel = bulk_channel;
      bulk_channel = NULL;  // Ownership transferred.
    }
    iree_slim_mutex_unlock(&server->session_mutex);

    iree_net_bulk_channel_release(bulk_channel);
  } else {
    iree_net_bulk_channel_release(bulk_channel);
    iree_async_buffer_pool_free(header_pool);
    iree_status_t shutdown_status = iree_net_session_shutdown(
        session, /*reason_code=*/0,
        iree_make_cstring_view("bulk channel setup failed"));
    iree_status_ignore(shutdown_status);
    iree_status_ignore(status);
  }

  iree_net_session_release(session);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Session callbacks
//===----------------------------------------------------------------------===//

void iree_hal_remote_server_on_session_ready(
    void* user_data, iree_net_session_t* session,
    const iree_net_session_topology_t* remote_topology) {
  iree_hal_remote_server_session_t* entry =
      (iree_hal_remote_server_session_t*)user_data;
  iree_hal_remote_server_t* server = entry->server;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)remote_topology;

  // If we're shutting down, immediately GOAWAY this newly-ready session.
  iree_slim_mutex_lock(&server->session_mutex);
  bool is_stopping = server->state != IREE_HAL_REMOTE_SERVER_STATE_STARTING &&
                     server->state != IREE_HAL_REMOTE_SERVER_STATE_RUNNING;
  iree_slim_mutex_unlock(&server->session_mutex);
  if (is_stopping) {
    iree_status_t goaway_status = iree_net_session_shutdown(
        session, /*reason_code=*/0, iree_make_cstring_view("server stopping"));
    iree_status_ignore(goaway_status);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Open application endpoints in the same order as the client: queue first,
  // then bulk. Endpoint-ready callbacks create and activate their channels.
  iree_hal_remote_server_endpoint_context_t* queue_context = NULL;
  iree_status_t status = iree_allocator_malloc(
      server->host_allocator, sizeof(*queue_context), (void**)&queue_context);
  if (iree_status_is_ok(status)) {
    queue_context->server = server;
    queue_context->session = session;
    iree_net_session_retain(session);
    queue_context->host_allocator = server->host_allocator;

    iree_net_endpoint_ready_callback_t endpoint_callback = {
        .fn = iree_hal_remote_server_on_queue_endpoint_ready,
        .user_data = queue_context,
    };
    status = iree_net_session_open_endpoint(session, endpoint_callback);
    if (!iree_status_is_ok(status)) {
      iree_net_session_release(session);
      iree_allocator_free(server->host_allocator, queue_context);
    }
  }
  iree_hal_remote_server_endpoint_context_t* bulk_context = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
        server->host_allocator, sizeof(*bulk_context), (void**)&bulk_context);
  }
  if (iree_status_is_ok(status)) {
    bulk_context->server = server;
    bulk_context->session = session;
    iree_net_session_retain(session);
    bulk_context->host_allocator = server->host_allocator;

    iree_net_endpoint_ready_callback_t endpoint_callback = {
        .fn = iree_hal_remote_server_on_bulk_endpoint_ready,
        .user_data = bulk_context,
    };
    status = iree_net_session_open_endpoint(session, endpoint_callback);
    if (!iree_status_is_ok(status)) {
      iree_net_session_release(session);
      iree_allocator_free(server->host_allocator, bulk_context);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_status_t shutdown_status = iree_net_session_shutdown(
        session, /*reason_code=*/0,
        iree_make_cstring_view("application endpoint setup failed"));
    iree_status_ignore(shutdown_status);
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_remote_server_on_session_goaway(void* user_data,
                                              iree_net_session_t* session,
                                              uint32_t reason_code,
                                              iree_string_view_t message) {
  iree_hal_remote_server_session_t* entry =
      (iree_hal_remote_server_session_t*)user_data;
  iree_hal_remote_server_t* server = entry->server;
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)reason_code;
  (void)message;
  // Client initiated graceful shutdown. Detach and release the session's
  // application channels immediately; queued command completion references
  // retain the queue channel and observe a detached endpoint.
  iree_hal_remote_server_remove_session(server, session);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_remote_server_on_session_error(void* user_data,
                                             iree_net_session_t* session,
                                             iree_status_t status) {
  iree_hal_remote_server_session_t* entry =
      (iree_hal_remote_server_session_t*)user_data;
  iree_hal_remote_server_t* server = entry->server;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Log and consume the error.
  iree_status_ignore(status);

  // Remove the failed session from tracking.
  iree_hal_remote_server_remove_session(server, session);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_remote_server_on_session_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status) {
  (void)user_data;
  iree_hal_remote_control_send_allocation_t* allocation =
      (iree_hal_remote_control_send_allocation_t*)(uintptr_t)
          operation_user_data;
  if (allocation) {
    iree_allocator_t host_allocator = allocation->host_allocator;
    iree_allocator_free(host_allocator, allocation);
  }
  iree_status_ignore(status);
}

iree_status_t iree_hal_remote_server_on_control_data(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  iree_hal_remote_server_session_t* entry =
      (iree_hal_remote_server_session_t*)user_data;

  // Parse control envelope.
  if (payload.data_length < sizeof(iree_hal_remote_control_envelope_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "control data too small for envelope: %" PRIhsz
                            " bytes",
                            payload.data_length);
  }
  const iree_hal_remote_control_envelope_t* envelope =
      (const iree_hal_remote_control_envelope_t*)payload.data;

  // Server should not receive responses.
  if (envelope->message_flags & IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "server received unexpected IS_RESPONSE flag");
  }

  // Message body starts after the envelope.
  const uint8_t* body =
      payload.data + sizeof(iree_hal_remote_control_envelope_t);
  iree_host_size_t body_length =
      payload.data_length - sizeof(iree_hal_remote_control_envelope_t);

  // Dispatch by message type.
  switch (envelope->message_type) {
    case IREE_HAL_REMOTE_CONTROL_BUFFER_ALLOC:
      return iree_hal_remote_server_handle_buffer_alloc(entry, envelope, body,
                                                        body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_MAP:
      return iree_hal_remote_server_handle_buffer_map(entry, envelope, body,
                                                      body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_UNMAP:
      return iree_hal_remote_server_handle_buffer_unmap(entry, envelope, body,
                                                        body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_QUERY_HEAPS:
      return iree_hal_remote_server_handle_buffer_query_heaps(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_EXECUTABLE_UPLOAD:
      return iree_hal_remote_server_handle_executable_upload(entry, envelope,
                                                             body, body_length);
    case IREE_HAL_REMOTE_CONTROL_COMMAND_BUFFER_UPLOAD:
      return iree_hal_remote_server_handle_command_buffer_upload(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_RESOURCE_RELEASE_BATCH:
      return iree_hal_remote_server_handle_resource_release_batch(entry, body,
                                                                  body_length);
    default:
      // For request/response messages, send an UNIMPLEMENTED error back.
      // For fire-and-forget messages, just return an error.
      if (!(envelope->message_flags &
            IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET)) {
        return iree_hal_remote_server_send_error_response(
            entry->server->host_allocator, entry->session, envelope,
            iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "unhandled control message type 0x%04x",
                             envelope->message_type));
      }
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unhandled control message type 0x%04x",
                              envelope->message_type);
  }
}
