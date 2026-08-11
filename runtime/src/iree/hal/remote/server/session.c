// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/session.h"

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/semaphore.h"
#include "iree/hal/remote/protocol/commands.h"
#include "iree/hal/remote/protocol/common.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/protocol/queue.h"
#include "iree/hal/remote/server/bulk.h"
#include "iree/hal/remote/server/bulk_session.h"
#include "iree/hal/remote/server/file_index.h"
#include "iree/hal/remote/server/profile.h"
#include "iree/hal/remote/server/profile_relay.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/util/queue_header_pool.h"
#include "iree/net/channel/bulk/bulk_channel.h"
#include "iree/net/channel/control/frame.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/status_wire.h"

// Inline arrays used while replaying uncommon synchronization commands.
// Larger command payloads spill to the host allocator instead of imposing a
// protocol limit.
#define IREE_HAL_REMOTE_INLINE_EVENT_COUNT 32
#define IREE_HAL_REMOTE_INLINE_BARRIER_COUNT 32

// Bounds retained ADVANCE records when the peer stops accepting progress.
// Crossing the bound terminalizes the session instead of admitting unbounded
// per-command state.
#define IREE_HAL_REMOTE_MAX_PENDING_ADVANCES 4096

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
  // Fixed ADVANCE payload prefix.
  iree_hal_remote_advance_payload_t advance_payload;
  // BUFFER_ALLOCA provisional-to-resolved mapping, valid when count is one.
  iree_hal_remote_resolution_entry_t resolution;
  // Serialized failure status appended to |advance_payload|, or NULL.
  uint8_t* status_wire;
} iree_hal_remote_server_command_completion_t;

static void iree_hal_remote_server_release_command_completion(
    iree_hal_remote_server_command_completion_t* completion) {
  if (!completion) return;
  iree_allocator_t host_allocator = completion->host_allocator;
  iree_allocator_free(host_allocator, completion->status_wire);
  iree_net_queue_channel_release(completion->queue_channel);
  iree_hal_semaphore_release(completion->local_semaphore);
  iree_hal_remote_server_release(completion->server);
  iree_allocator_free(host_allocator, completion);
}

static iree_status_t iree_hal_remote_server_allocate_command_completion(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    iree_net_queue_channel_t* queue_channel,
    const iree_async_frontier_t* signal_frontier,
    iree_hal_remote_server_command_completion_t** out_completion) {
  *out_completion = NULL;
  iree_hal_remote_server_command_completion_t* completion = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      server->host_allocator, sizeof(*completion), (void**)&completion));
  memset(completion, 0, sizeof(*completion));
  completion->host_allocator = server->host_allocator;
  completion->server = server;
  iree_hal_remote_server_retain(completion->server);
  completion->session_slot = session_slot;
  completion->session_id = session_id;
  completion->queue_channel = queue_channel;
  iree_net_queue_channel_retain(completion->queue_channel);
  iree_async_single_frontier_initialize(&completion->signal_frontier,
                                        signal_frontier->entries[0].axis,
                                        signal_frontier->entries[0].epoch);
  *out_completion = completion;
  return iree_ok_status();
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

struct iree_hal_remote_server_pending_queue_command_t {
  // Next command parked on the same provisional resource ID.
  iree_hal_remote_server_pending_queue_command_t* next;
  // Queue channel retained for the eventual ADVANCE frame.
  iree_net_queue_channel_t* queue_channel;
  // Host allocator used to free this command.
  iree_allocator_t host_allocator;
  // Receive-buffer lease keeping zero-copy frontier/payload pointers valid.
  iree_async_buffer_lease_t lease;
  // Provisional resource ID that must resolve before dispatch.
  iree_hal_remote_resource_id_t provisional_id;
  // Wait frontier referenced by |lease|; NULL if absent.
  const iree_async_frontier_t* wait_frontier;
  // Signal frontier referenced by |lease|.
  const iree_async_frontier_t* signal_frontier;
  // Queue command payload referenced by |lease|.
  iree_const_byte_span_t command_data;
};

static const iree_async_frontier_t*
iree_hal_remote_server_pending_queue_command_wait_frontier(
    iree_hal_remote_server_pending_queue_command_t* command) {
  return command->wait_frontier;
}

static const iree_async_frontier_t*
iree_hal_remote_server_pending_queue_command_signal_frontier(
    iree_hal_remote_server_pending_queue_command_t* command) {
  return command->signal_frontier;
}

static iree_const_byte_span_t iree_hal_remote_server_pending_queue_command_data(
    iree_hal_remote_server_pending_queue_command_t* command) {
  return command->command_data;
}

static void iree_hal_remote_server_free_pending_queue_command(
    iree_hal_remote_server_pending_queue_command_t* command) {
  if (!command) return;
  iree_allocator_t host_allocator = command->host_allocator;
  iree_async_buffer_lease_release(&command->lease);
  iree_net_queue_channel_release(command->queue_channel);
  iree_allocator_free(host_allocator, command);
}

static void iree_hal_remote_server_free_pending_queue_commands(
    iree_hal_remote_server_pending_queue_command_t* pending_list) {
  while (pending_list) {
    iree_hal_remote_server_pending_queue_command_t* next = pending_list->next;
    iree_hal_remote_server_free_pending_queue_command(pending_list);
    pending_list = next;
  }
}

static iree_status_t iree_hal_remote_server_observe_submission_frontier(
    iree_hal_remote_server_session_t* session_slot,
    const iree_async_frontier_t* signal_frontier);
iree_status_t iree_hal_remote_server_on_queue_command(
    void* user_data, uint32_t stream_id,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t command_data, iree_async_buffer_lease_t* lease);

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
    iree_net_sequence_node_t** out_pending_completions,
    iree_net_sequence_node_t** out_pending_advances) {
  *out_pending_releases = NULL;
  *out_pending_completions = NULL;
  *out_pending_advances = NULL;
  iree_net_sequence_window_take_pending(
      &session_slot->observed_submission_window, out_pending_releases);
  iree_net_sequence_window_deinitialize(
      &session_slot->observed_submission_window);

  iree_net_sequence_window_take_pending(&session_slot->completed_signal_window,
                                        out_pending_completions);
  iree_net_sequence_window_deinitialize(&session_slot->completed_signal_window);

  *out_pending_advances = session_slot->pending_advances.head;
  memset(&session_slot->pending_advances, 0,
         sizeof(session_slot->pending_advances));
  session_slot->queue_flags = 0;
}

void iree_hal_remote_server_session_deinitialize_windows(
    iree_hal_remote_server_session_t* session_slot) {
  iree_net_sequence_node_t* pending_releases = NULL;
  iree_net_sequence_node_t* pending_completions = NULL;
  iree_net_sequence_node_t* pending_advances = NULL;
  iree_hal_remote_server_session_take_window_nodes(
      session_slot, &pending_releases, &pending_completions, &pending_advances);
  iree_hal_remote_server_free_resource_release_nodes(pending_releases);
  iree_hal_remote_server_free_command_completion_nodes(pending_completions);
  iree_hal_remote_server_free_command_completion_nodes(pending_advances);
}

void iree_hal_remote_server_session_deinitialize_provisionals(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < session_slot->provisional_map.count; ++i) {
    iree_hal_remote_server_free_pending_queue_commands(
        session_slot->provisional_map.pending_heads[i]);
  }
  iree_allocator_free(host_allocator,
                      session_slot->provisional_map.provisional_ids);
  iree_allocator_free(host_allocator,
                      session_slot->provisional_map.resolved_ids);
  iree_allocator_free(host_allocator, session_slot->provisional_map.states);
  iree_allocator_free(host_allocator,
                      session_slot->provisional_map.status_codes);
  iree_allocator_free(host_allocator,
                      session_slot->provisional_map.pending_heads);
  iree_allocator_free(host_allocator,
                      session_slot->provisional_map.pending_tails);
  memset(&session_slot->provisional_map, 0,
         sizeof(session_slot->provisional_map));
}

static iree_hal_remote_file_registration_capabilities_t
iree_hal_remote_server_file_registration_capabilities(
    iree_net_carrier_t* carrier) {
  iree_hal_remote_file_registration_capabilities_t capabilities =
      IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE;
  if (carrier) {
    iree_net_carrier_capabilities_t carrier_capabilities =
        iree_net_carrier_capabilities(carrier);
    if (iree_any_bit_set(carrier_capabilities,
                         IREE_NET_CARRIER_CAPABILITY_POSIX_FD_TRANSFER)) {
      capabilities |= IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_POSIX_FD;
    }
    if (iree_any_bit_set(carrier_capabilities,
                         IREE_NET_CARRIER_CAPABILITY_WIN32_HANDLE_TRANSFER)) {
      capabilities |= IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_WIN32_HANDLE;
    }
  }
  return capabilities;
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
// Physical memory resources
//===----------------------------------------------------------------------===//

typedef struct iree_hal_remote_server_physical_memory_t {
  // HAL resource header used to store the wrapper in the session resource
  // table.
  iree_hal_resource_t resource;
  // Allocator that produced |physical_memory| and must free it.
  iree_hal_allocator_t* allocator;
  // Opaque physical memory handle owned by this wrapper.
  iree_hal_physical_memory_t* physical_memory;
  // Host allocator used to free this wrapper.
  iree_allocator_t host_allocator;
} iree_hal_remote_server_physical_memory_t;

static iree_status_t iree_hal_remote_server_physical_memory_free(
    iree_hal_remote_server_physical_memory_t* physical_memory) {
  if (!physical_memory->physical_memory) return iree_ok_status();
  iree_status_t status = iree_hal_allocator_physical_memory_free(
      physical_memory->allocator, physical_memory->physical_memory);
  if (iree_status_is_ok(status)) {
    physical_memory->physical_memory = NULL;
  }
  return status;
}

static void iree_hal_remote_server_physical_memory_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_remote_server_physical_memory_t* physical_memory =
      (iree_hal_remote_server_physical_memory_t*)resource;
  iree_allocator_t host_allocator = physical_memory->host_allocator;
  iree_status_ignore(
      iree_hal_remote_server_physical_memory_free(physical_memory));
  iree_hal_allocator_release(physical_memory->allocator);
  iree_allocator_free(host_allocator, physical_memory);
}

static const iree_hal_resource_vtable_t
    iree_hal_remote_server_physical_memory_vtable = {
        .destroy = iree_hal_remote_server_physical_memory_destroy,
};

static iree_status_t iree_hal_remote_server_physical_memory_create(
    iree_hal_allocator_t* allocator,
    iree_hal_physical_memory_t* physical_memory,
    iree_allocator_t host_allocator,
    iree_hal_remote_server_physical_memory_t** out_physical_memory) {
  *out_physical_memory = NULL;
  iree_hal_remote_server_physical_memory_t* wrapper = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*wrapper), (void**)&wrapper);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_remote_server_physical_memory_vtable,
                                 &wrapper->resource);
    iree_hal_allocator_retain(allocator);
    wrapper->allocator = allocator;
    wrapper->physical_memory = physical_memory;
    wrapper->host_allocator = host_allocator;
    *out_physical_memory = wrapper;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Buffer params helpers
//===----------------------------------------------------------------------===//

static iree_hal_buffer_params_t iree_hal_remote_server_wire_params_to_hal(
    iree_hal_remote_buffer_params_t wire_params) {
  return (iree_hal_buffer_params_t){
      .usage = (iree_hal_buffer_usage_t)wire_params.usage,
      .access = (iree_hal_memory_access_t)wire_params.access,
      .type = (iree_hal_memory_type_t)wire_params.type,
      .queue_affinity = (iree_hal_queue_affinity_t)wire_params.queue_affinity,
      .min_alignment = (iree_device_size_t)wire_params.min_alignment,
  };
}

static iree_hal_remote_buffer_params_t
iree_hal_remote_server_buffer_to_wire_params(iree_hal_buffer_t* buffer) {
  iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer);
  return (iree_hal_remote_buffer_params_t){
      .usage = iree_hal_buffer_allowed_usage(buffer),
      .access = (uint16_t)iree_hal_buffer_allowed_access(buffer),
      .type = iree_hal_buffer_memory_type(buffer),
      .queue_affinity = (uint64_t)placement.queue_affinity,
      .min_alignment = 0,
  };
}

//===----------------------------------------------------------------------===//
// Virtual buffer resources
//===----------------------------------------------------------------------===//

static bool iree_hal_remote_server_find_virtual_buffer(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t resource_id, iree_host_size_t* out_index) {
  *out_index = 0;
  for (iree_host_size_t i = 0; i < session_slot->virtual_buffer_map.count;
       ++i) {
    if (session_slot->virtual_buffer_map.resource_ids[i] == resource_id) {
      *out_index = i;
      return true;
    }
  }
  return false;
}

static bool iree_hal_remote_server_is_virtual_buffer(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t resource_id) {
  iree_host_size_t index = 0;
  return iree_hal_remote_server_find_virtual_buffer(session_slot, resource_id,
                                                    &index);
}

static iree_status_t iree_hal_remote_server_track_virtual_buffer(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t resource_id,
    iree_allocator_t host_allocator) {
  iree_host_size_t index = 0;
  if (iree_hal_remote_server_find_virtual_buffer(session_slot, resource_id,
                                                 &index)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "virtual buffer 0x%016" PRIx64 " is already tracked", resource_id);
  }

  iree_host_size_t minimum_capacity =
      session_slot->virtual_buffer_map.count + 1;
  if (minimum_capacity > session_slot->virtual_buffer_map.capacity) {
    iree_host_size_t new_capacity = session_slot->virtual_buffer_map.capacity;
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        host_allocator, minimum_capacity,
        sizeof(*session_slot->virtual_buffer_map.resource_ids), &new_capacity,
        (void**)&session_slot->virtual_buffer_map.resource_ids));
    session_slot->virtual_buffer_map.capacity = new_capacity;
  }

  session_slot->virtual_buffer_map
      .resource_ids[session_slot->virtual_buffer_map.count++] = resource_id;
  return iree_ok_status();
}

static void iree_hal_remote_server_untrack_virtual_buffer(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t resource_id) {
  iree_host_size_t index = 0;
  if (!iree_hal_remote_server_find_virtual_buffer(session_slot, resource_id,
                                                  &index)) {
    return;
  }
  --session_slot->virtual_buffer_map.count;
  session_slot->virtual_buffer_map.resource_ids[index] =
      session_slot->virtual_buffer_map
          .resource_ids[session_slot->virtual_buffer_map.count];
}

static iree_status_t iree_hal_remote_server_release_virtual_buffer(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t resource_id) {
  iree_hal_buffer_t* virtual_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &session_slot->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
          resource_id);
  if (!virtual_buffer) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "virtual buffer 0x%016" PRIx64 " not found",
                            resource_id);
  }

  iree_hal_allocator_t* allocator =
      iree_hal_device_allocator(session_slot->server->devices[0]);
  iree_status_t status =
      iree_hal_allocator_virtual_memory_release(allocator, virtual_buffer);
  if (iree_status_is_ok(status)) {
    // virtual_memory_release destroys |virtual_buffer| directly after enforcing
    // the backend's no-live-mappings contract. The resource table's retain is
    // now only a stale bookkeeping edge, so detach without releasing it again.
    (void)iree_hal_remote_resource_table_detach(&session_slot->resource_table,
                                                resource_id);
    iree_hal_remote_server_untrack_virtual_buffer(session_slot, resource_id);
  }
  return status;
}

void iree_hal_remote_server_session_deinitialize_resource_table(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator) {
  while (session_slot->virtual_buffer_map.count > 0) {
    iree_hal_remote_resource_id_t resource_id =
        session_slot->virtual_buffer_map
            .resource_ids[session_slot->virtual_buffer_map.count - 1];
    iree_status_t status = iree_hal_remote_server_release_virtual_buffer(
        session_slot, resource_id);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      // The backend rejected release, usually because mappings are still live.
      // Detach the reservation so generic resource-table teardown cannot run a
      // plain buffer destructor that bypasses the allocator's mapping registry.
      (void)iree_hal_remote_resource_table_detach(&session_slot->resource_table,
                                                  resource_id);
      iree_hal_remote_server_untrack_virtual_buffer(session_slot, resource_id);
    }
  }
  iree_allocator_free(host_allocator,
                      session_slot->virtual_buffer_map.resource_ids);
  memset(&session_slot->virtual_buffer_map, 0,
         sizeof(session_slot->virtual_buffer_map));

  iree_hal_remote_resource_table_deinitialize(&session_slot->resource_table,
                                              host_allocator);
}

//===----------------------------------------------------------------------===//
// Session removal
//===----------------------------------------------------------------------===//

static void iree_hal_remote_server_decrement_active_session_count_locked(
    iree_hal_remote_server_t* server,
    iree_hal_remote_server_stopped_callback_t* out_stopped_callback) {
  IREE_ASSERT_GT(server->active_session_count, 0u);
  --server->active_session_count;
  if (server->state == IREE_HAL_REMOTE_SERVER_STATE_STOPPING &&
      server->active_session_count == 0 && !server->listener) {
    server->state = IREE_HAL_REMOTE_SERVER_STATE_STOPPED;
    *out_stopped_callback = server->stopped_callback;
  }
}

void iree_hal_remote_server_session_complete_bulk_drain(
    iree_hal_remote_server_session_t* session_slot) {
  iree_hal_remote_server_t* server = session_slot->server;
  iree_hal_remote_server_bulk_session_t* bulk_session = NULL;
  iree_hal_remote_server_stopped_callback_t stopped_callback;
  memset(&stopped_callback, 0, sizeof(stopped_callback));

  iree_slim_mutex_lock(&server->session_mutex);
  if (iree_any_bit_set(
          session_slot->flags,
          IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING)) {
    session_slot->flags &=
        ~IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING;
    bulk_session = session_slot->bulk_session;
    iree_hal_remote_server_decrement_active_session_count_locked(
        server, &stopped_callback);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  iree_hal_remote_server_bulk_session_free(bulk_session);

  if (bulk_session) {
    iree_slim_mutex_lock(&server->session_mutex);
    if (session_slot->bulk_session == bulk_session) {
      session_slot->bulk_session = NULL;
    }
    iree_slim_mutex_unlock(&server->session_mutex);
  }
  if (stopped_callback.fn) {
    stopped_callback.fn(stopped_callback.user_data);
  }
}

void iree_hal_remote_server_remove_session(iree_hal_remote_server_t* server,
                                           iree_net_session_t* session) {
  iree_net_queue_channel_t* queue_channel = NULL;

  // Snapshot resources for cleanup outside the lock.
  iree_hal_remote_server_session_t resource_table_snapshot;
  memset(&resource_table_snapshot, 0, sizeof(resource_table_snapshot));
  iree_hal_remote_server_epoch_semaphore_map_t epoch_semaphore_map_snapshot;
  memset(&epoch_semaphore_map_snapshot, 0,
         sizeof(epoch_semaphore_map_snapshot));
  iree_hal_remote_server_session_t provisional_snapshot;
  memset(&provisional_snapshot, 0, sizeof(provisional_snapshot));
  iree_net_sequence_node_t* pending_releases = NULL;
  iree_net_sequence_node_t* pending_completions = NULL;
  iree_net_sequence_node_t* pending_advances = NULL;

  iree_slim_mutex_lock(&server->session_mutex);
  int32_t slot = iree_hal_remote_server_find_session_slot(server, session);
  if (slot >= 0) {
    // Snapshot references to clean up outside the lock.
    queue_channel = server->sessions[slot].queue_channel;
    server->sessions[slot].queue_channel = NULL;

    resource_table_snapshot.server = server;
    resource_table_snapshot.resource_table =
        server->sessions[slot].resource_table;
    memset(&server->sessions[slot].resource_table, 0,
           sizeof(server->sessions[slot].resource_table));
    resource_table_snapshot.virtual_buffer_map =
        server->sessions[slot].virtual_buffer_map;
    memset(&server->sessions[slot].virtual_buffer_map, 0,
           sizeof(server->sessions[slot].virtual_buffer_map));

    iree_hal_remote_server_epoch_semaphore_map_move(
        &server->sessions[slot].epoch_semaphore_map,
        &epoch_semaphore_map_snapshot);

    provisional_snapshot.provisional_map =
        server->sessions[slot].provisional_map;
    memset(&server->sessions[slot].provisional_map, 0,
           sizeof(server->sessions[slot].provisional_map));

    iree_hal_remote_server_session_take_window_nodes(
        &server->sessions[slot], &pending_releases, &pending_completions,
        &pending_advances);

    server->sessions[slot].session = NULL;
    server->sessions[slot].carrier = NULL;
    server->sessions[slot].file_registration_capabilities =
        IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE;
    server->sessions[slot].session_id = 0;
    server->sessions[slot].flags |=
        IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING;
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  if (slot < 0) return;  // Already removed (e.g., double callback).

  iree_hal_remote_server_profile_relay_cancel(&server->sessions[slot]);
  iree_hal_remote_server_profile_relay_deinitialize(&server->sessions[slot]);

  // Release all resources in the table.
  iree_hal_remote_server_session_deinitialize_resource_table(
      &resource_table_snapshot, server->host_allocator);

  iree_hal_remote_server_epoch_semaphore_map_deinitialize(
      &epoch_semaphore_map_snapshot, server->host_allocator);

  iree_hal_remote_server_session_deinitialize_provisionals(
      &provisional_snapshot, server->host_allocator);

  // Release owner-managed nodes that were pending in sequence windows.
  iree_hal_remote_server_free_resource_release_nodes(pending_releases);
  iree_hal_remote_server_free_command_completion_nodes(pending_completions);
  iree_hal_remote_server_free_command_completion_nodes(pending_advances);

  // Stop new bulk callbacks before failing transfers. DATA payload sends are
  // zero-copy, so the bulk state drains asynchronously until send completions
  // retire any staging memory referenced by the transport.
  iree_hal_remote_server_bulk_session_detach_channel(&server->sessions[slot]);

  // Detach channels from their endpoints before releasing the session. Command
  // completions may hold retained references to the queue channel that outlive
  // the session. Detach clears endpoint callbacks while endpoints are alive and
  // zeroes endpoint references so eventual channel destroy does not UAF.
  iree_net_queue_channel_detach(queue_channel);
  iree_net_queue_channel_release(queue_channel);
  iree_net_session_release(session);

  iree_hal_remote_server_bulk_session_deinitialize_transfers(
      &server->sessions[slot]);
}

//===----------------------------------------------------------------------===//
// Command completion
//===----------------------------------------------------------------------===//

typedef struct iree_hal_remote_server_session_failure_operation_t {
  // NOP used to enter the session's proactor thread.
  iree_async_nop_operation_t nop;
  // Server retained until failure publication completes.
  iree_hal_remote_server_t* server;
  // Session retained until failure publication completes.
  iree_net_session_t* session;
  // Terminal status transferred to iree_net_session_fail().
  iree_status_t status;
  // Host allocator used for this operation.
  iree_allocator_t host_allocator;
} iree_hal_remote_server_session_failure_operation_t;

static void iree_hal_remote_server_session_failure_operation_complete(
    void* user_data, iree_async_operation_t* operation,
    iree_status_t operation_status, iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_remote_server_session_failure_operation_t* failure_operation =
      (iree_hal_remote_server_session_failure_operation_t*)user_data;
  iree_hal_remote_server_t* server = failure_operation->server;
  iree_net_session_t* session = failure_operation->session;
  iree_allocator_t host_allocator = failure_operation->host_allocator;
  iree_status_t status = failure_operation->status;

  if (!iree_status_is_ok(operation_status)) {
    status = iree_status_join(
        status, iree_status_annotate(
                    operation_status,
                    IREE_SV("failed to dispatch terminal session failure")));
  }
  iree_net_session_fail(session, status);

  iree_net_session_release(session);
  iree_allocator_free(host_allocator, failure_operation);
  iree_hal_remote_server_release(server);
}

static void iree_hal_remote_server_fail_active_session(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    iree_status_t status) {
  iree_net_session_t* session = NULL;
  iree_hal_remote_server_t* server = session_slot->server;
  iree_slim_mutex_lock(&server->session_mutex);
  if (session_slot->session_id == session_id && session_slot->session) {
    session = session_slot->session;
    iree_net_session_retain(session);
  }
  iree_slim_mutex_unlock(&server->session_mutex);
  if (!session) {
    iree_status_free(status);
    return;
  }

  iree_hal_remote_server_session_failure_operation_t* failure_operation = NULL;
  iree_status_t schedule_status =
      iree_allocator_malloc(server->host_allocator, sizeof(*failure_operation),
                            (void**)&failure_operation);
  if (iree_status_is_ok(schedule_status)) {
    memset(failure_operation, 0, sizeof(*failure_operation));
    failure_operation->server = server;
    iree_hal_remote_server_retain(server);
    failure_operation->session = session;
    failure_operation->status = status;
    failure_operation->host_allocator = server->host_allocator;
    iree_async_operation_initialize(
        &failure_operation->nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        iree_hal_remote_server_session_failure_operation_complete,
        failure_operation);
    schedule_status = iree_async_proactor_submit_one(
        server->proactor, &failure_operation->nop.base);
  }
  if (!iree_status_is_ok(schedule_status)) {
    if (failure_operation) {
      iree_hal_remote_server_t* retained_server = failure_operation->server;
      iree_allocator_t host_allocator = failure_operation->host_allocator;
      iree_allocator_free(host_allocator, failure_operation);
      iree_hal_remote_server_release(retained_server);
    }
    iree_net_session_release(session);
    iree_status_abort(iree_status_join(
        status,
        iree_status_annotate(schedule_status,
                             IREE_SV("failed to schedule session failure"))));
  }
}

// Converts |status| into an owned ADVANCE payload and consumes it.
// Serialization is best effort: allocation failure preserves the status code
// while omitting the diagnostic wire data.
static void iree_hal_remote_server_prepare_failure_advance(
    iree_hal_remote_server_command_completion_t* completion,
    iree_status_t status) {
  memset(&completion->advance_payload, 0, sizeof(completion->advance_payload));
  completion->advance_payload.status_code = (uint16_t)iree_status_code(status);

  iree_host_size_t wire_size = 0;
  iree_net_status_wire_size(status, &wire_size);
  iree_status_t serialize_status = iree_ok_status();
  if (wire_size > UINT32_MAX) {
    serialize_status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                        "serialized status wire length %" PRIhsz
                                        " exceeds protocol limit %" PRIu32,
                                        wire_size, UINT32_MAX);
  } else {
    serialize_status =
        iree_allocator_malloc(completion->host_allocator, wire_size,
                              (void**)&completion->status_wire);
  }
  if (iree_status_is_ok(serialize_status)) {
    serialize_status = iree_net_status_wire_serialize(
        status, iree_make_byte_span(completion->status_wire, wire_size));
  }
  if (iree_status_is_ok(serialize_status)) {
    completion->advance_payload.status_wire_length = (uint32_t)wire_size;
  } else {
    iree_status_free(serialize_status);
    iree_allocator_free(completion->host_allocator, completion->status_wire);
    completion->status_wire = NULL;
  }
  iree_status_free(status);
}

static iree_status_t iree_hal_remote_server_send_command_completion(
    iree_hal_remote_server_command_completion_t* completion) {
  iree_async_frontier_t* signal_frontier =
      iree_async_single_frontier_as_frontier(&completion->signal_frontier);

  iree_async_span_t spans[3];
  iree_host_size_t span_count = 0;
  if (completion->advance_payload.resolution_count > 0 ||
      completion->advance_payload.status_code != IREE_STATUS_OK) {
    spans[span_count++] = iree_async_span_from_ptr(
        &completion->advance_payload, sizeof(completion->advance_payload));
  }
  if (completion->advance_payload.resolution_count > 0) {
    spans[span_count++] = iree_async_span_from_ptr(
        &completion->resolution, sizeof(completion->resolution));
  }
  if (completion->advance_payload.status_wire_length > 0) {
    spans[span_count++] = iree_async_span_from_ptr(
        completion->status_wire,
        completion->advance_payload.status_wire_length);
  }
  iree_net_queue_channel_t* queue_channel = completion->queue_channel;
  iree_net_queue_channel_retain(queue_channel);
  iree_status_t status = iree_net_queue_channel_send_advance(
      queue_channel, signal_frontier,
      iree_async_span_list_make(spans, span_count),
      (uint64_t)(uintptr_t)completion);
  iree_net_queue_channel_release(queue_channel);
  return status;
}

static iree_status_t iree_hal_remote_server_enqueue_advances_locked(
    iree_hal_remote_server_session_t* session_slot,
    iree_net_sequence_node_t* advance_list) {
  iree_host_size_t list_count = 0;
  iree_net_sequence_node_t* list_tail = NULL;
  for (iree_net_sequence_node_t* node = advance_list; node; node = node->next) {
    list_tail = node;
    ++list_count;
  }
  if (list_count > IREE_HAL_REMOTE_MAX_PENDING_ADVANCES -
                       session_slot->pending_advances.count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "remote queue retained ADVANCE limit exceeded (%u)",
                            IREE_HAL_REMOTE_MAX_PENDING_ADVANCES);
  }
  if (session_slot->pending_advances.tail) {
    session_slot->pending_advances.tail->next = advance_list;
  } else {
    session_slot->pending_advances.head = advance_list;
  }
  session_slot->pending_advances.tail = list_tail;
  session_slot->pending_advances.count += list_count;
  return iree_ok_status();
}

// Requests one retry after a new record or transport resource becomes
// available. An active drainer will either consume the record itself or record
// a send-ready edge if it encounters backpressure.
static bool iree_hal_remote_server_request_advance_drain_locked(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id) {
  if (session_slot->session_id != session_id || !session_slot->session ||
      !session_slot->pending_advances.head ||
      iree_any_bit_set(
          session_slot->queue_flags,
          IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_DRAIN_ACTIVE)) {
    return false;
  }
  session_slot->queue_flags &=
      ~IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_BACKPRESSURED;
  session_slot->queue_flags |=
      IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_DRAIN_ACTIVE;
  return true;
}

static void iree_hal_remote_server_drain_advances(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id) {
  iree_hal_remote_server_t* server = session_slot->server;
  while (true) {
    iree_hal_remote_server_command_completion_t* completion = NULL;
    iree_slim_mutex_lock(&server->session_mutex);
    if (session_slot->session_id == session_id && session_slot->session &&
        session_slot->pending_advances.head) {
      iree_net_sequence_node_t* node = session_slot->pending_advances.head;
      session_slot->pending_advances.head = node->next;
      if (!session_slot->pending_advances.head) {
        session_slot->pending_advances.tail = NULL;
      }
      --session_slot->pending_advances.count;
      node->next = NULL;
      completion = iree_containerof(
          node, iree_hal_remote_server_command_completion_t, sequence_node);
    } else if (session_slot->session_id == session_id) {
      session_slot->queue_flags &=
          ~(IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_DRAIN_ACTIVE |
            IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_READY_PENDING);
    }
    iree_slim_mutex_unlock(&server->session_mutex);
    if (!completion) return;

    iree_status_t send_status =
        iree_hal_remote_server_send_command_completion(completion);
    if (iree_status_is_ok(send_status)) {
      // The queue channel completion callback now owns |completion|. It may
      // already have released it when the transport completes synchronously.
      continue;
    }

    if (iree_status_code(send_status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      iree_status_free(send_status);
      bool retry_immediately = false;
      bool completion_requeued = false;
      iree_slim_mutex_lock(&server->session_mutex);
      if (session_slot->session_id == completion->session_id &&
          session_slot->session) {
        completion->sequence_node.next = session_slot->pending_advances.head;
        session_slot->pending_advances.head = &completion->sequence_node;
        if (!session_slot->pending_advances.tail) {
          session_slot->pending_advances.tail = &completion->sequence_node;
        }
        ++session_slot->pending_advances.count;
        completion_requeued = true;
        session_slot->queue_flags |=
            IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_BACKPRESSURED;
        if (iree_any_bit_set(
                session_slot->queue_flags,
                IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_READY_PENDING)) {
          session_slot->queue_flags &=
              ~(IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_BACKPRESSURED |
                IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_READY_PENDING);
          retry_immediately = true;
        } else {
          session_slot->queue_flags &=
              ~IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_DRAIN_ACTIVE;
        }
      }
      iree_slim_mutex_unlock(&server->session_mutex);
      if (!completion_requeued) {
        iree_hal_remote_server_release_command_completion(completion);
      }
      if (retry_immediately) continue;
      return;
    }

    iree_hal_remote_server_fail_active_session(
        session_slot, completion->session_id,
        iree_status_annotate(send_status,
                             IREE_SV("failed to send remote queue ADVANCE")));
    iree_hal_remote_server_release_command_completion(completion);
    return;
  }
}

static void iree_hal_remote_server_queue_ready_completions(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id,
    iree_net_sequence_node_t* ready_list) {
  if (!ready_list) return;
  iree_hal_remote_server_t* server = session_slot->server;
  bool start_drain = false;
  bool completions_transferred = false;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&server->session_mutex);
  if (session_slot->session_id == session_id && session_slot->session &&
      !iree_any_bit_set(session_slot->queue_flags,
                        IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_TERMINAL)) {
    status = iree_hal_remote_server_enqueue_advances_locked(session_slot,
                                                            ready_list);
    completions_transferred = iree_status_is_ok(status);
    if (completions_transferred) {
      start_drain = iree_hal_remote_server_request_advance_drain_locked(
          session_slot, session_id);
    }
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_fail_active_session(session_slot, session_id,
                                               status);
  }
  if (!completions_transferred) {
    iree_hal_remote_server_free_command_completion_nodes(ready_list);
  } else if (start_drain) {
    iree_hal_remote_server_drain_advances(session_slot, session_id);
  }
}

// Terminalizes the queue axis with |status| and consumes |completion|. Pending
// successful completions that were already ordered remain ahead of the failure
// ADVANCE; deferred and later completions are reclaimed without publication.
static void iree_hal_remote_server_terminalize_queue(
    iree_hal_remote_server_command_completion_t* completion,
    iree_status_t status) {
  iree_hal_remote_server_prepare_failure_advance(completion, status);

  iree_hal_remote_server_session_t* session_slot = completion->session_slot;
  iree_hal_remote_server_t* server = completion->server;
  uint64_t session_id = completion->session_id;
  iree_net_sequence_node_t* discarded_completions = NULL;
  bool completion_transferred = false;
  bool start_drain = false;
  iree_status_t enqueue_status = iree_ok_status();
  iree_slim_mutex_lock(&server->session_mutex);
  if (session_slot->session_id == completion->session_id &&
      session_slot->session &&
      !iree_any_bit_set(session_slot->queue_flags,
                        IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_TERMINAL)) {
    session_slot->queue_flags |= IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_TERMINAL;
    iree_net_sequence_window_take_pending(
        &session_slot->completed_signal_window, &discarded_completions);
    enqueue_status = iree_hal_remote_server_enqueue_advances_locked(
        session_slot, &completion->sequence_node);
    completion_transferred = iree_status_is_ok(enqueue_status);
    if (completion_transferred) {
      start_drain = iree_hal_remote_server_request_advance_drain_locked(
          session_slot, session_id);
    }
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  iree_hal_remote_server_free_command_completion_nodes(discarded_completions);
  if (!iree_status_is_ok(enqueue_status)) {
    iree_hal_remote_server_fail_active_session(session_slot, session_id,
                                               enqueue_status);
  }
  if (!completion_transferred) {
    iree_hal_remote_server_release_command_completion(completion);
  } else if (start_drain) {
    iree_hal_remote_server_drain_advances(session_slot, session_id);
  }
}

// Fails a command whose signal frontier was accepted but could not be
// submitted. The first failure publishes one terminal ADVANCE; later failures
// only retire their observed submission ownership.
static iree_status_t iree_hal_remote_server_fail_queue_command(
    iree_hal_remote_server_session_t* session_slot,
    const iree_async_frontier_t* signal_frontier, iree_status_t status) {
  iree_hal_remote_server_t* server = session_slot->server;
  iree_net_queue_channel_t* queue_channel = NULL;
  uint64_t session_id = 0;
  iree_slim_mutex_lock(&server->session_mutex);
  if (session_slot->session && session_slot->queue_channel &&
      !iree_any_bit_set(session_slot->queue_flags,
                        IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_TERMINAL)) {
    session_id = session_slot->session_id;
    queue_channel = session_slot->queue_channel;
    iree_net_queue_channel_retain(queue_channel);
  }
  iree_slim_mutex_unlock(&server->session_mutex);

  if (queue_channel) {
    iree_hal_remote_server_command_completion_t* completion = NULL;
    iree_status_t allocation_status =
        iree_hal_remote_server_allocate_command_completion(
            server, session_slot, session_id, queue_channel, signal_frontier,
            &completion);
    if (iree_status_is_ok(allocation_status)) {
      iree_hal_remote_server_terminalize_queue(completion, status);
    } else {
      iree_hal_remote_server_fail_active_session(
          session_slot, session_id,
          iree_status_join(status, allocation_status));
    }
    iree_net_queue_channel_release(queue_channel);
  } else {
    iree_status_free(status);
  }

  return iree_hal_remote_server_observe_submission_frontier(session_slot,
                                                            signal_frontier);
}

static void iree_hal_remote_server_notify_queue_send_ready(
    iree_hal_remote_server_session_t* session_slot, uint64_t session_id) {
  iree_hal_remote_server_t* server = session_slot->server;
  bool start_drain = false;
  iree_slim_mutex_lock(&server->session_mutex);
  if (session_slot->session_id == session_id && session_slot->session) {
    if (iree_any_bit_set(
            session_slot->queue_flags,
            IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_DRAIN_ACTIVE)) {
      session_slot->queue_flags |=
          IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_READY_PENDING;
    } else if (session_slot->pending_advances.head &&
               iree_any_bit_set(
                   session_slot->queue_flags,
                   IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_BACKPRESSURED)) {
      session_slot->queue_flags &=
          ~IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_ADVANCE_BACKPRESSURED;
      start_drain = iree_hal_remote_server_request_advance_drain_locked(
          session_slot, session_id);
    }
  }
  iree_slim_mutex_unlock(&server->session_mutex);
  if (start_drain) {
    iree_hal_remote_server_drain_advances(session_slot, session_id);
  }
}

void iree_hal_remote_server_on_queue_send_complete(void* user_data,
                                                   uint64_t operation_user_data,
                                                   iree_status_t status) {
  (void)user_data;
  iree_hal_remote_server_command_completion_t* completion =
      (iree_hal_remote_server_command_completion_t*)(uintptr_t)
          operation_user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_fail_active_session(
        completion->session_slot, completion->session_id,
        iree_status_annotate(status,
                             IREE_SV("remote queue ADVANCE transport failed")));
  } else {
    iree_status_free(status);
    iree_hal_remote_server_notify_queue_send_ready(completion->session_slot,
                                                   completion->session_id);
  }
  iree_hal_remote_server_release_command_completion(completion);
}

void iree_hal_remote_server_on_queue_send_ready(void* user_data) {
  iree_hal_remote_server_session_t* session_slot =
      (iree_hal_remote_server_session_t*)user_data;
  iree_hal_remote_server_t* server = session_slot->server;
  iree_slim_mutex_lock(&server->session_mutex);
  uint64_t session_id = session_slot->session_id;
  iree_slim_mutex_unlock(&server->session_mutex);
  iree_hal_remote_server_notify_queue_send_ready(session_slot, session_id);
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
  bool queue_terminal = false;

  iree_hal_remote_server_t* server = completion->server;
  iree_hal_remote_server_session_t* session_slot = completion->session_slot;
  iree_slim_mutex_lock(&server->session_mutex);
  session_active = session_slot->session_id == completion->session_id &&
                   session_slot->session != NULL;
  if (session_active) {
    removed_semaphore = iree_hal_remote_server_epoch_semaphore_map_remove(
        &session_slot->epoch_semaphore_map, signal_axis, signal_epoch);
    queue_terminal = iree_any_bit_set(
        session_slot->queue_flags, IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_TERMINAL);
    if (!queue_terminal && iree_status_is_ok(status)) {
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

  if (!session_active || queue_terminal) {
    iree_status_free(status);
    iree_hal_remote_server_free_command_completion_nodes(ready_completions);
  } else if (iree_status_is_ok(status)) {
    iree_status_free(status);
    iree_hal_remote_server_queue_ready_completions(
        session_slot, completion->session_id, ready_completions);
  } else {
    // Execution failure or invalid completion bookkeeping terminalizes the
    // queue axis. No later success may be published after this failure.
    iree_hal_remote_server_free_command_completion_nodes(ready_completions);
    iree_hal_remote_server_terminalize_queue(completion, status);
    completion_transferred = true;
  }

  if (!completion_transferred) {
    iree_hal_remote_server_release_command_completion(completion);
  }

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Provisional→resolved resource ID mapping
//===----------------------------------------------------------------------===//

static bool iree_hal_remote_server_find_provisional(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id, iree_host_size_t* out_index) {
  *out_index = 0;
  for (iree_host_size_t i = 0; i < session_slot->provisional_map.count; ++i) {
    if (session_slot->provisional_map.provisional_ids[i] == provisional_id) {
      *out_index = i;
      return true;
    }
  }
  return false;
}

static iree_status_t iree_hal_remote_server_reserve_provisional_map(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator) {
  // Cap distinct provisional IDs at the resource table capacity. Pending
  // provisionals that later resolve will consume resource table slots, so the
  // map should not admit more live names than the session can materialize.
  if (session_slot->provisional_map.count >=
      session_slot->resource_table.capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "provisional map full");
  }
  iree_host_size_t minimum_capacity = session_slot->provisional_map.count + 1;
  iree_status_t status = iree_ok_status();
  if (minimum_capacity > session_slot->provisional_map.capacity) {
    iree_host_size_t provisional_capacity =
        session_slot->provisional_map.capacity;
    status = iree_allocator_grow_array(
        host_allocator, minimum_capacity, sizeof(iree_hal_remote_resource_id_t),
        &provisional_capacity,
        (void**)&session_slot->provisional_map.provisional_ids);

    iree_host_size_t resolved_capacity = session_slot->provisional_map.capacity;
    if (iree_status_is_ok(status)) {
      status = iree_allocator_grow_array(
          host_allocator, minimum_capacity,
          sizeof(iree_hal_remote_resource_id_t), &resolved_capacity,
          (void**)&session_slot->provisional_map.resolved_ids);
    }

    iree_host_size_t state_capacity = session_slot->provisional_map.capacity;
    if (iree_status_is_ok(status)) {
      status = iree_allocator_grow_array(
          host_allocator, minimum_capacity,
          sizeof(iree_hal_remote_server_provisional_state_t), &state_capacity,
          (void**)&session_slot->provisional_map.states);
    }

    iree_host_size_t status_code_capacity =
        session_slot->provisional_map.capacity;
    if (iree_status_is_ok(status)) {
      status = iree_allocator_grow_array(
          host_allocator, minimum_capacity, sizeof(iree_status_code_t),
          &status_code_capacity,
          (void**)&session_slot->provisional_map.status_codes);
    }

    iree_host_size_t head_capacity = session_slot->provisional_map.capacity;
    if (iree_status_is_ok(status)) {
      status = iree_allocator_grow_array(
          host_allocator, minimum_capacity,
          sizeof(iree_hal_remote_server_pending_queue_command_t*),
          &head_capacity, (void**)&session_slot->provisional_map.pending_heads);
    }

    iree_host_size_t tail_capacity = session_slot->provisional_map.capacity;
    if (iree_status_is_ok(status)) {
      status = iree_allocator_grow_array(
          host_allocator, minimum_capacity,
          sizeof(iree_hal_remote_server_pending_queue_command_t*),
          &tail_capacity, (void**)&session_slot->provisional_map.pending_tails);
    }

    if (iree_status_is_ok(status)) {
      session_slot->provisional_map.capacity =
          iree_min(iree_min(provisional_capacity, resolved_capacity),
                   iree_min(state_capacity,
                            iree_min(status_code_capacity,
                                     iree_min(head_capacity, tail_capacity))));
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_server_insert_provisional(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id,
    iree_hal_remote_server_provisional_state_t state,
    iree_hal_remote_resource_id_t resolved_id, iree_allocator_t host_allocator,
    iree_host_size_t* out_index) {
  *out_index = 0;
  iree_status_t status = iree_hal_remote_server_reserve_provisional_map(
      session_slot, host_allocator);
  if (iree_status_is_ok(status)) {
    iree_host_size_t index = session_slot->provisional_map.count++;
    session_slot->provisional_map.provisional_ids[index] = provisional_id;
    session_slot->provisional_map.resolved_ids[index] = resolved_id;
    session_slot->provisional_map.states[index] = state;
    session_slot->provisional_map.status_codes[index] = IREE_STATUS_OK;
    session_slot->provisional_map.pending_heads[index] = NULL;
    session_slot->provisional_map.pending_tails[index] = NULL;
    *out_index = index;
  }
  return status;
}

// Prepares a provisional ID for a control-channel operation that will resolve
// later. Queue commands that arrive first can already have created the pending
// entry; in that case FILE_OPEN claims the same entry and later resolves it.
static iree_status_t iree_hal_remote_server_prepare_provisional(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id,
    iree_allocator_t host_allocator) {
  iree_host_size_t index = 0;
  iree_status_t status = iree_ok_status();
  if (iree_hal_remote_server_find_provisional(session_slot, provisional_id,
                                              &index)) {
    if (session_slot->provisional_map.states[index] ==
        IREE_HAL_REMOTE_SERVER_PROVISIONAL_RESOLVED) {
      status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "provisional resource 0x%016" PRIx64
                                " already has a resolved mapping",
                                provisional_id);
    } else if (session_slot->provisional_map.states[index] ==
               IREE_HAL_REMOTE_SERVER_PROVISIONAL_FAILED) {
      iree_status_code_t status_code =
          session_slot->provisional_map.status_codes[index];
      if (status_code == IREE_STATUS_OK) {
        status_code = IREE_STATUS_FAILED_PRECONDITION;
      }
      status = iree_make_status(status_code,
                                "provisional resource 0x%016" PRIx64
                                " already failed resolution",
                                provisional_id);
    }
  } else {
    status = iree_hal_remote_server_insert_provisional(
        session_slot, provisional_id,
        IREE_HAL_REMOTE_SERVER_PROVISIONAL_PENDING,
        /*resolved_id=*/0, host_allocator, &index);
  }
  return status;
}

// Stores or completes a provisional→resolved resource ID mapping. Used by
// BUFFER_ALLOCA, FILE_OPEN, and similar operations so later commands
// referencing the provisional ID can be resolved.
static iree_status_t iree_hal_remote_server_store_provisional(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id,
    iree_hal_remote_resource_id_t resolved_id, iree_allocator_t host_allocator,
    iree_hal_remote_server_pending_queue_command_t** out_pending_commands) {
  if (out_pending_commands) *out_pending_commands = NULL;

  iree_host_size_t index = 0;
  iree_status_t status = iree_ok_status();
  if (iree_hal_remote_server_find_provisional(session_slot, provisional_id,
                                              &index)) {
    if (session_slot->provisional_map.states[index] ==
        IREE_HAL_REMOTE_SERVER_PROVISIONAL_RESOLVED) {
      status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "provisional resource 0x%016" PRIx64
                                " already has a resolved mapping",
                                provisional_id);
    } else if (session_slot->provisional_map.states[index] ==
               IREE_HAL_REMOTE_SERVER_PROVISIONAL_FAILED) {
      iree_status_code_t status_code =
          session_slot->provisional_map.status_codes[index];
      if (status_code == IREE_STATUS_OK) {
        status_code = IREE_STATUS_FAILED_PRECONDITION;
      }
      status = iree_make_status(status_code,
                                "provisional resource 0x%016" PRIx64
                                " already failed resolution",
                                provisional_id);
    }
  } else {
    status = iree_hal_remote_server_insert_provisional(
        session_slot, provisional_id,
        IREE_HAL_REMOTE_SERVER_PROVISIONAL_PENDING,
        /*resolved_id=*/0, host_allocator, &index);
  }

  if (iree_status_is_ok(status)) {
    session_slot->provisional_map.resolved_ids[index] = resolved_id;
    session_slot->provisional_map.states[index] =
        IREE_HAL_REMOTE_SERVER_PROVISIONAL_RESOLVED;
    session_slot->provisional_map.status_codes[index] = IREE_STATUS_OK;
    if (out_pending_commands) {
      *out_pending_commands =
          session_slot->provisional_map.pending_heads[index];
    } else {
      iree_hal_remote_server_free_pending_queue_commands(
          session_slot->provisional_map.pending_heads[index]);
    }
    session_slot->provisional_map.pending_heads[index] = NULL;
    session_slot->provisional_map.pending_tails[index] = NULL;
  }

  return status;
}

// Marks a provisional ID as failed and detaches any queue commands waiting for
// its resolution.
static void iree_hal_remote_server_fail_provisional(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id,
    iree_status_code_t status_code, iree_allocator_t host_allocator,
    iree_hal_remote_server_pending_queue_command_t** out_pending_commands) {
  if (out_pending_commands) *out_pending_commands = NULL;

  iree_host_size_t index = 0;
  if (!iree_hal_remote_server_find_provisional(session_slot, provisional_id,
                                               &index)) {
    iree_status_ignore(iree_hal_remote_server_insert_provisional(
        session_slot, provisional_id, IREE_HAL_REMOTE_SERVER_PROVISIONAL_FAILED,
        /*resolved_id=*/0, host_allocator, &index));
  }
  if (index < session_slot->provisional_map.count) {
    session_slot->provisional_map.resolved_ids[index] = 0;
    session_slot->provisional_map.states[index] =
        IREE_HAL_REMOTE_SERVER_PROVISIONAL_FAILED;
    session_slot->provisional_map.status_codes[index] = status_code;
    if (out_pending_commands) {
      *out_pending_commands =
          session_slot->provisional_map.pending_heads[index];
    } else {
      iree_hal_remote_server_free_pending_queue_commands(
          session_slot->provisional_map.pending_heads[index]);
    }
    session_slot->provisional_map.pending_heads[index] = NULL;
    session_slot->provisional_map.pending_tails[index] = NULL;
  }
}

// Removes a provisional mapping by provisional_id. Called when the resolved
// resource is released or a pending open fails to prevent unbounded map growth.
static iree_hal_remote_server_pending_queue_command_t*
iree_hal_remote_server_remove_provisional(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id) {
  iree_hal_remote_server_pending_queue_command_t* pending_commands = NULL;
  iree_host_size_t index = 0;
  if (iree_hal_remote_server_find_provisional(session_slot, provisional_id,
                                              &index)) {
    pending_commands = session_slot->provisional_map.pending_heads[index];
    iree_host_size_t last = session_slot->provisional_map.count - 1;
    if (index != last) {
      session_slot->provisional_map.provisional_ids[index] =
          session_slot->provisional_map.provisional_ids[last];
      session_slot->provisional_map.resolved_ids[index] =
          session_slot->provisional_map.resolved_ids[last];
      session_slot->provisional_map.states[index] =
          session_slot->provisional_map.states[last];
      session_slot->provisional_map.status_codes[index] =
          session_slot->provisional_map.status_codes[last];
      session_slot->provisional_map.pending_heads[index] =
          session_slot->provisional_map.pending_heads[last];
      session_slot->provisional_map.pending_tails[index] =
          session_slot->provisional_map.pending_tails[last];
    }
    --session_slot->provisional_map.count;
  }
  return pending_commands;
}

static iree_status_t iree_hal_remote_server_enqueue_pending_queue_command(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id,
    iree_net_queue_channel_t* queue_channel,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t command_data, iree_async_buffer_lease_t* lease,
    iree_allocator_t host_allocator) {
  iree_status_t status = iree_ok_status();
  if (!lease || !lease->release.fn) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "queue message endpoint did not provide retainable receive storage for "
        "deferred provisional resource 0x%016" PRIx64,
        provisional_id);
  }

  iree_hal_remote_server_pending_queue_command_t* command = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*command),
                                   (void**)&command);
  }
  if (iree_status_is_ok(status)) {
    memset(command, 0, sizeof(*command));
    command->host_allocator = host_allocator;
    command->provisional_id = provisional_id;
    command->queue_channel = queue_channel;
    iree_net_queue_channel_retain(command->queue_channel);

    command->wait_frontier = wait_frontier;
    command->signal_frontier = signal_frontier;
    command->command_data = command_data;
  }

  iree_host_size_t index = 0;
  if (iree_status_is_ok(status) && !iree_hal_remote_server_find_provisional(
                                       session_slot, provisional_id, &index)) {
    status = iree_hal_remote_server_insert_provisional(
        session_slot, provisional_id,
        IREE_HAL_REMOTE_SERVER_PROVISIONAL_PENDING,
        /*resolved_id=*/0, host_allocator, &index);
  }
  if (iree_status_is_ok(status)) {
    if (session_slot->provisional_map.states[index] ==
        IREE_HAL_REMOTE_SERVER_PROVISIONAL_RESOLVED) {
      status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "provisional resource 0x%016" PRIx64
                                " resolved while queue command "
                                "was being parked",
                                provisional_id);
    } else if (session_slot->provisional_map.states[index] ==
               IREE_HAL_REMOTE_SERVER_PROVISIONAL_FAILED) {
      iree_status_code_t status_code =
          session_slot->provisional_map.status_codes[index];
      if (status_code == IREE_STATUS_OK) {
        status_code = IREE_STATUS_FAILED_PRECONDITION;
      }
      status = iree_make_status(status_code,
                                "provisional resource 0x%016" PRIx64
                                " failed before queue command arrived",
                                provisional_id);
    } else {
      // Transfer receive storage only after every operation that can reject
      // the enqueue has succeeded. On failure the caller still owns |lease|
      // and may use frontier or command pointers until its callback returns.
      command->lease = *lease;
      memset(lease, 0, sizeof(*lease));
      if (session_slot->provisional_map.pending_tails[index]) {
        session_slot->provisional_map.pending_tails[index]->next = command;
      } else {
        session_slot->provisional_map.pending_heads[index] = command;
      }
      session_slot->provisional_map.pending_tails[index] = command;
      command = NULL;
    }
  }
  iree_hal_remote_server_free_pending_queue_command(command);
  return status;
}

static bool iree_hal_remote_server_is_provisional_resolved(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t provisional_id) {
  iree_host_size_t index = 0;
  return iree_hal_remote_server_find_provisional(session_slot, provisional_id,
                                                 &index) &&
         session_slot->provisional_map.states[index] ==
             IREE_HAL_REMOTE_SERVER_PROVISIONAL_RESOLVED;
}

static bool iree_hal_remote_server_file_command_waits_on_provisional(
    iree_hal_remote_server_session_t* session_slot,
    iree_const_byte_span_t command_data,
    iree_hal_remote_resource_id_t* out_provisional_id) {
  *out_provisional_id = 0;
  if (command_data.data_length < sizeof(iree_hal_remote_queue_op_header_t)) {
    return false;
  }
  const iree_hal_remote_queue_op_header_t* op_header =
      (const iree_hal_remote_queue_op_header_t*)command_data.data;
  iree_hal_remote_resource_id_t file_id = 0;
  if (op_header->type == IREE_HAL_REMOTE_QUEUE_OP_FILE_READ &&
      command_data.data_length >= sizeof(iree_hal_remote_file_read_op_t)) {
    const iree_hal_remote_file_read_op_t* op =
        (const iree_hal_remote_file_read_op_t*)command_data.data;
    file_id = op->source_file_id;
  } else if (op_header->type == IREE_HAL_REMOTE_QUEUE_OP_FILE_WRITE &&
             command_data.data_length >=
                 sizeof(iree_hal_remote_file_write_op_t)) {
    const iree_hal_remote_file_write_op_t* op =
        (const iree_hal_remote_file_write_op_t*)command_data.data;
    file_id = op->target_file_id;
  }

  bool waits_on_provisional = false;
  if (file_id != 0 && IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(file_id) &&
      !iree_hal_remote_server_is_provisional_resolved(session_slot, file_id)) {
    *out_provisional_id = file_id;
    waits_on_provisional = true;
  }
  return waits_on_provisional;
}

static iree_status_t iree_hal_remote_server_fail_pending_queue_command(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_server_pending_queue_command_t* pending_command,
    iree_status_t status) {
  IREE_ASSERT_ARGUMENT(pending_command);
  const iree_async_frontier_t* signal_frontier =
      iree_hal_remote_server_pending_queue_command_signal_frontier(
          pending_command);
  iree_status_t observe_status = iree_hal_remote_server_fail_queue_command(
      session_slot, signal_frontier, status);
  iree_hal_remote_server_free_pending_queue_command(pending_command);
  return observe_status;
}

static iree_status_t iree_hal_remote_server_fail_pending_queue_commands(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_server_pending_queue_command_t* pending_commands,
    iree_status_t status) {
  if (!pending_commands) {
    return iree_status_ignore(status);
  }
  iree_status_t fail_status = iree_ok_status();
  while (pending_commands) {
    iree_hal_remote_server_pending_queue_command_t* next =
        pending_commands->next;
    pending_commands->next = NULL;
    iree_status_t command_status = next ? iree_status_clone(status) : status;
    if (!next) status = iree_ok_status();
    fail_status = iree_status_join(
        fail_status, iree_hal_remote_server_fail_pending_queue_command(
                         session_slot, pending_commands, command_status));
    pending_commands = next;
  }
  iree_status_ignore(status);
  return fail_status;
}

static iree_status_t iree_hal_remote_server_process_pending_queue_command(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_server_pending_queue_command_t* pending_command) {
  const iree_async_frontier_t* wait_frontier =
      iree_hal_remote_server_pending_queue_command_wait_frontier(
          pending_command);
  const iree_async_frontier_t* signal_frontier =
      iree_hal_remote_server_pending_queue_command_signal_frontier(
          pending_command);
  iree_const_byte_span_t command_data =
      iree_hal_remote_server_pending_queue_command_data(pending_command);
  iree_status_t status = iree_hal_remote_server_on_queue_command(
      session_slot, /*stream_id=*/0, wait_frontier, signal_frontier,
      command_data, /*lease=*/NULL);
  if (!iree_status_is_ok(status)) {
    status = iree_hal_remote_server_fail_pending_queue_command(
        session_slot, pending_command, status);
  } else {
    iree_hal_remote_server_free_pending_queue_command(pending_command);
  }
  return status;
}

// Resolves a resource ID that may be provisional. If the ID has the
// PROVISIONAL flag set, looks up a resolved mapping and returns it. Pending or
// unknown provisionals return unchanged so resource table lookups fail loudly.
static iree_hal_remote_resource_id_t iree_hal_remote_server_resolve_resource_id(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_resource_id_t resource_id) {
  if (!IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(resource_id)) {
    return resource_id;
  }
  for (iree_host_size_t i = 0; i < session_slot->provisional_map.count; ++i) {
    if (session_slot->provisional_map.provisional_ids[i] == resource_id &&
        session_slot->provisional_map.states[i] ==
            IREE_HAL_REMOTE_SERVER_PROVISIONAL_RESOLVED) {
      return session_slot->provisional_map.resolved_ids[i];
    }
  }
  return resource_id;
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
        iree_hal_remote_server_epoch_semaphore_map_lookup(
            &session_slot->epoch_semaphore_map, wait_frontier->entries[i].axis,
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
    status = iree_hal_remote_server_allocate_command_completion(
        server, session_slot, session_id, queue_channel, signal_frontier,
        &completion);
  }

  if (iree_status_is_ok(status)) {
    completion->local_semaphore = local_semaphore;
    iree_hal_semaphore_retain(local_semaphore);
    completion->timepoint.callback = iree_hal_remote_server_on_command_complete;
    completion->timepoint.user_data = completion;
  }

  // Store epoch→semaphore mapping for future wait frontier resolution.
  bool epoch_mapping_stored = false;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&server->session_mutex);
    if (session_slot->session_id == session_id && session_slot->session) {
      status = iree_hal_remote_server_epoch_semaphore_map_insert(
          &session_slot->epoch_semaphore_map, signal_axis, signal_epoch,
          local_semaphore, server->host_allocator);
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
      completion->advance_payload.resolution_count =
          op_context->resolution_count;
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
        removed_semaphore = iree_hal_remote_server_epoch_semaphore_map_remove(
            &session_slot->epoch_semaphore_map, signal_axis, signal_epoch);
      }
      iree_slim_mutex_unlock(&server->session_mutex);
      iree_hal_semaphore_release(removed_semaphore);
    }

    if (local_semaphore) {
      iree_hal_semaphore_fail(local_semaphore,
                              iree_status_clone(failure_status));
    }
    if (completion) {
      iree_hal_remote_server_terminalize_queue(completion, failure_status);
      completion = NULL;
    } else {
      iree_hal_remote_server_fail_active_session(session_slot, session_id,
                                                 failure_status);
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

  // Inline source data follows the op struct.
  if (op->length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "BUFFER_UPDATE inline data exceeds host capacity");
  }
  iree_host_size_t inline_data_offset = 0;
  iree_host_size_t update_required = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_buffer_update_op_t), &update_required,
      IREE_STRUCT_FIELD((iree_host_size_t)op->length, uint8_t,
                        &inline_data_offset)));
  if (context->command_data.data_length < update_required) {
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
        context->session_slot->server->host_allocator,
        /*out_pending_commands=*/NULL);
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

static iree_status_t iree_hal_remote_server_submit_client_file_read(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_client_file_read_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CLIENT_FILE_READ command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_client_file_read_op_t));
  }
  const iree_hal_remote_client_file_read_op_t* op =
      (const iree_hal_remote_client_file_read_op_t*)context->command_data.data;
  if (op->transfer_id == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CLIENT_FILE_READ transfer_id must be non-zero");
  }
  if (op->read_flags != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported CLIENT_FILE_READ flags: 0x%" PRIx64,
                            op->read_flags);
  }
  if (op->length > IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "CLIENT_FILE_READ length %" PRIu64
                            " exceeds device size max %" PRIu64,
                            op->length, (uint64_t)IREE_DEVICE_SIZE_MAX);
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
        "CLIENT_FILE_READ target buffer 0x%016" PRIx64 " not found", target_id);
  }

  return iree_hal_remote_server_bulk_submit_client_file_read(
      context->session_slot, local_device, wait_list, signal_list,
      op->transfer_id, target_buffer, op->target_offset,
      (iree_device_size_t)op->length, (iree_hal_read_flags_t)op->read_flags);
}

static iree_status_t iree_hal_remote_server_submit_client_file_write(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_client_file_write_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CLIENT_FILE_WRITE command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_client_file_write_op_t));
  }
  const iree_hal_remote_client_file_write_op_t* op =
      (const iree_hal_remote_client_file_write_op_t*)context->command_data.data;
  if (op->transfer_id == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CLIENT_FILE_WRITE transfer_id must be non-zero");
  }
  if (op->write_flags != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported CLIENT_FILE_WRITE flags: 0x%" PRIx64,
                            op->write_flags);
  }
  if (op->length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "CLIENT_FILE_WRITE length %" PRIu64
                            " exceeds host size max %" PRIhsz,
                            op->length, IREE_HOST_SIZE_MAX);
  }

  iree_hal_remote_resource_id_t source_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->source_buffer_id);
  iree_hal_buffer_t* source_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, source_id);
  if (!source_buffer) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "CLIENT_FILE_WRITE source buffer 0x%016" PRIx64
                            " not found",
                            source_id);
  }

  return iree_hal_remote_server_bulk_submit_client_file_write(
      context->session_slot, local_device, wait_list, signal_list,
      op->transfer_id, source_buffer, op->source_offset,
      (iree_device_size_t)op->length, (iree_hal_write_flags_t)op->write_flags);
}

static iree_status_t iree_hal_remote_server_submit_file_read(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_file_read_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "FILE_READ command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_file_read_op_t));
  }
  const iree_hal_remote_file_read_op_t* op =
      (const iree_hal_remote_file_read_op_t*)context->command_data.data;
  if (op->read_flags != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported FILE_READ flags: 0x%" PRIx64,
                            op->read_flags);
  }
  if (op->length > IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "FILE_READ length %" PRIu64
                            " exceeds device size max %" PRIu64,
                            op->length, (uint64_t)IREE_DEVICE_SIZE_MAX);
  }

  iree_hal_remote_resource_id_t source_file_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->source_file_id);
  iree_hal_file_t* source_file =
      (iree_hal_file_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_FILE, source_file_id);
  if (!source_file) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "FILE_READ source file 0x%016" PRIx64 " not found",
                            source_file_id);
  }

  iree_hal_remote_resource_id_t target_buffer_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->target_buffer_id);
  iree_hal_buffer_t* target_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, target_buffer_id);
  if (!target_buffer) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "FILE_READ target buffer 0x%016" PRIx64 " not found", target_buffer_id);
  }

  return iree_hal_device_queue_read(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      source_file, op->source_offset, target_buffer, op->target_offset,
      (iree_device_size_t)op->length, (iree_hal_read_flags_t)op->read_flags);
}

static iree_status_t iree_hal_remote_server_submit_file_write(
    void* user_data, iree_hal_device_t* local_device,
    iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list) {
  iree_hal_remote_server_op_context_t* context =
      (iree_hal_remote_server_op_context_t*)user_data;
  if (context->command_data.data_length <
      sizeof(iree_hal_remote_file_write_op_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "FILE_WRITE command too short: %" PRIhsz
                            " < %" PRIhsz,
                            context->command_data.data_length,
                            sizeof(iree_hal_remote_file_write_op_t));
  }
  const iree_hal_remote_file_write_op_t* op =
      (const iree_hal_remote_file_write_op_t*)context->command_data.data;
  if (op->write_flags != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported FILE_WRITE flags: 0x%" PRIx64,
                            op->write_flags);
  }
  if (op->length > IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "FILE_WRITE length %" PRIu64
                            " exceeds device size max %" PRIu64,
                            op->length, (uint64_t)IREE_DEVICE_SIZE_MAX);
  }

  iree_hal_remote_resource_id_t source_buffer_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->source_buffer_id);
  iree_hal_buffer_t* source_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, source_buffer_id);
  if (!source_buffer) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "FILE_WRITE source buffer 0x%016" PRIx64
                            " not found",
                            source_buffer_id);
  }

  iree_hal_remote_resource_id_t target_file_id =
      iree_hal_remote_server_resolve_resource_id(context->session_slot,
                                                 op->target_file_id);
  iree_hal_file_t* target_file =
      (iree_hal_file_t*)iree_hal_remote_resource_table_lookup(
          &context->session_slot->resource_table,
          IREE_HAL_REMOTE_RESOURCE_TYPE_FILE, target_file_id);
  if (!target_file) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "FILE_WRITE target file 0x%016" PRIx64 " not found",
                            target_file_id);
  }

  return iree_hal_device_queue_write(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      source_buffer, op->source_offset, target_file, op->target_offset,
      (iree_device_size_t)op->length, (iree_hal_write_flags_t)op->write_flags);
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

  // Parse variable-length constants and bindings.
  iree_host_size_t constants_offset = 0;
  iree_host_size_t bindings_offset = 0;
  iree_host_size_t required_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_dispatch_op_t), &required_length,
      IREE_STRUCT_FIELD(op->constant_count, uint32_t, &constants_offset),
      IREE_STRUCT_FIELD_ALIGNED(op->binding_count, iree_hal_remote_binding_t, 8,
                                &bindings_offset)));
  if (context->command_data.data_length < required_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DISPATCH command truncated: bindings");
  }

  iree_const_byte_span_t constants = iree_make_const_byte_span(
      context->command_data.data + constants_offset,
      (iree_host_size_t)op->constant_count * sizeof(uint32_t));

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
      executable, iree_hal_executable_function_from_value(op->function_value),
      local_config, constants, binding_list,
      (iree_hal_dispatch_flags_t)op->dispatch_flags);
}

//===----------------------------------------------------------------------===//
// Control channel dispatch
//===----------------------------------------------------------------------===//

static iree_hal_remote_control_envelope_t
iree_hal_remote_server_make_response_envelope(
    const iree_hal_remote_control_envelope_t* request_envelope) {
  iree_hal_remote_control_envelope_t envelope;
  memset(&envelope, 0, sizeof(envelope));
  envelope.message_type = request_envelope->message_type;
  envelope.message_flags = IREE_HAL_REMOTE_CONTROL_FLAG_IS_RESPONSE;
  envelope.request_id = request_envelope->request_id;
  return envelope;
}

static bool iree_hal_remote_server_response_fits_inline_copy(
    iree_host_size_t response_length) {
  if (IREE_NET_FRAME_SENDER_INLINE_FRAME_CAPACITY <
      IREE_NET_CONTROL_FRAME_HEADER_SIZE) {
    return false;
  }
  return response_length <= IREE_NET_FRAME_SENDER_INLINE_FRAME_CAPACITY -
                                IREE_NET_CONTROL_FRAME_HEADER_SIZE;
}

static iree_status_t iree_hal_remote_server_response_length(
    iree_host_size_t body_header_length, iree_host_size_t data_length,
    iree_host_size_t* out_response_length) {
  *out_response_length = 0;
  iree_host_size_t body_length = 0;
  iree_status_t status =
      iree_host_size_checked_add(body_header_length, data_length, &body_length)
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "response body size overflow");
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_add(
          sizeof(iree_hal_remote_control_envelope_t) +
              sizeof(iree_hal_remote_control_response_prefix_t),
          body_length, out_response_length)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "response payload size overflow");
  }
  return status;
}

// Sends a control channel response by synchronously copying stack/local spans
// into the net frame sender's retained inline storage.
static iree_status_t iree_hal_remote_server_send_response_copy(
    iree_net_session_t* session,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_code_t status_code, const void* body_header,
    iree_host_size_t body_header_length, const void* data,
    iree_host_size_t data_length) {
  iree_hal_remote_control_envelope_t envelope =
      iree_hal_remote_server_make_response_envelope(request_envelope);
  iree_hal_remote_control_response_prefix_t prefix;
  memset(&prefix, 0, sizeof(prefix));
  prefix.status_code = (uint32_t)status_code;

  iree_async_span_t spans[4];
  iree_host_size_t span_count = 0;
  spans[span_count++] = iree_async_span_from_ptr(&envelope, sizeof(envelope));
  spans[span_count++] = iree_async_span_from_ptr(&prefix, sizeof(prefix));
  if (body_header && body_header_length > 0) {
    spans[span_count++] =
        iree_async_span_from_ptr((void*)body_header, body_header_length);
  }
  if (data && data_length > 0) {
    spans[span_count++] = iree_async_span_from_ptr((void*)data, data_length);
  }

  iree_async_span_list_t payload = iree_async_span_list_make(spans, span_count);
  return iree_net_session_send_control_data_copy(session, /*flags=*/0, payload,
                                                 /*operation_user_data=*/0);
}

// Sends a control channel response. The request envelope is used to echo the
// request_id and message_type.
static iree_status_t iree_hal_remote_server_send_response(
    iree_allocator_t host_allocator, iree_net_session_t* session,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_code_t status_code, const void* body,
    iree_host_size_t body_length) {
  (void)host_allocator;
  iree_host_size_t response_length = 0;
  iree_status_t status =
      iree_hal_remote_server_response_length(body_length, 0, &response_length);
  if (iree_status_is_ok(status)) {
    if (iree_hal_remote_server_response_fits_inline_copy(response_length)) {
      status = iree_hal_remote_server_send_response_copy(
          session, request_envelope, status_code, body, body_length, NULL, 0);
    } else {
      // Control responses must stay in the frame sender's inline storage.
      // Large payloads belong on the bulk channel; report a code-only error
      // instead of allocating retained response storage on the server hot path.
      status = iree_hal_remote_server_send_response_copy(
          session, request_envelope,
          status_code == IREE_STATUS_OK ? IREE_STATUS_RESOURCE_EXHAUSTED
                                        : status_code,
          NULL, 0, NULL, 0);
    }
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

  // Compute serialized size and serialize the full status.
  iree_host_size_t wire_size = 0;
  iree_net_status_wire_size(status, &wire_size);

  uint8_t* wire_buffer = NULL;
  iree_status_t serialize_status =
      iree_allocator_malloc(host_allocator, wire_size, (void**)&wire_buffer);
  if (iree_status_is_ok(serialize_status)) {
    serialize_status = iree_net_status_wire_serialize(
        status, iree_make_byte_span(wire_buffer, wire_size));
  }

  iree_status_t send_status = iree_ok_status();
  if (iree_status_is_ok(serialize_status)) {
    send_status = iree_hal_remote_server_send_response(host_allocator, session,
                                                       request_envelope, code,
                                                       wire_buffer, wire_size);
  } else {
    // Allocation or serialization failed; send code-only response.
    iree_status_ignore(serialize_status);
    send_status = iree_hal_remote_server_send_response(
        host_allocator, session, request_envelope, code, NULL, 0);
  }

  iree_allocator_free(host_allocator, wire_buffer);
  iree_status_ignore(status);
  return send_status;
}

iree_status_t iree_hal_remote_server_session_send_response(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_code_t status_code, const void* body,
    iree_host_size_t body_length) {
  if (!session_slot->session) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "remote session is no longer active");
  }
  return iree_hal_remote_server_send_response(
      session_slot->server->host_allocator, session_slot->session,
      request_envelope, status_code, body, body_length);
}

iree_status_t iree_hal_remote_server_session_send_error_response(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_t status) {
  if (!session_slot->session) {
    iree_status_ignore(status);
    return iree_make_status(IREE_STATUS_ABORTED,
                            "remote session is no longer active");
  }
  return iree_hal_remote_server_send_error_response(
      session_slot->server->host_allocator, session_slot->session,
      request_envelope, status);
}

// Handles EVENT_CREATE: creates an event on the local device and assigns a
// resource slot in the session's table.
static iree_status_t iree_hal_remote_server_handle_event_create(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  const iree_hal_remote_event_create_request_t* request = NULL;
  iree_hal_event_t* event = NULL;
  iree_hal_remote_resource_id_t resolved_id = 0;

  iree_status_t status = iree_ok_status();
  if (body_length < sizeof(iree_hal_remote_event_create_request_t)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "EVENT_CREATE body too small: %" PRIhsz " bytes",
                              body_length);
  }
  if (iree_status_is_ok(status)) {
    request = (const iree_hal_remote_event_create_request_t*)body;
    if (request->reserved != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "EVENT_CREATE reserved field is nonzero");
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_t* local_device = entry->server->devices[0];
    status = iree_hal_event_create(
        local_device, (iree_hal_queue_affinity_t)request->queue_affinity,
        (iree_hal_event_flags_t)request->flags, &event);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_resource_table_assign(
        &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_EVENT, event,
        &resolved_id);
  }

  iree_hal_event_release(event);

  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }
  iree_hal_remote_event_create_response_t response = {
      .resolved_id = resolved_id,
  };
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
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

static iree_status_t
iree_hal_remote_server_handle_buffer_virtual_query_capabilities(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length <
      sizeof(iree_hal_remote_buffer_virtual_query_capabilities_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "BUFFER_VIRTUAL_QUERY_CAPABILITIES body too small: %" PRIhsz
            " bytes",
            body_length));
  }

  const iree_hal_remote_buffer_virtual_query_capabilities_request_t* request =
      (const iree_hal_remote_buffer_virtual_query_capabilities_request_t*)body;
  if (request->flags != 0 || request->reserved != 0) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "BUFFER_VIRTUAL_QUERY_CAPABILITIES reserved fields must "
            "be 0"));
  }

  iree_hal_allocator_t* allocator =
      iree_hal_device_allocator(entry->server->devices[0]);
  iree_hal_remote_buffer_virtual_query_capabilities_response_t response;
  memset(&response, 0, sizeof(response));
  response.supports_virtual_memory =
      iree_hal_allocator_supports_virtual_memory(allocator) ? 1u : 0u;
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

static iree_status_t
iree_hal_remote_server_handle_buffer_virtual_query_granularity(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length <
      sizeof(iree_hal_remote_buffer_virtual_query_granularity_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "BUFFER_VIRTUAL_QUERY_GRANULARITY body too small: %" PRIhsz
            " bytes",
            body_length));
  }

  const iree_hal_remote_buffer_virtual_query_granularity_request_t* request =
      (const iree_hal_remote_buffer_virtual_query_granularity_request_t*)body;
  iree_hal_allocator_t* allocator =
      iree_hal_device_allocator(entry->server->devices[0]);
  iree_hal_buffer_params_t params =
      iree_hal_remote_server_wire_params_to_hal(request->params);

  iree_device_size_t minimum_page_size = 0;
  iree_device_size_t recommended_page_size = 0;
  iree_status_t status = iree_hal_allocator_virtual_memory_query_granularity(
      allocator, params, &minimum_page_size, &recommended_page_size);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  iree_hal_remote_buffer_virtual_query_granularity_response_t response = {
      .minimum_page_size = (uint64_t)minimum_page_size,
      .recommended_page_size = (uint64_t)recommended_page_size,
  };
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

static iree_status_t iree_hal_remote_server_handle_buffer_virtual_reserve(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_virtual_reserve_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_VIRTUAL_RESERVE body too small: %" PRIhsz
                         " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_virtual_reserve_request_t* request =
      (const iree_hal_remote_buffer_virtual_reserve_request_t*)body;
  iree_status_t status = iree_ok_status();
  if (request->size > IREE_DEVICE_SIZE_MAX) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "BUFFER_VIRTUAL_RESERVE size %" PRIu64
                              " exceeds device size max %" PRIu64,
                              request->size, (uint64_t)IREE_DEVICE_SIZE_MAX);
  }

  iree_hal_allocator_t* allocator =
      iree_hal_device_allocator(entry->server->devices[0]);
  iree_hal_buffer_t* virtual_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_virtual_memory_reserve(
        allocator, (iree_hal_queue_affinity_t)request->queue_affinity,
        (iree_device_size_t)request->size, &virtual_buffer);
  }
  if (iree_status_is_ok(status) && !virtual_buffer) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "BUFFER_VIRTUAL_RESERVE allocator returned success without a buffer");
  }

  iree_hal_remote_resource_id_t resolved_id = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_resource_table_assign(
        &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
        virtual_buffer, &resolved_id);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_track_virtual_buffer(
        entry, resolved_id, entry->server->host_allocator);
  }

  iree_hal_remote_buffer_virtual_reserve_response_t response;
  memset(&response, 0, sizeof(response));
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_placement_t placement =
        iree_hal_buffer_allocation_placement(virtual_buffer);
    response.resolved_id = resolved_id;
    response.params =
        iree_hal_remote_server_buffer_to_wire_params(virtual_buffer);
    response.allocation_size =
        (uint64_t)iree_hal_buffer_allocation_size(virtual_buffer);
    response.placement_flags = placement.flags;
  } else if (virtual_buffer) {
    if (resolved_id != 0) {
      iree_hal_buffer_t* table_buffer =
          (iree_hal_buffer_t*)iree_hal_remote_resource_table_detach(
              &entry->resource_table, resolved_id);
      iree_hal_buffer_release(table_buffer);
      iree_hal_remote_server_untrack_virtual_buffer(entry, resolved_id);
    }
    status = iree_status_join(status, iree_hal_allocator_virtual_memory_release(
                                          allocator, virtual_buffer));
    virtual_buffer = NULL;
  }

  iree_hal_buffer_release(virtual_buffer);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

static iree_status_t iree_hal_remote_server_handle_buffer_virtual_release(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_virtual_release_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_VIRTUAL_RELEASE body too small: %" PRIhsz
                         " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_virtual_release_request_t* request =
      (const iree_hal_remote_buffer_virtual_release_request_t*)body;
  iree_hal_remote_resource_id_t resolved_id =
      iree_hal_remote_server_resolve_resource_id(entry, request->buffer_id);
  iree_status_t status =
      iree_hal_remote_server_is_virtual_buffer(entry, resolved_id)
          ? iree_hal_remote_server_release_virtual_buffer(entry, resolved_id)
          : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "BUFFER_VIRTUAL_RELEASE buffer 0x%016" PRIx64
                             " is not a virtual reservation",
                             resolved_id);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }
  return iree_hal_remote_server_send_response(entry->server->host_allocator,
                                              entry->session, envelope,
                                              IREE_STATUS_OK, NULL, 0);
}

static iree_status_t iree_hal_remote_server_handle_buffer_physical_alloc(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_physical_alloc_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_PHYSICAL_ALLOC body too small: %" PRIhsz
                         " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_physical_alloc_request_t* request =
      (const iree_hal_remote_buffer_physical_alloc_request_t*)body;
  iree_status_t status = iree_ok_status();
  if (request->size > IREE_DEVICE_SIZE_MAX) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "BUFFER_PHYSICAL_ALLOC size %" PRIu64
                              " exceeds device size max %" PRIu64,
                              request->size, (uint64_t)IREE_DEVICE_SIZE_MAX);
  }

  iree_hal_allocator_t* allocator =
      iree_hal_device_allocator(entry->server->devices[0]);
  iree_hal_buffer_params_t params =
      iree_hal_remote_server_wire_params_to_hal(request->params);
  iree_hal_physical_memory_t* physical_memory = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_physical_memory_allocate(
        allocator, params, (iree_device_size_t)request->size,
        entry->server->host_allocator, &physical_memory);
  }
  if (iree_status_is_ok(status) && !physical_memory) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "BUFFER_PHYSICAL_ALLOC allocator returned success without physical "
        "memory");
  }

  iree_hal_remote_server_physical_memory_t* wrapper = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_physical_memory_create(
        allocator, physical_memory, entry->server->host_allocator, &wrapper);
    if (iree_status_is_ok(status)) physical_memory = NULL;
  }

  iree_hal_remote_resource_id_t resolved_id = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_resource_table_assign(
        &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_PHYSICAL_MEMORY,
        wrapper, &resolved_id);
  }

  iree_hal_resource_release(wrapper);
  if (physical_memory) {
    status = iree_status_join(status, iree_hal_allocator_physical_memory_free(
                                          allocator, physical_memory));
  }
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  iree_hal_remote_buffer_physical_alloc_response_t response = {
      .resolved_id = resolved_id,
  };
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

static iree_status_t iree_hal_remote_server_handle_buffer_physical_free(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_physical_free_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_PHYSICAL_FREE body too small: %" PRIhsz
                         " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_physical_free_request_t* request =
      (const iree_hal_remote_buffer_physical_free_request_t*)body;
  iree_hal_remote_resource_id_t resolved_id =
      iree_hal_remote_server_resolve_resource_id(entry,
                                                 request->physical_memory_id);
  iree_hal_remote_server_physical_memory_t* physical_memory =
      (iree_hal_remote_server_physical_memory_t*)
          iree_hal_remote_resource_table_lookup(
              &entry->resource_table,
              IREE_HAL_REMOTE_RESOURCE_TYPE_PHYSICAL_MEMORY, resolved_id);
  iree_status_t status = iree_ok_status();
  if (!physical_memory) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "BUFFER_PHYSICAL_FREE physical memory 0x%016" PRIx64 " not found",
        resolved_id);
  } else {
    status = iree_hal_remote_server_physical_memory_free(physical_memory);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_t* detached =
        (iree_hal_resource_t*)iree_hal_remote_resource_table_detach(
            &entry->resource_table, resolved_id);
    iree_hal_resource_release(detached);
  }
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }
  return iree_hal_remote_server_send_response(entry->server->host_allocator,
                                              entry->session, envelope,
                                              IREE_STATUS_OK, NULL, 0);
}

static iree_status_t iree_hal_remote_server_handle_buffer_virtual_map(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_virtual_map_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_VIRTUAL_MAP body too small: %" PRIhsz " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_virtual_map_request_t* request =
      (const iree_hal_remote_buffer_virtual_map_request_t*)body;
  iree_hal_remote_resource_id_t buffer_id =
      iree_hal_remote_server_resolve_resource_id(entry, request->buffer_id);
  iree_hal_remote_resource_id_t physical_memory_id =
      iree_hal_remote_server_resolve_resource_id(entry,
                                                 request->physical_memory_id);
  iree_hal_buffer_t* virtual_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
          buffer_id);
  iree_hal_remote_server_physical_memory_t* physical_memory =
      (iree_hal_remote_server_physical_memory_t*)
          iree_hal_remote_resource_table_lookup(
              &entry->resource_table,
              IREE_HAL_REMOTE_RESOURCE_TYPE_PHYSICAL_MEMORY,
              physical_memory_id);

  iree_status_t status = iree_ok_status();
  if (!virtual_buffer) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "BUFFER_VIRTUAL_MAP virtual buffer 0x%016" PRIx64
                              " not found",
                              buffer_id);
  } else if (!iree_hal_remote_server_is_virtual_buffer(entry, buffer_id)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "BUFFER_VIRTUAL_MAP buffer 0x%016" PRIx64
                              " is not a virtual reservation",
                              buffer_id);
  } else if (!physical_memory) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "BUFFER_VIRTUAL_MAP physical memory 0x%016" PRIx64
                              " not found",
                              physical_memory_id);
  } else if (request->virtual_offset > IREE_DEVICE_SIZE_MAX ||
             request->physical_offset > IREE_DEVICE_SIZE_MAX ||
             request->size > IREE_DEVICE_SIZE_MAX) {
    status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "BUFFER_VIRTUAL_MAP range exceeds device size max");
  } else {
    iree_hal_allocator_t* allocator =
        iree_hal_device_allocator(entry->server->devices[0]);
    status = iree_hal_allocator_virtual_memory_map(
        allocator, virtual_buffer, (iree_device_size_t)request->virtual_offset,
        physical_memory->physical_memory,
        (iree_device_size_t)request->physical_offset,
        (iree_device_size_t)request->size);
  }

  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }
  return iree_hal_remote_server_send_response(entry->server->host_allocator,
                                              entry->session, envelope,
                                              IREE_STATUS_OK, NULL, 0);
}

static iree_status_t iree_hal_remote_server_handle_buffer_virtual_unmap(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_virtual_unmap_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_VIRTUAL_UNMAP body too small: %" PRIhsz
                         " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_virtual_unmap_request_t* request =
      (const iree_hal_remote_buffer_virtual_unmap_request_t*)body;
  iree_hal_remote_resource_id_t buffer_id =
      iree_hal_remote_server_resolve_resource_id(entry, request->buffer_id);
  iree_hal_buffer_t* virtual_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
          buffer_id);

  iree_status_t status = iree_ok_status();
  if (!virtual_buffer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "BUFFER_VIRTUAL_UNMAP virtual buffer 0x%016" PRIx64 " not found",
        buffer_id);
  } else if (!iree_hal_remote_server_is_virtual_buffer(entry, buffer_id)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "BUFFER_VIRTUAL_UNMAP buffer 0x%016" PRIx64
                              " is not a virtual reservation",
                              buffer_id);
  } else if (request->virtual_offset > IREE_DEVICE_SIZE_MAX ||
             request->size > IREE_DEVICE_SIZE_MAX) {
    status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "BUFFER_VIRTUAL_UNMAP range exceeds device size max");
  } else {
    iree_hal_allocator_t* allocator =
        iree_hal_device_allocator(entry->server->devices[0]);
    status = iree_hal_allocator_virtual_memory_unmap(
        allocator, virtual_buffer, (iree_device_size_t)request->virtual_offset,
        (iree_device_size_t)request->size);
  }

  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }
  return iree_hal_remote_server_send_response(entry->server->host_allocator,
                                              entry->session, envelope,
                                              IREE_STATUS_OK, NULL, 0);
}

static iree_status_t iree_hal_remote_server_handle_buffer_virtual_protect(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_virtual_protect_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_VIRTUAL_PROTECT body too small: %" PRIhsz
                         " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_virtual_protect_request_t* request =
      (const iree_hal_remote_buffer_virtual_protect_request_t*)body;
  iree_hal_remote_resource_id_t buffer_id =
      iree_hal_remote_server_resolve_resource_id(entry, request->buffer_id);
  iree_hal_buffer_t* virtual_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
          buffer_id);

  iree_status_t status = iree_ok_status();
  if (!virtual_buffer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "BUFFER_VIRTUAL_PROTECT virtual buffer 0x%016" PRIx64 " not found",
        buffer_id);
  } else if (!iree_hal_remote_server_is_virtual_buffer(entry, buffer_id)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "BUFFER_VIRTUAL_PROTECT buffer 0x%016" PRIx64
                              " is not a virtual reservation",
                              buffer_id);
  } else if (request->virtual_offset > IREE_DEVICE_SIZE_MAX ||
             request->size > IREE_DEVICE_SIZE_MAX) {
    status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "BUFFER_VIRTUAL_PROTECT range exceeds device size "
                         "max");
  } else {
    iree_hal_allocator_t* allocator =
        iree_hal_device_allocator(entry->server->devices[0]);
    status = iree_hal_allocator_virtual_memory_protect(
        allocator, virtual_buffer, (iree_device_size_t)request->virtual_offset,
        (iree_device_size_t)request->size,
        (iree_hal_queue_affinity_t)request->queue_affinity,
        (iree_hal_memory_protection_t)request->protection);
  }

  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }
  return iree_hal_remote_server_send_response(entry->server->host_allocator,
                                              entry->session, envelope,
                                              IREE_STATUS_OK, NULL, 0);
}

static iree_status_t iree_hal_remote_server_handle_buffer_virtual_advise(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length < sizeof(iree_hal_remote_buffer_virtual_advise_request_t)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_VIRTUAL_ADVISE body too small: %" PRIhsz
                         " bytes",
                         body_length));
  }

  const iree_hal_remote_buffer_virtual_advise_request_t* request =
      (const iree_hal_remote_buffer_virtual_advise_request_t*)body;
  iree_hal_remote_resource_id_t buffer_id =
      iree_hal_remote_server_resolve_resource_id(entry, request->buffer_id);
  iree_hal_buffer_t* virtual_buffer =
      (iree_hal_buffer_t*)iree_hal_remote_resource_table_lookup(
          &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
          buffer_id);

  iree_status_t status = iree_ok_status();
  if (!virtual_buffer) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "BUFFER_VIRTUAL_ADVISE virtual buffer 0x%016" PRIx64 " not found",
        buffer_id);
  } else if (!iree_hal_remote_server_is_virtual_buffer(entry, buffer_id)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "BUFFER_VIRTUAL_ADVISE buffer 0x%016" PRIx64
                              " is not a virtual reservation",
                              buffer_id);
  } else if (request->virtual_offset > IREE_DEVICE_SIZE_MAX ||
             request->size > IREE_DEVICE_SIZE_MAX) {
    status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "BUFFER_VIRTUAL_ADVISE range exceeds device size max");
  } else {
    iree_hal_allocator_t* allocator =
        iree_hal_device_allocator(entry->server->devices[0]);
    status = iree_hal_allocator_virtual_memory_advise(
        allocator, virtual_buffer, (iree_device_size_t)request->virtual_offset,
        (iree_device_size_t)request->size,
        (iree_hal_queue_affinity_t)request->queue_affinity,
        (iree_hal_memory_advice_t)request->advice);
  }

  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }
  return iree_hal_remote_server_send_response(entry->server->host_allocator,
                                              entry->session, envelope,
                                              IREE_STATUS_OK, NULL, 0);
}

// Sends a control channel response with a variable-length data payload.
static iree_status_t iree_hal_remote_server_send_response_with_data(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* request_envelope,
    iree_status_code_t status_code, const void* body_header,
    iree_host_size_t body_header_length, const void* data,
    iree_host_size_t data_length) {
  iree_host_size_t response_length = 0;
  iree_status_t status = iree_hal_remote_server_response_length(
      body_header_length, data_length, &response_length);
  if (iree_status_is_ok(status)) {
    if (iree_hal_remote_server_response_fits_inline_copy(response_length)) {
      status = iree_hal_remote_server_send_response_copy(
          entry->session, request_envelope, status_code, body_header,
          body_header_length, data, data_length);
    } else {
      // Control responses must stay in the frame sender's inline storage.
      // Large payloads belong on the bulk channel; report a code-only error
      // instead of allocating retained response storage on the server hot path.
      status = iree_hal_remote_server_send_response_copy(
          entry->session, request_envelope,
          status_code == IREE_STATUS_OK ? IREE_STATUS_RESOURCE_EXHAUSTED
                                        : status_code,
          NULL, 0, NULL, 0);
    }
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

// Imports writes from completed local queue operations before host access.
// Command completions publish under |session_mutex| before sending ADVANCE.
// A client may issue BUFFER_MAP after observing that ADVANCE, but the network
// round trip alone is not a C memory synchronization edge when the client and
// server share a process.
static void iree_hal_remote_server_acquire_queue_completions(
    iree_hal_remote_server_session_t* entry) {
  iree_slim_mutex_lock(&entry->server->session_mutex);
  iree_slim_mutex_unlock(&entry->server->session_mutex);
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
  if (request->flags & ~IREE_HAL_REMOTE_BUFFER_MAP_FLAG_BULK_TRANSFER) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_MAP unsupported flags: 0x%08x",
                         request->flags));
  }
  iree_hal_remote_resource_id_t buffer_id =
      iree_hal_remote_server_resolve_resource_id(entry, request->buffer_id);

  iree_hal_remote_server_acquire_queue_completions(entry);

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
    const bool use_bulk = iree_all_bits_set(
        request->flags, IREE_HAL_REMOTE_BUFFER_MAP_FLAG_BULK_TRANSFER);
    if (use_bulk && request->transfer_id == 0) {
      return iree_hal_remote_server_send_error_response(
          entry->server->host_allocator, entry->session, envelope,
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "BUFFER_MAP bulk transfer requires transfer_id"));
    } else if (!use_bulk && request->transfer_id != 0) {
      return iree_hal_remote_server_send_error_response(
          entry->server->host_allocator, entry->session, envelope,
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "BUFFER_MAP inline transfer provided transfer_id"));
    }

    if (use_bulk) {
      iree_hal_device_t* local_device = entry->server->devices[0];
      iree_status_t status =
          iree_hal_remote_server_bulk_submit_client_file_write(
              entry, local_device, iree_hal_semaphore_list_empty(),
              iree_hal_semaphore_list_empty(), request->transfer_id, buffer,
              offset, length, IREE_HAL_WRITE_FLAG_NONE);
      if (!iree_status_is_ok(status)) {
        return iree_hal_remote_server_send_error_response(
            entry->server->host_allocator, entry->session, envelope, status);
      }
      iree_hal_remote_buffer_map_response_t response = {
          .mapped_offset = offset,
          .mapped_length = length,
          .transfer_id = request->transfer_id,
      };
      return iree_hal_remote_server_send_response(
          entry->server->host_allocator, entry->session, envelope,
          IREE_STATUS_OK, &response, sizeof(response));
    }

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
    } else {
      iree_host_size_t response_length = 0;
      status = iree_hal_remote_server_response_length(
          sizeof(iree_hal_remote_buffer_map_response_t), response_data_length,
          &response_length);
      if (iree_status_is_ok(status) &&
          !iree_hal_remote_server_response_fits_inline_copy(response_length)) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "BUFFER_MAP inline response requires bulk transfer for %" PRIhsz
            " bytes",
            response_data_length);
      }
    }
    if (iree_status_is_ok(status)) {
      if (length == 0) {
        // Nothing to map.
      } else if (iree_hal_remote_server_buffer_supports_scoped_mapping(
                     buffer)) {
        status = iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, offset,
                                           length, &mapping);
        if (iree_status_is_ok(status)) {
          did_map = true;
          response_data = mapping.contents.data;
          response_data_length = mapping.contents.data_length;
        }
      } else {
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "BUFFER_MAP of non-host-visible buffer 0x%016" PRIx64
            " requires async remote staging",
            buffer_id);
      }
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
        .transfer_id = 0,
    };
    iree_status_t send_status = iree_hal_remote_server_send_response_with_data(
        entry, envelope, IREE_STATUS_OK, &response, sizeof(response),
        response_data, response_data_length);

    if (did_map) {
      iree_status_ignore(iree_hal_buffer_unmap_range(&mapping));
    }
    return send_status;
  }

  if (request->flags != 0 || request->transfer_id != 0) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_MAP bulk transfer requires READ access"));
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
      .transfer_id = 0,
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
  if (request->flags & ~IREE_HAL_REMOTE_BUFFER_UNMAP_FLAG_BULK_TRANSFER) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_UNMAP unsupported flags: 0x%08x",
                         request->flags));
  }
  if (request->reserved != 0) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_UNMAP reserved field must be 0"));
  }
  iree_hal_remote_resource_id_t buffer_id =
      iree_hal_remote_server_resolve_resource_id(entry, request->buffer_id);
  iree_device_size_t offset = (iree_device_size_t)request->offset;
  iree_device_size_t length = (iree_device_size_t)request->length;
  const bool use_bulk = iree_all_bits_set(
      request->flags, IREE_HAL_REMOTE_BUFFER_UNMAP_FLAG_BULK_TRANSFER);

  const uint8_t* data = body + sizeof(iree_hal_remote_buffer_unmap_request_t);
  iree_host_size_t data_length =
      body_length - sizeof(iree_hal_remote_buffer_unmap_request_t);
  if (use_bulk && request->transfer_id == 0) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_UNMAP bulk transfer requires transfer_id"));
  } else if (!use_bulk && request->transfer_id != 0) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_UNMAP inline transfer provided transfer_id"));
  } else if (use_bulk && data_length != 0) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "BUFFER_UNMAP bulk request carried %" PRIhsz
                         " unexpected inline bytes",
                         data_length));
  } else if (!use_bulk && length <= IREE_HOST_SIZE_MAX &&
             data_length < (iree_host_size_t)length) {
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
  } else if (use_bulk && length == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "BUFFER_UNMAP bulk transfer length must be > 0");
  } else if (use_bulk) {
    iree_hal_device_t* local_device = entry->server->devices[0];
    status = iree_hal_remote_server_bulk_submit_buffer_unmap(
        entry, local_device, envelope, request->transfer_id, buffer, offset,
        length);
    if (!iree_status_is_ok(status)) {
      return iree_hal_remote_server_send_error_response(
          entry->server->host_allocator, entry->session, envelope, status);
    }
    return iree_ok_status();
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

  if (IREE_HAL_REMOTE_RESOURCE_ID_TYPE(request->provisional_id) !=
          IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE ||
      !IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(request->provisional_id)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "EXECUTABLE_UPLOAD provisional id must have "
                         "provisional EXECUTABLE type"));
  }

  // Bulk executable uploads are not implemented yet.
  if (request->upload_flags != IREE_HAL_REMOTE_UPLOAD_FLAG_INLINE_DATA) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "EXECUTABLE_UPLOAD: only INLINE_DATA is supported"));
  }
  if (request->bulk_transfer_id != 0) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "EXECUTABLE_UPLOAD inline request has a bulk transfer ID"));
  }

  const iree_hal_executable_load_flags_t allowed_load_flags =
      IREE_HAL_EXECUTABLE_LOAD_FLAG_ALLOW_OPTIMIZATION |
      IREE_HAL_EXECUTABLE_LOAD_FLAG_ENABLE_DEBUGGING |
      IREE_HAL_EXECUTABLE_LOAD_FLAG_DISABLE_VERIFICATION;
  if (request->load_flags & ~allowed_load_flags) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "EXECUTABLE_UPLOAD load flags contain unknown bits "
                         "0x%08" PRIx32,
                         request->load_flags & ~allowed_load_flags));
  }

  if (request->target_ordinal > IREE_HOST_SIZE_MAX) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "EXECUTABLE_UPLOAD target ordinal exceeds host "
                         "capacity"));
  }
  if (request->data_length > IREE_HOST_SIZE_MAX) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "EXECUTABLE_UPLOAD data_length exceeds host capacity"));
  }

  iree_hal_remote_server_t* server = entry->server;
  iree_hal_device_t* device = server->devices[0];
  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(iree_hal_device_spec(device));
  const iree_host_size_t target_ordinal =
      (iree_host_size_t)request->target_ordinal;
  if (target_ordinal >= executable_spec->target_count) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "EXECUTABLE_UPLOAD target ordinal %" PRIhsz
                         " exceeds target count %" PRIhsz,
                         target_ordinal, executable_spec->target_count));
  }

  // Extract specialization constants and inline artifact data.
  iree_host_size_t constants_offset = 0;
  iree_host_size_t data_offset = 0;
  iree_host_size_t total_required = 0;
  iree_status_t layout_status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_executable_upload_request_t), &total_required,
      IREE_STRUCT_FIELD_ALIGNED(request->constant_count, uint32_t, 8,
                                &constants_offset),
      IREE_STRUCT_FIELD_ALIGNED((iree_host_size_t)request->data_length, uint8_t,
                                8, &data_offset));
  if (!iree_status_is_ok(layout_status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, layout_status);
  }
  if (body_length < total_required) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "EXECUTABLE_UPLOAD inline data truncated"));
  }
  const uint32_t* constants = (const uint32_t*)(body + constants_offset);
  const uint8_t* inline_data = body + data_offset;

  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.flags = (iree_hal_executable_load_flags_t)request->load_flags;
  load_params.executable_data = iree_make_const_byte_span(
      inline_data, (iree_host_size_t)request->data_length);
  load_params.constant_count = (iree_host_size_t)request->constant_count;
  load_params.constants = constants;

  iree_hal_executable_t* executable = NULL;
  iree_status_t status = iree_hal_device_load_executable(
      device, (iree_hal_queue_affinity_t)request->queue_affinity,
      &executable_spec->targets[target_ordinal], &load_params, &executable);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  // Query function count from the loaded executable.
  iree_host_size_t function_count =
      iree_hal_executable_function_count(executable);
  if (function_count > UINT32_MAX) {
    iree_hal_executable_release(executable);
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope,
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "EXECUTABLE_UPLOAD function count %" PRIhsz
                         " exceeds wire limit %" PRIu32,
                         function_count, UINT32_MAX));
  }

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
  response.function_count = (uint32_t)function_count;
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

static iree_status_t iree_hal_remote_server_lookup_executable(
    iree_hal_remote_server_session_t* entry,
    iree_hal_remote_resource_id_t executable_id,
    iree_hal_executable_t** out_executable) {
  iree_hal_remote_resource_id_t resolved_id =
      iree_hal_remote_server_resolve_resource_id(entry, executable_id);
  iree_hal_executable_t* executable =
      (iree_hal_executable_t*)iree_hal_remote_resource_table_lookup(
          &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE,
          resolved_id);
  if (!executable) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "executable 0x%016" PRIx64 " not found",
                            resolved_id);
  }
  *out_executable = executable;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_handle_executable_query_function(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  iree_status_t status = iree_ok_status();
  if (body_length <
      sizeof(iree_hal_remote_executable_query_function_request_t)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "EXECUTABLE_QUERY_FUNCTION body too small: %" PRIhsz " bytes",
        body_length);
  }

  const iree_hal_remote_executable_query_function_request_t* request = NULL;
  iree_hal_executable_t* executable = NULL;
  iree_hal_executable_function_info_t info;
  memset(&info, 0, sizeof(info));
  if (iree_status_is_ok(status)) {
    request = (const iree_hal_remote_executable_query_function_request_t*)body;
    status = iree_hal_remote_server_lookup_executable(
        entry, request->executable_id, &executable);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_function_info(
        executable,
        iree_hal_executable_function_from_value(request->function_value),
        &info);
  }
  if (iree_status_is_ok(status) && info.name.size > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "function name length %" PRIhsz
                              " exceeds wire limit %u",
                              info.name.size, (unsigned)UINT16_MAX);
  }
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  iree_hal_remote_executable_query_function_response_t response;
  memset(&response, 0, sizeof(response));
  response.flags = (uint64_t)info.flags;
  memcpy(response.workgroup_size, info.workgroup_size,
         sizeof(response.workgroup_size));
  response.occupancy_reserved = info.occupancy_info.reserved;
  response.constant_byte_length = info.constant_byte_length;
  response.binding_count = info.binding_count;
  response.parameter_count = info.parameter_count;
  response.name_length = (uint16_t)info.name.size;
  return iree_hal_remote_server_send_response_with_data(
      entry, envelope, IREE_STATUS_OK, &response, sizeof(response),
      info.name.data, info.name.size);
}

static iree_status_t iree_hal_remote_server_handle_executable_query_parameters(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  iree_status_t status = iree_ok_status();
  if (body_length <
      sizeof(iree_hal_remote_executable_query_parameters_request_t)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "EXECUTABLE_QUERY_PARAMETERS body too small: %" PRIhsz " bytes",
        body_length);
  }

  const iree_hal_remote_executable_query_parameters_request_t* request = NULL;
  iree_hal_executable_t* executable = NULL;
  iree_hal_executable_function_info_t info;
  memset(&info, 0, sizeof(info));
  if (iree_status_is_ok(status)) {
    request =
        (const iree_hal_remote_executable_query_parameters_request_t*)body;
    if (request->reserved[0] != 0 || request->reserved[1] != 0 ||
        request->reserved[2] != 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "EXECUTABLE_QUERY_PARAMETERS request reserved field is nonzero");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_lookup_executable(
        entry, request->executable_id, &executable);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_function_info(
        executable,
        iree_hal_executable_function_from_value(request->function_value),
        &info);
  }
  iree_host_size_t parameter_count = 0;
  if (iree_status_is_ok(status)) {
    parameter_count = iree_min((iree_host_size_t)request->capacity,
                               (iree_host_size_t)info.parameter_count);
  }

  iree_hal_executable_function_parameter_t stack_parameters[64];
  iree_hal_executable_function_parameter_t* parameters = stack_parameters;
  bool parameters_heap_allocated = false;
  if (iree_status_is_ok(status) &&
      parameter_count > IREE_ARRAYSIZE(stack_parameters)) {
    status = iree_allocator_malloc_array(
        entry->server->host_allocator, parameter_count,
        sizeof(iree_hal_executable_function_parameter_t), (void**)&parameters);
    parameters_heap_allocated = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status) && parameter_count > 0) {
    status = iree_hal_executable_function_parameters(
        executable,
        iree_hal_executable_function_from_value(request->function_value),
        parameter_count, parameters);
  }

  iree_host_size_t name_data_length = 0;
  const iree_hal_executable_function_parameter_flags_t known_parameter_flags =
      IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET;
  for (iree_host_size_t i = 0; i < parameter_count && iree_status_is_ok(status);
       ++i) {
    if (parameters[i].type >
        IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter has invalid type %" PRIu8,
                                parameters[i].type);
    } else if (iree_any_bit_set(parameters[i].flags, ~known_parameter_flags)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter has unsupported flags 0x%04" PRIx16,
                                parameters[i].flags);
    } else if (parameters[i].name.size > UINT16_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter name length %" PRIhsz
                                " exceeds wire limit %u",
                                parameters[i].name.size, (unsigned)UINT16_MAX);
    } else if (!iree_host_size_checked_add(name_data_length,
                                           parameters[i].name.size,
                                           &name_data_length)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter name storage overflow");
    } else if (name_data_length > UINT32_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter name data length %" PRIhsz
                                " exceeds wire limit %" PRIu32,
                                name_data_length, UINT32_MAX);
    }
  }

  iree_host_size_t parameters_offset = 0;
  iree_host_size_t names_offset = 0;
  iree_host_size_t data_length = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        0, &data_length,
        IREE_STRUCT_FIELD(parameter_count,
                          iree_hal_remote_executable_function_parameter_t,
                          &parameters_offset),
        IREE_STRUCT_FIELD(name_data_length, char, &names_offset));
  }

  iree_hal_remote_executable_query_parameters_response_t response;
  memset(&response, 0, sizeof(response));
  uint64_t data_storage[(IREE_NET_FRAME_SENDER_INLINE_FRAME_CAPACITY +
                         sizeof(uint64_t) - 1) /
                        sizeof(uint64_t)];
  if (iree_status_is_ok(status)) {
    iree_host_size_t response_length = 0;
    status = iree_hal_remote_server_response_length(
        sizeof(response), data_length, &response_length);
    if (iree_status_is_ok(status) &&
        !iree_hal_remote_server_response_fits_inline_copy(response_length)) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "EXECUTABLE_QUERY_PARAMETERS response length %" PRIhsz
          " exceeds inline control response capacity",
          response_length);
    }
  }
  if (iree_status_is_ok(status)) {
    memset(data_storage, 0, data_length);
    uint8_t* data_bytes = (uint8_t*)data_storage;
    iree_hal_remote_executable_function_parameter_t* wire_parameters =
        (iree_hal_remote_executable_function_parameter_t*)(data_bytes +
                                                           parameters_offset);
    char* name_cursor = (char*)data_bytes + names_offset;
    for (iree_host_size_t i = 0; i < parameter_count; ++i) {
      wire_parameters[i].offset = parameters[i].offset;
      wire_parameters[i].native_abi_offset = parameters[i].native_abi_offset;
      wire_parameters[i].flags = parameters[i].flags;
      wire_parameters[i].name_length = (uint16_t)parameters[i].name.size;
      wire_parameters[i].type = parameters[i].type;
      wire_parameters[i].size = parameters[i].size;
      if (parameters[i].name.size > 0) {
        memcpy(name_cursor, parameters[i].name.data, parameters[i].name.size);
        name_cursor += parameters[i].name.size;
      }
    }
    response.parameter_count = (uint16_t)parameter_count;
    response.name_data_length = (uint32_t)name_data_length;
  }

  if (parameters_heap_allocated) {
    iree_allocator_free(entry->server->host_allocator, parameters);
  }
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  return iree_hal_remote_server_send_response_with_data(
      entry, envelope, IREE_STATUS_OK, &response, sizeof(response),
      data_storage, data_length);
}

static iree_status_t iree_hal_remote_server_handle_executable_lookup_global(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  iree_status_t status = iree_ok_status();
  if (body_length <
      sizeof(iree_hal_remote_executable_lookup_global_request_t)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "EXECUTABLE_LOOKUP_GLOBAL body too small: %" PRIhsz " bytes",
        body_length);
  }

  const iree_hal_remote_executable_lookup_global_request_t* request = NULL;
  iree_host_size_t name_offset = 0;
  iree_host_size_t required_length = 0;
  if (iree_status_is_ok(status)) {
    request = (const iree_hal_remote_executable_lookup_global_request_t*)body;
    if (request->reserved0 != 0 || request->reserved1 != 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "EXECUTABLE_LOOKUP_GLOBAL request reserved field is nonzero");
    }
  }
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_executable_lookup_global_request_t),
        &required_length,
        IREE_STRUCT_FIELD(request->name_length, char, &name_offset));
  }
  if (iree_status_is_ok(status) && body_length < required_length) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "EXECUTABLE_LOOKUP_GLOBAL name truncated");
  }

  iree_hal_executable_t* executable = NULL;
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  iree_hal_executable_global_info_t global_info;
  memset(&global_info, 0, sizeof(global_info));
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_lookup_executable(
        entry, request->executable_id, &executable);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t name =
        iree_make_string_view((const char*)body + name_offset,
                              (iree_host_size_t)request->name_length);
    status = iree_hal_executable_try_lookup_global_by_name(executable, name,
                                                           &found, &global);
  }
  if (iree_status_is_ok(status) && found) {
    status = iree_hal_executable_global_info(executable, global, &global_info);
  }
  if (iree_status_is_ok(status) && found &&
      !iree_hal_executable_global_is_valid(global)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "EXECUTABLE_LOOKUP_GLOBAL returned an invalid global token");
  }

  iree_hal_remote_executable_lookup_global_response_t response;
  memset(&response, 0, sizeof(response));
  if (iree_status_is_ok(status) && found) {
    response.byte_length = (uint64_t)global_info.byte_length;
  }
  if (iree_status_is_ok(status)) {
    response.found = found ? 1u : 0u;
  }

  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }

  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

static iree_status_t iree_hal_remote_server_handle_executable_global_buffer(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  iree_status_t status = iree_ok_status();
  if (body_length <
      sizeof(iree_hal_remote_executable_global_buffer_request_t)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "EXECUTABLE_GLOBAL_BUFFER body too small: %" PRIhsz " bytes",
        body_length);
  }

  const iree_hal_remote_executable_global_buffer_request_t* request = NULL;
  iree_host_size_t name_offset = 0;
  iree_host_size_t required_length = 0;
  if (iree_status_is_ok(status)) {
    request = (const iree_hal_remote_executable_global_buffer_request_t*)body;
    if (request->reserved0 != 0 || request->reserved1 != 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "EXECUTABLE_GLOBAL_BUFFER request reserved field is nonzero");
    }
  }
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_executable_global_buffer_request_t),
        &required_length,
        IREE_STRUCT_FIELD(request->name_length, char, &name_offset));
  }
  if (iree_status_is_ok(status) && body_length < required_length) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "EXECUTABLE_GLOBAL_BUFFER name truncated");
  }

  iree_hal_executable_t* executable = NULL;
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  iree_hal_buffer_t* local_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_lookup_executable(
        entry, request->executable_id, &executable);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t name =
        iree_make_string_view((const char*)body + name_offset,
                              (iree_host_size_t)request->name_length);
    status = iree_hal_executable_try_lookup_global_by_name(executable, name,
                                                           &found, &global);
  }
  if (iree_status_is_ok(status) && !found) {
    status =
        iree_make_status(IREE_STATUS_NOT_FOUND, "executable global not found");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_global_buffer(
        executable, global, (iree_hal_queue_affinity_t)request->queue_affinity,
        &local_buffer);
  }
  if (iree_status_is_ok(status) && !local_buffer) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "EXECUTABLE_GLOBAL_BUFFER returned success without a buffer");
  }

  iree_hal_remote_resource_id_t resolved_id = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_resource_table_assign(
        &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER,
        local_buffer, &resolved_id);
  }

  iree_hal_remote_executable_global_buffer_response_t response;
  memset(&response, 0, sizeof(response));
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_placement_t placement =
        iree_hal_buffer_allocation_placement(local_buffer);
    response.resolved_id = resolved_id;
    response.params.usage = iree_hal_buffer_allowed_usage(local_buffer);
    response.params.access =
        (uint16_t)iree_hal_buffer_allowed_access(local_buffer);
    response.params.type = iree_hal_buffer_memory_type(local_buffer);
    response.params.queue_affinity = (uint64_t)placement.queue_affinity;
    response.params.min_alignment = 0;
    response.byte_length = (uint64_t)iree_hal_buffer_byte_length(local_buffer);
    response.placement_flags = placement.flags;
  }

  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
  }
  return iree_hal_remote_server_send_response(
      entry->server->host_allocator, entry->session, envelope, IREE_STATUS_OK,
      &response, sizeof(response));
}

//===----------------------------------------------------------------------===//
// Command stream replay
//===----------------------------------------------------------------------===//

typedef struct iree_hal_remote_server_barrier_list_t {
  // Memory barriers passed to the local HAL command buffer.
  iree_hal_memory_barrier_t* memory_barriers;

  // Buffer barriers passed to the local HAL command buffer.
  iree_hal_buffer_barrier_t* buffer_barriers;

  // Inline memory barrier storage for the common small-count case.
  iree_hal_memory_barrier_t
      inline_memory_barriers[IREE_HAL_REMOTE_INLINE_BARRIER_COUNT];

  // Inline buffer barrier storage for the common small-count case.
  iree_hal_buffer_barrier_t
      inline_buffer_barriers[IREE_HAL_REMOTE_INLINE_BARRIER_COUNT];
} iree_hal_remote_server_barrier_list_t;

typedef struct iree_hal_remote_server_event_list_t {
  // Events passed to the local HAL command buffer.
  const iree_hal_event_t** events;

  // Inline event storage for the common small-count case.
  const iree_hal_event_t* inline_events[IREE_HAL_REMOTE_INLINE_EVENT_COUNT];
} iree_hal_remote_server_event_list_t;

static void iree_hal_remote_server_barrier_list_initialize(
    iree_hal_remote_server_barrier_list_t* list) {
  memset(list, 0, sizeof(*list));
}

static void iree_hal_remote_server_barrier_list_deinitialize(
    iree_hal_remote_server_barrier_list_t* list,
    iree_allocator_t host_allocator) {
  if (list->memory_barriers &&
      list->memory_barriers != list->inline_memory_barriers) {
    iree_allocator_free(host_allocator, list->memory_barriers);
  }
  if (list->buffer_barriers &&
      list->buffer_barriers != list->inline_buffer_barriers) {
    iree_allocator_free(host_allocator, list->buffer_barriers);
  }
}

static void iree_hal_remote_server_event_list_initialize(
    iree_hal_remote_server_event_list_t* list) {
  memset(list, 0, sizeof(*list));
}

static void iree_hal_remote_server_event_list_deinitialize(
    iree_hal_remote_server_event_list_t* list,
    iree_allocator_t host_allocator) {
  if (list->events && list->events != list->inline_events) {
    iree_allocator_free(host_allocator, list->events);
  }
}

static iree_status_t iree_hal_remote_server_prepare_barrier_list(
    iree_hal_remote_server_session_t* session_slot,
    iree_host_size_t memory_barrier_count,
    const iree_hal_remote_memory_barrier_t* wire_memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_remote_buffer_barrier_t* wire_buffer_barriers,
    const char* command_name, iree_hal_remote_server_barrier_list_t* list) {
  iree_allocator_t host_allocator = session_slot->server->host_allocator;
  iree_status_t status = iree_ok_status();

  if (memory_barrier_count > 0) {
    if (memory_barrier_count <= IREE_HAL_REMOTE_INLINE_BARRIER_COUNT) {
      list->memory_barriers = list->inline_memory_barriers;
    } else {
      status = iree_allocator_malloc_array(host_allocator, memory_barrier_count,
                                           sizeof(*list->memory_barriers),
                                           (void**)&list->memory_barriers);
    }
  }
  for (iree_host_size_t i = 0;
       i < memory_barrier_count && iree_status_is_ok(status); ++i) {
    list->memory_barriers[i].source_scope =
        (iree_hal_access_scope_t)wire_memory_barriers[i].source_scope;
    list->memory_barriers[i].target_scope =
        (iree_hal_access_scope_t)wire_memory_barriers[i].target_scope;
  }

  if (iree_status_is_ok(status) && buffer_barrier_count > 0) {
    if (buffer_barrier_count <= IREE_HAL_REMOTE_INLINE_BARRIER_COUNT) {
      list->buffer_barriers = list->inline_buffer_barriers;
    } else {
      status = iree_allocator_malloc_array(host_allocator, buffer_barrier_count,
                                           sizeof(*list->buffer_barriers),
                                           (void**)&list->buffer_barriers);
    }
  }
  for (iree_host_size_t i = 0;
       i < buffer_barrier_count && iree_status_is_ok(status); ++i) {
    list->buffer_barriers[i].source_scope =
        (iree_hal_access_scope_t)wire_buffer_barriers[i].source_scope;
    list->buffer_barriers[i].target_scope =
        (iree_hal_access_scope_t)wire_buffer_barriers[i].target_scope;
    status = iree_hal_remote_server_resolve_command_buffer_ref(
        session_slot, wire_buffer_barriers[i].buffer_id,
        wire_buffer_barriers[i].buffer_slot, wire_buffer_barriers[i].offset,
        wire_buffer_barriers[i].length, command_name,
        &list->buffer_barriers[i].buffer_ref);
  }

  return status;
}

static iree_status_t iree_hal_remote_server_prepare_event_list(
    iree_hal_remote_server_session_t* session_slot,
    iree_host_size_t event_count,
    const iree_hal_remote_resource_id_t* wire_event_ids,
    const char* command_name, iree_hal_remote_server_event_list_t* list) {
  iree_allocator_t host_allocator = session_slot->server->host_allocator;
  iree_status_t status = iree_ok_status();

  if (event_count > 0) {
    if (event_count <= IREE_HAL_REMOTE_INLINE_EVENT_COUNT) {
      list->events = list->inline_events;
    } else {
      status = iree_allocator_malloc_array(host_allocator, event_count,
                                           sizeof(*list->events),
                                           (void**)&list->events);
    }
  }
  for (iree_host_size_t i = 0; i < event_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_remote_resource_id_t event_id =
        iree_hal_remote_server_resolve_resource_id(session_slot,
                                                   wire_event_ids[i]);
    iree_hal_event_t* event =
        (iree_hal_event_t*)iree_hal_remote_resource_table_lookup(
            &session_slot->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_EVENT,
            event_id);
    if (event) {
      list->events[i] = event;
    } else {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "%s event 0x%016" PRIx64 " not found",
                                command_name, event_id);
    }
  }

  return status;
}

static iree_status_t iree_hal_remote_server_replay_execution_barrier_cmd(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer, const uint8_t* cmd_data) {
  const iree_hal_remote_execution_barrier_cmd_t* cmd =
      (const iree_hal_remote_execution_barrier_cmd_t*)cmd_data;
  const iree_host_size_t memory_barriers_offset = sizeof(*cmd);
  const iree_host_size_t buffer_barriers_offset =
      memory_barriers_offset +
      cmd->memory_barrier_count * sizeof(iree_hal_remote_memory_barrier_t);
  iree_hal_remote_server_barrier_list_t barrier_list;
  iree_hal_remote_server_barrier_list_initialize(&barrier_list);

  iree_status_t status = iree_hal_remote_server_prepare_barrier_list(
      session_slot, cmd->memory_barrier_count,
      (const iree_hal_remote_memory_barrier_t*)(cmd_data +
                                                memory_barriers_offset),
      cmd->buffer_barrier_count,
      (const iree_hal_remote_buffer_barrier_t*)(cmd_data +
                                                buffer_barriers_offset),
      "EXECUTION_BARRIER", &barrier_list);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_execution_barrier(
        local_command_buffer,
        (iree_hal_execution_stage_t)cmd->source_stage_mask,
        (iree_hal_execution_stage_t)cmd->target_stage_mask,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, cmd->memory_barrier_count,
        barrier_list.memory_barriers, cmd->buffer_barrier_count,
        barrier_list.buffer_barriers);
  }

  iree_hal_remote_server_barrier_list_deinitialize(
      &barrier_list, session_slot->server->host_allocator);
  return status;
}

static iree_status_t iree_hal_remote_server_replay_event_signal_cmd(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer, const uint8_t* cmd_data) {
  const iree_hal_remote_event_signal_cmd_t* cmd =
      (const iree_hal_remote_event_signal_cmd_t*)cmd_data;
  iree_hal_remote_resource_id_t event_id =
      iree_hal_remote_server_resolve_resource_id(session_slot, cmd->event_id);
  iree_hal_event_t* event =
      (iree_hal_event_t*)iree_hal_remote_resource_table_lookup(
          &session_slot->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_EVENT,
          event_id);
  if (!event) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "EVENT_SIGNAL event 0x%016" PRIx64 " not found",
                            event_id);
  }
  return iree_hal_command_buffer_signal_event(
      local_command_buffer, event,
      (iree_hal_execution_stage_t)cmd->source_stage_mask);
}

static iree_status_t iree_hal_remote_server_replay_event_reset_cmd(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer, const uint8_t* cmd_data) {
  const iree_hal_remote_event_reset_cmd_t* cmd =
      (const iree_hal_remote_event_reset_cmd_t*)cmd_data;
  iree_hal_remote_resource_id_t event_id =
      iree_hal_remote_server_resolve_resource_id(session_slot, cmd->event_id);
  iree_hal_event_t* event =
      (iree_hal_event_t*)iree_hal_remote_resource_table_lookup(
          &session_slot->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_EVENT,
          event_id);
  if (!event) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "EVENT_RESET event 0x%016" PRIx64 " not found",
                            event_id);
  }
  return iree_hal_command_buffer_reset_event(
      local_command_buffer, event,
      (iree_hal_execution_stage_t)cmd->source_stage_mask);
}

static iree_status_t iree_hal_remote_server_replay_event_wait_cmd(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer, const uint8_t* cmd_data) {
  const iree_hal_remote_event_wait_cmd_t* cmd =
      (const iree_hal_remote_event_wait_cmd_t*)cmd_data;
  const iree_host_size_t event_ids_offset = sizeof(*cmd);
  const iree_host_size_t memory_barriers_offset =
      event_ids_offset +
      cmd->event_count * sizeof(iree_hal_remote_resource_id_t);
  const iree_host_size_t buffer_barriers_offset =
      memory_barriers_offset +
      cmd->memory_barrier_count * sizeof(iree_hal_remote_memory_barrier_t);
  iree_hal_remote_server_event_list_t event_list;
  iree_hal_remote_server_barrier_list_t barrier_list;
  iree_hal_remote_server_event_list_initialize(&event_list);
  iree_hal_remote_server_barrier_list_initialize(&barrier_list);

  iree_status_t status = iree_hal_remote_server_prepare_event_list(
      session_slot, cmd->event_count,
      (const iree_hal_remote_resource_id_t*)(cmd_data + event_ids_offset),
      "EVENT_WAIT", &event_list);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_prepare_barrier_list(
        session_slot, cmd->memory_barrier_count,
        (const iree_hal_remote_memory_barrier_t*)(cmd_data +
                                                  memory_barriers_offset),
        cmd->buffer_barrier_count,
        (const iree_hal_remote_buffer_barrier_t*)(cmd_data +
                                                  buffer_barriers_offset),
        "EVENT_WAIT", &barrier_list);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_wait_events(
        local_command_buffer, cmd->event_count, event_list.events,
        (iree_hal_execution_stage_t)cmd->source_stage_mask,
        (iree_hal_execution_stage_t)cmd->target_stage_mask,
        cmd->memory_barrier_count, barrier_list.memory_barriers,
        cmd->buffer_barrier_count, barrier_list.buffer_barriers);
  }

  iree_hal_remote_server_event_list_deinitialize(
      &event_list, session_slot->server->host_allocator);
  iree_hal_remote_server_barrier_list_deinitialize(
      &barrier_list, session_slot->server->host_allocator);
  return status;
}

// Replays a single DISPATCH command from the serialized stream.
static iree_status_t iree_hal_remote_server_replay_dispatch_cmd(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer, const uint8_t* cmd_data) {
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

  const iree_host_size_t constants_offset = sizeof(*cmd);
  const iree_host_size_t bindings_offset = iree_host_align(
      constants_offset + cmd->constant_count * sizeof(uint32_t), 8);
  const uint8_t* constants_data = cmd_data + constants_offset;
  iree_const_byte_span_t constants = iree_make_const_byte_span(
      constants_data, (iree_host_size_t)cmd->constant_count * sizeof(uint32_t));

  // Parse and resolve bindings.
  const iree_hal_remote_binding_t* wire_bindings =
      (const iree_hal_remote_binding_t*)(cmd_data + bindings_offset);
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
      iree_hal_executable_function_from_value(cmd->function_value), config,
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
    iree_hal_remote_command_view_t command;
    iree_status_t status = iree_hal_remote_command_parse(
        iree_make_const_byte_span(stream_data + offset, stream_length - offset),
        &command);
    if (!iree_status_is_ok(status)) {
      return iree_status_annotate_f(status, "command at offset %" PRIhsz,
                                    offset);
    }
    const iree_hal_remote_cmd_header_t* header = &command.header;
    const uint8_t* command_data = command.bytes.data;

    switch (header->type) {
      case IREE_HAL_REMOTE_CMD_EXECUTION_BARRIER:
        status = iree_hal_remote_server_replay_execution_barrier_cmd(
            session_slot, local_command_buffer, command_data);
        break;

      case IREE_HAL_REMOTE_CMD_EVENT_SIGNAL:
        status = iree_hal_remote_server_replay_event_signal_cmd(
            session_slot, local_command_buffer, command_data);
        break;

      case IREE_HAL_REMOTE_CMD_EVENT_RESET:
        status = iree_hal_remote_server_replay_event_reset_cmd(
            session_slot, local_command_buffer, command_data);
        break;

      case IREE_HAL_REMOTE_CMD_EVENT_WAIT:
        status = iree_hal_remote_server_replay_event_wait_cmd(
            session_slot, local_command_buffer, command_data);
        break;

      case IREE_HAL_REMOTE_CMD_BUFFER_FILL: {
        const iree_hal_remote_buffer_fill_cmd_t* cmd =
            (const iree_hal_remote_buffer_fill_cmd_t*)command_data;
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
        const iree_hal_remote_buffer_update_cmd_t* cmd =
            (const iree_hal_remote_buffer_update_cmd_t*)command_data;
        const void* source_data = command_data + sizeof(*cmd);
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
        const iree_hal_remote_buffer_copy_cmd_t* cmd =
            (const iree_hal_remote_buffer_copy_cmd_t*)command_data;
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
            session_slot, local_command_buffer, command_data);
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

    offset += command.bytes.data_length;
  }
  return iree_ok_status();
}

// Records a serialized command stream into an unrecorded local command buffer.
// Once recording begins, end is always paired with it so that drivers can
// finalize or unwind any state retained before a replay failure.
static iree_status_t iree_hal_remote_server_record_command_stream(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer, const uint8_t* stream_data,
    iree_host_size_t stream_length) {
  iree_status_t status = iree_hal_command_buffer_begin(local_command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_replay_command_stream(
        session_slot, local_command_buffer, stream_data, stream_length);
    status = iree_status_join(
        status, iree_hal_command_buffer_end(local_command_buffer));
  }
  return status;
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
  iree_host_size_t data_offset = 0;
  iree_host_size_t required_length = 0;
  iree_status_t layout_status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_command_buffer_upload_request_t), &required_length,
      IREE_STRUCT_FIELD((iree_host_size_t)request->data_length, uint8_t,
                        &data_offset));
  if (!iree_status_is_ok(layout_status)) {
    return iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, layout_status);
  }
  if (body_length < required_length) {
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
    status = iree_hal_remote_server_record_command_stream(
        entry, command_buffer, stream_data, stream_length);
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
  iree_host_size_t bindings_offset = 0;
  iree_host_size_t stream_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_command_buffer_execute_op_t), &stream_offset,
      IREE_STRUCT_FIELD(op->binding_count, iree_hal_remote_binding_t,
                        &bindings_offset)));
  if (stream_offset > command_length) {
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
    status = iree_hal_remote_server_record_command_stream(
        session_slot, local_command_buffer, stream_data, stream_length);
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
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < resource_count; ++i) {
    iree_hal_remote_resource_id_t resolved_id =
        iree_hal_remote_server_resolve_resource_id(entry, resource_ids[i]);
    if (!IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(resolved_id)) {
      if (iree_hal_remote_server_is_virtual_buffer(entry, resolved_id)) {
        status = iree_status_join(
            status,
            iree_hal_remote_server_release_virtual_buffer(entry, resolved_id));
      } else {
        iree_hal_remote_resource_table_release(&entry->resource_table,
                                               resolved_id);
      }
    }
    if (resolved_id != resource_ids[i] ||
        IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(resource_ids[i])) {
      iree_hal_remote_server_pending_queue_command_t* pending_commands =
          iree_hal_remote_server_remove_provisional(entry, resource_ids[i]);
      status = iree_status_join(
          status,
          iree_hal_remote_server_fail_pending_queue_commands(
              entry, pending_commands,
              iree_make_status(IREE_STATUS_ABORTED,
                               "resource release canceled pending provisional "
                               "resource 0x%016" PRIx64,
                               resource_ids[i])));
    }
  }
  return status;
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
  while (ready_list) {
    iree_net_sequence_node_t* next = ready_list->next;
    iree_hal_remote_server_resource_release_node_t* release_node =
        iree_containerof(ready_list,
                         iree_hal_remote_server_resource_release_node_t,
                         sequence_node);
    ready_list->next = NULL;
    status = iree_status_join(
        status,
        iree_hal_remote_server_release_resource_node(entry, release_node));
    ready_list = next;
  }
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
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_resource_release_batch_t), &expected_size,
      IREE_STRUCT_FIELD(resource_count, iree_hal_remote_resource_id_t, NULL)));
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
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_resource_release_op_t), &expected_size,
      IREE_STRUCT_FIELD(resource_count, iree_hal_remote_resource_id_t, NULL)));
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

static iree_status_t iree_hal_remote_server_observe_submission_frontier(
    iree_hal_remote_server_session_t* session_slot,
    const iree_async_frontier_t* signal_frontier) {
  uint64_t signal_epoch = signal_frontier->entries[0].epoch;

  iree_net_sequence_node_t* ready_releases = NULL;
  bool session_active = false;
  iree_hal_remote_server_t* server = session_slot->server;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&server->session_mutex);
  session_active = session_slot->session != NULL;
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
  return status;
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
iree_status_t iree_hal_remote_server_on_queue_command(
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

  status = iree_hal_remote_server_validate_queue_frontier(
      session_slot, "signal", signal_frontier);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_remote_server_t* server = session_slot->server;
  iree_slim_mutex_lock(&server->session_mutex);
  bool queue_terminal = iree_any_bit_set(
      session_slot->queue_flags, IREE_HAL_REMOTE_SERVER_QUEUE_FLAG_TERMINAL);
  iree_slim_mutex_unlock(&server->session_mutex);
  if (queue_terminal) {
    status = iree_hal_remote_server_observe_submission_frontier(
        session_slot, signal_frontier);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  status = iree_hal_remote_server_validate_queue_frontier(session_slot, "wait",
                                                          wait_frontier);
  if (!iree_status_is_ok(status)) {
    status = iree_hal_remote_server_fail_queue_command(session_slot,
                                                       signal_frontier, status);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_remote_resource_id_t pending_file_id = 0;
  if (op_header && iree_hal_remote_server_file_command_waits_on_provisional(
                       session_slot, command_data, &pending_file_id)) {
    if (!session_slot->queue_channel) {
      status = iree_status_from_code(IREE_STATUS_ABORTED);
    } else {
      status = iree_hal_remote_server_enqueue_pending_queue_command(
          session_slot, pending_file_id, session_slot->queue_channel,
          wait_frontier, signal_frontier, command_data, lease,
          session_slot->server->host_allocator);
    }
    if (!iree_status_is_ok(status) && session_slot->queue_channel) {
      status = iree_hal_remote_server_fail_queue_command(
          session_slot, signal_frontier, status);
    }
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
      case IREE_HAL_REMOTE_QUEUE_OP_CLIENT_FILE_READ:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_client_file_read, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_CLIENT_FILE_WRITE:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_client_file_write, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_FILE_READ:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_file_read, &op_context);
        break;
      case IREE_HAL_REMOTE_QUEUE_OP_FILE_WRITE:
        status = iree_hal_remote_server_submit_command(
            session_slot, wait_frontier, signal_frontier,
            iree_hal_remote_server_submit_file_write, &op_context);
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
        status = iree_hal_remote_server_fail_queue_command(
            session_slot, signal_frontier, status);
        break;
    }
  } else {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "COMMAND payload too short for op header: "
                              "%" PRIhsz " bytes",
                              command_data.data_length);
    status = iree_hal_remote_server_fail_queue_command(session_slot,
                                                       signal_frontier, status);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_remote_server_process_pending_queue_commands(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_remote_server_pending_queue_command_t* pending_commands) {
  iree_status_t status = iree_ok_status();
  while (pending_commands) {
    iree_hal_remote_server_pending_queue_command_t* next =
        pending_commands->next;
    pending_commands->next = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_remote_server_process_pending_queue_command(
          session_slot, pending_commands);
    } else {
      iree_status_t command_status = iree_status_clone(status);
      iree_status_t fail_status =
          iree_hal_remote_server_fail_pending_queue_command(
              session_slot, pending_commands, command_status);
      status = iree_status_join(status, fail_status);
    }
    pending_commands = next;
  }
  return status;
}

// Handles FILE_OPEN: resolves a server-side logical file name through the
// configured file index and assigns the imported HAL file to the session table.
static iree_status_t iree_hal_remote_server_handle_file_open(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  const bool fire_and_forget = iree_all_bits_set(
      envelope->message_flags, IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET);
  iree_status_t status = iree_ok_status();
  const iree_hal_remote_file_open_request_t* request = NULL;
  iree_string_view_t logical_name = iree_string_view_empty();

  if (body_length < sizeof(iree_hal_remote_file_open_request_t)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "FILE_OPEN body too small: %" PRIhsz " bytes",
                              body_length);
  }
  if (iree_status_is_ok(status)) {
    request = (const iree_hal_remote_file_open_request_t*)body;
    if (request->flags != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "FILE_OPEN flags must be 0");
    }
  }
  if (iree_status_is_ok(status) &&
      IREE_HAL_REMOTE_RESOURCE_ID_TYPE(request->provisional_id) !=
          IREE_HAL_REMOTE_RESOURCE_TYPE_FILE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "FILE_OPEN provisional id must have FILE type");
  }
  if (iree_status_is_ok(status) &&
      !IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(request->provisional_id)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "FILE_OPEN provisional id must be provisional");
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t path_offset = 0;
    iree_host_size_t path_end = 0;
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_file_open_request_t), &path_end,
        IREE_STRUCT_FIELD(request->path_length, uint8_t, &path_offset));
    if (iree_status_is_ok(status) && path_end > body_length) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "FILE_OPEN path string truncated");
    }
    if (iree_status_is_ok(status)) {
      logical_name = iree_make_string_view((const char*)body + path_offset,
                                           request->path_length);
    }
  }

  bool provisional_prepared = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_prepare_provisional(
        entry, request->provisional_id, entry->server->host_allocator);
    provisional_prepared = iree_status_is_ok(status);
  }

  iree_hal_memory_access_t granted_access = 0;
  iree_io_file_handle_t* file_handle = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_file_index_open(
        entry->server->options.file_index, logical_name,
        (iree_hal_memory_access_t)request->mode, entry->server->host_allocator,
        &file_handle, &granted_access);
  }

  iree_hal_file_t* file = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_import(
        entry->server->devices[0], IREE_HAL_QUEUE_AFFINITY_ANY, granted_access,
        file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file);
  }

  uint64_t file_size = 0;
  iree_hal_remote_resource_id_t resolved_id = 0;
  if (iree_status_is_ok(status)) {
    file_size = iree_hal_file_length(file);
    status = iree_hal_remote_resource_table_assign(
        &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_FILE, file,
        &resolved_id);
  }
  iree_hal_remote_server_pending_queue_command_t* pending_commands = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_store_provisional(
        entry, request->provisional_id, resolved_id,
        entry->server->host_allocator, &pending_commands);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_free_pending_queue_commands(pending_commands);
    pending_commands = NULL;
  }

  if (!iree_status_is_ok(status) && resolved_id != 0) {
    iree_hal_remote_resource_table_release(&entry->resource_table, resolved_id);
    resolved_id = 0;
  }
  iree_status_t pending_failure_status = iree_ok_status();
  if (!iree_status_is_ok(status) && provisional_prepared) {
    iree_hal_remote_server_pending_queue_command_t* pending_commands = NULL;
    iree_hal_remote_server_fail_provisional(
        entry, request->provisional_id, iree_status_code(status),
        entry->server->host_allocator, &pending_commands);
    pending_failure_status = iree_hal_remote_server_fail_pending_queue_commands(
        entry, pending_commands, iree_status_clone(status));
  }

  iree_hal_file_release(file);
  iree_io_file_handle_release(file_handle);

  iree_status_t send_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    if (!fire_and_forget) {
      iree_hal_remote_file_open_response_t response;
      memset(&response, 0, sizeof(response));
      response.resolved_id = resolved_id;
      response.file_size = file_size;
      response.granted_access = granted_access;
      send_status = iree_hal_remote_server_send_response(
          entry->server->host_allocator, entry->session, envelope,
          IREE_STATUS_OK, &response, sizeof(response));
    }
    send_status = iree_status_join(
        send_status, iree_hal_remote_server_process_pending_queue_commands(
                         entry, pending_commands));
    pending_commands = NULL;
  } else if (!fire_and_forget) {
    send_status = iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
    status = iree_ok_status();
  } else {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  send_status = iree_status_join(send_status, pending_failure_status);
  iree_hal_remote_server_free_pending_queue_commands(pending_commands);
  return send_status;
}

static iree_status_t iree_hal_remote_server_handle_file_close(
    iree_hal_remote_server_session_t* entry, const uint8_t* body,
    iree_host_size_t body_length) {
  iree_status_t status = iree_ok_status();
  if (body_length < sizeof(iree_hal_remote_file_close_t)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "FILE_CLOSE body too small: %" PRIhsz " bytes",
                              body_length);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_remote_file_close_t* request =
        (const iree_hal_remote_file_close_t*)body;
    iree_hal_remote_resource_id_t resolved_id =
        iree_hal_remote_server_resolve_resource_id(entry, request->file_id);
    if (!IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(resolved_id)) {
      iree_hal_remote_resource_table_release(&entry->resource_table,
                                             resolved_id);
    }
    if (resolved_id != request->file_id ||
        IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(request->file_id)) {
      iree_hal_remote_server_pending_queue_command_t* pending_commands =
          iree_hal_remote_server_remove_provisional(entry, request->file_id);
      status = iree_hal_remote_server_fail_pending_queue_commands(
          entry, pending_commands,
          iree_make_status(IREE_STATUS_ABORTED,
                           "FILE_CLOSE canceled pending provisional file "
                           "0x%016" PRIx64,
                           request->file_id));
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_server_handle_file_register(
    iree_hal_remote_server_session_t* entry,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  const bool fire_and_forget = iree_all_bits_set(
      envelope->message_flags, IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET);
  const iree_hal_remote_file_registration_capabilities_t
      supported_capabilities = entry->file_registration_capabilities;
  const iree_hal_memory_access_t allowed_access_mask =
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;

  iree_status_t status = iree_ok_status();
  const iree_hal_remote_file_register_request_t* request = NULL;
  if (body_length < sizeof(iree_hal_remote_file_register_request_t)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "FILE_REGISTER body too small: %" PRIhsz " bytes",
                              body_length);
  }
  if (iree_status_is_ok(status)) {
    request = (const iree_hal_remote_file_register_request_t*)body;
    if (request->reserved != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "FILE_REGISTER reserved field must be 0");
    }
  }
  if (iree_status_is_ok(status) &&
      IREE_HAL_REMOTE_RESOURCE_ID_TYPE(request->provisional_id) !=
          IREE_HAL_REMOTE_RESOURCE_TYPE_FILE) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "FILE_REGISTER provisional id must have FILE type");
  }
  if (iree_status_is_ok(status) &&
      !IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(request->provisional_id)) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "FILE_REGISTER provisional id must be provisional");
  }
  if (iree_status_is_ok(status) && request->access_flags == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "FILE_REGISTER access flags must be nonzero");
  }
  if (iree_status_is_ok(status) &&
      (request->access_flags & ~allowed_access_mask) != 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "FILE_REGISTER access flags 0x%08x contain "
                              "unsupported bits outside read/write",
                              request->access_flags);
  }

  iree_host_size_t required_length = 0;
  iree_host_size_t handle_payload_offset = 0;
  if (iree_status_is_ok(status)) {
    status =
        IREE_STRUCT_LAYOUT(sizeof(*request), &required_length,
                           IREE_STRUCT_FIELD(request->handle_payload_length,
                                             uint8_t, &handle_payload_offset));
  }
  if (iree_status_is_ok(status) && body_length < required_length) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "FILE_REGISTER handle payload truncated: need %" PRIhsz
                         " bytes, got %" PRIhsz " bytes",
                         required_length, body_length);
  }
  if (iree_status_is_ok(status) && request->external_type > UINT8_MAX) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "FILE_REGISTER external file type %u is out of range",
                         request->external_type);
  }

  iree_hal_remote_file_registration_capabilities_t required_capability =
      IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE;
  if (iree_status_is_ok(status)) {
    required_capability =
        iree_hal_remote_file_registration_capability_for_external_type(
            (iree_hal_remote_file_external_type_t)request->external_type);
    if (required_capability ==
        IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "FILE_REGISTER external file type %u is not defined",
                           request->external_type);
    }
  }

  bool provisional_prepared = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_prepare_provisional(
        entry, request->provisional_id, entry->server->host_allocator);
    provisional_prepared = iree_status_is_ok(status);
  }

  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(supported_capabilities, required_capability)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "FILE_REGISTER external file type %u requires capability 0x%08x, "
        "but this server supports 0x%08x",
        request->external_type, required_capability, supported_capabilities);
  }
  if (iree_status_is_ok(status) && !entry->carrier) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "FILE_REGISTER requires a session carrier");
  }
  iree_const_byte_span_t handle_payload = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    handle_payload = iree_make_const_byte_span(body + handle_payload_offset,
                                               request->handle_payload_length);
  }

  iree_net_file_handle_transfer_type_t transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_NONE;
  if (iree_status_is_ok(status)) {
    switch ((iree_hal_remote_file_external_type_t)request->external_type) {
      case IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_POSIX_FD:
        transfer_type = IREE_NET_FILE_HANDLE_TRANSFER_TYPE_POSIX_FD;
        break;
      case IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_WIN32_HANDLE:
        transfer_type = IREE_NET_FILE_HANDLE_TRANSFER_TYPE_WIN32_HANDLE;
        break;
      default:
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "FILE_REGISTER external file type %u is not importable",
            request->external_type);
        break;
    }
  }

  iree_io_file_handle_t* file_handle = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_carrier_import_file_handle(
        entry->carrier, transfer_type, handle_payload,
        entry->server->host_allocator, &file_handle);
  }
  if (iree_status_is_ok(status) &&
      !iree_io_file_handle_uses_async_io(file_handle)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "FILE_REGISTER imported file handle is not opened for async I/O");
  }

  iree_hal_file_t* file = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_import(
        entry->server->devices[0], IREE_HAL_QUEUE_AFFINITY_ANY,
        (iree_hal_memory_access_t)request->access_flags, file_handle,
        IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file);
  }

  uint64_t file_size = 0;
  iree_hal_remote_resource_id_t resolved_id = 0;
  if (iree_status_is_ok(status)) {
    file_size = iree_hal_file_length(file);
    status = iree_hal_remote_resource_table_assign(
        &entry->resource_table, IREE_HAL_REMOTE_RESOURCE_TYPE_FILE, file,
        &resolved_id);
  }
  iree_hal_remote_server_pending_queue_command_t* pending_commands = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_store_provisional(
        entry, request->provisional_id, resolved_id,
        entry->server->host_allocator, &pending_commands);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_free_pending_queue_commands(pending_commands);
    pending_commands = NULL;
  }

  if (!iree_status_is_ok(status) && resolved_id != 0) {
    iree_hal_remote_resource_table_release(&entry->resource_table, resolved_id);
    resolved_id = 0;
  }

  iree_status_t pending_failure_status = iree_ok_status();
  if (!iree_status_is_ok(status) && provisional_prepared) {
    iree_hal_remote_server_pending_queue_command_t* pending_commands = NULL;
    iree_hal_remote_server_fail_provisional(
        entry, request->provisional_id, iree_status_code(status),
        entry->server->host_allocator, &pending_commands);
    pending_failure_status = iree_hal_remote_server_fail_pending_queue_commands(
        entry, pending_commands, iree_status_clone(status));
  }

  iree_hal_file_release(file);
  iree_io_file_handle_release(file_handle);

  iree_status_t send_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    if (!fire_and_forget) {
      iree_hal_remote_file_register_response_t response;
      memset(&response, 0, sizeof(response));
      response.resolved_id = resolved_id;
      response.file_size = file_size;
      send_status = iree_hal_remote_server_send_response(
          entry->server->host_allocator, entry->session, envelope,
          IREE_STATUS_OK, &response, sizeof(response));
    }
    send_status = iree_status_join(
        send_status, iree_hal_remote_server_process_pending_queue_commands(
                         entry, pending_commands));
    pending_commands = NULL;
  } else if (!fire_and_forget) {
    send_status = iree_hal_remote_server_send_error_response(
        entry->server->host_allocator, entry->session, envelope, status);
    status = iree_ok_status();
  }
  send_status = iree_status_join(send_status, pending_failure_status);
  iree_hal_remote_server_free_pending_queue_commands(pending_commands);
  return iree_status_join(status, send_status);
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
  iree_hal_remote_server_session_t* session_slot =
      (iree_hal_remote_server_session_t*)user_data;
  iree_hal_remote_server_t* server = session_slot->server;
  iree_slim_mutex_lock(&server->session_mutex);
  uint64_t session_id = session_slot->session_id;
  iree_slim_mutex_unlock(&server->session_mutex);
  iree_hal_remote_server_fail_active_session(
      session_slot, session_id,
      iree_status_annotate(status, IREE_SV("remote queue transport failed")));
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
        .on_command = iree_hal_remote_server_on_queue_command,
        .on_advance = iree_hal_remote_server_on_advance,
        .on_transport_error = iree_hal_remote_server_on_queue_transport_error,
        .on_send_complete = iree_hal_remote_server_on_queue_send_complete,
        .on_send_ready = iree_hal_remote_server_on_queue_send_ready,
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
    iree_async_buffer_pool_release(header_pool);
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
    iree_net_bulk_channel_callbacks_t callbacks =
        iree_hal_remote_server_bulk_session_channel_callbacks(
            &server->sessions[slot]);
    status =
        iree_net_bulk_channel_create(endpoint, /*options=*/NULL, header_pool,
                                     callbacks, host_allocator, &bulk_channel);
    header_pool = NULL;  // bulk_channel_create consumes the pool.
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_bulk_channel_activate(bulk_channel);
  }

  bool bulk_channel_attached = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_session_attach_channel(
        &server->sessions[slot], bulk_channel);
    bulk_channel_attached = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_session_flush_receive_window(
        &server->sessions[slot]);
  }

  if (iree_status_is_ok(status)) {
    iree_net_bulk_channel_release(bulk_channel);
    bulk_channel = NULL;
  } else {
    if (bulk_channel_attached) {
      iree_net_bulk_channel_t* attached_channel =
          iree_hal_remote_server_bulk_session_take_channel(
              &server->sessions[slot]);
      iree_net_bulk_channel_release(attached_channel);
    }
    if (bulk_channel) {
      iree_net_bulk_channel_release(bulk_channel);
    } else {
      iree_async_buffer_pool_release(header_pool);
    }
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

  entry->carrier = iree_net_session_carrier(session);
  entry->file_registration_capabilities =
      iree_hal_remote_server_file_registration_capabilities(entry->carrier);

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

  // The session has already published this terminal status to its frontier
  // axes. This callback owns the diagnostic and completes server-side teardown.
  iree_status_free(status);
  iree_hal_remote_server_remove_session(server, session);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_remote_server_on_session_send_complete(
    void* user_data, uint64_t operation_user_data, iree_status_t status) {
  (void)operation_user_data;
  iree_hal_remote_server_session_t* entry =
      (iree_hal_remote_server_session_t*)user_data;
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_on_session_error(user_data, entry->session, status);
  } else {
    iree_status_free(status);
  }
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
    case IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_QUERY_CAPABILITIES:
      return iree_hal_remote_server_handle_buffer_virtual_query_capabilities(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_QUERY_GRANULARITY:
      return iree_hal_remote_server_handle_buffer_virtual_query_granularity(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_RESERVE:
      return iree_hal_remote_server_handle_buffer_virtual_reserve(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_RELEASE:
      return iree_hal_remote_server_handle_buffer_virtual_release(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_PHYSICAL_ALLOC:
      return iree_hal_remote_server_handle_buffer_physical_alloc(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_PHYSICAL_FREE:
      return iree_hal_remote_server_handle_buffer_physical_free(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_MAP:
      return iree_hal_remote_server_handle_buffer_virtual_map(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_UNMAP:
      return iree_hal_remote_server_handle_buffer_virtual_unmap(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_PROTECT:
      return iree_hal_remote_server_handle_buffer_virtual_protect(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_ADVISE:
      return iree_hal_remote_server_handle_buffer_virtual_advise(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_EXECUTABLE_UPLOAD:
      return iree_hal_remote_server_handle_executable_upload(entry, envelope,
                                                             body, body_length);
    case IREE_HAL_REMOTE_CONTROL_EXECUTABLE_QUERY_FUNCTION:
      return iree_hal_remote_server_handle_executable_query_function(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_EXECUTABLE_QUERY_PARAMETERS:
      return iree_hal_remote_server_handle_executable_query_parameters(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_EXECUTABLE_LOOKUP_GLOBAL:
      return iree_hal_remote_server_handle_executable_lookup_global(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_EXECUTABLE_GLOBAL_BUFFER:
      return iree_hal_remote_server_handle_executable_global_buffer(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_EVENT_CREATE:
      return iree_hal_remote_server_handle_event_create(entry, envelope, body,
                                                        body_length);
    case IREE_HAL_REMOTE_CONTROL_COMMAND_BUFFER_UPLOAD:
      return iree_hal_remote_server_handle_command_buffer_upload(
          entry, envelope, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_FILE_OPEN:
      return iree_hal_remote_server_handle_file_open(entry, envelope, body,
                                                     body_length);
    case IREE_HAL_REMOTE_CONTROL_FILE_CLOSE:
      return iree_hal_remote_server_handle_file_close(entry, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_FILE_REGISTER:
      return iree_hal_remote_server_handle_file_register(entry, envelope, body,
                                                         body_length);
    case IREE_HAL_REMOTE_CONTROL_FILE_UNREGISTER:
      return iree_hal_remote_server_handle_file_close(entry, body, body_length);
    case IREE_HAL_REMOTE_CONTROL_RESOURCE_RELEASE_BATCH:
      return iree_hal_remote_server_handle_resource_release_batch(entry, body,
                                                                  body_length);
    case IREE_HAL_REMOTE_CONTROL_PROFILING_BEGIN:
      return iree_hal_remote_server_handle_profiling_begin(entry, envelope,
                                                           body, body_length);
    case IREE_HAL_REMOTE_CONTROL_PROFILING_FLUSH:
      return iree_hal_remote_server_handle_profiling_flush(entry, envelope,
                                                           body, body_length);
    case IREE_HAL_REMOTE_CONTROL_PROFILING_END:
      return iree_hal_remote_server_handle_profiling_end(entry, envelope, body,
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
