// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_BUFFER_H_
#define IREE_HAL_REMOTE_CLIENT_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/pool.h"
#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_client_device_t iree_hal_remote_client_device_t;

// Remote buffer proxy. Wraps the HAL buffer base with a server-assigned
// resource_id. All immutable properties (memory_type, allowed_access,
// allowed_usage, allocation_size) are cached in the base struct and served
// locally — no round-trip for property queries.
typedef struct iree_hal_remote_client_buffer_t {
  // Base HAL buffer storing cached buffer metadata.
  iree_hal_buffer_t base;

  // Host allocator used for proxy storage.
  iree_allocator_t host_allocator;

  // Back-pointer to the owning device for release notifications.
  iree_hal_remote_client_device_t* device;

  // Current provisional or server-assigned resource ID.
  // May transition atomically while application threads submit queue work.
  iree_atomic_uint64_t resource_id;

  // Materialized backing buffer for logical wrappers, or NULL for direct views.
  iree_hal_buffer_t* backing_buffer;

  // Callback invoked when this proxy is destroyed.
  iree_hal_buffer_release_callback_t release_callback;

  // Pool that owns allocation_reservation while armed, or NULL.
  iree_hal_pool_t* allocation_pool;

  // Pool reservation returned by queue_dealloca or destroy when armed.
  iree_hal_pool_reservation_t allocation_reservation;

  // Nonzero while allocation_reservation still belongs to this proxy.
  iree_atomic_int32_t allocation_reservation_armed;

  // Nonzero once queue_dealloca has made the logical allocation unusable.
  iree_atomic_int32_t deallocated;

  // True if destroy should release resource_id on the remote server.
  bool owns_remote_resource;

  // Staging allocation for the active mapping, or NULL when unmapped.
  uint8_t* active_mapping_data;

  // Access bits granted to the active mapping.
  iree_hal_memory_access_t active_mapping_access;

  // Buffer offset where active_mapping_data begins.
  iree_device_size_t active_mapping_offset;

  // Length of active_mapping_data in bytes.
  iree_device_size_t active_mapping_length;

  // Count of active read-only staging mappings that are not tracked.
  uint32_t active_untracked_mapping_count;
} iree_hal_remote_client_buffer_t;

// Creates a buffer proxy wrapping a server-assigned resource.
iree_status_t iree_hal_remote_client_buffer_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_device_size_t byte_length,
    iree_hal_buffer_placement_flags_t placement_flags,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Creates an unbacked logical buffer proxy for a queue allocation. The proxy is
// materialized later with iree_hal_remote_client_buffer_set_backing().
iree_status_t iree_hal_remote_client_buffer_create_unbacked(
    iree_hal_remote_client_device_t* device,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_buffer_placement_flags_t placement_flags,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Creates a buffer proxy view over an existing remote root allocation.
iree_status_t iree_hal_remote_client_buffer_create_view(
    iree_hal_remote_client_device_t* device, iree_hal_buffer_t* root_buffer,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    const iree_hal_buffer_params_t* params,
    iree_hal_buffer_placement_flags_t placement_flags,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Materializes an unbacked logical buffer with a concrete remote backing view.
iree_status_t iree_hal_remote_client_buffer_set_backing(
    iree_hal_buffer_t* buffer, iree_hal_buffer_t* backing_buffer);

// Attaches a queue-allocation pool reservation to |buffer|.
void iree_hal_remote_client_buffer_attach_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation);

// Releases an attached reservation if still armed.
void iree_hal_remote_client_buffer_release_reservation(
    iree_hal_buffer_t* buffer, const iree_async_frontier_t* death_frontier);

// Returns true if |buffer| owns an armed pool reservation.
bool iree_hal_remote_client_buffer_has_reservation(iree_hal_buffer_t* buffer);

// Marks the logical queue allocation as deallocated.
void iree_hal_remote_client_buffer_mark_deallocated(iree_hal_buffer_t* buffer);

// Returns true if the logical queue allocation has been deallocated.
bool iree_hal_remote_client_buffer_is_deallocated(iree_hal_buffer_t* buffer);

// Resolves a buffer reference to wire resource and absolute offset values.
iree_status_t iree_hal_remote_client_buffer_resolve_wire_ref(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_hal_remote_resource_id_t* out_resource_id, uint64_t* out_byte_offset);

// Resolves a buffer range to wire resource, offset, and length values.
iree_status_t iree_hal_remote_client_buffer_resolve_wire_range(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length,
    iree_hal_remote_resource_id_t* out_resource_id, uint64_t* out_byte_offset,
    uint64_t* out_byte_length);

// Returns a snapshot of the resource_id from a remote client buffer proxy.
// Handles subspan buffers by traversing to the root allocation. A provisional
// ID may become canonical concurrently; either identity is valid on the wire.
static inline iree_hal_remote_resource_id_t
iree_hal_remote_client_buffer_resource_id(iree_hal_buffer_t* buffer) {
  iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
  if (allocated) buffer = allocated;
  return (iree_hal_remote_resource_id_t)iree_atomic_load(
      &((iree_hal_remote_client_buffer_t*)buffer)->resource_id,
      iree_memory_order_relaxed);
}

// Updates the resource_id on a buffer proxy. Used to resolve provisional
// IDs to canonical server-assigned IDs when the ADVANCE frame arrives.
// Application threads may concurrently read either valid identity while
// submitting work. The update must precede the corresponding semaphore signal
// so callers that wait for allocation completion observe the canonical ID.
static inline void iree_hal_remote_client_buffer_set_resource_id(
    iree_hal_buffer_t* buffer, iree_hal_remote_resource_id_t resolved_id) {
  iree_atomic_store(&((iree_hal_remote_client_buffer_t*)buffer)->resource_id,
                    resolved_id, iree_memory_order_relaxed);
}

// Disarms server-side resource release for a buffer whose remote allocation
// was explicitly released by another allocator API.
static inline void iree_hal_remote_client_buffer_disown_remote_resource(
    iree_hal_buffer_t* buffer) {
  iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffer);
  if (allocated) buffer = allocated;
  ((iree_hal_remote_client_buffer_t*)buffer)->owns_remote_resource = false;
  iree_atomic_store(&((iree_hal_remote_client_buffer_t*)buffer)->resource_id, 0,
                    iree_memory_order_relaxed);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_BUFFER_H_
