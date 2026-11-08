// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_DEVICE_H_
#define LIBHRX_SRC_BINDING_COMMON_DEVICE_H_

#include "common/context.h"
#include "common/execution_resource.h"
#include "common/host_callback.h"
#include "common/tls.h"
#include "hrx_runtime.h"
#include "iree/base/internal/arena.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Device types
//===----------------------------------------------------------------------===//

// Maximum number of devices supported by the stream HAL.
// This avoids dynamic enumeration overhead during initialization.
#define IREE_HAL_STREAMING_MAX_DEVICES 64

// Device registry entry for multi-device support.
typedef struct iree_hal_streaming_device_t {
  // Device ordinal in the global registry.
  iree_host_size_t ordinal;

  // HRX device handle (owns the HAL device and driver).
  hrx_device_t hrx_device;

  // HAL device extracted from hrx_device for direct HAL calls.
  // Streaming is always built from the same source tree as libhrx and
  // shares internal representations. Accessed via hrx_device_hal().
  iree_hal_device_t* hal_device;
  iree_hal_device_info_t info;

  // Borrowed callout owner shared by every device in this frontend registry.
  iree_hal_streaming_host_callback_queue_t* host_callbacks;

  // Immutable execution-resource sets interned for copied compatibility API
  // values. Entries live until this device incarnation is deinitialized.
  iree_hal_streaming_execution_resource_table_t execution_resource_table;

  // Device capabilities.
  uint32_t compute_capability_major;
  uint32_t compute_capability_minor;
  // Total HIP-visible memory reported for the device.
  iree_device_size_t total_memory;
  // Approximate HIP-visible free memory tracked atomically by the binding.
  iree_atomic_uint64_t free_memory;
  // True when cooperative launches are supported by the device.
  bool supports_cooperative_launch;

  // GCN architecture name (e.g., "gfx942:sramecc+:xnack-").
  char gcn_arch_name[64];

  // Device properties cache.
  uint32_t max_threads_per_block;
  uint32_t max_block_dim[3];
  uint32_t max_grid_dim[3];
  uint32_t warp_size;
  uint32_t multiprocessor_count;

  // Occupancy calculation properties.
  uint32_t max_threads_per_multiprocessor;
  uint32_t max_blocks_per_multiprocessor;
  uint32_t max_registers_per_multiprocessor;
  uint32_t max_shared_memory_per_multiprocessor;
  uint32_t max_registers_per_block;
  // Default shared-memory capacity available to one block.
  uint32_t max_shared_memory_per_block;
  // Maximum shared-memory capacity available to an opted-in block.
  uint32_t max_shared_memory_per_block_optin;

  // Arena block pool for transient host allocations.
  // Shared by all graphs created from this device.
  iree_arena_block_pool_t block_pool;

  // Primary context flags.
  iree_hal_streaming_context_flags_t primary_context_flags;

  // Serializes primary-context publication and allocation-pool selection.
  iree_slim_mutex_t primary_context_mutex;

  // Fully initialized primary context, published under primary_context_mutex.
  iree_hal_streaming_context_t* primary_context;

  // Primary context reference count.
  // When > 0, the primary context is retained and must not be destroyed.
  // When reaches 0, the primary context is destroyed.
  // Protected by primary_context_mutex.
  int32_t primary_context_ref_count;

  // Default device allocation pool, protected by primary_context_mutex.
  hrx_mem_pool_t default_mem_pool;
  // Current device allocation pool, protected by primary_context_mutex.
  hrx_mem_pool_t current_mem_pool;

  // Guards graph-memory accounting and cached physical blocks.
  iree_slim_mutex_t graph_memory_mutex;
  // Logical bytes in accepted live graph allocations.
  uint64_t graph_memory_used_current;
  // High-water graph-memory bytes visible via hipGraphMemAttrUsedMemHigh.
  uint64_t graph_memory_used_high;
  // Physical extents retained for accepted allocations and cached reuse.
  uint64_t graph_memory_reserved_current;
  // High-water graph-memory reservation visible via
  // hipGraphMemAttrReservedMemHigh.
  uint64_t graph_memory_reserved_high;
  // Unmapped physical graph blocks, reusable after their captured timepoints.
  struct iree_hal_streaming_graph_memory_block_t* graph_memory_blocks;
} iree_hal_streaming_device_t;

// Global device registry for multi-device management.
typedef struct iree_hal_streaming_device_registry_t {
  // Host allocator for internal allocations.
  iree_allocator_t host_allocator;

  // Immutable HAL device-creation extension chain selected at initialization.
  const iree_hal_device_create_params_extension_t* device_extensions;

  // Global initialization state.
  bool initialized;

  // Thread-exit cleanup for current and stacked context references.
  iree_hal_streaming_tls_key_t context_key;

  // Ready compatibility callouts, with one lazily created frontend worker.
  iree_hal_streaming_host_callback_queue_t host_callbacks;

  iree_slim_mutex_t mutex;

  // Fixed-size array of registered devices.
  iree_hal_streaming_device_t devices[IREE_HAL_STREAMING_MAX_DEVICES];
  iree_host_size_t device_count;

  // Global context tracking for cleanup.
  // All created contexts are tracked here to ensure proper cleanup.
  struct {
    iree_slim_mutex_t mutex;
    iree_hal_streaming_context_t* head;
    iree_hal_streaming_context_t* tail;
  } context_list;
} iree_hal_streaming_device_registry_t;

// Accessor for the global device registry.
// Synchronization: none (read-only access).
iree_hal_streaming_device_registry_t* iree_hal_streaming_device_registry(void);

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

// Synchronization: none (queries static device count).
iree_status_t iree_hal_streaming_device_count(iree_host_size_t* out_count);

// Returns the borrowed entry for |ordinal|, or NULL if the registry is not
// initialized or the ordinal is out of range.
// Synchronization: none (queries immutable registry membership).
iree_hal_streaming_device_t* iree_hal_streaming_device_entry(
    iree_hal_streaming_device_ordinal_t ordinal);

// Selects the borrowed provisioned queue defining the device's primary
// compatibility execution domain. Dynamic domains acquire queues from the same
// family. |out_queue| is unchanged on failure.
// Synchronization: none (queries immutable device facts).
iree_status_t iree_hal_streaming_device_select_primary_queue(
    iree_hal_streaming_device_t* device, iree_hal_queue_t** out_queue);

// Streaming submits primarily from the CPU. Selects HOST_LOCAL when the
// family exposes it, or DEFAULT when host-local placement is not selectable.
iree_hal_queue_placement_t iree_hal_streaming_queue_family_select_placement(
    const iree_hal_queue_family_t* family);

// Acquires the primary streaming queue, preferring host-local command storage.
// Reuses a suitable provisioned queue; otherwise acquires from its family when
// dynamic acquisition supports the preference. The caller owns one reference
// and keeps the device alive. Allocation failures propagate without fallback.
iree_status_t iree_hal_streaming_device_acquire_primary_queue(
    iree_hal_streaming_device_t* device, iree_hal_queue_t** out_queue);

// Acquires a bulk-transfer queue matching |primary_queue|'s family, placement,
// priority, features, and execution resources. Reuses another provisioned queue
// before acquiring dynamically. A static family with no compatible secondary
// retains its primary queue. The caller owns one reference and keeps the device
// alive; native allocation failures propagate without changing the policy.
iree_status_t iree_hal_streaming_device_acquire_transfer_queue(
    iree_hal_streaming_device_t* device, iree_hal_queue_t* primary_queue,
    iree_hal_queue_t** out_queue);

// Synchronization: none (queries device properties).
iree_status_t iree_hal_streaming_device_name(
    iree_hal_streaming_device_ordinal_t ordinal, char* name,
    iree_host_size_t name_size);

// Queries a string-valued device property owned by the streaming layer.
// Supported (category, key) pairs:
//   ("hal.device", "name")         -> device display name.
//   ("hal.device", "path")         -> HAL device path (architecture).
//   ("hal.device", "architecture") -> GCN/gfx architecture name.
// Returns IREE_STATUS_NOT_FOUND for unknown category/key pairs, or
// IREE_STATUS_OUT_OF_RANGE if |value_size| is too small to hold the property
// (including the null terminator).
iree_status_t iree_hal_streaming_device_get_string_property(
    iree_hal_streaming_device_ordinal_t ordinal, const char* category,
    const char* key, char* value, iree_host_size_t value_size);

// Queries the binding's approximate free memory and immutable total memory.
// Both outputs are zero on failure.
// Synchronization: relaxed atomic load of the free-memory estimate.
iree_status_t iree_hal_streaming_device_memory_info(
    iree_hal_streaming_device_ordinal_t ordinal,
    iree_device_size_t* out_free_memory, iree_device_size_t* out_total_memory);

// Synchronization: none (queries context state).
iree_status_t iree_hal_streaming_device_primary_context_state(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    iree_hal_streaming_context_flags_t* out_flags, bool* out_active);

// Gets or creates the primary context for a device (thread-safe).
// This performs lazy initialization of the primary context on first access.
// Synchronization: thread-safe (serializes initialization and publication).
iree_status_t iree_hal_streaming_device_get_or_create_primary_context(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_context_t** out_context);

// Retains the primary context, creating it if necessary, and increments its
// device-level usage count. The caller must balance the returned owning
// reference with iree_hal_streaming_device_release_primary_context.
// |out_context| is unchanged on failure.
// Synchronization: thread-safe (serializes initialization and retention).
iree_status_t iree_hal_streaming_device_retain_primary_context(
    iree_hal_streaming_device_t* device,
    iree_hal_streaming_context_t** out_context);

// Releases one primary-context reference and decrements its device-level usage
// count. Destroys the device-owned context when the count reaches zero.
// Synchronization: context (waits for idle when destroying).
iree_status_t iree_hal_streaming_device_release_primary_context(
    iree_hal_streaming_device_t* device);

// Synchronization: none (sets flags for future context creation).
iree_status_t iree_hal_streaming_device_set_primary_context_flags(
    iree_hal_streaming_device_ordinal_t device_ordinal,
    const iree_hal_streaming_context_flags_t* flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_DEVICE_H_
