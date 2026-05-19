// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Proactor pool: a process-level factory for NUMA-pinned proactors.
//
// Creates proactors on-demand per NUMA node, allowing HAL devices, network
// sessions, and other subsystems to share I/O infrastructure with proper NUMA
// locality. The pool is ref-counted and each initialized pool entry is
// independently ref-counted. Devices that may request proactors over their
// lifetime usually retain the pool; subsystems that need only one
// proactor/runner pair can acquire that entry and release the aggregate pool.
//
// Proactors are created lazily: nothing is allocated until pool_get() or
// pool_get_for_node() is called. This makes pool creation effectively free —
// create a pool, pass it to device creation, release it. If no driver requests
// a proactor, no resources are allocated.
//
// ## Poll runners
//
// Proactors are caller-driven: they only make progress when poll() is called.
// The pool supports an optional runner factory that creates a poll runner for
// each proactor on-demand. The standard runner creates a dedicated poll thread
// (see proactor_runner_thread.h). On platforms without C threads (wasm), the
// host event loop drives polling and no runner is needed.
//
// The default options (iree_async_proactor_pool_options_default) select the
// appropriate runner for the platform: threaded on native, none on wasm.
//
// ## Typical usage
//
//   // Create pool and pass to device creation.
//   iree_async_proactor_pool_t* pool = NULL;
//   IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
//       iree_numa_node_count(), /*node_ids=*/NULL,
//       iree_async_proactor_pool_options_default(),
//       allocator, &pool));
//
//   // Device retains the pool — caller can release immediately.
//   iree_hal_device_create_params_t create_params =
//       iree_hal_device_create_params_default();
//   create_params.proactor_pool = pool;
//   IREE_RETURN_IF_ERROR(iree_hal_driver_create_default_device(
//       driver, &create_params, allocator, &device));
//   iree_async_proactor_pool_release(pool);
//
//   // At shutdown: releasing the device releases the pool (and runners).
//   iree_hal_device_release(device);
//
// If a subsystem stores only one proactor and relies on the default poll
// runner, use iree_async_proactor_pool_acquire() and retain the returned entry
// for as long as automatic progress is required. Retaining the bare proactor
// keeps the proactor object alive but does not keep a pool-owned runner alive.
//
// ## NUMA mapping
//
// When |node_ids| are provided, each proactor's runner is pinned to the
// corresponding NUMA node (if the runner supports affinity). When |node_ids|
// is NULL, all runners get unspecified affinity (OS chooses). On single-node
// systems (node_count=1), the pool degenerates to a single proactor — this is
// the common case and works well.

#ifndef IREE_ASYNC_UTIL_PROACTOR_POOL_H_
#define IREE_ASYNC_UTIL_PROACTOR_POOL_H_

#include "iree/async/proactor.h"
#include "iree/async/util/proactor_pool_types.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_async_proactor_pool_t
//===----------------------------------------------------------------------===//

// Options for configuring proactor pool creation.
typedef struct iree_async_proactor_pool_options_t {
  // Options applied to each proactor created by the pool.
  iree_async_proactor_options_t proactor_options;

  // Optional proactor creator. NULL selects the platform-optimal creator.
  // A caller can provide an existing backend creator when its workload has a
  // stronger execution-policy requirement than the platform default.
  iree_async_proactor_pool_proactor_create_fn_t proactor_create;

  // Optional runner factory for creating poll runners that drive proactors.
  // When create is non-NULL, the pool calls it for each proactor during
  // pool_get(). When create is NULL (zero-initialized), proactors are created
  // without a runner and the caller is responsible for polling.
  iree_async_proactor_pool_runner_factory_t runner;
} iree_async_proactor_pool_options_t;

// Returns default pool options appropriate for the current platform.
// On native platforms: creates a threaded poll runner per proactor.
// On wasm: no runner (the JS event loop drives polling).
iree_async_proactor_pool_options_t iree_async_proactor_pool_options_default(
    void);

typedef struct iree_async_proactor_pool_t iree_async_proactor_pool_t;
typedef struct iree_async_proactor_pool_entry_t
    iree_async_proactor_pool_entry_t;

// Creates a pool with capacity for |node_count| proactors.
//
// No proactors or runners are created during pool creation — they are created
// on-demand when pool_get() or pool_get_for_node() is first called for each
// entry. This makes pool creation effectively free. |node_count| must be >= 1.
//
// If |node_ids| is non-NULL, it must point to |node_count| NUMA node IDs.
// When a runner is created on-demand, the node ID is passed to the runner
// factory for NUMA-aware pinning. If |node_ids| is NULL, runners get no
// affinity hint (suitable for single-node systems).
//
// The pool retains all created proactors and runners. Releasing the pool (when
// the ref count reaches zero) releases its references to all entries. Entries
// retained by consumers continue running until their own final release.
iree_status_t iree_async_proactor_pool_create(
    iree_host_size_t node_count, const uint32_t* node_ids,
    iree_async_proactor_pool_options_t options, iree_allocator_t allocator,
    iree_async_proactor_pool_t** out_pool);

// Retains a reference to the pool.
void iree_async_proactor_pool_retain(iree_async_proactor_pool_t* pool);

// Releases a reference to the pool. When the count reaches zero, the pool drops
// its entry references and is freed. Entries retained by consumers remain
// alive.
void iree_async_proactor_pool_release(iree_async_proactor_pool_t* pool);

// Returns the number of proactors in the pool.
iree_host_size_t iree_async_proactor_pool_count(
    const iree_async_proactor_pool_t* pool);

// Acquires the pool entry at the given dense |index| (0-based), creating its
// proactor and runner on-demand if this is the first access for that index.
//
// The returned entry is retained and must be released with
// iree_async_proactor_pool_entry_release(). Retaining the entry keeps the
// proactor and pool-owned runner alive even after the aggregate pool is
// released.
iree_status_t iree_async_proactor_pool_acquire(
    iree_async_proactor_pool_t* pool, iree_host_size_t index,
    iree_async_proactor_pool_entry_t** out_entry);

// Returns the proactor at the given dense |index| (0-based), creating it
// on-demand if this is the first access for that index. The proactor and its
// runner (if the factory is set) are created lazily.
//
// The returned proactor is borrowed from the pool entry. Retain the pool or
// acquire the entry if automatic progress must continue after this call.
iree_status_t iree_async_proactor_pool_get(
    iree_async_proactor_pool_t* pool, iree_host_size_t index,
    iree_async_proactor_t** out_proactor);

// Returns the NUMA node ID for the proactor at |index|, or UINT32_MAX if no
// node ID was specified during creation.
uint32_t iree_async_proactor_pool_node_id(
    const iree_async_proactor_pool_t* pool, iree_host_size_t index);

// Acquires the entry associated with the given NUMA |node_id|, creating it
// on-demand if this is the first access for that node.
//
// If an exact match exists, returns that entry. If no exact match is found
// (e.g., the pool was created for a subset of nodes), returns the first
// entry in the pool as a fallback.
//
// The returned entry is retained and must be released with
// iree_async_proactor_pool_entry_release().
iree_status_t iree_async_proactor_pool_acquire_for_node(
    iree_async_proactor_pool_t* pool, uint32_t node_id,
    iree_async_proactor_pool_entry_t** out_entry);

// Returns the proactor associated with the given NUMA |node_id|, creating it
// on-demand if this is the first access for that node.
//
// If an exact match exists, returns that proactor. If no exact match is found
// (e.g., the pool was created for a subset of nodes), returns the first
// proactor in the pool as a fallback.
//
// The returned proactor is borrowed from the pool entry. Retain the pool or
// acquire the entry if automatic progress must continue after this call.
iree_status_t iree_async_proactor_pool_get_for_node(
    iree_async_proactor_pool_t* pool, uint32_t node_id,
    iree_async_proactor_t** out_proactor);

// Retains a reference to a pool entry.
void iree_async_proactor_pool_entry_retain(
    iree_async_proactor_pool_entry_t* entry);

// Releases a reference to a pool entry. When the count reaches zero, the
// runner is stopped, the proactor is released, and the entry is freed.
void iree_async_proactor_pool_entry_release(
    iree_async_proactor_pool_entry_t* entry);

// Returns the proactor owned by |entry|. Borrowed for the lifetime of |entry|.
iree_async_proactor_t* iree_async_proactor_pool_entry_proactor(
    iree_async_proactor_pool_entry_t* entry);

// Returns the NUMA node ID associated with |entry|.
uint32_t iree_async_proactor_pool_entry_node_id(
    const iree_async_proactor_pool_entry_t* entry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_PROACTOR_POOL_H_
