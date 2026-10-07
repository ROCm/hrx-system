// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Proactor pool: a process-level factory for NUMA-pinned proactors.
//
// Creates proactors on-demand per NUMA node, allowing HAL devices, network
// sessions, and other subsystems to share I/O infrastructure with proper NUMA
// locality. The aggregate pool and each initialized entry are independently
// reference counted. Devices that may select proactors throughout their
// lifetime retain the pool; consumers that need one proactor/runner pair can
// retain only that entry.
//
// Entries, proactors, and runners are created lazily when a slot is first
// accessed. Pool creation performs one allocation for the pool and its
// lightweight slot table but creates no OS resources or threads.
//
// ## Poll runners
//
// Proactors are caller-driven: they only make progress when poll() is called.
// The pool supports an optional runner factory that creates a poll runner for
// each proactor on-demand. The standard runner creates a dedicated poll thread
// (see proactor_thread_runner.h). On platforms without C threads (wasm), the
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
// A subsystem that selects one proactor and then releases the aggregate pool
// must acquire the entry so that automatic progress remains alive:
//
//   iree_async_proactor_pool_entry_t* entry = NULL;
//   IREE_RETURN_IF_ERROR(
//       iree_async_proactor_pool_acquire(pool, 0, &entry));
//   iree_async_proactor_t* proactor =
//       iree_async_proactor_pool_entry_proactor(entry);
//   iree_async_proactor_pool_release(pool);
//   // Use proactor while entry remains retained...
//   iree_async_proactor_pool_entry_release(entry);
//
// ## NUMA mapping
//
// A specified node applies to both backend-owned workers and the poll runner.
// The default thread runner establishes affinity before polling; lazy entry
// creation fails if affinity cannot be established. Custom runner factories
// must honor the node, or report failure. Caller-driven polling is responsible
// for establishing its own affinity. NULL node_ids or UINT32_MAX entries leave
// runner placement unspecified. These constraints do not bind allocations.

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
  // Options applied to each proactor created by the pool. An explicit entry
  // node overrides worker_affinity with that node's CPU affinity.
  iree_async_proactor_options_t proactor_options;

  // Optional proactor creator. NULL selects the platform-optimal creator.
  // A caller can provide an existing backend creator when its workload has a
  // stronger execution-policy requirement than the platform default.
  iree_async_proactor_pool_proactor_create_fn_t proactor_create;

  // Optional runner factory for creating poll runners that drive proactors.
  // When create is non-NULL, the pool calls it for each proactor during
  // first access. When create is NULL (zero-initialized), proactors are
  // created without a runner and the caller is responsible for polling.
  // The callbacks and user_data are copied into initialized entries and must
  // remain valid until all entries acquired from the pool are released.
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
// No entries, proactors, or runners are created during pool creation. They are
// created on-demand when a slot is first accessed. |node_count| must be >= 1.
//
// If |node_ids| is non-NULL, it must point to |node_count| NUMA node IDs.
// When a runner is created on-demand, the node ID is passed to the runner
// factory and backend workers for NUMA-aware pinning. If |node_ids| is NULL,
// runner placement is unconstrained (suitable for single-node systems).
//
// The pool owns one reference to every created entry. Releasing the pool drops
// those references and stops entries with no other owners. Entries retained by
// consumers continue running until their own final release.
iree_status_t iree_async_proactor_pool_create(
    iree_host_size_t node_count, const uint32_t* node_ids,
    iree_async_proactor_pool_options_t options, iree_allocator_t allocator,
    iree_async_proactor_pool_t** out_pool);

// Retains a reference to the pool.
void iree_async_proactor_pool_retain(iree_async_proactor_pool_t* pool);

// Releases a reference to the pool. When the count reaches zero, all
// pool-owned entry references are released and the pool is freed. Consumer-
// retained entries remain alive.
void iree_async_proactor_pool_release(iree_async_proactor_pool_t* pool);

// Returns the number of proactor slots in the pool.
iree_host_size_t iree_async_proactor_pool_count(
    const iree_async_proactor_pool_t* pool);

// Acquires the pool entry at dense |index|, creating its proactor and runner
// on-demand if this is the first access for that slot.
//
// The returned entry is retained and must be released with
// iree_async_proactor_pool_entry_release(). Retaining the entry keeps both the
// proactor and its pool-created runner alive after the aggregate pool is
// released.
iree_status_t iree_async_proactor_pool_acquire(
    iree_async_proactor_pool_t* pool, iree_host_size_t index,
    iree_async_proactor_pool_entry_t** out_entry);

// Returns the proactor at the given dense |index| (0-based), creating it
// on-demand if this is the first access for that index. The proactor and its
// runner (if the factory is set) are created lazily.
//
// The returned proactor is borrowed from its entry. The caller must retain the
// aggregate pool or acquire the entry for as long as automatic progress is
// required. Retaining only the proactor does not retain its runner.
iree_status_t iree_async_proactor_pool_get(
    iree_async_proactor_pool_t* pool, iree_host_size_t index,
    iree_async_proactor_t** out_proactor);

// Returns the NUMA node ID for the proactor at |index|, or UINT32_MAX if no
// node ID was specified during creation.
uint32_t iree_async_proactor_pool_node_id(
    const iree_async_proactor_pool_t* pool, iree_host_size_t index);

// Acquires the entry associated with |node_id|, creating it on-demand if this
// is the first access for that node.
//
// A concrete node requires an exact match; otherwise returns NOT_FOUND and
// clears the output. UINT32_MAX selects the first entry without a locality
// constraint. Release the returned entry with
// iree_async_proactor_pool_entry_release().
iree_status_t iree_async_proactor_pool_acquire_for_node(
    iree_async_proactor_pool_t* pool, uint32_t node_id,
    iree_async_proactor_pool_entry_t** out_entry);

// Returns the proactor associated with the given NUMA |node_id|, creating it
// on-demand if this is the first access for that node.
//
// A concrete node requires an exact match; otherwise returns NOT_FOUND and
// clears the output. UINT32_MAX selects the first entry without a locality
// constraint. An entry with unspecified placement does not match a concrete
// node request.
//
// The returned proactor is borrowed from its entry. The caller must retain the
// aggregate pool or acquire the entry for as long as automatic progress is
// required. Retaining only the proactor does not retain its runner.
iree_status_t iree_async_proactor_pool_get_for_node(
    iree_async_proactor_pool_t* pool, uint32_t node_id,
    iree_async_proactor_t** out_proactor);

// Retains |entry| for the caller.
void iree_async_proactor_pool_entry_retain(
    iree_async_proactor_pool_entry_t* entry);

// Releases an entry reference. The final release requests its runner to stop,
// relinquishes runner ownership, releases the proactor, and frees the entry.
// An external caller waits for runner teardown. A final release from a runner
// callback defers runner teardown until the callback and poll call return.
void iree_async_proactor_pool_entry_release(
    iree_async_proactor_pool_entry_t* entry);

// Returns the proactor owned by |entry|, borrowed for the entry lifetime.
iree_async_proactor_t* iree_async_proactor_pool_entry_proactor(
    const iree_async_proactor_pool_entry_t* entry);

// Returns the NUMA node ID associated with |entry|, or UINT32_MAX if no node
// was specified for its slot.
uint32_t iree_async_proactor_pool_entry_node_id(
    const iree_async_proactor_pool_entry_t* entry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_PROACTOR_POOL_H_
