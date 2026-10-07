// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_DEVICE_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_DEVICE_H_

#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/runtime/residency.h"
#include "experimental/loom_serve/storage/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// One serving device, its async I/O services, semaphore domain and ordered
// execution timelines. One host owner serializes model calls and physical
// residency transitions. Model programs and logical state remain independent;
// their virtual reservations share this owner's physical allocation budget.
typedef struct loom_serve_device_t loom_serve_device_t;

typedef enum loom_serve_device_backing_e {
  // Ordinary fixed allocations, including device address sanitizer use.
  LOOM_SERVE_DEVICE_BACKING_FIXED = 0,
  // Stable virtual roots with independently reclaimable physical slabs.
  LOOM_SERVE_DEVICE_BACKING_ELASTIC = 1,
} loom_serve_device_backing_t;

typedef struct loom_serve_device_options_t {
  // HAL device URI, borrowed during creation.
  iree_string_view_t uri;
  // Physical backing strategy for all models sharing this owner.
  loom_serve_device_backing_t backing;
  // Physical slab bytes; zero selects the allocator recommendation.
  iree_device_size_t slab_size;
  // Maximum committed virtual-pool bytes; zero imposes no serving limit.
  uint64_t memory_limit;
} loom_serve_device_options_t;

// Creates the device identified by a HAL URI and selects the first provisioned
// queue supporting each of dispatch and transfer. Missing roles fail creation.
// This establishes runtime ownership, not compiler support for the device.
// On failure no owner is returned and all partial resources are released.
iree_status_t loom_serve_device_create(
    const loom_serve_device_options_t* options,
    loom_serve_device_t** out_device, iree_allocator_t host_allocator);

// Releases the device and then its async services, without an implicit wait.
// The caller first drains execution while all borrowed host I/O is alive and
// releases all model/JIT/VM resources referencing this domain. Those resources
// must not outlive the owner. NULL is permitted.
// Profiling belongs to the device, not individual models. Outstanding virtual
// reservations preserve the owner and return failure rather than dangling a
// physical domain after a model cleanup failure.
iree_status_t loom_serve_device_destroy(loom_serve_device_t* device);

// Serving-managed backing across every model sharing this device. Virtual
// statistics are zero for fixed backing. Workspace statistics are atomic HAL
// observations, not a coherent device-wide snapshot. The sum of commitments
// excludes host snapshots, fixed buffers, loader storage and driver overhead;
// HAL providers may also hide physical allocation padding.
typedef struct loom_serve_device_memory_statistics_t {
  // Virtual parameter/state reservations and their physical slab backing.
  loom_serve_memory_statistics_t retained;
  // Shared command workspace, including reusable idle backing.
  iree_hal_pool_stats_t workspace;
} loom_serve_device_memory_statistics_t;

// Called by the host owner; copies observations without waiting for device
// work.
loom_serve_device_memory_statistics_t loom_serve_device_memory_statistics(
    const loom_serve_device_t* device);

// Cold owner-controlled trim toward target_bytes of reported managed backing.
// Releases idle workspace first, then unpinned parameter groups in LRU order.
// Live workspace, pinned parameters and model-owned mutable state survive.
// An unreachable target is not an error; statistics expose the reached extent.
// Consumers pin parameter groups through retirement; only those already
// retired are eligible. Does not wait for live workspace or destroy queues.
// Platform unmap/free failures propagate to the owner.
iree_status_t loom_serve_device_trim(loom_serve_device_t* device,
                                     uint64_t target_bytes);

// Borrowed handles valid through owner destruction. The group establishes the
// semaphore namespace; the exact queues are also used by the execution object.
iree_hal_device_t* loom_serve_device_handle(const loom_serve_device_t* device);
iree_hal_device_group_t* loom_serve_device_group(
    const loom_serve_device_t* device);
iree_hal_queue_t* loom_serve_device_dispatch_queue(
    const loom_serve_device_t* device);
iree_hal_queue_t* loom_serve_device_transfer_queue(
    const loom_serve_device_t* device);
loom_serve_execution_t* loom_serve_device_execution(
    const loom_serve_device_t* device);
// Borrowed shared physical owner, or NULL for explicitly fixed backing.
loom_serve_memory_pool_t* loom_serve_device_memory_pool(
    const loom_serve_device_t* device);
// Borrowed parameter admission, pinning and LRU policy shared by all models.
loom_serve_residency_cache_t* loom_serve_device_residency_cache(
    const loom_serve_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_DEVICE_H_
