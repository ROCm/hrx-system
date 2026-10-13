// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_DEVICE_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_DEVICE_H_

#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/runtime/host/host.h"
#include "experimental/loom_serve/runtime/residency.h"
#include "experimental/loom_serve/storage/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// Application owner of all cooperating devices, one immutable HAL topology and
// frontier namespace, profiling session and retained host services. Models
// drain before this owner is destroyed. Group membership creates no execution
// edges and implies no cross-device buffer visibility.
typedef struct loom_serve_device_group_t loom_serve_device_group_t;

// Borrowed device record within the application group. Its queues, workspace,
// retained-memory budget and residency cache are independent of other records.
// One task owner serializes model calls and physical residency transitions;
// models sharing this record share its allocation budget, not their timelines.
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

// Creates each driver once and every requested HAL device, seals one group,
// then constructs each record's pools and selects its first provisioned
// dispatch/transfer queues.
// Missing roles fail creation. device_count must be nonzero and within the HAL
// topology limit; options are borrowed only during creation. This establishes
// runtime ownership, not compiler support or cross-device tensor access.
// CPU task devices share the host executor instead of creating private workers.
// Retains host through all device and asynchronous service lifetimes. Failure
// returns no group and releases partial state, including earlier devices.
iree_status_t loom_serve_device_group_create(
    loom_serve_host_t* host, iree_host_size_t device_count,
    const loom_serve_device_options_t* options,
    loom_serve_device_group_t** out_group, iree_allocator_t host_allocator);

// Releases all device resources and the host reference without an implicit
// device wait. The caller first drains every model domain and releases all
// model/JIT/VM resources while host I/O remains alive. NULL is permitted.
// Any outstanding virtual reservation preserves the entire group and returns
// failure before partial teardown can invalidate another device's resources.
iree_status_t loom_serve_device_group_destroy(loom_serve_device_group_t* group);

// Number of immutable device records owned by the group.
iree_host_size_t loom_serve_device_group_count(
    const loom_serve_device_group_t* group);

// Borrows a record through group destruction. device_index is a caller-resolved
// ordinal less than the group's count.
loom_serve_device_t* loom_serve_device_group_at(
    loom_serve_device_group_t* group, iree_host_size_t device_index);

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

// Borrowed handles valid through group destruction. Every device record returns
// the same HAL group; execution domains retain their own exact queues.
iree_hal_device_t* loom_serve_device_handle(const loom_serve_device_t* device);
// Borrowed native source for retained model buffers. Access is qualified for
// this record's dispatch and transfer families, without public host mappings.
// All allocated buffers must retire before the group is destroyed. Command
// transients use the execution workspace instead of this unbounded source.
iree_hal_pool_t* loom_serve_device_storage_pool(
    const loom_serve_device_t* device);
// Borrowed application host, which outlives this device and its models.
loom_serve_host_t* loom_serve_device_host(const loom_serve_device_t* device);
iree_hal_device_group_t* loom_serve_device_group(
    const loom_serve_device_t* device);
iree_hal_queue_t* loom_serve_device_dispatch_queue(
    const loom_serve_device_t* device);
iree_hal_queue_t* loom_serve_device_transfer_queue(
    const loom_serve_device_t* device);
// Creates an owned model timeline over this device's shared workspace. It must
// be drained and released before the device owner. Sharing physical backing
// introduces no dependency edges between models.
iree_status_t loom_serve_device_create_execution(
    loom_serve_device_t* device, loom_serve_execution_t** out_execution);
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
