// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_DEVICE_POOL_H_
#define LOOM_SERVE_RUNTIME_DEVICE_POOL_H_

#include "loom/serve/runtime/execution/workspace.h"
#include "loom/serve/runtime/parameter/residency.h"

#ifdef __cplusplus
extern "C" {
#endif

// Explicit storage, admission, and reclamation owner. Consumers share policy
// by borrowing this object, independently of their execution device. The HAL
// group and allocation queue outlive the pool and all of its consumers.
typedef struct loom_serve_pool_t loom_serve_pool_t;

typedef enum loom_serve_pool_backing_e {
  // Ordinary fixed allocations, including device address sanitizer use.
  LOOM_SERVE_POOL_BACKING_FIXED = 0,
  // Stable virtual roots with independently reclaimable physical slabs.
  LOOM_SERVE_POOL_BACKING_ELASTIC = 1,
} loom_serve_pool_backing_t;

// Application-selected identity and complete set of consumers. Device bits
// index the immutable application group, not native driver ordinals. Several
// pools may name the same devices without sharing storage or admission policy.
typedef struct loom_serve_pool_binding_t {
  // Application key, borrowed from its owner through pool retirement.
  iree_string_view_t name;
  // Nonempty set of device records allowed to execute against this pool.
  uint64_t device_mask;
} loom_serve_pool_binding_t;

typedef enum loom_serve_pool_placement_kind_e {
  // Let the HAL choose among routes satisfying the complete access scope.
  LOOM_SERVE_POOL_PLACEMENT_AUTOMATIC = 0,
  // Require the allocation domain of one group device's physical device.
  LOOM_SERVE_POOL_PLACEMENT_DEVICE = 1,
  // Require initial allocation on an operating-system host NUMA node.
  LOOM_SERVE_POOL_PLACEMENT_HOST_NUMA = 2,
} loom_serve_pool_placement_kind_t;

// Application coordinate resolved once against the sealed HAL group. The
// selected allocation domain is independent of the pool's consumer membership.
typedef struct loom_serve_pool_placement_t {
  // Domain selector; zero retains ordinary automatic placement.
  loom_serve_pool_placement_kind_t kind;
  // Group device index for DEVICE, OS node number for HOST_NUMA; ignored for
  // AUTOMATIC. DEVICE requires an unambiguous physical device in that record.
  uint32_t ordinal;
} loom_serve_pool_placement_t;

typedef struct loom_serve_pool_options_t {
  // Stable virtual roots with independently reclaimable physical backing.
  bool elastic;
  // Preferred parameter and workspace slab extent, in bytes.
  iree_device_size_t weight_slab_size;
  // Preferred sparse-state slab extent, in bytes.
  iree_device_size_t state_slab_size;
  // Elastic physical-backing allowance in bytes; zero uses source capacity.
  // An explicit allowance can tighten that capacity, not increase it.
  // Ordinary native storage is accounted separately and remains outside it.
  uint64_t memory_limit;
  // Fixed workspace sanitizer policy, qualified by the HAL child allocator.
  iree_hal_asan_pool_options_t asan;
  // Required backing domain for both ordinary and elastic physical sources.
  loom_serve_pool_placement_t placement;
} loom_serve_pool_options_t;

// Qualifies the complete simultaneous access scope before publishing an owner.
// Creates metadata and native sources, not model payload. All mutable pool and
// residency operations use one serialized host owner. Physical placement and
// execution ordering are not inferred from the selected allocation queue.
iree_status_t loom_serve_pool_allocate(iree_hal_device_group_t* group,
                                       iree_hal_pool_scope_t scope,
                                       iree_hal_queue_t* allocation_queue,
                                       const loom_serve_pool_options_t* options,
                                       iree_allocator_t host_allocator,
                                       loom_serve_pool_t** out_pool);

// True while registered models, parameter banks, retained output, or elastic
// reservations still borrow the owner. The pool's own idle workspace ranges
// are excluded. NULL is unused. Called on the serialized owner.
bool loom_serve_pool_is_in_use(const loom_serve_pool_t* pool);

// Releases a retired owner. Live consumers or failed native cleanup
// preserve remaining ownership for the caller. NULL is permitted. All model,
// execution, and result consumers retire before the caller destroys the pool.
iree_status_t loom_serve_pool_free(loom_serve_pool_t* pool);

// Serving-managed backing across every consumer of this owner. Storage and
// retained have separate native sources; workspace is a breakdown of storage
// for fixed backing or retained for elastic backing, not additional physical
// consumption. Virtual statistics are zero for fixed backing. Native statistics
// are independent atomic observations, not a coherent snapshot. Commitments
// exclude host snapshots, loader storage, and driver overhead; HAL providers
// may also hide physical allocation padding.
typedef struct loom_serve_pool_statistics_t {
  // Ordinary native buffers, including fixed parameters, state, and workspace.
  iree_hal_pool_stats_t storage;
  // All virtual reservations and backing, including weights, state, workspace.
  loom_serve_memory_statistics_t retained;
  // Shared command workspace, including reusable idle backing.
  iree_hal_pool_stats_t workspace;
} loom_serve_pool_statistics_t;

// Called by the host owner; copies observations without waiting for device
// work.
loom_serve_pool_statistics_t loom_serve_pool_statistics(
    const loom_serve_pool_t* pool);

// Cold owner-controlled trim toward target_bytes of managed backing. Releases
// idle workspace first, then unpinned parameters in LRU order. Live workspace,
// pinned parameters, and mutable model state survive. An unreachable target
// is not an error; statistics expose the reached extent. Native returns can
// remain visible until their queued sweep retires. Does not wait for live
// work, destroy queues, or trim any execution device's compiled products.
// Fixed parameter and state allocations remain until their model releases them.
iree_status_t loom_serve_pool_trim(loom_serve_pool_t* pool,
                                   uint64_t target_bytes);

// Independent idle policies applied by the serialized serving owner. Zero
// disables a policy; both durations must be nonnegative. Mutable model state,
// sessions, and prefix history are outside these policies.
typedef struct loom_serve_pool_idle_policy_t {
  // Idle interval after the last command workspace root retires.
  iree_duration_t workspace_timeout;
  // Idle interval after the last invocation or explicit parameter pin releases.
  // Only elastic parameter backing can be deactivated independently.
  iree_duration_t parameter_timeout;
} loom_serve_pool_idle_policy_t;

// Schedules expired backing releases without waiting for native deallocation.
// Returns pending while enabled resources await retirement or idle expiry;
// native releases already scheduled use the ordinary capacity observer.
// Live invocation pins and explicit parameter pins prevent deactivation.
iree_status_t loom_serve_pool_trim_idle(loom_serve_pool_t* pool,
                                        loom_serve_pool_idle_policy_t policy,
                                        iree_time_t now, bool* out_pending);

// Borrowed resources outliving every model, execution, and output using them.
iree_hal_pool_t* loom_serve_pool_storage(const loom_serve_pool_t* pool);
loom_serve_workspace_t* loom_serve_pool_workspace(
    const loom_serve_pool_t* pool);
loom_serve_memory_pool_t* loom_serve_pool_memory(const loom_serve_pool_t* pool);
loom_serve_memory_geometry_t loom_serve_pool_weight_geometry(
    const loom_serve_pool_t* pool);
loom_serve_memory_geometry_t loom_serve_pool_state_geometry(
    const loom_serve_pool_t* pool);
loom_serve_residency_cache_t* loom_serve_pool_residency_cache(
    const loom_serve_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_DEVICE_POOL_H_
