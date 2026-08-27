// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_DEVICE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_DEVICE_H_

#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/api.h"
#include "iree/hal/drivers/amdxdna/async_queue.h"
#include "iree/hal/drivers/amdxdna/completion_queue.h"
#include "iree/hal/drivers/amdxdna/native.h"

struct iree_async_proactor_pool_t;
struct iree_async_proactor_t;
typedef struct iree_hal_amdxdna_device_chain_command_cache_t
    iree_hal_amdxdna_device_chain_command_cache_t;
typedef struct iree_hal_amdxdna_device_context_cache_t
    iree_hal_amdxdna_device_context_cache_t;
typedef struct iree_hal_amdxdna_device_single_command_cache_t
    iree_hal_amdxdna_device_single_command_cache_t;
typedef struct iree_hal_amdxdna_device {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // Backend allocator used for HAL buffer allocation; the underlying BOs are
  // created through the shim device/context APIs.
  iree_hal_allocator_t* device_allocator;
  iree_hal_channel_provider_t* channel_provider;
  iree_hal_device_spec_t* device_spec;
  iree_hal_device_topology_info_t topology_info;

  // Proactor pool is retained for the device lifetime (same contract as
  // iree_hal_device_create_params_t).
  iree_async_proactor_pool_t* proactor_pool;
  // Borrowed from the pool while the pool is retained.
  iree_async_proactor_t* proactor;

  // block pool used for command buffer allocations, uses a larger block size
  // since command buffers can contain inlined data
  iree_arena_block_pool_t block_pool;

  // Per-device async work queue. Defers HAL queue ops until their wait
  // semaphores are satisfied, then runs them on a single worker thread.
  iree_hal_amdxdna_async_queue_t* async_queue;
  // Dedicated native-completion worker. The async queue issues work; this queue
  // waits native submissions, runs post-completion actions, and signals HAL
  // semaphores after resources are no longer in use by the NPU.
  iree_hal_amdxdna_completion_queue_t* completion_queue;
  // Shared epoch advanced by all amdxdna worker queues for this HAL device.
  iree_atomic_uint64_t queue_epoch;

  // Default pool backend exposed via query_queue_pool_backend; required so
  // CTS pool tests can create explicit pools (passthrough/TLSF/fixed-block)
  // that share the device's slab provider + completion notification.
  iree_hal_slab_provider_t* default_slab_provider;
  iree_async_notification_t* default_pool_notification;
  iree_hal_pool_t* default_pool;

  // Frontier tracker borrowed from the runtime topology_info during
  // assign_topology_info. Used to advance the queue epoch on each successful
  // signal so pool epoch_query results stay coherent.
  iree_hal_device_topology_info_t topology_info_resolved;
  iree_async_frontier_tracker_t* frontier_tracker;
  iree_async_axis_t frontier_axis;

  iree_hal_amdxdna_native_device_t* native_device;
  iree_hal_amdxdna_native_c_device_caps_t native_caps;
  // Native hardware-context cache for control-packet bootstrap PDIs.
  // Implementation-private so HAL-facing code does not expose STL maps/locks.
  iree_hal_amdxdna_device_context_cache_t* context_cache;
  // Native parent-chain cache for module-style command chains.
  // Implementation-private and device-owned so cached command BOs cannot
  // outlive the native device/context they belong to.
  iree_hal_amdxdna_device_chain_command_cache_t* chain_command_cache;
  // Native single-dispatch cache for module-style commands.
  // This owns prepared command/control BOs keyed by the device-visible dispatch
  // signature and is destroyed before the native device. Created during device
  // construction so public dispatch never races a lazy getter.
  iree_hal_amdxdna_device_single_command_cache_t* single_command_cache;
  // Maximum slots that fit in one ERT_CMD_CHAIN exec BO (constant per device).
  // Lazily computed on first flush; the chain flush splits into this many
  // slots per submitted chain. Atomic because a multi-worker submission path
  // (which a complete HAL would have, to exploit the driver's depth-4
  // in-flight cap) could race two first-time probes; the probe is idempotent
  // (returns the same value every call on a given device) so double-probe is
  // safe, but we still need atomic load/store to avoid torn reads on the
  // 0 -> max sentinel transition.
  iree_atomic_uint32_t chain_max_slots;
  // True when creation successfully changed hardware power mode away from the
  // default and teardown should best-effort restore the default.
  bool power_mode_applied;
  // should come last; see the definition of total_size below in
  // iree_hal_amdxdna_device_create
  iree_string_view_t identifier;
} iree_hal_amdxdna_device;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Casts an opaque iree_hal_device_t* to the amdxdna implementation type.
// Verifies the vtable; returns nullptr on a foreign device. Exposed for tests
// that need to poke implementation-internal state (e.g. force chain_max_slots
// to a small value to exercise the chunking code path).
iree_hal_amdxdna_device* iree_hal_amdxdna_device_cast(
    iree_hal_device_t* base_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DEVICE_H_
