// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_TSAN_H_
#define IREE_HAL_DRIVERS_AMD_GPU_TSAN_H_

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold storage budget for the serialized workgroup-race shadow arena.
typedef struct iree_hal_amd_gpu_tsan_options_t {
  // Bytes per queue; instrumented dispatches reuse the complete arena. The
  // compiler's unsigned 32-bit byte offsets permit at most 2^32 bytes.
  iree_device_size_t shadow_size;
} iree_hal_amd_gpu_tsan_options_t;

// Initializes a 512 MiB arena per queue.
void iree_hal_amd_gpu_tsan_options_initialize(
    iree_hal_amd_gpu_tsan_options_t* out_options);

// Runtime-published TSAN configuration read by instrumented device code.
//
// Queue-scoped executable variants publish one shared shadow arena. The
// compiler sizes each workgroup record from the final dispatch LDS size.
// Instrumented dispatches execute serially within the queue; dispatch IDs
// distinguish reused generations without host mutation of the arena.
typedef struct iree_hal_amd_gpu_tsan_config_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Enabled (bit 0) and queue-shadow (bit 1) runtime facilities.
  uint32_t flags;
  // Reserved for future TSAN configuration. Must be zero.
  uint32_t reserved0;
  // Device-visible base of the queue shadow arena.
  uint64_t shadow_base;
  // Total bytes in the queue shadow arena.
  uint64_t shadow_size;
} iree_hal_amd_gpu_tsan_config_t;
static_assert(sizeof(iree_hal_amd_gpu_tsan_config_t) == 32,
              "TSAN config size is part of the device ABI");
static_assert(offsetof(iree_hal_amd_gpu_tsan_config_t, shadow_base) == 16,
              "TSAN config shadow fields must follow the fixed header");

// One queue's native shadow owner, retained by its parent device until every
// queue and executable cleanup has joined. Ordinary devices allocate no state.
typedef struct iree_hal_amd_gpu_tsan_t {
  // Native backing for the mutable queue shadow arena.
  iree_hal_amd_fabric_memory_t memory;
  // Immutable shader configuration copied into this queue's executable variant.
  iree_hal_amd_gpu_tsan_config_t config;
} iree_hal_amd_gpu_tsan_t;

// Validates the public byte budget and prepares the shader configuration
// without allocating. Addresses remain zero until initialize establishes
// backing.
iree_status_t iree_hal_amd_gpu_tsan_calculate_config(
    const iree_hal_amd_gpu_tsan_options_t* options,
    iree_hal_amd_gpu_tsan_config_t* out_config);

// Allocates and publishes one queue's shadow on the cold calling thread.
// Prefers local backing whose host writes can be published to the exact queue
// family, then system backing. Native accesses use this context's GPU address
// domain. Failure leaves
// deinitializable state; the parent joins GPU uses before deinitialization.
iree_status_t iree_hal_amd_gpu_tsan_initialize(
    const iree_hal_amd_gpu_context_t* context, uint32_t queue_family_ordinal,
    const iree_hal_amd_gpu_tsan_options_t* options,
    iree_hal_amd_gpu_tsan_t* out_tsan);

// Releases native backing after queue execution and executable cleanup join.
void iree_hal_amd_gpu_tsan_deinitialize(
    const iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_tsan_t* tsan);

typedef struct iree_hal_amd_gpu_tsan_registry_t
    iree_hal_amd_gpu_tsan_registry_t;

// Embedded in an instrumented image owner; callbacks run under the cold
// registry lock. The registry borrows clients and never retains executables.
typedef struct iree_hal_amd_gpu_tsan_client_t {
  // Registry containing this client, or NULL before registration/after removal.
  iree_hal_amd_gpu_tsan_registry_t* registry;
  // Previous live image, for constant-time removal during final release.
  struct iree_hal_amd_gpu_tsan_client_t* previous;
  // Next live image.
  struct iree_hal_amd_gpu_tsan_client_t* next;
  // Borrowed image owner, valid through removal.
  void* user_data;
  // Prepares the image's native placement before a queue can accept work.
  iree_status_t (*prepare)(void* user_data, uint32_t ordinal,
                           const iree_hal_amd_gpu_tsan_config_t* config);
  // Detaches a placement after native use joins; absent placements are a no-op.
  void (*release)(void* user_data, uint32_t ordinal);
} iree_hal_amd_gpu_tsan_client_t;

// Optional device-owned cold membership, allocated only with shader TSAN.
// Queue zero is provisioned for the device lifetime and supplies each image's
// canonical placement. Additional shadows exist only while their queues live.
struct iree_hal_amd_gpu_tsan_registry_t {
  // Borrowed native address domain, live through every client and queue.
  const iree_hal_amd_gpu_context_t* context;
  // Allocator for individual shadow owners.
  iree_allocator_t host_allocator;
  // Serializes cold image and queue membership, never dispatch or retirement.
  iree_slim_mutex_t mutex;
  // Immutable shadow geometry shared by all queues.
  iree_hal_amd_gpu_tsan_options_t options;
  // Exact native family using these shadows.
  uint32_t family_ordinal;
  // Initial coordinates whose shadows remain live until device destruction.
  uint32_t provisioned_count;
  // Highest initialized coordinate plus one.
  uint32_t count;
  // Borrowed live images with queue-local shader globals.
  iree_hal_amd_gpu_tsan_client_t* clients;
  // Stable coordinate directory; only live queues allocate shadow backing.
  iree_hal_amd_gpu_tsan_t* queues[256];
};

// Initializes cold membership without allocating native backing.
void iree_hal_amd_gpu_tsan_registry_initialize(
    const iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    uint32_t provisioned_count, const iree_hal_amd_gpu_tsan_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_tsan_registry_t* registry);

// Releases all shadows after queues and clients have been destroyed.
void iree_hal_amd_gpu_tsan_registry_deinitialize(
    iree_hal_amd_gpu_tsan_registry_t* registry);

// Prepares a shadow and all existing images before exposing a native queue.
// Failure rolls back this coordinate without changing other live queues.
iree_status_t iree_hal_amd_gpu_tsan_registry_add_queue(
    iree_hal_amd_gpu_tsan_registry_t* registry, uint32_t ordinal);

// Releases image placements and shadow backing after the queue's native join.
// NULL registries and absent coordinates require no work.
void iree_hal_amd_gpu_tsan_registry_remove_queue(
    iree_hal_amd_gpu_tsan_registry_t* registry, uint32_t ordinal);

// Prepares every live coordinate and links the client only on success.
iree_status_t iree_hal_amd_gpu_tsan_registry_register(
    iree_hal_amd_gpu_tsan_registry_t* registry,
    iree_hal_amd_gpu_tsan_client_t* client);

// Unlinks a client before its owner releases placements. No native work occurs.
void iree_hal_amd_gpu_tsan_registry_unregister(
    iree_hal_amd_gpu_tsan_client_t* client);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_TSAN_H_
