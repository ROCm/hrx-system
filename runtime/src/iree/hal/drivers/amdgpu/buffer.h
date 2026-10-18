// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_BUFFER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/atomic_memory.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_buffer_t iree_hal_amdgpu_buffer_t;

// Native slots shared by direct and queue-ordered ROCr storage. Registered host
// storage can have different host and agent addresses. Public mapping grants
// are independent of both execution bindings.
enum iree_hal_amdgpu_buffer_binding_index_e {
  IREE_HAL_AMDGPU_BUFFER_BINDING_DEVICE_ADDRESS = 0,
  IREE_HAL_AMDGPU_BUFFER_BINDING_HOST = 1,
  IREE_HAL_AMDGPU_BUFFER_BINDING_MAPPING_SOURCE = 2,
  IREE_HAL_AMDGPU_BUFFER_BINDING_COUNT = 3,
};

// Inline native facts borrowed by prepared views for their allocation epoch.
typedef struct iree_hal_amdgpu_buffer_native_t {
  // Generic native slots at offset zero, referenced by buffer->memory.bindings.
  iree_hal_buffer_native_binding_t
      bindings[IREE_HAL_AMDGPU_BUFFER_BINDING_COUNT];
  // Immutable width and scope cells supported by the backing allocation.
  iree_hal_amdgpu_atomic_memory_cell_flags_t atomic_memory_cells;
} iree_hal_amdgpu_buffer_native_t;

// Complete ROCr native format, including private atomic capability cells.
const iree_hal_buffer_binding_layout_t* iree_hal_amdgpu_buffer_binding_layout(
    void);

// Returns captured facts from a qualified, prepared AMDGPU buffer.
static inline const iree_hal_amdgpu_buffer_native_t*
iree_hal_amdgpu_buffer_native(const iree_hal_buffer_t* buffer) {
  return (const iree_hal_amdgpu_buffer_native_t*)buffer->memory.bindings;
}

// Publishes a borrowed native mapping descriptor before the buffer escapes its
// constructor. The provider keeps it live for the buffer's allocation epoch.
void iree_hal_amdgpu_buffer_prepare_mapping_source(iree_hal_buffer_t* buffer,
                                                   const void* descriptor,
                                                   uint32_t granularity);

// Per-physical-device pool of materialized AMDGPU HAL buffer wrappers.
//
// The pool only owns the host-side iree_hal_amdgpu_buffer_t storage. Backing
// HSA memory ownership remains expressed by each buffer's release callback or
// direct HSA allocation ownership.
typedef struct iree_hal_amdgpu_buffer_pool_t {
  // Per-physical-device host block pool used for cold wrapper-block growth.
  iree_arena_block_pool_t* block_pool;

  // Head of the lock-free return stack pushed by AMDGPU buffer destroy.
  iree_atomic_intptr_t return_head;

  // Serializes acquire-side cache pops, return-stack migration, and growth.
  iree_slim_mutex_t mutex;

  // Head of the mutex-protected acquire-side cache.
  iree_hal_amdgpu_buffer_t* acquire_head;

  // First host block owned by this pool.
  iree_arena_block_t* block_head;

  // Last host block owned by this pool.
  iree_arena_block_t* block_tail;

#if !defined(NDEBUG)
  // Number of wrappers currently retained by users or in-flight operations.
  iree_atomic_int32_t live_count;
#endif  // !defined(NDEBUG)
} iree_hal_amdgpu_buffer_pool_t;

// Initializes a per-physical-device materialized buffer wrapper pool.
//
// No wrapper memory is allocated until the first acquire. Wrapper storage grows
// in blocks borrowed from |block_pool| and returned during deinitialization.
iree_status_t iree_hal_amdgpu_buffer_pool_initialize(
    iree_arena_block_pool_t* block_pool,
    iree_hal_amdgpu_buffer_pool_t* out_pool);

// Deinitializes the pool and releases all cold-grown wrapper blocks.
//
// All buffers allocated from the pool must have been released before this is
// called. Violating that lifetime contract is a device teardown/use-after-free
// bug and is checked in debug builds.
void iree_hal_amdgpu_buffer_pool_deinitialize(
    iree_hal_amdgpu_buffer_pool_t* pool);

// Wraps an HSA memory pool allocation in an iree_hal_buffer_t.
// If |release_callback| is null the buffer owns the HSA allocation and frees
// it directly on destroy. Otherwise the callback owns teardown/release of the
// wrapped memory and any associated pool bookkeeping.
//
// |allocation_size| is the full size of the HSA allocation and may be larger
// than the logical |byte_length| exposed through the HAL buffer.
// |device_pointer| is the agent address. |host_pointer| is the independent
// host address, or NULL when host access is not prepared.
iree_status_t iree_hal_amdgpu_buffer_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_amdgpu_atomic_memory_cell_flags_t atomic_memory_cells,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    void* device_pointer, void* host_pointer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Wraps an HSA memory pool allocation in a pooled iree_hal_buffer_t wrapper.
//
// The returned buffer has the same memory ownership semantics as
// iree_hal_amdgpu_buffer_create(), but its host-side wrapper storage is
// returned to |pool| instead of |host_allocator| when the final reference is
// released. Native bindings address byte zero of the complete
// |allocation_size| extent; |byte_offset| locates the visible view within it.
// |pool| must outlive all buffers allocated from it.
iree_status_t iree_hal_amdgpu_buffer_create_pooled(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_amdgpu_atomic_memory_cell_flags_t atomic_memory_cells,
    iree_device_size_t allocation_size, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, void* device_pointer, void* host_pointer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_amdgpu_buffer_pool_t* pool, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

// Records direct native ownership in Tracy until this buffer releases it.
// Only direct allocator buffers use this stream: imported storage and pool
// wrappers have independent owners. Native release must finish before this
// buffer's release callback returns.
void iree_hal_amdgpu_buffer_trace_allocation(iree_hal_buffer_t* buffer);

// Tags |buffer| with a profiling allocation id.
//
// Only direct synchronous allocator buffers should be tagged. Pool/materialized
// and queue_alloca transient buffers have their own pool/reservation event
// streams and must not be double-counted as standalone buffer allocations.
void iree_hal_amdgpu_buffer_set_profile_allocation(
    iree_hal_buffer_t* buffer, uint64_t session_id, uint64_t allocation_id,
    uint64_t pool_id, uint32_t physical_device_ordinal,
    iree_device_size_t alignment);

// Returns true if |buffer| directly wraps ROCr storage.
bool iree_hal_amdgpu_buffer_isa(const iree_hal_buffer_t* buffer);

// Returns the agent base pointer for a buffer qualified for ROCr execution, or
// NULL if an unscoped buffer is not ROCr-owned or has no staged storage. Scoped
// buffers have already passed the queue/recording family boundary. Their
// wrapper owner may be another backend. Host accessibility is independent of
// this address.
//
// This is allocation coordinate zero and must be offset by
// iree_hal_buffer_byte_offset and the operation-relative offset when computing
// execution addresses. |buffer| must be the allocated buffer (not a subspan);
// callers should use iree_hal_buffer_allocated_buffer() to unwrap first.
void* iree_hal_amdgpu_buffer_device_pointer(iree_hal_buffer_t* buffer);

// Returns the immutable atomic memory cells supported by |buffer|.
//
// Transient buffers resolve their staged backing once; nested views carry
// prepared facts. Scoped foreign storage uses the generic backing's qualified
// operation families, never an AMDGPU-private extension of its binding array.
// Unqualified storage and unmaterialized transient buffers support no cells.
iree_hal_amdgpu_atomic_memory_cell_flags_t
iree_hal_amdgpu_buffer_atomic_memory_cells(iree_hal_buffer_t* buffer);

// Returns true if |buffer| is a direct AMDGPU buffer using exactly
// |release_callback| to release its backing storage.
bool iree_hal_amdgpu_buffer_uses_release_callback(
    iree_hal_buffer_t* buffer,
    iree_hal_buffer_release_callback_t release_callback);

// Disarms callback-owned storage that has already been released externally.
//
// |buffer| must be a direct AMDGPU buffer using exactly |release_callback|.
// The backing pointer and callback are cleared so destroying the remaining
// buffer wrapper performs no storage operation.
void iree_hal_amdgpu_buffer_disarm_storage(
    iree_hal_buffer_t* buffer,
    iree_hal_buffer_release_callback_t release_callback);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_BUFFER_H_
