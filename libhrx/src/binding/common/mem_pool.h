// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_MEM_POOL_H_
#define LIBHRX_SRC_BINDING_COMMON_MEM_POOL_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "libhrx/src/libhrx/mem_pool_backing.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;

//===----------------------------------------------------------------------===//
// HIP/CUDA memory pool policy
//===----------------------------------------------------------------------===//

typedef enum iree_hal_streaming_mem_pool_attr_t {
  IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 0,
  IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 1,
  IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2,
  // Retained-byte floor contributed to the device's shared backing cache.
  // Compatible pools share storage, so their floors combine by maximum.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_RELEASE_THRESHOLD = 3,
  // Bytes borrowed by this pool, including padding and fragmentation. Empty
  // slabs return to the device cache and cease to be charged to this pool;
  // shared idle storage is not counted again for every potential borrower.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT = 4,
  // Peak bytes borrowed by this pool. Writing zero resets the watermark.
  IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_HIGH = 5,
  IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_CURRENT = 6,
  IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_HIGH = 7,
} iree_hal_streaming_mem_pool_attr_t;

typedef struct iree_hal_streaming_mem_pool_props_t {
  // Platform allocation-handle types requested when the pool was created.
  uint32_t alloc_handle_type;
  // Platform location type for the pool's backing storage.
  uint32_t location_type;
  // Platform device ordinal for the pool's backing storage.
  int location_id;
  // Maximum live requested bytes, or zero when the pool is unbounded. This
  // excludes native alignment, padding, and sanitizer quarantine overhead.
  size_t max_size;
} iree_hal_streaming_mem_pool_props_t;

typedef struct iree_hal_streaming_mem_pool_reuse_t {
  // True when internal dependency insertion is allowed for reuse.
  bool allow_internal_dependencies;

  // True when event dependencies are honored for reuse.
  bool follow_event_dependencies;

  // True when opportunistic reuse is allowed.
  bool allow_opportunistic;
} iree_hal_streaming_mem_pool_reuse_t;

// Frontend policy over a shared HAL storage cache. Native HRX pools do not
// inherit these compatibility limits, reuse attributes, or logical counters.
typedef struct iree_hal_streaming_mem_pool_t {
  // References held by public handles and bounded allocations.
  iree_atomic_ref_count_t ref_count;

  // Retained device owning the cache shared by compatible public pools.
  hrx_device_t device;

  // HIP/CUDA-style creation properties used for attribute queries.
  iree_hal_streaming_mem_pool_props_t props;

  // TLSF HAL pool serving suballocated and dedicated ordinary backing.
  iree_hal_pool_t* hal_pool;

  // Embedded request linked into the device's shared retention policy.
  hrx_mem_pool_retention_t retention;

  // Bytes charged to the immutable |props.max_size| allocation limit.
  size_t allocation_budget_current;

  // HIP/CUDA release threshold attribute in bytes.
  uint64_t release_threshold;

  // Allocations that have retained the HAL pool but have not reserved from it.
  uint32_t inflight_allocation_count;

  // Compatibility reuse choices, sampled together under |mutex|.
  iree_hal_streaming_mem_pool_reuse_t reuse;

  // Current backing borrowed by this child, excluding shared idle storage.
  uint64_t reserved_mem_current;

  // Peak backing borrowed by this child, including padding and fragmentation.
  uint64_t reserved_mem_high;

  // Current logical bytes charged to live frontend allocations.
  uint64_t used_mem_current;

  // Peak logical bytes charged to live frontend allocations.
  uint64_t used_mem_high;

  // Platform-native pool handle, if one is imported or exported.
  void* platform_handle;

  // Guards mutable pool attributes and lazy HAL pool creation.
  iree_slim_mutex_t mutex;
} iree_hal_streaming_mem_pool_t;

iree_status_t iree_hal_streaming_mem_pool_create(
    hrx_device_t device, const iree_hal_streaming_mem_pool_props_t* props,
    iree_hal_streaming_mem_pool_t** out_pool);
void iree_hal_streaming_mem_pool_retain(iree_hal_streaming_mem_pool_t* pool);
void iree_hal_streaming_mem_pool_release(iree_hal_streaming_mem_pool_t* pool);
iree_status_t iree_hal_streaming_mem_pool_get_attribute(
    iree_hal_streaming_mem_pool_t* pool,
    iree_hal_streaming_mem_pool_attr_t attr, uint64_t* out_value);
iree_status_t iree_hal_streaming_mem_pool_set_attribute(
    iree_hal_streaming_mem_pool_t* pool,
    iree_hal_streaming_mem_pool_attr_t attr, uint64_t value);
// Returns one consistent snapshot of the compatibility reuse choices. This
// does not query backing statistics or perform native allocation work.
iree_hal_streaming_mem_pool_reuse_t iree_hal_streaming_mem_pool_query_reuse(
    iree_hal_streaming_mem_pool_t* pool);

// Returns unused child backing and trims the shared device cache, preserving
// |min_bytes_to_keep| and every other pool's current retention request. This
// replaces this pool's floor until an attribute update or automatic release
// applies its configured threshold again. It does not allocate missing bytes.
iree_status_t iree_hal_streaming_mem_pool_trim(
    iree_hal_streaming_mem_pool_t* pool, size_t min_bytes_to_keep);

// Returns unused child backing and applies the configured release threshold to
// the device's shared cache, preserving other pools' retention requests. This
// is intended for stream-ordered free completion paths. Reclamation runs on
// the memory owner without waiting here; statistics converge after queued
// whole-slab returns and the ordered cache sweep complete.
iree_status_t iree_hal_streaming_mem_pool_release_unused(
    iree_hal_streaming_mem_pool_t* pool);

// Records a logical allocation backed by |pool|. Logical usage is distinct
// from physical TLSF reservations, which can remain retained for later
// stream-ordered reuse.
void iree_hal_streaming_mem_pool_record_logical_allocation(
    iree_hal_streaming_mem_pool_t* pool, size_t size);

// Records retirement of a logical allocation backed by |pool|. The physical
// reservation remains owned by its buffer until that buffer is released or
// reused.
void iree_hal_streaming_mem_pool_record_logical_free(
    iree_hal_streaming_mem_pool_t* pool, size_t size);

// Allocates a buffer with at least 256-byte alignment and the device access
// permissions captured by |pool|. The caller keeps |pool| alive until the
// returned buffer is released. Frontend allocation wrappers own that reference
// independently of the application's public pool handle.
iree_status_t iree_hal_streaming_mem_pool_allocate_buffer(
    iree_hal_streaming_mem_pool_t* pool, size_t size, hrx_buffer_t* out_buffer);

// Memory access flags for memory pools (for HIP API conversion).
typedef enum iree_hal_streaming_mem_access_flag_bits_e {
  IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_NONE = 0ull,
  IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READ = 1ull << 0,
  IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READWRITE =
      (1ull << 1) | IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READ,
} iree_hal_streaming_mem_access_flags_t;

// Memory pool location types (for HIP API conversion).
typedef enum iree_hal_streaming_mem_location_type_e {
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_INVALID = 0,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_DEVICE,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA,
  IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
} iree_hal_streaming_mem_location_type_t;

// Device pool accessors.
// Returns a device-owned pool handle that remains valid while selected.
iree_hal_streaming_mem_pool_t* iree_hal_streaming_device_default_mem_pool(
    iree_hal_streaming_device_t* device);
// Returns a device-owned pool handle that remains valid while selected.
iree_hal_streaming_mem_pool_t* iree_hal_streaming_device_mem_pool(
    iree_hal_streaming_device_t* device);
// Retains the selected pool for use outside the device lock. The caller must
// release the returned handle with iree_hal_streaming_mem_pool_release.
iree_hal_streaming_mem_pool_t* iree_hal_streaming_device_retain_mem_pool(
    iree_hal_streaming_device_t* device);
// Retains the device default pool for use outside the device lock. The caller
// must release the returned handle with iree_hal_streaming_mem_pool_release.
iree_hal_streaming_mem_pool_t*
iree_hal_streaming_device_retain_default_mem_pool(
    iree_hal_streaming_device_t* device);
iree_status_t iree_hal_streaming_device_ensure_default_mem_pool(
    iree_hal_streaming_device_t* device);
// Replaces the selected pool while preserving any in-flight pool users.
void iree_hal_streaming_device_set_mem_pool(
    iree_hal_streaming_device_t* device, iree_hal_streaming_mem_pool_t* pool);
// Restores the default pool only when |pool| is the selected pool.
void iree_hal_streaming_device_reset_mem_pool_if_current(
    iree_hal_streaming_device_t* device, iree_hal_streaming_mem_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_MEM_POOL_H_
