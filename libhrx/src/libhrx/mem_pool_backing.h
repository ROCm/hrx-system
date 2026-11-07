// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_MEM_POOL_BACKING_H_
#define LIBHRX_SRC_LIBHRX_MEM_POOL_BACKING_H_

#include "hrx_runtime.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/buffer.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

// A public pool's request to retain shared storage. The backing owner guards
// all fields; the public pool embeds this record and detaches it before dying.
typedef struct hrx_mem_pool_retention_t {
  // Previous request attached to the device's backing owner, or NULL.
  struct hrx_mem_pool_retention_t* previous;
  // Next request attached to the device's backing owner, or NULL.
  struct hrx_mem_pool_retention_t* next;
  // Floor applied by subsequent trims; it never triggers allocation.
  uint64_t min_bytes_to_keep;
} hrx_mem_pool_retention_t;

// Lazily created owned-storage classes with different native requirements.
typedef struct hrx_mem_pool_sources_t {
  // Ordinary allocations with no requirement for independent native aliases.
  iree_hal_pool_t* ordinary;
  // Allocations prepared with the native mapping-source interface.
  iree_hal_pool_t* mapping;
} hrx_mem_pool_sources_t;

// Device-owned backing shared by the device's public memory pools. The device
// outlives its pools and buffers; the cache survives individual pool teardown.
typedef struct hrx_mem_pool_backing_t {
  // Guards lazy construction and retention requests, never payload allocation.
  iree_slim_mutex_t mutex;
  // Retained native storage cache, created on first pool allocation.
  iree_hal_pool_t* cache;
  // Retained sources for device storage, acquired only when requested.
  hrx_mem_pool_sources_t device;
  // Retained sources for uncached device storage.
  hrx_mem_pool_sources_t uncached_device;
  // Retained sources with required coherent public host access.
  hrx_mem_pool_sources_t host;
  // Retained sources with coherent host access and uncached device access.
  hrx_mem_pool_sources_t uncached_host;
  // Retained fixed-block pool for coherent eight-byte native queue signals.
  iree_hal_pool_t* signals;
  // Retained import-only source for caller-owned host pointers.
  iree_hal_pool_t* host_import_source;
  // Retained import-only source for native device addresses.
  iree_hal_pool_t* device_import_source;
  // Immutable child policy geometry selected with the cache's slab class.
  iree_hal_tlsf_pool_options_t pool_options;
  // Borrowed list of requests embedded in live public pools.
  hrx_mem_pool_retention_t* retention_head;
  // Maximum requested floor; all clients can reuse the same retained bytes.
  uint64_t min_bytes_to_keep;
} hrx_mem_pool_backing_t;

void hrx_mem_pool_backing_initialize(hrx_mem_pool_backing_t* backing);

// Releases shared storage before native device teardown. All public pools,
// buffers, and reservations must have retired.
void hrx_mem_pool_backing_deinitialize(hrx_mem_pool_backing_t* backing);

// Returns the HRX buffer usage for a family's declared roles. Physical sources
// and virtual reservations use the same access contract, including copy-only
// families without requiring dispatch access from them.
iree_hal_buffer_usage_t hrx_mem_pool_family_usage(
    const iree_hal_queue_family_t* family);

// Creates one independent allocation policy over the device's shared cache.
// The caller retains the device; construction acquires no native payload.
iree_status_t hrx_mem_pool_backing_create_pool(hrx_device_t device,
                                               iree_hal_pool_t** out_pool);

// Retains the device-owned source matching |memory_type| for an ordinary
// allocation. The caller must release the returned pool. Host-local storage
// receives required coherent host access; device-uncached storage receives a
// hard uncached family requirement; their combination preserves both. All
// other storage prefers the device's physical placement. MAPPING_SOURCE usage
// selects native backing that can be aliased into explicit reservations on
// every device in the sealed group; unsupported complete access sets fail
// construction. The returned pool publishes its mapping geometry before any
// payload allocation.
iree_status_t hrx_mem_pool_backing_retain_allocation_source(
    hrx_device_t device, hrx_memory_type_t memory_type,
    hrx_buffer_usage_t usage, iree_hal_pool_t** out_pool);

// Retains the device-owned eight-byte signal pool. Signals have coherent host
// access and native 32/64-bit waits and stores on the dispatch family. Small
// allocations share backing; one idle slab is retained until device teardown.
iree_status_t hrx_mem_pool_backing_retain_signal_pool(
    hrx_device_t device, iree_hal_pool_t** out_pool);

// Retains the device-owned import-only source for |storage_class|. Portable
// host pointers and native device addresses are the supported HRX interop
// classes. The caller must release the returned pool.
iree_status_t hrx_mem_pool_backing_retain_import_source(
    hrx_device_t device, iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_t** out_pool);

void hrx_mem_pool_backing_attach(hrx_mem_pool_backing_t* backing,
                                 hrx_mem_pool_retention_t* retention);
void hrx_mem_pool_backing_detach(hrx_mem_pool_backing_t* backing,
                                 hrx_mem_pool_retention_t* retention);

// Changes this pool's floor for subsequent shared-cache trims. Already queued
// trimming can complete; increasing the floor does not restore released bytes.
void hrx_mem_pool_backing_set_retention(hrx_mem_pool_backing_t* backing,
                                        hrx_mem_pool_retention_t* retention,
                                        uint64_t min_bytes_to_keep);

// Updates the caller's floor and trims toward the combined floor. Other pools'
// requests remain effective. Native release runs asynchronously on the owner.
void hrx_mem_pool_backing_trim(hrx_mem_pool_backing_t* backing,
                               hrx_mem_pool_retention_t* retention,
                               uint64_t min_bytes_to_keep);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_MEM_POOL_BACKING_H_
