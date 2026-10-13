// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_CONTEXT_CACHE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_CONTEXT_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdxdna_device iree_hal_amdxdna_device;
typedef struct iree_hal_amdxdna_device_context_cache_t
    iree_hal_amdxdna_device_context_cache_t;
typedef struct iree_hal_amdxdna_context_cache_lease_t
    iree_hal_amdxdna_context_cache_lease_t;

// Injectable native-context operations used by hermetic cache-policy tests and
// optional device integration hooks. NULL callbacks use the native DDI
// directly. The retain/release callbacks retain/release opaque native context
// refs with the same ownership contract as
// iree_hal_amdxdna_native_context_ref_{retain,release}.
typedef struct iree_hal_amdxdna_context_cache_ops_t {
  iree_status_t (*create_context)(
      void* user_data,
      const iree_hal_amdxdna_native_c_context_image_t* context_image,
      bool* out_context_pool_exhausted,
      iree_hal_amdxdna_native_context_ref_t** out_context_ref);
  iree_hal_amdxdna_native_context_ref_t* (*retain_context)(
      void* user_data, iree_hal_amdxdna_native_context_ref_t* context_ref);
  void (*release_context)(void* user_data,
                          iree_hal_amdxdna_native_context_ref_t* context_ref);
  // Optional hook invoked before the cache releases an entry-owned context ref.
  // Device-level users can use this to invalidate resources keyed by the
  // context/queue lifetime. The callback must not retain |context_ref|.
  void (*before_release_context)(
      void* user_data, iree_hal_amdxdna_native_context_ref_t* context_ref);
  // Optional native resource-pressure recovery hook. Called once after context
  // creation returns UNAVAILABLE so idle command resources can be dropped
  // before retrying. Idle-only, so it never tears down in-flight work. Runs
  // with the context-cache mutex held.
  void (*reclaim_create_unavailable)(void* user_data);
  // Drops one idle command-cache entry and returns true when an entry was
  // freed. Used to trim toward a heap target before context creation. Idle-only.
  // Runs with the context-cache mutex held.
  bool (*reclaim_one_idle_command)(void* user_data);
  // Shared DEV-heap size and construction reserve, or 0 when the backend has no
  // bounded heap. When set, the cache proactively drops idle command resources
  // before creating a context whose image would push live heap occupancy past
  // (max - reserve), so the PDI allocation and the command construction that
  // follows succeed on the first try instead of tripping ENOSPC and recovering
  // reactively.
  iree_host_size_t max_shared_code_memory_bytes;
  iree_host_size_t shared_code_memory_miss_reserve_bytes;
} iree_hal_amdxdna_context_cache_ops_t;

// Full identity of a cached hardware context. Two contexts are interchangeable
// only when they use the same native context image inputs. For Linux KMQ PDI
// contexts this is PDI + CU name; for native-xclbin backends this is the xclbin
// image. Dispatch control code is command identity, not context identity: the
// native context constructor does not consume it, and splitting one logical PDI
// context by control-code content creates redundant hwctx objects.
typedef struct iree_hal_amdxdna_context_cache_key_t {
  iree_const_byte_span_t pdi;
  iree_const_byte_span_t xclbin;
  iree_string_view_t kernel_name;
  // AIE4 empty-context column count. 0 for PDI/xclbin images.
  uint32_t partition_cols;
} iree_hal_amdxdna_context_cache_key_t;

// Returns true when a context cached under `lhs` may be safely reused to
// satisfy a request bearing `rhs`, i.e. all native context-image key fields are
// byte-for-byte equal. The cache uses this for lookup; it is also exported so
// the reuse-safety contract can be unit tested without a device.
bool iree_hal_amdxdna_context_cache_key_equal(
    const iree_hal_amdxdna_context_cache_key_t* lhs,
    const iree_hal_amdxdna_context_cache_key_t* rhs);

// Creates a bounded device context cache sized by `hardware_context_budget`
// (from iree_hal_amdxdna_native_c_device_caps_t::max_hardware_contexts). See
// iree_hal_amdxdna_context_cache_resolve_capacity for how the capacity is
// chosen.
iree_hal_amdxdna_device_context_cache_t*
iree_hal_amdxdna_device_context_cache_create(
    iree_allocator_t host_allocator, iree_host_size_t hardware_context_budget);

// Creates a cache using injected context operations. Intended for hermetic
// policy/lifetime tests; production code should use
// iree_hal_amdxdna_device_context_cache_create.
iree_hal_amdxdna_device_context_cache_t*
iree_hal_amdxdna_device_context_cache_create_with_ops(
    iree_allocator_t host_allocator, iree_host_size_t hardware_context_budget,
    const iree_hal_amdxdna_context_cache_ops_t* ops, void* user_data);

// Resolves the cache capacity: the device budget if nonzero, else a built-in
// default. Exported so the policy can be unit tested without a device.
iree_host_size_t iree_hal_amdxdna_context_cache_resolve_capacity(
    iree_host_size_t hardware_context_budget);

// Bounds the cumulative resident context-image bytes the cache keeps alive so
// the shared device code-memory heap is never fully consumed by cached hardware
// contexts. When |budget_bytes| is nonzero, get_or_create proactively evicts
// idle (unleased) LRU contexts before creating a new one so that the resident
// image footprint stays within budget; leased/in-flight contexts are never
// force-evicted by this path. Zero (the default) disables the memory bound and
// keeps the count-only behavior. See
// iree_hal_amdxdna_shared_code_memory_context_image_budget for the value.
void iree_hal_amdxdna_context_cache_set_context_image_budget(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_host_size_t budget_bytes);

void iree_hal_amdxdna_device_context_cache_destroy(
    iree_hal_amdxdna_device_context_cache_t* context_cache);

void iree_hal_amdxdna_device_context_cache_clear(
    iree_hal_amdxdna_device_context_cache_t* context_cache);

// Drops entries no executable still leases. Leased entries stay.
void iree_hal_amdxdna_device_context_cache_drop_unleased(
    iree_hal_amdxdna_device_context_cache_t* context_cache);

iree_host_size_t iree_hal_amdxdna_device_context_cache_count(
    iree_hal_amdxdna_device_context_cache_t* context_cache);

// Sum of native context-image bytes represented by cache entries. Exported for
// hermetic policy tests; production admission uses native live ownership so
// evicted contexts retained by in-flight work remain charged.
iree_host_size_t iree_hal_amdxdna_context_cache_cached_image_bytes(
    iree_hal_amdxdna_device_context_cache_t* context_cache);

// Evicts idle (unleased) LRU contexts, freeing their native context resources.
// If |force_leased| is true, also force-evicts one leased LRU entry after idle
// entries are gone.
void iree_hal_amdxdna_context_cache_reclaim(
    iree_hal_amdxdna_device_context_cache_t* context_cache, bool force_leased);

// Implements lookup/create independently of the HAL device wrapper. Exported
// for hermetic cache-policy tests; production callers use
// iree_hal_amdxdna_device_get_or_create_context below.
// `partition_cols` distinguishes empty AIE4 contexts; pass 0 for PDI/xclbin.
iree_status_t iree_hal_amdxdna_context_cache_get_or_create(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_native_device_t* native_device,
    uint32_t context_image_models, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    uint32_t partition_cols,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref);

// Like get_or_create, but returns a lease. Unleased LRU entries are evicted
// first; if the cap is full of leases (FLM pinning executables across model
// switches), the LRU leased entry is force-evicted and later retain returns
// NULL. When requested, |out_context_ref| is retained atomically with lease
// creation so force-eviction cannot invalidate the lease between pinning and
// acquiring the dispatch reference.
iree_status_t iree_hal_amdxdna_context_cache_pin(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_native_device_t* native_device,
    uint32_t context_image_models, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    uint32_t partition_cols,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref,
    iree_hal_amdxdna_context_cache_lease_t** out_lease);

// Retains the native context referenced by |lease| for one dispatch or CU-open
// operation. The returned context ref must be released by the caller.
iree_hal_amdxdna_native_context_ref_t*
iree_hal_amdxdna_context_cache_lease_retain_context(
    iree_hal_amdxdna_context_cache_lease_t* lease);

// Releases a cache entry lease created by iree_hal_amdxdna_context_cache_pin.
void iree_hal_amdxdna_context_cache_lease_release(
    iree_hal_amdxdna_context_cache_lease_t* lease);

// Returns a native context for the control-packet bootstrap `pdi`/`xclbin`
// and CU/export name, creating and caching it on first use. PDI, xclbin, and
// empty AIE4 START_DPU contexts are keyed by image bytes, name, and
// `partition_cols`.
iree_status_t iree_hal_amdxdna_device_get_or_create_context(
    iree_hal_amdxdna_device* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    uint32_t partition_cols,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref);

iree_status_t iree_hal_amdxdna_device_pin_context(
    iree_hal_amdxdna_device* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    uint32_t partition_cols,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref,
    iree_hal_amdxdna_context_cache_lease_t** out_lease);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_CONTEXT_CACHE_H_
