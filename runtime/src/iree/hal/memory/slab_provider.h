// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_SLAB_PROVIDER_H_
#define IREE_HAL_MEMORY_SLAB_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/atomic.h"
#include "iree/hal/buffer.h"
#include "iree/hal/memory/asan_pool.h"
#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_slab_provider_t iree_hal_slab_provider_t;
typedef struct iree_hal_slab_provider_vtable_t iree_hal_slab_provider_vtable_t;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// Immutable properties of slabs acquired from a provider.
typedef struct iree_hal_slab_provider_properties_t {
  // Memory type properties shared by every slab from the provider.
  iree_hal_memory_type_t memory_type;

  // Buffer usage bits supported by every slab from the provider.
  iree_hal_buffer_usage_t supported_usage;

  // Queue families that may access buffers materialized from the provider.
  // IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY means every queue family in the
  // containing logical device.
  iree_hal_queue_family_affinity_t queue_family_affinity;

  // Guaranteed power-of-two alignment of acquired slab byte zero.
  iree_device_size_t allocation_alignment;

  // Largest power-of-two alignment accepted for an individual acquisition.
  iree_device_size_t max_allocation_alignment;

  // Smallest independently maintained byte granule; one for coherent storage.
  iree_device_size_t maintenance_alignment;

  // Atomic operations supported by naturally aligned slab locations.
  iree_hal_atomic_operation_capabilities_t atomic_operations;
} iree_hal_slab_provider_properties_t;

// Running statistics for a slab provider (and any inner providers in its
// chain). Stats accumulate from the innermost provider outward, with each
// layer adding its own contributions.
typedef struct iree_hal_slab_provider_stats_t {
  // Cumulative slabs acquired from the platform/driver.
  uint64_t total_acquired;
  // Cumulative slabs released to the platform/driver.
  uint64_t total_released;

  // Cache-layer statistics. Zero for non-caching providers.
  struct {
    // Number of slabs currently in the cache freelist.
    uint32_t count;
    // Acquires satisfied from the freelist (fast path).
    uint64_t hit_count;
    // Acquires that required waiting or fell through to the inner provider.
    uint64_t miss_count;
    // Current EMA of reuse interval in nanoseconds. Tracks how quickly
    // cached slabs are recycled. Short intervals indicate bursty reuse.
    uint64_t ema_reuse_interval_nanoseconds;
    // Cumulative time spent in prefault callbacks (nanoseconds).
    uint64_t prefault_time_nanoseconds;
  } cache;
} iree_hal_slab_provider_stats_t;

// Maximum number of providers tracked in a visited set. If a provider chain
// is deeper than this, stats queries bail at the top level.
#define IREE_HAL_SLAB_PROVIDER_MAX_VISITED 32

// Set of already-visited providers for deduplicating stats queries across
// shared provider chains. Multiple pools may share the same provider (or
// provider chain); the visited set prevents double-counting.
//
// Stack-allocate before a stats query:
//   iree_hal_slab_provider_visited_set_t visited = {0};
typedef struct iree_hal_slab_provider_visited_set_t {
  // Number of providers already recorded in |providers|.
  iree_host_size_t count;

  // Providers visited during the current recursive stats query.
  const iree_hal_slab_provider_t* providers[IREE_HAL_SLAB_PROVIDER_MAX_VISITED];
} iree_hal_slab_provider_visited_set_t;

//===----------------------------------------------------------------------===//
// iree_hal_slab_t
//===----------------------------------------------------------------------===//

// A contiguous region of physical memory acquired from a slab provider.
// Slabs are the backing store for pool offset allocators: the pool manages
// offsets within [0, length), and the slab provider materializes buffers that
// reference physical memory at those offsets.
//
// |base_ptr| and |provider_handle| are provider-owned payloads, not generic
// pool data. Host providers may store a host pointer in |base_ptr|. Device
// providers may store a device VA token, leave |base_ptr| NULL, and keep the
// real driver handle in |provider_handle|. Generic pools must not dereference
// |base_ptr| or inspect |provider_handle| directly.
//
// To expose a reservation as a HAL buffer, pools pass the slab plus an offset
// range to iree_hal_slab_provider_wrap_buffer(), which creates a
// provider-specific buffer referencing the physical memory.
typedef struct iree_hal_slab_t {
  // Base address of the slab's memory region.
  uint8_t* base_ptr;

  // Length of the slab in bytes.
  iree_device_size_t length;

  // Provider-specific opaque handle used to release the slab. The provider
  // interprets this however it needs: a pointer to internal state, an
  // allocation handle, a VMM mapping descriptor, etc.
  uint64_t provider_handle;

  // Native payload geometry. Owned acquisitions report their complete payload
  // charge; imported attachments may have unknown source allocation geometry.
  iree_hal_buffer_allocation_info_t native_allocation;

  // Source-qualified maintenance envelope in slab coordinates. WHOLE_BUFFER
  // extends to slab length; owned slabs may control additional native padding.
  iree_hal_buffer_range_t maintenance_range;

  // Cache class established by the source mapping, or UNKNOWN without a view.
  iree_hal_host_cacheability_t host_cacheability;

  // Native or external-source physical identity, when established.
  iree_hal_physical_memory_id_t physical_backing_id;

  // Guaranteed alignment of logical slab byte zero for this acquisition.
  iree_device_size_t allocation_alignment;
} iree_hal_slab_t;

// One native attachment to caller-selected external storage. Properties
// describe this exact attachment and may be narrower than provider-owned slab
// properties, particularly for alignment and maintenance granularity.
typedef struct iree_hal_imported_slab_t {
  // Provider-owned attachment returned in a native release batch.
  iree_hal_slab_t slab;
  // Offset of slab byte zero within the caller's external storage extent.
  iree_device_size_t storage_offset;
  // Exact immutable properties of this attachment.
  iree_hal_slab_provider_properties_t properties;
} iree_hal_imported_slab_t;

//===----------------------------------------------------------------------===//
// Typed native slab batches
//===----------------------------------------------------------------------===//

// Geometry already validated by the source pool, including hidden backing.
typedef struct iree_hal_slab_request_t {
  // Minimum physical range length in bytes.
  iree_device_size_t min_length;
  // Nonzero power-of-two alignment of slab byte zero.
  iree_device_size_t min_alignment;
} iree_hal_slab_request_t;

typedef enum iree_hal_slab_operation_type_e {
  IREE_HAL_SLAB_OPERATION_ACQUIRE = 0,
  IREE_HAL_SLAB_OPERATION_RELEASE,
} iree_hal_slab_operation_type_t;

// Backend-owned linkage while a typed native operation is pending. There is
// no common execution queue: a provider decides where its native calls run.
// The two operation kinds cannot carry arbitrary work or execution waits.
typedef struct iree_hal_slab_operation_t {
  // Intrusive link available to the provider until terminal completion.
  struct iree_hal_slab_operation_t* next;
  // Provider borrowed through submission and terminal completion.
  iree_hal_slab_provider_t* provider;
  // Concrete request type, initialized by the provider entry point.
  iree_hal_slab_operation_type_t type;
} iree_hal_slab_operation_t;

typedef struct iree_hal_slab_acquire_callback_t {
  // Takes ownership of the acquisition status after the whole batch resolves.
  void(IREE_API_PTR* fn)(void* user_data, iree_status_t status);
  // Borrowed callback state.
  void* user_data;
} iree_hal_slab_acquire_callback_t;

typedef struct iree_hal_slab_release_callback_t {
  // Called after the provider relinquishes every native slab in the batch.
  void(IREE_API_PTR* fn)(void* user_data);
  // Borrowed callback state.
  void* user_data;
} iree_hal_slab_release_callback_t;

// Caller-owned descriptor and arrays, stable through both submission and
// completion. Exactly one completion may run inline or later. On success all
// outputs own native slabs; on failure every staged slab has been released and
// outputs are zero. Output storage is private until completion.
typedef struct iree_hal_slab_acquire_request_t {
  // Backend linkage initialized by the provider entry point.
  iree_hal_slab_operation_t operation;
  // Number of input geometries and output slabs.
  iree_host_size_t count;
  // Validated native allocation geometry, borrowed through completion.
  const iree_hal_slab_request_t* requests;
  // Staged native outputs, borrowed through completion.
  iree_hal_slab_t* slabs;
  // Exactly one terminal result for the complete batch.
  iree_hal_slab_acquire_callback_t completion;
} iree_hal_slab_acquire_request_t;

// Infallible submission using storage secured during acquisition. The caller
// establishes actual retirement and releases every native view before submit.
// The descriptor and slab array remain allocated through completion. Submission
// consumes their contents; providers may compact the pending native work in
// place.
typedef struct iree_hal_slab_release_request_t {
  // Backend linkage initialized by the provider entry point.
  iree_hal_slab_operation_t operation;
  // Number of slabs returned; providers may reduce it as entries are consumed.
  iree_host_size_t count;
  // Mutable native descriptors consumed by the provider through completion.
  iree_hal_slab_t* slabs;
  // Exactly one terminal completion after all native release attempts.
  iree_hal_slab_release_callback_t completion;
} iree_hal_slab_release_request_t;

//===----------------------------------------------------------------------===//
// iree_hal_slab_provider_t
//===----------------------------------------------------------------------===//

// Acquires and releases large chunks of physical memory from the platform or
// GPU driver. This is the only component that makes driver API calls for
// memory allocation; everything above it (offset allocators, pools) is pure
// host-side bookkeeping.
//
// Slab providers are ref-counted. Pools retain their slab provider at creation
// time and release it on destruction, ensuring the provider outlives all pools
// that use it. Slab providers are typically created by the device at init time
// and shared across pools on the same device. Different memory classes
// (DEVICE_LOCAL, HOST_VISIBLE, HOST_CACHED) require separate slab providers
// because they draw from different physical memory.
//
// Concrete implementations embed this struct at offset 0.
struct iree_hal_slab_provider_t {
  // Reference count controlling provider lifetime.
  iree_atomic_ref_count_t ref_count;

  // Concrete provider implementation hooks.
  const iree_hal_slab_provider_vtable_t* vtable;

  // Native-owner allocator, valid through the final provider release. Source
  // pools use it for presecured retirement records that can outlive their
  // caller-owned public pool metadata.
  iree_allocator_t host_allocator;
};

// Initializes the base slab provider fields. Called by concrete
// implementations during their create function.
void iree_hal_slab_provider_initialize(
    const iree_hal_slab_provider_vtable_t* vtable,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t* provider);

// Retains a reference to the slab provider.
void iree_hal_slab_provider_retain(iree_hal_slab_provider_t* provider);

// Releases a reference. Destroys the provider when the count reaches zero.
void iree_hal_slab_provider_release(iree_hal_slab_provider_t* provider);

// Submits a complete native acquisition batch. Capture cannot fail: native
// allocation failures are delivered through the exactly-once completion.
// The caller supplies validated geometry and all descriptor/output storage.
void iree_hal_slab_provider_acquire_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_acquire_request_t* request);

// Explicitly blocking convenience for cold callers acquiring one slab.
// Allocation and queue progress paths use acquire_slabs instead.
iree_status_t iree_hal_slab_provider_acquire_slab_sync(
    iree_hal_slab_provider_t* provider, iree_device_size_t min_length,
    iree_device_size_t min_alignment, iree_hal_slab_t* out_slab);

// Returns a borrowed HOST_POINTER view of the source's maintenance envelope.
// Pool import has validated its bounds. Native registration providers use this
// view to limit cold cache preparation to lines intersecting the authorized
// extent. Native page coverage may extend beyond it. The original external
// transport retains its ownership and release payload.
iree_hal_external_storage_t iree_hal_slab_provider_registration_view(
    const iree_hal_external_storage_t* storage);

// Attaches the provider to a caller-selected external storage extent. The
// provider borrows |storage| for this call and acquires any native references
// required by the returned slab. It never consumes or releases the external
// transport owner. A provider may attach only the source-authorized
// registration envelope; out_import->storage_offset locates that slab within
// |storage|. On failure |out_import| remains zeroed.
iree_status_t iree_hal_slab_provider_import_slab(
    iree_hal_slab_provider_t* provider,
    const iree_hal_external_storage_t* storage,
    iree_hal_imported_slab_t* out_import);

// Queries native geometry required to import |storage_class|. Returns false
// and zeros |out_requirements| when this provider cannot attach that class.
bool iree_hal_slab_provider_query_import_requirements(
    const iree_hal_slab_provider_t* provider,
    iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_import_requirements_t* out_requirements);

// Submits native release without allocating capture storage. Completion
// reports relinquished ownership even if the native API could only leak on
// release failure; concrete providers own those cleanup diagnostics.
void iree_hal_slab_provider_release_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_release_request_t* request);

// Explicitly blocking convenience for cold callers returning one slab.
// Logical pool release uses an acquisition-backed release descriptor instead.
void iree_hal_slab_provider_release_slab_sync(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab);

// Wraps a byte range within |slab| as a HAL buffer using the provider-specific
// buffer implementation for that slab's memory.
//
// |slab_offset| and |allocation_size| define the byte range relative to the
// slab's base. The range must lie fully inside [0, slab->length).
// |allowed_access| and |allowed_usage| are the permissions captured by the
// owning pool's memory contract; zero usage grants no operations.
iree_status_t iree_hal_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer);

// Validates that |provider| can advise allocations shaped by |options|.
//
// Disabled ASAN options are always valid and do not dispatch to the provider.
// Enabled options must first satisfy target-neutral ASAN validation and then
// pass provider-specific capability checks.
iree_status_t iree_hal_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* provider,
    const iree_hal_asan_pool_options_t* options);

// Advises the provider of an ASAN backing range lifecycle transition.
//
// |backing_offset| identifies the beginning of |layout|'s backing range within
// |slab|. The provider uses |layout| to locate the user-visible range and
// poison/unpoison bytes in its target-specific shadow state according to
// |advice_flags|. The caller establishes actual completion of prior accesses
// before changing shadow state. This hook must be infallible after enabled ASAN
// options have been accepted by iree_hal_slab_provider_validate_asan_options().
void iree_hal_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout);

// Prepares an exclusively owned, retired range (page faulting, first touch,
// etc.). May change contents inside the range, never outside it. The caller
// establishes actual completion of prior users before requesting preparation.
void iree_hal_slab_provider_prefault(iree_hal_slab_provider_t* provider,
                                     const iree_hal_slab_t* slab,
                                     iree_device_size_t offset,
                                     iree_device_size_t length);

// Releases unused cached resources. Passes |flags| through to the provider
// and any inner providers in the chain.
void iree_hal_slab_provider_trim(iree_hal_slab_provider_t* provider,
                                 iree_hal_pool_trim_flags_t flags);

// Accumulates statistics from the provider (and any inner providers).
// |visited| prevents double-counting across shared provider chains.
void iree_hal_slab_provider_query_stats(
    const iree_hal_slab_provider_t* provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats);

// Queries the memory properties of slabs from this provider.
void iree_hal_slab_provider_query_properties(
    const iree_hal_slab_provider_t* provider,
    iree_hal_slab_provider_properties_t* out_properties);

// Returns true if |provider| has already been visited (should be skipped).
// If not visited, adds it to the set and returns false. Returns true without
// adding if the set is full (conservative: skip rather than double-count).
bool iree_hal_slab_provider_visited(
    iree_hal_slab_provider_visited_set_t* visited,
    const iree_hal_slab_provider_t* provider);

// Prepared buffer facts embedded beside stable pool-owned slab metadata.
// The slab, provider, progress owner, and tracker are borrowed for that
// lifetime.
typedef struct iree_hal_slab_buffer_backing_t {
  // Immutable facts shared by materialized views of this slab.
  iree_hal_buffer_backing_facts_t facts;
  // Native lifecycle dispatch using this object's stable address.
  iree_hal_buffer_range_advice_t advice;
  // Native provider borrowed from the allocation owner.
  iree_hal_slab_provider_t* provider;
  // Stable native slab descriptor borrowed from the allocation owner.
  const iree_hal_slab_t* slab;
} iree_hal_slab_buffer_backing_t;

// Initializes facts after the slab descriptor reaches its stable owned address.
// |properties| describes this exact slab rather than merely the provider's
// owned-allocation route.
void iree_hal_slab_buffer_backing_initialize(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
    const iree_hal_slab_provider_properties_t* properties,
    iree_async_notification_t* notification,
    iree_async_frontier_tracker_t* tracker,
    iree_hal_slab_buffer_backing_t* out_backing);

//===----------------------------------------------------------------------===//
// iree_hal_slab_provider_t vtable
//===----------------------------------------------------------------------===//

// Implementer conveniences for platforms exposing scalar native operations.
// These perform the complete batch on the calling context selected by the
// concrete provider, roll back partial acquisition, and invoke completion.
// They contain no scheduling, waits, or execution-context selection.
void iree_hal_slab_provider_acquire_slabs_inline(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_acquire_request_t* request,
    iree_status_t (*acquire_slab)(iree_hal_slab_provider_t* provider,
                                  iree_device_size_t min_length,
                                  iree_device_size_t min_alignment,
                                  iree_hal_slab_t* out_slab),
    void (*release_slab)(iree_hal_slab_provider_t* provider,
                         const iree_hal_slab_t* slab));
void iree_hal_slab_provider_release_slabs_inline(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_release_request_t* request,
    void (*release_slab)(iree_hal_slab_provider_t* provider,
                         const iree_hal_slab_t* slab));

struct iree_hal_slab_provider_vtable_t {
  // Destroys a concrete slab provider implementation.
  void (*destroy)(iree_hal_slab_provider_t* provider);

  // Accepts a typed native acquisition using caller-owned stable storage.
  // The implementation supplies exactly one completion, including on failure.
  void (*acquire_slabs)(iree_hal_slab_provider_t* provider,
                        iree_hal_slab_acquire_request_t* request);

  // Returns actually retired slabs using presecured descriptor storage.
  // Native release attempts finish before the exactly-once completion.
  void (*release_slabs)(iree_hal_slab_provider_t* provider,
                        iree_hal_slab_release_request_t* request);

  // Wraps [slab_offset, slab_offset + allocation_size) in |slab| as a HAL
  // buffer whose backing and concrete buffer type are owned by this provider.
  //
  // Generic pools must use this hook instead of interpreting slab payload
  // fields directly. For CPU providers this typically wraps a host pointer in
  // iree_hal_heap_buffer_t; GPU or remote providers can create device-specific
  // buffer objects that retain opaque driver state from |slab|.
  //
  // The returned buffer invokes |release_callback| when destroyed. The pool
  // uses that callback to release its reservation bookkeeping; the provider
  // remains responsible for releasing whole slabs through release_slabs().
  // If |release_callback.fn| is NULL, the returned buffer is still a borrowed
  // slab view and must not release the slab or underlying physical storage.
  iree_status_t (*wrap_buffer)(
      iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
      iree_device_size_t slab_offset, iree_device_size_t allocation_size,
      iree_hal_memory_access_t allowed_access,
      iree_hal_buffer_usage_t allowed_usage,
      iree_hal_buffer_release_callback_t release_callback,
      iree_hal_buffer_t** out_buffer);

  // Validates that this provider can advise allocations shaped by enabled
  // ASAN options. Disabled options are handled by the public wrapper and do
  // not dispatch here.
  iree_status_t (*validate_asan_options)(
      const iree_hal_slab_provider_t* provider,
      const iree_hal_asan_pool_options_t* options);

  // Advises the provider of an ASAN backing range lifecycle transition.
  void (*advise_asan_range)(iree_hal_slab_provider_t* provider,
                            const iree_hal_slab_t* slab,
                            iree_device_size_t backing_offset,
                            iree_hal_asan_range_advice_flags_t advice_flags,
                            const iree_hal_asan_allocation_layout_t* layout);

  // Prepares only the requested range after prior accesses have completed.
  // Providers whose allocations already commit every page may do nothing.
  void (*prefault)(iree_hal_slab_provider_t* provider,
                   const iree_hal_slab_t* slab, iree_device_size_t offset,
                   iree_device_size_t length);

  // Releases unused cached resources. Caching providers release freelisted
  // slabs back to their inner provider. Non-caching providers do nothing.
  // |flags| controls which resources are released (ALL vs EXCESS).
  void (*trim)(iree_hal_slab_provider_t* provider,
               iree_hal_pool_trim_flags_t flags);

  // Accumulates this provider's statistics into |out_stats|. Wrapping
  // providers call into their inner provider first, then add their own
  // contributions. |visited| prevents double-counting when multiple pools
  // share a provider chain; providers already in the set return immediately.
  void (*query_stats)(const iree_hal_slab_provider_t* provider,
                      iree_hal_slab_provider_visited_set_t* visited,
                      iree_hal_slab_provider_stats_t* out_stats);

  // Queries the memory properties of slabs from this provider.
  void (*query_properties)(const iree_hal_slab_provider_t* provider,
                           iree_hal_slab_provider_properties_t* out_properties);

  // Attaches to external storage without consuming its
  // transport owner. NULL when this provider cannot import external storage.
  iree_status_t (*import_slab)(iree_hal_slab_provider_t* provider,
                               const iree_hal_external_storage_t* storage,
                               iree_hal_imported_slab_t* out_import);

  // Queries immutable external storage geometry without native calls.
  bool (*query_import_requirements)(
      const iree_hal_slab_provider_t* provider,
      iree_hal_external_storage_class_t storage_class,
      iree_hal_pool_import_requirements_t* out_requirements);
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_SLAB_PROVIDER_H_
