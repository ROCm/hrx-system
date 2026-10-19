// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_VIRTUAL_BUFFER_H_
#define IREE_HAL_VIRTUAL_BUFFER_H_

#include "iree/hal/memory_scope.h"
#include "iree/hal/queue.h"
#include "iree/hal/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Explicit mapping state supplied by the caller. These operations do not cause
// HAL to discover or track installed sources. Backends can use native bind and
// protection primitives when their APIs cannot replace arbitrary intervals.
typedef enum iree_hal_memory_mapping_operation_e {
  // Replaces any previous mapping in the range, including holes. A NULL source
  // clears the range. This is the default for callers managing sparse ranges.
  IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE = 0,
  // Installs a non-NULL source in a range the caller knows is unbound.
  IREE_HAL_MEMORY_MAPPING_OPERATION_BIND = 1,
  // Removes a complete bound range. Requires a NULL source and zero access.
  IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND = 2,
  // Changes access to a bound range. The caller supplies the same source and
  // source offset currently bound there, including when revoking all access.
  IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT = 3,
} iree_hal_memory_mapping_operation_t;

// Bit set indexed by iree_hal_memory_mapping_operation_t.
typedef uint32_t iree_hal_memory_mapping_operations_t;
#define IREE_HAL_MEMORY_MAPPING_OPERATIONS_ALL UINT32_C(0xF)

// Native address geometry for a complete simultaneous scope. A zero
// max_byte_length means no backend supports that scope. These limits describe
// addresses only; backing pools and their prepared mapping sources may impose
// additional alignment and size requirements. Address availability is checked
// at creation and is not guaranteed by a successful capability query.
typedef struct iree_hal_virtual_buffer_capabilities_t {
  // Largest supported address interval, in bytes; zero when unsupported.
  iree_device_size_t max_byte_length;
  // Required multiple of the reservation length and minimum base alignment.
  iree_device_size_t reservation_granularity;
  // Minimum independently replaceable destination offset and length unit.
  iree_device_size_t mapping_granularity;
  // Largest accepted power-of-two minimum base alignment.
  iree_device_size_t max_alignment;
  // Operations supported by every projection of the requested scope.
  iree_hal_memory_mapping_operations_t operations;
} iree_hal_virtual_buffer_capabilities_t;

// Queries geometry before acquiring addresses or physical backing. The same
// complete-scope qualification is used by creation. Unsupported combinations
// return OK with zeroed capabilities; invalid scope, native query, and metadata
// allocation failures propagate. The allocator is used only for temporary
// construction metadata. This is a cold planning operation.
IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_query_capabilities(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_capabilities_t* out_capabilities);

// Geometry of an address reservation, independent of physical storage pools.
// Zero alignment/granularity selects the backend's minimum. The achieved
// granularity is queried after creation. Mapping offsets and interior lengths
// are multiples of that unit; only the root reservation's final unit may be
// shorter. Reservation lengths still satisfy native capabilities. A larger
// granularity may reduce native mapping work (for example, the number of
// independently replaceable section views on Windows).
typedef struct iree_hal_virtual_buffer_options_t {
  // Nonzero length of the stable virtual buffer, in bytes.
  iree_device_size_t byte_length;
  // Minimum power-of-two alignment of the virtual buffer's first byte.
  iree_device_size_t min_alignment;
  // Minimum power-of-two independently replaceable mapping unit, in bytes.
  iree_device_size_t mapping_granularity;
} iree_hal_virtual_buffer_options_t;

// Reserves addresses without allocating physical payload and returns one owned
// ordinary buffer reference. Buffer references, subspans and child pools keep
// the native addresses alive until final release. Installed backing remains
// caller-owned and must stay live until checked removal completes after its
// last users. Accessing an unbound range is invalid. The sealed group outlives
// its buffers. Failure returns NULL and transfers no cleanup obligation.
IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_create(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Achieved mapping geometry of a virtual buffer or an ordinary view of one.
typedef struct iree_hal_virtual_buffer_info_t {
  // Visible byte zero in the native mapping target's coordinates.
  iree_device_size_t mapping_byte_offset;
  // Independently replaceable unit, in bytes; the root's final unit may be
  // short.
  iree_device_size_t mapping_granularity;
  // Operations supported across the admitted native projections.
  iree_hal_memory_mapping_operations_t operations;
  // Number of native page-table projections owned by the root buffer.
  uint32_t projection_count;
} iree_hal_virtual_buffer_info_t;

// Copies achieved geometry after any asynchronous allocation has committed.
// Returns UNAVAILABLE when the current buffer has no virtual mapping target.
IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_query_info(
    const iree_hal_buffer_t* buffer, iree_hal_virtual_buffer_info_t* out_info);

// Resolves one concrete admitted site to its canonical native projection.
// Equal IDs on the same native root share page tables. ANY is not a concrete
// site. Allocation prerequisites must be complete before calling this query.
IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_resolve_projection(
    const iree_hal_buffer_t* buffer, iree_hal_memory_scope_t scope,
    uint32_t* out_projection);

// Removes all mappings in the selected native projections without releasing
// addresses. The view must cover the complete native target; a child range
// cannot clear its siblings. ANY selects every site granted to this view.
// The caller has completed all affected execution and accepted mapping work
// and serializes mutation. This cold call may enter the driver and wait for
// native removal. It is admitted after partial mapping failure, preserves that
// failure, and returns the independent removal result. Only success permits
// reuse of detached backing, subject to its other aliases and users.
// Partial healthy removal remains a queue_map operation.
IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_detach(
    iree_hal_buffer_t* buffer, iree_hal_memory_scope_t scope);

// One change to a virtual range. Mapping borrows physical storage; it does not
// copy contents, transfer ownership, or retain backing after queue completion.
// Target offsets in root coordinates satisfy the reservation's mapping
// granularity. Lengths are whole units or end exactly at the root reservation's
// end. A subspan ending inside the root does not permit a short final unit.
// Native source offsets and lengths satisfy the source's independently prepared
// granularity, which may be smaller than the target's replacement unit.
// Binding and protection changes require their declared prior state. HAL does
// not validate that state or retain the source after completion. Backing with
// NONE access may be left detached natively; later protection changes supply
// the source again so the backend can install it without retained bookkeeping.
typedef struct iree_hal_memory_mapping_t {
  // Borrowed virtual buffer or view whose backing will change.
  iree_hal_buffer_t* target;
  // Exact admitted site, or ANY for all sites granted to the target view.
  iree_hal_memory_scope_t scope;
  // Destination interval relative to the target view's first byte.
  iree_hal_buffer_range_t target_range;
  // Ordinary buffer from a pool prepared with the MAPPING_SOURCE interface for
  // each selected site. NULL detaches the target range.
  iree_hal_buffer_t* source;
  // Byte offset within the logical source buffer; zero when detaching.
  iree_device_size_t source_offset;
  // Required native access permissions; NONE revokes access. A non-NULL source
  // with NONE access remains a logical binding owned entirely by the caller.
  iree_hal_memory_access_t access;
  // Prior-state contract and requested update; zero selects general
  // replacement.
  iree_hal_memory_mapping_operation_t operation;
} iree_hal_memory_mapping_t;

//===----------------------------------------------------------------------===//
// Virtual buffer implementation contract
//===----------------------------------------------------------------------===//

// Resolved source identity and target coordinates for one native projection.
// Public ranges and access have been validated and transient sources committed
// before native update. Arrays are sorted by target offset and nonoverlapping.
typedef struct iree_hal_memory_mapping_update_t {
  // Exact admitted target site, preserved across native projection aliases.
  uint32_t scope_id;
  // Target interval in reservation coordinates.
  iree_hal_buffer_range_t target_range;
  // Prepared native source; a NULL descriptor detaches the interval.
  iree_hal_buffer_native_binding_t source;
  // Exact requested access permissions.
  iree_hal_memory_access_t access;
  // Caller-declared binding state, already validated against backend support.
  iree_hal_memory_mapping_operation_t operation;
} iree_hal_memory_mapping_update_t;

// Shared prefix for native virtual-buffer implementations. The ordinary buffer
// resource owns the address interval and uses its single buffer vtable for
// native operations and final destruction. Concrete implementations own their
// bindings, projection table and backing facts through final buffer release.
typedef struct iree_hal_virtual_buffer_t {
  // Reference-counted buffer owning the native address interval.
  iree_hal_buffer_t buffer;
  // Allocator for the concrete native buffer and its immutable metadata.
  iree_allocator_t host_allocator;
  // Achieved mapping unit shared by all native projections.
  iree_device_size_t mapping_granularity;
  // Number of distinct native page-table projections.
  uint32_t projection_count;
  // Supported mapping operations, immutable after creation.
  iree_hal_memory_mapping_operations_t operations;
  // Native descriptor schema accepted for mapping sources.
  iree_hal_memory_backend_type_t source_type;
  // Scope-indexed native projections; INVALID for excluded sites and ANY.
  const uint32_t* projections;
} iree_hal_virtual_buffer_t;

// Borrowed mapping target after a view's allocation prerequisites complete.
// An empty value means no committed virtual target. The original view and its
// allocation epoch keep the root and immutable facts live through the query.
// Native operations use root coordinates; submissions retain the original view
// so child-pool release callbacks remain attached to the captured range.
typedef struct iree_hal_virtual_buffer_view_t {
  // Canonical native target, or NULL when unavailable.
  iree_hal_virtual_buffer_t* root;
  // Visible byte zero in the native root, in bytes.
  iree_device_size_t byte_offset;
} iree_hal_virtual_buffer_view_t;

// Resolves a trusted buffer's currently committed virtual mapping view.
// Subspans apply their flat offsets; transient and instrumented roots forward
// through their buffer vtable. No allocation or native operation is performed.
IREE_API_EXPORT iree_hal_virtual_buffer_view_t
iree_hal_buffer_virtual_view(const iree_hal_buffer_t* buffer);

// Initializes the ordinary buffer resource and retains its memory contract.
// The native constructor fills projection_count/projections before publication.
// Virtual backing facts describe address alignment and group progress services.
IREE_API_EXPORT void iree_hal_virtual_buffer_initialize(
    iree_hal_memory_contract_t* contract,
    const iree_hal_buffer_native_binding_t* bindings,
    const iree_hal_buffer_backing_facts_t* backing,
    iree_hal_memory_type_t memory_type, iree_device_size_t byte_length,
    iree_device_size_t mapping_granularity,
    iree_hal_memory_mapping_operations_t operations,
    iree_hal_memory_backend_type_t source_type,
    const iree_hal_buffer_vtable_t* vtable, iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_t* buffer);

// Common buffer methods for native implementations with coherent host views.
// Backends with different cache or host-view policies supply their own methods.
IREE_API_EXPORT iree_hal_virtual_buffer_view_t
iree_hal_virtual_buffer_query_root(const iree_hal_buffer_t* buffer);
IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_export_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length,
    iree_hal_external_storage_class_t requested_class,
    iree_hal_external_storage_t* out_storage,
    iree_hal_buffer_range_t* out_logical_range);
IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_map_range(
    iree_hal_buffer_t* buffer, iree_hal_mapping_mode_t mode,
    iree_hal_memory_access_t access, iree_hal_buffer_map_flags_t flags,
    iree_device_size_t offset, iree_device_size_t length,
    iree_hal_buffer_mapping_t* mapping);
IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_unmap_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_buffer_mapping_t* mapping);
IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_invalidate_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length);
IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_flush_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length);

// One normalized, nonoverlapping vector targeting a native projection.
// Resolution establishes these trusted entries before any native mutation.
typedef struct iree_hal_memory_mapping_batch_entry_t {
  // Canonical target borrowed through the captured original buffer views.
  iree_hal_virtual_buffer_t* target;
  // Native page-table projection within the target.
  uint32_t projection;
  // Number of updates in this projection's contiguous vector.
  iree_host_size_t update_count;
  // Validated root coordinates, permissions and prepared native sources.
  const iree_hal_memory_mapping_update_t* updates;
} iree_hal_memory_mapping_batch_entry_t;

// Queue-owned transaction capture. One allocation holds original descriptors,
// normalized entries and native-source scratch. Retained original buffer views
// keep their normal lifetime obligations through completion or cancellation.
// They do not extend an explicitly ended asynchronous allocation epoch.
typedef struct iree_hal_memory_mapping_batch_t {
  // Submission resource released after completion or cancellation.
  iree_hal_resource_t resource;
  // Maximum entries/updates, available before deferred targets are committed.
  iree_host_size_t capacity;
  // Number of resolved target/projection vectors.
  iree_host_size_t entry_count;
  // Resolved entries ordered by native root, projection and target offset.
  const iree_hal_memory_mapping_batch_entry_t* entries;
  // Total resolved updates across all entries.
  iree_host_size_t update_count;
} iree_hal_memory_mapping_batch_t;

// Retains the captured mapping transaction. NULL is allowed.
static inline void iree_hal_memory_mapping_batch_retain(
    iree_hal_memory_mapping_batch_t* batch) {
  iree_hal_resource_retain(batch);
}

// Releases a transaction reference and its captured views on final release.
// NULL is allowed.
static inline void iree_hal_memory_mapping_batch_release(
    iree_hal_memory_mapping_batch_t* batch) {
  iree_hal_resource_release(batch);
}

// Checks public logical arguments and captures the original buffer views.
// Empty transactions return NULL; failure transfers no ownership. Native target
// and source resolution runs after allocation prerequisites. The queue supplies
// its host allocator independently of any target's committed backing.
IREE_API_EXPORT iree_status_t iree_hal_memory_mapping_batch_create(
    iree_hal_queue_t* queue, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings, iree_hal_queue_map_flags_t flags,
    iree_allocator_t host_allocator,
    iree_hal_memory_mapping_batch_t** out_batch);

// Resolves all native targets and sources after prerequisite allocations have
// committed. Validates original-view grants and bounds before translating into
// root coordinates, grouping projections and rejecting overlaps through
// aliases. Failure leaves all native mappings unchanged.
IREE_API_EXPORT iree_status_t
iree_hal_memory_mapping_batch_resolve(iree_hal_memory_mapping_batch_t* batch);

// Captures a batch for the queue's existing cold host-action execution owner.
// The returned call owns one resource reference. Its callback resolves and
// updates after prerequisites; cancellation only releases captured references.
IREE_API_EXPORT iree_status_t iree_hal_memory_mapping_prepare(
    iree_hal_queue_t* queue, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings, iree_hal_queue_map_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_host_call_t* out_call);

// Runs synchronous native updates through an existing queue host-call owner.
// Native queue implementations provide queue.map for queue-ordered publication.
IREE_API_EXPORT iree_status_t iree_hal_memory_mapping_queue_map(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VIRTUAL_BUFFER_H_
