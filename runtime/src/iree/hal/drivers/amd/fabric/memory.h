// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_H_

#include "amdf/amdf.h"
#include "iree/async/types.h"
#include "iree/hal/device_event.h"
#include "iree/hal/slab_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

// One native backing with an optional persistent host view. Borrowed scopes,
// native devices, and the API table outlive this owner and every device use.
// Addresses belong to prepared binding tables, not to the backing itself.
typedef struct iree_hal_amd_fabric_memory_t {
  // Owned native storage, consumed at deinitialization.
  amdf_memory_t* handle;
  // Owned optional host view, destroyed before the native storage.
  amdf_host_mapping_t* mapping;
  // Borrowed host pointer and logical extent; empty without a host view.
  iree_byte_span_t contents;
  // Actual permissions of the prepared host view, or zero when absent.
  amdf_memory_map_flags_t host_access;
  // Native capabilities of the prepared host view, independent of I/O pins.
  amdf_host_mapping_features_t host_features;
  // Cache class of the native or registered host view, when established.
  iree_hal_host_cacheability_t host_cacheability;
  // Achieved attachment geometry and physical identity, captured once.
  amdf_memory_info_t info;
  // Storage-relative envelope available for host cache maintenance.
  iree_hal_buffer_range_t maintenance_range;
  // Borrowed export policy, dominated by the owning slab provider.
  iree_hal_pool_export_classes_t exports;
  // Owned I/O pins, released before the persistent host view. The consumer
  // attaches prepared entries and joins detachment before native release.
  iree_async_buffer_registration_state_t registrations;
} iree_hal_amd_fabric_memory_t;

// Private suffix after the generic entries of a fabric native binding table.
// Copies borrow the same physical owner and preserve its lifetime contract.
// Translated host bindings establish byte correspondence for nested views.
typedef struct iree_hal_amd_fabric_memory_binding_facts_t {
  // Borrowed physical storage, including its optional prepared host view.
  const iree_hal_amd_fabric_memory_t* memory;
  // Provider-owned native access ordinals indexed by the HAL memory scope ID.
  const uint32_t* scope_access_ordinals;
} iree_hal_amd_fabric_memory_binding_facts_t;

// Prepared file registrations placed with one physical slab. Regions borrow
// the persistent host view and expire before native storage is released.
typedef struct iree_hal_amd_fabric_io_binding_table_t {
  // Borrowed native backing used for exact file/device visibility queries.
  amdf_memory_t* memory;
  // Borrowed persistent ordinary host view admitted for native file I/O.
  amdf_host_mapping_t* mapping;
  // Borrowed regions in native access-ordinal order, NULL for excluded devices.
  // Devices sharing a proactor share one region and one set of retained pins.
  iree_async_region_t** regions;
} iree_hal_amd_fabric_io_binding_table_t;

// Prepared source metadata owned by one physical slab. Native updates receive
// logical offsets while HAL views preserve native coordinates for alignment.
typedef struct iree_hal_amd_fabric_mapping_source_t {
  // Borrowed prepared backing; no installed alias retains this storage.
  const iree_hal_amd_fabric_memory_t* storage;
  // Number of prepared native consumers in |accesses|.
  uint32_t access_count;
  // Borrowed provider-owned consumers in native memory access-ordinal order.
  // A shared translation group may select a different consumer than the HAL
  // scope naming this source; cold mapping batches resolve its exact ordinal.
  const amdf_memory_device_access_t* accesses;
  // Native payload preceding logical source byte zero.
  uint64_t logical_byte_offset;
  // Optional file registrations owned by this same physical slab.
  const iree_hal_amd_fabric_io_binding_table_t* io_bindings;
} iree_hal_amd_fabric_mapping_source_t;

// One queue site's immutable promise, checked before an attachment is exposed.
typedef struct iree_hal_amd_fabric_memory_atomic_access_t {
  // Index in the native device access array.
  uint32_t access_ordinal;
  // Native queue family supplying the promised atomic operations.
  uint32_t queue_family_ordinal;
  // Qualified backing operations and reach after execution-site filtering.
  iree_hal_atomic_operation_capabilities_t operations;
} iree_hal_amd_fabric_memory_atomic_access_t;

// Qualified construction facts copied by a provider. Accesses and scope are
// borrowed; native construction establishes all accesses before publication.
typedef struct iree_hal_amd_fabric_memory_options_t {
  // Scope accepting the selected profile.
  amdf_memory_scope_t* scope;
  // Qualified native construction and mapping capabilities.
  amdf_memory_profile_t profile;
  // Required owned-storage placement; imports preserve their source placement.
  amdf_memory_placement_t placement;
  // Number of entries in |accesses|.
  uint32_t access_count;
  // Complete caller-ordered native device access set.
  const amdf_memory_device_access_t* accesses;
  // Backing properties required during construction.
  amdf_memory_flags_t required_flags;
  // Persistent host access, or zero to omit the mapping entirely.
  amdf_memory_map_flags_t host_access;
  // The persistent CPU view must permit ordinary CPU atomic accesses.
  bool host_atomic_access;
  // Number of exact queue sites requiring atomic attachment admission.
  uint32_t atomic_access_count;
  // Borrowed immutable promises, live through allocation/import preparation.
  const iree_hal_amd_fabric_memory_atomic_access_t* atomic_accesses;
  // Qualified exports; class storage outlives the native memory owner.
  iree_hal_pool_export_classes_t exports;
  // Owned allocation prepares detached I/O pins against its persistent host
  // view. Imports leave registration to their consumer.
  struct {
    // Number of distinct proactors requiring prepared pins on owned backing.
    iree_host_size_t proactor_count;
    // Borrowed proactors, live through native backing and pin release.
    iree_async_proactor_t* const* proactors;
  } io;
} iree_hal_amd_fabric_memory_options_t;

// Converts independent native operation and reach facts. Unknown operation bits
// confer no HAL support; narrower reach never establishes system scope.
iree_hal_atomic_operation_capabilities_t
iree_hal_amd_fabric_memory_translate_atomics(
    amdf_atomic_operations_t operations_32,
    amdf_atomic_operations_t operations_64, amdf_atomic_reach_t reach);

// Maps transport vocabulary only. NONE means libamdf has no representation;
// native profile qualification establishes actual support and provenance.
amdf_external_memory_type_t iree_hal_amd_fabric_memory_external_type(
    iree_hal_external_storage_type_t type);

// Finds one exact transport/provenance/role in a captured native profile.
// The returned entry is borrowed from |profile|; NULL means unsupported.
const amdf_external_memory_support_t*
iree_hal_amd_fabric_memory_external_support(
    const amdf_memory_profile_t* profile,
    iree_hal_external_storage_class_t storage_class,
    amdf_external_memory_support_flags_t required_flags);

// Narrows one captured construction profile to the exact import/export
// geometry. NONE selects owned allocation. Returns false if no extent can
// satisfy the correlated requirements. Transport support is already qualified.
bool iree_hal_amd_fabric_memory_constrain_profile(
    iree_hal_external_storage_class_t source_class,
    iree_hal_pool_export_classes_t exports, amdf_memory_profile_t* profile);

// Qualifies CPU writes to one prospective device access before allocation.
// Options describe owned, host-visible backing and a writable host view.
// A false result means this placement has no qualified publication path;
// native query failures propagate independently. This cold query owns no state.
iree_status_t iree_hal_amd_fabric_memory_query_host_publication(
    const amdf_api_t* api, const iree_hal_amd_fabric_memory_options_t* options,
    uint32_t access_ordinal, uint32_t queue_family_ordinal,
    bool* out_supported);

// Allocates native backing, rounding the extent to the profile granularity.
// A nonzero host_access creates the requested view before publication.
// Requested I/O pins are prepared against that view but remain detached. The
// consumer attaches them before I/O. Failure leaves out_memory empty and
// releases any prepared pins before the native view; cleanup failures report to
// event_sink.
// Optional out_release_info receives unpublished backing loss independently
// of the returned construction status and is initialized on every path.
iree_status_t iree_hal_amd_fabric_memory_allocate(
    const amdf_api_t* api, const iree_hal_amd_fabric_memory_options_t* options,
    uint64_t byte_length, uint64_t alignment,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amd_fabric_memory_t* out_memory,
    amdf_memory_release_info_t* out_release_info);

// Registers caller-owned host storage using a qualified REGISTER profile.
// The provider supplies a borrowed registration view of the source's
// maintenance envelope, with logical byte zero at the registered pointer.
// The caller excludes all cache lines intersecting that envelope during the
// call, including aliases, while native construction prepares cache readiness.
// The source is borrowed throughout, including rollback after attachment.
// Success owns only the native attachment and optional persistent mapping.
iree_status_t iree_hal_amd_fabric_memory_register(
    const amdf_api_t* api, const iree_hal_amd_fabric_memory_options_t* options,
    const iree_hal_external_storage_t* storage,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amd_fabric_memory_t* out_memory);

// Imports a borrowed native transport using a qualified IMPORT profile.
// Success owns an independent attachment. The source transport, including its
// callback, remains entirely with the caller on both success and failure.
iree_status_t iree_hal_amd_fabric_memory_import(
    const amdf_api_t* api, const iree_hal_amd_fabric_memory_options_t* options,
    const iree_hal_external_storage_t* storage,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amd_fabric_memory_t* out_memory);

// Releases I/O pins and native ownership after all uses retire, without waiting
// for them. Prepared registrations must be detached before entry. Native
// release failures are diagnosed; consumed handles are never retried.
// The result distinguishes consumed metadata from relinquished owned backing.
amdf_memory_release_info_t iree_hal_amd_fabric_memory_deinitialize(
    const amdf_api_t* api, iree_hal_device_event_sink_t event_sink,
    iree_hal_amd_fabric_memory_t* memory);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_H_
