// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEMORY_RESOURCE_H_
#define AMDF_SRC_MEMORY_RESOURCE_H_

#include "amdf/amdf.h"
#include "libamdf/src/child_tracker.h"
#include "libamdf/src/memory_pair.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_memory_vtable_t {
  // Exports one logical range as a complete owned transport value. Failure
  // releases every partial resource and leaves the result storage empty.
  amdf_status_t (*export_external)(amdf_memory_t* memory,
                                   uint32_t access_ordinal,
                                   const amdf_memory_export_info_t* export_info,
                                   amdf_external_memory_t* out_value);
  // Describes one concrete attachment and exact local queue family.
  // Hot metadata path: reads retained facts without locks, allocation, lazy
  // setup, native queries or scans over other resources or access records.
  amdf_status_t (*describe_site)(
      amdf_memory_t* memory, uint32_t access_ordinal,
      uint32_t queue_family_ordinal,
      amdf_memory_site_description_t* out_description);
  // Creates one explicit host mapping. Failure releases every partial resource;
  // success returns one complete mapping.
  amdf_status_t (*map)(amdf_memory_t* memory, uint32_t access_ordinal,
                       const amdf_host_mapping_capabilities_t* capabilities,
                       const amdf_memory_map_info_t* map_info,
                       amdf_host_mapping_t** out_mapping);
  // Releases the exact native state owned by a memory implementation.
  amdf_status_t (*destroy_native)(amdf_memory_t* memory,
                                  uint32_t access_ordinal);
  // Discards unpublished native metadata without native calls. Any surviving
  // native resources retain their dependent backing after terminal failure.
  void (*abandon_native)(amdf_memory_t* memory, uint32_t access_ordinal);
} amdf_memory_vtable_t;

// Immutable consumer facts, indexed directly by the caller's access ordinal.
typedef struct amdf_memory_access_state_t {
  // Device borrowed without retention or lifetime tracking. The caller keeps
  // it live through successful native memory teardown.
  amdf_device_t* device;
  // Complete consumer properties established before publication.
  amdf_memory_access_info_t info;
  // Cached interface bases; info.address_kinds identifies available entries.
  uint64_t addresses[AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE + 1];
  // Native operations selected before preparation begins.
  const amdf_memory_vtable_t* vtable;
  // Owned native state, including partial preparation, released through vtable.
  void* native;
  // Direct index of the access owning native state for this consumer. Only
  // that slot has a non-NULL native pointer; address metadata remains local.
  uint32_t native_owner_ordinal;
  // Native contract used for this access, distinct from the scope ordinal.
  uint32_t native_profile_ordinal;
} amdf_memory_access_state_t;

struct amdf_memory_t {
  // Host allocator copied for direct terminal teardown.
  amdf_allocator_t host_allocator;
  // Exact allocation scope borrowed without retention. Its execution owner
  // qualifies PRIVATE addresses and must outlive this resource.
  amdf_memory_scope_t* scope;
  // Immutable properties established before publication.
  amdf_memory_info_t info;
  // Caller-ordered records in this allocation's tail. CPU-only memory reserves
  // one private native slot here, with no device or publicly queryable access.
  amdf_memory_access_state_t* accesses;
  // Slot owning the backing and its host view; zero for CPU-only memory.
  uint32_t backing_access_ordinal;
  // Host-view limits selected during construction, independent of consumers.
  amdf_host_mapping_capabilities_t host_mapping;
  // Number of live mappings and commands borrowing this memory.
  amdf_child_tracker_t children;
};

// Registers one child that borrows `memory`. Used by command publication:
// atomic bookkeeping only, without locks, allocation or lazy initialization.
// The shared borrow counter may contend; this is not a wait-free operation.
amdf_status_t amdf_memory_register_child(amdf_memory_t* memory);

// Releases one child borrow. Also runs on no-syscall command retirement paths;
// performs only atomic bookkeeping, with the same contention contract above.
void amdf_memory_unregister_child(amdf_memory_t* memory);

// Returns the host allocator copied by the memory attachment.
amdf_allocator_t amdf_memory_host_allocator(const amdf_memory_t* memory);

// Allocates one owner and its caller-ordered native/access slots. It contains
// no native resources yet and retains no instance or device.
amdf_status_t amdf_memory_resource_allocate(amdf_allocator_t host_allocator,
                                            uint32_t access_count,
                                            amdf_memory_t** out_memory);

// Releases consumers before backing, preserving failed native state.
amdf_status_t amdf_memory_release_native(amdf_memory_t* memory);

// Attempts unpublished rollback once, then consumes the host metadata even on
// terminal native failure. Dependent native backing remains unrecycled.
amdf_status_t amdf_memory_discard(amdf_memory_t* memory);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEMORY_RESOURCE_H_
