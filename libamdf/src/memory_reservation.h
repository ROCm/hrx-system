// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEMORY_RESERVATION_H_
#define AMDF_SRC_MEMORY_RESERVATION_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_memory_reservation_profile_t
    amdf_memory_reservation_profile_t;

// Explicit caller indices prepared by one native reservation provider.
typedef struct amdf_memory_reservation_group_t {
  // Number of participating consumers, always nonzero.
  uint32_t access_count;
  // Borrowed caller ordinals in their original order.
  const uint32_t* access_ordinals;
  // Borrowed construction facts indexed by original caller ordinal.
  const amdf_memory_reservation_profile_t* profiles;
} amdf_memory_reservation_group_t;

typedef struct amdf_memory_reservation_vtable_t {
  // Prepares a validated group in an unpublished common owner. Every return
  // leaves acquired native state there for terminal construction cleanup.
  amdf_status_t (*prepare)(
      amdf_memory_reservation_t* reservation,
      const amdf_memory_reservation_create_info_t* create_info,
      const amdf_memory_reservation_group_t* group);
  // Applies validated, destination-ordered descriptors. Before native mutation
  // begins, sets mutation_started; failures before that point preserve state.
  amdf_status_t (*update)(amdf_memory_reservation_t* reservation,
                          uint32_t access_ordinal, uint32_t update_count,
                          const amdf_memory_mapping_update_t* updates,
                          bool* mutation_started);
  // Clears one complete native mapping group without releasing its owner or
  // addresses. Native failure poisons the group; no source knowledge is needed.
  amdf_status_t (*detach)(amdf_memory_reservation_t* reservation,
                          uint32_t access_ordinal);
  // Consumes one native owner's state, even on failure. Common metadata and
  // other independent owners remain live until common terminal destruction.
  amdf_status_t (*destroy)(amdf_memory_reservation_t* reservation,
                           uint32_t owner_ordinal);
} amdf_memory_reservation_vtable_t;

// A consumer of the process page tables, including biased device DMA views.
// Native admission establishes that the callback can prepare every page of a
// qualified source without modifying payload. NULL leaves ordinary CPU faults.
typedef struct amdf_host_reservation_projection_t {
  // Constant displacement from the process address to the consumer's address.
  uint64_t address_byte_offset;
  // Interpretation of the displaced address.
  amdf_memory_address_kind_t address_kind;
  // Native page size used by prepare_translations, independent of mapping
  // units.
  uint64_t page_size;
  // Populates an installed, pinned source for a device sharing process PTEs.
  void (*prepare_translations)(void* pointer, uint64_t byte_length,
                               uint64_t page_size);
} amdf_host_reservation_projection_t;

// Native limits and preparation protocol, resolved before resource creation.
struct amdf_memory_reservation_profile_t {
  // Complete live-domain geometry and permission envelope.
  amdf_memory_reservation_capabilities_t capabilities;
  // Matching operations prepare compatible consumers together.
  const amdf_memory_reservation_vtable_t* vtable;
  // Native device borrowed by prepare; no reference or initialization is taken.
  void* native_device;
  // Identity of the live consumer; zero for the CPU.
  amdf_device_id_t device_id;
  // Consumer reset epoch; zero for the CPU.
  uint64_t reset_epoch;
  // Process-page-table projection, used only by the host reservation
  // operations.
  amdf_host_reservation_projection_t host;
};

typedef struct amdf_memory_reservation_access_state_t {
  // Live consumer borrowed without retention.
  amdf_device_t* device;
  // Immutable address and alias-domain facts.
  amdf_memory_reservation_access_info_t info;
  // Native source-offset unit shared by the canonical page-table projection;
  // independent of the caller's larger destination replacement unit.
  uint64_t source_granularity;
  // Native permissions that every nonempty mapping must include.
  amdf_memory_access_t required_access;
  // First native mutation failure for the canonical mapping access.
  amdf_status_t failure;
  // Native operations selected for this consumer before preparation.
  const amdf_memory_reservation_vtable_t* vtable;
  // Owned native state, non-NULL only in its native owner record.
  void* native;
  // Caller ordinal owning this consumer's native state.
  uint32_t native_owner_ordinal;
  // Consumer ordinal within that native owner, independent of caller order.
  uint32_t native_access_ordinal;
  // Native host preparation retained only by a process-address-space owner.
  struct {
    // Smallest page unit required by the group's process-PTE consumers.
    uint64_t page_size;
    // Eager PTE preparation, or NULL when every consumer supports CPU faults.
    void (*prepare_translations)(void* pointer, uint64_t byte_length,
                                 uint64_t page_size);
  } host;
} amdf_memory_reservation_access_state_t;

struct amdf_memory_reservation_t {
  // Allocator copied from the instance, not a lifetime reference.
  amdf_allocator_t host_allocator;
  // Number of caller-ordered borrowed consumer records.
  uint32_t access_count;
  // Records stored in this common owner's allocation tail.
  amdf_memory_reservation_access_state_t* accesses;
};

amdf_status_t AMDF_CALL amdf_device_query_memory_reservation_capabilities(
    amdf_device_t* device, uint32_t address_domain_ordinal,
    amdf_memory_reservation_capabilities_t* out_capabilities);

amdf_status_t AMDF_CALL
amdf_instance_query_host_memory_reservation_capabilities(
    amdf_instance_t* instance,
    amdf_memory_reservation_capabilities_t* out_capabilities);

amdf_status_t AMDF_CALL amdf_memory_reservation_create(
    amdf_instance_t* instance,
    const amdf_memory_reservation_create_info_t* create_info,
    amdf_memory_reservation_t** out_reservation);

amdf_status_t AMDF_CALL amdf_memory_reservation_map(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal,
    const amdf_memory_map_info_t* map_info, amdf_host_mapping_t** out_mapping);

amdf_status_t AMDF_CALL amdf_memory_reservation_query_access_info(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal,
    amdf_memory_reservation_access_info_t* out_info);

amdf_status_t AMDF_CALL amdf_memory_reservation_update(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal,
    uint32_t update_count, const amdf_memory_mapping_update_t* updates);

amdf_status_t AMDF_CALL amdf_memory_reservation_detach(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal);

amdf_status_t AMDF_CALL
amdf_memory_reservation_destroy(amdf_memory_reservation_t* reservation);

amdf_status_t AMDF_CALL amdf_kernel_queue_update_memory(
    amdf_kernel_queue_t* queue, amdf_memory_reservation_t* reservation,
    uint32_t access_ordinal, uint32_t update_count,
    const amdf_memory_mapping_update_t* updates, uint64_t* out_submission);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEMORY_RESERVATION_H_
