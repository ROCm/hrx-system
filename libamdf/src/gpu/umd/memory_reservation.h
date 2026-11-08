// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_MEMORY_RESERVATION_H_
#define AMDF_SRC_GPU_UMD_MEMORY_RESERVATION_H_

#include "libamdf/src/gpu/umd/kernel_queue.h"
#include "libamdf/src/gpu/umd/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_gpu_umd_memory_reservation_t
    amdf_gpu_umd_memory_reservation_t;

// True when both live devices address the same native page tables. This cached
// identity query does not open, acquire or prepare a VM.
bool amdf_gpu_umd_devices_share_mapping_domain(const amdf_gpu_umd_device_t* a,
                                               const amdf_gpu_umd_device_t* b);

typedef struct amdf_gpu_umd_mapping_update_t {
  // Reservation-relative destination offset.
  uint64_t target_byte_offset;
  // Nonzero validated range length.
  uint64_t byte_length;
  // Prepared ordinary native backing, or NULL for detach.
  amdf_gpu_umd_memory_t* memory;
  // Native backing-relative offset, including the logical prefix exactly once.
  uint64_t source_byte_offset;
  // Exact installed permissions, or zero for detach.
  amdf_memory_access_t access;
  // Borrowed prior backing, or NULL for an unbound range. Native publication
  // consumes this source identity without retaining installed mapping state.
  amdf_gpu_umd_memory_t* previous_memory;
} amdf_gpu_umd_mapping_update_t;

// Cached live native-domain limits; unsupported domains leave output unchanged.
amdf_status_t amdf_gpu_umd_query_memory_reservation_capabilities(
    amdf_gpu_umd_device_t* device, uint32_t address_domain_ordinal,
    amdf_memory_reservation_capabilities_t* out_capabilities);

// Reserves one common GPU address for the explicitly live device group.
// A nonzero common_address borrows an already-reserved process interval and
// requires that exact GPU address; its lifetime remains with the caller.
// Failure consumes partial metadata/resources and preserves the output.
amdf_status_t amdf_gpu_umd_memory_reservation_create(
    amdf_allocator_t host_allocator, uint32_t device_count,
    amdf_gpu_umd_device_t* const* devices, uint64_t byte_length,
    uint64_t minimum_alignment, uint64_t common_address,
    amdf_gpu_umd_memory_reservation_t** out_reservation);

// Reads complete immutable address and alias-domain facts established by
// create.
void amdf_gpu_umd_memory_reservation_query_access(
    const amdf_gpu_umd_memory_reservation_t* reservation,
    uint32_t access_ordinal, uint64_t* out_address,
    uint32_t* out_mapping_access_ordinal, uint64_t* out_mapping_granularity,
    amdf_memory_access_t* out_required_access);

// Applies trusted destination-ordered ranges. All native descriptor allocation
// and source admission precedes setting mutation_started and changing the VM.
// The caller owns affected execution ordering and installed backing lifetimes.
amdf_status_t amdf_gpu_umd_memory_reservation_update(
    amdf_gpu_umd_memory_reservation_t* reservation, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    bool* mutation_started);

// Enqueues the same trusted updates on a native queue of the selected consumer.
// Accepted work owns submission even when a subsequent dependency step fails.
amdf_status_t amdf_gpu_umd_memory_reservation_update_on_queue(
    amdf_gpu_umd_memory_reservation_t* reservation,
    amdf_gpu_umd_kernel_queue_t* queue, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    uint64_t submission, bool* mutation_started, bool* accepted);

// Clears the selected native group and publishes translation removal, ordered
// after any accepted host-update prefix. Retains metadata, addresses and native
// publication resources on every result. The caller has retired affected
// execution and queue-ordered mutations. Native failure may leave partial
// state.
amdf_status_t amdf_gpu_umd_memory_reservation_detach(
    amdf_gpu_umd_memory_reservation_t* reservation, uint32_t access_ordinal);

// Checked complete removal before VA release; consumes metadata on any result.
amdf_status_t amdf_gpu_umd_memory_reservation_destroy(
    amdf_gpu_umd_memory_reservation_t* reservation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_MEMORY_RESERVATION_H_
