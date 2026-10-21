// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_MEMORY_H_
#define AMDF_SRC_GPU_UMD_MEMORY_H_

#include "amdf/amdf.h"
#include "libamdf/src/memory_pair.h"
#include "libamdf/src/memory_profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_gpu_umd_memory_t amdf_gpu_umd_memory_t;
typedef struct amdf_gpu_umd_host_mapping_t amdf_gpu_umd_host_mapping_t;
typedef struct amdf_gpu_umd_device_t amdf_gpu_umd_device_t;

// Native GPU memory properties established before publication.
typedef struct amdf_gpu_umd_memory_result_t {
  // Achieved attachment properties.
  amdf_memory_flags_t flags;
  // Byte offset of logical byte zero in the physical backing.
  uint64_t source_byte_offset;
  // Logical attachment length in bytes.
  uint64_t byte_length;
  // Guaranteed allocation-base alignment in every supported address space.
  uint64_t alignment;
  // Complete native physical allocation or registered page-cover length.
  uint64_t native_allocation_byte_length;
  // Granularity of the native allocation length.
  uint64_t native_allocation_granularity;
  // Identity of the physical backing within the provider instance.
  amdf_physical_memory_id_t physical_backing_id;
  // Stable GPU virtual base.
  uint64_t device_address;
} amdf_gpu_umd_memory_result_t;

// Native host mapping properties established before publication.
typedef struct amdf_gpu_umd_host_mapping_result_t {
  // Achieved host access flags.
  amdf_memory_map_flags_t flags;
  // First mapped byte.
  void* pointer;
  // Mapped byte length.
  uint64_t byte_length;
  // Immutable host visibility policy shared with pre-allocation queries.
  amdf_memory_host_description_t visibility;
} amdf_gpu_umd_host_mapping_result_t;

// Copies one immutable memory profile supported by `device`.
amdf_status_t amdf_gpu_umd_device_query_memory_profile(
    amdf_gpu_umd_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile);

// Prepares physical backing and native access in an already-live owner's
// initially NULL `memory_state` slot. This is a one-shot transition: native
// progress remains in that slot on failure for explicit destruction. Failure
// before metadata allocation leaves the slot NULL. Only success publishes
// complete properties to `out_result`; no rollback occurs inside preparation.
// Successful preparation upholds each resolved consumer profile's atomic
// operation contract; those masks remain per-consumer profile properties.
// Peers are a borrowed span from the selected allocation domain in the same
// instance, with identical exact permissions. A zero count supplies NULL.
amdf_status_t amdf_gpu_umd_memory_prepare(
    amdf_gpu_umd_device_t* device, uint32_t peer_count,
    amdf_gpu_umd_device_t* const* peer_devices,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result);

// Prepares native access to borrowed external memory with the same ownership
// protocol as memory_prepare. Acquired native references are independent of the
// input's release obligation. This operation never invokes its release
// callback; the public construction owner consumes the move only after complete
// success.
amdf_status_t amdf_gpu_umd_memory_prepare_import(
    amdf_gpu_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result);

// Exports an owned native payload and its release callback. The public owner
// fills transport type, logical range, backing identity and provenance from its
// validated request and profile. Failure leaves the output unchanged.
amdf_status_t amdf_gpu_umd_memory_export(
    amdf_gpu_umd_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value);

// Borrows the eagerly prepared host-alias source of backing constructed with
// HOST_MAPPING_SOURCE. This infallible query performs no native operation.
uintptr_t amdf_gpu_umd_memory_host_mapping_source(
    amdf_gpu_umd_memory_t* memory);

// Describes immutable access facts against an exact local queue family.
// This shared query touches no native state and performs no allocation, lock,
// system call or lazy initialization. Unqualified cache semantics return
// UNSUPPORTED without publishing a partial description.
amdf_status_t amdf_gpu_umd_memory_describe_site(
    const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description);

// Describes owned SYSTEM backing with qualified CPU/CP control-word atomics.
// Shared STORE support establishes the system serialization domain for each
// width; callers separately intersect the memory and queue operation masks.
// Exact coherent read/write access and a PM4 or AQL queue are required. Other
// sites inherit the ordinary visibility description without atomic reach.
amdf_status_t amdf_gpu_umd_memory_describe_system_store_site(
    const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description);

// Releases partial or complete native state and its metadata. Failure leaves
// partial state for the common owner to abandon after preserving required
// backing; this is not a surviving public memory handle.
amdf_status_t amdf_gpu_umd_memory_destroy(amdf_gpu_umd_memory_t* memory);

// Remaining payload charge for an allocating owner, including partial
// preparation or release. Reads retained native ownership without native calls.
// Empty address reservations do not retain payload. The caller must not use
// this allocation accounting query for registered or imported source storage.
uint64_t amdf_gpu_umd_memory_remaining_backing_byte_length(
    const amdf_gpu_umd_memory_t* memory);

// Consumes remaining metadata without attempting any native operation.
// After terminal cleanup failure, native resources and dependent backing leak;
// the caller preserves any separately owned backing those resources can reach.
void amdf_gpu_umd_memory_abandon(amdf_gpu_umd_memory_t* memory);

// Borrows a view of the memory's persistent native host mapping. Current native
// providers need no allocation or system call; the common host view owns the
// borrow and keeps the memory live by caller contract, not by reference count.
amdf_status_t amdf_gpu_umd_memory_map(
    amdf_gpu_umd_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info,
    amdf_gpu_umd_host_mapping_t** out_mapping,
    amdf_gpu_umd_host_mapping_result_t* out_result);

// Performs one host cache operation over a validated memory-relative range.
// The common mapping boundary translates its view-relative offset once.
amdf_status_t amdf_gpu_umd_host_mapping_cache_control(
    amdf_gpu_umd_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t memory_byte_offset, uint64_t byte_length);

// Releases a lightweight host view; its memory owns the native mapping.
void amdf_gpu_umd_host_mapping_destroy(amdf_gpu_umd_host_mapping_t* mapping);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_MEMORY_H_
