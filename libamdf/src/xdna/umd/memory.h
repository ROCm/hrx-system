// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MEMORY_H_
#define AMDF_SRC_XDNA_UMD_MEMORY_H_

#include "amdf/amdf.h"
#include "libamdf/src/memory_pair.h"
#include "libamdf/src/memory_profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_xdna_umd_memory_t amdf_xdna_umd_memory_t;
typedef struct amdf_xdna_umd_host_mapping_t amdf_xdna_umd_host_mapping_t;
typedef struct amdf_xdna_umd_device_t amdf_xdna_umd_device_t;
typedef struct amdf_xdna_umd_context_t amdf_xdna_umd_context_t;

// Native memory properties established before publication.
typedef struct amdf_xdna_umd_memory_result_t {
  // Achieved attachment properties.
  amdf_memory_flags_t flags;
  // Atomic operations supported by 32-bit words in this attachment.
  amdf_atomic_operations_t atomic_operations_32;
  // Atomic operations supported by 64-bit words in this attachment.
  amdf_atomic_operations_t atomic_operations_64;
  // Byte offset of logical byte zero in the physical backing.
  uint64_t source_byte_offset;
  // Logical attachment length in bytes.
  uint64_t byte_length;
  // Guaranteed logical-base alignment for every established address kind.
  uint64_t alignment;
  // Complete native physical allocation or registered page-cover length.
  uint64_t native_allocation_byte_length;
  // Granularity of the native allocation length.
  uint64_t native_allocation_granularity;
  // Identity of the physical backing within the provider instance.
  amdf_physical_memory_id_t physical_backing_id;
  // Address of logical byte zero consumed by XDNA firmware interfaces.
  uint64_t device_address;
  // Address kinds established for the complete logical range.
  amdf_memory_address_kinds_t address_kinds;
  // Native-translated shim DMA base when the DMA kind is established.
  uint64_t dma_address;
} amdf_xdna_umd_memory_result_t;

// Native host mapping properties established before publication.
typedef struct amdf_xdna_umd_host_mapping_result_t {
  // Achieved host access flags.
  amdf_memory_map_flags_t flags;
  // First mapped byte.
  void* pointer;
  // Mapped byte length.
  uint64_t byte_length;
  // Host cache behavior of the mapped pages.
  amdf_host_cacheability_t cacheability;
  // Host cache-line length in bytes.
  uint32_t cache_line_size;
  // Available CPU flush operation, independent of device coherence.
  amdf_cache_transition_t flush;
  // Available CPU invalidate operation, independent of device coherence.
  amdf_cache_transition_t invalidate;
} amdf_xdna_umd_host_mapping_result_t;

// Copies one immutable memory profile supported by `device`.
amdf_status_t amdf_xdna_umd_device_query_memory_profile(
    amdf_xdna_umd_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile);

// Copies the complete private storage contract of an admitted context. This
// consumes cached admission facts and performs no native operation.
void amdf_xdna_umd_context_query_memory_profile(
    amdf_xdna_umd_context_t* context,
    amdf_memory_native_profile_t* out_profile);

// Prepares one context-qualified private allocation in the public memory
// owner's slot. The context is borrowed, never retained. Partial native state
// follows the same local rollback protocol as ordinary memory preparation.
amdf_status_t amdf_xdna_umd_memory_prepare_private(
    amdf_xdna_umd_context_t* context,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_xdna_umd_memory_t** memory_state,
    amdf_xdna_umd_memory_result_t* out_result);

// Prepares physical backing and native access in an already-live owner's
// initially NULL `memory_state` slot. This is a one-shot transition: native
// progress remains in that slot on failure for explicit destruction. Failure
// before metadata allocation leaves the slot NULL. Only success publishes
// complete properties to `out_result`; no rollback occurs inside preparation.
amdf_status_t amdf_xdna_umd_memory_prepare(
    amdf_xdna_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_xdna_umd_memory_t** memory_state,
    amdf_xdna_umd_memory_result_t* out_result);

// Prepares native access to borrowed external memory with the same ownership
// protocol as memory_prepare. Acquired native references are independent of the
// input's release obligation. This operation never invokes its release
// callback; the public construction owner consumes the move only after complete
// success.
amdf_status_t amdf_xdna_umd_memory_prepare_import(
    amdf_xdna_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_xdna_umd_memory_t** memory_state,
    amdf_xdna_umd_memory_result_t* out_result);

// Exports an owned native payload and its release callback. The public owner
// fills transport type, logical range, backing identity and provenance from its
// validated request and profile. Failure leaves the output unchanged.
amdf_status_t amdf_xdna_umd_memory_export(
    amdf_xdna_umd_memory_t* memory,
    const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value);

// Describes one concrete attachment and exact local queue family.
amdf_status_t amdf_xdna_umd_memory_describe_site(
    amdf_xdna_umd_memory_t* memory, const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description);

// Releases partial or complete native state and its metadata. Failure retains
// the remaining state with the caller and may leave its backing referenced.
amdf_status_t amdf_xdna_umd_memory_destroy(amdf_xdna_umd_memory_t* memory);

// Consumes unpublished metadata without attempting any native operation.
// After terminal cleanup failure, native resources and dependent backing leak;
// the caller preserves any separately owned backing those resources can reach.
void amdf_xdna_umd_memory_abandon(amdf_xdna_umd_memory_t* memory);

// Creates one explicit host mapping.
amdf_status_t amdf_xdna_umd_memory_map(
    amdf_xdna_umd_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info,
    amdf_xdna_umd_host_mapping_t** out_mapping,
    amdf_xdna_umd_host_mapping_result_t* out_result);

// Performs one host cache ownership transition.
amdf_status_t amdf_xdna_umd_host_mapping_cache_control(
    amdf_xdna_umd_host_mapping_t* mapping,
    amdf_host_cache_operation_t operation, uint64_t byte_offset,
    uint64_t byte_length);

// Releases one explicit host mapping.
amdf_status_t amdf_xdna_umd_host_mapping_destroy(
    amdf_xdna_umd_host_mapping_t* mapping);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_MEMORY_H_
