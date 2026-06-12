// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_METADATA_H_
#define IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_METADATA_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/kernarg_layout.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_executable_metadata_t
//===----------------------------------------------------------------------===//

// Executable metadata source used for diagnostics and optional validation.
typedef enum iree_hal_amdgpu_executable_metadata_source_e {
  // Metadata source is not yet known.
  IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_UNKNOWN = 0,
  // Metadata was populated from an IREE/Loom ELF note.
  IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_IREE_NOTE = 1,
  // Metadata was populated from AMDGPU MessagePack.
  IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_HSACO_MESSAGEPACK = 2,
  // Metadata was populated from ELF kernel symbols without decoded metadata.
  IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_ELF_SYMBOLS = 3,
} iree_hal_amdgpu_executable_metadata_source_t;

// Export metadata flags.
typedef enum iree_hal_amdgpu_executable_export_flag_bits_e {
  // No export flags are set.
  IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_NONE = 0u,
  // Dispatch config must provide a non-zero workgroup size.
  IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_DISPATCH_WORKGROUP_SIZE =
      1u << 0,
  // Export can only be launched with caller-provided native kernargs.
  IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_CUSTOM_DIRECT_ONLY = 1u << 1,
} iree_hal_amdgpu_executable_export_flag_bits_t;
typedef uint32_t iree_hal_amdgpu_executable_export_flags_t;

// Reference to one layout record stored in the metadata layout blob.
typedef struct iree_hal_amdgpu_kernarg_layout_ref_t {
  // Byte offset of the layout record in
  // iree_hal_amdgpu_executable_metadata_t::layout_blob.
  iree_host_size_t byte_offset;
} iree_hal_amdgpu_kernarg_layout_ref_t;

// Returns an invalid layout reference.
static inline iree_hal_amdgpu_kernarg_layout_ref_t
iree_hal_amdgpu_kernarg_layout_ref_invalid(void) {
  return (iree_hal_amdgpu_kernarg_layout_ref_t){
      /*.byte_offset=*/IREE_HOST_SIZE_MAX,
  };
}

// Returns true if |ref| references a layout record.
static inline bool iree_hal_amdgpu_kernarg_layout_ref_is_valid(
    iree_hal_amdgpu_kernarg_layout_ref_t ref) {
  return ref.byte_offset != IREE_HOST_SIZE_MAX;
}

// Count summary used to allocate executable metadata storage.
typedef struct iree_hal_amdgpu_executable_metadata_counts_t {
  // Number of executable exports.
  iree_host_size_t export_count;
  // Number of reflected HAL parameter records.
  iree_host_size_t parameter_count;
  // Total byte capacity required for immutable kernarg layout records.
  iree_host_size_t layout_blob_byte_length;
} iree_hal_amdgpu_executable_metadata_counts_t;

// Cold reflected metadata for one executable export.
typedef struct iree_hal_amdgpu_executable_reflection_t {
  // Public export name used by iree_hal_executable_lookup.
  iree_string_view_t name;
  // Native kernel descriptor symbol name used for HSA symbol lookup.
  iree_string_view_t symbol_name;
  // First parameter record for this export.
  uint32_t parameter_offset;
  // Number of parameter records for this export.
  uint32_t parameter_count;
} iree_hal_amdgpu_executable_reflection_t;

// Hot provider-neutral metadata for one executable export.
typedef struct iree_hal_amdgpu_executable_export_t {
  // Export dispatch/layout feature flags.
  iree_hal_amdgpu_executable_export_flags_t flags;
  // Fixed XYZ workgroup size, or zeroes when dispatch config must provide it.
  uint32_t workgroup_size[3];
  // Fixed group segment byte size reported by executable metadata.
  uint32_t fixed_group_segment_size;
  // Fixed private segment byte size reported by executable metadata.
  uint32_t fixed_private_segment_size;
  // Maximum dynamic group-memory byte count accepted for this export.
  uint32_t max_dynamic_workgroup_local_memory;
  // Native kernarg layout record for normal metadata-described dispatch.
  iree_hal_amdgpu_kernarg_layout_ref_t kernarg_layout;
} iree_hal_amdgpu_executable_export_t;

// Provider-neutral executable metadata populated during executable load.
typedef struct iree_hal_amdgpu_executable_metadata_t {
  // Allocator that owns this metadata allocation.
  iree_allocator_t host_allocator;
  // Metadata source used for diagnostics and optional cross-checking.
  iree_hal_amdgpu_executable_metadata_source_t source;
  // Target ID or ISA string borrowed from loaded executable bytes.
  iree_string_view_t target;
  // Loaded code-object bytes borrowed from HSA loader storage.
  iree_const_byte_span_t code_object_data;
  // Number of executable exports.
  iree_host_size_t export_count;
  // Hot provider-neutral export records.
  iree_hal_amdgpu_executable_export_t* exports;
  // Cold reflection records parallel to exports.
  iree_hal_amdgpu_executable_reflection_t* reflection;
  // Number of reflected HAL parameter records.
  iree_host_size_t parameter_count;
  // Reflected HAL parameter records for all exports.
  iree_hal_executable_function_parameter_t* parameters;
  // Total byte capacity of layout_blob.
  iree_host_size_t layout_blob_capacity;
  // Number of bytes already allocated from layout_blob.
  iree_host_size_t layout_blob_used;
  // Immutable kernarg layout record storage.
  uint8_t* layout_blob;
} iree_hal_amdgpu_executable_metadata_t;

// Calculates the allocation size required for metadata with |counts|.
iree_status_t iree_hal_amdgpu_executable_metadata_storage_size(
    const iree_hal_amdgpu_executable_metadata_counts_t* counts,
    iree_host_size_t* out_storage_byte_length);

// Allocates zero-initialized executable metadata storage sized by |counts|.
iree_status_t iree_hal_amdgpu_executable_metadata_allocate(
    const iree_hal_amdgpu_executable_metadata_counts_t* counts,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_executable_metadata_t** out_metadata);

// Releases metadata storage allocated by
// iree_hal_amdgpu_executable_metadata_allocate.
void iree_hal_amdgpu_executable_metadata_free(
    iree_hal_amdgpu_executable_metadata_t* metadata);

// Allocates one layout record from |metadata|'s layout blob.
iree_status_t iree_hal_amdgpu_executable_metadata_append_layout(
    iree_hal_amdgpu_executable_metadata_t* metadata,
    iree_host_size_t layout_byte_length,
    iree_hal_amdgpu_kernarg_layout_ref_t* out_ref,
    iree_byte_span_t* out_storage);

// Resolves a valid layout reference into immutable metadata-owned storage.
iree_status_t iree_hal_amdgpu_executable_metadata_resolve_layout(
    const iree_hal_amdgpu_executable_metadata_t* metadata,
    iree_hal_amdgpu_kernarg_layout_ref_t ref,
    const iree_hal_amdgpu_kernarg_layout_t** out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_METADATA_H_
