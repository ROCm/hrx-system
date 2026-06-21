// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_REGION_H_
#define EXPERIMENTAL_ID4_PIPELINE_REGION_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maximum tensor rank represented inline in pipeline planning structures.
#define ID4_PIPELINE_TENSOR_MAX_RANK 4

// Ephemeral authoring context for an executable pipeline region.
typedef struct id4_pipeline_region_builder_t id4_pipeline_region_builder_t;

// Region builder mode.
typedef enum id4_pipeline_region_builder_mode_e {
  // Runs planning, lifetime validation, and statistics without HAL recording.
  ID4_PIPELINE_REGION_BUILDER_MODE_DRY_RUN = 0,
  // Runs planning and records effects into a borrowed HAL command buffer.
  ID4_PIPELINE_REGION_BUILDER_MODE_RECORD = 1,
} id4_pipeline_region_builder_mode_t;

// Region builder behavior flags.
typedef uint32_t id4_pipeline_region_builder_flags_t;

// Region builder behavior flag bits.
typedef enum id4_pipeline_region_builder_flag_bits_e {
  // Disables local slab range reuse after release and epoch advancement.
  ID4_PIPELINE_REGION_BUILDER_FLAG_DISABLE_LOCAL_REUSE = 1u << 0,
} id4_pipeline_region_builder_flag_bits_t;

// Tensor storage class selected by a tensor handle.
typedef enum id4_pipeline_tensor_storage_class_e {
  // Invalid tensor handle.
  ID4_PIPELINE_TENSOR_STORAGE_CLASS_INVALID = 0,
  // Region-local transient tensor acquired from the local slab.
  ID4_PIPELINE_TENSOR_STORAGE_CLASS_LOCAL = 1,
  // Tensor imported from an existing issue-time binding-table slot.
  ID4_PIPELINE_TENSOR_STORAGE_CLASS_BOUND = 2,
} id4_pipeline_tensor_storage_class_t;

// Tensor access flags observed by a dispatch.
typedef uint32_t id4_pipeline_tensor_access_flags_t;

// Tensor access flag bits.
typedef enum id4_pipeline_tensor_access_flag_bits_e {
  // Dispatch reads the tensor contents.
  ID4_PIPELINE_TENSOR_ACCESS_READ = 1u << 0,
  // Dispatch writes the tensor contents.
  ID4_PIPELINE_TENSOR_ACCESS_WRITE = 1u << 1,
} id4_pipeline_tensor_access_flag_bits_t;

// Tensor import flags.
typedef uint32_t id4_pipeline_tensor_import_flags_t;

// Tensor import flag bits.
typedef enum id4_pipeline_tensor_import_flag_bits_e {
  // Imported tensor contents are initialized before the region begins.
  ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED = 1u << 0,
} id4_pipeline_tensor_import_flag_bits_t;

// Fixed-rank tensor shape used by region planning.
typedef struct id4_pipeline_tensor_shape_t {
  // Number of used dimensions in dims.
  uint32_t rank;
  // Dimension extents for ranks up to ID4_PIPELINE_TENSOR_MAX_RANK.
  uint64_t dims[ID4_PIPELINE_TENSOR_MAX_RANK];
} id4_pipeline_tensor_shape_t;

// Tensor layout requested from a region.
typedef struct id4_pipeline_tensor_layout_t {
  // Human-readable tensor name borrowed for the call and copied by the builder.
  iree_string_view_t name;
  // Tensor shape.
  id4_pipeline_tensor_shape_t shape;
  // Tensor byte length.
  iree_device_size_t byte_length;
  // Required base alignment in bytes. Zero selects byte alignment.
  iree_device_size_t alignment;
} id4_pipeline_tensor_layout_t;

// Value handle for a tensor known to a region builder.
typedef struct id4_pipeline_tensor_t {
  // Tensor storage class.
  id4_pipeline_tensor_storage_class_t storage_class;
  // Builder-local tensor ordinal.
  uint32_t ordinal;
  // Issue-time binding-table slot containing this tensor's backing storage.
  uint32_t binding_slot;
  // Byte offset into binding_slot.
  iree_device_size_t offset;
  // Byte length visible to dispatches.
  iree_device_size_t length;
  // Tensor shape.
  id4_pipeline_tensor_shape_t shape;
} id4_pipeline_tensor_t;

// Existing tensor imported into a region from a binding-table slot.
typedef struct id4_pipeline_tensor_import_t {
  // Tensor layout and diagnostic name.
  id4_pipeline_tensor_layout_t layout;
  // Issue-time binding-table slot containing this tensor's backing storage.
  uint32_t binding_slot;
  // Byte offset into binding_slot.
  iree_device_size_t offset;
  // Import behavior flags.
  id4_pipeline_tensor_import_flags_t flags;
} id4_pipeline_tensor_import_t;

// Kernel descriptor consumed by region dispatch.
typedef struct id4_pipeline_region_kernel_t {
  // Stable kernel name used in diagnostics.
  iree_string_view_t name;
  // HAL executable required when the builder is in RECORD mode.
  iree_hal_executable_t* executable;
  // HAL executable function required when the builder is in RECORD mode.
  iree_hal_executable_function_t function;
  // Exact tensor binding count expected by the kernel ABI.
  iree_host_size_t binding_count;
  // Exact constant byte length expected by the kernel ABI.
  iree_host_size_t constant_byte_length;
} id4_pipeline_region_kernel_t;

// Tensor binding passed to one region dispatch.
typedef struct id4_pipeline_region_dispatch_binding_t {
  // Tensor bound to the dispatch argument.
  id4_pipeline_tensor_t tensor;
  // Access performed by the dispatch.
  id4_pipeline_tensor_access_flags_t access;
} id4_pipeline_region_dispatch_binding_t;

// Region statistics accumulated by dry-run or record builders.
typedef struct id4_pipeline_region_statistics_t {
  // Number of dispatch and barrier operations authored.
  iree_host_size_t operation_count;
  // Number of dispatch operations authored.
  iree_host_size_t dispatch_count;
  // Number of barrier operations authored.
  iree_host_size_t barrier_count;
  // Current epoch after authored barriers.
  uint32_t current_epoch;
  // Number of local tensor acquire operations.
  iree_host_size_t local_acquire_count;
  // Number of local tensor release operations.
  iree_host_size_t local_release_count;
  // Number of local slab ranges reused after an epoch boundary.
  iree_host_size_t local_reuse_count;
  // Number of imported bound tensors.
  iree_host_size_t bound_import_count;
  // Local slab byte length required by the region.
  iree_device_size_t local_slab_byte_length;
  // Peak local slab byte length reached by the region.
  iree_device_size_t local_slab_high_water_mark;
} id4_pipeline_region_statistics_t;

// Options for creating a region builder.
typedef struct id4_pipeline_region_builder_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Region name borrowed for the create call and copied by the builder.
  iree_string_view_t region_name;
  // Builder execution mode.
  id4_pipeline_region_builder_mode_t mode;
  // Builder behavior flags.
  id4_pipeline_region_builder_flags_t flags;
  // Arena block pool used for all builder transient allocations.
  iree_arena_block_pool_t* block_pool;
  // HAL command buffer borrowed in RECORD mode and ignored in DRY_RUN mode.
  // Record-mode callers must pass a command buffer that has already begun.
  iree_hal_command_buffer_t* command_buffer;
  // Exact issue-time binding-table capacity for the region.
  iree_host_size_t binding_capacity;
  // Binding-table slot reserved for the local transient slab.
  uint32_t local_binding_slot;
} id4_pipeline_region_builder_create_options_t;

// Creates an ephemeral region builder.
iree_status_t id4_pipeline_region_builder_create(
    const id4_pipeline_region_builder_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_region_builder_t** out_builder);

// Destroys a region builder and releases its transient arena blocks.
void id4_pipeline_region_builder_destroy(
    id4_pipeline_region_builder_t* builder);

// Returns the builder mode.
id4_pipeline_region_builder_mode_t id4_pipeline_region_builder_mode(
    const id4_pipeline_region_builder_t* builder);

// Returns the current builder epoch.
uint32_t id4_pipeline_region_builder_current_epoch(
    const id4_pipeline_region_builder_t* builder);

// Copies accumulated region statistics into |out_statistics|.
void id4_pipeline_region_builder_statistics(
    const id4_pipeline_region_builder_t* builder,
    id4_pipeline_region_statistics_t* out_statistics);

// Acquires an uninitialized local transient tensor from the region slab.
iree_status_t id4_pipeline_region_acquire_tensor(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_tensor_layout_t* layout,
    id4_pipeline_tensor_t* out_tensor);

// Imports an initialized or uninitialized tensor from a binding-table slot.
iree_status_t id4_pipeline_region_import_tensor(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_tensor_import_t* import,
    id4_pipeline_tensor_t* out_tensor);

// Releases a local tensor so its storage can be reused after an epoch boundary.
iree_status_t id4_pipeline_region_release_tensor(
    id4_pipeline_region_builder_t* builder, id4_pipeline_tensor_t tensor);

// Authors a dispatch and optionally records it into the HAL command buffer.
iree_status_t id4_pipeline_region_dispatch(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_region_kernel_t* kernel,
    iree_hal_dispatch_config_t dispatch_config,
    iree_const_byte_span_t constants, iree_host_size_t binding_count,
    const id4_pipeline_region_dispatch_binding_t* bindings,
    iree_hal_dispatch_flags_t flags);

// Authors an execution barrier, advances the epoch, and optionally records it.
iree_status_t id4_pipeline_region_barrier(
    id4_pipeline_region_builder_t* builder,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_REGION_H_
