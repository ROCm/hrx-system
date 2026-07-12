// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_REGION_H_
#define EXPERIMENTAL_ID4_PIPELINE_REGION_H_

#include <stdint.h>

#include "experimental/id4/pipeline/kernel_library.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Current inline tensor rank storage limit in pipeline planning structures.
#define ID4_PIPELINE_TENSOR_MAX_RANK 5

// Ephemeral authoring context for an executable pipeline region.
typedef struct id4_pipeline_region_builder_t id4_pipeline_region_builder_t;

// Sealed executable region prepared from a recording region builder.
typedef struct id4_pipeline_prepared_region_t id4_pipeline_prepared_region_t;

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

// Region dispatch binding behavior flags.
typedef uint32_t id4_pipeline_region_dispatch_binding_flags_t;

// Region dispatch binding behavior flag bits.
typedef enum id4_pipeline_region_dispatch_binding_flag_bits_e {
  // write_range describes the byte interval written by this binding.
  ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_WRITE_RANGE = 1u << 0,
  // read_range describes the byte interval read by this binding.
  ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_READ_RANGE = 1u << 1,
  // This full-tensor write intentionally destroys overlapping alias contents.
  ID4_PIPELINE_REGION_DISPATCH_BINDING_FLAG_DESTRUCTIVE_ALIAS_WRITE = 1u << 2,
} id4_pipeline_region_dispatch_binding_flag_bits_t;

// Tensor byte interval relative to the bound logical tensor.
typedef struct id4_pipeline_region_tensor_byte_range_t {
  // Byte offset from the start of the logical tensor.
  iree_device_size_t offset;
  // Byte length of the interval.
  iree_device_size_t length;
} id4_pipeline_region_tensor_byte_range_t;

// Tensor import flags.
typedef uint32_t id4_pipeline_tensor_import_flags_t;

// Tensor import flag bits.
typedef enum id4_pipeline_tensor_import_flag_bits_e {
  // Imported tensor contents are initialized before the region begins.
  ID4_PIPELINE_TENSOR_IMPORT_FLAG_INITIALIZED = 1u << 0,
} id4_pipeline_tensor_import_flag_bits_t;

// Local tensor subview behavior flags.
typedef uint32_t id4_pipeline_region_subview_flags_t;

// Local tensor subview behavior flag bits.
typedef enum id4_pipeline_region_subview_flag_bits_e {
  // The new logical tensor starts uninitialized regardless of source contents.
  ID4_PIPELINE_REGION_SUBVIEW_FLAG_DISCARD_CONTENTS = 1u << 0,
} id4_pipeline_region_subview_flag_bits_t;

// Local tensor lifetime behavior flags.
typedef uint32_t id4_pipeline_region_local_lifetime_flags_t;

// Local tensor lifetime behavior flag bits.
typedef enum id4_pipeline_region_local_lifetime_flag_bits_e {
  // Tensor storage was acquired from a previously released local slab range.
  ID4_PIPELINE_REGION_LOCAL_LIFETIME_FLAG_REUSED = 1u << 0,
  // Tensor is a logical subview and does not own a local slab range.
  ID4_PIPELINE_REGION_LOCAL_LIFETIME_FLAG_SUBVIEW = 1u << 1,
} id4_pipeline_region_local_lifetime_flag_bits_t;

// Scalar element type for planned tensor values.
typedef enum id4_pipeline_tensor_dtype_e {
  // Invalid element type.
  ID4_PIPELINE_TENSOR_DTYPE_INVALID = 0,
  // IEEE 754 single-precision floating point.
  ID4_PIPELINE_TENSOR_DTYPE_F32 = 1,
  // IEEE 754 half-precision floating point.
  ID4_PIPELINE_TENSOR_DTYPE_F16 = 2,
  // Brain floating point 16-bit value.
  ID4_PIPELINE_TENSOR_DTYPE_BF16 = 3,
  // Signed 32-bit integer.
  ID4_PIPELINE_TENSOR_DTYPE_I32 = 4,
  // Unsigned 32-bit integer.
  ID4_PIPELINE_TENSOR_DTYPE_U32 = 5,
  // E4M3 8-bit floating point value.
  ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3 = 6,
} id4_pipeline_tensor_dtype_t;

// Inline tensor shape used by region planning.
typedef struct id4_pipeline_tensor_shape_t {
  // Number of used dimensions in dims.
  uint32_t rank;
  // Dimension extents stored inline up to the current rank limit.
  uint64_t dims[ID4_PIPELINE_TENSOR_MAX_RANK];
} id4_pipeline_tensor_shape_t;

// Tensor layout requested from a region.
typedef struct id4_pipeline_tensor_layout_t {
  // Human-readable tensor name borrowed for the call and copied by the builder.
  iree_string_view_t name;
  // Scalar element type.
  id4_pipeline_tensor_dtype_t dtype;
  // Tensor shape.
  id4_pipeline_tensor_shape_t shape;
  // Tensor byte length.
  iree_device_size_t byte_length;
  // Required base alignment in bytes. Zero selects byte alignment.
  iree_device_size_t alignment;
} id4_pipeline_tensor_layout_t;

// Returns the dense byte length for one element of |dtype|, or zero if invalid.
iree_device_size_t id4_pipeline_tensor_dtype_byte_length(
    id4_pipeline_tensor_dtype_t dtype);

// Returns the canonical lower-case fixture spelling for |dtype|.
iree_string_view_t id4_pipeline_tensor_dtype_format(
    id4_pipeline_tensor_dtype_t dtype);

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
  // Tensor-relative byte ranges initialized before the region begins.
  const id4_pipeline_region_tensor_byte_range_t* initialized_ranges;
  // Number of entries in initialized_ranges.
  iree_host_size_t initialized_range_count;
} id4_pipeline_tensor_import_t;

// Local tensor lifetime record emitted by a region builder.
typedef struct id4_pipeline_region_local_lifetime_t {
  // Human-readable tensor name owned by the containing record.
  iree_string_view_t name;
  // Builder-local tensor ordinal.
  uint32_t ordinal;
  // Builder-local ordinal of the tensor owning the backing allocation.
  uint32_t storage_root_ordinal;
  // Lifetime behavior flags.
  id4_pipeline_region_local_lifetime_flags_t flags;
  // Scalar element type.
  id4_pipeline_tensor_dtype_t dtype;
  // Tensor shape.
  id4_pipeline_tensor_shape_t shape;
  // Tensor byte length.
  iree_device_size_t byte_length;
  // Required base alignment in bytes.
  iree_device_size_t alignment;
  // Byte offset into the local slab binding.
  iree_device_size_t offset;
  // Byte offset from the storage root to this logical tensor.
  iree_device_size_t storage_byte_offset;
  // Operation ordinal where the tensor was acquired.
  iree_host_size_t acquire_operation_ordinal;
  // Epoch where the tensor was acquired.
  uint32_t acquire_epoch;
  // Operation ordinal where the tensor was released, or IREE_HOST_SIZE_MAX.
  iree_host_size_t release_operation_ordinal;
  // Epoch where the tensor was released, or UINT32_MAX.
  uint32_t release_epoch;
  // Operation ordinal where the root storage became dead.
  iree_host_size_t storage_release_operation_ordinal;
  // Epoch where the root storage became dead.
  uint32_t storage_release_epoch;
} id4_pipeline_region_local_lifetime_t;

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

// Loom kernel dispatch descriptor consumed by model authoring code.
typedef struct id4_pipeline_region_loom_kernel_t {
  // Stable specialization key for diagnostics, plans, and executable lookup.
  iree_string_view_t specialization_key;
  // Stable module path resolved through the kernel library.
  iree_string_view_t module_path;
  // Exported kernel function name inside the module.
  iree_string_view_t function_name;
  // HAL executable required when the builder is in RECORD mode.
  iree_hal_executable_t* executable;
  // HAL executable function required when the builder is in RECORD mode.
  iree_hal_executable_function_t function;
  // Exact tensor binding count expected by the kernel ABI.
  iree_host_size_t binding_count;
  // Exact constant byte length expected by the kernel ABI.
  iree_host_size_t constant_byte_length;
  // Number of Loom config bindings used for this specialization.
  iree_host_size_t config_binding_count;
  // Loom config bindings borrowed for the dispatch call.
  const id4_pipeline_kernel_config_binding_t* config_bindings;
} id4_pipeline_region_loom_kernel_t;

// Planned Loom kernel specialization authored into a region.
typedef struct id4_pipeline_region_kernel_plan_t {
  // Stable specialization key.
  iree_string_view_t specialization_key;
  // Stable module path resolved through the kernel library.
  iree_string_view_t module_path;
  // Exported kernel function name inside the module.
  iree_string_view_t function_name;
  // Number of copied Loom config bindings.
  iree_host_size_t config_binding_count;
  // Copied Loom config bindings.
  const id4_pipeline_kernel_config_binding_t* config_bindings;
} id4_pipeline_region_kernel_plan_t;

// Tensor binding passed to one region dispatch.
typedef struct id4_pipeline_region_dispatch_binding_t {
  // Tensor bound to the dispatch argument.
  id4_pipeline_tensor_t tensor;
  // Access performed by the dispatch.
  id4_pipeline_tensor_access_flags_t access;
  // Dispatch binding behavior flags.
  id4_pipeline_region_dispatch_binding_flags_t flags;
  // Tensor-relative byte range covered by reads when READ_RANGE is set.
  id4_pipeline_region_tensor_byte_range_t read_range;
  // Tensor-relative byte range covered by writes when WRITE_RANGE is set.
  id4_pipeline_region_tensor_byte_range_t write_range;
} id4_pipeline_region_dispatch_binding_t;

// Region statistics accumulated by dry-run or record builders.
typedef struct id4_pipeline_region_statistics_t {
  // Number of dispatch and barrier operations authored.
  iree_host_size_t operation_count;
  // Number of dispatch operations authored.
  iree_host_size_t dispatch_count;
  // Number of tensor copy operations authored.
  iree_host_size_t copy_count;
  // Number of barrier operations authored.
  iree_host_size_t barrier_count;
  // Current epoch after authored barriers.
  uint32_t current_epoch;
  // Number of local tensor acquire operations.
  iree_host_size_t local_acquire_count;
  // Number of local tensor subview operations.
  iree_host_size_t local_subview_count;
  // Number of local tensor release operations.
  iree_host_size_t local_release_count;
  // Number of local slab ranges reused after an epoch boundary.
  iree_host_size_t local_reuse_count;
  // Number of imported bound tensors.
  iree_host_size_t bound_import_count;
  // Local slab byte length required by the region.
  iree_device_size_t local_slab_byte_length;
  // Peak concurrently live local tensor bytes.
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

// Options for sealing a record-mode builder as a prepared region.
typedef struct id4_pipeline_prepared_region_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Device group retained while the prepared region can be issued.
  iree_hal_device_group_t* device_group;
  // Device-group index used to issue the prepared region.
  iree_host_size_t device_index;
  // Queue affinity used for alloca, execute, and dealloca operations.
  iree_hal_queue_affinity_t queue_affinity;
  // HAL allocation-pool operand passed to queue_alloca for the local slab.
  iree_hal_pool_t* local_slab_pool;
  // HAL buffer parameters used for the local transient slab.
  iree_hal_buffer_params_t local_slab_params;
  // HAL queue-alloca flags for the local transient slab.
  iree_hal_alloca_flags_t local_slab_alloca_flags;
  // HAL queue-dealloca flags for the local transient slab.
  iree_hal_dealloca_flags_t local_slab_dealloca_flags;
} id4_pipeline_prepared_region_create_options_t;

// Options for issuing a prepared region once.
typedef struct id4_pipeline_prepared_region_issue_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Timeline semaphores that must be reached before region issue begins.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Timeline semaphores signaled after local transient deallocation completes.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Issue-time binding table. The local slab slot must be empty.
  iree_hal_buffer_binding_table_t binding_table;
  // HAL queue-execute flags.
  iree_hal_execute_flags_t execute_flags;
} id4_pipeline_prepared_region_issue_options_t;

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

// Selects the borrowed HAL command buffer receiving subsequent record-mode
// operations. Switching after operations have been authored requires an idle
// epoch produced by an execution barrier.
iree_status_t id4_pipeline_region_builder_set_recording_command_buffer(
    id4_pipeline_region_builder_t* builder,
    iree_hal_command_buffer_t* command_buffer);

// Returns the current builder epoch.
uint32_t id4_pipeline_region_builder_current_epoch(
    const id4_pipeline_region_builder_t* builder);

// Copies accumulated region statistics into |out_statistics|.
void id4_pipeline_region_builder_statistics(
    const id4_pipeline_region_builder_t* builder,
    id4_pipeline_region_statistics_t* out_statistics);

// Returns the number of authored Loom kernel specializations.
iree_host_size_t id4_pipeline_region_builder_kernel_count(
    const id4_pipeline_region_builder_t* builder);

// Returns authored Loom kernel specialization |index| or NULL when out of
// range.
const id4_pipeline_region_kernel_plan_t* id4_pipeline_region_builder_kernel_at(
    const id4_pipeline_region_builder_t* builder, iree_host_size_t index);

// Returns the number of local tensor lifetime records.
iree_host_size_t id4_pipeline_region_builder_local_lifetime_count(
    const id4_pipeline_region_builder_t* builder);

// Clones local tensor lifetime records into |host_allocator|.
iree_status_t id4_pipeline_region_builder_clone_local_lifetimes(
    const id4_pipeline_region_builder_t* builder,
    iree_allocator_t host_allocator, iree_host_size_t* out_lifetime_count,
    id4_pipeline_region_local_lifetime_t** out_lifetimes);

// Releases local tensor lifetime records cloned from a region builder.
void id4_pipeline_region_local_lifetime_list_release(
    iree_host_size_t lifetime_count,
    id4_pipeline_region_local_lifetime_t* lifetimes,
    iree_allocator_t host_allocator);

// Acquires an uninitialized local transient tensor from the region slab.
iree_status_t id4_pipeline_region_acquire_tensor(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_tensor_layout_t* layout,
    id4_pipeline_tensor_t* out_tensor);

// Creates a logical tensor view into an acquired local tensor.
iree_status_t id4_pipeline_region_subview_tensor(
    id4_pipeline_region_builder_t* builder, id4_pipeline_tensor_t source,
    const id4_pipeline_tensor_layout_t* layout,
    iree_device_size_t source_byte_offset,
    id4_pipeline_region_subview_flags_t flags,
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

// Authors a Loom dispatch and records the specialization in the region plan.
iree_status_t id4_pipeline_region_dispatch_loom(
    id4_pipeline_region_builder_t* builder,
    const id4_pipeline_region_loom_kernel_t* kernel,
    iree_hal_dispatch_config_t dispatch_config,
    iree_const_byte_span_t constants, iree_host_size_t binding_count,
    const id4_pipeline_region_dispatch_binding_t* bindings,
    iree_hal_dispatch_flags_t flags);

// Authors a tensor-to-tensor copy and optionally records it.
iree_status_t id4_pipeline_region_copy_tensor(
    id4_pipeline_region_builder_t* builder, id4_pipeline_tensor_t source,
    id4_pipeline_tensor_t target, iree_hal_copy_flags_t flags);

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

// Seals a record-mode builder into an immutable prepared region.
iree_status_t id4_pipeline_prepared_region_create(
    const id4_pipeline_region_builder_t* builder,
    const id4_pipeline_prepared_region_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_prepared_region_t** out_prepared_region);

// Retains |prepared_region| for the caller.
void id4_pipeline_prepared_region_retain(
    id4_pipeline_prepared_region_t* prepared_region);

// Releases |prepared_region| from the caller.
void id4_pipeline_prepared_region_release(
    id4_pipeline_prepared_region_t* prepared_region);

// Issues |prepared_region| with explicit caller waits and final signals.
iree_status_t id4_pipeline_prepared_region_issue(
    id4_pipeline_prepared_region_t* prepared_region,
    const id4_pipeline_prepared_region_issue_options_t* options);

// Copies realized region statistics into |out_statistics|.
void id4_pipeline_prepared_region_statistics(
    const id4_pipeline_prepared_region_t* prepared_region,
    id4_pipeline_region_statistics_t* out_statistics);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_REGION_H_
