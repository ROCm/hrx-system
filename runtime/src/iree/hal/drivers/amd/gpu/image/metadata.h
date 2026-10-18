// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable GPU kernel ABI and reflection records. All string views borrow from
// the source retained by their owner; these records contain no native
// resources.

#ifndef IREE_HAL_DRIVERS_AMD_GPU_IMAGE_METADATA_H_
#define IREE_HAL_DRIVERS_AMD_GPU_IMAGE_METADATA_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kernel argument ABI classification from AMDGPU metadata `.value_kind`.
typedef enum iree_hal_amd_gpu_metadata_argument_kind_e {
  // Unknown or unsupported value kind.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_UNKNOWN = 0,
  // Argument bytes are copied directly into the kernarg segment.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_BY_VALUE,
  // Argument is a pointer to global memory.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_GLOBAL_BUFFER,
  // Argument is a pointer to dynamically allocated LDS.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_DYNAMIC_SHARED_POINTER,
  // Argument is an image descriptor pointer.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_IMAGE,
  // Argument is a sampler descriptor pointer.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_SAMPLER,
  // Argument is an OpenCL pipe pointer.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_PIPE,
  // Argument is an OpenCL device enqueue queue pointer.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_QUEUE,
  // Argument reserves hidden ABI space but does not need a value.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_NONE,
  // Global work-item offset in the X dimension.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GLOBAL_OFFSET_X,
  // Global work-item offset in the Y dimension.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GLOBAL_OFFSET_Y,
  // Global work-item offset in the Z dimension.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GLOBAL_OFFSET_Z,
  // Runtime printf buffer pointer.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_PRINTF_BUFFER,
  // Hostcall service buffer pointer.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_HOSTCALL_BUFFER,
  // Default device-enqueue queue pointer.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_DEFAULT_QUEUE,
  // Device-enqueue ancestor completion state pointer.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_COMPLETION_ACTION,
  // Cooperative multigrid synchronization state pointer.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_MULTIGRID_SYNC_ARG,
  // Number of complete workgroups in the X dimension.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_X,
  // Number of complete workgroups in the Y dimension.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_Y,
  // Number of complete workgroups in the Z dimension.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_BLOCK_COUNT_Z,
  // Full workgroup size in the X dimension.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GROUP_SIZE_X,
  // Full workgroup size in the Y dimension.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GROUP_SIZE_Y,
  // Full workgroup size in the Z dimension.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GROUP_SIZE_Z,
  // Work-items in the partial X workgroup, or zero when absent.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_REMAINDER_X,
  // Work-items in the partial Y workgroup, or zero when absent.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_REMAINDER_Y,
  // Work-items in the partial Z workgroup, or zero when absent.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_REMAINDER_Z,
  // Grid dimensionality, between one and three.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_GRID_DIMS,
  // Device malloc/free V1 heap service pointer.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_HEAP_V1,
  // Dynamically allocated LDS size in bytes.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_DYNAMIC_LDS_SIZE,
  // High 32 bits of the private aperture base on GFX8.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_PRIVATE_BASE,
  // High 32 bits of the shared aperture base on GFX8.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_SHARED_BASE,
  // Native queue ABI pointer for pre-GFX9 trap handling.
  IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_QUEUE_PTR,
} iree_hal_amd_gpu_metadata_argument_kind_t;

// Returns true for known runtime-supplied arguments, including reserved slots.
// Unknown value kinds remain UNKNOWN and require explicit admission by a
// caller.
static inline bool iree_hal_amd_gpu_metadata_argument_is_hidden(
    iree_hal_amd_gpu_metadata_argument_kind_t kind) {
  return kind >= IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_NONE;
}

// Source-level address space. Unspecified is distinct from generic addressing.
typedef enum iree_hal_amd_gpu_metadata_address_space_e {
  IREE_HAL_AMD_GPU_METADATA_ADDRESS_SPACE_UNSPECIFIED = 0,
  IREE_HAL_AMD_GPU_METADATA_ADDRESS_SPACE_PRIVATE,
  IREE_HAL_AMD_GPU_METADATA_ADDRESS_SPACE_GLOBAL,
  IREE_HAL_AMD_GPU_METADATA_ADDRESS_SPACE_CONSTANT,
  IREE_HAL_AMD_GPU_METADATA_ADDRESS_SPACE_LOCAL,
  IREE_HAL_AMD_GPU_METADATA_ADDRESS_SPACE_GENERIC,
  IREE_HAL_AMD_GPU_METADATA_ADDRESS_SPACE_REGION,
} iree_hal_amd_gpu_metadata_address_space_t;

// Argument memory access, independent of HAL buffer usage/placement policy.
typedef enum iree_hal_amd_gpu_metadata_access_e {
  // No qualifier supplied; effective access must be derived conservatively.
  IREE_HAL_AMD_GPU_METADATA_ACCESS_UNSPECIFIED = 0,
  IREE_HAL_AMD_GPU_METADATA_ACCESS_READ_ONLY = 1,
  IREE_HAL_AMD_GPU_METADATA_ACCESS_WRITE_ONLY = 2,
  IREE_HAL_AMD_GPU_METADATA_ACCESS_READ_WRITE = 3,
} iree_hal_amd_gpu_metadata_access_t;

// Decoded kernel argument metadata.
typedef struct iree_hal_amd_gpu_metadata_argument_t {
  // Source-level argument name from `.name`, if present.
  iree_string_view_t name;
  // Byte offset in the kernel's kernarg segment.
  uint32_t offset;
  // Byte length of the argument payload within the kernarg segment.
  uint32_t size;
  // Storage alignment in bytes when explicitly available; otherwise 0.
  uint32_t alignment;
  // Parsed classification of |value_kind|.
  iree_hal_amd_gpu_metadata_argument_kind_t kind;
  // Raw `.value_kind` string view borrowed from the metadata blob.
  iree_string_view_t value_kind;
  // Source type name, if supplied.
  iree_string_view_t type_name;
  // Deprecated descriptive value type, if supplied by an older producer.
  iree_string_view_t value_type;
  // Required alignment of the pointee in bytes, or zero when unspecified.
  uint32_t pointee_alignment;
  // Source-level address-space qualifier.
  iree_hal_amd_gpu_metadata_address_space_t address_space;
  // Declared source-level access qualifier.
  iree_hal_amd_gpu_metadata_access_t access;
  // More precise access determined by the compiler, when supplied.
  iree_hal_amd_gpu_metadata_access_t actual_access;
  // Effective access after applying actual access, declared access, and const.
  iree_hal_amd_gpu_metadata_access_t effective_access;
  // Source-level argument qualifiers.
  struct {
    // Pointee data is const qualified.
    bool is_const;
    // Pointer is restrict qualified.
    bool is_restrict;
    // Pointee accesses are volatile qualified.
    bool is_volatile;
    // Argument is pipe qualified.
    bool is_pipe;
  } qualifiers;
} iree_hal_amd_gpu_metadata_argument_t;

// Kernel invocation phase imposed by the code object.
typedef enum iree_hal_amd_gpu_metadata_kernel_kind_e {
  // Application entry point.
  IREE_HAL_AMD_GPU_METADATA_KERNEL_KIND_NORMAL = 0,
  // Completes after loading and before any normal or finalization kernel.
  IREE_HAL_AMD_GPU_METADATA_KERNEL_KIND_INITIALIZER,
  // Completes after normal kernels and before unloading the code object.
  IREE_HAL_AMD_GPU_METADATA_KERNEL_KIND_FINALIZER,
} iree_hal_amd_gpu_metadata_kernel_kind_t;

// Decoded per-kernel metadata.
typedef struct iree_hal_amd_gpu_metadata_kernel_t {
  // Source-level kernel name from `.name`, if present.
  iree_string_view_t name;
  // Kernel descriptor symbol name from `.symbol`, usually `foo.kd`.
  iree_string_view_t symbol_name;
  // Export reflection name from `.name` or `.symbol` with a `.kd` suffix
  // removed.
  iree_string_view_t reflection_name;
  // Bytes required to clone all argument names for this kernel.
  iree_host_size_t argument_name_storage_size;
  // Kernel kernarg segment size from `.kernarg_segment_size`.
  uint32_t kernarg_segment_size;
  // Kernel kernarg segment alignment from `.kernarg_segment_align`.
  uint32_t kernarg_segment_alignment;
  // Fixed group segment size from `.group_segment_fixed_size`.
  uint32_t group_segment_fixed_size;
  // Fixed private segment size from `.private_segment_fixed_size`.
  uint32_t private_segment_fixed_size;
  // Maximum total work-items per workgroup from `.max_flat_workgroup_size`.
  uint32_t max_flat_workgroup_size;
  // Vector registers allocated per work-item from `.vgpr_count`.
  uint32_t vgpr_count;
  // Required workgroup size from `.reqd_workgroup_size`, if present.
  uint32_t required_workgroup_size[3];
  // True when |required_workgroup_size| constrains dispatch dimensions.
  bool has_required_workgroup_size;
  // True when `.uniform_work_group_size` requires full workgroups.
  bool uniform_workgroup_size;
  // Workgroup-cluster size from `.cluster_dims`, or zeros when omitted.
  uint32_t workgroup_cluster_size[3];
  // True when |workgroup_cluster_size| requires a nontrivial cluster.
  bool has_workgroup_cluster_size;
  // True when `.args` describes the kernarg ABI, including an empty list.
  // False means argument reflection was omitted, not that there are no
  // arguments.
  bool has_argument_metadata;
  // Number of argument records in |arguments|.
  iree_host_size_t argument_count;
  // Argument records borrowed from the owning metadata object.
  const iree_hal_amd_gpu_metadata_argument_t* arguments;
  // Source-language reflection independent of the argument ABI.
  struct {
    // Source language name, if supplied.
    iree_string_view_t language;
    // Source language major and minor versions, or zeros when omitted.
    uint32_t language_version[2];
    // Source scalar/vector type hint, if supplied.
    iree_string_view_t vector_type_hint;
  } source;
  // Device-enqueue runtime symbol to receive the kernel address, if supplied.
  iree_string_view_t device_enqueue_symbol;
  // Module lifecycle invocation phase.
  iree_hal_amd_gpu_metadata_kernel_kind_t kind;
  // Work-items in a wavefront; independent of workgroup size.
  uint32_t wavefront_size;
  // Scalar registers used per wavefront before hardware allocation rounding.
  uint32_t sgpr_count;
  // Accumulator registers used per work-item, or zero when omitted.
  uint32_t agpr_count;
  // Scalar-register stores to compiler-allocated spill locations.
  uint32_t sgpr_spill_count;
  // Vector-register stores to compiler-allocated spill locations.
  uint32_t vgpr_spill_count;
  // Suggested workgroup dimensions; all zeros when unspecified.
  uint32_t workgroup_size_hint[3];
  // Per-axis maximum workgroup counts; zero denotes an unspecified axis.
  uint32_t max_workgroup_count[3];
  // True when dispatch requires stack beyond the fixed private segment.
  bool uses_dynamic_stack;
  // True when workgroup processor mode is selected instead of CU mode.
  bool workgroup_processor_mode;
  // True when the source specified workgroup processor mode explicitly.
  bool has_workgroup_processor_mode;
  // GFX1250 revision required by the producer, such as "A0" or "B0".
  // Empty means no revision annotation. Relevant only to GFX1250 admission.
  iree_string_view_t gfx1250_revision;
} iree_hal_amd_gpu_metadata_kernel_t;

// One printf call's decoded wire layout. Calls retain their numeric IDs so the
// runtime can resolve device output without reparsing metadata strings.
typedef struct iree_hal_amd_gpu_metadata_printf_t {
  // Producer-assigned ID carried by the device printf record.
  uint32_t id;
  // Format string, which may itself contain colons.
  iree_string_view_t format;
  // Number of formatted arguments, excluding the format string.
  uint32_t argument_count;
  // Each formatted argument's size in bytes, borrowed from module storage.
  const uint32_t* argument_sizes;
} iree_hal_amd_gpu_metadata_printf_t;

// Decoded module metadata. One allocation owns all kernel, argument, and printf
// records; strings borrow from the admitted source payloads. The owning image
// keeps those payloads alive until this description is deinitialized.
typedef struct iree_hal_amd_gpu_metadata_t {
  // Allocator releasing the record allocation.
  iree_allocator_t host_allocator;
  // Owned record allocation, independent of any native executable storage.
  void* storage;
  // AMDHSA metadata schema major and minor versions.
  uint32_t version[2];
  // Target ISA name, if supplied by the source.
  iree_string_view_t target;
  // Bytes needed to copy all kernel reflection names without terminators.
  iree_host_size_t reflection_name_storage_size;
  // Bytes needed to copy all argument names without terminators.
  iree_host_size_t argument_name_storage_size;
  // Number of kernel records.
  iree_host_size_t kernel_count;
  // Owned kernel records in source order.
  iree_hal_amd_gpu_metadata_kernel_t* kernels;
  // Total number of argument records referenced by the kernels.
  iree_host_size_t argument_count;
  // Owned contiguous argument records.
  iree_hal_amd_gpu_metadata_argument_t* arguments;
  // Number of printf call records.
  iree_host_size_t printf_count;
  // Owned printf records sorted by ID, with no duplicates.
  iree_hal_amd_gpu_metadata_printf_t* printfs;
} iree_hal_amd_gpu_metadata_t;

// Releases the record allocation without releasing borrowed source bytes.
void iree_hal_amd_gpu_metadata_deinitialize(
    iree_hal_amd_gpu_metadata_t* metadata);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_IMAGE_METADATA_H_
