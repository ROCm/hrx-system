// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_METADATA_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_METADATA_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// AMDGPU Code Object Metadata
//===----------------------------------------------------------------------===//

// Kernel argument ABI classification from AMDGPU metadata `.value_kind`.
typedef enum iree_hal_amdgpu_hsaco_metadata_arg_kind_e {
  // Unknown or unsupported value kind.
  IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_UNKNOWN = 0,
  // Argument bytes are copied directly into the kernarg segment.
  IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE,
  // Argument is a pointer to global memory.
  IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER,
  // Argument is a pointer to dynamically allocated LDS.
  IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_DYNAMIC_SHARED_POINTER,
  // Argument is an image descriptor pointer.
  IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_IMAGE,
  // Argument is a sampler descriptor pointer.
  IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_SAMPLER,
  // Argument is an OpenCL pipe pointer.
  IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_PIPE,
  // Argument is an OpenCL device enqueue queue pointer.
  IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_QUEUE,
  // Argument is a hidden ABI/runtime value.
  IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN,
  // Argument reserves hidden ABI space but does not need a value.
  IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN_NONE,
} iree_hal_amdgpu_hsaco_metadata_arg_kind_t;

// Decoded kernel argument metadata.
typedef struct iree_hal_amdgpu_hsaco_metadata_arg_t {
  // Source-level argument name from `.name`, if present.
  iree_string_view_t name;
  // Byte offset in the kernel's kernarg segment.
  uint32_t offset;
  // Byte length of the kernarg storage for this argument.
  uint32_t size;
  // Storage alignment in bytes when explicitly available; otherwise 0.
  uint32_t alignment;
  // Parsed classification of |value_kind|.
  iree_hal_amdgpu_hsaco_metadata_arg_kind_t kind;
  // Raw `.value_kind` string view borrowed from the metadata blob.
  iree_string_view_t value_kind;
  // Raw `.address_space` string view borrowed from the metadata blob, if any.
  iree_string_view_t address_space;
  // Effective access string view borrowed from the metadata blob, if any.
  // `.actual_access` is preferred over `.access` when both are present.
  iree_string_view_t access;
} iree_hal_amdgpu_hsaco_metadata_arg_t;

// Decoded per-kernel metadata.
typedef struct iree_hal_amdgpu_hsaco_metadata_kernel_t {
  // Source-level kernel name from `.name`, if present.
  iree_string_view_t name;
  // Kernel descriptor symbol name from `.symbol`, usually `foo.kd`.
  iree_string_view_t symbol_name;
  // Export reflection name from `.name` or `.symbol` with a `.kd` suffix
  // removed.
  iree_string_view_t reflection_name;
  // Bytes required to clone all argument names for this kernel.
  iree_host_size_t arg_name_storage_size;
  // Kernel kernarg segment size from `.kernarg_segment_size`.
  uint32_t kernarg_segment_size;
  // Kernel kernarg segment alignment from `.kernarg_segment_align`.
  uint32_t kernarg_segment_alignment;
  // Fixed group segment size from `.group_segment_fixed_size`.
  uint32_t group_segment_fixed_size;
  // Fixed private segment size from `.private_segment_fixed_size`.
  uint32_t private_segment_fixed_size;
  // Required workgroup size from `.reqd_workgroup_size`, if present.
  uint32_t required_workgroup_size[3];
  // True when |required_workgroup_size| was present.
  bool has_required_workgroup_size;
  // Number of argument records in |args|.
  iree_host_size_t arg_count;
  // Argument records borrowed from the owning metadata object.
  const iree_hal_amdgpu_hsaco_metadata_arg_t* args;
} iree_hal_amdgpu_hsaco_metadata_kernel_t;

// ELF kernel symbol with no decoded AMDGPU metadata entry.
typedef struct iree_hal_amdgpu_hsaco_metadata_elf_kernel_symbol_t {
  // Function/export name, usually the symbol name without a `.kd` suffix.
  iree_string_view_t name;
  // Kernel descriptor symbol name used to resolve the HSA kernel object.
  iree_string_view_t symbol_name;
} iree_hal_amdgpu_hsaco_metadata_elf_kernel_symbol_t;

// Decoded AMDGPU code object metadata.
//
// All string views and |message_pack_data| borrow from |elf_data|. Callers must
// keep the ELF bytes alive for as long as this metadata object is in use.
typedef struct iree_hal_amdgpu_hsaco_metadata_t {
  // Allocator used for kernel and argument arrays.
  iree_allocator_t host_allocator;
  // Borrowed ELF bytes used as the source of all string views.
  iree_const_byte_span_t elf_data;
  // Borrowed AMDGPU MessagePack note descriptor payload.
  iree_const_byte_span_t message_pack_data;
  // Borrowed target ISA string from `amdhsa.target`, if present.
  iree_string_view_t target;
  // Bytes required to clone all kernel reflection names.
  iree_host_size_t reflection_name_storage_size;
  // Bytes required to clone all decoded argument names.
  iree_host_size_t arg_name_storage_size;
  // Number of decoded kernels.
  iree_host_size_t kernel_count;
  // Decoded kernel records.
  iree_hal_amdgpu_hsaco_metadata_kernel_t* kernels;
  // Number of ELF kernel symbols that have no decoded metadata entry.
  iree_host_size_t elf_kernel_symbol_count;
  // ELF-only kernel symbols. These are not reflected metadata and must only be
  // used by custom-direct native kernarg launch paths.
  iree_hal_amdgpu_hsaco_metadata_elf_kernel_symbol_t* elf_kernel_symbols;
  // Total number of decoded argument records.
  iree_host_size_t arg_count;
  // Contiguous argument storage referenced by |kernels|.
  iree_hal_amdgpu_hsaco_metadata_arg_t* args;
} iree_hal_amdgpu_hsaco_metadata_t;

// Requirements for materializing export parameter reflection.
typedef struct iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t {
  // Number of HAL-visible parameters after hidden ABI arguments are skipped.
  uint16_t parameter_count;
  // Number of 32-bit constants consumed by reflected by-value parameters.
  uint16_t constant_count;
  // Number of HAL bindings consumed by reflected global-buffer parameters.
  uint16_t binding_count;
  // Bytes required to clone all reflected parameter names for this kernel.
  iree_host_size_t name_storage_size;
} iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t;

// Analysis result for raw HSACO kernels that may already follow the HAL ABI.
typedef struct iree_hal_amdgpu_hsaco_metadata_hal_abi_analysis_t {
  // True when the decoded kernarg metadata exactly matches the HAL ABI.
  bool is_compatible;
  // Requirements for normal HAL reflection when |is_compatible| is true.
  iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t requirements;
} iree_hal_amdgpu_hsaco_metadata_hal_abi_analysis_t;

// Initializes |out_metadata| from a raw AMDGPU ELF code object.
//
// This locates the `AMDGPU`/`NT_AMDGPU_METADATA` note and decodes only the
// fields needed for kernel argument reflection. The parser accepts a normal
// LLVM-produced 64-bit little-endian AMDGPU ELF. It intentionally does not
// implement HIP fat binary, clang offload bundle, or compressed code object
// handling.
iree_status_t iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
    iree_const_byte_span_t elf_data, iree_allocator_t host_allocator,
    iree_hal_amdgpu_hsaco_metadata_t* out_metadata);

// Releases storage owned by |metadata|.
void iree_hal_amdgpu_hsaco_metadata_deinitialize(
    iree_hal_amdgpu_hsaco_metadata_t* metadata);

// Calculates the storage and export-info counts required by the HAL ABI
// reflection projection for |kernel|.
//
// This projection maps `.value_kind == "global_buffer"` arguments to bindings
// using compact binding ordinals and `.value_kind == "by_value"` arguments to
// constants using compact byte offsets. Reflected by-value sizes must be whole
// 32-bit constants. Hidden ABI arguments are skipped. Any other visible
// argument kind fails with IREE_STATUS_INVALID_ARGUMENT so callers do not
// accidentally publish partial reflection for unsupported ABIs. This helper
// does not prove that raw kernarg offsets follow the HAL ABI; raw HSACO callers
// must use iree_hal_amdgpu_hsaco_metadata_analyze_hal_abi first.
iree_status_t
iree_hal_amdgpu_hsaco_metadata_calculate_hal_abi_export_parameter_requirements(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t*
        out_requirements);

// Populates |out_parameters| using the HAL ABI reflection projection.
//
// |parameter_capacity| and |name_storage_capacity| must satisfy the
// requirements returned by
// iree_hal_amdgpu_hsaco_metadata_calculate_hal_abi_export_parameter_requirements.
// Reflected parameter names are cloned into |name_storage| and borrowed by the
// returned parameter records. No NUL terminators are written or required.
iree_status_t iree_hal_amdgpu_hsaco_metadata_populate_hal_abi_export_parameters(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_host_size_t parameter_capacity,
    iree_hal_executable_function_parameter_t* out_parameters,
    iree_host_size_t name_storage_capacity, char* name_storage);

// Analyzes whether |kernel| can be dispatched through the normal HAL ABI.
//
// Compatible raw HSACO kernels must place all global-buffer bindings first as
// tightly packed 64-bit entries, followed by all by-value constants as tightly
// packed 32-bit multiples. Hidden HIP/OpenCL ABI arguments are accepted only as
// a suffix that begins at the 8-byte aligned end of the explicit HAL arguments.
// Any visible native padding, interleaving, unsupported argument kind, or
// binding after a constant makes the kernel incompatible with normal HAL
// dispatch. Incompatible kernels may still be usable through custom-direct
// dispatch if the native projection supports their visible arguments.
iree_status_t iree_hal_amdgpu_hsaco_metadata_analyze_hal_abi(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_hal_amdgpu_hsaco_metadata_hal_abi_analysis_t* out_analysis);

// Calculates the storage and export-info counts required by the native raw
// kernarg reflection projection for |kernel|.
//
// This projection preserves raw kernarg byte offsets for all visible parameters
// and covers the full aligned native kernarg segment as constants so
// custom-direct prepacked argument buffers can be copied without compacting or
// dropping padding/trailing bytes. Use this only for raw HSACO executables.
iree_status_t
iree_hal_amdgpu_hsaco_metadata_calculate_native_kernarg_export_parameter_requirements(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_hal_amdgpu_hsaco_metadata_export_parameter_requirements_t*
        out_requirements);

// Populates |out_parameters| using the native raw kernarg reflection
// projection.
iree_status_t
iree_hal_amdgpu_hsaco_metadata_populate_native_kernarg_export_parameters(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_host_size_t parameter_capacity,
    iree_hal_executable_function_parameter_t* out_parameters,
    iree_host_size_t name_storage_capacity, char* name_storage);

// Finds a decoded kernel by its descriptor symbol name.
iree_status_t iree_hal_amdgpu_hsaco_metadata_find_kernel_by_symbol(
    const iree_hal_amdgpu_hsaco_metadata_t* metadata,
    iree_string_view_t symbol_name,
    const iree_hal_amdgpu_hsaco_metadata_kernel_t** out_kernel);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_HSACO_METADATA_H_
