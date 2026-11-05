// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SPIRV_H_
#define IREE_HAL_DRIVERS_VULKAN_SPIRV_H_

#include "iree/base/api.h"
#include "iree/hal/executable.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reflected compute entry point inside a SPIR-V module.
typedef struct iree_hal_vulkan_spirv_compute_entry_point_t {
  // NUL-terminated entry point name borrowed from the SPIR-V module words.
  iree_string_view_t name;

  // SPIR-V result id of the entry point function.
  uint32_t id;

  // Static local workgroup size declared by OpExecutionMode LocalSize.
  uint32_t workgroup_size[3];
} iree_hal_vulkan_spirv_compute_entry_point_t;

typedef enum iree_hal_vulkan_spirv_bda_memory_model_e {
  // No BDA-compatible OpMemoryModel was declared.
  IREE_HAL_VULKAN_SPIRV_BDA_MEMORY_MODEL_NONE = 0u,

  // OpMemoryModel PhysicalStorageBuffer64 GLSL450.
  IREE_HAL_VULKAN_SPIRV_BDA_MEMORY_MODEL_GLSL450 = 1u,

  // OpMemoryModel PhysicalStorageBuffer64 Vulkan.
  IREE_HAL_VULKAN_SPIRV_BDA_MEMORY_MODEL_VULKAN = 2u,
} iree_hal_vulkan_spirv_bda_memory_model_t;

typedef enum iree_hal_vulkan_spirv_module_capability_bits_e {
  // No recognized OpCapability declarations were found.
  IREE_HAL_VULKAN_SPIRV_MODULE_CAPABILITY_NONE = 0u,

  // OpCapability PhysicalStorageBufferAddresses.
  IREE_HAL_VULKAN_SPIRV_MODULE_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES =
      0x1u,

  // OpCapability VulkanMemoryModel.
  IREE_HAL_VULKAN_SPIRV_MODULE_CAPABILITY_VULKAN_MEMORY_MODEL = 0x2u,
} iree_hal_vulkan_spirv_module_capability_bits_t;

typedef uint32_t iree_hal_vulkan_spirv_module_capabilities_t;

// Module-wide SPIR-V facts used while preparing Vulkan executables.
typedef struct iree_hal_vulkan_spirv_module_analysis_t {
  // SPIR-V version word from the verified module header.
  uint32_t version;

  // BDA-compatible OpMemoryModel variant declared by the module.
  iree_hal_vulkan_spirv_bda_memory_model_t bda_memory_model;

  // Recognized OpCapability declarations present in the module.
  iree_hal_vulkan_spirv_module_capabilities_t capabilities;

  // Whether any OpDecorate declares DescriptorSet or Binding.
  bool has_descriptor_binding_decorations;

  // Number of OpVariable declarations in the PushConstant storage class.
  iree_host_size_t push_constant_variable_count;

  // Result type id of the sole PushConstant OpVariable, or 0 otherwise.
  uint32_t single_push_constant_pointer_type_id;

  // Whether any OpVariable declares a descriptor-backed storage class.
  bool has_descriptor_storage_class_variables;

  // Number of OpEntryPoint declarations in the GLCompute execution model.
  iree_host_size_t compute_entry_point_count;

  // Byte length needed to copy all compute entry point names with NULs.
  iree_host_size_t compute_entry_point_name_storage_size;
} iree_hal_vulkan_spirv_module_analysis_t;

// Per-binding requirements reflected from SPIR-V BDA metadata.
typedef struct iree_hal_vulkan_spirv_bda_binding_requirement_t {
  // Minimum device address alignment required by the shader.
  uint32_t minimum_alignment;

  // Minimum buffer range byte length required by the shader.
  uint64_t minimum_length;
} iree_hal_vulkan_spirv_bda_binding_requirement_t;

// Compact logical parameter retained from SPIR-V BDA metadata.
typedef struct iree_hal_vulkan_spirv_bda_parameter_t {
  // HAL-visible parameter kind.
  iree_hal_executable_function_parameter_type_t type;

  // Logical byte length excluding raw-BDA slot padding; ignored for bindings.
  uint8_t size;

  // HAL constants-table byte offset or binding ordinal, depending on type.
  uint16_t offset;
} iree_hal_vulkan_spirv_bda_parameter_t;
static_assert(sizeof(iree_hal_vulkan_spirv_bda_parameter_t) == 4,
              "Vulkan BDA parameters must remain compact");

// Per-entry SPIR-V BDA dispatch metadata reflected from OpModuleProcessed
// strings.
typedef struct iree_hal_vulkan_spirv_bda_dispatch_metadata_t {
  // Whether any recognized iree.vulkan.bda.v1 metadata string was found.
  bool is_present;

  // Required subgroup width, or zero when the entry has no fixed-width
  // contract.
  uint32_t subgroup_size;

  // Push-constant byte offset of iree_hal_vulkan_bda_dispatch_root_v1_t.
  uint32_t root_push_constant_offset;

  // Push-constant byte length reserved for the hidden BDA root.
  uint32_t root_push_constant_length;

  // Push-constant byte offset of the first HAL inline constant.
  uint32_t constant_push_constant_offset;

  // Byte length of HAL inline constants accepted by the pipeline.
  uint32_t constant_byte_length;

  // Whether |binding_count| was explicitly declared.
  bool binding_count_known;

  // Number of HAL buffer bindings accepted by the pipeline.
  uint16_t binding_count;

  // Number of entries in |binding_requirements|.
  iree_host_size_t binding_requirement_count;

  // Per-binding requirements owned by this metadata object.
  iree_hal_vulkan_spirv_bda_binding_requirement_t* binding_requirements;

  // Number of logical parameters in |parameters|.
  uint16_t parameter_count;

  // Logical parameters in source order, owned by this metadata object.
  iree_hal_vulkan_spirv_bda_parameter_t* parameters;
} iree_hal_vulkan_spirv_bda_dispatch_metadata_t;

typedef enum iree_hal_vulkan_spirv_bda_verification_flag_bits_e {
  // No additional BDA module verification requirements.
  IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_NONE = 0u,

  // Require the hidden BDA root push-constant block shape.
  IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_REQUIRE_PUSH_CONSTANT_ROOT = 0x1u,
} iree_hal_vulkan_spirv_bda_verification_flag_bits_t;

typedef uint32_t iree_hal_vulkan_spirv_bda_verification_flags_t;

// Verifies the structural SPIR-V header and instruction stream.
iree_status_t iree_hal_vulkan_spirv_verify_module(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count);

// Analyzes module-wide facts while verifying the structural instruction stream.
iree_status_t iree_hal_vulkan_spirv_analyze_module(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_hal_vulkan_spirv_module_analysis_t* out_analysis);

// Verifies the BDA v1 hidden root push-constant block shape.
iree_status_t iree_hal_vulkan_spirv_verify_bda_root_push_constant_layout(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count);

// Verifies that |analysis| and |spirv_words| satisfy the BDA executable ABI.
iree_status_t iree_hal_vulkan_spirv_verify_bda_module_analysis(
    const iree_hal_vulkan_spirv_module_analysis_t* analysis,
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_hal_vulkan_spirv_bda_verification_flags_t verification_flags);

// Verifies that a SPIR-V module satisfies the BDA executable ABI.
iree_status_t iree_hal_vulkan_spirv_verify_bda_module(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_hal_vulkan_spirv_bda_verification_flags_t verification_flags);

// Verifies a BDA executable entry point and returns its local workgroup size.
iree_status_t iree_hal_vulkan_spirv_verify_bda_entry_point(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_string_view_t entry_point,
    iree_hal_vulkan_spirv_bda_verification_flags_t verification_flags,
    uint32_t out_workgroup_size[3]);

// Parses BDA dispatch metadata from OpModuleProcessed strings for all compute
// entries. |entry_points| is the verified result of parse_compute_entry_points;
// |out_metadata| has |entry_point_count| elements in that same entry order.
//
// Recognized strings use these decimal ASCII forms, where <name> is the exact
// exported OpEntryPoint name (not a SPIR-V ID that optimizers may renumber):
//
//   iree.vulkan.bda.v1[<name>]
//   iree.vulkan.bda.v1[<name>].subgroup_size=<invocations>
//   iree.vulkan.bda.v1[<name>].root=<byte_offset>,<byte_length>
//   iree.vulkan.bda.v1[<name>].constant_offset=<byte_offset>
//   iree.vulkan.bda.v1[<name>].constant_length=<byte_length>
//   iree.vulkan.bda.v1[<name>].bindings=<count>
//   iree.vulkan.bda.v1[<name>].binding.<ordinal>=<alignment>,<minimum_length>
//   iree.vulkan.bda.parameters.v1[<name>]=<parameter>[;<parameter>...]
//
// Logical parameter tokens are `b<binding_ordinal>` for bindings and
// `c<constant_byte_offset>,<logical_byte_length>` for constants. Constant
// offsets name packed 32-bit raw-BDA slots while the byte length excludes slot
// padding. Token order is source order. An empty value declares an empty
// logical signature. The independent extension namespace lets runtimes that
// predate logical reflection ignore the record while retaining the v1 physical
// dispatch contract.
//
// The last ']' terminates the name; names may themselves contain brackets.
// Record order is independent of entry order. Defaults are root=0,32,
// constant_offset=32, constant_length=0 and an unspecified binding count.
// Missing metadata leaves that entry's is_present false. Unknown versions and
// other tools' strings are ignored. Recognized malformed records, unknown entry
// names, and duplicate fields fail load. Unscoped v1 records are invalid.
// A subgroup size is a nonzero power of two. Its absence permits the device's
// default width before SPIR-V 1.6 and varying widths from SPIR-V 1.6 onward;
// its presence requires that exact execution width. Nondefault widths and all
// fixed widths in SPIR-V 1.6 require a static LocalSize and enabled compute
// subgroup-size control. Fixed-width entries whose LocalSize X is divisible
// by that width execute with full subgroups. This establishes a dense partition
// of the workgroup, without specifying the mapping of local invocation IDs to
// subgroup IDs or lanes.
//
// The caller owns the output array. On success, release each element with
// iree_hal_vulkan_spirv_bda_dispatch_metadata_deinitialize. On failure all
// elements are reset and no allocations remain owned by the array.
iree_status_t iree_hal_vulkan_spirv_parse_bda_dispatch_metadata(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t entry_point_count,
    const iree_hal_vulkan_spirv_compute_entry_point_t* entry_points,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_spirv_bda_dispatch_metadata_t* out_metadata);

// Releases storage owned by |metadata|.
void iree_hal_vulkan_spirv_bda_dispatch_metadata_deinitialize(
    iree_hal_vulkan_spirv_bda_dispatch_metadata_t* metadata,
    iree_allocator_t host_allocator);

// Parses compute entry points and their static local workgroup sizes.
//
// Entry point names are borrowed from |spirv_words| and remain valid only while
// the module words remain live. Duplicate compute entry point names fail load.
iree_status_t iree_hal_vulkan_spirv_parse_compute_entry_points(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_host_size_t entry_point_capacity,
    iree_hal_vulkan_spirv_compute_entry_point_t* out_entry_points);

// Parses the static local workgroup size for |entry_point|.
//
// |out_entry_point_found| may be NULL. If the entry point exists but has no
// LocalSize execution mode, |out_workgroup_size| is left zeroed.
iree_status_t iree_hal_vulkan_spirv_parse_compute_workgroup_size(
    const uint32_t* spirv_words, iree_host_size_t spirv_word_count,
    iree_string_view_t entry_point, bool* out_entry_point_found,
    uint32_t out_workgroup_size[3]);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_SPIRV_H_
