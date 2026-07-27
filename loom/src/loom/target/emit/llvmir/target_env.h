// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVM target environment descriptions.
//
// Target environments collect target/ABI state chosen by a target provider, not
// by the generic LLVM IR module builder. This header owns the shared data model
// and target-neutral helpers; concrete target profile presets live in opt-in
// provider packages.

#ifndef LOOM_TARGET_LLVMIR_TARGET_ENV_H_
#define LOOM_TARGET_LLVMIR_TARGET_ENV_H_

#include "loom/target/emit/llvmir/module.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_llvmir_object_format_e {
  LOOM_LLVMIR_OBJECT_FORMAT_UNKNOWN = 0,
  LOOM_LLVMIR_OBJECT_FORMAT_ELF = 1,
  LOOM_LLVMIR_OBJECT_FORMAT_COFF = 2,
  LOOM_LLVMIR_OBJECT_FORMAT_MACHO = 3,
} loom_llvmir_object_format_t;

typedef enum loom_llvmir_target_profile_kind_e {
  LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT = 0,
  LOOM_LLVMIR_TARGET_PROFILE_KERNEL = 1,
} loom_llvmir_target_profile_kind_t;

typedef enum loom_llvmir_kernel_profile_flag_bits_e {
  LOOM_LLVMIR_KERNEL_PROFILE_FLAG_ALWAYSINLINE = 1u << 0,
} loom_llvmir_kernel_profile_flag_bits_t;
typedef uint32_t loom_llvmir_kernel_profile_flags_t;

enum {
  LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT = 2,
  LOOM_LLVMIR_TARGET_PROFILE_MAX_LLC_ARGUMENT_COUNT = 3,
  LOOM_LLVMIR_TARGET_PROFILE_LLC_ARGUMENT_STORAGE_LENGTH = 512,
};

typedef struct loom_llvmir_target_address_spaces_t {
  // Generic/default pointer address space.
  uint32_t generic;
  // Device or process global memory address space.
  uint32_t global;
  // Workgroup/shared memory address space.
  uint32_t local;
  // Constant memory address space.
  uint32_t constant;
  // Per-invocation/private memory address space.
  uint32_t private_memory;
  // Target-specific buffer-resource pointer address space, or UINT32_MAX.
  uint32_t buffer_resource;
} loom_llvmir_target_address_spaces_t;

typedef struct loom_llvmir_workgroup_size_t {
  // Workgroup size along the x dimension.
  uint32_t x;
  // Workgroup size along the y dimension.
  uint32_t y;
  // Workgroup size along the z dimension.
  uint32_t z;
} loom_llvmir_workgroup_size_t;

typedef struct loom_llvmir_kernel_coordinate_intrinsics_t {
  // LLVM intrinsic returning the x workitem coordinate, or empty if
  // unsupported.
  iree_string_view_t workitem_id_x;
  // LLVM intrinsic returning the y workitem coordinate, or empty if
  // unsupported.
  iree_string_view_t workitem_id_y;
  // LLVM intrinsic returning the z workitem coordinate, or empty if
  // unsupported.
  iree_string_view_t workitem_id_z;
  // LLVM intrinsic returning the x workgroup coordinate, or empty if
  // unsupported.
  iree_string_view_t workgroup_id_x;
  // LLVM intrinsic returning the y workgroup coordinate, or empty if
  // unsupported.
  iree_string_view_t workgroup_id_y;
  // LLVM intrinsic returning the z workgroup coordinate, or empty if
  // unsupported.
  iree_string_view_t workgroup_id_z;
} loom_llvmir_kernel_coordinate_intrinsics_t;

typedef struct loom_llvmir_target_env_t {
  // Stable target environment name for diagnostics and tests.
  iree_string_view_t name;
  // LLVM target triple.
  iree_string_view_t target_triple;
  // LLVM data layout, or empty when supplied by a downstream toolchain.
  iree_string_view_t data_layout;
  // Linkable object format produced by this target.
  loom_llvmir_object_format_t object_format;
  // Default pointer bit width for target-independent layout decisions.
  uint32_t default_pointer_bitwidth;
  // Index bit width chosen for lowered index values.
  uint32_t index_bitwidth;
  // Offset bit width chosen for lowered byte offsets.
  uint32_t offset_bitwidth;
  // Address space assignments used by target-specific ABI lowering.
  loom_llvmir_target_address_spaces_t address_spaces;
} loom_llvmir_target_env_t;

typedef struct loom_llvmir_kernel_profile_t {
  // LLVM calling convention required for the kernel entry point.
  loom_llvmir_calling_convention_t calling_convention;
  // Metadata attachment name for selected fixed workgroup size.
  iree_string_view_t required_workgroup_size_metadata_name;
  // ABI-selected fixed workgroup size attached to each kernel entry point, or
  // zero when launch selection remains dynamic.
  loom_llvmir_workgroup_size_t required_workgroup_size;
  // Lower flat workgroup size bound advertised to LLVM.
  uint32_t flat_workgroup_size_min;
  // Upper flat workgroup size bound advertised to LLVM.
  uint32_t flat_workgroup_size_max;
  // Fixed subgroup size in invocations, or zero when the target profile does
  // not select a fixed subgroup size.
  uint32_t subgroup_size;
  // Target-specific flag word for descriptor-backed kernel binding resources.
  uint32_t binding_resource_flags;
  // Optional LLVM function-attribute key for the flat workgroup size range.
  iree_string_view_t flat_workgroup_size_attr_name;
  // Optional LLVM function-attribute key declaring uniform workgroup size.
  iree_string_view_t uniform_workgroup_size_attr_name;
  // Kernel entry attribute flags.
  loom_llvmir_kernel_profile_flags_t flags;
  // Target-specific LLVM intrinsics for kernel coordinate materialization.
  loom_llvmir_kernel_coordinate_intrinsics_t coordinate_intrinsics;
  // ABI-required attrs for descriptor-backed binding pointer parameters.
  loom_llvmir_attr_t binding_parameter_attrs
      [LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT];
  // Number of valid entries in |binding_parameter_attrs|.
  iree_host_size_t binding_parameter_attr_count;
} loom_llvmir_kernel_profile_t;

typedef struct loom_llvmir_target_profile_t {
  // Stable target profile name for diagnostics and tests.
  iree_string_view_t name;
  // Target environment selected by this ABI profile.
  const loom_llvmir_target_env_t* target_env;
  // ABI family represented by this profile.
  loom_llvmir_target_profile_kind_t kind;
  // Optimization default target CPU, or empty when supplied per variant.
  iree_string_view_t target_cpu;
  // Optimization default target feature string, or empty when supplied per
  // variant.
  iree_string_view_t target_features;
  // x86 packed-dot feature bits interpreted by target-specific vector lowering.
  uint64_t x86_packed_dot_feature_bits;
  // ABI-required linkage for exported object functions or kernel entry points.
  loom_llvmir_linkage_t exported_linkage;
  // Kernel entry-point projection parameters.
  loom_llvmir_kernel_profile_t kernel;
} loom_llvmir_target_profile_t;

typedef struct loom_llvmir_target_profile_storage_t {
  // Derived LLVM target environment storage referenced by |profile|.
  loom_llvmir_target_env_t target_env;
  // Derived LLVM target profile adapter referencing |target_env|.
  loom_llvmir_target_profile_t profile;
} loom_llvmir_target_profile_storage_t;

typedef struct loom_llvmir_target_profile_llc_arguments_t {
  // Backing storage for argv-style llc argument strings.
  char storage[LOOM_LLVMIR_TARGET_PROFILE_MAX_LLC_ARGUMENT_COUNT]
              [LOOM_LLVMIR_TARGET_PROFILE_LLC_ARGUMENT_STORAGE_LENGTH];
  // Argument views pointing into storage.
  iree_string_view_t values[LOOM_LLVMIR_TARGET_PROFILE_MAX_LLC_ARGUMENT_COUNT];
  // Number of valid entries in values.
  iree_host_size_t count;
} loom_llvmir_target_profile_llc_arguments_t;

void loom_llvmir_target_env_module_config(
    const loom_llvmir_target_env_t* target_env, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config);

void loom_llvmir_target_profile_module_config(
    const loom_llvmir_target_profile_t* profile, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config);

// Derives an LLVMIR adapter profile from generic target records and an
// LLVM-owned projection row. The returned |out_storage->profile.target_env|
// points at |out_storage->target_env|.
void loom_llvmir_target_profile_storage_initialize_from_bundle(
    const loom_target_bundle_t* bundle,
    const loom_llvmir_target_profile_t* projected_profile,
    loom_llvmir_target_profile_storage_t* out_storage);

// Builds argv-style llc target arguments for |profile|. The returned views
// point into |out_arguments| and remain valid until it is overwritten.
iree_status_t loom_llvmir_target_profile_llc_arguments(
    const loom_llvmir_target_profile_t* profile,
    loom_llvmir_target_profile_llc_arguments_t* out_arguments);

// Writes ABI-required calling-convention attrs for a HAL kernel binding
// pointer. Memory facts such as noalias, nonnull, and pointee alignment must be
// proven on the individual binding value before they are attached.
// The caller provides temporary storage; loom_llvmir_function_add_parameter()
// copies the attrs into the target module when attaching them to a parameter.
void loom_llvmir_target_profile_kernel_binding_attrs(
    const loom_llvmir_target_profile_t* profile, loom_llvmir_attr_t* attrs,
    iree_host_size_t* out_attr_count);

// Adds the ABI and optimization attr group for a HAL kernel entry point.
// Attribute values are derived from |profile| at materialization time so
// profile copies can safely customize workgroup-size policy.
iree_status_t loom_llvmir_target_profile_add_kernel_attr_group(
    loom_llvmir_module_t* module, const loom_llvmir_target_profile_t* profile,
    loom_llvmir_attr_group_id_t* out_group_id);

// Attaches fixed-workgroup-size metadata when the profile selected one.
iree_status_t loom_llvmir_target_profile_attach_kernel_metadata(
    loom_llvmir_function_t* function,
    const loom_llvmir_target_profile_t* profile);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_TARGET_ENV_H_
