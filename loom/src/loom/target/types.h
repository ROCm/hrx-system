// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared target payload types.
//
// This header intentionally contains only target-neutral data shapes. It must
// not include target-family descriptors, generated target records, feature
// catalogs, scheduling tables, or emitter-specific helpers. Those live under
// opt-in target packages so AOT and JIT builds can exclude every target family
// they do not use.

#ifndef LOOM_TARGET_TYPES_H_
#define LOOM_TARGET_TYPES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_target_codegen_format_t;
typedef enum loom_target_codegen_format_e {
  LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN = 0,
  LOOM_TARGET_CODEGEN_FORMAT_LLVMIR = 1,
  LOOM_TARGET_CODEGEN_FORMAT_SPIRV = 2,
  LOOM_TARGET_CODEGEN_FORMAT_VM = 3,
  LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE = 4,
  LOOM_TARGET_CODEGEN_FORMAT_WASM = 5,
} loom_target_codegen_format_e;

typedef uint8_t loom_target_artifact_format_t;
typedef enum loom_target_artifact_format_e {
  LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN = 0,
  LOOM_TARGET_ARTIFACT_FORMAT_ELF = 1,
  LOOM_TARGET_ARTIFACT_FORMAT_COFF = 2,
  LOOM_TARGET_ARTIFACT_FORMAT_MACHO = 3,
  LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY = 4,
  LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE = 5,
  LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY = 6,
  LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_TEXT = 7,
  LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_BITCODE = 8,
} loom_target_artifact_format_e;

typedef uint8_t loom_target_abi_kind_t;
typedef enum loom_target_abi_kind_e {
  LOOM_TARGET_ABI_UNKNOWN = 0,
  LOOM_TARGET_ABI_OBJECT_FUNCTION = 1,
  LOOM_TARGET_ABI_HAL_KERNEL = 2,
  LOOM_TARGET_ABI_VM_MODULE_FUNCTION = 3,
  LOOM_TARGET_ABI_SHADER_ENTRY_POINT = 4,
  LOOM_TARGET_ABI_WASM_FUNCTION = 5,
} loom_target_abi_kind_e;

typedef uint8_t loom_target_linkage_t;
typedef enum loom_target_linkage_e {
  LOOM_TARGET_LINKAGE_DEFAULT = 0,
  LOOM_TARGET_LINKAGE_DSO_LOCAL = 1,
} loom_target_linkage_e;

// Returns the assembly spelling for |format| used in diagnostics.
static inline iree_string_view_t loom_target_codegen_format_name(
    loom_target_codegen_format_t format) {
  switch (format) {
    case LOOM_TARGET_CODEGEN_FORMAT_LLVMIR:
      return IREE_SV("llvmir");
    case LOOM_TARGET_CODEGEN_FORMAT_SPIRV:
      return IREE_SV("spirv");
    case LOOM_TARGET_CODEGEN_FORMAT_VM:
      return IREE_SV("vm");
    case LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE:
      return IREE_SV("low_native");
    case LOOM_TARGET_CODEGEN_FORMAT_WASM:
      return IREE_SV("wasm");
    case LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

// Returns the assembly spelling for |format| used in diagnostics.
static inline iree_string_view_t loom_target_artifact_format_name(
    loom_target_artifact_format_t format) {
  switch (format) {
    case LOOM_TARGET_ARTIFACT_FORMAT_ELF:
      return IREE_SV("elf");
    case LOOM_TARGET_ARTIFACT_FORMAT_COFF:
      return IREE_SV("coff");
    case LOOM_TARGET_ARTIFACT_FORMAT_MACHO:
      return IREE_SV("macho");
    case LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY:
      return IREE_SV("spirv_binary");
    case LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE:
      return IREE_SV("vm_bytecode");
    case LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY:
      return IREE_SV("wasm_binary");
    case LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_TEXT:
      return IREE_SV("llvmir_text");
    case LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_BITCODE:
      return IREE_SV("llvmir_bitcode");
    case LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

// Returns the assembly spelling for |abi_kind| used in diagnostics.
static inline iree_string_view_t loom_target_abi_kind_name(
    loom_target_abi_kind_t abi_kind) {
  switch (abi_kind) {
    case LOOM_TARGET_ABI_OBJECT_FUNCTION:
      return IREE_SV("object_function");
    case LOOM_TARGET_ABI_HAL_KERNEL:
      return IREE_SV("hal_kernel");
    case LOOM_TARGET_ABI_VM_MODULE_FUNCTION:
      return IREE_SV("vm_module_function");
    case LOOM_TARGET_ABI_SHADER_ENTRY_POINT:
      return IREE_SV("shader_entry_point");
    case LOOM_TARGET_ABI_WASM_FUNCTION:
      return IREE_SV("wasm_function");
    case LOOM_TARGET_ABI_UNKNOWN:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

// Returns the assembly spelling for |linkage| used in diagnostics.
static inline iree_string_view_t loom_target_linkage_name(
    loom_target_linkage_t linkage) {
  switch (linkage) {
    case LOOM_TARGET_LINKAGE_DSO_LOCAL:
      return IREE_SV("dso_local");
    case LOOM_TARGET_LINKAGE_DEFAULT:
      return IREE_SV("default");
  }
  return IREE_SV("unknown");
}

typedef struct loom_target_memory_space_map_t {
  // Generic/default pointer address space.
  uint32_t generic;
  // Device or process global memory address space.
  uint32_t global;
  // Workgroup/shared memory address space.
  uint32_t workgroup;
  // Constant memory address space.
  uint32_t constant;
  // Per-invocation/private memory address space.
  uint32_t private_memory;
  // Host-visible memory address space, or UINT32_MAX when unavailable.
  uint32_t host;
  // Descriptor/resource identity address space, or UINT32_MAX when unavailable.
  uint32_t descriptor;
} loom_target_memory_space_map_t;

typedef struct loom_target_workgroup_size_t {
  // Workgroup size along the x dimension.
  uint32_t x;
  // Workgroup size along the y dimension.
  uint32_t y;
  // Workgroup size along the z dimension.
  uint32_t z;
} loom_target_workgroup_size_t;

typedef struct loom_target_grid_size_t {
  // Maximum dispatched grid size along the x dimension.
  uint32_t x;
  // Maximum dispatched grid size along the y dimension.
  uint32_t y;
  // Maximum dispatched grid size along the z dimension.
  uint32_t z;
} loom_target_grid_size_t;

typedef struct loom_target_workgroup_count_limit_t {
  // Maximum dispatch workgroup count along the x dimension.
  uint32_t x;
  // Maximum dispatch workgroup count along the y dimension.
  uint32_t y;
  // Maximum dispatch workgroup count along the z dimension.
  uint32_t z;
} loom_target_workgroup_count_limit_t;

typedef struct loom_target_dispatch_workgroup_count_t {
  // Dispatch workgroup count along the x dimension.
  uint32_t x;
  // Dispatch workgroup count along the y dimension.
  uint32_t y;
  // Dispatch workgroup count along the z dimension.
  uint32_t z;
} loom_target_dispatch_workgroup_count_t;

typedef struct loom_target_snapshot_t {
  // Stable snapshot name for diagnostics and tests.
  iree_string_view_t name;
  // Primary emitter family for this snapshot.
  loom_target_codegen_format_t codegen_format;
  // Linkable or loadable artifact format produced for this snapshot.
  loom_target_artifact_format_t artifact_format;
  // Default pointer bit width for target-independent layout decisions.
  uint32_t default_pointer_bitwidth;
  // Index bit width chosen for lowered index values.
  uint32_t index_bitwidth;
  // Offset bit width chosen for lowered byte offsets.
  uint32_t offset_bitwidth;
  // Maximum API/hardware local workgroup size per dimension. Zero dimensions
  // mean the target has not supplied a tighter limit.
  loom_target_workgroup_size_t max_workgroup_size;
  // Maximum flat local workgroup size. Zero means the target has not supplied a
  // tighter total-workitem limit across the workgroup dimensions.
  uint32_t max_flat_workgroup_size;
  // Maximum bytes a single function may reserve in workgroup storage. Zero
  // means the target has not supplied a tighter function-local storage limit.
  uint64_t max_workgroup_storage_bytes;
  // Fixed subgroup size in invocations. Zero means the target has not supplied
  // a target-wide fixed subgroup size.
  uint32_t subgroup_size;
  // Maximum API/hardware dispatched grid size per dimension, in workitems.
  // Zero dimensions mean the target has not supplied a tighter limit.
  loom_target_grid_size_t max_grid_size;
  // Maximum flat dispatched grid size, in workitems. Zero means the target has
  // not supplied a tighter total-workitem limit across the grid dimensions.
  uint64_t max_flat_grid_size;
  // Maximum API/hardware dispatch workgroup count per dimension. Zero
  // dimensions mean the target has not supplied a tighter limit.
  loom_target_workgroup_count_limit_t max_workgroup_count;
  // Address space assignments used by target-specific ABI lowering.
  loom_target_memory_space_map_t memory_spaces;
} loom_target_snapshot_t;

typedef struct loom_target_hal_kernel_abi_t {
  // Function-selected fixed workgroup size for a kernel entry point. A zero
  // dimension means the function has not selected a fixed size yet.
  loom_target_workgroup_size_t required_workgroup_size;
  // Optimization lower flat workgroup size advertised to the backend.
  uint32_t flat_workgroup_size_min;
  // Optimization upper flat workgroup size advertised to the backend.
  uint32_t flat_workgroup_size_max;
  // ABI-required raw buffer resource flags for global binding resources.
  uint32_t buffer_resource_flags;
} loom_target_hal_kernel_abi_t;

typedef struct loom_target_export_plan_t {
  // Stable export plan name for diagnostics and tests.
  iree_string_view_t name;
  // Output symbol exposed by this export, or empty to preserve the source name.
  iree_string_view_t export_symbol;
  // Callable/package ABI used for this export.
  loom_target_abi_kind_t abi_kind;
  // ABI-required linkage for exported object functions or entry points.
  loom_target_linkage_t linkage;
  // HAL kernel ABI facts when abi_kind is LOOM_TARGET_ABI_HAL_KERNEL.
  loom_target_hal_kernel_abi_t hal_kernel;
} loom_target_export_plan_t;

typedef struct loom_target_config_t {
  // Stable config name for diagnostics and tests.
  iree_string_view_t name;
  // Provider-defined target-contract selection key, or empty for default
  // policy.
  iree_string_view_t contract_set_key;
  // Provider-defined target-contract feature bitset, or zero for default
  // policy.
  uint64_t contract_feature_bits;
} loom_target_config_t;

typedef struct loom_target_bundle_t {
  // Stable bundle name for diagnostics and tests.
  iree_string_view_t name;
  // Frozen codegen target snapshot selected for this bundle.
  const loom_target_snapshot_t* snapshot;
  // Export and ABI plan selected for this bundle.
  const loom_target_export_plan_t* export_plan;
  // Legalization and target-contract config selected for this bundle.
  const loom_target_config_t* config;
} loom_target_bundle_t;

typedef struct loom_target_selection_t {
  // Runtime-selected effective target bundle, or NULL when source IR target
  // records alone select the target contract.
  const loom_target_bundle_t* bundle;
  // Target-owned immutable payload associated with |bundle| or with the source
  // selected target bundle when |bundle| is NULL. Core compiler code passes
  // this through and never interprets it.
  const void* data;
} loom_target_selection_t;

// Returns an empty selected target overlay.
static inline loom_target_selection_t loom_target_selection_empty(void) {
  return (loom_target_selection_t){0};
}

// Returns true when |selection| has no selected target overlay.
static inline bool loom_target_selection_is_empty(
    loom_target_selection_t selection) {
  return selection.bundle == NULL && selection.data == NULL;
}

typedef struct loom_target_bundle_table_t {
  // Selector-indexed target row pointers. Entry zero is normally NULL because
  // generated enums reserve zero for absent/unknown cases.
  const loom_target_bundle_t* const* values;
  // Number of entries in |values|.
  uint8_t count;
} loom_target_bundle_table_t;

enum loom_target_projection_value_bits_e {
  // Enum attr projected into a uint8_t target enum field.
  LOOM_TARGET_PROJECTION_VALUE_ENUM_U8 = 1,
  // I64 attr projected into a uint32_t field after verification.
  LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32 = 2,
  // I64 attr projected into a uint64_t field after verification.
  LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64 = 3,
  // String attr projected into an iree_string_view_t field.
  LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW = 4,
};
typedef uint8_t loom_target_projection_value_kind_t;

typedef struct loom_target_projection_t {
  // Byte offset into loom_target_bundle_storage_t for the destination field.
  uint16_t storage_offset;
  // Attribute index on the target-like op.
  uint8_t attr_index;
  // Projection operation used to copy the present attr payload.
  loom_target_projection_value_kind_t value_kind;
} loom_target_projection_t;

static_assert(sizeof(loom_target_projection_t) == 4,
              "loom_target_projection_t must be exactly 4 bytes");

typedef struct loom_target_like_descriptor_t {
  // Direct selector-indexed bundle table for a target-like op family.
  const loom_target_bundle_table_t* bundle_table;
  // Optional projection rows for typed attrs that override the selected bundle.
  const loom_target_projection_t* projections;
  // Number of entries in |projections|.
  uint8_t projection_count;
} loom_target_like_descriptor_t;

typedef struct loom_target_bundle_storage_t {
  // Materialized target snapshot owned by this storage object.
  loom_target_snapshot_t snapshot;
  // Materialized function ABI/export contract owned by this storage object.
  loom_target_export_plan_t export_plan;
  // Materialized target configuration owned by this storage object.
  loom_target_config_t config;
  // Bundle view pointing at snapshot, export_plan, and config above.
  loom_target_bundle_t bundle;
} loom_target_bundle_storage_t;

// Rebinds |storage->bundle| to the payloads embedded in |storage|. Call this
// after copying a storage object by value; the embedded bundle is a view and
// its pointers must refer to the same enclosing storage object.
static inline void loom_target_bundle_storage_rebind(
    loom_target_bundle_storage_t* storage) {
  storage->bundle.snapshot = &storage->snapshot;
  storage->bundle.export_plan = &storage->export_plan;
  storage->bundle.config = &storage->config;
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_TYPES_H_
