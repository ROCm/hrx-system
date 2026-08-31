// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Detached native-object contribution emission for one AIE2P Low leaf.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_OBJECT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_OBJECT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"
#include "loom/target/arch/amd/xdna/elf_format.h"
#include "loom/target/emit/native/object.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable semantic target identity carried by every AIE2P core contribution.
#define LOOM_AIE2P_LEAF_TARGET_IDENTITY IREE_SV("amd.xdna.aie2p.core")

// Stable object-function ABI identity carried by every AIE2P core
// contribution.
#define LOOM_AIE2P_LEAF_ABI_IDENTITY IREE_SV("aie2p-object-function")

enum loom_aie2p_leaf_capability_flag_bits_e {
  // The leaf imports values that the array linker must bind to physical
  // registers before the core starts.
  LOOM_AIE2P_LEAF_CAPABILITY_FLAG_RESOURCE_IMPORTS = 1u << 0,
  // The leaf contains native fixups that the array linker must resolve.
  LOOM_AIE2P_LEAF_CAPABILITY_FLAG_NATIVE_FIXUPS = 1u << 1,
  // The leaf requires initialized local data bytes.
  LOOM_AIE2P_LEAF_CAPABILITY_FLAG_INITIALIZED_DATA = 1u << 2,
  // The leaf requires zero-filled local data bytes.
  LOOM_AIE2P_LEAF_CAPABILITY_FLAG_ZERO_FILL = 1u << 3,
  // The leaf requires function-local storage.
  LOOM_AIE2P_LEAF_CAPABILITY_FLAG_FUNCTION_STORAGE = 1u << 4,
  // The leaf contains materialized spill storage and traffic.
  LOOM_AIE2P_LEAF_CAPABILITY_FLAG_MATERIALIZED_SPILLS = 1u << 5,
};
typedef uint32_t loom_aie2p_leaf_capability_flags_t;

// Exact storage required in one placement domain.
typedef struct loom_aie2p_leaf_storage_requirement_t {
  // Required byte length, excluding placement padding outside this domain.
  uint64_t byte_length;
  // Minimum placement alignment, or zero when no storage is required.
  uint64_t minimum_alignment;
} loom_aie2p_leaf_storage_requirement_t;

enum loom_aie2p_leaf_resource_flag_bits_e {
  // Resource extent is supplied through the Low extent operand.
  LOOM_AIE2P_LEAF_RESOURCE_FLAG_DYNAMIC_EXTENT = 1u << 0,
  // Resource carries a static byte extent.
  LOOM_AIE2P_LEAF_RESOURCE_FLAG_STATIC_EXTENT = 1u << 1,
  // Resource carries a cache-swizzle byte stride.
  LOOM_AIE2P_LEAF_RESOURCE_FLAG_CACHE_SWIZZLE_STRIDE = 1u << 2,
};
typedef uint16_t loom_aie2p_leaf_resource_flags_t;

// One detached Low resource import and its final physical-register binding.
typedef struct loom_aie2p_leaf_resource_import_t {
  // Resource table index selected by low.resource.
  uint64_t index;
  // Static resource extent when STATIC_EXTENT is set, otherwise zero.
  uint64_t extent;
  // Cache-swizzle byte stride when CACHE_SWIZZLE_STRIDE is set, otherwise
  // zero.
  uint32_t cache_swizzle_stride;
  // First AIE2P physical-register ID occupied by the imported value.
  uint32_t physical_register;
  // Number of logical physical-register units occupied by the imported value.
  uint32_t physical_register_count;
  // Physical register carrying a dynamic extent, or UINT32_MAX when absent.
  uint32_t extent_physical_register;
  // AIE2P descriptor-set register class owning the physical register.
  uint16_t descriptor_register_class_id;
  // AIE2P register class carrying a dynamic extent, or zero when absent.
  uint16_t extent_descriptor_register_class_id;
  // Number of physical-register units carrying a dynamic extent.
  uint32_t extent_physical_register_count;
  // Optional resource metadata carried by this record.
  loom_aie2p_leaf_resource_flags_t flags;
  // Low ABI import kind.
  loom_low_resource_import_kind_t import_kind;
  // Outer Loom type kind of the imported source value.
  loom_type_kind_t source_type_kind;
} loom_aie2p_leaf_resource_import_t;

// Exact physical facts retained after all expensive leaf compilation work.
//
// The record owns no compiler IR. Array planning and final linking may retain,
// serialize, and cache it together with the native object contribution.
typedef struct loom_aie2p_leaf_realization_t {
  // Stable target contract identity.
  iree_string_view_t target_identity;
  // Stable object-function ABI identity.
  iree_string_view_t abi_identity;
  // Index of the entry definition in object.symbols.
  uint32_t entry_symbol_index;
  // ELF machine identity required by the contribution.
  uint16_t elf_machine;
  // XDNA target generation required by the contribution.
  loom_xdna_target_generation_t target_generation;
  // Processor-specific ELF flags required by the contribution.
  uint32_t elf_flags;
  // Physical features the array linker must realize.
  loom_aie2p_leaf_capability_flags_t capability_flags;
  // Core program-memory footprint.
  loom_aie2p_leaf_storage_requirement_t code;
  // Read-only local-data footprint.
  loom_aie2p_leaf_storage_requirement_t read_only_data;
  // Initialized writable local-data footprint.
  loom_aie2p_leaf_storage_requirement_t initialized_data;
  // Zero-filled writable local-data footprint.
  loom_aie2p_leaf_storage_requirement_t zero_fill;
  // Function stack-storage footprint.
  loom_aie2p_leaf_storage_requirement_t stack;
  // Function scratch-storage footprint.
  loom_aie2p_leaf_storage_requirement_t scratch;
  // Function private-storage footprint.
  loom_aie2p_leaf_storage_requirement_t private_storage;
  // Function workgroup-storage footprint.
  loom_aie2p_leaf_storage_requirement_t workgroup_storage;
  // Materialized spill payload footprint, already included in the applicable
  // function-storage domain above.
  loom_aie2p_leaf_storage_requirement_t spill;
  // Detached resource imports in Low entry-block order.
  const loom_aie2p_leaf_resource_import_t* resource_imports;
  // Number of records in resource_imports.
  iree_host_size_t resource_import_count;
} loom_aie2p_leaf_realization_t;

// Complete detached product of one AIE2P core compilation.
typedef struct loom_aie2p_leaf_contribution_t {
  // Generic native sections, symbols, and fixups.
  loom_native_object_contribution_t object;
  // Exact AIE2P planning and linking facts.
  loom_aie2p_leaf_realization_t realization;
} loom_aie2p_leaf_contribution_t;

// Emits one arena-owned native leaf contribution from |plan|.
//
// Section bytes and names are copied into |arena| and no longer borrow the
// source Loom module. Symbols, fixups, resource bindings, and realization
// facts remain contribution-relative so a later array-image linker can gather
// independently compiled leaves without reopening worker object files or
// retaining compiler IR.
iree_status_t loom_aie2p_leaf_object_emit(
    const loom_aie2p_bundle_plan_t* plan, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_contribution_t* out_contribution);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_OBJECT_H_
