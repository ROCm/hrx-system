// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native AMDGPU memory packet legality, addressing, and selection.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_PACKET_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_PACKET_H_

#include "loom/target/arch/amdgpu/lower/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_memory_packet_selection_context_t {
  // Source module being lowered.
  const loom_module_t* module;
  // Source value facts used for offset and storage decisions.
  const loom_value_fact_table_t* fact_table;
  // Active target-low descriptor set.
  const loom_low_descriptor_set_t* descriptor_set;
  // Source view-region facts for alias and read-only classification.
  const loom_view_region_table_t* view_regions;
  // Source function containing the memory access.
  loom_func_like_t source_function;
  // Target configuration bundle being specialized.
  const loom_target_bundle_t* bundle;
  // Target instruction restrictions active for packet selection.
  loom_amdgpu_instruction_constraint_bits_t instruction_constraints;
  // Cached dynamic-term materialization facts for the source access.
  const loom_amdgpu_memory_dynamic_term_materialization_plan_t*
      materialization_plan;
  // The retained storage root has a per-lane vector address.
  bool root_prefers_vgpr;
  // The source carrier decision requires every result packet in scalar storage.
  bool requires_scalar_payload;
} loom_amdgpu_memory_packet_selection_context_t;

typedef struct loom_amdgpu_descriptor_offset_immediate_info_t {
  // Low immediate kind required by all offset immediate fields.
  loom_low_immediate_kind_t kind;
  // Minimum encoded value accepted by signed offset immediate fields.
  int64_t signed_min;
  // Maximum encoded value accepted by every offset immediate field.
  uint64_t unsigned_max;
  // Byte count represented by one encoded offset unit.
  uint32_t unit_byte_count;
} loom_amdgpu_descriptor_offset_immediate_info_t;

// Records whether a static byte offset is representable by native addressing.
bool loom_amdgpu_memory_access_static_byte_offset_is_usable(
    int64_t static_byte_offset,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic);

// Returns whether the packet is a two-element word/doubleword LDS pair.
bool loom_amdgpu_memory_access_is_ds_pair(
    const loom_amdgpu_memory_access_t* access);

// Returns whether every element of this physical packet is adjacent in memory.
bool loom_amdgpu_memory_access_has_contiguous_vector_lanes(
    const loom_amdgpu_memory_access_t* access);

// Selects one native packet from canonical term materialization facts. Logical
// decomposition owns the payload span and the required result register class.
bool loom_amdgpu_memory_access_select_packet(
    const loom_amdgpu_memory_packet_selection_context_t* selection_context,
    loom_low_source_memory_operation_kind_t kind, bool allow_global_smem,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic);

// Reads common offset-immediate limits from a descriptor.
bool loom_amdgpu_descriptor_offset_immediate_info(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint16_t expected_offset_immediate_count,
    loom_low_immediate_kind_t expected_kind,
    loom_amdgpu_descriptor_offset_immediate_info_t* out_info);

// Records the first source dynamic term relevant to a flat-address rejection.
void loom_amdgpu_memory_access_record_flat_dynamic_address_rejection(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic);

// Routes all dynamic source terms through the VGPR byte-address operand.
void loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(
    loom_amdgpu_memory_access_t* access);

// Selects a full-width flat address with every dynamic term routed through
// VADDR. The byte offset initially belongs entirely to the pointer; packet
// selection may split off a descriptor immediate afterward.
bool loom_amdgpu_memory_access_select_flat_address(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic);

// Splits a flat packet's static byte offset between the selected descriptor's
// immediate and its full-width pointer. Shared by observations and updates.
bool loom_amdgpu_memory_access_select_flat_offset(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic);

// Selects an explicit VGPR byte-offset expression for source access
// instrumentation.
//
// The selected access has all static byte offset, including target-assigned
// workgroup/private alloca layout, and all dynamic byte terms routed through
// VADDR. It does not select a memory packet descriptor. Callers can pass the
// result to loom_amdgpu_emit_memory_vaddr with no base value to materialize the
// exact u32 memory-space-relative byte offset.
bool loom_amdgpu_memory_access_select_u32_vaddr_byte_offset(
    const loom_module_t* module,
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic);

// Selects a bounds-checked buffer packet and partitions its static offset.
// Dynamic contributions retain their canonical proof even when a physical
// realization supplies the descriptor base and vector offset.
bool loom_amdgpu_memory_access_try_select_buffer(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_source_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic);

// Returns the stable report/diagnostic reason key explaining why an LDS access
// did or did not select the DS addtid address form.
iree_string_view_t loom_amdgpu_memory_ds_addtid_reason_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_target_bundle_t* bundle,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_PACKET_H_
