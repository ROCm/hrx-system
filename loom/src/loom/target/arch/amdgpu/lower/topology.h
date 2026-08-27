// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU launch topology and source-allocation layout lowering helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_TOPOLOGY_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_TOPOLOGY_H_

#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_low_legality_context_t
    loom_target_low_legality_context_t;
typedef struct loom_amdgpu_source_alloca_layout_t
    loom_amdgpu_source_alloca_layout_t;

// Returns an initialized analysis with no source allocation entries. This is
// used by cold helper paths that intentionally lack a source-to-low or
// low-legality context; it is still a real analysis object, not a nullable
// cache sentinel.
const loom_amdgpu_source_alloca_layout_t*
loom_amdgpu_source_alloca_layout_empty(void);

// Returns the function-local source allocation layout analysis for |context|.
// The returned object is allocated from the lowering arena and remains valid
// until the current source function lowering finishes.
iree_status_t loom_amdgpu_source_alloca_layout_for_lower_context(
    loom_low_lower_context_t* context,
    const loom_amdgpu_source_alloca_layout_t** out_layout);

// Records one selected source buffer allocation in the lowering analysis.
// Source-to-low planning calls this while visiting buffer.alloca ops, so later
// selectors can resolve allocation roots without scanning source IR.
iree_status_t loom_amdgpu_source_alloca_layout_record_lower_alloca(
    loom_low_lower_context_t* context, const loom_op_t* alloca_op,
    uint64_t byte_length);

// Emits one physical low-storage root for each planned source-allocation slot.
// Mutually exclusive logical allocations share a slot and therefore share its
// root. Source allocation plans are complete before entry setup, so each root
// carries the final maximum capacity and alignment of its occupants.
iree_status_t loom_amdgpu_source_alloca_layout_emit_low_storage_roots(
    loom_low_lower_context_t* context);

// Returns the function-local source allocation layout analysis for low-legality
// verification. The returned object is allocated from the legality context's
// scratch arena and populated by the existing target-low verification walk.
iree_status_t loom_amdgpu_source_alloca_layout_for_low_legality(
    loom_target_low_legality_context_t* context,
    const loom_amdgpu_source_alloca_layout_t** out_layout);

// Records one source buffer allocation in the low-legality analysis. Target-low
// verification calls this while visiting buffer.alloca ops, so later provider
// checks can resolve allocation roots without scanning source IR.
iree_status_t loom_amdgpu_source_alloca_layout_record_low_legality_alloca(
    loom_target_low_legality_context_t* context, const loom_op_t* alloca_op,
    uint64_t byte_length);

// Resolves the analyzed storage base for a source buffer.alloca root in the
// requested memory space. Returns false when the analysis cannot prove the root
// has a statically encodable storage base in that memory space.
bool loom_amdgpu_source_alloca_layout_lookup_root(
    const loom_amdgpu_source_alloca_layout_t* layout,
    loom_value_fact_memory_space_t memory_space, loom_value_id_t root_value_id,
    uint64_t* out_byte_offset);

// Resolves the emitted physical low-storage root for a planned source
// allocation. Entry setup must have emitted the roots before this is called
// during body lowering. Packet planning separately consumes the allocation's
// analyzed suballocation offset through lookup_root.
void loom_amdgpu_source_alloca_layout_lookup_low_storage(
    const loom_amdgpu_source_alloca_layout_t* layout,
    loom_value_fact_memory_space_t memory_space, loom_value_id_t root_value_id,
    loom_value_id_t* out_storage_value_id);

// Returns the exact wavefront size selected by the active target bundle.
uint32_t loom_amdgpu_target_wavefront_size(const loom_target_bundle_t* bundle);

// Returns the native execution partition width available to subgroup
// communication for |source_wavefront_size| on |target_facts|.
//
// Source target records may request a kernel wavefront size that is wider than
// the processor's default/native execution partition. Workgroup collectives can
// stitch native partitions together through LDS, but direct subgroup operations
// must not claim semantic communication wider than this value.
uint32_t loom_amdgpu_target_native_subgroup_width(
    const loom_amdgpu_target_facts_t* target_facts,
    uint32_t source_wavefront_size);

// Returns whether a direct subgroup operation with |required_width| lanes can
// be represented by native subgroup communication for the selected target.
bool loom_amdgpu_target_supports_direct_subgroup_width(
    const loom_amdgpu_target_facts_t* target_facts,
    uint32_t source_wavefront_size, uint32_t required_width);

// Selects the active target wavefront size when it is valid for native
// subgroup lowering.
bool loom_amdgpu_select_subgroup_wavefront_size(
    loom_low_lower_context_t* context, uint32_t* out_wavefront_size);

// Selects whether a direct subgroup operation with |required_width| semantic
// lanes can be represented by native subgroup communication for the selected
// target.
bool loom_amdgpu_select_direct_subgroup_width(loom_low_lower_context_t* context,
                                              uint32_t source_wavefront_size,
                                              uint32_t required_width);

// Selects the active target wavefront size when full-wave native subgroup
// communication is available.
bool loom_amdgpu_select_full_wave_direct_subgroup_width(
    loom_low_lower_context_t* context, uint32_t* out_wavefront_size);

// Returns the fixed per-dimension workgroup size required by the source
// function or target ABI.
bool loom_amdgpu_required_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, loom_target_workgroup_size_t* out_size);

// Returns the fixed per-dimension workgroup size required by the source
// function or target ABI, using |fact_table| to prove launch-config values
// that are not literal constants in the current IR.
bool loom_amdgpu_required_workgroup_size_from_facts(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle,
    const loom_value_fact_table_t* fact_table,
    loom_target_workgroup_size_t* out_size);

// Returns the fixed flat workgroup size required by the source function or
// target ABI.
bool loom_amdgpu_required_flat_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t* out_flat_size);

// Returns the fixed flat workgroup size required by the source function or
// target ABI, using |fact_table| to prove launch-config values that are not
// literal constants in the current IR.
bool loom_amdgpu_required_flat_workgroup_size_from_facts(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle,
    const loom_value_fact_table_t* fact_table, uint32_t* out_flat_size);

// Emits the current invocation lane id within its subgroup as a VGPR value.
iree_status_t loom_amdgpu_emit_current_subgroup_lane_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_lane_id);

// Emits the flattened local workitem id for the active kernel launch.
iree_status_t loom_amdgpu_emit_current_workitem_linear_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_linear_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_TOPOLOGY_H_
