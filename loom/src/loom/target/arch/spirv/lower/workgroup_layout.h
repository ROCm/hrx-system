// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V Workgroup scalar-carrier and physical storage layout analysis.

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_LAYOUT_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_LAYOUT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves the allocation-wide register class for a source Workgroup view.
// |out_is_workgroup| distinguishes ordinary views from Workgroup views that
// cannot be represented with one coherent scalar carrier.
iree_status_t loom_spirv_resolve_workgroup_view_reg_class(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_is_workgroup, uint16_t* out_reg_class_id);

// Resolves the same allocation-wide Workgroup view carrier during read-only
// target-contract queries.
iree_status_t loom_spirv_resolve_workgroup_contract_view_reg_class(
    const loom_target_contract_query_environment_t* environment,
    loom_value_id_t source_value_id, bool* out_is_workgroup,
    uint16_t* out_reg_class_id);

// Records one selected Workgroup allocation in its exact-carrier physical
// arena when it has a supported carrier and a potentially accessed footprint.
// Footprint-free, incompatible, or unrepresentable roots remain on the
// dedicated-storage path; subsequent source-plan demand analysis may elide an
// unused plan. |out_packed| is true only when the root joined an arena.
iree_status_t loom_spirv_workgroup_layout_record_alloca(
    loom_low_lower_context_t* context, const loom_op_t* alloca_op,
    uint64_t byte_length, uint64_t byte_alignment, bool* out_packed);

typedef struct loom_spirv_workgroup_storage_root_requirement_t {
  // Physical byte length of the emitted storage root.
  uint64_t byte_length;
  // Required base alignment of the emitted storage root.
  uint64_t byte_alignment;
} loom_spirv_workgroup_storage_root_requirement_t;

// Returns the number of retained physical storage roots in emission order.
iree_host_size_t loom_spirv_workgroup_layout_storage_root_count(
    const loom_low_lower_context_t* context);

// Returns one retained physical storage-root requirement in emission order.
loom_spirv_workgroup_storage_root_requirement_t
loom_spirv_workgroup_layout_storage_root_requirement(
    const loom_low_lower_context_t* context, iree_host_size_t index);

// Emits one physical Low Workgroup storage root for every populated scalar
// carrier arena.
iree_status_t loom_spirv_workgroup_layout_emit_storage_roots(
    loom_low_lower_context_t* context);

// Resolves the emitted Low storage root for a packed source allocation.
void loom_spirv_workgroup_layout_lookup_low_storage(
    const loom_low_lower_context_t* context, loom_value_id_t root_value_id,
    loom_value_id_t* out_low_storage_value_id);

// Returns a source allocation's byte origin in its physical Workgroup arena.
// Dedicated roots retain origin zero. Consumes the retained packing decision.
uint64_t loom_spirv_workgroup_layout_root_byte_offset(
    const loom_low_lower_context_t* context, loom_value_id_t root_value_id);

// Returns the packed allocation-root bias consumed by common source-memory
// rule matching. Non-Workgroup and dedicated roots return zero.
uint64_t loom_spirv_workgroup_layout_source_memory_root_byte_offset(
    void* user_data, const loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_memory_access);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_LAYOUT_H_
