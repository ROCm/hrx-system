// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU launch topology and LDS layout lowering helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_TOPOLOGY_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_TOPOLOGY_H_

#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves the source-order workgroup-storage base for a workgroup
// buffer.alloca root. The layout must match the low.storage.reserve order
// emitted by buffer lowering.
bool loom_amdgpu_source_lds_layout_lookup_root(
    const loom_value_fact_table_t* fact_table, loom_func_like_t source_function,
    loom_value_id_t root_value_id, uint64_t* out_byte_offset);

// Returns the exact wavefront size selected by the active target bundle.
iree_status_t loom_amdgpu_target_wavefront_size(
    const loom_target_bundle_t* bundle, uint32_t* out_wavefront_size);

// Returns the fixed per-dimension workgroup size required by the source
// function or target ABI.
bool loom_amdgpu_required_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, loom_target_workgroup_size_t* out_size);

// Returns the fixed flat workgroup size required by the source function or
// target ABI.
bool loom_amdgpu_required_flat_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t* out_flat_size);

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
