// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Architected workgroup and clustered-launch identity materialization.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CLUSTER_PREAMBLE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CLUSTER_PREAMBLE_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/ops/kernel/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_target_facts_t loom_amdgpu_target_facts_t;

typedef struct loom_amdgpu_cluster_preamble_demands_t {
  // First source op requiring each global workgroup coordinate.
  const loom_op_t* workgroup_id_ops[LOOM_KERNEL_DIMENSION_COUNT_];
  // First source op requiring each cluster coordinate.
  const loom_op_t* cluster_id_ops[LOOM_KERNEL_DIMENSION_COUNT_];
  // First source op requiring each within-cluster workgroup coordinate.
  const loom_op_t* cluster_workgroup_id_ops[LOOM_KERNEL_DIMENSION_COUNT_];
  // First source op requiring the flat within-cluster workgroup coordinate.
  const loom_op_t* cluster_workgroup_flat_id_op;
} loom_amdgpu_cluster_preamble_demands_t;

// Returns the statically proven nontrivial cluster size, when present.
bool loom_amdgpu_cluster_preamble_required_nontrivial_size(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_value_fact_table_t* fact_table,
    loom_target_workgroup_cluster_size_t* out_size);

// Returns whether the selected processor carries workgroup coordinates in
// architected TTMP launch state.
bool loom_amdgpu_cluster_preamble_target_uses_architected_workgroup_ids(
    const loom_amdgpu_target_facts_t* target_facts);

// Returns whether the selected processor defines clustered launch state.
bool loom_amdgpu_cluster_preamble_target_supports_cluster_launch_state(
    const loom_amdgpu_target_facts_t* target_facts);

// Returns the static cluster extent along |dimension|.
uint32_t loom_amdgpu_cluster_preamble_size_dimension(
    const loom_target_workgroup_cluster_size_t* size,
    loom_kernel_dimension_t dimension);

// Reports whether the selected target carries workgroup identity in
// architected TTMP launch state.
iree_status_t loom_amdgpu_cluster_preamble_uses_architected_workgroup_ids(
    loom_low_lower_context_t* context,
    bool* out_uses_architected_workgroup_ids);

// Reports whether the current function uses an extended clustered-dispatch
// packet. Only a statically nontrivial cluster selects that packet ABI.
iree_status_t loom_amdgpu_cluster_preamble_uses_clustered_dispatch(
    loom_low_lower_context_t* context, bool* out_uses_clustered_dispatch);

// Imports the fixed architected launch-state registers required by |demands|.
iree_status_t loom_amdgpu_cluster_preamble_emit_live_ins(
    loom_low_lower_context_t* context,
    const loom_amdgpu_cluster_preamble_demands_t* demands);

// Materializes and binds identity values after structural ABI imports.
iree_status_t loom_amdgpu_cluster_preamble_emit_entry_setup(
    loom_low_lower_context_t* context);

// Looks up a materialized global workgroup coordinate.
iree_status_t loom_amdgpu_cluster_preamble_lookup_workgroup_id(
    loom_low_lower_context_t* context, loom_kernel_dimension_t dimension,
    loom_value_id_t* out_low_value_id);

// Returns the current static cluster extent along |dimension|.
iree_status_t loom_amdgpu_cluster_preamble_lookup_size(
    loom_low_lower_context_t* context, loom_kernel_dimension_t dimension,
    uint32_t* out_size);

// Loads a dynamic cluster count from the extended dispatch packet.
iree_status_t loom_amdgpu_cluster_preamble_emit_cluster_count(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_ptr, loom_kernel_dimension_t dimension,
    loom_type_t result_type, loom_value_id_t* out_low_value_id);

// Materializes a dynamic global workgroup count from the cluster count.
iree_status_t loom_amdgpu_cluster_preamble_emit_workgroup_count(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dispatch_ptr, loom_kernel_dimension_t dimension,
    loom_type_t result_type, loom_value_id_t* out_low_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CLUSTER_PREAMBLE_H_
