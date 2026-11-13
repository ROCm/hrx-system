// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for target-low spill traffic.
//
// Target-low allocation materialization can select concrete AMDGPU
// private-segment scratch recipes before mutating IR and emit them directly.
// The same machinery also lowers authored low.spill/low.reload operations.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_SPILL_LOWERING_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_SPILL_LOWERING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_spill_plan_t
    loom_low_allocation_spill_plan_t;
typedef struct loom_low_allocation_table_t loom_low_allocation_table_t;
typedef struct loom_low_spill_materialization_plan_t
    loom_low_spill_materialization_plan_t;
typedef struct loom_low_spill_materialization_traffic_plan_t
    loom_low_spill_materialization_traffic_plan_t;

typedef struct loom_amdgpu_spill_lowering_result_t {
  // Number of user-facing diagnostics emitted while lowering spill traffic.
  uint32_t error_count;
  // True when lowering replaced at least one spill or reload operation.
  bool lowered_traffic;
  // Scratch-arena-owned helper value IDs created by spill lowering
  // that must remain in registers during subsequent allocation rounds.
  const loom_value_id_t* required_register_value_ids;
  // Number of entries in |required_register_value_ids|.
  iree_host_size_t required_register_value_count;
  // Scratch-arena-owned VGPR values defined while EXEC is widened beyond the
  // surrounding program point's semantic lane mask.
  const loom_value_id_t* expanded_execution_value_ids;
  // Number of entries in |expanded_execution_value_ids|.
  iree_host_size_t expanded_execution_value_count;
  // Scratch-arena-owned register values created as single-result private setup
  // for one consumer while lowering reloads.
  const loom_value_id_t* per_user_placement_value_ids;
  // Number of entries in |per_user_placement_value_ids|.
  iree_host_size_t per_user_placement_value_count;
} loom_amdgpu_spill_lowering_result_t;

// Selects immutable AMDGPU packet recipes for one generic spill
// materialization plan. The selected recipes are emitted directly while the
// generic plan applies its ordered SSA and CFG mutations.
// |user_data| points to the function's immutable
// loom_amdgpu_storage_layout_segment_sizes_t prefix throughout frame
// rebuilding.
iree_status_t loom_amdgpu_plan_spill_materialization_traffic(
    void* user_data, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t spill_plan_count,
    const loom_low_spill_materialization_plan_t* materialization_plan,
    iree_arena_allocator_t* arena,
    loom_low_spill_materialization_traffic_plan_t* out_plan);

// Rewrites low.spill/low.reload in |function_op| into descriptor-backed AMDGPU
// scratch load/store packets. The descriptor set must be the already-selected
// target-low descriptor set for the function target. VGPR and AGPR traffic
// retains its payload bank and execution mask; SGPR traffic broadcasts and
// gathers a wave-wide snapshot through VGPRs. Scratch addresses use VGPRs
// independently of the payload bank.
// |segment_prefix| is the same fixed placement used by allocation repair and
// final native storage projection.
iree_status_t loom_amdgpu_lower_spill_traffic(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_storage_layout_segment_sizes_t segment_prefix,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* scratch_arena,
    loom_amdgpu_spill_lowering_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_SPILL_LOWERING_H_
