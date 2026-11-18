// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sparse AMDGPU wait-action accumulation and immutable table publication.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_ACTIONS_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_ACTIONS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet_hazard_plan.h"
#include "loom/codegen/low/packet_progress.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/planning/wait_classification.h"
#include "loom/util/segmented_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_amdgpu_wait_plan_action_t;

// Sparse append state and its exact retained action sequence.
typedef struct loom_amdgpu_wait_actions_t {
  // Stable segments populated in action order.
  loom_segmented_storage_t segments;
  // Current append segment, or NULL before the first action.
  void* tail;
  // Number of populated actions in |tail|.
  iree_host_size_t tail_count;
  // Contiguous retained action rows after finalization.
  const struct loom_amdgpu_wait_plan_action_t* actions;
  // Total number of populated action rows.
  iree_host_size_t action_count;
  // Number of actions that become canonical residual hazards.
  iree_host_size_t hazard_event_count;
} loom_amdgpu_wait_actions_t;

// Initializes an empty action builder.
void loom_amdgpu_wait_actions_initialize(loom_amdgpu_wait_actions_t* actions);

// Appends |action| to the sparse transient stream.
iree_status_t loom_amdgpu_wait_actions_append(
    loom_amdgpu_wait_actions_t* actions,
    const struct loom_amdgpu_wait_plan_action_t* action,
    iree_arena_allocator_t* transient_arena);

// Copies the sparse stream into exact retained storage.
iree_status_t loom_amdgpu_wait_actions_finalize(
    loom_amdgpu_wait_actions_t* actions, iree_arena_allocator_t* arena);

// Projects immutable action and classification facts into common Low tables.
iree_status_t loom_amdgpu_wait_actions_build_hazard_plan(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_classification_t* classification,
    const uint64_t* elided_wait_nodes, iree_host_size_t progress_event_count,
    const loom_amdgpu_wait_actions_t* actions, iree_arena_allocator_t* arena,
    loom_low_packet_hazard_plan_t* out_hazard_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_ACTIONS_H_
