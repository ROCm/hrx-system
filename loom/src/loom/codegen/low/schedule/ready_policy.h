// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared ready membership and pair-setup policy for low scheduling.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_READY_POLICY_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_READY_POLICY_H_

#include "loom/codegen/low/schedule/context.h"
#include "loom/codegen/low/schedule/ready_frontier.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mutable ready policy for one scheduling run.
typedef struct loom_low_schedule_ready_policy_t {
  // Shared ready membership and nomination heaps.
  loom_low_schedule_ready_frontier_t frontier;
  // Dependencies from ready structural setup nodes, indexed by consumer node.
  uint32_t* setup_dependency_counts;
} loom_low_schedule_ready_policy_t;

// Initializes an empty policy with |view_count| indexed ready views.
iree_status_t loom_low_schedule_ready_policy_initialize(
    const loom_low_schedule_build_state_t* state, uint32_t node_count,
    uint8_t view_count, loom_low_schedule_ready_policy_t* out_policy);

// Inserts |node_index| into the ready policy with precomputed nomination keys.
void loom_low_schedule_ready_policy_insert(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_ready_policy_t* policy, uint32_t node_index,
    const loom_low_schedule_ready_keys_t* keys);

// Removes |node_index| from the ready policy.
void loom_low_schedule_ready_policy_remove(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_ready_policy_t* policy, uint32_t node_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_READY_POLICY_H_
