// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Register capacity observed at retained call boundaries.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_CALL_PRESSURE_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_CALL_PRESSURE_H_

#include "loom/codegen/low/schedule/context.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_schedule_pressure_state_t
    loom_low_schedule_pressure_state_t;
typedef struct loom_low_schedule_candidate_score_t
    loom_low_schedule_candidate_score_t;

// Validates retained call contracts and builds sparse boundary capacities.
iree_status_t loom_low_schedule_call_pressure_initialize(
    loom_low_schedule_build_state_t* state);

// Adds transient pressure debt when live values exceed the storage preserved by
// a retained call candidate.
void loom_low_schedule_call_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node_index, loom_low_schedule_candidate_score_t* score);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_CALL_PRESSURE_H_
