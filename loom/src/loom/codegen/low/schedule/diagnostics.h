// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured diagnostic emission for target-low scheduling.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_DIAGNOSTICS_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_DIAGNOSTICS_H_

#include "loom/codegen/low/schedule/context.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_low_schedule_emit_missing_descriptor(
    loom_low_schedule_build_state_t* state, const loom_op_t* op,
    iree_string_view_t opcode);

iree_status_t loom_low_schedule_emit_dependency_cycle(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_failure_t* failure);

iree_status_t loom_low_schedule_emit_pressure_diagnostics(
    loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness);

iree_status_t loom_low_schedule_emit_candidate_decision_diagnostics(
    loom_low_schedule_build_state_t* state);

iree_status_t loom_low_schedule_emit_model_diagnostics(
    loom_low_schedule_build_state_t* state);

iree_status_t loom_low_schedule_emit_resource_diagnostics(
    loom_low_schedule_build_state_t* state);

iree_status_t loom_low_schedule_emit_hazard_gap_diagnostics(
    loom_low_schedule_build_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_DIAGNOSTICS_H_
