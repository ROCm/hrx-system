// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Deterministic AMDGPU wait-state plan diagnostics.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_STATES_FORMAT_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_STATES_FORMAT_H_

#include "iree/base/string_builder.h"
#include "loom/target/arch/amdgpu/planning/wait_states.h"

#ifdef __cplusplus
extern "C" {
#endif

// Formats the wait-state plan as compact deterministic diagnostic text.
iree_status_t loom_amdgpu_wait_state_plan_format_text(
    const loom_amdgpu_wait_state_plan_t* plan,
    const loom_low_packet_hazard_plan_t* hazards,
    iree_string_builder_t* builder);

// Formats the wait-state plan, common progress table, and common hazard sidecar
// as deterministic JSON for diagnostics and structured tooling.
iree_status_t loom_amdgpu_wait_state_plan_format_json(
    const loom_amdgpu_wait_state_plan_t* plan,
    const loom_low_packet_hazard_plan_t* hazards,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_STATES_FORMAT_H_
