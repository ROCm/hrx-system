// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native function planning from an accepted target-low frame.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_PLAN_BUILD_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_PLAN_BUILD_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/error/emitter.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/emit/native/amdgpu/function_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds the native decisions common to kernel entries and callable helpers.
// User-dependent preflight failures emit diagnostics and leave |out_accepted|
// false. |arena| must outlive |out_plan|.
iree_status_t loom_amdgpu_function_plan_build(
    const loom_low_emission_frame_t* frame, iree_diagnostic_emitter_t emitter,
    const loom_amdgpu_packet_plan_options_t* packet_options,
    iree_arena_allocator_t* arena, bool* out_accepted,
    loom_amdgpu_function_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_PLAN_BUILD_H_
