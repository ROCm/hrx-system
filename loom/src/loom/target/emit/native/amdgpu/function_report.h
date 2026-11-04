// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU compile-report adaptation for one completed native function.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_REPORT_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_REPORT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/frame.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/function_plan.h"
#include "loom/target/reporting/low_mix.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Records counter and native destination waits in one ordered report view.
// The immutable plans keep their own ownership and are not copied or mutated.
iree_status_t loom_amdgpu_function_report_record_plan(
    loom_target_compile_report_t* report,
    const loom_amdgpu_packet_plan_t* packet_plan);

// Records target-native body emission facts against their originating Low
// frame.
//
// This adapter owns the cross-layer reporting join. Native emission produces
// |stream| without inspecting report policy or running Low analyses.
iree_status_t loom_amdgpu_function_report_record_emission(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame,
    const loom_target_compile_report_low_dynamic_context_t* dynamic_context,
    const loom_amdgpu_function_plan_t* plan,
    const loom_amdgpu_encoded_instruction_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_FUNCTION_REPORT_H_
