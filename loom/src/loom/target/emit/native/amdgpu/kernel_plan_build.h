// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native planning from an accepted target-low emission frame.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_PLAN_BUILD_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_PLAN_BUILD_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/error/emitter.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/emit/native/amdgpu/kernel_plan.h"
#include "loom/target/emit/native/amdgpu/program_resources.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_kernel_plan_options_t {
  // Verified ABI facts captured before allocation.
  const loom_amdgpu_hal_kernel_abi_verify_result_t* abi_verify;
  // Optional complete helper-closure resource facts. Null for a call-free
  // kernel, preserving its function-local storage and allocation resources.
  const loom_amdgpu_program_resources_t* program_resources;
  // Structured diagnostic emitter for target-dependent admission failures.
  iree_diagnostic_emitter_t emitter;
  // Optional compile report receiving native planning facts.
  loom_target_compile_report_t* report;
} loom_amdgpu_kernel_plan_options_t;

// Builds and admits the complete target-native plan for |frame|.
//
// User-dependent target failures emit diagnostics and leave |out_accepted|
// false. An accepted plan contains every semantic and packet-planning decision
// required by byte and listing emission. |arena| must outlive |out_plan|.
iree_status_t loom_amdgpu_kernel_plan_build(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_kernel_plan_options_t* options,
    iree_arena_allocator_t* arena, bool* out_accepted,
    loom_amdgpu_kernel_plan_t* out_plan);

// Builds and admits the entry-only native plan around an already compiled
// function plan. This is the common boundary for call-free kernels and entries
// participating in a retained native program.
iree_status_t loom_amdgpu_kernel_plan_build_from_function(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_function_plan_t* function_plan,
    const loom_amdgpu_kernel_plan_options_t* options,
    iree_arena_allocator_t* arena, bool* out_accepted,
    loom_amdgpu_kernel_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_PLAN_BUILD_H_
