// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native kernel emission from a completed target-low frame.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_EMISSION_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_EMISSION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/frame.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"
#include "loom/target/emit/native/amdgpu/preflight.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds one native kernel contribution and optional assembly listing.
//
// This is the ownership boundary for target packet planning and native
// instruction reporting. Assembly emission consumes the packet plan and exact
// branch layout produced for machine-code emission so both products describe
// the same kernel.
iree_status_t loom_amdgpu_kernel_emission_build(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout,
    const loom_amdgpu_hal_kernel_abi_verify_result_t* abi_verify,
    const loom_amdgpu_native_preflight_t* preflight,
    iree_string_builder_t* target_listing, loom_target_compile_report_t* report,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution,
    iree_arena_allocator_t* table_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_EMISSION_H_
