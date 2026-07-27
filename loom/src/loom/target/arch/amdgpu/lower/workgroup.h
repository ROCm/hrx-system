// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for workgroup collective source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_WORKGROUP_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_WORKGROUP_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects native AMDGPU packets for a source workgroup reduce.
iree_status_t loom_amdgpu_select_kernel_workgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_workgroup_reduce_plan_t* out_plan, bool* out_selected);

// Lowers a source workgroup reduce through native subgroup and LDS packets.
iree_status_t loom_amdgpu_lower_kernel_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_reduce_plan_t* plan);

// Verifies source workgroup reduce legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_workgroup_reduce(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Returns the compile-report key for a selected workgroup reduce publication
// strategy.
iree_string_view_t loom_amdgpu_workgroup_reduce_publication_report_key(
    loom_amdgpu_workgroup_reduce_publication_kind_t publication_kind);

// Selects native AMDGPU cross-lane and LDS packets for a source workgroup scan.
iree_status_t loom_amdgpu_select_kernel_workgroup_scan_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_workgroup_scan_plan_t* out_plan, bool* out_selected);

// Lowers a source workgroup scan through subgroup scans plus LDS cross-wave
// prefixes when needed.
iree_status_t loom_amdgpu_lower_kernel_workgroup_scan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_scan_plan_t* plan);

// Verifies source workgroup scan legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_workgroup_scan(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_WORKGROUP_H_
