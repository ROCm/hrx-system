// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-memory packet bank-service reporting.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_BANK_SERVICE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_BANK_SERVICE_H_

#include "loom/analysis/symbolic_expr.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/target/arch/amdgpu/analysis/lds_bank_service.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/address.h"

#ifdef __cplusplus
extern "C" {
#endif

// Calculates address evidence for a selected LDS packet from canonical source
// terms. |workgroup_size| is a fixed, complete-wave geometry established by the
// active-subgroup proof. The selected packet has a proven 32-bit DS address.
//
// Workitem coordinates follow native X-fastest order; subgroup-lane coordinates
// reset for each wave. Both admit retained digit projections and can be mixed
// in one address. Other contributions must be proven subgroup-uniform.
// |expressions| contains the completed shared source-index analysis. Exact
// evidence requires the same phase profile for every wave and every
// compatible common subword residue. This consumes retained plans and facts
// only; it neither queries nor traverses source IR.
// Active-lane evidence is supplied separately by the caller.
void loom_amdgpu_memory_calculate_source_bank_service(
    const loom_amdgpu_lds_bank_service_model_t* model,
    const loom_low_source_memory_access_plan_t* source,
    const loom_symbolic_expr_context_t* expressions,
    const loom_target_workgroup_size_t* workgroup_size,
    loom_low_lower_memory_bank_service_report_t* out_report);

// Populates bank-service evidence for a selected source-memory packet.
//
// Non-LDS packets leave the output empty. LDS packets without a target, packet,
// or wave-size model report "unmodeled" with an explicit reason.
// A selected model always produces either an exact result or an explicit
// unknown proof reason. Analysis runs only while detail report rows are
// requested.
iree_status_t loom_amdgpu_memory_report_bank_service(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_descriptor_t* descriptor,
    const loom_low_source_memory_access_plan_t* source,
    loom_low_lower_memory_bank_service_report_t* out_report);

// Populates bank-service evidence for a selected fragment-memory packet.
//
// The compiled fragment address layout is the sole source of lane, register,
// and packed-element coordinates. Source dynamic terms must prove a common
// subgroup translation before an exact result is reported. |descriptor| names
// the issued instruction; |packet| can describe a pair of low/high-half loads.
iree_status_t loom_amdgpu_fragment_memory_report_bank_service(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index,
    const loom_amdgpu_fragment_memory_packet_offset_t* runtime_offset,
    loom_low_lower_memory_bank_service_report_t* out_report);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_BANK_SERVICE_H_
