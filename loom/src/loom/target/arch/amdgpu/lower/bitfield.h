// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for vector bitfield source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_BITFIELD_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_BITFIELD_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects an AMDGPU vector.bitfield.extract* plan.
iree_status_t loom_amdgpu_select_vector_bitfield_extract_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitfield_extract_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.bitfield.extract* op to AMDGPU descriptor-backed low
// packets.
iree_status_t loom_amdgpu_lower_vector_bitfield_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_extract_plan_t* plan);

// Selects an AMDGPU vector.bitfield.insert plan.
iree_status_t loom_amdgpu_select_vector_bitfield_insert_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitfield_insert_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.bitfield.insert op to AMDGPU descriptor-backed low
// packets.
iree_status_t loom_amdgpu_lower_vector_bitfield_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_insert_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_BITFIELD_H_
