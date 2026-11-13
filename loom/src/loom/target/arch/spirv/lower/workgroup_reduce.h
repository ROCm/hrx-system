// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_REDUCE_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_REDUCE_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Retains typed payloads and subgroup composition at one convergence point.
iree_status_t loom_spirv_select_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Emits the retained native reduction with shared topology and synchronization.
iree_status_t loom_spirv_emit_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Appends each implicit typed scratch reservation to the function's existing
// storage extent, including alignment between differently typed payloads.
// Returns false when the aggregate exceeds the 64-bit address space.
bool loom_spirv_workgroup_reduce_measure_storage(loom_low_lower_plan_t plan,
                                                 uint64_t* inout_byte_extent);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_REDUCE_H_
