// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Portable workgroup collective composition.

#ifndef LOOM_TRANSFORMS_KERNEL_WORKGROUP_COLLECTIVE_LEGALIZATION_H_
#define LOOM_TRANSFORMS_KERNEL_WORKGROUP_COLLECTIVE_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rewrites a workgroup reduction into subgroup reductions and typed workgroup
// scratch communication. Hierarchical partials and final results occupy
// disjoint slots across publication phases. A single-subgroup workgroup needs
// no scratch.
iree_status_t loom_kernel_target_legalize_workgroup_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Rewrites a workgroup scan into subgroup scans/reductions and typed workgroup
// scratch communication. Hierarchical partials and subgroup adjustments occupy
// disjoint slots across publication phases. A single-subgroup workgroup needs
// no scratch.
iree_status_t loom_kernel_target_legalize_workgroup_scan(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_WORKGROUP_COLLECTIVE_LEGALIZATION_H_
