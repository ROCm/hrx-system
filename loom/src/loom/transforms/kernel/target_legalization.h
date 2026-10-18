// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic kernel reference legalizers.

#ifndef LOOM_TRANSFORMS_KERNEL_TARGET_LEGALIZATION_H_
#define LOOM_TRANSFORMS_KERNEL_TARGET_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the generic kernel legalizer provider. Pipelines should compose this
// after target-specific providers so native collectives win before their
// portable decompositions.
const loom_target_legalizer_provider_t* loom_kernel_target_legalizer_provider(
    void);

// Normalizes logical shuffle addressing to an in-range physical indexed
// exchange, retaining logical bounds and source participation in the validity
// result. Targets with custom source-low selection call this when their native
// addressing preconditions are not satisfied.
iree_status_t loom_kernel_target_legalize_subgroup_shuffle(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Rewrites a subgroup scan into portable subgroup primitives. Target
// legalizers may call this when an otherwise legal scan violates a native
// implementation precondition. Unsupported cluster shapes are left unchanged
// for final legality to diagnose. Except under explicit reference-only policy,
// expansion waits for final legalization so source cleanup can share inputs
// before constructing the shuffle/select network.
iree_status_t loom_kernel_target_legalize_subgroup_scan(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_TARGET_LEGALIZATION_H_
