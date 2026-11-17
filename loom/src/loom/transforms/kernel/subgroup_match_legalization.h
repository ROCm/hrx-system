// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_KERNEL_SUBGROUP_MATCH_LEGALIZATION_H_
#define LOOM_TRANSFORMS_KERNEL_SUBGROUP_MATCH_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Portable i32 equality recipes registered by the shared kernel provider.
// Target-native match contracts retain precedence; broadcast, vote, and mask
// contracts own the resulting primitives' target capability requirements.

// Matches a fixed subgroup size representable by the verified i32/i64 mask.
// Match-any enumerates physical source lanes, so an unknown size cannot select
// this recipe. Inactive source lanes are excluded before broadcasting.
bool loom_kernel_subgroup_match_any_is_supported(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op);

// Builds one guarded broadcast/ballot per physical lane, selecting the ballot
// for each invocation's own value. The matcher establishes the lane bound.
iree_status_t loom_kernel_target_legalize_subgroup_match_any(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Compares against the first active lane and votes on equality. The all-equal
// result selects the active mask or zero. This recipe does not enumerate lanes.
iree_status_t loom_kernel_target_legalize_subgroup_match_all(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_SUBGROUP_MATCH_LEGALIZATION_H_
