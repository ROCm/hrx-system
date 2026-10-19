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

// Splits a wide subgroup payload only when each target-native packet preserves
// the authored addressing and participation contract. Target payload adapters
// use this before changing the value representation; rejected packet contracts
// leave the operation unchanged for shared normalization or payload conversion.
iree_status_t loom_kernel_target_legalize_subgroup_packets(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Matches logical shuffle addressing that differs from the physical indexed
// normal form. Payload adapters normalize it once before splitting the value.
bool loom_kernel_match_subgroup_shuffle(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op);

// Exchanges scalar or rank-1 i1 payloads through shape-preserving i32 values.
// Targets whose predicates occupy lane masks use this for shuffle, broadcast,
// and broadcast.first. Shuffle source validity remains independent of payload
// conversion.
iree_status_t loom_kernel_target_legalize_subgroup_predicate(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Normalizes logical shuffle addressing to an in-range physical indexed
// exchange, retaining logical bounds and source participation in the validity
// result. Targets with custom source-low selection call this when their native
// addressing preconditions are not satisfied.
iree_status_t loom_kernel_target_legalize_subgroup_shuffle(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Rewrites a subgroup reduction into the shared active-rank shuffle tree.
// Target legalizers retain native cells by accepting them before this
// reference provider runs. Logical cluster attributes and sparse active lanes
// are preserved by the same rank mapping used for subgroup scans.
iree_status_t loom_kernel_target_legalize_subgroup_reduce(
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
