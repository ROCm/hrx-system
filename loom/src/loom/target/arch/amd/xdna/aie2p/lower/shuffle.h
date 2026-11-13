// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P fixed vector-permutation lowering.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_SHUFFLE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_SHUFFLE_H_

#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the complete fixed vector-permutation family admitted by AIE2P
// register carriers. Non-shuffle and carrierless operations leave |out_plan|
// empty.
iree_status_t loom_aie2p_select_shuffle_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan);

// Publishes an identity permutation's actual source carrier.
iree_status_t loom_aie2p_finalize_shuffle_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Retains the shuffle source carrier for callback emission.
void loom_aie2p_mark_shuffle_plan_demands(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan);

// Describes the selected packet-routing or broadcast/select realization.
void loom_aie2p_describe_shuffle_plan(loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_low_lower_plan_t plan,
                                      loom_low_lower_plan_report_t* out_report);

// Emits the selected fixed vector-permutation realization.
iree_status_t loom_aie2p_emit_shuffle_plan(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_SHUFFLE_H_
