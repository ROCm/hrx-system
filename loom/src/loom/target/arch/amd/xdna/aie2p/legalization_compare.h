// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_COMPARE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_COMPARE_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lowers F16, BF16, and F32 comparisons through packed integer order keys.
// Preserves NaNs, signed zeros, infinities, and subnormal ordering without
// converting floating-point values or extracting individual lanes.
iree_status_t loom_aie2p_legalize_vector_cmpf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LEGALIZATION_COMPARE_H_
