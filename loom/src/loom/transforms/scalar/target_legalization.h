// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic scalar reference legalizers.

#ifndef LOOM_TRANSFORMS_SCALAR_TARGET_LEGALIZATION_H_
#define LOOM_TRANSFORMS_SCALAR_TARGET_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the generic scalar legalizer provider. Pipelines should compose this
// after target-specific providers so native target rewrites win before scalar
// reference decomposition.
const loom_target_legalizer_provider_t* loom_scalar_target_legalizer_provider(
    void);

// Rewrites an i8/i16 scalar.andi, scalar.ori, scalar.xori, scalar.shli,
// scalar.shrsi, or scalar.shrui through an i32 carrier. The caller must
// establish that the selected target supports the required i32 operation and
// narrow integer extension/truncation operations.
iree_status_t loom_scalar_target_legalize_narrow_integer_binary_reference(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SCALAR_TARGET_LEGALIZATION_H_
