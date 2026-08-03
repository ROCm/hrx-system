// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic view reference legalizers.

#ifndef LOOM_TRANSFORMS_VIEW_TARGET_LEGALIZATION_H_
#define LOOM_TRANSFORMS_VIEW_TARGET_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the generic view legalizer provider. Pipelines should compose this
// after target-specific providers so native target atomics win before portable
// compare-exchange decomposition.
const loom_target_legalizer_provider_t* loom_view_target_legalizer_provider(
    void);

// Rewrites a floating-point atomic add to a bitwise compare-exchange loop.
// The caller must establish that the selected target lacks the native atomic
// add but supports compare-exchange for the payload width.
iree_status_t loom_view_target_legalize_atomic_addf_reference(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VIEW_TARGET_LEGALIZATION_H_
