// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic vector reference legalizers.

#ifndef LOOM_TRANSFORMS_VECTOR_TARGET_LEGALIZATION_H_
#define LOOM_TRANSFORMS_VECTOR_TARGET_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the generic vector legalizer provider. Pipelines should compose this
// after target-specific providers so native target rewrites win before scalar
// reference decomposition.
const loom_target_legalizer_provider_t* loom_vector_target_legalizer_provider(
    void);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TARGET_LEGALIZATION_H_
