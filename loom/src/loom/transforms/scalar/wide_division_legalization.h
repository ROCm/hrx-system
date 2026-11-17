// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Portable wide integer quotient and remainder legalization.

#ifndef LOOM_TRANSFORMS_SCALAR_WIDE_DIVISION_LEGALIZATION_H_
#define LOOM_TRANSFORMS_SCALAR_WIDE_DIVISION_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the shared reference provider for exact signed/unsigned i64 and
// nonnegative index division over i32 arithmetic contracts. Compose it after
// target-specific providers and before generic quotient/remainder splitting.
const loom_target_legalizer_provider_t*
loom_scalar_wide_division_legalizer_provider(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SCALAR_WIDE_DIVISION_LEGALIZATION_H_
