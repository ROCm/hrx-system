// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_CONVERSION_TARGET_LEGALIZATION_H_
#define LOOM_TRANSFORMS_CONVERSION_TARGET_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns reference decompositions shared by scalar and vector conversions.
// Compose after target providers so directly supported conversions retain their
// native lowering. Intermediate formats preserve the original rounding result.
// F16/FP8-to-I64 stages through exact F32 and I32/U32; BF16 retains a wide
// integer conversion because its defined results can exceed 32 bits.
const loom_target_legalizer_provider_t*
loom_conversion_target_legalizer_provider(void);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CONVERSION_TARGET_LEGALIZATION_H_
