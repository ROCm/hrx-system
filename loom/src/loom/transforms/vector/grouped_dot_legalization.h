// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_GROUPED_DOT_LEGALIZATION_H_
#define LOOM_TRANSFORMS_VECTOR_GROUPED_DOT_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Partitions static dot2f, dot4i, dot8i4, and dot4f8 operations into packets
// admitted by the target contract. Input and accumulator intervals retain the
// authored grouping ratio and row-major order. Slices consume the original SSA
// values, including any captured memory reads. Dynamic shapes and operations
// without a complete bounded native partition remain with ordinary
// legalization.
//
// This provider performs native-target rewrites and precedes the reference
// provider so the caller's legalization policy governs any decomposition.
const loom_target_legalizer_provider_t*
loom_vector_grouped_dot_legalizer_provider(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_GROUPED_DOT_LEGALIZATION_H_
