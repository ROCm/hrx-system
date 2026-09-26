// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Static multidimensional vector shape legalization.

#ifndef LOOM_TRANSFORMS_VECTOR_SHAPE_LEGALIZATION_H_
#define LOOM_TRANSFORMS_VECTOR_SHAPE_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rewrites supported static multidimensional vector operations into rank-one
// structural operations and shape-only bitcasts. Returns without rewriting
// dynamic shapes or rank-one operations.
iree_status_t loom_vector_static_shape_rewrite_op(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_SHAPE_LEGALIZATION_H_
