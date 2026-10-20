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

// Rewrites supported static vector operations into rank-one structural
// operations and shape-only bitcasts. Static and dynamically indexed tail
// extracts, rectangular slices, and mixed static/dynamic scalar extracts and
// inserts are linearized while their row-major source shape is visible.
// Rectangles retain maximal contiguous runs. An exclusive ordinary load may
// become a load of the selected rectangle when shared read-motion analysis
// permits moving it to the slice. Rank-one singleton broadcasts become one
// extract and one splat. Returns without rewriting dynamic shapes or expansions
// above the static operation bound.
// Rank-one slices and scalar extracts project directly into retained regular
// vector components when the selected range fits one component.
iree_status_t loom_vector_static_shape_rewrite_op(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten);

// Rewrites a vector.load or vector.store over a static multidimensional vector
// footprint into rank-one memory operations. Contiguous footprints use one
// access through a row-major view bitcast. Rectangular default-layout
// footprints use bounded innermost-row accesses at adjusted coordinates.
// Returns without rewriting dynamic vector shapes, encoded layouts, or
// expansions above the static operation bound.
iree_status_t loom_vector_static_shape_memory_rewrite_op(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_SHAPE_LEGALIZATION_H_
