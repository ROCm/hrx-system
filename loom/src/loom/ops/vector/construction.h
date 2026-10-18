// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_VECTOR_CONSTRUCTION_H_
#define LOOM_OPS_VECTOR_CONSTRUCTION_H_

#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects a static rank-one slice or scalar extraction into one retained
// regular vector component. The indexed partition preserves the component's
// SSA value instead of rematerializing its producer. Spanning or dynamic
// projections and sources without a regular partition remain unchanged.
iree_status_t loom_vector_project_regular_partition(loom_op_t* op,
                                                    loom_rewriter_t* rewriter,
                                                    bool* out_changed);

// Folds a pure single-vector-result operation to vector.from_elements when
// retained small-static-lane facts provide an exact constant for every lane.
// The caller selects operations whose constant construction is canonical;
// unknown lanes and NaN facts without retained payloads remain unchanged.
iree_status_t loom_vector_fold_constant_lanes(loom_op_t* op,
                                              loom_rewriter_t* rewriter,
                                              bool* out_changed);

// Folds a vector.from_elements of every unmodified source lane, in order, to
// that source. This identity introduces no operation and is valid at any phase.
// Nonmatching constructions remain unchanged and allocate no scratch storage.
iree_status_t loom_vector_from_elements_fold_source(loom_op_t* op,
                                                    loom_rewriter_t* rewriter,
                                                    bool* out_changed);

// Combines identical scalar rounding/conversion chains on ordered source lanes
// into elementwise vector operations. Only run before target legalization:
// scalarization deliberately decomposes these operations into the same chains.
// The bounded matcher borrows verified IR and allocates only on a match.
iree_status_t loom_vector_from_elements_combine_lanes(loom_op_t* op,
                                                      loom_rewriter_t* rewriter,
                                                      bool* out_changed);

// Combines a complete rank-one chain of static scalar vector.insert operations
// into vector.from_elements. The outermost write to each lane wins; lanes not
// written by the chain can be retained from vector.splat or
// vector.from_elements bases. Only run before target legalization, which may
// deliberately linearize vector.from_elements back into scalar insertions.
iree_status_t loom_vector_insert_combine_elements(loom_op_t* op,
                                                  loom_rewriter_t* rewriter,
                                                  bool* out_changed);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_VECTOR_CONSTRUCTION_H_
