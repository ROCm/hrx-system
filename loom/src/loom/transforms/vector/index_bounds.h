// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Coordinate bounds retained when lowering vector accesses.

#ifndef LOOM_TRANSFORMS_VECTOR_INDEX_BOUNDS_H_
#define LOOM_TRANSFORMS_VECTOR_INDEX_BOUNDS_H_

#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Retains the in-bounds access contract for one vector coordinate before its
// shape is erased. |extent| is the number of accessed lanes along |axis|, not
// the flattened access size. Returns |index| unchanged when the axis is dynamic
// or its existing facts already prove the range. Only the returned alias is
// refined; other uses of |index| keep their original domain.
iree_status_t loom_vector_assume_axis_index(loom_rewriter_t* rewriter,
                                            loom_type_t vector_type,
                                            uint8_t axis, int64_t extent,
                                            loom_value_id_t index,
                                            loom_location_id_t location,
                                            loom_value_id_t* out_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_INDEX_BOUNDS_H_
