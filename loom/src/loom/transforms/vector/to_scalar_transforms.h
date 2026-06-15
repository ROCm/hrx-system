// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Numeric transform lane programs.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_TRANSFORMS_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_TRANSFORMS_H_

#include "loom/transforms/vector/to_scalar_lanes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates transform descriptors before aggregate lowering emits lane IR.
iree_status_t loom_vector_to_scalar_validate_transform(
    loom_vector_to_scalar_state_t* state);

iree_status_t loom_vector_to_scalar_build_transform_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_TRANSFORMS_H_
