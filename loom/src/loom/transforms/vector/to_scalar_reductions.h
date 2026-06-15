// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reduction and dot-product lowering.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_REDUCTIONS_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_REDUCTIONS_H_

#include "loom/transforms/vector/to_scalar_lanes.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_vector_to_scalar_lower_reduce(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

iree_status_t loom_vector_to_scalar_lower_reduce_axes(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

iree_status_t loom_vector_to_scalar_lower_dotf(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_REDUCTIONS_H_
