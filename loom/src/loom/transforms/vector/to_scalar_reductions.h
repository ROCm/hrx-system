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

typedef enum loom_vector_to_scalar_reduce_input_mode_e {
  // Rebuilds each input lane from decomposable producer operations.
  LOOM_VECTOR_TO_SCALAR_REDUCE_INPUT_MODE_REMATERIALIZE = 0,
  // Extracts each lane from the input aggregate without rebuilding producers.
  LOOM_VECTOR_TO_SCALAR_REDUCE_INPUT_MODE_CAPTURED = 1,
} loom_vector_to_scalar_reduce_input_mode_t;

iree_status_t loom_vector_to_scalar_lower_reduce(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_reduce_input_mode_t input_mode,
    loom_value_id_t* out_replacement);

iree_status_t loom_vector_to_scalar_lower_reduce_axes(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

iree_status_t loom_vector_to_scalar_lower_dotf(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_REDUCTIONS_H_
