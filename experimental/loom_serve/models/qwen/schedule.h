// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_MODELS_QWEN_SCHEDULE_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_MODELS_QWEN_SCHEDULE_H_

#include "experimental/loom_serve/scheduling/packing.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Five token classes through 512 and five span classes through 16.
enum { LOOM_SERVE_QWEN_DEFAULT_SHAPE_CAPACITY = 25 };

// Cold catalog construction for validated row_count in [1, 16] and maximum
// token capacity in [1, 512]. Token classes start at 32; span classes at one.
// Each axis doubles and includes its exact terminal capacity, including odd
// counts. Shapes with more spans than tokens are omitted. Writes at most
// DEFAULT_SHAPE_CAPACITY entries and returns the nonzero count.
iree_host_size_t loom_serve_qwen_default_shapes(
    iree_host_size_t row_count, iree_host_size_t token_capacity,
    loom_serve_packing_shape_t* shapes);

// Reserves page-rounded capacity for a validated request through completion.
// input_count includes retained and appended input. The final selected output
// stays pending and does not enter KV. Legal speculative steps may transiently
// write past that committed endpoint, even when output credit is only two.
// Positive input/output counts already fit context_capacity; proposal_depth is
// zero or three and block_size is positive. No allocation or state mutation.
iree_host_size_t loom_serve_qwen_request_reservation(
    iree_host_size_t context_capacity, iree_host_size_t input_count,
    iree_host_size_t output_count, iree_host_size_t proposal_depth,
    iree_host_size_t block_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_MODELS_QWEN_SCHEDULE_H_
