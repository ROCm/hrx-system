// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SCHEDULE_H_
#define LOOM_SERVE_SERVICE_TEXT_SCHEDULE_H_

#include "iree/base/api.h"
#include "loom/serve/service/text/packing.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_text_schedule_mode_e {
  // One shared weight traversal for all admitted prompt and decode spans.
  LOOM_SERVE_TEXT_SCHEDULE_PACKED = 0,
  // Same ready-span partition, executed with ordinary isolated model stages.
  LOOM_SERVE_TEXT_SCHEDULE_ISOLATED,
  // Isolated prefill math even for decode, for a same-math packed control.
  LOOM_SERVE_TEXT_SCHEDULE_MATCHED,
} loom_serve_text_schedule_mode_t;

typedef enum loom_serve_text_packing_mode_e {
  // Fill one epoch with all ready prompt and decode spans.
  LOOM_SERVE_TEXT_PACKING_MIXED = 0,
  // Pack only the first ready row's phase, using the same rotating priority.
  LOOM_SERVE_TEXT_PACKING_SEPARATE,
} loom_serve_text_packing_mode_t;

// Seventeen token classes through 262144 and five span classes through 16.
enum { LOOM_SERVE_TEXT_DEFAULT_SHAPE_CAPACITY = 85 };

// Cold catalog construction for validated row_count in [1, 16] and maximum
// token capacity in [1, 262144]. Token classes start at four; span classes at
// one. Each axis doubles and includes its exact terminal capacity, including
// odd counts. Shapes with more spans than tokens are omitted. Writes at most
// DEFAULT_SHAPE_CAPACITY entries and returns the nonzero count.
iree_host_size_t loom_serve_text_default_shapes(
    iree_host_size_t row_count, iree_host_size_t token_capacity,
    loom_serve_packing_shape_t* shapes);

// Reserves page-rounded capacity for a validated request through completion.
// input_count includes retained and appended input. The final selected output
// stays pending and does not enter KV. Legal speculative steps may transiently
// write past that committed endpoint, even when output credit is only two.
// Positive input/output counts already fit context_capacity; proposal_depth is
// within the source-validated draft depth and block_size is positive. No
// allocation or state mutation.
iree_host_size_t loom_serve_text_request_reservation(
    iree_host_size_t context_capacity, iree_host_size_t input_count,
    iree_host_size_t output_count, iree_host_size_t proposal_depth,
    iree_host_size_t block_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_SCHEDULE_H_
