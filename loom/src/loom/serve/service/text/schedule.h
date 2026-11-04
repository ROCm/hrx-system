// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SCHEDULE_H_
#define LOOM_SERVE_SERVICE_TEXT_SCHEDULE_H_

#include "iree/base/api.h"
#include "loom/serve/service/text/packing.h"
#include "loom/service/text/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold catalog construction for a positive token capacity fitting signed i32.
// Token buckets start at one, double, and include the exact terminal capacity.
// Each bucket includes span counts from one through its token count with the
// same progression. Every token can therefore advance a distinct session;
// initial resident storage does not constrain the reusable dispatch catalog.
// Returns the nonzero count; shapes may be NULL to query storage first.
iree_host_size_t loom_serve_text_default_shapes(
    iree_host_size_t token_capacity, loom_serve_packing_shape_t* shapes);

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
