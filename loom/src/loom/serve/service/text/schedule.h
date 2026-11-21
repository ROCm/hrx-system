// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SCHEDULE_H_
#define LOOM_SERVE_SERVICE_TEXT_SCHEDULE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

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
