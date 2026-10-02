// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_QWEN_FLAGS_H_
#define EXPERIMENTAL_LOOM_SERVE_QWEN_FLAGS_H_

#include "experimental/loom_serve/qwen_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared source, weight, and specialization flags for every Qwen entry point.
// The caller parses ordinary IREE flags before invoking these functions.
iree_host_size_t loom_serve_qwen_shape_count_from_flags(void);
bool loom_serve_qwen_mtp_from_flags(void);
iree_status_t loom_serve_qwen_model_create_from_flags(
    iree_host_size_t row_count, iree_allocator_t host_allocator,
    loom_serve_qwen_model_t** out_model);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_QWEN_FLAGS_H_
