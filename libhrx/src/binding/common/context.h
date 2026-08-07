// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_CONTEXT_H_
#define IREE_EXPERIMENTAL_STREAMING_CONTEXT_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;

// Orders subsequent legacy-default-stream work after all work submitted before
// this call on blocking streams. The dependencies are encoded on the device
// timeline; this function does not wait for them on the host.
iree_status_t
iree_hal_streaming_context_enqueue_legacy_default_dependency_barrier(
    iree_hal_streaming_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_CONTEXT_H_
